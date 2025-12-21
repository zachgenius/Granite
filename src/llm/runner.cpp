#include "llm_internal.h"

namespace granite {

// Thread-local RNG for sampling
thread_local std::mt19937 g_rng{std::random_device{}()};

Result<std::unique_ptr<LLMRunner>> LLMRunner::load(const std::string& path) {
    auto runner = std::make_unique<LLMRunner>();

    // Create backend
    runner->backend_ = create_default_backend();
    if (!runner->backend_) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to create backend");
    }

    auto init_result = runner->backend_->initialize();
    if (!init_result.ok()) {
        return init_result.error();
    }

    // Load model
    auto model_result = TransformerModel::load(path, runner->backend_.get());
    if (!model_result.ok()) {
        return model_result.error();
    }
    runner->model_ = std::move(model_result).take();

    // Load tokenizer
    auto gguf_result = GGUFFile::open(path);
    if (!gguf_result.ok()) {
        return gguf_result.error();
    }

    auto tok_result = Tokenizer::from_gguf(gguf_result.value());
    if (!tok_result.ok()) {
        GRANITE_LOG_WARN("Failed to load tokenizer: {}", tok_result.error().message());
    } else {
        runner->tokenizer_ = std::move(tok_result).take();
    }

    // Allocate KV cache
    auto kv_result = KVCache::allocate(
        runner->model_.config(),
        runner->model_.config().max_seq_len,
        runner->backend_.get());
    if (!kv_result.ok()) {
        return kv_result.error();
    }
    runner->kv_cache_ = std::move(kv_result).take();

#ifdef GRANITE_HAS_METAL
    // Allocate GPU KV cache for faster decode
    auto gpu_cache_result = runner->model_.allocate_gpu_kv_cache(
        runner->model_.config().max_seq_len);
    if (!gpu_cache_result.ok()) {
        GRANITE_LOG_WARN("Failed to allocate GPU KV cache: {}",
                         gpu_cache_result.error().message());
        // Continue without GPU KV cache - will use CPU path
    }
#endif

    return runner;
}

Result<std::string> LLMRunner::generate(
    const std::string& prompt,
    const GenerationConfig& config)
{
    std::string result;
    auto status = generate_streaming(prompt, config, [&](const std::string& token) {
        result += token;
        return true;
    });

    if (!status.ok()) {
        return status.error();
    }
    return result;
}

Result<void> LLMRunner::generate_streaming(
    const std::string& prompt,
    const GenerationConfig& config,
    TokenCallback callback)
{
    cancelled_ = false;

    if (!tokenizer_.is_loaded()) {
        GRANITE_FAIL(ErrorCode::InvalidState, "Tokenizer not loaded");
    }

    // Tokenize prompt
    auto prompt_tokens = tokenizer_.encode(prompt, true);

    // Clear KV cache for new generation
    kv_cache_.clear();

    // Create token tensor
    std::vector<int64_t> prompt_shape = {1, static_cast<int64_t>(prompt_tokens.size())};
    auto ids_result = Tensor::allocate(prompt_shape, DataType::INT32, backend_.get());
    if (!ids_result.ok()) {
        return ids_result.error();
    }
    auto ids = std::move(ids_result).take();

    // Copy tokens
    auto map_ids = backend_->map_buffer(ids.buffer());
    if (map_ids.ok()) {
        auto* ptr = static_cast<int32_t*>(map_ids.value());
        std::copy(prompt_tokens.begin(), prompt_tokens.end(), ptr);
        backend_->unmap_buffer(ids.buffer());
    }

    // Prefill
    auto logits_result = model_.forward(ids, &kv_cache_, 0);
    if (!logits_result.ok()) {
        return logits_result.error();
    }
    auto logits = std::move(logits_result).take();

    // Get last position logits and sample
    std::vector<int32_t> generated_tokens = prompt_tokens;

    for (int i = 0; i < config.max_tokens; i++) {
        if (cancelled_) {
            break;
        }

        // Sample next token
        int32_t next_token = sample(logits, config);

        // Check stop tokens
        if (std::find(config.stop_tokens.begin(), config.stop_tokens.end(),
                      next_token) != config.stop_tokens.end()) {
            break;
        }
        if (next_token == tokenizer_.eos_token()) {
            break;
        }

        // Decode and callback
        std::string token_str = tokenizer_.decode_token(next_token);
        if (!callback(token_str)) {
            break;
        }

        generated_tokens.push_back(next_token);

        // Generate next logits
        logits_result = model_.forward_single(next_token, kv_cache_);
        if (!logits_result.ok()) {
            return logits_result.error();
        }
        logits = std::move(logits_result).take();
    }

    return {};
}

void LLMRunner::cancel() {
    cancelled_ = true;
}

void LLMRunner::reset() {
    kv_cache_.clear();
}

int32_t LLMRunner::sample(const Tensor& logits, const GenerationConfig& config) {
    auto map_l = backend_->map_buffer(logits.buffer());
    if (!map_l.ok()) {
        return 0;
    }

    const auto* l = static_cast<const float*>(map_l.value());
    int seq_len = static_cast<int>(logits.size(1));
    int vocab_size = static_cast<int>(logits.size(2));

    const float* last_logits = l + (seq_len - 1) * vocab_size;
    std::vector<float> probs(last_logits, last_logits + vocab_size);

    backend_->unmap_buffer(logits.buffer());

    // Greedy decoding
    if (!config.do_sample) {
        return static_cast<int32_t>(
            std::max_element(probs.begin(), probs.end()) - probs.begin());
    }

    // Apply temperature
    if (config.temperature != 1.0f && config.temperature > 0) {
        for (auto& p : probs) {
            p /= config.temperature;
        }
    }

    // Softmax
    float max_val = *std::max_element(probs.begin(), probs.end());
    float sum = 0;
    for (auto& p : probs) {
        p = std::exp(p - max_val);
        sum += p;
    }
    for (auto& p : probs) {
        p /= sum;
    }

    // Top-k
    if (config.top_k > 0 && config.top_k < vocab_size) {
        std::vector<std::pair<float, int>> indexed;
        indexed.reserve(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            indexed.emplace_back(probs[i], i);
        }
        std::partial_sort(indexed.begin(), indexed.begin() + config.top_k,
                         indexed.end(), std::greater<>());

        float threshold = indexed[config.top_k - 1].first;
        for (int i = 0; i < vocab_size; i++) {
            if (probs[i] < threshold) {
                probs[i] = 0;
            }
        }

        sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (auto& p : probs) {
            p /= sum;
        }
    }

    // Top-p
    if (config.top_p < 1.0f) {
        std::vector<std::pair<float, int>> indexed;
        indexed.reserve(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            if (probs[i] > 0) {
                indexed.emplace_back(probs[i], i);
            }
        }
        std::sort(indexed.begin(), indexed.end(), std::greater<>());

        float cumsum = 0;
        size_t cutoff = indexed.size();
        for (size_t i = 0; i < indexed.size(); i++) {
            cumsum += indexed[i].first;
            if (cumsum > config.top_p) {
                cutoff = i + 1;
                break;
            }
        }

        std::unordered_set<int> keep;
        for (size_t i = 0; i < cutoff; i++) {
            keep.insert(indexed[i].second);
        }
        for (int i = 0; i < vocab_size; i++) {
            if (keep.find(i) == keep.end()) {
                probs[i] = 0;
            }
        }

        sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (auto& p : probs) {
            p /= sum;
        }
    }

    // Sample from distribution
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return dist(g_rng);
}

void LLMRunner::apply_repetition_penalty(
    std::vector<float>& logits,
    const std::vector<int32_t>& past_tokens,
    float penalty)
{
    if (penalty == 1.0f) return;

    for (int32_t token : past_tokens) {
        if (token >= 0 && token < static_cast<int32_t>(logits.size())) {
            if (logits[token] > 0) {
                logits[token] /= penalty;
            } else {
                logits[token] *= penalty;
            }
        }
    }
}

}  // namespace granite
