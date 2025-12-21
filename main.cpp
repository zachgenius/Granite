#include <granite/granite.h>
#include <iostream>
#include <string>
#include <chrono>
#include <cstring>
#include <cstdlib>

// Simple FP16 to FP32 conversion for debugging
inline float debug_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <model.gguf> [options] [prompt]\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  --draft-model <path>  Enable speculative decoding with draft model\n";
    std::cerr << "  --max-tokens <n>      Maximum tokens to generate (default: 20)\n";
    std::cerr << "  --spec-k <n>          Initial speculation depth (default: 4)\n";
    std::cerr << "\nExamples:\n";
    std::cerr << "  " << program << " model.gguf \"Hello\"\n";
    std::cerr << "  " << program << " large.gguf --draft-model small.gguf \"Hello\"\n";
}

int main(int argc, char* argv[]) {
    granite::init_logging(spdlog::level::info);

    GRANITE_LOG_INFO("Granite Embedded Inference Framework v{}", granite::version_string());

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse arguments
    std::string model_path;
    std::string draft_model_path;
    std::string prompt = "Hello";
    int max_tokens = 20;
    int spec_k = 4;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--draft-model" && i + 1 < argc) {
            draft_model_path = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--spec-k" && i + 1 < argc) {
            spec_k = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            if (model_path.empty()) {
                model_path = arg;
            } else {
                prompt = arg;
            }
        }
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Speculative decoding mode
    if (!draft_model_path.empty()) {
        GRANITE_LOG_INFO("Speculative decoding mode:");
        GRANITE_LOG_INFO("  Target model: {}", model_path);
        GRANITE_LOG_INFO("  Draft model:  {}", draft_model_path);
        GRANITE_LOG_INFO("  Initial K:    {}", spec_k);

        auto runner_result = granite::SpeculativeRunner::load(model_path, draft_model_path);
        if (!runner_result.ok()) {
            GRANITE_LOG_ERROR("Failed to load models: {}", runner_result.error().message());
            return 1;
        }
        auto runner = std::move(runner_result).take();

        granite::GenerationConfig gen_config;
        gen_config.max_tokens = max_tokens;
        gen_config.do_sample = false;  // Greedy for speculative

        granite::SpeculativeConfig spec_config;
        spec_config.initial_k = spec_k;

        GRANITE_LOG_INFO("Generating with speculative decoding...");
        std::cout << prompt << std::flush;

        auto start = std::chrono::high_resolution_clock::now();
        int token_count = 0;

        auto status = runner->generate_streaming(prompt, gen_config, [&](const std::string& token) {
            std::cout << token << std::flush;
            token_count++;
            return true;
        }, nullptr, spec_config);

        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "\n" << std::endl;

        if (!status.ok()) {
            GRANITE_LOG_ERROR("Generation failed: {}", status.error().message());
            return 1;
        }

        auto& stats = runner->stats();
        float tok_per_sec = token_count > 0 && ms > 0 ? (token_count * 1000.0f / ms) : 0;
        GRANITE_LOG_INFO("Generated {} tokens in {} ms ({:.1f} tok/s)",
                         token_count, ms, tok_per_sec);
        GRANITE_LOG_INFO("Speculative stats: {} drafted, {} accepted ({:.1f}%), {} target forwards",
                         stats.total_draft_tokens, stats.total_accepted_tokens,
                         stats.acceptance_rate() * 100.0f, stats.total_target_forwards);

        return 0;
    }

    GRANITE_LOG_INFO("Loading model: {}", model_path);

    // Load GGUF
    auto gguf_result = granite::GGUFFile::open(model_path);
    if (!gguf_result.ok()) {
        GRANITE_LOG_ERROR("Failed to open GGUF: {}", gguf_result.error().message());
        return 1;
    }
    auto& gguf = gguf_result.value();

    // Parse config
    auto config_result = granite::parse_model_config(gguf);
    if (!config_result.ok()) {
        GRANITE_LOG_ERROR("Failed to parse config: {}", config_result.error().message());
        return 1;
    }
    auto& config = config_result.value();

    GRANITE_LOG_INFO("Model: {} layers, {} hidden, {} heads",
                     config.num_layers, config.hidden_dim, config.num_heads);

    // Load tokenizer
    auto tok_result = granite::Tokenizer::from_gguf(gguf);
    if (!tok_result.ok()) {
        GRANITE_LOG_ERROR("Failed to load tokenizer: {}", tok_result.error().message());
        return 1;
    }
    auto& tokenizer = tok_result.value();

    // Create backend
    auto backend = granite::create_default_backend();
    if (!backend) {
        GRANITE_LOG_ERROR("Failed to create backend");
        return 1;
    }
    backend->initialize();

    // Load model
    GRANITE_LOG_INFO("Loading weights...");
    auto load_start = std::chrono::high_resolution_clock::now();

    auto model_result = granite::TransformerModel::load(model_path, backend.get());
    if (!model_result.ok()) {
        GRANITE_LOG_ERROR("Failed to load model: {}", model_result.error().message());
        return 1;
    }
    auto model = std::move(model_result).take();

    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
    GRANITE_LOG_INFO("Model loaded in {} ms", load_ms);

    // Allocate KV cache (CPU)
    auto cache_result = granite::KVCache::allocate(config, 512, backend.get());
    if (!cache_result.ok()) {
        GRANITE_LOG_ERROR("Failed to allocate KV cache: {}", cache_result.error().message());
        return 1;
    }
    auto kv_cache = std::move(cache_result).take();

#ifdef GRANITE_HAS_METAL
    // Allocate GPU KV cache for faster decode
    auto gpu_cache_result = model.allocate_gpu_kv_cache(512);
    if (!gpu_cache_result.ok()) {
        GRANITE_LOG_WARN("Failed to allocate GPU KV cache: {}", gpu_cache_result.error().message());
        // Continue without GPU KV cache - will use CPU path
    }

#endif

    // Debug: Check actual weight shapes and sample values
    GRANITE_LOG_INFO("Checking loaded weights:");
    std::vector<std::string> debug_weights = {
        "token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
        "blk.0.attn_v.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"
    };
    for (const auto& name : debug_weights) {
        const auto* w = model.get_weight(name);
        if (w) {
            std::string shape = "[";
            for (size_t i = 0; i < w->shape().size(); i++) {
                if (i > 0) shape += ", ";
                shape += std::to_string(w->shape()[i]);
            }
            shape += "]";

            // Sample a few values from the weight
            auto map_result = backend->map_buffer(w->buffer());
            if (map_result.ok()) {
                const uint16_t* data = static_cast<const uint16_t*>(map_result.value());
                std::string vals;
                for (int i = 0; i < 5; i++) {
                    if (i > 0) vals += ", ";
                    vals += std::to_string(debug_fp16_to_fp32(data[i]));
                }
                backend->unmap_buffer(w->buffer());
                GRANITE_LOG_INFO("  {}: {} {} first5=[{}]", name, granite::dtype_name(w->dtype()), shape, vals);
            } else {
                GRANITE_LOG_INFO("  {}: {} {}", name, granite::dtype_name(w->dtype()), shape);
            }
        } else {
            GRANITE_LOG_WARN("  {}: NOT FOUND", name);
        }
    }

    // Tokenize prompt
    GRANITE_LOG_INFO("Prompt: \"{}\"", prompt);
    auto tokens = tokenizer.encode(prompt, true);
    std::string token_ids_str;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) token_ids_str += ", ";
        token_ids_str += std::to_string(tokens[i]);
    }
    GRANITE_LOG_INFO("Token IDs: [{}]", token_ids_str);
    GRANITE_LOG_INFO("Tokens: {}", tokens.size());

    // Debug: Check embedding values for BOS token (token 1)
    const auto* emb_weight = model.get_weight("token_embd.weight");
    if (emb_weight) {
        int hidden_dim = config.hidden_dim;
        auto emb_map = backend->map_buffer(emb_weight->buffer());
        if (emb_map.ok()) {
            const uint16_t* emb_data = static_cast<const uint16_t*>(emb_map.value());
            // Check embedding at row 1 (BOS token)
            std::string bos_vals;
            for (int i = 0; i < 5; i++) {
                if (i > 0) bos_vals += ", ";
                bos_vals += std::to_string(debug_fp16_to_fp32(emb_data[1 * hidden_dim + i]));
            }
            GRANITE_LOG_INFO("Embedding for BOS (token 1) first5: [{}]", bos_vals);

            // Check if maybe it's transposed: column 1 instead of row 1
            std::string bos_col_vals;
            int vocab_size = config.vocab_size;
            for (int i = 0; i < 5; i++) {
                if (i > 0) bos_col_vals += ", ";
                bos_col_vals += std::to_string(debug_fp16_to_fp32(emb_data[i * vocab_size + 1]));
            }
            GRANITE_LOG_INFO("Embedding for BOS (token 1) as COLUMN first5: [{}]", bos_col_vals);

            backend->unmap_buffer(emb_weight->buffer());
        }
    }

    // Create token tensor
    std::vector<int64_t> token_shape = {1, static_cast<int64_t>(tokens.size())};
    auto token_tensor_result = granite::Tensor::allocate(token_shape, granite::DataType::INT32, backend.get());
    if (!token_tensor_result.ok()) {
        GRANITE_LOG_ERROR("Failed to allocate token tensor");
        return 1;
    }
    auto token_tensor = std::move(token_tensor_result).take();

    // Copy tokens to tensor
    auto map_result = backend->map_buffer(token_tensor.buffer());
    if (map_result.ok()) {
        int32_t* data = static_cast<int32_t*>(map_result.value());
        std::copy(tokens.begin(), tokens.end(), data);
        backend->unmap_buffer(token_tensor.buffer());
    }

    // Run forward pass
    GRANITE_LOG_INFO("Running forward pass...");
    auto forward_start = std::chrono::high_resolution_clock::now();

    auto logits_result = model.forward(token_tensor, &kv_cache, 0);

    auto forward_end = std::chrono::high_resolution_clock::now();
    auto forward_ms = std::chrono::duration_cast<std::chrono::milliseconds>(forward_end - forward_start).count();

    if (!logits_result.ok()) {
        GRANITE_LOG_ERROR("Forward pass failed: {}", logits_result.error().message());
        return 1;
    }
    auto& logits = logits_result.value();

    GRANITE_LOG_INFO("Forward pass completed in {} ms", forward_ms);
    GRANITE_LOG_INFO("Logits shape: [{}, {}, {}]",
                     logits.size(0), logits.size(1), logits.size(2));

    // Get top tokens from last position
    auto logits_map = backend->map_buffer(logits.buffer());
    if (logits_map.ok()) {
        const float* logits_data = static_cast<const float*>(logits_map.value());
        int seq_len = static_cast<int>(logits.size(1));
        int vocab_size = static_cast<int>(logits.size(2));

        // Get logits for last position
        const float* last_logits = logits_data + (seq_len - 1) * vocab_size;

        // Find top 5 tokens
        std::vector<std::pair<float, int>> scores;
        for (int i = 0; i < vocab_size; i++) {
            scores.emplace_back(last_logits[i], i);
        }
        std::partial_sort(scores.begin(), scores.begin() + 5, scores.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

        GRANITE_LOG_INFO("Top 5 next tokens:");
        for (int i = 0; i < 5; i++) {
            std::string token_str = tokenizer.decode_token(scores[i].second);
            GRANITE_LOG_INFO("  {}: \"{}\" (logit: {:.2f})",
                           scores[i].second, token_str, scores[i].first);
        }

        backend->unmap_buffer(logits.buffer());
    }

    // Simple generation loop
    GRANITE_LOG_INFO("");
    GRANITE_LOG_INFO("Generating text (greedy, max {} tokens)...", max_tokens);
    std::cout << prompt << std::flush;

    int max_new_tokens = max_tokens;
    for (int i = 0; i < max_new_tokens; i++) {
        // Get last token logits
        auto logits_map2 = backend->map_buffer(logits.buffer());
        if (!logits_map2.ok()) break;

        const float* logits_data = static_cast<const float*>(logits_map2.value());
        int seq_len = static_cast<int>(logits.size(1));
        int vocab_size = static_cast<int>(logits.size(2));
        const float* last_logits = logits_data + (seq_len - 1) * vocab_size;

        // Greedy: find max
        int next_token = 0;
        float max_logit = last_logits[0];
        for (int j = 1; j < vocab_size; j++) {
            if (last_logits[j] > max_logit) {
                max_logit = last_logits[j];
                next_token = j;
            }
        }

        // Debug: show what we're getting
        GRANITE_LOG_INFO("Step {}: seq_len={}, next_token={}, max_logit={:.2f}, first5_logits=[{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]",
            i, seq_len, next_token, max_logit,
            last_logits[0], last_logits[1], last_logits[2], last_logits[3], last_logits[4]);

        backend->unmap_buffer(logits.buffer());

        // Check for EOS
        if (next_token == tokenizer.eos_token()) {
            break;
        }

        // Decode and print
        std::string token_str = tokenizer.decode_token(next_token);
        std::cout << token_str << std::flush;

        // Forward single token
        auto single_result = model.forward_single(next_token, kv_cache);
        if (!single_result.ok()) {
            GRANITE_LOG_ERROR("Generation failed: {}", single_result.error().message());
            break;
        }
        logits = std::move(single_result).take();
    }

    std::cout << "\n" << std::endl;
    GRANITE_LOG_INFO("Generation complete!");

    backend->shutdown();
    return 0;
}
