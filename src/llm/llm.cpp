#include "granite/llm.h"
#include "granite/operators.h"
#include "granite/log.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>
#include <unordered_set>

namespace granite {

// =============================================================================
// FP16 Helpers
// =============================================================================

namespace {

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign << 31;
            float result;
            std::memcpy(&result, &bits, 4);
            return result;
        }
        while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
    } else if (exp == 31) {
        uint32_t bits = (sign << 31) | 0x7F800000 | (mant << 13);
        float result;
        std::memcpy(&result, &bits, 4);
        return result;
    }

    uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

// Thread-local RNG for sampling
thread_local std::mt19937 g_rng{std::random_device{}()};

}  // namespace

// =============================================================================
// KVCache Implementation
// =============================================================================

Result<KVCache> KVCache::allocate(
    const ModelConfig& config,
    int max_seq_len,
    IComputeBackend* backend)
{
    KVCache cache;
    cache.max_seq_len_ = max_seq_len;
    cache.num_kv_heads_ = config.num_kv_heads;
    cache.head_dim_ = config.head_dim;
    cache.backend_ = backend;
    cache.current_len_ = 0;

    // Allocate per-layer cache
    cache.layers_.reserve(config.num_layers);

    std::vector<int64_t> cache_shape = {
        1,
        static_cast<int64_t>(config.num_kv_heads),
        static_cast<int64_t>(max_seq_len),
        static_cast<int64_t>(config.head_dim)
    };

    for (int i = 0; i < config.num_layers; i++) {
        LayerCache layer;

        // Keys: [1, num_kv_heads, max_seq_len, head_dim]
        auto keys_result = Tensor::allocate(cache_shape, DataType::FP16, backend);
        if (!keys_result.ok()) {
            return keys_result.error();
        }
        layer.keys = std::move(keys_result).take();

        // Values: [1, num_kv_heads, max_seq_len, head_dim]
        auto values_result = Tensor::allocate(cache_shape, DataType::FP16, backend);
        if (!values_result.ok()) {
            return values_result.error();
        }
        layer.values = std::move(values_result).take();

        cache.layers_.push_back(std::move(layer));
    }

    size_t total_bytes = cache.memory_bytes();
    GRANITE_LOG_INFO("Allocated KV cache: {} layers, {} max seq len, {:.1f} MB",
                     config.num_layers, max_seq_len,
                     static_cast<float>(total_bytes) / (1024 * 1024));

    return cache;
}

Result<void> KVCache::append(int layer, const Tensor& keys, const Tensor& values) {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) {
        GRANITE_FAIL(ErrorCode::InvalidArgument, "Invalid layer index");
    }

    int new_seq_len = static_cast<int>(keys.size(2));
    if (current_len_ + new_seq_len > max_seq_len_) {
        GRANITE_FAIL(ErrorCode::InvalidArgument,
                     fmt::format("KV cache overflow: {} + {} > {}",
                                 current_len_, new_seq_len, max_seq_len_));
    }

    auto& lc = layers_[layer];

    // Copy new keys/values to cache at current position
    // This is a simplified copy - in practice we'd use a more efficient method
    auto map_k_src = backend_->map_buffer(keys.buffer());
    auto map_v_src = backend_->map_buffer(values.buffer());
    auto map_k_dst = backend_->map_buffer(lc.keys.buffer());
    auto map_v_dst = backend_->map_buffer(lc.values.buffer());

    if (!map_k_src.ok() || !map_v_src.ok() || !map_k_dst.ok() || !map_v_dst.ok()) {
        return Error(ErrorCode::InternalError, "Failed to map buffers");
    }

    const auto* k_src = static_cast<const uint16_t*>(map_k_src.value());
    const auto* v_src = static_cast<const uint16_t*>(map_v_src.value());
    auto* k_dst = static_cast<uint16_t*>(map_k_dst.value());
    auto* v_dst = static_cast<uint16_t*>(map_v_dst.value());

    // Copy to the right position in cache
    // Shape: [1, num_kv_heads, seq_len, head_dim]
    size_t stride = static_cast<size_t>(max_seq_len_) * head_dim_;
    for (int h = 0; h < num_kv_heads_; h++) {
        size_t dst_offset = h * stride + current_len_ * head_dim_;
        size_t src_offset = h * new_seq_len * head_dim_;
        std::memcpy(k_dst + dst_offset, k_src + src_offset,
                    new_seq_len * head_dim_ * sizeof(uint16_t));
        std::memcpy(v_dst + dst_offset, v_src + src_offset,
                    new_seq_len * head_dim_ * sizeof(uint16_t));
    }

    backend_->unmap_buffer(keys.buffer());
    backend_->unmap_buffer(values.buffer());
    backend_->unmap_buffer(lc.keys.buffer());
    backend_->unmap_buffer(lc.values.buffer());

    // Only update current_len_ on the last layer to keep consistency
    if (layer == static_cast<int>(layers_.size()) - 1) {
        current_len_ += new_seq_len;
    }

    return {};
}

std::pair<Tensor, Tensor> KVCache::get(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) {
        // Return empty tensors
        return {Tensor{}, Tensor{}};
    }

    // Return views of the cache up to current_len_
    // For now, return the full tensors - the attention kernel will use current_len_
    return {layers_[layer].keys, layers_[layer].values};
}

void KVCache::clear() {
    current_len_ = 0;
}

size_t KVCache::memory_bytes() const {
    size_t total = 0;
    for (const auto& lc : layers_) {
        total += lc.keys.size_bytes() + lc.values.size_bytes();
    }
    return total;
}

// =============================================================================
// RoPE Cache Implementation
// =============================================================================

void RoPECache::initialize(int max_seq_len, int head_dim, float theta) {
    max_seq_len_ = max_seq_len;
    head_dim_ = head_dim;

    int half_dim = head_dim / 2;
    cos_.resize(max_seq_len * half_dim);
    sin_.resize(max_seq_len * half_dim);

    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
            float angle = static_cast<float>(pos) * freq;
            cos_[pos * half_dim + i] = std::cos(angle);
            sin_[pos * half_dim + i] = std::sin(angle);
        }
    }

    GRANITE_LOG_DEBUG("Initialized RoPE cache: max_seq={}, head_dim={}, theta={}",
                      max_seq_len, head_dim, theta);
}

Result<void> RoPECache::apply(Tensor& q, Tensor& k, int start_pos, IComputeBackend* backend) {
    if (!is_initialized()) {
        GRANITE_FAIL(ErrorCode::InvalidState, "RoPE cache not initialized");
    }

    // Q/K shape: [batch, seq_len, num_heads, head_dim]
    int seq_len = static_cast<int>(q.size(1));
    int num_heads_q = static_cast<int>(q.size(2));
    int num_heads_k = static_cast<int>(k.size(2));

    if (start_pos + seq_len > max_seq_len_) {
        GRANITE_FAIL(ErrorCode::InvalidArgument,
                     fmt::format("RoPE position out of range: {} + {} > {}",
                                 start_pos, seq_len, max_seq_len_));
    }

    auto map_q = backend->map_buffer(q.buffer());
    auto map_k = backend->map_buffer(k.buffer());

    if (!map_q.ok() || !map_k.ok()) {
        return Error(ErrorCode::InternalError, "Failed to map buffers");
    }

    auto* q_data = static_cast<float*>(map_q.value());
    auto* k_data = static_cast<float*>(map_k.value());

    int half_dim = head_dim_ / 2;

    // Apply RoPE to Q
    for (int s = 0; s < seq_len; s++) {
        int pos = start_pos + s;
        for (int h = 0; h < num_heads_q; h++) {
            float* q_ptr = q_data + s * num_heads_q * head_dim_ + h * head_dim_;
            for (int i = 0; i < half_dim; i++) {
                float cos_val = cos_[pos * half_dim + i];
                float sin_val = sin_[pos * half_dim + i];

                float q0 = q_ptr[2 * i];
                float q1 = q_ptr[2 * i + 1];

                q_ptr[2 * i]     = q0 * cos_val - q1 * sin_val;
                q_ptr[2 * i + 1] = q0 * sin_val + q1 * cos_val;
            }
        }
    }

    // Apply RoPE to K
    for (int s = 0; s < seq_len; s++) {
        int pos = start_pos + s;
        for (int h = 0; h < num_heads_k; h++) {
            float* k_ptr = k_data + s * num_heads_k * head_dim_ + h * head_dim_;
            for (int i = 0; i < half_dim; i++) {
                float cos_val = cos_[pos * half_dim + i];
                float sin_val = sin_[pos * half_dim + i];

                float k0 = k_ptr[2 * i];
                float k1 = k_ptr[2 * i + 1];

                k_ptr[2 * i]     = k0 * cos_val - k1 * sin_val;
                k_ptr[2 * i + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }

    backend->unmap_buffer(q.buffer());
    backend->unmap_buffer(k.buffer());

    return {};
}

// =============================================================================
// Tokenizer Implementation
// =============================================================================

Result<Tokenizer> Tokenizer::from_gguf(const GGUFFile& gguf) {
    Tokenizer tok;

    // Load vocabulary
    auto tokens = gguf.get_metadata_as<std::vector<std::string>>("tokenizer.ggml.tokens");
    if (!tokens) {
        GRANITE_FAIL(ErrorCode::InvalidFormat, "No tokenizer vocabulary in GGUF");
    }
    tok.vocab_ = *tokens;

    // Build token to ID map
    for (size_t i = 0; i < tok.vocab_.size(); i++) {
        tok.token_to_id_[tok.vocab_[i]] = static_cast<int32_t>(i);
    }

    // Load special tokens
    if (auto bos = gguf.get_metadata_as<uint32_t>("tokenizer.ggml.bos_token_id")) {
        tok.bos_token_ = static_cast<int32_t>(*bos);
    }
    if (auto eos = gguf.get_metadata_as<uint32_t>("tokenizer.ggml.eos_token_id")) {
        tok.eos_token_ = static_cast<int32_t>(*eos);
    }
    if (auto pad = gguf.get_metadata_as<uint32_t>("tokenizer.ggml.padding_token_id")) {
        tok.pad_token_ = static_cast<int32_t>(*pad);
    }
    if (auto unk = gguf.get_metadata_as<uint32_t>("tokenizer.ggml.unknown_token_id")) {
        tok.unk_token_ = static_cast<int32_t>(*unk);
    }

    // Load merges (BPE)
    auto merges = gguf.get_metadata_as<std::vector<std::string>>("tokenizer.ggml.merges");
    if (merges) {
        for (size_t i = 0; i < merges->size(); i++) {
            const auto& merge = (*merges)[i];
            size_t space_pos = merge.find(' ');
            if (space_pos != std::string::npos) {
                std::string first = merge.substr(0, space_pos);
                std::string second = merge.substr(space_pos + 1);
                tok.merges_.emplace_back(first, second);
                tok.merge_ranks_[{first, second}] = static_cast<int>(i);
            }
        }
    }

    GRANITE_LOG_INFO("Loaded tokenizer: vocab_size={}, bos={}, eos={}",
                     tok.vocab_.size(), tok.bos_token_, tok.eos_token_);

    return tok;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> tokens;

    if (add_bos) {
        tokens.push_back(bos_token_);
    }

    if (text.empty()) {
        return tokens;
    }

    // Simple character-level tokenization as fallback
    // A full BPE implementation would be more complex
    std::vector<std::string> chars;
    for (size_t i = 0; i < text.size(); ) {
        // Handle UTF-8 (simplified)
        unsigned char c = static_cast<unsigned char>(text[i]);
        int char_len = 1;
        if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        chars.push_back(text.substr(i, char_len));
        i += char_len;
    }

    // Apply BPE merges
    while (chars.size() > 1) {
        int best_idx = -1;
        int best_rank = INT_MAX;

        for (size_t i = 0; i < chars.size() - 1; i++) {
            auto it = merge_ranks_.find({chars[i], chars[i + 1]});
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx < 0) break;

        // Merge
        chars[best_idx] = chars[best_idx] + chars[best_idx + 1];
        chars.erase(chars.begin() + best_idx + 1);
    }

    // Convert to token IDs
    for (const auto& tok_str : chars) {
        auto it = token_to_id_.find(tok_str);
        if (it != token_to_id_.end()) {
            tokens.push_back(it->second);
        } else {
            // Try to find as byte tokens
            for (unsigned char c : tok_str) {
                // Many tokenizers use byte fallback tokens
                std::string byte_tok = "<0x" + fmt::format("{:02X}", c) + ">";
                auto byte_it = token_to_id_.find(byte_tok);
                if (byte_it != token_to_id_.end()) {
                    tokens.push_back(byte_it->second);
                } else {
                    tokens.push_back(unk_token_);
                }
            }
        }
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string result;
    for (int32_t id : tokens) {
        result += decode_token(id);
    }
    return result;
}

std::string Tokenizer::decode_token(int32_t token) const {
    if (token == bos_token_ || token == eos_token_ || token == pad_token_) {
        return "";
    }
    if (token >= 0 && token < static_cast<int32_t>(vocab_.size())) {
        const std::string& tok = vocab_[token];
        // Handle special tokens like "▁" (sentencepiece space)
        std::string result = tok;
        size_t pos = 0;
        while ((pos = result.find("▁", pos)) != std::string::npos) {
            result.replace(pos, 3, " ");  // "▁" is 3 bytes in UTF-8
            pos += 1;
        }
        return result;
    }
    return "";
}

// =============================================================================
// Model Config Parsing
// =============================================================================

Result<ModelConfig> parse_model_config(const GGUFFile& gguf) {
    ModelConfig config;

    // Get architecture
    auto arch = gguf.get_architecture();
    if (!arch) {
        GRANITE_FAIL(ErrorCode::InvalidFormat, "No architecture in GGUF");
    }
    config.architecture = *arch;

    // Get model name
    if (auto name = gguf.get_metadata_as<std::string>("general.name")) {
        config.name = *name;
    }

    // Architecture-specific parameters
    std::string prefix = config.architecture + ".";

    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "context_length")) {
        config.max_seq_len = static_cast<int>(*v);
    }
    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "embedding_length")) {
        config.hidden_dim = static_cast<int>(*v);
    }
    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "feed_forward_length")) {
        config.intermediate_dim = static_cast<int>(*v);
    }
    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "block_count")) {
        config.num_layers = static_cast<int>(*v);
    }
    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "attention.head_count")) {
        config.num_heads = static_cast<int>(*v);
    }
    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "attention.head_count_kv")) {
        config.num_kv_heads = static_cast<int>(*v);
    } else {
        config.num_kv_heads = config.num_heads;  // Default to MHA
    }

    // RoPE parameters
    if (auto v = gguf.get_metadata_as<float>(prefix + "rope.freq_base")) {
        config.rope_theta = *v;
    }
    if (auto v = gguf.get_metadata_as<float>(prefix + "attention.layer_norm_rms_epsilon")) {
        config.rms_norm_eps = *v;
    }

    // Vocab size from embedding tensor
    if (auto* emb = gguf.find_tensor("token_embd.weight")) {
        config.vocab_size = static_cast<int>(emb->dimensions[0]);
    }

    // Compute derived values
    config.compute_derived();

    GRANITE_LOG_INFO("Model config: arch={}, layers={}, hidden={}, heads={}/{}, vocab={}",
                     config.architecture, config.num_layers, config.hidden_dim,
                     config.num_heads, config.num_kv_heads, config.vocab_size);

    return config;
}

// =============================================================================
// TransformerModel Implementation
// =============================================================================

Result<TransformerModel> TransformerModel::load(
    const std::string& path,
    IComputeBackend* backend)
{
    TransformerModel model;
    model.backend_ = backend;

    // Open GGUF file
    auto gguf_result = GGUFFile::open(path);
    if (!gguf_result.ok()) {
        return gguf_result.error();
    }
    model.gguf_ = std::make_unique<GGUFFile>(std::move(gguf_result).take());

    // Parse config
    auto config_result = parse_model_config(*model.gguf_);
    if (!config_result.ok()) {
        return config_result.error();
    }
    model.config_ = std::move(config_result).take();

    // Initialize RoPE cache
    model.rope_cache_.initialize(
        model.config_.max_seq_len,
        model.config_.head_dim,
        model.config_.rope_theta);

    // Load weights
    ModelLoader loader(backend);
    auto weights_result = loader.load_weights(*model.gguf_);
    if (!weights_result.ok()) {
        return weights_result.error();
    }
    model.weights_ = std::move(weights_result).take();

    GRANITE_LOG_INFO("Loaded model: {} weights", model.weights_.size());

    return model;
}

const Tensor* TransformerModel::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        return nullptr;
    }
    return &it->second;
}

Result<Tensor> TransformerModel::embed(const Tensor& token_ids) {
    const Tensor* emb_weight = get_weight("token_embd.weight");
    if (!emb_weight) {
        GRANITE_FAIL(ErrorCode::InvalidState, "Embedding weight not found");
    }

    // token_ids: [batch, seq_len] INT32
    // emb_weight: [vocab_size, hidden_dim]
    // output: [batch, seq_len, hidden_dim]

    int batch = static_cast<int>(token_ids.size(0));
    int seq_len = static_cast<int>(token_ids.size(1));
    int hidden_dim = config_.hidden_dim;

    std::vector<int64_t> output_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(hidden_dim)
    };
    auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        return output_result.error();
    }
    auto output = std::move(output_result).take();

    // Map buffers
    auto map_ids = backend_->map_buffer(token_ids.buffer());
    auto map_emb = backend_->map_buffer(emb_weight->buffer());
    auto map_out = backend_->map_buffer(output.buffer());

    if (!map_ids.ok() || !map_emb.ok() || !map_out.ok()) {
        return Error(ErrorCode::InternalError, "Failed to map buffers");
    }

    const auto* ids = static_cast<const int32_t*>(map_ids.value());
    const auto* emb = static_cast<const uint16_t*>(map_emb.value());  // FP16
    auto* out = static_cast<float*>(map_out.value());

    // Embedding lookup
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            int token_id = ids[b * seq_len + s];
            if (token_id < 0 || token_id >= config_.vocab_size) {
                token_id = 0;  // Fallback to first token
            }

            const uint16_t* emb_row = emb + token_id * hidden_dim;
            float* out_row = out + (b * seq_len + s) * hidden_dim;

            for (int d = 0; d < hidden_dim; d++) {
                out_row[d] = fp16_to_fp32(emb_row[d]);
            }
        }
    }

    backend_->unmap_buffer(token_ids.buffer());
    backend_->unmap_buffer(emb_weight->buffer());
    backend_->unmap_buffer(output.buffer());

    return output;
}

Result<Tensor> TransformerModel::forward(
    const Tensor& token_ids,
    KVCache* kv_cache,
    int start_pos)
{
    // Embed tokens
    auto hidden_result = embed(token_ids);
    if (!hidden_result.ok()) {
        return hidden_result.error();
    }
    auto hidden = std::move(hidden_result).take();

    // Process through transformer layers
    for (int layer = 0; layer < config_.num_layers; layer++) {
        auto block_result = transformer_block(hidden, layer, kv_cache, start_pos);
        if (!block_result.ok()) {
            return block_result.error();
        }
        hidden = std::move(block_result).take();
    }

    // Final RMSNorm
    const Tensor* norm_weight = get_weight("output_norm.weight");
    if (norm_weight) {
        // Apply RMSNorm (simplified inline implementation)
        auto map_h = backend_->map_buffer(hidden.buffer());
        auto map_w = backend_->map_buffer(norm_weight->buffer());

        if (map_h.ok() && map_w.ok()) {
            auto* h = static_cast<float*>(map_h.value());
            const auto* w = static_cast<const uint16_t*>(map_w.value());

            int batch = static_cast<int>(hidden.size(0));
            int seq_len = static_cast<int>(hidden.size(1));
            int dim = config_.hidden_dim;

            for (int b = 0; b < batch; b++) {
                for (int s = 0; s < seq_len; s++) {
                    float* row = h + (b * seq_len + s) * dim;

                    // Compute RMS
                    float sum_sq = 0;
                    for (int d = 0; d < dim; d++) {
                        sum_sq += row[d] * row[d];
                    }
                    float rms = std::sqrt(sum_sq / dim + config_.rms_norm_eps);
                    float inv_rms = 1.0f / rms;

                    // Normalize and scale
                    for (int d = 0; d < dim; d++) {
                        row[d] = row[d] * inv_rms * fp16_to_fp32(w[d]);
                    }
                }
            }

            backend_->unmap_buffer(hidden.buffer());
            backend_->unmap_buffer(norm_weight->buffer());
        }
    }

    // Output projection (lm_head)
    const Tensor* output_weight = get_weight("output.weight");
    if (!output_weight) {
        // Try tied embeddings
        output_weight = get_weight("token_embd.weight");
    }
    if (!output_weight) {
        GRANITE_FAIL(ErrorCode::InvalidState, "Output weight not found");
    }

    // MatMul: hidden @ output_weight.T -> logits
    int batch = static_cast<int>(hidden.size(0));
    int seq_len = static_cast<int>(hidden.size(1));

    std::vector<int64_t> logits_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(config_.vocab_size)
    };
    auto logits_result = Tensor::allocate(logits_shape, DataType::FP32, backend_);
    if (!logits_result.ok()) {
        return logits_result.error();
    }
    auto logits = std::move(logits_result).take();

    // Simple matmul implementation
    auto map_h = backend_->map_buffer(hidden.buffer());
    auto map_w = backend_->map_buffer(output_weight->buffer());
    auto map_l = backend_->map_buffer(logits.buffer());

    if (!map_h.ok() || !map_w.ok() || !map_l.ok()) {
        return Error(ErrorCode::InternalError, "Failed to map buffers");
    }

    const auto* h = static_cast<const float*>(map_h.value());
    const auto* w = static_cast<const uint16_t*>(map_w.value());  // [vocab, hidden]
    auto* l = static_cast<float*>(map_l.value());

    int hidden_dim = config_.hidden_dim;
    int vocab_size = config_.vocab_size;

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            const float* h_row = h + (b * seq_len + s) * hidden_dim;
            float* l_row = l + (b * seq_len + s) * vocab_size;

            for (int v = 0; v < vocab_size; v++) {
                float sum = 0;
                const uint16_t* w_row = w + v * hidden_dim;
                for (int d = 0; d < hidden_dim; d++) {
                    sum += h_row[d] * fp16_to_fp32(w_row[d]);
                }
                l_row[v] = sum;
            }
        }
    }

    backend_->unmap_buffer(hidden.buffer());
    backend_->unmap_buffer(output_weight->buffer());
    backend_->unmap_buffer(logits.buffer());

    return logits;
}

Result<Tensor> TransformerModel::forward_single(int32_t token_id, KVCache& kv_cache) {
    // Create single-token tensor
    std::vector<int64_t> ids_shape = {1, 1};
    auto ids_result = Tensor::allocate(ids_shape, DataType::INT32, backend_);
    if (!ids_result.ok()) {
        return ids_result.error();
    }
    auto ids = std::move(ids_result).take();

    auto map_ids = backend_->map_buffer(ids.buffer());
    if (map_ids.ok()) {
        auto* ptr = static_cast<int32_t*>(map_ids.value());
        ptr[0] = token_id;
        backend_->unmap_buffer(ids.buffer());
    }

    int start_pos = kv_cache.seq_len();
    return forward(ids, &kv_cache, start_pos);
}

// Helper: Apply RMSNorm in place
Tensor TransformerModel::apply_rms_norm(const Tensor& input, const Tensor* weight) {
    int batch = static_cast<int>(input.size(0));
    int seq_len = static_cast<int>(input.size(1));
    int dim = static_cast<int>(input.size(2));

    // Allocate output
    std::vector<int64_t> shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(dim)
    };
    auto output_result = Tensor::allocate(shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        return input;  // Return input unchanged on failure
    }
    auto output = std::move(output_result).take();

    auto map_in = backend_->map_buffer(input.buffer());
    auto map_out = backend_->map_buffer(output.buffer());

    if (!map_in.ok() || !map_out.ok()) {
        return input;
    }

    const float* in_data = static_cast<const float*>(map_in.value());
    float* out_data = static_cast<float*>(map_out.value());

    const uint16_t* w_data = nullptr;
    if (weight) {
        auto map_w = backend_->map_buffer(weight->buffer());
        if (map_w.ok()) {
            w_data = static_cast<const uint16_t*>(map_w.value());
        }
    }

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            const float* row = in_data + (b * seq_len + s) * dim;
            float* out_row = out_data + (b * seq_len + s) * dim;

            // Compute RMS
            float sum_sq = 0;
            for (int d = 0; d < dim; d++) {
                sum_sq += row[d] * row[d];
            }
            float rms = std::sqrt(sum_sq / dim + config_.rms_norm_eps);
            float inv_rms = 1.0f / rms;

            // Normalize and scale
            for (int d = 0; d < dim; d++) {
                float val = row[d] * inv_rms;
                if (w_data) {
                    val *= fp16_to_fp32(w_data[d]);
                }
                out_row[d] = val;
            }
        }
    }

    backend_->unmap_buffer(input.buffer());
    backend_->unmap_buffer(output.buffer());
    if (weight) {
        backend_->unmap_buffer(weight->buffer());
    }

    return output;
}

// Helper: Add two tensors element-wise
Tensor TransformerModel::add_tensors(const Tensor& a, const Tensor& b) {
    int batch = static_cast<int>(a.size(0));
    int seq_len = static_cast<int>(a.size(1));
    int dim = static_cast<int>(a.size(2));

    std::vector<int64_t> shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(dim)
    };
    auto output_result = Tensor::allocate(shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        return a;
    }
    auto output = std::move(output_result).take();

    auto map_a = backend_->map_buffer(a.buffer());
    auto map_b = backend_->map_buffer(b.buffer());
    auto map_out = backend_->map_buffer(output.buffer());

    if (!map_a.ok() || !map_b.ok() || !map_out.ok()) {
        return a;
    }

    const float* a_data = static_cast<const float*>(map_a.value());
    const float* b_data = static_cast<const float*>(map_b.value());
    float* out_data = static_cast<float*>(map_out.value());

    size_t total = static_cast<size_t>(batch) * seq_len * dim;
    for (size_t i = 0; i < total; i++) {
        out_data[i] = a_data[i] + b_data[i];
    }

    backend_->unmap_buffer(a.buffer());
    backend_->unmap_buffer(b.buffer());
    backend_->unmap_buffer(output.buffer());

    return output;
}

Result<Tensor> TransformerModel::transformer_block(
    const Tensor& hidden,
    int layer,
    KVCache* kv_cache,
    int start_pos)
{
    std::string prefix = "blk." + std::to_string(layer) + ".";

    int batch = static_cast<int>(hidden.size(0));
    int seq_len = static_cast<int>(hidden.size(1));
    int hidden_dim = config_.hidden_dim;

    // Get weights
    const Tensor* attn_norm_weight = get_weight(prefix + "attn_norm.weight");
    const Tensor* ffn_norm_weight = get_weight(prefix + "ffn_norm.weight");

    // Clone hidden for residual (we'll modify it in place for RMSNorm)
    std::vector<int64_t> hidden_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(hidden_dim)
    };
    auto residual_result = Tensor::allocate(hidden_shape, DataType::FP32, backend_);
    if (!residual_result.ok()) {
        return residual_result.error();
    }
    auto residual = std::move(residual_result).take();

    // Copy hidden to residual
    auto map_h = backend_->map_buffer(hidden.buffer());
    auto map_r = backend_->map_buffer(residual.buffer());
    if (map_h.ok() && map_r.ok()) {
        std::memcpy(map_r.value(), map_h.value(), hidden.size_bytes());
        backend_->unmap_buffer(hidden.buffer());
        backend_->unmap_buffer(residual.buffer());
    }

    // 1. RMSNorm for attention
    auto attn_input = apply_rms_norm(hidden, attn_norm_weight);

    // 2. Attention with KV cache
    auto attn_result = attention(attn_input, layer, kv_cache, start_pos);
    if (!attn_result.ok()) {
        return attn_result.error();
    }
    auto attn_output = std::move(attn_result).take();

    // 3. Residual add: hidden = residual + attn_output
    auto post_attn = add_tensors(residual, attn_output);

    // 4. RMSNorm for FFN
    auto ffn_input = apply_rms_norm(post_attn, ffn_norm_weight);

    // 5. Feed forward (SwiGLU)
    auto ffn_result = feed_forward(ffn_input, layer);
    if (!ffn_result.ok()) {
        return ffn_result.error();
    }
    auto ffn_output = std::move(ffn_result).take();

    // 6. Residual add: output = post_attn + ffn_output
    return add_tensors(post_attn, ffn_output);
}

Result<Tensor> TransformerModel::attention(
    const Tensor& hidden,
    int layer,
    KVCache* kv_cache,
    int start_pos)
{
    std::string prefix = "blk." + std::to_string(layer) + ".";

    // Get attention weights
    const Tensor* wq = get_weight(prefix + "attn_q.weight");
    const Tensor* wk = get_weight(prefix + "attn_k.weight");
    const Tensor* wv = get_weight(prefix + "attn_v.weight");
    const Tensor* wo = get_weight(prefix + "attn_output.weight");

    if (!wq || !wk || !wv || !wo) {
        GRANITE_FAIL(ErrorCode::InvalidState,
                     "Missing attention weights for layer " + std::to_string(layer));
    }

    int batch = static_cast<int>(hidden.size(0));
    int seq_len = static_cast<int>(hidden.size(1));
    int hidden_dim = config_.hidden_dim;
    int num_heads = config_.num_heads;
    int num_kv_heads = config_.num_kv_heads;
    int head_dim = config_.head_dim;
    int kv_dim = num_kv_heads * head_dim;

    // Allocate Q, K, V tensors
    std::vector<int64_t> q_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(num_heads),
        static_cast<int64_t>(head_dim)
    };
    std::vector<int64_t> kv_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(num_kv_heads),
        static_cast<int64_t>(head_dim)
    };

    auto q_result = Tensor::allocate(q_shape, DataType::FP32, backend_);
    auto k_result = Tensor::allocate(kv_shape, DataType::FP32, backend_);
    auto v_result = Tensor::allocate(kv_shape, DataType::FP32, backend_);

    if (!q_result.ok() || !k_result.ok() || !v_result.ok()) {
        GRANITE_FAIL(ErrorCode::AllocationFailed, "Failed to allocate Q/K/V tensors");
    }

    auto q = std::move(q_result).take();
    auto k = std::move(k_result).take();
    auto v = std::move(v_result).take();

    // Project Q, K, V
    // hidden: [batch, seq_len, hidden_dim]
    // wq: [num_heads * head_dim, hidden_dim]
    // output: [batch, seq_len, num_heads, head_dim]
    auto map_h = backend_->map_buffer(hidden.buffer());
    auto map_q = backend_->map_buffer(q.buffer());
    auto map_k = backend_->map_buffer(k.buffer());
    auto map_v = backend_->map_buffer(v.buffer());
    auto map_wq = backend_->map_buffer(wq->buffer());
    auto map_wk = backend_->map_buffer(wk->buffer());
    auto map_wv = backend_->map_buffer(wv->buffer());

    if (!map_h.ok() || !map_q.ok() || !map_k.ok() || !map_v.ok() ||
        !map_wq.ok() || !map_wk.ok() || !map_wv.ok()) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to map attention buffers");
    }

    const float* h_data = static_cast<const float*>(map_h.value());
    float* q_data = static_cast<float*>(map_q.value());
    float* k_data = static_cast<float*>(map_k.value());
    float* v_data = static_cast<float*>(map_v.value());
    const uint16_t* wq_data = static_cast<const uint16_t*>(map_wq.value());
    const uint16_t* wk_data = static_cast<const uint16_t*>(map_wk.value());
    const uint16_t* wv_data = static_cast<const uint16_t*>(map_wv.value());

    // Q projection: [batch, seq, hidden] @ [hidden, num_heads*head_dim] -> [batch, seq, num_heads, head_dim]
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            const float* h_row = h_data + (b * seq_len + s) * hidden_dim;

            // Q
            float* q_row = q_data + (b * seq_len + s) * num_heads * head_dim;
            for (int i = 0; i < num_heads * head_dim; i++) {
                float sum = 0;
                for (int d = 0; d < hidden_dim; d++) {
                    sum += h_row[d] * fp16_to_fp32(wq_data[i * hidden_dim + d]);
                }
                q_row[i] = sum;
            }

            // K
            float* k_row = k_data + (b * seq_len + s) * num_kv_heads * head_dim;
            for (int i = 0; i < kv_dim; i++) {
                float sum = 0;
                for (int d = 0; d < hidden_dim; d++) {
                    sum += h_row[d] * fp16_to_fp32(wk_data[i * hidden_dim + d]);
                }
                k_row[i] = sum;
            }

            // V
            float* v_row = v_data + (b * seq_len + s) * num_kv_heads * head_dim;
            for (int i = 0; i < kv_dim; i++) {
                float sum = 0;
                for (int d = 0; d < hidden_dim; d++) {
                    sum += h_row[d] * fp16_to_fp32(wv_data[i * hidden_dim + d]);
                }
                v_row[i] = sum;
            }
        }
    }

    backend_->unmap_buffer(hidden.buffer());
    backend_->unmap_buffer(wq->buffer());
    backend_->unmap_buffer(wk->buffer());
    backend_->unmap_buffer(wv->buffer());

    // Apply RoPE to Q and K
    auto rope_result = rope_cache_.apply(q, k, start_pos, backend_);
    if (!rope_result.ok()) {
        GRANITE_LOG_WARN("RoPE failed: {}", rope_result.error().message());
    }

    backend_->unmap_buffer(q.buffer());
    backend_->unmap_buffer(k.buffer());
    backend_->unmap_buffer(v.buffer());

    // Update KV cache if provided
    if (kv_cache) {
        // Convert K, V to FP16 for cache
        std::vector<int64_t> kv_cache_shape = {
            1,
            static_cast<int64_t>(num_kv_heads),
            static_cast<int64_t>(seq_len),
            static_cast<int64_t>(head_dim)
        };
        auto k_fp16_result = Tensor::allocate(kv_cache_shape, DataType::FP16, backend_);
        auto v_fp16_result = Tensor::allocate(kv_cache_shape, DataType::FP16, backend_);

        if (k_fp16_result.ok() && v_fp16_result.ok()) {
            auto k_fp16 = std::move(k_fp16_result).take();
            auto v_fp16 = std::move(v_fp16_result).take();

            // Convert to FP16 (simplified - assumes batch=1)
            auto map_k_src = backend_->map_buffer(k.buffer());
            auto map_v_src = backend_->map_buffer(v.buffer());
            auto map_k_dst = backend_->map_buffer(k_fp16.buffer());
            auto map_v_dst = backend_->map_buffer(v_fp16.buffer());

            if (map_k_src.ok() && map_v_src.ok() && map_k_dst.ok() && map_v_dst.ok()) {
                const float* k_src = static_cast<const float*>(map_k_src.value());
                const float* v_src = static_cast<const float*>(map_v_src.value());
                uint16_t* k_dst = static_cast<uint16_t*>(map_k_dst.value());
                uint16_t* v_dst = static_cast<uint16_t*>(map_v_dst.value());

                // Transpose and convert: [batch, seq, num_kv_heads, head_dim] -> [1, num_kv_heads, seq, head_dim]
                for (int h = 0; h < num_kv_heads; h++) {
                    for (int s = 0; s < seq_len; s++) {
                        for (int d = 0; d < head_dim; d++) {
                            int src_idx = s * num_kv_heads * head_dim + h * head_dim + d;
                            int dst_idx = h * seq_len * head_dim + s * head_dim + d;
                            k_dst[dst_idx] = fp32_to_fp16(k_src[src_idx]);
                            v_dst[dst_idx] = fp32_to_fp16(v_src[src_idx]);
                        }
                    }
                }

                backend_->unmap_buffer(k.buffer());
                backend_->unmap_buffer(v.buffer());
                backend_->unmap_buffer(k_fp16.buffer());
                backend_->unmap_buffer(v_fp16.buffer());

                kv_cache->append(layer, k_fp16, v_fp16);
            }
        }
    }

    // Compute attention scores and output
    // For now, simplified single-sequence attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Total sequence length (including cached)
    int total_seq = kv_cache ? kv_cache->seq_len() : seq_len;

    // Allocate attention output
    std::vector<int64_t> output_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(hidden_dim)
    };
    auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        return output_result.error();
    }
    auto output = std::move(output_result).take();

    // Map output and weight
    auto map_out = backend_->map_buffer(output.buffer());
    auto map_wo = backend_->map_buffer(wo->buffer());
    map_q = backend_->map_buffer(q.buffer());

    if (!map_out.ok() || !map_wo.ok() || !map_q.ok()) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to map output buffers");
    }

    float* out_data = static_cast<float*>(map_out.value());
    const uint16_t* wo_data = static_cast<const uint16_t*>(map_wo.value());
    q_data = static_cast<float*>(map_q.value());

    // Get K, V from cache or use current
    const float* k_all = nullptr;
    const float* v_all = nullptr;
    std::vector<float> k_buffer, v_buffer;

    if (kv_cache && kv_cache->seq_len() > 0) {
        auto [cached_k, cached_v] = kv_cache->get(layer);
        auto map_ck = backend_->map_buffer(cached_k.buffer());
        auto map_cv = backend_->map_buffer(cached_v.buffer());

        if (map_ck.ok() && map_cv.ok()) {
            // Convert cached FP16 to FP32 and transpose
            const uint16_t* ck = static_cast<const uint16_t*>(map_ck.value());
            const uint16_t* cv = static_cast<const uint16_t*>(map_cv.value());

            k_buffer.resize(total_seq * num_kv_heads * head_dim);
            v_buffer.resize(total_seq * num_kv_heads * head_dim);

            // [1, num_kv_heads, total_seq, head_dim] -> [total_seq, num_kv_heads, head_dim]
            for (int h = 0; h < num_kv_heads; h++) {
                for (int s = 0; s < total_seq; s++) {
                    for (int d = 0; d < head_dim; d++) {
                        int src_idx = h * kv_cache->max_seq_len() * head_dim + s * head_dim + d;
                        int dst_idx = s * num_kv_heads * head_dim + h * head_dim + d;
                        k_buffer[dst_idx] = fp16_to_fp32(ck[src_idx]);
                        v_buffer[dst_idx] = fp16_to_fp32(cv[src_idx]);
                    }
                }
            }

            backend_->unmap_buffer(cached_k.buffer());
            backend_->unmap_buffer(cached_v.buffer());

            k_all = k_buffer.data();
            v_all = v_buffer.data();
        }
    } else {
        map_k = backend_->map_buffer(k.buffer());
        map_v = backend_->map_buffer(v.buffer());
        k_all = static_cast<const float*>(map_k.value());
        v_all = static_cast<const float*>(map_v.value());
        total_seq = seq_len;
    }

    // Compute attention for each head
    int heads_per_kv = num_heads / num_kv_heads;  // For GQA

    // Temporary buffer for context vectors
    std::vector<float> context(batch * seq_len * num_heads * head_dim);

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / heads_per_kv;  // GQA: which KV head to use

            for (int q_pos = 0; q_pos < seq_len; q_pos++) {
                // Current query position in the full sequence
                int abs_pos = start_pos + q_pos;

                // Compute attention scores
                std::vector<float> scores(total_seq);
                const float* q_vec = q_data + (b * seq_len + q_pos) * num_heads * head_dim + h * head_dim;

                for (int k_pos = 0; k_pos < total_seq; k_pos++) {
                    // Causal mask: can only attend to positions <= current
                    if (k_pos > abs_pos) {
                        scores[k_pos] = -std::numeric_limits<float>::infinity();
                    } else {
                        const float* k_vec = k_all + k_pos * num_kv_heads * head_dim + kv_h * head_dim;
                        float dot = 0;
                        for (int d = 0; d < head_dim; d++) {
                            dot += q_vec[d] * k_vec[d];
                        }
                        scores[k_pos] = dot * scale;
                    }
                }

                // Softmax
                float max_score = *std::max_element(scores.begin(), scores.end());
                float sum = 0;
                for (int i = 0; i < total_seq; i++) {
                    scores[i] = std::exp(scores[i] - max_score);
                    sum += scores[i];
                }
                for (int i = 0; i < total_seq; i++) {
                    scores[i] /= sum;
                }

                // Weighted sum of values
                float* ctx = context.data() + (b * seq_len + q_pos) * num_heads * head_dim + h * head_dim;
                std::fill(ctx, ctx + head_dim, 0.0f);

                for (int v_pos = 0; v_pos < total_seq; v_pos++) {
                    const float* v_vec = v_all + v_pos * num_kv_heads * head_dim + kv_h * head_dim;
                    float w = scores[v_pos];
                    for (int d = 0; d < head_dim; d++) {
                        ctx[d] += w * v_vec[d];
                    }
                }
            }
        }
    }

    // Unmap K, V if not using cache
    if (!kv_cache || kv_cache->seq_len() == 0) {
        backend_->unmap_buffer(k.buffer());
        backend_->unmap_buffer(v.buffer());
    }

    backend_->unmap_buffer(q.buffer());

    // Output projection: context @ wo
    // context: [batch, seq, num_heads * head_dim]
    // wo: [hidden_dim, num_heads * head_dim]
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            const float* ctx_row = context.data() + (b * seq_len + s) * num_heads * head_dim;
            float* out_row = out_data + (b * seq_len + s) * hidden_dim;

            for (int d = 0; d < hidden_dim; d++) {
                float sum = 0;
                for (int i = 0; i < num_heads * head_dim; i++) {
                    sum += ctx_row[i] * fp16_to_fp32(wo_data[d * num_heads * head_dim + i]);
                }
                out_row[d] = sum;
            }
        }
    }

    backend_->unmap_buffer(output.buffer());
    backend_->unmap_buffer(wo->buffer());

    return output;
}

Result<Tensor> TransformerModel::feed_forward(const Tensor& hidden, int layer) {
    std::string prefix = "blk." + std::to_string(layer) + ".";

    // SwiGLU FFN: output = down(silu(gate(x)) * up(x))
    const Tensor* w_gate = get_weight(prefix + "ffn_gate.weight");
    const Tensor* w_up = get_weight(prefix + "ffn_up.weight");
    const Tensor* w_down = get_weight(prefix + "ffn_down.weight");

    if (!w_gate || !w_up || !w_down) {
        GRANITE_FAIL(ErrorCode::InvalidState,
                     "Missing FFN weights for layer " + std::to_string(layer));
    }

    int batch = static_cast<int>(hidden.size(0));
    int seq_len = static_cast<int>(hidden.size(1));
    int hidden_dim = config_.hidden_dim;
    int intermediate_dim = config_.intermediate_dim;

    // Allocate output
    std::vector<int64_t> output_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(hidden_dim)
    };
    auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        return output_result.error();
    }
    auto output = std::move(output_result).take();

    // Map buffers
    auto map_h = backend_->map_buffer(hidden.buffer());
    auto map_o = backend_->map_buffer(output.buffer());
    auto map_wg = backend_->map_buffer(w_gate->buffer());
    auto map_wu = backend_->map_buffer(w_up->buffer());
    auto map_wd = backend_->map_buffer(w_down->buffer());

    if (!map_h.ok() || !map_o.ok() || !map_wg.ok() || !map_wu.ok() || !map_wd.ok()) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to map FFN buffers");
    }

    const float* h_data = static_cast<const float*>(map_h.value());
    float* o_data = static_cast<float*>(map_o.value());
    const uint16_t* wg_data = static_cast<const uint16_t*>(map_wg.value());
    const uint16_t* wu_data = static_cast<const uint16_t*>(map_wu.value());
    const uint16_t* wd_data = static_cast<const uint16_t*>(map_wd.value());

    // Temporary buffers for intermediate values
    std::vector<float> gate(intermediate_dim);
    std::vector<float> up(intermediate_dim);

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            const float* h_row = h_data + (b * seq_len + s) * hidden_dim;
            float* o_row = o_data + (b * seq_len + s) * hidden_dim;

            // gate = h @ w_gate.T
            // up = h @ w_up.T
            for (int i = 0; i < intermediate_dim; i++) {
                float gate_sum = 0;
                float up_sum = 0;
                for (int d = 0; d < hidden_dim; d++) {
                    gate_sum += h_row[d] * fp16_to_fp32(wg_data[i * hidden_dim + d]);
                    up_sum += h_row[d] * fp16_to_fp32(wu_data[i * hidden_dim + d]);
                }
                // SiLU activation on gate: x * sigmoid(x)
                float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_sum));
                gate[i] = gate_sum * sigmoid_gate;
                up[i] = up_sum;
            }

            // output = (gate * up) @ w_down.T
            for (int d = 0; d < hidden_dim; d++) {
                float sum = 0;
                for (int i = 0; i < intermediate_dim; i++) {
                    sum += (gate[i] * up[i]) * fp16_to_fp32(wd_data[d * intermediate_dim + i]);
                }
                o_row[d] = sum;
            }
        }
    }

    backend_->unmap_buffer(hidden.buffer());
    backend_->unmap_buffer(output.buffer());
    backend_->unmap_buffer(w_gate->buffer());
    backend_->unmap_buffer(w_up->buffer());
    backend_->unmap_buffer(w_down->buffer());

    return output;
}

// =============================================================================
// LLMRunner Implementation
// =============================================================================

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
    // Need to reopen GGUF for tokenizer (model took ownership)
    auto gguf_result = GGUFFile::open(path);
    if (!gguf_result.ok()) {
        return gguf_result.error();
    }

    auto tok_result = Tokenizer::from_gguf(gguf_result.value());
    if (!tok_result.ok()) {
        GRANITE_LOG_WARN("Failed to load tokenizer: {}", tok_result.error().message());
        // Continue without tokenizer - user can set one later
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
    GRANITE_LOG_DEBUG("Prompt tokens: {}", prompt_tokens.size());

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
            break;  // User cancelled via callback
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
    // Get logits for last position
    auto map_l = backend_->map_buffer(logits.buffer());
    if (!map_l.ok()) {
        return 0;
    }

    const auto* l = static_cast<const float*>(map_l.value());
    int seq_len = static_cast<int>(logits.size(1));
    int vocab_size = static_cast<int>(logits.size(2));

    // Get last position
    const float* last_logits = l + (seq_len - 1) * vocab_size;

    // Copy to vector for manipulation
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

        // Renormalize
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

        // Zero out tokens beyond cutoff
        std::unordered_set<int> keep;
        for (size_t i = 0; i < cutoff; i++) {
            keep.insert(indexed[i].second);
        }
        for (int i = 0; i < vocab_size; i++) {
            if (keep.find(i) == keep.end()) {
                probs[i] = 0;
            }
        }

        // Renormalize
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
