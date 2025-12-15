#pragma once

#include "granite/types.h"
#include "granite/error.h"
#include "granite/tensor.h"
#include "granite/backend.h"
#include "granite/gguf.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <unordered_map>

namespace granite {

// =============================================================================
// Model Configuration
// =============================================================================

struct ModelConfig {
    std::string architecture;
    std::string name;

    int vocab_size = 0;
    int hidden_dim = 0;
    int intermediate_dim = 0;
    int num_layers = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int max_seq_len = 2048;

    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;

    bool tie_word_embeddings = false;

    // Compute head_dim if not explicitly set
    void compute_derived() {
        if (head_dim == 0 && hidden_dim > 0 && num_heads > 0) {
            head_dim = hidden_dim / num_heads;
        }
    }
};

// =============================================================================
// KV Cache
// =============================================================================

class KVCache {
public:
    KVCache() = default;

    /// Allocate cache for given config
    [[nodiscard]] static Result<KVCache> allocate(
        const ModelConfig& config,
        int max_seq_len,
        IComputeBackend* backend);

    /// Append new K/V tensors for a layer
    Result<void> append(int layer, const Tensor& keys, const Tensor& values);

    /// Get cached K/V for a layer (up to current length)
    [[nodiscard]] std::pair<Tensor, Tensor> get(int layer) const;

    /// Get current sequence length
    [[nodiscard]] int seq_len() const { return current_len_; }

    /// Get max sequence length
    [[nodiscard]] int max_seq_len() const { return max_seq_len_; }

    /// Clear cache (reset for new sequence)
    void clear();

    /// Memory usage in bytes
    [[nodiscard]] size_t memory_bytes() const;

    /// Check if cache is allocated
    [[nodiscard]] bool is_allocated() const { return !layers_.empty(); }

private:
    struct LayerCache {
        Tensor keys;    // [1, num_kv_heads, max_seq_len, head_dim]
        Tensor values;  // [1, num_kv_heads, max_seq_len, head_dim]
    };

    std::vector<LayerCache> layers_;
    int max_seq_len_ = 0;
    int current_len_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    IComputeBackend* backend_ = nullptr;
};

// =============================================================================
// RoPE (Rotary Position Embeddings)
// =============================================================================

class RoPECache {
public:
    RoPECache() = default;

    /// Initialize cos/sin cache for given config
    void initialize(int max_seq_len, int head_dim, float theta = 10000.0f);

    /// Apply RoPE to Q and K tensors
    /// Q/K shape: [batch, seq_len, num_heads, head_dim]
    Result<void> apply(Tensor& q, Tensor& k, int start_pos, IComputeBackend* backend);

    /// Get cos values for position range
    [[nodiscard]] const std::vector<float>& cos_cache() const { return cos_; }

    /// Get sin values for position range
    [[nodiscard]] const std::vector<float>& sin_cache() const { return sin_; }

    [[nodiscard]] bool is_initialized() const { return !cos_.empty(); }

private:
    std::vector<float> cos_;  // [max_seq_len, head_dim/2]
    std::vector<float> sin_;  // [max_seq_len, head_dim/2]
    int max_seq_len_ = 0;
    int head_dim_ = 0;
};

// =============================================================================
// BPE Tokenizer
// =============================================================================

class Tokenizer {
public:
    Tokenizer() = default;

    /// Load tokenizer from GGUF file
    [[nodiscard]] static Result<Tokenizer> from_gguf(const GGUFFile& gguf);

    /// Encode text to token IDs
    [[nodiscard]] std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const;

    /// Decode token IDs to text
    [[nodiscard]] std::string decode(const std::vector<int32_t>& tokens) const;

    /// Decode single token
    [[nodiscard]] std::string decode_token(int32_t token) const;

    /// Get special token IDs
    [[nodiscard]] int32_t bos_token() const { return bos_token_; }
    [[nodiscard]] int32_t eos_token() const { return eos_token_; }
    [[nodiscard]] int32_t pad_token() const { return pad_token_; }

    /// Get vocabulary size
    [[nodiscard]] size_t vocab_size() const { return vocab_.size(); }

    /// Check if loaded
    [[nodiscard]] bool is_loaded() const { return !vocab_.empty(); }

private:
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>{}(p.first) ^
                   (std::hash<std::string>{}(p.second) << 1);
        }
    };

    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_ranks_;

    int32_t bos_token_ = 1;
    int32_t eos_token_ = 2;
    int32_t pad_token_ = 0;
    int32_t unk_token_ = 0;
};

// =============================================================================
// Generation Config
// =============================================================================

struct GenerationConfig {
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;

    int max_tokens = 256;
    std::vector<int32_t> stop_tokens = {};

    bool do_sample = true;  // false = greedy decoding
};

// =============================================================================
// Transformer Model
// =============================================================================

class TransformerModel {
public:
    TransformerModel() = default;
    ~TransformerModel() = default;

    // Disable copy
    TransformerModel(const TransformerModel&) = delete;
    TransformerModel& operator=(const TransformerModel&) = delete;

    // Allow move
    TransformerModel(TransformerModel&&) = default;
    TransformerModel& operator=(TransformerModel&&) = default;

    /// Load model from GGUF file
    [[nodiscard]] static Result<TransformerModel> load(
        const std::string& path,
        IComputeBackend* backend);

    /// Get model config
    [[nodiscard]] const ModelConfig& config() const { return config_; }

    /// Forward pass for prefill (multiple tokens)
    /// Input: token_ids [batch, seq_len]
    /// Output: logits [batch, seq_len, vocab_size]
    [[nodiscard]] Result<Tensor> forward(
        const Tensor& token_ids,
        KVCache* kv_cache = nullptr,
        int start_pos = 0);

    /// Forward pass for single token (decode mode)
    [[nodiscard]] Result<Tensor> forward_single(
        int32_t token_id,
        KVCache& kv_cache);

    /// Get embedding for token IDs
    [[nodiscard]] Result<Tensor> embed(const Tensor& token_ids);

    /// Get weight tensor by name
    [[nodiscard]] const Tensor* get_weight(const std::string& name) const;

    /// Check if model is loaded
    [[nodiscard]] bool is_loaded() const { return !weights_.empty(); }

private:
    ModelConfig config_;
    std::unordered_map<std::string, Tensor> weights_;
    RoPECache rope_cache_;
    IComputeBackend* backend_ = nullptr;
    std::unique_ptr<GGUFFile> gguf_;

    // Layer forward pass
    Result<Tensor> transformer_block(
        const Tensor& hidden,
        int layer,
        KVCache* kv_cache,
        int start_pos);

    // Attention forward
    Result<Tensor> attention(
        const Tensor& hidden,
        int layer,
        KVCache* kv_cache,
        int start_pos);

    // FFN forward (SwiGLU)
    Result<Tensor> feed_forward(const Tensor& hidden, int layer);

    // Helper functions
    Tensor apply_rms_norm(const Tensor& input, const Tensor* weight);
    Tensor add_tensors(const Tensor& a, const Tensor& b);
};

// =============================================================================
// LLM Runner
// =============================================================================

using TokenCallback = std::function<bool(const std::string& token)>;

class LLMRunner {
public:
    LLMRunner() = default;
    ~LLMRunner() = default;

    /// Load model from GGUF file
    [[nodiscard]] static Result<std::unique_ptr<LLMRunner>> load(
        const std::string& path);

    /// Generate text (blocking)
    [[nodiscard]] Result<std::string> generate(
        const std::string& prompt,
        const GenerationConfig& config = {});

    /// Generate text with streaming callback
    /// Callback returns false to stop generation
    [[nodiscard]] Result<void> generate_streaming(
        const std::string& prompt,
        const GenerationConfig& config,
        TokenCallback callback);

    /// Cancel ongoing generation
    void cancel();

    /// Get model config
    [[nodiscard]] const ModelConfig& config() const { return model_.config(); }

    /// Get tokenizer
    [[nodiscard]] const Tokenizer& tokenizer() const { return tokenizer_; }

    /// Reset KV cache (for new conversation)
    void reset();

private:
    std::unique_ptr<IComputeBackend> backend_;
    TransformerModel model_;
    Tokenizer tokenizer_;
    KVCache kv_cache_;

    std::atomic<bool> cancelled_{false};

    // Sampling
    int32_t sample(const Tensor& logits, const GenerationConfig& config);
    void apply_repetition_penalty(
        std::vector<float>& logits,
        const std::vector<int32_t>& past_tokens,
        float penalty);
};

// =============================================================================
// Utility Functions
// =============================================================================

/// Parse model config from GGUF metadata
Result<ModelConfig> parse_model_config(const GGUFFile& gguf);

}  // namespace granite
