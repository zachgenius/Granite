#pragma once

#include "granite/types.h"
#include "granite/error.h"
#include "granite/tensor.h"
#include "granite/backend.h"
#include "granite/config.h"
#include "granite/gguf.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <unordered_map>
#include <chrono>
#include <deque>

namespace granite {

// =============================================================================
// Raw Quantized Weight (for GPU acceleration)
// =============================================================================

struct RawWeight {
    BufferHandle buffer;          // Raw quantized data
    GGMLType quant_type;          // Quantization type (Q4_K, Q8_0, etc.)
    std::vector<int64_t> shape;   // Original shape
    size_t size_bytes = 0;        // Size in bytes
};

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
// GPU KV Cache (Metal-native)
// =============================================================================

#ifdef GRANITE_HAS_METAL
// Forward declaration - actual type is in Metal framework
struct GPUKVCache {
    struct LayerCache {
        void* k_cache = nullptr;  // MTL::Buffer* [num_kv_heads, max_seq_len, head_dim]
        void* v_cache = nullptr;  // MTL::Buffer*
    };

    std::vector<LayerCache> layers;
    int max_seq_len = 0;
    int current_len = 0;
    int num_kv_heads = 0;
    int head_dim = 0;

    bool is_allocated() const { return !layers.empty(); }
    int seq_len() const { return current_len; }
    void clear() { current_len = 0; }
    void increment_len(int delta = 1) { current_len += delta; }
    void truncate(int new_len) {
        if (new_len >= 0 && new_len < current_len) {
            current_len = new_len;
        }
    }
};
#endif

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

    /// Increment sequence length (for GPU path that writes directly to cache)
    void increment_seq_len(int delta = 1) { current_len_ += delta; }
    void set_seq_len(int len) { current_len_ = len; }

    /// Truncate cache to a specific length (for speculative decoding rollback)
    void truncate(int new_len);

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
// Speculative Decoding Config
// =============================================================================

struct SpeculativeConfig {
    int initial_k = 4;              // Starting speculation depth
    int min_k = 1;                  // Minimum tokens to speculate
    int max_k = 8;                  // Maximum tokens to speculate
    float target_acceptance = 0.8f; // Target acceptance rate for adaptive K
    bool enabled = true;            // Enable/disable speculative decoding

    // Tree speculation settings
    bool use_tree = false;          // Enable tree-based speculation
    int tree_width = 2;             // Branching factor (top-k at each level)
    int tree_depth = 4;             // Maximum tree depth
};

// =============================================================================
// Speculation Tree (for tree-based speculative decoding)
// =============================================================================

class SpeculationTree {
public:
    struct Node {
        int32_t token;
        int parent_idx;             // -1 for root
        int depth;                  // Distance from root
        float log_prob;             // Log probability from draft model
        std::vector<int> children;
    };

    SpeculationTree() = default;

    // Build tree from draft model starting from last token
    void build(int width, int depth);

    // Add a node to the tree
    int add_node(int32_t token, int parent_idx, float log_prob = 0.0f);

    // Get all leaf node indices
    std::vector<int> get_leaves() const;

    // Get path from root to node (inclusive)
    std::vector<int> get_path(int node_idx) const;

    // Get parent indices for all nodes (for attention mask)
    std::vector<int> get_parent_indices() const;

    // Flatten tree tokens in BFS order for batch processing
    std::vector<int32_t> flatten_tokens() const;

    // Get number of nodes
    size_t size() const { return nodes_.size(); }

    // Access nodes
    const Node& node(int idx) const { return nodes_[idx]; }
    Node& node(int idx) { return nodes_[idx]; }

    // Get node pointer (for speculative_runner compatibility)
    const Node* get_node(int idx) const {
        if (idx < 0 || idx >= static_cast<int>(nodes_.size())) return nullptr;
        return &nodes_[idx];
    }

    // Get maximum depth in tree
    int max_depth() const {
        int max_d = 0;
        for (const auto& n : nodes_) {
            max_d = std::max(max_d, n.depth);
        }
        return max_d;
    }

    // Clear tree
    void clear() { nodes_.clear(); }

    // Find longest accepted path given target verification results
    // Returns token indices along the accepted path
    std::vector<int32_t> find_accepted_path(
        const std::vector<int32_t>& target_choices  // argmax at each position
    ) const;

private:
    std::vector<Node> nodes_;
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

    /// Load model from GGUF file with runtime configuration
    /// Config controls attention backend selection, memory limits, etc.
    [[nodiscard]] static Result<TransformerModel> load(
        const std::string& path,
        IComputeBackend* backend,
        const Config& config);

    /// Get model config
    [[nodiscard]] const ModelConfig& config() const { return model_config_; }

    /// Get runtime config
    [[nodiscard]] const Config& runtime_config() const { return runtime_config_; }

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

    /// Forward pass for batch of tokens (speculative decoding verification)
    /// Returns logits for ALL positions [1, num_tokens, vocab_size]
    [[nodiscard]] Result<Tensor> forward_batch(
        const std::vector<int32_t>& tokens,
        KVCache* kv_cache,
        int start_pos);

    /// Forward pass for tree-structured tokens (tree speculative decoding)
    /// Uses tree attention mask where each node attends only to ancestors
    /// Returns logits for ALL nodes [1, num_nodes, vocab_size]
    [[nodiscard]] Result<Tensor> forward_tree(
        const std::vector<int32_t>& tokens,
        const std::vector<int>& parent_indices,  // -1 for root
        KVCache* kv_cache,
        int start_pos);

    /// Get embedding for token IDs
    [[nodiscard]] Result<Tensor> embed(const Tensor& token_ids);

    /// Get weight tensor by name
    [[nodiscard]] const Tensor* get_weight(const std::string& name) const;

    /// Check if model is loaded
    [[nodiscard]] bool is_loaded() const { return !weights_.empty(); }

    /// Enable/disable GPU acceleration
    void set_use_gpu(bool use_gpu) { use_gpu_ = use_gpu; }
    [[nodiscard]] bool use_gpu() const { return use_gpu_; }

    /// Get raw weight by name (for GPU path)
    [[nodiscard]] const RawWeight* get_raw_weight(const std::string& name) const;

    /// Get the compute backend used by this model
    [[nodiscard]] IComputeBackend* backend() const { return backend_; }

#ifdef GRANITE_HAS_METAL
    /// Allocate GPU KV cache
    Result<void> allocate_gpu_kv_cache(int max_seq_len);

    /// Sync CPU KV cache to GPU KV cache (for transitioning from prefill to decode)
    Result<void> sync_cpu_to_gpu_kv_cache(KVCache* kv_cache);

    /// Get GPU KV cache
    GPUKVCache* gpu_kv_cache() { return gpu_kv_cache_.get(); }
    const GPUKVCache* gpu_kv_cache() const { return gpu_kv_cache_.get(); }
#endif

private:
    ModelConfig model_config_;
    Config runtime_config_;  // Runtime configuration (attention backend, memory, etc.)
    std::unordered_map<std::string, Tensor> weights_;
    std::unordered_map<std::string, RawWeight> raw_weights_;  // Raw quantized for GPU
    RoPECache rope_cache_;
    IComputeBackend* backend_ = nullptr;
    std::unique_ptr<GGUFFile> gguf_;
    bool use_gpu_ = false;

#ifdef GRANITE_HAS_METAL
    std::unique_ptr<GPUKVCache> gpu_kv_cache_;

    // Decode buffer pool - preallocated buffers for single-token decode
    struct DecodeBufferPool {
        bool initialized = false;
        Tensor attn_input;      // [1, 1, hidden_dim]
        Tensor post_attn;       // [1, 1, hidden_dim]
        Tensor ffn_input;       // [1, 1, hidden_dim]
        Tensor block_output;    // [1, 1, hidden_dim]
        Tensor attn_layer_out;  // [1, 1, hidden_dim] - for attention layer output
        Tensor norm_out;        // [1, 1, hidden_dim]
        Tensor logits;          // [1, 1, vocab_size]
        // Attention-specific GPU buffers (raw Metal buffers)
        void* q_buf = nullptr;       // [num_heads * head_dim]
        void* k_buf = nullptr;       // [num_kv_heads * head_dim]
        void* v_buf = nullptr;       // [num_kv_heads * head_dim]
        void* attn_out_buf = nullptr; // [num_heads * head_dim]
        // FFN-specific GPU buffers
        void* ffn_gate_buf = nullptr; // [intermediate_dim]
        void* ffn_up_buf = nullptr;   // [intermediate_dim]
    };
    std::unique_ptr<DecodeBufferPool> decode_pool_;

    // Initialize decode buffer pool
    Result<void> init_decode_pool();
#endif

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

    // GPU-accelerated FFN (uses raw Q4_K weights)
    Result<Tensor> feed_forward_gpu(const Tensor& hidden, int layer);

    // GPU-accelerated attention
    Result<Tensor> attention_gpu(
        const Tensor& hidden,
        int layer,
        KVCache* kv_cache,
        int start_pos);

    // Full GPU attention (no CPU sync until output)
    Result<Tensor> attention_full_gpu(
        const Tensor& hidden,
        int layer,
        int start_pos);

    // Tree attention for speculative decoding
    // tree_mask[i][j] = true if node i can attend to node j
    Result<Tensor> transformer_block_tree(
        const Tensor& hidden,
        int layer,
        KVCache* kv_cache,
        int start_pos,
        const std::vector<std::vector<bool>>& tree_mask);

    Result<Tensor> attention_tree(
        const Tensor& hidden,
        int layer,
        KVCache* kv_cache,
        int start_pos,
        const std::vector<std::vector<bool>>& tree_mask);

    // GPU-accelerated tree attention using parent_indices directly
    // Returns empty optional if GPU not available, falls back to CPU
    Result<Tensor> attention_tree_gpu(
        const Tensor& hidden,
        int layer,
        KVCache* kv_cache,
        int start_pos,
        const std::vector<int>& parent_indices);

    // GPU-accelerated transformer block for tree speculation
    Result<Tensor> transformer_block_tree_gpu(
        const Tensor& hidden,
        int layer,
        KVCache* kv_cache,
        int start_pos,
        const std::vector<int>& parent_indices);

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
// Speculative Runner (Draft Model + Target Model)
// =============================================================================

class SpeculativeRunner {
public:
    SpeculativeRunner() = default;
    ~SpeculativeRunner() = default;

    /// Load target and draft models from GGUF files
    [[nodiscard]] static Result<std::unique_ptr<SpeculativeRunner>> load(
        const std::string& target_path,
        const std::string& draft_path);

    /// Generate text (blocking)
    [[nodiscard]] Result<std::string> generate(
        const std::string& prompt,
        const GenerationConfig& config = {},
        const SpeculativeConfig& spec_config = {});

    /// Generate text with streaming callback
    /// Callback returns false to stop generation
    [[nodiscard]] Result<void> generate_streaming(
        const std::string& prompt,
        const GenerationConfig& config,
        TokenCallback callback,
        const SpeculativeConfig& spec_config = {});

    /// Cancel ongoing generation
    void cancel();

    /// Get target model config
    [[nodiscard]] const ModelConfig& config() const { return target_model_.config(); }

    /// Get tokenizer
    [[nodiscard]] const Tokenizer& tokenizer() const { return tokenizer_; }

    /// Reset KV caches (for new conversation)
    void reset();

    /// Get speculative decoding stats
    struct Stats {
        int total_draft_tokens = 0;      // Total tokens drafted
        int total_accepted_tokens = 0;   // Tokens accepted by target
        int total_target_forwards = 0;   // Number of target model batch forwards
        float acceptance_rate() const {
            return total_draft_tokens > 0
                ? static_cast<float>(total_accepted_tokens) / total_draft_tokens
                : 0.0f;
        }
    };
    [[nodiscard]] const Stats& stats() const { return stats_; }

private:
    std::unique_ptr<IComputeBackend> target_backend_;
    std::unique_ptr<IComputeBackend> draft_backend_;
    TransformerModel target_model_;
    TransformerModel draft_model_;
    Tokenizer tokenizer_;  // Shared tokenizer (same vocab assumed)
    KVCache target_kv_cache_;
    KVCache draft_kv_cache_;

    std::atomic<bool> cancelled_{false};
    Stats stats_;

    // Core speculative decoding methods
    std::vector<int32_t> draft_tokens(int k, int32_t last_token);
    std::vector<int32_t> verify_tokens(
        const std::vector<int32_t>& candidates,
        int32_t last_accepted_token);
    void sync_kv_caches(int accepted_count, int drafted_count);
    int adapt_k(int current_k, float acceptance_rate, const SpeculativeConfig& config);

    // Tree-based speculative decoding methods
    void build_tree(SpeculationTree& tree, int32_t root_token, int width, int depth);
    std::vector<int32_t> verify_tree(SpeculationTree& tree, int32_t last_accepted_token);
    std::vector<std::pair<int32_t, float>> get_top_k(const Tensor& logits, int k);
    Result<void> generate_streaming_tree(
        const std::string& prompt,
        const GenerationConfig& config,
        TokenCallback callback,
        const SpeculativeConfig& spec_config);

    // Sampling
    int32_t sample(const Tensor& logits, const GenerationConfig& config);
    int32_t argmax(const Tensor& logits);
};

// =============================================================================
// Continuous Batching - KV Cache Pool
// =============================================================================

class KVCachePool {
public:
    KVCachePool() = default;
    ~KVCachePool() = default;

    /// Allocate pool with N cache slots
    [[nodiscard]] Result<void> allocate(
        int num_slots,
        const ModelConfig& config,
        int max_seq_len,
        IComputeBackend* backend);

    /// Acquire a free cache slot (-1 if none available)
    int acquire_slot();

    /// Release a cache slot back to the pool
    void release_slot(int slot);

    /// Get KV cache for a specific slot
    KVCache& get_cache(int slot) { return caches_[slot]; }
    const KVCache& get_cache(int slot) const { return caches_[slot]; }

    /// Check if slot is in use
    bool is_slot_in_use(int slot) const { return slot_in_use_[slot]; }

    /// Get number of slots
    int num_slots() const { return static_cast<int>(caches_.size()); }

    /// Get number of free slots
    int num_free_slots() const;

private:
    std::vector<KVCache> caches_;
    std::vector<bool> slot_in_use_;
    IComputeBackend* backend_ = nullptr;
};

// =============================================================================
// PagedAttention - Memory-Efficient KV Cache with Block-Based Paging
// =============================================================================

// Block configuration for PagedAttention
struct PagedKVConfig {
    int block_size = 16;         // Tokens per block (power of 2 recommended)
    int num_blocks = 256;        // Total physical blocks in the pool
    int num_layers = 0;          // Number of transformer layers
    int num_kv_heads = 0;        // Number of KV heads
    int head_dim = 0;            // Head dimension
};

// BlockManager: Manages physical KV cache blocks
// Uses a simple free-list allocator for O(1) allocation/deallocation
class BlockManager {
public:
    BlockManager() = default;

    /// Initialize block manager with given config
    [[nodiscard]] Result<void> initialize(
        const PagedKVConfig& config,
        IComputeBackend* backend);

    /// Allocate a free block, returns block index (-1 if none available)
    int allocate_block();

    /// Free a block back to the pool
    void free_block(int block_idx);

    /// Get number of free blocks
    [[nodiscard]] int num_free_blocks() const { return num_free_; }

    /// Get total number of blocks
    [[nodiscard]] int num_total_blocks() const { return num_blocks_; }

    /// Get block size (tokens per block)
    [[nodiscard]] int block_size() const { return block_size_; }

    /// Get K cache buffer for a layer (all blocks contiguous)
    /// Shape: [num_blocks, block_size, num_kv_heads, head_dim]
    [[nodiscard]] Tensor& k_cache(int layer) { return k_blocks_[layer]; }
    [[nodiscard]] const Tensor& k_cache(int layer) const { return k_blocks_[layer]; }

    /// Get V cache buffer for a layer
    [[nodiscard]] Tensor& v_cache(int layer) { return v_blocks_[layer]; }
    [[nodiscard]] const Tensor& v_cache(int layer) const { return v_blocks_[layer]; }

    /// Check if initialized
    [[nodiscard]] bool is_initialized() const { return !k_blocks_.empty(); }

    /// Get config
    [[nodiscard]] const PagedKVConfig& config() const { return config_; }

private:
    PagedKVConfig config_;
    int num_blocks_ = 0;
    int block_size_ = 0;
    int num_free_ = 0;
    std::vector<int> free_list_;          // Stack of free block indices
    std::vector<bool> block_allocated_;   // Track which blocks are allocated
    std::vector<Tensor> k_blocks_;        // [num_layers] of [num_blocks, block_size, num_kv_heads, head_dim]
    std::vector<Tensor> v_blocks_;        // [num_layers]
    IComputeBackend* backend_ = nullptr;
};

// PagedKVCache: Per-sequence view into the shared block pool
// Maintains a block table mapping logical positions to physical blocks
class PagedKVCache {
public:
    PagedKVCache() = default;

    /// Create a new paged cache using the given block manager
    explicit PagedKVCache(BlockManager* block_manager);

    /// Release all blocks back to manager
    void release();

    /// Append tokens to the cache (allocates new blocks as needed)
    /// Returns false if allocation failed (out of blocks)
    [[nodiscard]] bool append_tokens(int num_tokens);

    /// Get current sequence length
    [[nodiscard]] int seq_len() const { return seq_len_; }

    /// Get number of allocated blocks
    [[nodiscard]] int num_blocks() const { return static_cast<int>(block_table_.size()); }

    /// Get the block table (maps logical block -> physical block)
    [[nodiscard]] const std::vector<int>& block_table() const { return block_table_; }

    /// Get physical block index for a given token position
    [[nodiscard]] int get_physical_block(int token_pos) const;

    /// Get position within block for a given token position
    [[nodiscard]] int get_block_offset(int token_pos) const;

    /// Clear the cache (release all blocks, reset seq_len)
    void clear();

    /// Truncate to a specific length (for speculative decoding rollback)
    void truncate(int new_len);

    /// Get block manager
    [[nodiscard]] BlockManager* block_manager() { return block_manager_; }

private:
    BlockManager* block_manager_ = nullptr;
    std::vector<int> block_table_;  // Logical block index -> Physical block index
    int seq_len_ = 0;
};

// PagedKVPool: Pool of PagedKVCache instances for multiple concurrent requests
class PagedKVPool {
public:
    PagedKVPool() = default;
    ~PagedKVPool() = default;

    /// Initialize pool with shared block manager
    [[nodiscard]] Result<void> initialize(
        const ModelConfig& config,
        int max_sequences,          // Maximum concurrent sequences
        int max_total_tokens,       // Total tokens across all sequences
        int block_size,             // Tokens per block
        IComputeBackend* backend);

    /// Acquire a cache slot for a new sequence
    [[nodiscard]] int acquire_slot();

    /// Release a cache slot
    void release_slot(int slot);

    /// Get PagedKVCache for a slot
    [[nodiscard]] PagedKVCache& get_cache(int slot) { return caches_[slot]; }
    [[nodiscard]] const PagedKVCache& get_cache(int slot) const { return caches_[slot]; }

    /// Get the shared block manager
    [[nodiscard]] BlockManager& block_manager() { return block_manager_; }

    /// Get number of available slots
    [[nodiscard]] int num_free_slots() const;

    /// Get number of free blocks in pool
    [[nodiscard]] int num_free_blocks() const { return block_manager_.num_free_blocks(); }

    /// Check if initialized
    [[nodiscard]] bool is_initialized() const { return block_manager_.is_initialized(); }

private:
    BlockManager block_manager_;
    std::vector<PagedKVCache> caches_;
    std::vector<bool> slot_in_use_;
};

// =============================================================================
// Continuous Batching - Generation Request
// =============================================================================

struct GenerationRequest {
    int request_id = -1;
    std::string prompt;
    GenerationConfig config;
    TokenCallback callback;

    // Request state
    enum class State { PENDING, PREFILLING, DECODING, COMPLETED, FAILED };
    State state = State::PENDING;

    // Token state
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    int prefill_pos = 0;            // Progress through prefill

    // KV cache assignment
    int kv_cache_slot = -1;

    // Timing
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point first_token_time;
};

// =============================================================================
// Continuous Batching - Batch Scheduler
// =============================================================================

class BatchScheduler {
public:
    BatchScheduler() = default;
    ~BatchScheduler() = default;

    /// Initialize scheduler with model and cache pool (standard KV cache)
    [[nodiscard]] Result<void> initialize(
        TransformerModel* model,
        Tokenizer* tokenizer,
        int num_cache_slots,
        int max_batch_tokens = 256);

    /// Initialize scheduler with PagedKVPool for true batched decode
    /// This enables memory-efficient KV caching with scattered physical blocks
    /// and uses batched GPU kernels for decode
    [[nodiscard]] Result<void> initialize_paged(
        TransformerModel* model,
        Tokenizer* tokenizer,
        int num_cache_slots,
        int max_total_tokens,      // Total tokens across all sequences
        int block_size = 16,       // Tokens per physical block
        int max_batch_tokens = 256);

    /// Submit a new generation request
    /// Returns request ID
    int submit(const std::string& prompt,
               const GenerationConfig& config,
               TokenCallback callback = nullptr);

    /// Process one batch iteration
    /// Returns number of completed requests
    int step();

    /// Get and clear completed requests
    std::vector<GenerationRequest> take_completed();

    /// Cancel a request
    void cancel(int request_id);

    /// Cancel all requests
    void cancel_all();

    /// Check if any requests are pending/active
    bool has_pending() const;

    /// Get queue sizes
    int prefill_queue_size() const { return static_cast<int>(prefill_queue_.size()); }
    int decode_queue_size() const { return static_cast<int>(decode_queue_.size()); }

    /// Check if using paged attention
    bool is_paged() const { return use_paged_; }

private:
    // Batch assembly
    struct Batch {
        std::vector<int32_t> tokens;
        std::vector<int> positions;      // Position in sequence per token
        std::vector<int> kv_slots;       // KV cache slot per token
        std::vector<int> request_ids;    // Request ID per token
        bool is_prefill = false;
    };

    Batch assemble_prefill_batch();
    Batch assemble_decode_batch();
    void process_batch(const Batch& batch);
    void process_batch_paged(const Batch& batch);  // True batched decode with paged attention

    // Model and resources
    TransformerModel* model_ = nullptr;
    Tokenizer* tokenizer_ = nullptr;
    std::unique_ptr<KVCachePool> kv_pool_;
    std::unique_ptr<PagedKVPool> paged_kv_pool_;

    // Request management
    std::deque<std::unique_ptr<GenerationRequest>> prefill_queue_;
    std::deque<std::unique_ptr<GenerationRequest>> decode_queue_;
    std::vector<GenerationRequest> completed_;
    int next_request_id_ = 0;

    // Configuration
    int max_batch_tokens_ = 256;
    bool use_paged_ = false;
    int block_size_ = 16;
};

// =============================================================================
// Utility Functions
// =============================================================================

/// Parse model config from GGUF metadata
Result<ModelConfig> parse_model_config(const GGUFFile& gguf);

}  // namespace granite
