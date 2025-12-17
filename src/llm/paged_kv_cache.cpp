// PagedKVCache - Memory-efficient KV cache using block-based paging
//
// Key concepts:
// - Physical blocks: Fixed-size chunks of KV memory shared across all sequences
// - Block table: Per-sequence mapping from logical to physical blocks
// - Dynamic allocation: Blocks allocated on-demand as sequences grow
// - Memory sharing: Unused blocks returned to pool for other sequences

#include "llm_internal.h"

namespace granite {

// =============================================================================
// BlockManager Implementation
// =============================================================================

Result<void> BlockManager::initialize(
    const PagedKVConfig& config,
    IComputeBackend* backend)
{
    if (config.num_blocks <= 0 || config.block_size <= 0) {
        GRANITE_FAIL(ErrorCode::InvalidArgument, "Invalid block configuration");
    }

    config_ = config;
    backend_ = backend;
    num_blocks_ = config.num_blocks;
    block_size_ = config.block_size;

    // Initialize free list with all blocks
    free_list_.resize(num_blocks_);
    block_allocated_.resize(num_blocks_, false);
    for (int i = 0; i < num_blocks_; i++) {
        free_list_[i] = i;
    }
    num_free_ = num_blocks_;

    // Allocate K/V block storage for each layer
    // Shape: [num_blocks, block_size, num_kv_heads, head_dim] but stored as FP16
    // We use a flat layout: [num_blocks * block_size, num_kv_heads, head_dim]
    k_blocks_.resize(config.num_layers);
    v_blocks_.resize(config.num_layers);

    int64_t block_tokens = static_cast<int64_t>(num_blocks_) * block_size_;
    std::vector<int64_t> shape = {
        block_tokens,
        static_cast<int64_t>(config.num_kv_heads),
        static_cast<int64_t>(config.head_dim)
    };

    for (int layer = 0; layer < config.num_layers; layer++) {
        auto k_result = Tensor::allocate(shape, DataType::FP16, backend);
        auto v_result = Tensor::allocate(shape, DataType::FP16, backend);

        if (!k_result.ok() || !v_result.ok()) {
            GRANITE_FAIL(ErrorCode::AllocationFailed,
                        "Failed to allocate paged KV blocks for layer " + std::to_string(layer));
        }

        k_blocks_[layer] = std::move(k_result).take();
        v_blocks_[layer] = std::move(v_result).take();
    }

    size_t total_memory = 2 * config.num_layers * block_tokens *
                          config.num_kv_heads * config.head_dim * sizeof(uint16_t);
    GRANITE_LOG_INFO("BlockManager initialized: {} blocks x {} tokens, {} layers, {:.1f} MB",
                     num_blocks_, block_size_, config.num_layers,
                     total_memory / (1024.0 * 1024.0));

    return {};
}

int BlockManager::allocate_block() {
    if (num_free_ == 0) {
        return -1;  // Out of blocks
    }

    // Pop from free list (stack-style for cache locality)
    int block_idx = free_list_[--num_free_];
    block_allocated_[block_idx] = true;
    return block_idx;
}

void BlockManager::free_block(int block_idx) {
    if (block_idx < 0 || block_idx >= num_blocks_) {
        GRANITE_LOG_WARN("Invalid block index {} in free_block", block_idx);
        return;
    }

    if (!block_allocated_[block_idx]) {
        GRANITE_LOG_WARN("Double free of block {}", block_idx);
        return;
    }

    // Push to free list
    block_allocated_[block_idx] = false;
    free_list_[num_free_++] = block_idx;
}

// =============================================================================
// PagedKVCache Implementation
// =============================================================================

PagedKVCache::PagedKVCache(BlockManager* block_manager)
    : block_manager_(block_manager)
{
}

void PagedKVCache::release() {
    if (!block_manager_) return;

    // Free all allocated blocks
    for (int block_idx : block_table_) {
        block_manager_->free_block(block_idx);
    }
    block_table_.clear();
    seq_len_ = 0;
}

bool PagedKVCache::append_tokens(int num_tokens) {
    if (!block_manager_) return false;

    int block_size = block_manager_->block_size();
    int new_seq_len = seq_len_ + num_tokens;

    // Calculate how many blocks we need
    int blocks_needed = (new_seq_len + block_size - 1) / block_size;
    int current_blocks = static_cast<int>(block_table_.size());

    // Allocate new blocks if needed
    while (current_blocks < blocks_needed) {
        int new_block = block_manager_->allocate_block();
        if (new_block < 0) {
            // Out of blocks - allocation failed
            GRANITE_LOG_WARN("PagedKVCache: out of blocks, needed {} more",
                            blocks_needed - current_blocks);
            return false;
        }
        block_table_.push_back(new_block);
        current_blocks++;
    }

    seq_len_ = new_seq_len;
    return true;
}

int PagedKVCache::get_physical_block(int token_pos) const {
    if (!block_manager_) return -1;

    int block_size = block_manager_->block_size();
    int logical_block = token_pos / block_size;

    if (logical_block >= static_cast<int>(block_table_.size())) {
        return -1;
    }

    return block_table_[logical_block];
}

int PagedKVCache::get_block_offset(int token_pos) const {
    if (!block_manager_) return 0;
    return token_pos % block_manager_->block_size();
}

void PagedKVCache::clear() {
    release();
}

void PagedKVCache::truncate(int new_len) {
    if (new_len >= seq_len_) return;
    if (!block_manager_) return;

    int block_size = block_manager_->block_size();

    // Calculate how many blocks we need for new_len
    int blocks_needed = (new_len + block_size - 1) / block_size;
    if (new_len == 0) blocks_needed = 0;

    // Free excess blocks
    while (static_cast<int>(block_table_.size()) > blocks_needed) {
        int block_idx = block_table_.back();
        block_table_.pop_back();
        block_manager_->free_block(block_idx);
    }

    seq_len_ = new_len;
}

// =============================================================================
// PagedKVPool Implementation
// =============================================================================

Result<void> PagedKVPool::initialize(
    const ModelConfig& config,
    int max_sequences,
    int max_total_tokens,
    int block_size,
    IComputeBackend* backend)
{
    // Calculate number of blocks needed
    int num_blocks = (max_total_tokens + block_size - 1) / block_size;

    // Initialize block manager
    PagedKVConfig paged_config;
    paged_config.block_size = block_size;
    paged_config.num_blocks = num_blocks;
    paged_config.num_layers = config.num_layers;
    paged_config.num_kv_heads = config.num_kv_heads;
    paged_config.head_dim = config.head_dim;

    auto init_result = block_manager_.initialize(paged_config, backend);
    if (!init_result.ok()) {
        return init_result.error();
    }

    // Create cache slots
    caches_.resize(max_sequences);
    slot_in_use_.resize(max_sequences, false);

    for (int i = 0; i < max_sequences; i++) {
        caches_[i] = PagedKVCache(&block_manager_);
    }

    GRANITE_LOG_INFO("PagedKVPool: {} slots, {} blocks ({} tokens max), block_size={}",
                     max_sequences, num_blocks, max_total_tokens, block_size);

    return {};
}

int PagedKVPool::acquire_slot() {
    for (size_t i = 0; i < slot_in_use_.size(); i++) {
        if (!slot_in_use_[i]) {
            slot_in_use_[i] = true;
            caches_[i].clear();  // Ensure clean state
            return static_cast<int>(i);
        }
    }
    return -1;  // No slots available
}

void PagedKVPool::release_slot(int slot) {
    if (slot < 0 || slot >= static_cast<int>(slot_in_use_.size())) {
        return;
    }

    if (slot_in_use_[slot]) {
        caches_[slot].release();
        slot_in_use_[slot] = false;
    }
}

int PagedKVPool::num_free_slots() const {
    int count = 0;
    for (bool in_use : slot_in_use_) {
        if (!in_use) count++;
    }
    return count;
}

}  // namespace granite
