// KVCachePool - Pool of KV caches for continuous batching
//
// Manages multiple KV caches to serve concurrent generation requests.
// Each request acquires a slot from the pool and uses its dedicated cache.

#include "llm_internal.h"

namespace granite {

Result<void> KVCachePool::allocate(
    int num_slots,
    const ModelConfig& config,
    int max_seq_len,
    IComputeBackend* backend)
{
    backend_ = backend;
    caches_.clear();
    slot_in_use_.clear();

    caches_.reserve(num_slots);
    slot_in_use_.resize(num_slots, false);

    // Allocate KV cache for each slot
    for (int i = 0; i < num_slots; i++) {
        auto cache_result = KVCache::allocate(config, max_seq_len, backend);
        if (!cache_result.ok()) {
            GRANITE_FAIL(ErrorCode::AllocationFailed,
                         "Failed to allocate KV cache slot " + std::to_string(i));
        }
        caches_.push_back(std::move(cache_result).take());
    }

    GRANITE_LOG_INFO("KVCachePool: allocated {} slots, max_seq_len={}",
                     num_slots, max_seq_len);

    return {};
}

int KVCachePool::acquire_slot() {
    for (size_t i = 0; i < slot_in_use_.size(); i++) {
        if (!slot_in_use_[i]) {
            slot_in_use_[i] = true;
            caches_[i].clear();  // Reset cache for new sequence
            return static_cast<int>(i);
        }
    }
    return -1;  // No free slots
}

void KVCachePool::release_slot(int slot) {
    if (slot >= 0 && slot < static_cast<int>(slot_in_use_.size())) {
        slot_in_use_[slot] = false;
    }
}

int KVCachePool::num_free_slots() const {
    int count = 0;
    for (bool in_use : slot_in_use_) {
        if (!in_use) count++;
    }
    return count;
}

}  // namespace granite
