#pragma once

#include "granite/types.h"
#include "granite/error.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <optional>

namespace granite {

// Forward declarations
class IComputeBackend;

// =============================================================================
// Memory Statistics
// =============================================================================

struct MemoryStats {
    size_t total_allocated = 0;     // Total bytes currently allocated
    size_t peak_allocated = 0;      // Peak bytes ever allocated
    size_t total_pooled = 0;        // Bytes available in pool
    size_t pool_hits = 0;           // Successful reuses from pool
    size_t pool_misses = 0;         // Allocations requiring new buffers
    size_t aliased_savings = 0;     // Bytes saved through aliasing

    [[nodiscard]] double hit_rate() const {
        size_t total = pool_hits + pool_misses;
        return total > 0 ? static_cast<double>(pool_hits) / total : 0.0;
    }
};

// =============================================================================
// Buffer Allocation Request
// =============================================================================

struct BufferRequest {
    size_t size = 0;
    MemoryType memory_type = MemoryType::Device;
    size_t alignment = 16;  // Default 16-byte alignment for SIMD

    // For liveness tracking (used in graph execution)
    uint32_t first_use = 0;     // First step using this buffer
    uint32_t last_use = 0;      // Last step using this buffer

    // Allow aliasing with other buffers whose lifetimes don't overlap
    bool allow_aliasing = false;
};

// =============================================================================
// Pooled Buffer
// =============================================================================

struct PooledBuffer {
    BufferHandle handle;
    size_t size = 0;
    MemoryType memory_type = MemoryType::Device;
    bool in_use = false;

    // For aliasing: which other buffers share this physical allocation
    std::vector<uint32_t> aliases;
};

// =============================================================================
// Memory Manager
// =============================================================================

class MemoryManager {
public:
    explicit MemoryManager(IComputeBackend* backend);
    ~MemoryManager();

    // Disable copy
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    // -------------------------------------------------------------------------
    // Buffer Allocation
    // -------------------------------------------------------------------------

    /// Allocate a buffer, reusing from pool if available
    [[nodiscard]] Result<BufferHandle> allocate(const BufferRequest& request);

    /// Allocate a buffer with simple parameters
    [[nodiscard]] Result<BufferHandle> allocate(size_t size,
                                                 MemoryType type = MemoryType::Device);

    /// Release a buffer back to the pool for reuse
    void release(BufferHandle handle);

    /// Immediately destroy a buffer (not returned to pool)
    void destroy(BufferHandle handle);

    // -------------------------------------------------------------------------
    // Batch Operations (for graph execution)
    // -------------------------------------------------------------------------

    /// Plan memory allocation for a set of buffers with liveness info
    /// Returns optimal allocation respecting memory budget
    [[nodiscard]] Result<std::vector<BufferHandle>> plan_allocations(
        const std::vector<BufferRequest>& requests,
        std::optional<size_t> memory_budget = std::nullopt);

    /// Release all buffers allocated in a plan
    void release_plan(const std::vector<BufferHandle>& handles);

    // -------------------------------------------------------------------------
    // Pool Management
    // -------------------------------------------------------------------------

    /// Clear all unused buffers from the pool
    void clear_pool();

    /// Set maximum pool size (0 = unlimited)
    void set_max_pool_size(size_t max_bytes);

    /// Get maximum pool size
    [[nodiscard]] size_t max_pool_size() const { return max_pool_size_; }

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------

    /// Get memory statistics
    [[nodiscard]] const MemoryStats& stats() const { return stats_; }

    /// Reset statistics counters
    void reset_stats();

    /// Get current memory usage
    [[nodiscard]] size_t current_usage() const { return stats_.total_allocated; }

    /// Get peak memory usage
    [[nodiscard]] size_t peak_usage() const { return stats_.peak_allocated; }

private:
    IComputeBackend* backend_;
    MemoryStats stats_;
    size_t max_pool_size_ = 0;  // 0 = unlimited

    // Buffer pools organized by size bucket
    // Bucket key = size rounded up to power of 2
    std::unordered_map<size_t, std::vector<PooledBuffer>> device_pool_;
    std::unordered_map<size_t, std::vector<PooledBuffer>> shared_pool_;
    std::unordered_map<size_t, std::vector<PooledBuffer>> managed_pool_;

    // Track all active allocations for cleanup
    std::unordered_map<BufferHandle, PooledBuffer> active_buffers_;

    // Thread safety
    mutable std::mutex mutex_;

    // Internal helpers
    static size_t round_to_bucket(size_t size);
    std::unordered_map<size_t, std::vector<PooledBuffer>>& get_pool(MemoryType type);
    [[nodiscard]] std::optional<BufferHandle> try_get_from_pool(size_t size,
                                                                  MemoryType type);
    void return_to_pool(PooledBuffer buffer);
    void trim_pool_if_needed();
};

// =============================================================================
// Liveness Analysis (for memory planning)
// =============================================================================

struct LivenessInterval {
    uint32_t buffer_id;
    uint32_t first_use;
    uint32_t last_use;
    size_t size;
};

/// Compute which buffers can share memory based on non-overlapping lifetimes
/// Returns groups of buffer IDs that can alias the same physical memory
std::vector<std::vector<uint32_t>> compute_aliasing_groups(
    const std::vector<LivenessInterval>& intervals);

/// Compute minimum memory required for a set of buffers with aliasing
size_t compute_minimum_memory(const std::vector<LivenessInterval>& intervals);

}  // namespace granite
