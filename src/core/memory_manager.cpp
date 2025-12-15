#include "granite/memory.h"
#include "granite/backend.h"
#include "granite/log.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace granite {

// =============================================================================
// MemoryManager Implementation
// =============================================================================

MemoryManager::MemoryManager(IComputeBackend* backend)
    : backend_(backend)
{
    GRANITE_LOG_DEBUG("MemoryManager initialized");
}

MemoryManager::~MemoryManager() {
    // Destroy all pooled buffers
    clear_pool();

    // Destroy any remaining active buffers
    for (auto& [handle, buffer] : active_buffers_) {
        if (handle.valid()) {
            backend_->destroy_buffer(handle);
        }
    }
    active_buffers_.clear();

    GRANITE_LOG_DEBUG("MemoryManager destroyed - peak usage: {} bytes", stats_.peak_allocated);
}

// -----------------------------------------------------------------------------
// Buffer Allocation
// -----------------------------------------------------------------------------

Result<BufferHandle> MemoryManager::allocate(const BufferRequest& request) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Try to get from pool first
    if (auto pooled = try_get_from_pool(request.size, request.memory_type)) {
        stats_.pool_hits++;
        return *pooled;
    }

    // Need to allocate a new buffer
    stats_.pool_misses++;

    // Round up to bucket size for better pooling
    size_t alloc_size = round_to_bucket(request.size);

    BufferDesc desc;
    desc.size = alloc_size;
    desc.memory_type = request.memory_type;
    desc.allow_aliasing = request.allow_aliasing;

    auto result = backend_->create_buffer(desc);
    if (!result.ok()) {
        return result.error();
    }

    BufferHandle handle = result.value();

    // Track the allocation
    PooledBuffer pb;
    pb.handle = handle;
    pb.size = alloc_size;
    pb.memory_type = request.memory_type;
    pb.in_use = true;

    active_buffers_[handle] = pb;

    // Update stats
    stats_.total_allocated += alloc_size;
    if (stats_.total_allocated > stats_.peak_allocated) {
        stats_.peak_allocated = stats_.total_allocated;
    }

    GRANITE_LOG_TRACE("Allocated buffer: {} bytes (bucket: {} bytes)",
              request.size, alloc_size);

    return handle;
}

Result<BufferHandle> MemoryManager::allocate(size_t size, MemoryType type) {
    BufferRequest request;
    request.size = size;
    request.memory_type = type;
    return allocate(request);
}

void MemoryManager::release(BufferHandle handle) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = active_buffers_.find(handle);
    if (it == active_buffers_.end()) {
        GRANITE_LOG_WARN("Attempted to release unknown buffer handle");
        return;
    }

    PooledBuffer buffer = it->second;
    active_buffers_.erase(it);

    // Return to pool for reuse
    return_to_pool(buffer);

    GRANITE_LOG_TRACE("Released buffer to pool: {} bytes", buffer.size);
}

void MemoryManager::destroy(BufferHandle handle) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = active_buffers_.find(handle);
    if (it == active_buffers_.end()) {
        GRANITE_LOG_WARN("Attempted to destroy unknown buffer handle");
        return;
    }

    stats_.total_allocated -= it->second.size;
    backend_->destroy_buffer(handle);
    active_buffers_.erase(it);

    GRANITE_LOG_TRACE("Destroyed buffer");
}

// -----------------------------------------------------------------------------
// Batch Operations
// -----------------------------------------------------------------------------

Result<std::vector<BufferHandle>> MemoryManager::plan_allocations(
    const std::vector<BufferRequest>& requests,
    std::optional<size_t> memory_budget)
{
    std::vector<BufferHandle> handles;
    handles.reserve(requests.size());

    // Build liveness intervals for potential aliasing
    std::vector<LivenessInterval> intervals;
    for (size_t i = 0; i < requests.size(); i++) {
        if (requests[i].allow_aliasing) {
            intervals.push_back({
                static_cast<uint32_t>(i),
                requests[i].first_use,
                requests[i].last_use,
                requests[i].size
            });
        }
    }

    // Compute aliasing groups if we have liveness info
    std::vector<std::vector<uint32_t>> aliasing_groups;
    if (!intervals.empty()) {
        aliasing_groups = compute_aliasing_groups(intervals);
    }

    // Create a map from buffer ID to its aliasing group leader
    std::unordered_map<uint32_t, uint32_t> alias_leader;
    std::unordered_map<uint32_t, BufferHandle> group_handles;

    for (const auto& group : aliasing_groups) {
        if (group.size() > 1) {
            uint32_t leader = group[0];
            for (uint32_t id : group) {
                alias_leader[id] = leader;
            }
        }
    }

    // Allocate buffers
    for (size_t i = 0; i < requests.size(); i++) {
        const auto& req = requests[i];

        // Check if this buffer can alias with an already allocated one
        auto alias_it = alias_leader.find(static_cast<uint32_t>(i));
        if (alias_it != alias_leader.end() && alias_it->second != i) {
            // Use the leader's buffer
            auto handle_it = group_handles.find(alias_it->second);
            if (handle_it != group_handles.end()) {
                handles.push_back(handle_it->second);
                stats_.aliased_savings += req.size;
                continue;
            }
        }

        // Allocate new buffer
        auto result = allocate(req);
        if (!result.ok()) {
            // Rollback on failure
            for (auto& h : handles) {
                release(h);
            }
            return result.error();
        }

        BufferHandle handle = result.value();
        handles.push_back(handle);

        // Track for aliasing
        if (alias_it != alias_leader.end()) {
            group_handles[alias_it->second] = handle;
        }

        // Check memory budget
        if (memory_budget && stats_.total_allocated > *memory_budget) {
            GRANITE_LOG_WARN("Memory budget exceeded: {} / {} bytes",
                     stats_.total_allocated, *memory_budget);
        }
    }

    return handles;
}

void MemoryManager::release_plan(const std::vector<BufferHandle>& handles) {
    // Use a set to avoid releasing aliased buffers multiple times
    std::unordered_set<BufferHandle> released;

    for (const auto& handle : handles) {
        if (released.find(handle) == released.end()) {
            release(handle);
            released.insert(handle);
        }
    }
}

// -----------------------------------------------------------------------------
// Pool Management
// -----------------------------------------------------------------------------

void MemoryManager::clear_pool() {
    std::lock_guard<std::mutex> lock(mutex_);

    auto clear_pool_map = [this](auto& pool_map) {
        for (auto& [size, buffers] : pool_map) {
            for (auto& buffer : buffers) {
                if (buffer.handle.valid()) {
                    backend_->destroy_buffer(buffer.handle);
                }
            }
        }
        pool_map.clear();
    };

    clear_pool_map(device_pool_);
    clear_pool_map(shared_pool_);
    clear_pool_map(managed_pool_);

    stats_.total_pooled = 0;

    GRANITE_LOG_DEBUG("Cleared all buffer pools");
}

void MemoryManager::set_max_pool_size(size_t max_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_pool_size_ = max_bytes;
    trim_pool_if_needed();
}

void MemoryManager::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t current = stats_.total_allocated;
    stats_ = MemoryStats{};
    stats_.total_allocated = current;
    stats_.peak_allocated = current;
}

// -----------------------------------------------------------------------------
// Internal Helpers
// -----------------------------------------------------------------------------

size_t MemoryManager::round_to_bucket(size_t size) {
    if (size == 0) return 0;

    // Round up to next power of 2 for small sizes
    if (size <= 4096) {
        size_t bucket = 64;  // Minimum bucket size
        while (bucket < size) {
            bucket *= 2;
        }
        return bucket;
    }

    // For larger sizes, round to nearest 4KB multiple
    constexpr size_t page_size = 4096;
    return ((size + page_size - 1) / page_size) * page_size;
}

std::unordered_map<size_t, std::vector<PooledBuffer>>&
MemoryManager::get_pool(MemoryType type) {
    switch (type) {
        case MemoryType::Device:  return device_pool_;
        case MemoryType::Shared:  return shared_pool_;
        case MemoryType::Managed: return managed_pool_;
    }
    return device_pool_;
}

std::optional<BufferHandle> MemoryManager::try_get_from_pool(size_t size,
                                                              MemoryType type) {
    size_t bucket_size = round_to_bucket(size);
    auto& pool = get_pool(type);

    auto it = pool.find(bucket_size);
    if (it == pool.end() || it->second.empty()) {
        return std::nullopt;
    }

    // Get a buffer from the pool
    PooledBuffer buffer = it->second.back();
    it->second.pop_back();

    // Track as active
    buffer.in_use = true;
    active_buffers_[buffer.handle] = buffer;

    stats_.total_pooled -= buffer.size;

    return buffer.handle;
}

void MemoryManager::return_to_pool(PooledBuffer buffer) {
    buffer.in_use = false;

    // Check if pool is full
    if (max_pool_size_ > 0 && stats_.total_pooled + buffer.size > max_pool_size_) {
        // Destroy instead of pooling
        backend_->destroy_buffer(buffer.handle);
        stats_.total_allocated -= buffer.size;
        return;
    }

    auto& pool = get_pool(buffer.memory_type);
    pool[buffer.size].push_back(buffer);
    stats_.total_pooled += buffer.size;
}

void MemoryManager::trim_pool_if_needed() {
    if (max_pool_size_ == 0) return;

    while (stats_.total_pooled > max_pool_size_) {
        // Find and remove the largest pooled buffer
        bool found = false;
        auto trim_from_pool = [this, &found](auto& pool_map) {
            if (found) return;
            for (auto& [size, buffers] : pool_map) {
                if (!buffers.empty()) {
                    auto& buffer = buffers.back();
                    backend_->destroy_buffer(buffer.handle);
                    stats_.total_pooled -= buffer.size;
                    stats_.total_allocated -= buffer.size;
                    buffers.pop_back();
                    found = true;
                    return;
                }
            }
        };

        trim_from_pool(device_pool_);
        trim_from_pool(shared_pool_);
        trim_from_pool(managed_pool_);

        if (!found) break;
    }
}

// =============================================================================
// Liveness Analysis
// =============================================================================

std::vector<std::vector<uint32_t>> compute_aliasing_groups(
    const std::vector<LivenessInterval>& intervals)
{
    std::vector<std::vector<uint32_t>> groups;

    // Sort intervals by first_use
    std::vector<size_t> sorted_indices(intervals.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
        [&intervals](size_t a, size_t b) {
            return intervals[a].first_use < intervals[b].first_use;
        });

    // Track which buffers are assigned to groups
    std::vector<int> buffer_to_group(intervals.size(), -1);

    // Greedy interval coloring
    for (size_t idx : sorted_indices) {
        const auto& interval = intervals[idx];

        // Find a compatible group (one whose intervals don't overlap)
        int compatible_group = -1;
        for (size_t g = 0; g < groups.size(); g++) {
            bool overlaps = false;
            for (uint32_t other_id : groups[g]) {
                const auto& other = intervals[other_id];
                // Check for overlap
                if (!(interval.last_use < other.first_use ||
                      other.last_use < interval.first_use)) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                compatible_group = static_cast<int>(g);
                break;
            }
        }

        if (compatible_group >= 0) {
            groups[compatible_group].push_back(interval.buffer_id);
            buffer_to_group[idx] = compatible_group;
        } else {
            // Create new group
            groups.push_back({interval.buffer_id});
            buffer_to_group[idx] = static_cast<int>(groups.size() - 1);
        }
    }

    return groups;
}

size_t compute_minimum_memory(const std::vector<LivenessInterval>& intervals) {
    if (intervals.empty()) return 0;

    // Find max time step
    uint32_t max_step = 0;
    for (const auto& interval : intervals) {
        max_step = std::max(max_step, interval.last_use);
    }

    // For each time step, compute total live memory
    size_t max_live = 0;
    for (uint32_t step = 0; step <= max_step; step++) {
        size_t live = 0;
        for (const auto& interval : intervals) {
            if (step >= interval.first_use && step <= interval.last_use) {
                live += interval.size;
            }
        }
        max_live = std::max(max_live, live);
    }

    return max_live;
}

}  // namespace granite
