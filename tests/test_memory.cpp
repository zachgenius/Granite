#include <catch2/catch_test_macros.hpp>
#include <granite/granite.h>

using namespace granite;

TEST_CASE("MemoryManager basic operations", "[memory]") {
    auto backend = create_backend(BackendType::CPU);
    REQUIRE(backend != nullptr);
    REQUIRE(backend->initialize().ok());

    MemoryManager manager(backend.get());

    SECTION("allocate and release") {
        auto result = manager.allocate(1024);
        REQUIRE(result.ok());

        BufferHandle handle = result.value();
        REQUIRE(handle.valid());

        // Check stats
        REQUIRE(manager.current_usage() > 0);

        manager.release(handle);
    }

    SECTION("allocate different memory types") {
        auto device_result = manager.allocate(512, MemoryType::Device);
        REQUIRE(device_result.ok());

        auto shared_result = manager.allocate(512, MemoryType::Shared);
        REQUIRE(shared_result.ok());

        manager.release(device_result.value());
        manager.release(shared_result.value());
    }

    SECTION("pool reuse") {
        // Allocate and release
        auto result1 = manager.allocate(1024);
        REQUIRE(result1.ok());
        BufferHandle handle1 = result1.value();
        manager.release(handle1);

        // Reset stats to track pool behavior
        size_t initial_hits = manager.stats().pool_hits;

        // Allocate same size again - should reuse from pool
        auto result2 = manager.allocate(1024);
        REQUIRE(result2.ok());

        // Check that we got a pool hit
        REQUIRE(manager.stats().pool_hits > initial_hits);

        manager.release(result2.value());
    }

    SECTION("destroy vs release") {
        auto result = manager.allocate(1024);
        REQUIRE(result.ok());
        BufferHandle handle = result.value();

        // Destroy immediately (not pooled)
        manager.destroy(handle);

        // Memory should be freed, not pooled
        REQUIRE(manager.stats().total_pooled == 0);
    }
}

TEST_CASE("MemoryManager statistics", "[memory]") {
    auto backend = create_backend(BackendType::CPU);
    REQUIRE(backend != nullptr);
    REQUIRE(backend->initialize().ok());

    MemoryManager manager(backend.get());

    SECTION("peak memory tracking") {
        std::vector<BufferHandle> handles;

        // Allocate several buffers
        for (int i = 0; i < 5; i++) {
            auto result = manager.allocate(1024);
            REQUIRE(result.ok());
            handles.push_back(result.value());
        }

        size_t peak_during_alloc = manager.peak_usage();

        // Release all
        for (auto& h : handles) {
            manager.release(h);
        }

        // Peak should still reflect the max
        REQUIRE(manager.peak_usage() == peak_during_alloc);
    }

    SECTION("reset stats") {
        auto result = manager.allocate(2048);
        REQUIRE(result.ok());

        manager.reset_stats();

        // Current/peak should reflect current state
        REQUIRE(manager.current_usage() > 0);
        REQUIRE(manager.peak_usage() > 0);
        // But pool hits/misses should be reset
        REQUIRE(manager.stats().pool_hits == 0);

        manager.release(result.value());
    }
}

TEST_CASE("MemoryManager pool limits", "[memory]") {
    auto backend = create_backend(BackendType::CPU);
    REQUIRE(backend != nullptr);
    REQUIRE(backend->initialize().ok());

    MemoryManager manager(backend.get());

    SECTION("max pool size") {
        // Set a small max pool size
        manager.set_max_pool_size(2048);
        REQUIRE(manager.max_pool_size() == 2048);

        // Allocate and release a large buffer
        auto result = manager.allocate(4096);
        REQUIRE(result.ok());
        manager.release(result.value());

        // Should not be pooled because it exceeds max
        REQUIRE(manager.stats().total_pooled == 0);
    }

    SECTION("clear pool") {
        // Allocate and release to build up pool
        auto result = manager.allocate(1024);
        REQUIRE(result.ok());
        manager.release(result.value());

        REQUIRE(manager.stats().total_pooled > 0);

        manager.clear_pool();
        REQUIRE(manager.stats().total_pooled == 0);
    }
}

TEST_CASE("MemoryManager buffer requests", "[memory]") {
    auto backend = create_backend(BackendType::CPU);
    REQUIRE(backend != nullptr);
    REQUIRE(backend->initialize().ok());

    MemoryManager manager(backend.get());

    SECTION("buffer request with full options") {
        BufferRequest request;
        request.size = 2048;
        request.memory_type = MemoryType::Shared;
        request.alignment = 64;
        request.first_use = 0;
        request.last_use = 5;
        request.allow_aliasing = true;

        auto result = manager.allocate(request);
        REQUIRE(result.ok());
        manager.release(result.value());
    }
}

TEST_CASE("Liveness analysis", "[memory]") {
    SECTION("non-overlapping intervals can alias") {
        std::vector<LivenessInterval> intervals = {
            {0, 0, 2, 1024},  // buffer 0: steps 0-2
            {1, 3, 5, 1024},  // buffer 1: steps 3-5 (doesn't overlap with 0)
            {2, 0, 5, 512},   // buffer 2: steps 0-5 (overlaps with both)
        };

        auto groups = compute_aliasing_groups(intervals);

        // Buffer 0 and 1 should be able to share (non-overlapping)
        // Buffer 2 overlaps with everything so it's separate
        bool buffer_0_and_1_together = false;
        for (const auto& group : groups) {
            bool has_0 = std::find(group.begin(), group.end(), 0) != group.end();
            bool has_1 = std::find(group.begin(), group.end(), 1) != group.end();
            if (has_0 && has_1) {
                buffer_0_and_1_together = true;
            }
        }
        REQUIRE(buffer_0_and_1_together);
    }

    SECTION("compute minimum memory") {
        std::vector<LivenessInterval> intervals = {
            {0, 0, 2, 1024},  // 1KB active steps 0-2
            {1, 1, 3, 2048},  // 2KB active steps 1-3
            {2, 4, 5, 512},   // 512B active steps 4-5
        };

        size_t min_memory = compute_minimum_memory(intervals);

        // At steps 1-2, both buffer 0 and 1 are live: 1024 + 2048 = 3072
        REQUIRE(min_memory == 3072);
    }

    SECTION("empty intervals") {
        std::vector<LivenessInterval> empty;
        auto groups = compute_aliasing_groups(empty);
        REQUIRE(groups.empty());

        size_t min_mem = compute_minimum_memory(empty);
        REQUIRE(min_mem == 0);
    }
}

TEST_CASE("MemoryManager plan allocations", "[memory]") {
    auto backend = create_backend(BackendType::CPU);
    REQUIRE(backend != nullptr);
    REQUIRE(backend->initialize().ok());

    MemoryManager manager(backend.get());

    SECTION("basic plan allocation") {
        std::vector<BufferRequest> requests = {
            {1024, MemoryType::Device, 16, 0, 2, false},
            {2048, MemoryType::Device, 16, 0, 2, false},
        };

        auto result = manager.plan_allocations(requests);
        REQUIRE(result.ok());

        auto handles = result.take();
        REQUIRE(handles.size() == 2);
        REQUIRE(handles[0].valid());
        REQUIRE(handles[1].valid());

        manager.release_plan(handles);
    }

    SECTION("plan with aliasing") {
        // Two non-overlapping buffers that could alias
        std::vector<BufferRequest> requests = {
            {1024, MemoryType::Device, 16, 0, 1, true},  // Active 0-1
            {1024, MemoryType::Device, 16, 2, 3, true},  // Active 2-3 (can alias)
        };

        auto result = manager.plan_allocations(requests);
        REQUIRE(result.ok());

        auto handles = result.take();
        REQUIRE(handles.size() == 2);

        // With aliasing, both should point to the same buffer
        REQUIRE(handles[0] == handles[1]);

        manager.release_plan(handles);
    }

    SECTION("plan with memory budget warning") {
        // Small budget that will be exceeded
        std::vector<BufferRequest> requests = {
            {1024 * 1024, MemoryType::Device, 16, 0, 0, false},  // 1MB
        };

        // This should still succeed but log a warning
        auto result = manager.plan_allocations(requests, 512 * 1024);  // 512KB budget
        REQUIRE(result.ok());

        auto handles = result.take();
        manager.release_plan(handles);
    }
}
