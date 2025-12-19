// Measure tensor/buffer allocation overhead
#include <granite/granite.h>
#ifdef GRANITE_HAS_METAL
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <granite/metal_compute.h>
#endif
#include <iostream>
#include <chrono>

using namespace granite;
using Clock = std::chrono::high_resolution_clock;

int main() {
#ifndef GRANITE_HAS_METAL
    std::cerr << "Metal not available\n";
    return 1;
#else
    // Initialize
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MetalCompute gpu;
    gpu.initialize(device);

    auto backend = create_backend(BackendType::Metal);

    int num_iterations = 100;
    int num_allocs = 22 * 4;  // 22 layers × 4 tensors per layer

    std::cout << "Measuring allocation overhead (" << num_allocs << " allocations per iteration)...\n\n";

    // Measure Tensor::allocate
    {
        std::vector<int64_t> shape = {1, 512, 2048};  // pp512, hidden_dim=2048

        auto start = Clock::now();
        for (int iter = 0; iter < num_iterations; iter++) {
            for (int i = 0; i < num_allocs; i++) {
                auto result = Tensor::allocate(shape, DataType::FP32, backend.get());
                if (result.ok()) {
                    auto tensor = std::move(result).take();
                    // Tensor goes out of scope and is deallocated
                }
            }
        }
        auto end = Clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double per_iter_ms = total_ms / num_iterations;
        double per_alloc_us = (total_ms * 1000) / (num_iterations * num_allocs);

        std::cout << "Tensor::allocate:\n";
        std::cout << "  Total: " << total_ms << " ms for " << num_iterations << " iterations\n";
        std::cout << "  Per iteration: " << per_iter_ms << " ms\n";
        std::cout << "  Per allocation: " << per_alloc_us << " us\n\n";
    }

    // Measure gpu.create_buffer
    {
        size_t size = 512 * 2048 * sizeof(float);

        auto start = Clock::now();
        for (int iter = 0; iter < num_iterations; iter++) {
            for (int i = 0; i < num_allocs; i++) {
                MTL::Buffer* buf = gpu.create_buffer(size);
                buf->release();
            }
        }
        auto end = Clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double per_iter_ms = total_ms / num_iterations;
        double per_alloc_us = (total_ms * 1000) / (num_iterations * num_allocs);

        std::cout << "gpu.create_buffer:\n";
        std::cout << "  Total: " << total_ms << " ms for " << num_iterations << " iterations\n";
        std::cout << "  Per iteration: " << per_iter_ms << " ms\n";
        std::cout << "  Per allocation: " << per_alloc_us << " us\n\n";
    }

    // For reference: prefill takes ~330ms at pp512
    // If allocation is significant, we'd see a meaningful fraction here
    std::cout << "Reference: pp512 prefill takes ~330ms\n";
    std::cout << "Allocation overhead would be significant if > 10ms per iteration\n";

    return 0;
#endif
}
