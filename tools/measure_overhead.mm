// Measure various CPU-side overhead sources
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
    if (!backend) {
        std::cerr << "Failed to create backend\n";
        return 1;
    }

    // Initialize the backend
    auto init_result = backend->initialize();
    if (!init_result.ok()) {
        std::cerr << "Failed to initialize backend: " << init_result.error().message() << "\n";
        return 1;
    }

    int num_iterations = 10000;

    std::cout << "Measuring CPU-side overhead sources...\n\n";

    // Create a test tensor
    std::vector<int64_t> shape = {1, 512, 2048};
    auto tensor_result = Tensor::allocate(shape, DataType::FP32, backend.get());
    if (!tensor_result.ok()) {
        std::cerr << "Failed to allocate tensor\n";
        return 1;
    }
    auto tensor = std::move(tensor_result).take();

    // Measure get_native_buffer
    {
        auto start = Clock::now();
        for (int i = 0; i < num_iterations; i++) {
            volatile auto* buf = backend->get_native_buffer(tensor.buffer());
            (void)buf;
        }
        auto end = Clock::now();

        double total_us = std::chrono::duration<double, std::micro>(end - start).count();
        double per_call_ns = (total_us * 1000) / num_iterations;

        std::cout << "get_native_buffer:\n";
        std::cout << "  Per call: " << per_call_ns << " ns\n";
        std::cout << "  For 200 calls/prefill: " << (per_call_ns * 200 / 1e6) << " ms\n\n";
    }

    // Measure matmul_q4k call overhead (empty kernel, just dispatch setup)
    {
        // Create buffers for matmul
        MTL::Buffer* X = gpu.create_buffer(512 * 2048 * sizeof(float));
        MTL::Buffer* W = gpu.create_buffer(2048 * 22 * 144);  // Q4_K
        MTL::Buffer* Y = gpu.create_buffer(512 * 2048 * sizeof(float));

        // Warmup
        for (int i = 0; i < 10; i++) {
            gpu.matmul_q4k(X, W, Y, 512, 2048, 2048);
        }
        gpu.sync();

        // Measure dispatch time (without sync)
        auto start = Clock::now();
        for (int i = 0; i < num_iterations; i++) {
            gpu.matmul_q4k(X, W, Y, 512, 2048, 2048);
        }
        auto end = Clock::now();

        double total_us = std::chrono::duration<double, std::micro>(end - start).count();
        double per_call_us = total_us / num_iterations;

        std::cout << "matmul_q4k dispatch (no sync):\n";
        std::cout << "  Per call: " << per_call_us << " us\n";
        std::cout << "  For 88 calls/prefill: " << (per_call_us * 88 / 1000) << " ms\n\n";

        // Now sync to flush
        gpu.sync();

        X->release();
        W->release();
        Y->release();
    }

    // Measure sync overhead
    {
        // Create a simple buffer for dummy ops
        MTL::Buffer* X = gpu.create_buffer(1024 * sizeof(float));
        MTL::Buffer* Y = gpu.create_buffer(1024 * sizeof(float));

        // Measure sync time with minimal work
        auto start = Clock::now();
        for (int i = 0; i < 100; i++) {
            gpu.elementwise_add(X, Y, X, 1024);
            gpu.sync();
        }
        auto end = Clock::now();

        double total_us = std::chrono::duration<double, std::micro>(end - start).count();
        double per_sync_us = total_us / 100;

        std::cout << "sync() overhead (with tiny kernel):\n";
        std::cout << "  Per sync: " << per_sync_us << " us\n\n";

        X->release();
        Y->release();
    }

    // Count expected dispatches vs compute time
    std::cout << "=== Analysis ===\n";
    std::cout << "If each of 148 dispatches takes ~10us overhead:\n";
    std::cout << "  Total overhead: " << (148 * 10 / 1000.0) << " ms\n";
    std::cout << "\nActual prefill time: ~330ms\n";
    std::cout << "Expected compute time at 8.79 TFLOPS: ~128ms for pp512\n";
    std::cout << "Gap: ~200ms unaccounted for\n";

    return 0;
#endif
}
