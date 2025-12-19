// Q4_K MatMul Profiler - Captures GPU trace for Xcode analysis
//
// Usage: METAL_CAPTURE_ENABLED=1 ./build/profile_q4k <model.gguf>
// Output: /tmp/granite_gpu_*.gputrace (open in Xcode)

#include <granite/granite.h>
#ifdef GRANITE_HAS_METAL
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <granite/metal_compute.h>
#endif
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <cstdlib>

using namespace granite;
using Clock = std::chrono::high_resolution_clock;

// Generate random Q4_K blocks
void fill_random_q4k(uint8_t* data, size_t num_blocks) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    // Q4_K block: 2 half (d, dmin) + 12 scales + 128 qs = 144 bytes
    for (size_t b = 0; b < num_blocks; b++) {
        uint8_t* block = data + b * 144;
        for (int i = 0; i < 4; i++) block[i] = dist(rng);
        for (int i = 4; i < 16; i++) block[i] = dist(rng);
        for (int i = 16; i < 144; i++) block[i] = dist(rng);
    }
}

int main(int argc, char** argv) {
#ifndef GRANITE_HAS_METAL
    std::cerr << "Metal backend not available\n";
    return 1;
#else
    std::cout << "Q4_K MatMul GPU Profiler\n";
    std::cout << "========================\n\n";

    // Initialize Metal device
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to create Metal device\n";
        return 1;
    }
    std::cout << "Metal device: " << device->name()->utf8String() << "\n\n";

    // Initialize MetalCompute
    MetalCompute gpu;
    auto init_result = gpu.initialize(device);
    if (!init_result.ok()) {
        std::cerr << "Failed to initialize MetalCompute: " << init_result.error().message() << "\n";
        return 1;
    }

    // Test sizes matching TinyLlama prefill
    struct TestCase {
        uint32_t M, K, N;
        const char* name;
    };

    std::vector<TestCase> tests = {
        {32, 2048, 2048, "Q proj (pp32)"},
        {128, 2048, 2048, "Q proj (pp128)"},
        {256, 2048, 2048, "Q proj (pp256)"},
        {512, 2048, 2048, "Q proj (pp512)"},
        {512, 2048, 5632, "FFN gate (pp512)"},
        {512, 5632, 2048, "FFN down (pp512)"},
    };

    // Find max sizes
    uint32_t max_M = 512, max_K = 5632, max_N = 5632;

    std::cout << "Allocating test buffers...\n";

    // Allocate input/output as float
    MTL::Buffer* X = gpu.create_buffer(max_M * max_K * sizeof(float), true);
    MTL::Buffer* Y = gpu.create_buffer(max_M * max_N * sizeof(float), true);

    // Allocate Q4_K weights (144 bytes per block, K/256 blocks per row)
    size_t q4k_block_size = 144;
    size_t max_blocks_per_row = (max_K + 255) / 256;
    MTL::Buffer* W = gpu.create_buffer(max_N * max_blocks_per_row * q4k_block_size, true);

    if (!X || !Y || !W) {
        std::cerr << "Failed to allocate buffers\n";
        return 1;
    }

    // Initialize buffers
    {
        float* x_data = (float*)X->contents();
        for (size_t i = 0; i < max_M * max_K; i++) {
            x_data[i] = 0.01f * (i % 100);
        }
        fill_random_q4k((uint8_t*)W->contents(), max_N * max_blocks_per_row);
    }

    std::cout << "Warming up...\n";

    // Warmup runs
    for (int i = 0; i < 5; i++) {
        gpu.matmul_q4k(X, W, Y, 128, 2048, 2048);
        gpu.sync();
    }

    std::cout << "\n=== Starting GPU Capture ===\n";
    std::cout << "Running " << tests.size() << " test cases, 10 iterations each...\n\n";

    // Begin GPU capture
    bool capturing = gpu.begin_capture();
    if (!capturing) {
        std::cout << "Note: GPU capture not started. To enable:\n";
        std::cout << "  METAL_CAPTURE_ENABLED=1 ./build/profile_q4k\n\n";
    }

    // Run tests with capture
    std::cout << std::setw(20) << "Test"
              << std::setw(10) << "M"
              << std::setw(10) << "K"
              << std::setw(10) << "N"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "TFLOPS\n";
    std::cout << std::string(80, '-') << "\n";

    for (auto& test : tests) {
        // Run multiple iterations
        auto start = Clock::now();

        for (int iter = 0; iter < 10; iter++) {
            gpu.matmul_q4k(X, W, Y, test.M, test.K, test.N);
        }
        gpu.sync();

        auto end = Clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double time_per_iter = time_ms / 10.0;

        // Calculate TFLOPS (2*M*N*K FLOPs per matmul)
        double flops = 2.0 * test.M * test.N * test.K;
        double tflops = (flops / 1e12) / (time_per_iter / 1000.0);

        std::cout << std::setw(20) << test.name
                  << std::setw(10) << test.M
                  << std::setw(10) << test.K
                  << std::setw(10) << test.N
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_per_iter
                  << std::setw(15) << std::fixed << std::setprecision(2) << tflops << "\n";
    }

    // End capture
    if (capturing) {
        gpu.end_capture();
        std::cout << "\n=== GPU Capture Complete ===\n";
        std::cout << "Open the .gputrace file in Xcode to analyze:\n";
        std::cout << "  1. File > Open > /tmp/granite_gpu_*.gputrace\n";
        std::cout << "  2. Click on 'matmul_q4k_simdgroup' kernel\n";
        std::cout << "  3. Check Shader Profiler for:\n";
        std::cout << "     - ALU utilization\n";
        std::cout << "     - Memory bandwidth\n";
        std::cout << "     - Occupancy\n";
        std::cout << "     - Stall reasons\n";
    }

    std::cout << "\n=== Analysis Summary ===\n";
    std::cout << "M3 Max theoretical peak:\n";
    std::cout << "  FP32: ~14 TFLOPS\n";
    std::cout << "  FP16: ~28 TFLOPS\n";
    std::cout << "\nTarget: >5 TFLOPS for Q4_K matmul (llama.cpp achieves ~9.6 TFLOPS at pp512)\n";

    // Cleanup
    X->release();
    Y->release();
    W->release();

    return 0;
#endif
}
