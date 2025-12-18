// benchmark_matvec.cpp - Isolated benchmark for matvec kernel performance
//
// Tests raw kernel performance without transformer overhead
// Measures memory bandwidth utilization for Q4_K matvec

#include <granite/granite.h>
#ifdef GRANITE_HAS_METAL
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <granite/metal_compute.h>
#endif
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

using namespace granite;

struct MatvecBenchResult {
    double avg_time_us;
    double min_time_us;
    double max_time_us;
    double throughput_gb_s;
    double theoretical_gb_s;
    double efficiency_pct;
};

// Generate random Q4_K blocks
void fill_random_q4k(uint8_t* data, size_t num_blocks) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    // Q4_K block: 2 half (d, dmin) + 12 scales + 128 qs = 144 bytes
    for (size_t b = 0; b < num_blocks; b++) {
        uint8_t* block = data + b * 144;
        // d and dmin as half (random bits)
        for (int i = 0; i < 4; i++) block[i] = dist(rng);
        // scales (12 bytes)
        for (int i = 4; i < 16; i++) block[i] = dist(rng);
        // qs (128 bytes)
        for (int i = 16; i < 144; i++) block[i] = dist(rng);
    }
}

int main(int argc, char** argv) {
#ifndef GRANITE_HAS_METAL
    std::cerr << "Metal backend not available\n";
    return 1;
#else
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Q4_K Matvec Kernel Benchmark                           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    // Initialize Metal device
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to create Metal device\n";
        return 1;
    }
    std::cout << "Metal device: " << device->name()->utf8String() << "\n";

    // Initialize MetalCompute
    MetalCompute gpu;
    auto init_result = gpu.initialize(device);
    if (!init_result.ok()) {
        std::cerr << "Failed to initialize MetalCompute: " << init_result.error().message() << "\n";
        return 1;
    }
    std::cout << "\n";

    // Benchmark configuration
    struct TestCase {
        uint32_t K;  // Input dimension (hidden_dim)
        uint32_t N;  // Output dimension
        const char* name;
    };

    std::vector<TestCase> tests = {
        // TinyLlama sizes
        {2048, 2048, "Q projection (2048 -> 2048)"},
        {2048, 256, "K/V projection (2048 -> 256)"},
        {2048, 5632, "FFN gate/up (2048 -> 5632)"},
        {5632, 2048, "FFN down (5632 -> 2048)"},
        // Larger model sizes
        {4096, 4096, "Large Q (4096 -> 4096)"},
        {4096, 11008, "Large FFN (4096 -> 11008)"},
    };

    constexpr int warmup_runs = 10;
    constexpr int benchmark_runs = 100;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Running " << benchmark_runs << " iterations per test (+" << warmup_runs << " warmup)\n\n";
    std::cout << "┌────────────────────────────────────┬────────────┬────────────┬────────────┬──────────────┬─────────┐\n";
    std::cout << "│ Test                               │ Avg (µs)   │ Min (µs)   │ Max (µs)   │ Bandwidth    │ Eff %   │\n";
    std::cout << "├────────────────────────────────────┼────────────┼────────────┼────────────┼──────────────┼─────────┤\n";

    for (const auto& test : tests) {
        // Calculate sizes
        uint32_t num_blocks = test.N * (test.K / 256);  // Q4_K: 256 elements per block
        size_t weight_bytes = num_blocks * 144;  // 144 bytes per Q4_K block
        size_t input_bytes = test.K * sizeof(float);
        size_t output_bytes = test.N * sizeof(float);
        size_t total_bytes = weight_bytes + input_bytes + output_bytes;

        // Allocate buffers
        MTL::Buffer* x_buf = gpu.create_buffer(input_bytes);
        MTL::Buffer* w_buf = gpu.create_buffer(weight_bytes);
        MTL::Buffer* y_buf = gpu.create_buffer(output_bytes);

        if (!x_buf || !w_buf || !y_buf) {
            std::cerr << "Failed to allocate buffers for " << test.name << "\n";
            continue;
        }

        // Initialize with random data
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        float* x_ptr = static_cast<float*>(x_buf->contents());
        for (uint32_t i = 0; i < test.K; i++) x_ptr[i] = dist(rng);

        uint8_t* w_ptr = static_cast<uint8_t*>(w_buf->contents());
        fill_random_q4k(w_ptr, num_blocks);

        // Warmup
        for (int i = 0; i < warmup_runs; i++) {
            gpu.matvec_q4k(x_buf, w_buf, y_buf, test.K, test.N);
            gpu.sync();
        }

        // Benchmark - batch multiple operations before sync to amortize overhead
        std::vector<double> times;
        times.reserve(benchmark_runs);
        constexpr int batch_size = 10;  // Dispatch this many kernels before sync

        for (int i = 0; i < benchmark_runs; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < batch_size; j++) {
                gpu.matvec_q4k(x_buf, w_buf, y_buf, test.K, test.N);
            }
            gpu.sync();
            auto end = std::chrono::high_resolution_clock::now();

            double us = std::chrono::duration<double, std::micro>(end - start).count() / batch_size;
            times.push_back(us);
        }

        // Calculate statistics
        double avg = 0, min_t = times[0], max_t = times[0];
        for (double t : times) {
            avg += t;
            min_t = std::min(min_t, t);
            max_t = std::max(max_t, t);
        }
        avg /= times.size();

        // Calculate bandwidth (GB/s)
        double bandwidth_gb_s = (total_bytes / 1e9) / (avg / 1e6);

        // Theoretical bandwidth (M3 Max ~ 400 GB/s)
        double theoretical_gb_s = 400.0;
        double efficiency = (bandwidth_gb_s / theoretical_gb_s) * 100.0;

        // Print results
        std::cout << "│ " << std::setw(34) << std::left << test.name << " │ "
                  << std::setw(10) << std::right << avg << " │ "
                  << std::setw(10) << min_t << " │ "
                  << std::setw(10) << max_t << " │ "
                  << std::setw(7) << bandwidth_gb_s << " GB/s │ "
                  << std::setw(6) << efficiency << "% │\n";

        // Cleanup
        x_buf->release();
        w_buf->release();
        y_buf->release();
    }

    std::cout << "└────────────────────────────────────┴────────────┴────────────┴────────────┴──────────────┴─────────┘\n\n";

    // Additional analysis
    std::cout << "Analysis:\n";
    std::cout << "- M3 Max theoretical bandwidth: ~400 GB/s\n";
    std::cout << "- Good efficiency: > 60%\n";
    std::cout << "- If efficiency < 20%, kernel is compute-bound or has memory access issues\n";

    return 0;
#endif
}
