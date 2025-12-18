// benchmark_attention.mm - Isolated benchmark for attention kernel performance
//
// Tests raw attention kernel performance without transformer overhead

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

struct BenchResult {
    double avg_us;
    double min_us;
    double max_us;
    double bandwidth_gb_s;
    double efficiency_pct;
};

#ifdef GRANITE_HAS_METAL
BenchResult run_benchmark(
    MetalCompute& gpu,
    std::function<void()> kernel_func,
    size_t total_bytes,
    int warmup_runs,
    int benchmark_runs,
    int batch_size
) {
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        kernel_func();
        gpu.sync();
    }

    // Benchmark
    std::vector<double> times;
    times.reserve(benchmark_runs);

    for (int i = 0; i < benchmark_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < batch_size; j++) {
            kernel_func();
        }
        gpu.sync();
        auto end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(end - start).count() / batch_size;
        times.push_back(us);
    }

    // Statistics
    double avg = 0, min_t = times[0], max_t = times[0];
    for (double t : times) {
        avg += t;
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }
    avg /= times.size();

    double bandwidth_gb_s = (total_bytes / 1e9) / (avg / 1e6);
    double theoretical_gb_s = 400.0;  // M3 Max
    double efficiency = (bandwidth_gb_s / theoretical_gb_s) * 100.0;

    return {avg, min_t, max_t, bandwidth_gb_s, efficiency};
}
#endif

int main(int argc, char** argv) {
#ifndef GRANITE_HAS_METAL
    std::cerr << "Metal backend not available\n";
    return 1;
#else
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║            Attention Kernel Benchmark                            ║\n";
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
        uint32_t num_heads;
        uint32_t num_kv_heads;
        uint32_t seq_kv;
        uint32_t head_dim;
        const char* name;
    };

    std::vector<TestCase> tests = {
        // TinyLlama 1.1B sizes
        {32, 4, 256, 64, "TinyLlama seq=256"},
        {32, 4, 512, 64, "TinyLlama seq=512"},
        {32, 4, 1024, 64, "TinyLlama seq=1024"},
        {32, 4, 2048, 64, "TinyLlama seq=2048"},
        // Larger model sizes
        {32, 8, 512, 128, "7B-style seq=512"},
        {32, 8, 1024, 128, "7B-style seq=1024"},
        {32, 8, 2048, 128, "7B-style seq=2048"},
    };

    constexpr int warmup_runs = 5;
    constexpr int benchmark_runs = 50;
    constexpr int batch_size = 10;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Running " << benchmark_runs << " iterations, " << batch_size << " per batch\n\n";

    // Header
    std::cout << "┌────────────────────────┬──────────┬──────────┬──────────────┐\n";
    std::cout << "│ Test                   │  Avg µs  │  BW GB/s │  Eff %       │\n";
    std::cout << "├────────────────────────┼──────────┼──────────┼──────────────┤\n";

    for (const auto& test : tests) {
        // Calculate sizes
        size_t q_bytes = test.num_heads * test.head_dim * sizeof(float);
        size_t k_bytes = test.num_kv_heads * test.seq_kv * test.head_dim * sizeof(uint16_t);
        size_t v_bytes = k_bytes;
        size_t output_bytes = test.num_heads * test.head_dim * sizeof(float);
        size_t total_bytes = q_bytes + k_bytes + v_bytes + output_bytes;

        // Allocate buffers
        MTL::Buffer* q_buf = gpu.create_buffer(q_bytes);
        MTL::Buffer* k_buf = gpu.create_buffer(k_bytes);
        MTL::Buffer* v_buf = gpu.create_buffer(v_bytes);
        MTL::Buffer* output_buf = gpu.create_buffer(output_bytes);

        if (!q_buf || !k_buf || !v_buf || !output_buf) {
            std::cerr << "Failed to allocate buffers for " << test.name << "\n";
            continue;
        }

        // Initialize with random data
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        float* q_ptr = static_cast<float*>(q_buf->contents());
        for (uint32_t i = 0; i < test.num_heads * test.head_dim; i++) {
            q_ptr[i] = dist(rng);
        }

        // Initialize K/V as FP16
        uint16_t* k_ptr = static_cast<uint16_t*>(k_buf->contents());
        uint16_t* v_ptr = static_cast<uint16_t*>(v_buf->contents());
        for (uint32_t i = 0; i < test.num_kv_heads * test.seq_kv * test.head_dim; i++) {
            float val = dist(rng);
            uint32_t bits;
            memcpy(&bits, &val, sizeof(float));
            uint16_t sign = (bits >> 16) & 0x8000;
            int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
            uint16_t mant = (bits >> 13) & 0x3FF;
            if (exp <= 0) exp = 0;
            if (exp >= 31) exp = 31;
            k_ptr[i] = sign | (exp << 10) | mant;
            v_ptr[i] = k_ptr[i];
        }

        float scale = 1.0f / sqrtf(static_cast<float>(test.head_dim));
        uint32_t seq_q = 1;

        // Benchmark attention kernel
        auto kernel = [&]() {
            gpu.multihead_attention(
                q_buf, k_buf, v_buf, output_buf,
                test.num_heads, test.num_kv_heads, seq_q, test.seq_kv,
                test.head_dim, scale
            );
        };
        BenchResult result = run_benchmark(gpu, kernel, total_bytes,
                                           warmup_runs, benchmark_runs, batch_size);

        // Print results
        std::cout << "│ " << std::setw(22) << std::left << test.name << " │ "
                  << std::setw(8) << std::right << result.avg_us << " │ "
                  << std::setw(8) << result.bandwidth_gb_s << " │ "
                  << std::setw(6) << result.efficiency_pct << "%      │\n";

        // Cleanup
        q_buf->release();
        k_buf->release();
        v_buf->release();
        output_buf->release();
    }

    std::cout << "└────────────────────────┴──────────┴──────────┴──────────────┘\n\n";

    std::cout << "Legend:\n";
    std::cout << "  Eff %: Memory bandwidth efficiency (actual / 400 GB/s)\n";

    return 0;
#endif
}
