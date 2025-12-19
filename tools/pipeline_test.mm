// Test kernel pipelining - compare tight matmul loop vs prefill-like pattern
#include <granite/granite.h>
#ifdef GRANITE_HAS_METAL
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <granite/metal_compute.h>
#endif
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

using namespace granite;
using Clock = std::chrono::high_resolution_clock;

void fill_random_q4k(uint8_t* data, size_t num_blocks) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (size_t b = 0; b < num_blocks; b++) {
        uint8_t* block = data + b * 144;
        for (int i = 0; i < 144; i++) block[i] = dist(rng);
    }
}

int main() {
#ifndef GRANITE_HAS_METAL
    std::cerr << "Metal not available\n";
    return 1;
#else
    std::cout << "GPU Pipeline Performance Test\n";
    std::cout << "==============================\n\n";

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MetalCompute gpu;
    gpu.initialize(device);

    // Allocate buffers matching TinyLlama prefill at pp512
    // Per layer: Q, K, V, O matmuls + gate, up, down FFN
    int M = 512, hidden = 2048, intermediate = 5632;
    int q_dim = 2048, kv_dim = 256;

    // Input/output buffers
    MTL::Buffer* X = gpu.create_buffer(M * hidden * sizeof(float), true);
    MTL::Buffer* Y = gpu.create_buffer(M * hidden * sizeof(float), true);

    // Q4_K weight buffers (7 per layer × 22 layers = 154 total, but we reuse)
    std::vector<MTL::Buffer*> weights;
    weights.push_back(gpu.create_buffer(q_dim * 8 * 144, true));      // Q
    weights.push_back(gpu.create_buffer(kv_dim * 8 * 144, true));     // K
    weights.push_back(gpu.create_buffer(kv_dim * 8 * 144, true));     // V
    weights.push_back(gpu.create_buffer(q_dim * 8 * 144, true));      // O
    weights.push_back(gpu.create_buffer(intermediate * 8 * 144, true)); // Gate
    weights.push_back(gpu.create_buffer(intermediate * 8 * 144, true)); // Up
    weights.push_back(gpu.create_buffer(hidden * 22 * 144, true));    // Down

    // Initialize
    float* x_data = (float*)X->contents();
    for (int i = 0; i < M * hidden; i++) x_data[i] = 0.01f * (i % 100);
    for (auto* w : weights) fill_random_q4k((uint8_t*)w->contents(), w->length() / 144);

    // Intermediate buffers (simulating layer processing)
    MTL::Buffer* q_buf = gpu.create_buffer(M * q_dim * sizeof(float), true);
    MTL::Buffer* k_buf = gpu.create_buffer(M * kv_dim * sizeof(float), true);
    MTL::Buffer* gate_buf = gpu.create_buffer(M * intermediate * sizeof(float), true);

    // Warmup
    for (int i = 0; i < 3; i++) {
        gpu.matmul_q4k(X, weights[0], q_buf, M, hidden, q_dim);
        gpu.sync();
    }

    int num_layers = 22;
    int iterations = 5;

    // Test 1: Just matmuls, no intermediate ops
    std::cout << "Test 1: Sequential matmuls only (no RMSNorm, attention, etc.)\n";
    {
        auto start = Clock::now();
        for (int iter = 0; iter < iterations; iter++) {
            for (int layer = 0; layer < num_layers; layer++) {
                // Attention: Q, K, V, O
                gpu.matmul_q4k(X, weights[0], q_buf, M, hidden, q_dim);
                gpu.matmul_q4k(X, weights[1], k_buf, M, hidden, kv_dim);
                gpu.matmul_q4k(X, weights[2], k_buf, M, hidden, kv_dim);
                gpu.matmul_q4k(q_buf, weights[3], Y, M, q_dim, hidden);
                // FFN: gate, up, down
                gpu.matmul_q4k(Y, weights[4], gate_buf, M, hidden, intermediate);
                gpu.matmul_q4k(Y, weights[5], gate_buf, M, hidden, intermediate);
                gpu.matmul_q4k(gate_buf, weights[6], Y, M, intermediate, hidden);
            }
        }
        gpu.sync();
        auto end = Clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double per_iter = total_ms / iterations;

        // Calculate FLOPs
        double flops_per_layer = 2.0 * M * (
            hidden * q_dim +      // Q
            hidden * kv_dim +     // K
            hidden * kv_dim +     // V
            q_dim * hidden +      // O
            hidden * intermediate + // gate
            hidden * intermediate + // up
            intermediate * hidden   // down
        );
        double total_flops = flops_per_layer * num_layers;
        double tflops = (total_flops / 1e12) / (per_iter / 1000.0);

        std::cout << "  Time/iteration: " << std::fixed << std::setprecision(1) << per_iter << " ms\n";
        std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n\n";
    }

    // Test 2: Matmuls with RMSNorm and element-wise ops interleaved
    std::cout << "Test 2: Matmuls + RMSNorm + element-wise (simulating prefill)\n";
    {
        // Create norm weights (FP16)
        MTL::Buffer* norm_w = gpu.create_buffer(hidden * 2, true);  // 2 bytes per half
        MTL::Buffer* norm_out = gpu.create_buffer(M * hidden * sizeof(float), true);

        auto start = Clock::now();
        for (int iter = 0; iter < iterations; iter++) {
            for (int layer = 0; layer < num_layers; layer++) {
                // RMSNorm
                gpu.rms_norm_batch_f16(X, norm_w, norm_out, M, hidden, 1e-5f);

                // Attention matmuls
                gpu.matmul_q4k(norm_out, weights[0], q_buf, M, hidden, q_dim);
                gpu.matmul_q4k(norm_out, weights[1], k_buf, M, hidden, kv_dim);
                gpu.matmul_q4k(norm_out, weights[2], k_buf, M, hidden, kv_dim);
                gpu.matmul_q4k(q_buf, weights[3], Y, M, q_dim, hidden);

                // Residual add
                gpu.elementwise_add(X, Y, norm_out, M * hidden);

                // RMSNorm
                gpu.rms_norm_batch_f16(norm_out, norm_w, Y, M, hidden, 1e-5f);

                // FFN matmuls
                gpu.matmul_q4k(Y, weights[4], gate_buf, M, hidden, intermediate);
                gpu.matmul_q4k(Y, weights[5], gate_buf, M, hidden, intermediate);
                gpu.silu_mul(gate_buf, gate_buf, gate_buf, M * intermediate);
                gpu.matmul_q4k(gate_buf, weights[6], Y, M, intermediate, hidden);

                // Residual add
                gpu.elementwise_add(norm_out, Y, X, M * hidden);
            }
        }
        gpu.sync();
        auto end = Clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double per_iter = total_ms / iterations;

        // Same FLOPs for matmuls (ignoring RMSNorm/elementwise which are tiny)
        double flops_per_layer = 2.0 * M * (
            hidden * q_dim + hidden * kv_dim + hidden * kv_dim + q_dim * hidden +
            hidden * intermediate + hidden * intermediate + intermediate * hidden
        );
        double total_flops = flops_per_layer * num_layers;
        double tflops = (total_flops / 1e12) / (per_iter / 1000.0);

        std::cout << "  Time/iteration: " << std::fixed << std::setprecision(1) << per_iter << " ms\n";
        std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n\n";

        norm_w->release();
        norm_out->release();
    }

    // Test 3: Full prefill simulation including attention
    std::cout << "Test 3: Full prefill simulation (including attention)\n";
    {
        // Create norm weights (FP16)
        MTL::Buffer* norm_w = gpu.create_buffer(hidden * 2, true);
        MTL::Buffer* norm_out = gpu.create_buffer(M * hidden * sizeof(float), true);

        // KV cache (FP16)
        int max_seq = 1024;
        int num_kv_heads = 4, head_dim = 64;
        MTL::Buffer* k_cache = gpu.create_buffer(max_seq * num_kv_heads * head_dim * 2, true);  // FP16
        MTL::Buffer* v_cache = gpu.create_buffer(max_seq * num_kv_heads * head_dim * 2, true);

        // Attention output
        MTL::Buffer* attn_out = gpu.create_buffer(M * q_dim * sizeof(float), true);

        auto start = Clock::now();
        for (int iter = 0; iter < iterations; iter++) {
            for (int layer = 0; layer < num_layers; layer++) {
                // RMSNorm
                gpu.rms_norm_batch_f16(X, norm_w, norm_out, M, hidden, 1e-5f);

                // Q/K/V projections
                gpu.matmul_q4k(norm_out, weights[0], q_buf, M, hidden, q_dim);
                gpu.matmul_q4k(norm_out, weights[1], k_buf, M, hidden, kv_dim);
                gpu.matmul_q4k(norm_out, weights[2], k_buf, M, hidden, kv_dim);

                // RoPE
                gpu.rope_multihead(q_buf, k_buf, 32, num_kv_heads, M, head_dim, 0, 10000.0f);

                // KV cache append
                gpu.kv_cache_append(k_cache, v_cache, k_buf, k_buf,
                                   num_kv_heads, head_dim, 0, M, max_seq);

                // Attention
                float scale = 1.0f / std::sqrt((float)head_dim);
                gpu.multihead_attention(q_buf, k_cache, v_cache, attn_out,
                                       32, num_kv_heads, M, M, head_dim, scale);

                // O projection
                gpu.matmul_q4k(attn_out, weights[3], Y, M, q_dim, hidden);

                // Residual add
                gpu.elementwise_add(X, Y, norm_out, M * hidden);

                // RMSNorm
                gpu.rms_norm_batch_f16(norm_out, norm_w, Y, M, hidden, 1e-5f);

                // FFN
                gpu.matmul_q4k(Y, weights[4], gate_buf, M, hidden, intermediate);
                gpu.matmul_q4k(Y, weights[5], gate_buf, M, hidden, intermediate);
                gpu.silu_mul(gate_buf, gate_buf, gate_buf, M * intermediate);
                gpu.matmul_q4k(gate_buf, weights[6], Y, M, intermediate, hidden);

                // Residual add
                gpu.elementwise_add(norm_out, Y, X, M * hidden);
            }
        }
        gpu.sync();
        auto end = Clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double per_iter = total_ms / iterations;

        double flops_per_layer = 2.0 * M * (
            hidden * q_dim + hidden * kv_dim + hidden * kv_dim + q_dim * hidden +
            hidden * intermediate + hidden * intermediate + intermediate * hidden
        );
        double total_flops = flops_per_layer * num_layers;
        double tflops = (total_flops / 1e12) / (per_iter / 1000.0);

        std::cout << "  Time/iteration: " << std::fixed << std::setprecision(1) << per_iter << " ms\n";
        std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n\n";

        norm_w->release();
        norm_out->release();
        k_cache->release();
        v_cache->release();
        attn_out->release();
    }

    std::cout << "=== Reference ===\n";
    std::cout << "Isolated matmul: 8.79 TFLOPS\n";
    std::cout << "Full prefill: ~3.34 TFLOPS (1500 tok/s at pp512)\n";
    std::cout << "llama.cpp prefill: ~9.6 TFLOPS (4400 tok/s at pp512)\n";

    // Cleanup
    X->release();
    Y->release();
    q_buf->release();
    k_buf->release();
    gate_buf->release();
    for (auto* w : weights) w->release();

    return 0;
#endif
}
