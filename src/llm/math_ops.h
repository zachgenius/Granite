#pragma once

// Internal math operations for LLM
// FP16 conversion, BLAS wrappers, parallel utilities

#include <cstdint>
#include <cstddef>
#include <vector>
#include <thread>
#include <algorithm>

namespace granite {

// FP16 <-> FP32 conversion
float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);
void fp16_to_fp32_array(const uint16_t* src, float* dst, size_t n);

// Threading utilities
size_t get_num_threads();

template<typename Func>
void parallel_for(size_t n, size_t num_threads, Func&& f);

// Matrix operations
// C = A @ B^T where A: [M, K], B: [N, K] (transposed), C: [M, N]
void matmul_transb(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_transb_fp16(const float* A, const uint16_t* B_fp16, float* C, int M, int N, int K);

// Activations
void silu_inplace(float* x, int n);

// Element-wise operations
void elementwise_mul(const float* a, const float* b, float* c, int n);

}  // namespace granite

// Template implementation
namespace granite {

template<typename Func>
void parallel_for(size_t n, size_t num_threads, Func&& f) {
    if (num_threads <= 1 || n < num_threads) {
        for (size_t i = 0; i < n; i++) {
            f(i);
        }
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    size_t chunk_size = (n + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        if (start >= n) break;

        threads.emplace_back([&f, start, end]() {
            for (size_t i = start; i < end; i++) {
                f(i);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }
}

}  // namespace granite
