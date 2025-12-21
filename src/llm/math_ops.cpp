#include "math_ops.h"

#include <cstring>
#include <cmath>
#include <thread>
#include <vector>

#ifdef GRANITE_HAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

#ifdef GRANITE_HAS_OPENMP
#include <omp.h>
#endif

namespace granite {

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign << 31;
            float result;
            std::memcpy(&result, &bits, 4);
            return result;
        }
        while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
    } else if (exp == 31) {
        uint32_t bits = (sign << 31) | 0x7F800000 | (mant << 13);
        float result;
        std::memcpy(&result, &bits, 4);
        return result;
    }

    uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

size_t get_num_threads() {
    return std::thread::hardware_concurrency();
}

void fp16_to_fp32_array(const uint16_t* src, float* dst, size_t n) {
    size_t num_threads = get_num_threads();
    parallel_for(n, num_threads, [src, dst](size_t i) {
        dst[i] = fp16_to_fp32(src[i]);
    });
}

void matmul_transb(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
#ifdef GRANITE_HAS_ACCELERATE
    // Use Apple's BLAS: C = alpha * A * B^T + beta * C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, A, K,
                B, K,
                0.0f, C, N);
#else
    // Fallback: naive implementation with OpenMP
    #ifdef GRANITE_HAS_OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void matmul_transb_fp16(
    const float* A, const uint16_t* B_fp16, float* C,
    int M, int N, int K)
{
    // Convert B to FP32 and use BLAS
    std::vector<float> B_fp32(N * K);
    fp16_to_fp32_array(B_fp16, B_fp32.data(), N * K);
    matmul_transb(A, B_fp32.data(), C, M, N, K);
}

void silu_inplace(float* x, int n) {
#ifdef GRANITE_HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = val / (1.0f + std::exp(-val));
    }
}

void elementwise_mul(const float* a, const float* b, float* c, int n) {
#ifdef GRANITE_HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

}  // namespace granite
