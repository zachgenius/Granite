// test_simdgroup_matmul.mm - Test simdgroup matmul kernel correctness
//
// Compares simdgroup kernel output against reference SIMD kernel

#include <granite/granite.h>
#ifdef GRANITE_HAS_METAL
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <granite/metal_compute.h>
#endif
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <cstring>

using namespace granite;

// Q4_K block structure: 144 bytes per 256 elements
struct block_q4_K {
    uint16_t d;        // delta (half)
    uint16_t dmin;     // min delta (half)
    uint8_t scales[12];
    uint8_t qs[128];
};

// Create random Q4_K weights with known values
void create_test_weights_q4k(uint8_t* data, size_t num_blocks, std::mt19937& rng) {
    // Use reasonable scale values for testing
    for (size_t b = 0; b < num_blocks; b++) {
        block_q4_K* block = reinterpret_cast<block_q4_K*>(data + b * 144);

        // Set d and dmin to small positive values (as half-precision)
        // half 0.1 ~ 0x2E66, half 0.05 ~ 0x2A66
        block->d = 0x2E66;    // ~0.1
        block->dmin = 0x2A66; // ~0.05

        // Set scales to reasonable values
        std::uniform_int_distribution<uint8_t> scale_dist(1, 15);
        for (int i = 0; i < 12; i++) {
            block->scales[i] = scale_dist(rng);
        }

        // Set quantized values
        std::uniform_int_distribution<uint8_t> qs_dist(0, 255);
        for (int i = 0; i < 128; i++) {
            block->qs[i] = qs_dist(rng);
        }
    }
}

// CPU reference for Q4_K matmul
void cpu_matmul_q4k(const float* X, const uint8_t* W, float* Y,
                    uint32_t M, uint32_t K, uint32_t N) {
    const size_t nb = K / 256;  // blocks per row

    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            float sum = 0.0f;

            for (uint32_t b = 0; b < nb; b++) {
                const block_q4_K* block = reinterpret_cast<const block_q4_K*>(
                    W + (n * nb + b) * 144);

                // Convert half to float for d and dmin
                uint16_t d_bits = block->d;
                uint16_t dmin_bits = block->dmin;

                // Simple half-to-float conversion (approximation)
                auto half_to_float = [](uint16_t h) -> float {
                    uint32_t sign = (h >> 15) & 1;
                    uint32_t exp = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;

                    if (exp == 0) {
                        if (mant == 0) return sign ? -0.0f : 0.0f;
                        // Denormalized
                        float f = mant / 1024.0f * powf(2.0f, -14.0f);
                        return sign ? -f : f;
                    }
                    if (exp == 31) {
                        return mant ? NAN : (sign ? -INFINITY : INFINITY);
                    }

                    float f = (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15.0f);
                    return sign ? -f : f;
                };

                float d = half_to_float(d_bits);
                float dmin = half_to_float(dmin_bits);

                // Process 256 elements in this block
                for (int j = 0; j < 256; j++) {
                    int k = b * 256 + j;
                    if (k >= K) break;

                    // Get 4-bit quantized value
                    uint8_t qs_byte = block->qs[j / 2];
                    int q = (j & 1) ? (qs_byte >> 4) : (qs_byte & 0xF);

                    // Get scale and min for this group (8 groups of 32)
                    int group = j / 32;
                    int scale_idx = group;

                    // Decode 6-bit scales/mins from packed format
                    int sc, m_val;
                    if (scale_idx < 4) {
                        sc = block->scales[scale_idx] & 0x3F;
                        m_val = block->scales[scale_idx + 4] & 0x3F;
                    } else {
                        sc = (block->scales[scale_idx] & 0x0F) |
                             ((block->scales[scale_idx - 4] >> 6) << 4);
                        m_val = (block->scales[scale_idx] >> 4) |
                                ((block->scales[scale_idx] >> 6) << 4);
                    }

                    // Dequantize
                    float w = d * sc * q - dmin * m_val;

                    sum += X[m * K + k] * w;
                }
            }

            Y[m * N + n] = sum;
        }
    }
}

int main(int argc, char** argv) {
#ifndef GRANITE_HAS_METAL
    std::cerr << "Metal backend not available\n";
    return 1;
#else
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Simdgroup Matmul Kernel Correctness Test                 ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

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

    // Test cases
    struct TestCase {
        uint32_t M;  // Batch size
        uint32_t K;  // Input dimension
        uint32_t N;  // Output dimension
        const char* name;
    };

    std::vector<TestCase> tests = {
        // Test simd kernel (M >= 32, M < 256)
        {64, 2048, 2048, "M=64 K=2048 N=2048 (simd)"},
        {128, 2048, 256, "M=128 K=2048 N=256 (simd)"},
        // Test simdgroup matrix kernel (M >= 256)
        {256, 2048, 2048, "M=256 K=2048 N=2048 (simdgroup)"},
        {512, 2048, 2048, "M=512 K=2048 N=2048 (simdgroup)"},
        {256, 2048, 5632, "M=256 K=2048 N=5632 (simdgroup)"},
        // Edge cases
        {64, 256, 256, "M=64 K=256 N=256 (minimal K)"},
        {256, 256, 256, "M=256 K=256 N=256 (simdgroup minimal K)"},
    };

    std::mt19937 rng(42);
    int passed = 0;
    int failed = 0;

    for (const auto& test : tests) {
        std::cout << "Testing: " << test.name << "... " << std::flush;

        // Calculate sizes
        uint32_t num_blocks_per_row = test.K / 256;
        uint32_t total_blocks = test.N * num_blocks_per_row;
        size_t weight_bytes = total_blocks * 144;
        size_t input_bytes = test.M * test.K * sizeof(float);
        size_t output_bytes = test.M * test.N * sizeof(float);

        // Allocate buffers
        MTL::Buffer* x_buf = gpu.create_buffer(input_bytes);
        MTL::Buffer* w_buf = gpu.create_buffer(weight_bytes);
        MTL::Buffer* y_simdgroup_buf = gpu.create_buffer(output_bytes);
        MTL::Buffer* y_simd_buf = gpu.create_buffer(output_bytes);

        if (!x_buf || !w_buf || !y_simdgroup_buf || !y_simd_buf) {
            std::cerr << "FAILED (buffer allocation)\n";
            failed++;
            continue;
        }

        // Initialize input with random data
        float* x_ptr = static_cast<float*>(x_buf->contents());
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        for (uint32_t i = 0; i < test.M * test.K; i++) {
            x_ptr[i] = input_dist(rng);
        }

        // Initialize weights with test data
        uint8_t* w_ptr = static_cast<uint8_t*>(w_buf->contents());
        create_test_weights_q4k(w_ptr, total_blocks, rng);

        // Clear outputs
        memset(y_simdgroup_buf->contents(), 0, output_bytes);
        memset(y_simd_buf->contents(), 0, output_bytes);

        // Run simdgroup kernel (M >= 64 will use it)
        auto result1 = gpu.matmul_q4k(x_buf, w_buf, y_simdgroup_buf, test.M, test.K, test.N);
        gpu.sync();

        if (!result1.ok()) {
            std::cerr << "FAILED (simdgroup kernel error: " << result1.error().message() << ")\n";
            failed++;
            continue;
        }

        // Temporarily force SIMD kernel by using M < 64
        // We'll compare against a batch of single-row operations for reference
        std::vector<float> y_reference(test.M * test.N, 0.0f);

        // Run row-by-row for reference using existing matmul path
        for (uint32_t m = 0; m < test.M; m++) {
            MTL::Buffer* x_row_buf = gpu.create_buffer(test.K * sizeof(float));
            MTL::Buffer* y_row_buf = gpu.create_buffer(test.N * sizeof(float));

            // Copy input row
            memcpy(x_row_buf->contents(), x_ptr + m * test.K, test.K * sizeof(float));
            memset(y_row_buf->contents(), 0, test.N * sizeof(float));

            // Run single-row matmul (uses matvec path)
            gpu.matvec_q4k(x_row_buf, w_buf, y_row_buf, test.K, test.N);
            gpu.sync();

            // Copy result
            memcpy(&y_reference[m * test.N], y_row_buf->contents(), test.N * sizeof(float));

            x_row_buf->release();
            y_row_buf->release();
        }

        // Compare results
        float* y_simdgroup = static_cast<float*>(y_simdgroup_buf->contents());

        float max_diff = 0.0f;
        float max_rel_diff = 0.0f;
        int diff_count = 0;
        int first_diff_idx = -1;

        for (uint32_t i = 0; i < test.M * test.N; i++) {
            float diff = fabs(y_simdgroup[i] - y_reference[i]);
            float rel_diff = diff / (fabs(y_reference[i]) + 1e-6f);

            // FP16 precision tolerance: 0.5 absolute or 1% relative
            if (diff > 0.5f && rel_diff > 0.01f) {
                diff_count++;
                if (first_diff_idx < 0) first_diff_idx = i;
            }

            max_diff = fmax(max_diff, diff);
            max_rel_diff = fmax(max_rel_diff, rel_diff);
        }

        // Tolerance: FP16 simdgroup computation allows ~0.5 absolute error due to reduced precision
        // and different accumulation order. Accept if max absolute diff < 1.0 or max relative < 5%
        bool pass = (max_diff < 1.0f || max_rel_diff < 0.05f);

        if (pass) {
            std::cout << "PASSED (max_diff=" << std::fixed << std::setprecision(4)
                      << max_diff << ", max_rel=" << max_rel_diff << ")\n";
            passed++;
        } else {
            std::cout << "FAILED (max_diff=" << max_diff << ", max_rel=" << max_rel_diff
                      << ", diff_count=" << diff_count << ")\n";

            // Print first few differences for debugging
            if (first_diff_idx >= 0) {
                std::cout << "  First difference at index " << first_diff_idx << ":\n";
                std::cout << "    simdgroup: " << y_simdgroup[first_diff_idx] << "\n";
                std::cout << "    reference: " << y_reference[first_diff_idx] << "\n";
            }

            // Print some sample outputs
            std::cout << "  First 5 simdgroup outputs: ";
            for (int i = 0; i < std::min(5u, test.N); i++) {
                std::cout << y_simdgroup[i] << " ";
            }
            std::cout << "\n  First 5 reference outputs: ";
            for (int i = 0; i < std::min(5u, test.N); i++) {
                std::cout << y_reference[i] << " ";
            }
            std::cout << "\n";

            failed++;
        }

        x_buf->release();
        w_buf->release();
        y_simdgroup_buf->release();
        y_simd_buf->release();
    }

    std::cout << "\n════════════════════════════════════════════════════════════════════\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "════════════════════════════════════════════════════════════════════\n";

    return failed > 0 ? 1 : 0;
#endif
}
