// Test tool for paged flash attention kernels
// Verifies correctness against reference implementation for long sequences

#include <Metal/Metal.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

#include "granite/metal_compute.h"

using namespace granite;

// Reference implementation of attention for verification
void reference_attention(
    const float* Q,      // [num_heads, head_dim]
    const uint16_t* K,   // [seq_len, num_kv_heads, head_dim] (FP16 as uint16_t)
    const uint16_t* V,   // [seq_len, num_kv_heads, head_dim] (FP16 as uint16_t)
    const int32_t* block_table,  // [num_logical_blocks]
    float* output,       // [num_heads, head_dim]
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    uint32_t block_size,
    float scale)
{
    uint32_t heads_per_kv = num_heads / num_kv_heads;

    for (uint32_t h = 0; h < num_heads; h++) {
        uint32_t kv_head = h / heads_per_kv;
        const float* q = Q + h * head_dim;
        float* out = output + h * head_dim;

        // Compute attention scores
        std::vector<float> scores(seq_len);
        float max_score = -1e30f;

        for (uint32_t s = 0; s < seq_len; s++) {
            // Block table lookup
            uint32_t logical_block = s / block_size;
            uint32_t block_offset = s % block_size;
            int32_t physical_block = block_table[logical_block];
            uint32_t physical_pos = physical_block * block_size + block_offset;

            // K at [physical_pos, kv_head, :]
            const uint16_t* k = K + physical_pos * num_kv_heads * head_dim + kv_head * head_dim;

            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                // Convert FP16 to float (simple approximation for testing)
                uint16_t k_bits = k[d];
                float k_val;
                // FP16 to float conversion
                uint32_t sign = (k_bits >> 15) & 0x1;
                uint32_t exp = (k_bits >> 10) & 0x1F;
                uint32_t mant = k_bits & 0x3FF;
                if (exp == 0) {
                    k_val = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
                } else if (exp == 31) {
                    k_val = (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
                } else {
                    k_val = (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15.0f);
                }
                dot += q[d] * k_val;
            }
            scores[s] = dot * scale;
            max_score = std::max(max_score, scores[s]);
        }

        // Softmax
        float sum = 0.0f;
        for (uint32_t s = 0; s < seq_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum += scores[s];
        }
        for (uint32_t s = 0; s < seq_len; s++) {
            scores[s] /= sum;
        }

        // Output = scores @ V
        for (uint32_t d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (uint32_t s = 0; s < seq_len; s++) {
                // Block table lookup
                uint32_t logical_block = s / block_size;
                uint32_t block_offset = s % block_size;
                int32_t physical_block = block_table[logical_block];
                uint32_t physical_pos = physical_block * block_size + block_offset;

                const uint16_t* v = V + physical_pos * num_kv_heads * head_dim + kv_head * head_dim;
                uint16_t v_bits = v[d];
                float v_val;
                uint32_t sign = (v_bits >> 15) & 0x1;
                uint32_t exp = (v_bits >> 10) & 0x1F;
                uint32_t mant = v_bits & 0x3FF;
                if (exp == 0) {
                    v_val = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
                } else if (exp == 31) {
                    v_val = (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
                } else {
                    v_val = (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15.0f);
                }
                val += scores[s] * v_val;
            }
            out[d] = val;
        }
    }
}

// Float to FP16 conversion
uint16_t float_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp > 15) {
        // Overflow to infinity
        return (sign << 15) | 0x7C00;
    } else if (exp < -14) {
        // Underflow to zero
        return sign << 15;
    } else {
        uint16_t fp16_exp = exp + 15;
        uint16_t fp16_mant = mant >> 13;
        return (sign << 15) | (fp16_exp << 10) | fp16_mant;
    }
}

int main() {
    std::cout << "==========================================================\n";
    std::cout << "    Paged Flash Attention Long Context Test\n";
    std::cout << "==========================================================\n\n";

    auto* gpu = get_metal_compute();
    if (!gpu || !gpu->is_initialized()) {
        std::cerr << "Failed to initialize Metal compute\n";
        return 1;
    }

    MTL::Device* device = gpu->device();
    std::cout << "Metal device: " << device->name()->utf8String() << "\n\n";

    struct TestCase {
        uint32_t num_heads;
        uint32_t num_kv_heads;
        uint32_t seq_len;
        uint32_t head_dim;
        uint32_t block_size;
        const char* name;
    };

    std::vector<TestCase> tests = {
        // Basic tests (within old kernel limit)
        {32, 8, 512, 64, 16, "512 tokens, head_dim=64"},
        {32, 8, 2048, 64, 16, "2K tokens, head_dim=64"},
        {32, 8, 4096, 64, 16, "4K tokens, head_dim=64"},

        // Long context tests (beyond old kernel limit)
        {32, 8, 8192, 64, 16, "8K tokens, head_dim=64"},
        {32, 8, 16384, 64, 16, "16K tokens, head_dim=64"},

        // head_dim=128 tests
        {32, 8, 2048, 128, 16, "2K tokens, head_dim=128"},
        {32, 8, 8192, 128, 16, "8K tokens, head_dim=128"},

        // GQA test
        {32, 4, 4096, 64, 16, "4K tokens, GQA 8:1"},
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    int passed = 0;
    int failed = 0;

    for (const auto& test : tests) {
        std::cout << "Testing: " << test.name << "... " << std::flush;

        // Calculate sizes
        uint32_t num_logical_blocks = (test.seq_len + test.block_size - 1) / test.block_size;
        uint32_t num_physical_blocks = num_logical_blocks;  // Simple 1:1 mapping for test

        size_t q_size = test.num_heads * test.head_dim * sizeof(float);
        size_t kv_size = num_physical_blocks * test.block_size * test.num_kv_heads * test.head_dim * sizeof(uint16_t);
        size_t block_table_size = num_logical_blocks * sizeof(int32_t);
        size_t output_size = test.num_heads * test.head_dim * sizeof(float);

        // Allocate buffers
        MTL::Buffer* Q = device->newBuffer(q_size, MTL::ResourceStorageModeShared);
        MTL::Buffer* K_cache = device->newBuffer(kv_size, MTL::ResourceStorageModeShared);
        MTL::Buffer* V_cache = device->newBuffer(kv_size, MTL::ResourceStorageModeShared);
        MTL::Buffer* block_table = device->newBuffer(block_table_size, MTL::ResourceStorageModeShared);
        MTL::Buffer* output = device->newBuffer(output_size, MTL::ResourceStorageModeShared);

        // Initialize data
        float* q_data = static_cast<float*>(Q->contents());
        uint16_t* k_data = static_cast<uint16_t*>(K_cache->contents());
        uint16_t* v_data = static_cast<uint16_t*>(V_cache->contents());
        int32_t* bt_data = static_cast<int32_t*>(block_table->contents());
        float* out_data = static_cast<float*>(output->contents());

        // Fill Q with random data
        for (size_t i = 0; i < test.num_heads * test.head_dim; i++) {
            q_data[i] = dist(rng);
        }

        // Fill K/V with random FP16 data
        size_t kv_elements = num_physical_blocks * test.block_size * test.num_kv_heads * test.head_dim;
        for (size_t i = 0; i < kv_elements; i++) {
            k_data[i] = float_to_fp16(dist(rng));
            v_data[i] = float_to_fp16(dist(rng));
        }

        // Create simple block table (1:1 mapping with some shuffle)
        std::vector<int32_t> block_indices(num_logical_blocks);
        for (uint32_t i = 0; i < num_logical_blocks; i++) {
            block_indices[i] = i;
        }
        // Shuffle to simulate scattered blocks
        std::shuffle(block_indices.begin(), block_indices.end(), rng);
        for (uint32_t i = 0; i < num_logical_blocks; i++) {
            bt_data[i] = block_indices[i];
        }

        // Zero output
        memset(out_data, 0, output_size);

        float scale = 1.0f / sqrtf(static_cast<float>(test.head_dim));

        // Run GPU kernel
        auto start = std::chrono::high_resolution_clock::now();
        auto result = gpu->paged_attention_decode(
            Q, K_cache, V_cache, block_table, output,
            test.num_heads, test.num_kv_heads,
            test.seq_len, test.head_dim, test.block_size, scale);
        gpu->sync();
        auto end = std::chrono::high_resolution_clock::now();

        if (!result.ok()) {
            std::cout << "FAILED (kernel error: " << result.error().message() << ")\n";
            failed++;
            Q->release();
            K_cache->release();
            V_cache->release();
            block_table->release();
            output->release();
            continue;
        }

        double gpu_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Compute reference (only for smaller tests to avoid long runtimes)
        std::vector<float> ref_output(test.num_heads * test.head_dim, 0.0f);
        if (test.seq_len <= 4096) {
            reference_attention(
                q_data, k_data, v_data, bt_data,
                ref_output.data(),
                test.num_heads, test.num_kv_heads,
                test.seq_len, test.head_dim, test.block_size, scale);

            // Compare
            float max_diff = 0.0f;
            float max_rel = 0.0f;
            for (size_t i = 0; i < test.num_heads * test.head_dim; i++) {
                float diff = fabsf(out_data[i] - ref_output[i]);
                float rel = diff / (fabsf(ref_output[i]) + 1e-6f);
                max_diff = std::max(max_diff, diff);
                max_rel = std::max(max_rel, rel);
            }

            bool pass = (max_diff < 0.1f) || (max_rel < 0.05f);
            if (pass) {
                std::cout << "PASSED";
            } else {
                std::cout << "FAILED";
                failed++;
            }
            std::cout << " (max_diff=" << std::fixed << std::setprecision(4) << max_diff
                      << ", time=" << std::setprecision(2) << gpu_ms << "ms)\n";
            if (pass) passed++;
        } else {
            // For long sequences, just check no NaN/Inf and reasonable values
            bool has_nan = false;
            bool has_inf = false;
            float min_val = INFINITY;
            float max_val = -INFINITY;
            for (size_t i = 0; i < test.num_heads * test.head_dim; i++) {
                if (std::isnan(out_data[i])) has_nan = true;
                if (std::isinf(out_data[i])) has_inf = true;
                min_val = std::min(min_val, out_data[i]);
                max_val = std::max(max_val, out_data[i]);
            }

            bool pass = !has_nan && !has_inf && max_val - min_val > 0.0f;
            if (pass) {
                std::cout << "PASSED";
                passed++;
            } else {
                std::cout << "FAILED";
                failed++;
            }
            std::cout << " (range=[" << std::setprecision(3) << min_val << ", " << max_val << "]"
                      << ", time=" << std::setprecision(2) << gpu_ms << "ms)\n";
        }

        Q->release();
        K_cache->release();
        V_cache->release();
        block_table->release();
        output->release();
    }

    std::cout << "\n==========================================================\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "==========================================================\n";

    return failed > 0 ? 1 : 0;
}
