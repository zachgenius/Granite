// Test tool for CoreML/ANE attention backend
// Compares performance and correctness against Metal Flash Attention

#include <Metal/Metal.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#include "granite/metal_compute.h"
#include "granite/coreml_attention.h"

using namespace granite;

// Reference softmax for verification
void reference_softmax(const float* input, float* output, uint32_t n) {
    float max_val = input[0];
    for (uint32_t i = 1; i < n; i++) {
        max_val = std::max(max_val, input[i]);
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (uint32_t i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

// Reference attention for verification
void reference_attention(
    const float* Q,      // [num_heads, seq_q, head_dim]
    const uint16_t* K,   // [num_kv_heads, seq_kv, head_dim] (FP16)
    const uint16_t* V,   // [num_kv_heads, seq_kv, head_dim] (FP16)
    float* output,       // [num_heads, seq_q, head_dim]
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t seq_q,
    uint32_t seq_kv,
    uint32_t head_dim,
    float scale)
{
    uint32_t heads_per_kv = num_heads / num_kv_heads;

    for (uint32_t h = 0; h < num_heads; h++) {
        uint32_t kv_h = h / heads_per_kv;

        for (uint32_t sq = 0; sq < seq_q; sq++) {
            const float* q = Q + h * seq_q * head_dim + sq * head_dim;
            float* out = output + h * seq_q * head_dim + sq * head_dim;

            // Compute attention scores
            std::vector<float> scores(seq_kv);
            for (uint32_t sk = 0; sk < seq_kv; sk++) {
                const uint16_t* k = K + kv_h * seq_kv * head_dim + sk * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    // FP16 to float conversion
                    uint16_t k_bits = k[d];
                    uint32_t sign = (k_bits >> 15) & 0x1;
                    uint32_t exp = (k_bits >> 10) & 0x1F;
                    uint32_t mant = k_bits & 0x3FF;
                    float k_val;
                    if (exp == 0) {
                        k_val = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
                    } else if (exp == 31) {
                        k_val = (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
                    } else {
                        k_val = (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15.0f);
                    }
                    dot += q[d] * k_val;
                }
                scores[sk] = dot * scale;
            }

            // Softmax
            reference_softmax(scores.data(), scores.data(), seq_kv);

            // Output = scores @ V
            for (uint32_t d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (uint32_t sk = 0; sk < seq_kv; sk++) {
                    const uint16_t* v = V + kv_h * seq_kv * head_dim + sk * head_dim;
                    uint16_t v_bits = v[d];
                    uint32_t sign = (v_bits >> 15) & 0x1;
                    uint32_t exp = (v_bits >> 10) & 0x1F;
                    uint32_t mant = v_bits & 0x3FF;
                    float v_val;
                    if (exp == 0) {
                        v_val = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
                    } else if (exp == 31) {
                        v_val = (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
                    } else {
                        v_val = (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15.0f);
                    }
                    val += scores[sk] * v_val;
                }
                out[d] = val;
            }
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

    if (exp > 15) return (sign << 15) | 0x7C00;  // Infinity
    if (exp < -14) return sign << 15;  // Zero
    uint16_t fp16_exp = exp + 15;
    uint16_t fp16_mant = mant >> 13;
    return (sign << 15) | (fp16_exp << 10) | fp16_mant;
}

int main() {
    std::cout << "==========================================================\n";
    std::cout << "    CoreML/ANE Attention Backend Test\n";
    std::cout << "==========================================================\n\n";

    // Initialize Metal
    auto* gpu = get_metal_compute();
    if (!gpu || !gpu->is_initialized()) {
        std::cerr << "Failed to initialize Metal compute\n";
        return 1;
    }

    MTL::Device* device = gpu->device();
    std::cout << "Metal device: " << device->name()->utf8String() << "\n";

    // Check ANE availability
    std::cout << "ANE available: " << (CoreMLAttention::is_ane_available() ? "yes" : "no") << "\n\n";

    // Initialize CoreML
    auto* coreml = get_coreml_attention();
    if (!coreml) {
        std::cerr << "Failed to get CoreML attention instance\n";
        return 1;
    }

    auto init_result = coreml->initialize(device);
    if (!init_result.ok()) {
        std::cerr << "Failed to initialize CoreML: " << init_result.error().message() << "\n";
        return 1;
    }

    std::cout << "CoreML attention initialized\n\n";

    // Test configurations
    struct TestCase {
        uint32_t num_heads;
        uint32_t num_kv_heads;
        uint32_t seq_q;
        uint32_t seq_kv;
        uint32_t head_dim;
        const char* name;
    };

    std::vector<TestCase> tests = {
        // Decode (seq_q=1) - the main use case for CoreML/ANE
        {32, 8, 1, 128, 64, "Decode: 128 tokens, head_dim=64"},
        {32, 8, 1, 512, 64, "Decode: 512 tokens, head_dim=64"},
        {32, 8, 1, 1024, 64, "Decode: 1024 tokens, head_dim=64"},
        {32, 8, 1, 2048, 64, "Decode: 2048 tokens, head_dim=64"},
        {32, 8, 1, 4096, 64, "Decode: 4096 tokens, head_dim=64"},

        // head_dim=128
        {32, 8, 1, 512, 128, "Decode: 512 tokens, head_dim=128"},
        {32, 8, 1, 1024, 128, "Decode: 1024 tokens, head_dim=128"},
        {32, 8, 1, 2048, 128, "Decode: 2048 tokens, head_dim=128"},

        // GQA configurations
        {32, 4, 1, 512, 64, "Decode: GQA 8:1, 512 tokens"},
        {32, 2, 1, 512, 64, "Decode: GQA 16:1, 512 tokens"},

        // Note: Prefill (seq_q > 1) should use Metal kernels which are 300-600x faster
        // CoreML/ANE is primarily for power-efficient decode on iOS devices
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    int passed = 0;
    int failed = 0;

    for (const auto& test : tests) {
        std::cout << "Testing: " << test.name << "\n";

        // Allocate buffers
        size_t q_size = test.num_heads * test.seq_q * test.head_dim * sizeof(float);
        size_t kv_size = test.num_kv_heads * test.seq_kv * test.head_dim * sizeof(uint16_t);
        size_t out_size = test.num_heads * test.seq_q * test.head_dim * sizeof(float);

        MTL::Buffer* Q = device->newBuffer(q_size, MTL::ResourceStorageModeShared);
        MTL::Buffer* K = device->newBuffer(kv_size, MTL::ResourceStorageModeShared);
        MTL::Buffer* V = device->newBuffer(kv_size, MTL::ResourceStorageModeShared);
        MTL::Buffer* output_coreml = device->newBuffer(out_size, MTL::ResourceStorageModeShared);
        MTL::Buffer* output_metal = device->newBuffer(out_size, MTL::ResourceStorageModeShared);

        // Initialize data
        float* q_data = static_cast<float*>(Q->contents());
        uint16_t* k_data = static_cast<uint16_t*>(K->contents());
        uint16_t* v_data = static_cast<uint16_t*>(V->contents());

        for (size_t i = 0; i < test.num_heads * test.seq_q * test.head_dim; i++) {
            q_data[i] = dist(rng);
        }
        for (size_t i = 0; i < test.num_kv_heads * test.seq_kv * test.head_dim; i++) {
            k_data[i] = float_to_fp16(dist(rng));
            v_data[i] = float_to_fp16(dist(rng));
        }

        memset(output_coreml->contents(), 0, out_size);
        memset(output_metal->contents(), 0, out_size);

        float scale = 1.0f / sqrtf(static_cast<float>(test.head_dim));

        // Warmup
        coreml->multihead_attention(Q, K, V, output_coreml,
            test.num_heads, test.num_kv_heads, test.seq_q, test.seq_kv, test.head_dim, scale);
        gpu->multihead_attention(Q, K, V, output_metal,
            test.num_heads, test.num_kv_heads, test.seq_q, test.seq_kv, test.head_dim, scale);
        gpu->sync();

        // Benchmark CoreML
        const int num_iters = 10;
        auto coreml_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iters; i++) {
            coreml->multihead_attention(Q, K, V, output_coreml,
                test.num_heads, test.num_kv_heads, test.seq_q, test.seq_kv, test.head_dim, scale);
        }
        auto coreml_end = std::chrono::high_resolution_clock::now();
        double coreml_ms = std::chrono::duration<double, std::milli>(coreml_end - coreml_start).count() / num_iters;

        // Benchmark Metal
        auto metal_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iters; i++) {
            gpu->multihead_attention(Q, K, V, output_metal,
                test.num_heads, test.num_kv_heads, test.seq_q, test.seq_kv, test.head_dim, scale);
            gpu->sync();
        }
        auto metal_end = std::chrono::high_resolution_clock::now();
        double metal_ms = std::chrono::duration<double, std::milli>(metal_end - metal_start).count() / num_iters;

        // Compare outputs
        float* out_coreml_data = static_cast<float*>(output_coreml->contents());
        float* out_metal_data = static_cast<float*>(output_metal->contents());

        float max_diff = 0.0f;
        for (size_t i = 0; i < test.num_heads * test.seq_q * test.head_dim; i++) {
            float diff = fabsf(out_coreml_data[i] - out_metal_data[i]);
            max_diff = std::max(max_diff, diff);
        }

        // Check if outputs are actually non-zero
        float coreml_sum = 0.0f, metal_sum = 0.0f;
        for (size_t i = 0; i < test.num_heads * test.seq_q * test.head_dim; i++) {
            coreml_sum += fabsf(out_coreml_data[i]);
            metal_sum += fabsf(out_metal_data[i]);
        }

        bool pass = max_diff < 0.1f;  // FP16 tolerance
        if (pass) {
            std::cout << "  PASSED";
            passed++;
        } else {
            std::cout << "  FAILED";
            failed++;
        }
        std::cout << " (max_diff=" << std::fixed << std::setprecision(4) << max_diff
                  << ", CoreML=" << std::setprecision(2) << coreml_ms << "ms"
                  << ", Metal=" << metal_ms << "ms"
                  << ", ratio=" << std::setprecision(2) << (coreml_ms / metal_ms) << "x";

        // Debug: show sample values for failures or suspicious results
        if (!pass || coreml_sum < 0.01f || metal_sum < 0.01f) {
            std::cout << "\n    CoreML[0..3]=[" << out_coreml_data[0] << "," << out_coreml_data[1]
                      << "," << out_coreml_data[2] << "," << out_coreml_data[3] << "]";
            std::cout << "\n    Metal[0..3]=[" << out_metal_data[0] << "," << out_metal_data[1]
                      << "," << out_metal_data[2] << "," << out_metal_data[3] << "]";
            std::cout << "\n    sums: coreml=" << coreml_sum << ", metal=" << metal_sum;
        }
        std::cout << ")\n";

        Q->release();
        K->release();
        V->release();
        output_coreml->release();
        output_metal->release();
    }

    std::cout << "\n==========================================================\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "==========================================================\n";

    return failed > 0 ? 1 : 0;
}
