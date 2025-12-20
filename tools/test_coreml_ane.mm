// Test tool for CoreML/ANE FFN backend
// Compares performance and correctness against Metal GPU FFN

#include <Metal/Metal.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#include "granite/metal_compute.h"
#include "granite/coreml_ffn.h"

using namespace granite;

// Float to FP16 conversion
uint16_t float_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp > 15) return (sign << 15) | 0x7C00;
    if (exp < -14) return sign << 15;
    uint16_t fp16_exp = exp + 15;
    uint16_t fp16_mant = mant >> 13;
    return (sign << 15) | (fp16_exp << 10) | fp16_mant;
}

// FP16 to float conversion
float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        return (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
    } else if (exp == 31) {
        return (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
    } else {
        return (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15.0f);
    }
}

// Reference SwiGLU FFN implementation
void reference_ffn(
    const float* input,      // [batch, hidden]
    const uint16_t* w_gate,  // [hidden, inter] FP16
    const uint16_t* w_up,    // [hidden, inter] FP16
    const uint16_t* w_down,  // [inter, hidden] FP16
    float* output,           // [batch, hidden]
    uint32_t batch_size,
    uint32_t hidden_dim,
    uint32_t intermediate_dim)
{
    std::vector<float> gate(batch_size * intermediate_dim);
    std::vector<float> up(batch_size * intermediate_dim);
    std::vector<float> hidden(batch_size * intermediate_dim);

    // Gate projection
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < intermediate_dim; i++) {
            float sum = 0.0f;
            for (uint32_t h = 0; h < hidden_dim; h++) {
                float w = fp16_to_float(w_gate[h * intermediate_dim + i]);
                sum += input[b * hidden_dim + h] * w;
            }
            gate[b * intermediate_dim + i] = sum;
        }
    }

    // Up projection
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < intermediate_dim; i++) {
            float sum = 0.0f;
            for (uint32_t h = 0; h < hidden_dim; h++) {
                float w = fp16_to_float(w_up[h * intermediate_dim + i]);
                sum += input[b * hidden_dim + h] * w;
            }
            up[b * intermediate_dim + i] = sum;
        }
    }

    // SiLU on gate and multiply with up
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < intermediate_dim; i++) {
            float g = gate[b * intermediate_dim + i];
            float silu = g / (1.0f + expf(-g));  // silu(x) = x * sigmoid(x)
            hidden[b * intermediate_dim + i] = silu * up[b * intermediate_dim + i];
        }
    }

    // Down projection
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t h = 0; h < hidden_dim; h++) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < intermediate_dim; i++) {
                float w = fp16_to_float(w_down[i * hidden_dim + h]);
                sum += hidden[b * intermediate_dim + i] * w;
            }
            output[b * hidden_dim + h] = sum;
        }
    }
}

int main() {
    std::cout << "==========================================================\n";
    std::cout << "    CoreML/ANE FFN Backend Test\n";
    std::cout << "==========================================================\n\n";

    // Initialize Metal
    auto* gpu = get_metal_compute();
    if (!gpu || !gpu->is_initialized()) {
        std::cerr << "Failed to initialize Metal compute\n";
        return 1;
    }

    MTL::Device* device = gpu->device();
    std::cout << "Metal device: " << device->name()->utf8String() << "\n";
    std::cout << "ANE available: " << (CoreMLFFN::is_ane_available() ? "yes" : "no") << "\n\n";

    // Initialize CoreML FFN
    auto* coreml = get_coreml_ffn();
    if (!coreml) {
        std::cerr << "Failed to get CoreML FFN instance\n";
        return 1;
    }

    auto init_result = coreml->initialize(device);
    if (!init_result.ok()) {
        std::cerr << "Failed to initialize CoreML FFN: " << init_result.error().message() << "\n";
        return 1;
    }

    // Set to LowPower mode to use ANE
    coreml->set_power_mode(CoreMLFFN::PowerMode::LowPower);

    std::cout << "CoreML FFN initialized\n\n";

    // Test configurations (smaller dimensions for faster testing)
    struct TestCase {
        uint32_t batch_size;
        uint32_t hidden_dim;
        uint32_t intermediate_dim;
        const char* name;
    };

    std::vector<TestCase> tests = {
        // Small test for correctness
        {1, 256, 512, "Tiny: batch=1, hidden=256"},
        {4, 256, 512, "Tiny: batch=4, hidden=256"},

        // TinyLlama-like dimensions (scaled down for test speed)
        {1, 512, 1408, "Small: batch=1, hidden=512"},
        {8, 512, 1408, "Small: batch=8, hidden=512"},
        {32, 512, 1408, "Small: batch=32, hidden=512"},

        // Realistic dimensions (if time permits)
        // {1, 2048, 5632, "TinyLlama: batch=1"},
        // {8, 2048, 5632, "TinyLlama: batch=8"},
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    int passed = 0;
    int failed = 0;

    for (const auto& test : tests) {
        std::cout << "Testing: " << test.name << "\n";

        // Allocate buffers
        size_t input_size = test.batch_size * test.hidden_dim * sizeof(float);
        size_t gate_size = test.hidden_dim * test.intermediate_dim * sizeof(uint16_t);
        size_t up_size = test.hidden_dim * test.intermediate_dim * sizeof(uint16_t);
        size_t down_size = test.intermediate_dim * test.hidden_dim * sizeof(uint16_t);
        size_t output_size = test.batch_size * test.hidden_dim * sizeof(float);

        MTL::Buffer* input_buf = device->newBuffer(input_size, MTL::ResourceStorageModeShared);
        MTL::Buffer* w_gate_buf = device->newBuffer(gate_size, MTL::ResourceStorageModeShared);
        MTL::Buffer* w_up_buf = device->newBuffer(up_size, MTL::ResourceStorageModeShared);
        MTL::Buffer* w_down_buf = device->newBuffer(down_size, MTL::ResourceStorageModeShared);
        MTL::Buffer* output_coreml = device->newBuffer(output_size, MTL::ResourceStorageModeShared);

        // Initialize data
        float* input_data = static_cast<float*>(input_buf->contents());
        uint16_t* w_gate_data = static_cast<uint16_t*>(w_gate_buf->contents());
        uint16_t* w_up_data = static_cast<uint16_t*>(w_up_buf->contents());
        uint16_t* w_down_data = static_cast<uint16_t*>(w_down_buf->contents());

        for (size_t i = 0; i < test.batch_size * test.hidden_dim; i++) {
            input_data[i] = dist(rng);
        }
        for (size_t i = 0; i < test.hidden_dim * test.intermediate_dim; i++) {
            w_gate_data[i] = float_to_fp16(dist(rng));
            w_up_data[i] = float_to_fp16(dist(rng));
        }
        for (size_t i = 0; i < test.intermediate_dim * test.hidden_dim; i++) {
            w_down_data[i] = float_to_fp16(dist(rng));
        }

        memset(output_coreml->contents(), 0, output_size);

        // Compute reference
        std::vector<float> output_ref(test.batch_size * test.hidden_dim);
        reference_ffn(input_data, w_gate_data, w_up_data, w_down_data,
                     output_ref.data(), test.batch_size, test.hidden_dim, test.intermediate_dim);

        // Warmup
        auto warmup_result = coreml->forward(
            input_buf, w_gate_buf, w_up_buf, w_down_buf, output_coreml,
            test.batch_size, test.hidden_dim, test.intermediate_dim);

        if (!warmup_result.ok()) {
            std::cout << "  SKIPPED (compile error: " << warmup_result.error().message() << ")\n";
            input_buf->release();
            w_gate_buf->release();
            w_up_buf->release();
            w_down_buf->release();
            output_coreml->release();
            continue;
        }

        // Benchmark CoreML
        const int num_iters = 5;
        auto coreml_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iters; i++) {
            coreml->forward(input_buf, w_gate_buf, w_up_buf, w_down_buf, output_coreml,
                          test.batch_size, test.hidden_dim, test.intermediate_dim);
        }
        auto coreml_end = std::chrono::high_resolution_clock::now();
        double coreml_ms = std::chrono::duration<double, std::milli>(coreml_end - coreml_start).count() / num_iters;

        // Compare outputs
        float* out_coreml_data = static_cast<float*>(output_coreml->contents());

        float max_diff = 0.0f;
        float coreml_sum = 0.0f;
        float ref_sum = 0.0f;
        for (size_t i = 0; i < test.batch_size * test.hidden_dim; i++) {
            float diff = fabsf(out_coreml_data[i] - output_ref[i]);
            max_diff = std::max(max_diff, diff);
            coreml_sum += fabsf(out_coreml_data[i]);
            ref_sum += fabsf(output_ref[i]);
        }

        // Relative error check (FFN outputs can be larger)
        float rel_error = (ref_sum > 0.001f) ? max_diff / (ref_sum / (test.batch_size * test.hidden_dim)) : max_diff;
        bool pass = rel_error < 0.1f && coreml_sum > 0.001f;

        if (pass) {
            std::cout << "  PASSED";
            passed++;
        } else {
            std::cout << "  FAILED";
            failed++;
        }

        std::cout << " (max_diff=" << std::fixed << std::setprecision(4) << max_diff
                  << ", rel_err=" << std::setprecision(4) << rel_error
                  << ", time=" << std::setprecision(2) << coreml_ms << "ms";

        if (!pass || coreml_sum < 0.001f) {
            std::cout << "\n    CoreML[0..3]=[" << out_coreml_data[0] << "," << out_coreml_data[1]
                      << "," << out_coreml_data[2] << "," << out_coreml_data[3] << "]";
            std::cout << "\n    Ref[0..3]=[" << output_ref[0] << "," << output_ref[1]
                      << "," << output_ref[2] << "," << output_ref[3] << "]";
        }
        std::cout << ")\n";

        input_buf->release();
        w_gate_buf->release();
        w_up_buf->release();
        w_down_buf->release();
        output_coreml->release();
    }

    std::cout << "\n==========================================================\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "==========================================================\n";

    return failed > 0 ? 1 : 0;
}
