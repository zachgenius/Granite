// test_vulkan.cpp - Vulkan backend tests
//
// These tests are only compiled and run when GRANITE_HAS_VULKAN is defined.
// They validate the Vulkan compute kernels against CPU reference implementations.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "granite/backend.h"
#include "granite/operators.h"
#include "granite/log.h"

#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <filesystem>

#ifdef GRANITE_HAS_VULKAN

using namespace granite;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// Helper to create random test data
static std::vector<float> random_vector(size_t size, float min = -1.0f, float max = 1.0f) {
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(min, max);
    std::vector<float> data(size);
    for (auto& v : data) {
        v = dist(gen);
    }
    return data;
}

// CPU reference implementations
namespace ref {

void silu(const float* x, float* out, size_t size) {
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void gelu(const float* x, float* out, size_t size) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    for (size_t i = 0; i < size; i++) {
        float xi = x[i];
        float x3 = xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + std::tanh(SQRT_2_OVER_PI * (xi + 0.044715f * x3)));
    }
}

void add(const float* a, const float* b, float* out, size_t size) {
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}

void mul(const float* a, const float* b, float* out, size_t size) {
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] * b[i];
    }
}

void rms_norm(const float* x, const float* w, float* out, size_t size, float eps) {
    // Compute RMS
    float sum_sq = 0.0f;
    for (size_t i = 0; i < size; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = std::sqrt(sum_sq / size + eps);
    float scale = 1.0f / rms;

    // Apply normalization and weight
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] * scale * w[i];
    }
}

}  // namespace ref

class VulkanTestFixture {
public:
    VulkanTestFixture() {
        backend_ = create_backend(BackendType::Vulkan);
        if (backend_) {
            auto result = backend_->initialize();
            initialized_ = result.ok();
            if (!initialized_) {
                GRANITE_LOG_WARN("Failed to initialize Vulkan backend: {}",
                               result.error().message());
            }
        }
    }

    ~VulkanTestFixture() {
        if (backend_) {
            backend_->shutdown();
        }
    }

    bool is_available() const { return initialized_; }
    IComputeBackend* backend() { return backend_.get(); }

private:
    std::unique_ptr<IComputeBackend> backend_;
    bool initialized_ = false;
};

TEST_CASE("Vulkan backend initialization", "[vulkan]") {
    VulkanTestFixture fixture;

    if (!fixture.is_available()) {
        SKIP("Vulkan backend not available");
    }

    REQUIRE(fixture.backend() != nullptr);
    REQUIRE(fixture.backend()->is_initialized());

    auto caps = fixture.backend()->get_capabilities();
    REQUIRE(caps.name != nullptr);
    REQUIRE(std::strlen(caps.name) > 0);

    INFO("Vulkan device: " << caps.name);
    INFO("Max buffer size: " << caps.max_buffer_size);
    INFO("Shared memory: " << caps.shared_memory_size);
}

TEST_CASE("Vulkan buffer operations", "[vulkan]") {
    VulkanTestFixture fixture;

    if (!fixture.is_available()) {
        SKIP("Vulkan backend not available");
    }

    auto* backend = fixture.backend();

    SECTION("Create and destroy buffer") {
        BufferDesc desc{};
        desc.size = 1024;
        desc.memory_type = MemoryType::Shared;

        auto result = backend->create_buffer(desc);
        REQUIRE(result.ok());

        BufferHandle handle = result.value();
        REQUIRE(handle.valid());

        backend->destroy_buffer(handle);
    }

    SECTION("Write and read buffer") {
        const size_t size = 256;
        std::vector<float> input(size);
        for (size_t i = 0; i < size; i++) {
            input[i] = static_cast<float>(i);
        }

        BufferDesc desc{};
        desc.size = size * sizeof(float);
        desc.memory_type = MemoryType::Shared;

        auto buffer_result = backend->create_buffer(desc);
        REQUIRE(buffer_result.ok());
        BufferHandle handle = buffer_result.value();

        // Write data
        auto write_result = backend->write_buffer(handle, input.data(),
                                                  size * sizeof(float), 0);
        REQUIRE(write_result.ok());

        // Read back
        std::vector<float> output(size);
        auto read_result = backend->read_buffer(handle, output.data(),
                                                size * sizeof(float), 0);
        REQUIRE(read_result.ok());

        // Verify
        for (size_t i = 0; i < size; i++) {
            REQUIRE(output[i] == input[i]);
        }

        backend->destroy_buffer(handle);
    }
}

// Note: The following tests require shaderc for runtime shader compilation.
// They will be skipped if shaderc is not available and no pre-compiled
// SPIR-V shaders are loaded.

TEST_CASE("Vulkan SiLU kernel", "[vulkan][compute]") {
    VulkanTestFixture fixture;

    if (!fixture.is_available()) {
        SKIP("Vulkan backend not available");
    }

    // This test requires runtime shader compilation or pre-compiled shaders
    // For now, we just verify the backend can be initialized
    // Full kernel tests require VulkanCompute integration

    INFO("SiLU kernel test - requires VulkanCompute integration");
}

TEST_CASE("Vulkan GELU kernel", "[vulkan][compute]") {
    VulkanTestFixture fixture;

    if (!fixture.is_available()) {
        SKIP("Vulkan backend not available");
    }

    INFO("GELU kernel test - requires VulkanCompute integration");
}

TEST_CASE("Vulkan RMSNorm kernel", "[vulkan][compute]") {
    VulkanTestFixture fixture;

    if (!fixture.is_available()) {
        SKIP("Vulkan backend not available");
    }

    INFO("RMSNorm kernel test - requires VulkanCompute integration");
}

TEST_CASE("Vulkan add kernel", "[vulkan][compute]") {
    VulkanTestFixture fixture;

    if (!fixture.is_available()) {
        SKIP("Vulkan backend not available");
    }

    INFO("Add kernel test - requires VulkanCompute integration");
}

TEST_CASE("Vulkan mul kernel", "[vulkan][compute]") {
    VulkanTestFixture fixture;

    if (!fixture.is_available()) {
        SKIP("Vulkan backend not available");
    }

    INFO("Mul kernel test - requires VulkanCompute integration");
}

TEST_CASE("Vulkan MatMul operator", "[vulkan][compute]") {
    VulkanTestFixture fixture;

    if (!fixture.is_available()) {
        SKIP("Vulkan backend not available");
    }

    initialize_operators();

    auto* backend = fixture.backend();

    std::vector<int64_t> shape_a = {2, 3};
    std::vector<int64_t> shape_b = {3, 4};

    auto a_result = Tensor::allocate(shape_a, DataType::FP32, backend);
    auto b_result = Tensor::allocate(shape_b, DataType::FP32, backend);
    REQUIRE(a_result.ok());
    REQUIRE(b_result.ok());

    auto a = std::move(a_result).take();
    auto b = std::move(b_result).take();

    // Fill inputs
    {
        auto map_a = backend->map_buffer(a.buffer());
        auto map_b = backend->map_buffer(b.buffer());
        REQUIRE(map_a.ok());
        REQUIRE(map_b.ok());

        float* pa = static_cast<float*>(map_a.value());
        float* pb = static_cast<float*>(map_b.value());

        for (size_t i = 0; i < a.numel(); i++) {
            pa[i] = static_cast<float>(i + 1);
        }
        for (size_t i = 0; i < b.numel(); i++) {
            pb[i] = static_cast<float>(i + 1);
        }

        backend->unmap_buffer(a.buffer());
        backend->unmap_buffer(b.buffer());
    }

    auto out_result = ops::matmul(a, b);
    REQUIRE(out_result.ok());
    auto out = std::move(out_result).take();

    auto map_out = backend->map_buffer(out.buffer());
    REQUIRE(map_out.ok());
    const float* pout = static_cast<const float*>(map_out.value());

    // Reference compute
    float expected[8] = {};
    const float a_vals[6] = {1, 2, 3, 4, 5, 6};
    const float b_vals[12] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += a_vals[i * 3 + k] * b_vals[k * 4 + j];
            }
            expected[i * 4 + j] = sum;
        }
    }

    for (int i = 0; i < 8; i++) {
        REQUIRE_THAT(pout[i], WithinAbs(expected[i], 1e-5f));
    }

    backend->unmap_buffer(out.buffer());
}

TEST_CASE("Vulkan attention operator", "[vulkan][compute]") {
    VulkanTestFixture fixture;

    if (!fixture.is_available()) {
        SKIP("Vulkan backend not available");
    }

    initialize_operators();

    auto* backend = fixture.backend();

    const int64_t seq_q = 2;
    const int64_t seq_kv = 3;
    const int64_t head_dim = 4;

    std::vector<int64_t> shape_q = {seq_q, head_dim};
    std::vector<int64_t> shape_k = {head_dim, seq_kv};  // transposed
    std::vector<int64_t> shape_v = {seq_kv, head_dim};

    auto q_result = Tensor::allocate(shape_q, DataType::FP32, backend);
    auto k_result = Tensor::allocate(shape_k, DataType::FP32, backend);
    auto v_result = Tensor::allocate(shape_v, DataType::FP32, backend);
    REQUIRE(q_result.ok());
    REQUIRE(k_result.ok());
    REQUIRE(v_result.ok());

    auto q = std::move(q_result).take();
    auto k = std::move(k_result).take();
    auto v = std::move(v_result).take();

    {
        auto map_q = backend->map_buffer(q.buffer());
        auto map_k = backend->map_buffer(k.buffer());
        auto map_v = backend->map_buffer(v.buffer());
        REQUIRE(map_q.ok());
        REQUIRE(map_k.ok());
        REQUIRE(map_v.ok());

        float* pq = static_cast<float*>(map_q.value());
        float* pk = static_cast<float*>(map_k.value());
        float* pv = static_cast<float*>(map_v.value());

        for (int i = 0; i < seq_q * head_dim; i++) {
            pq[i] = static_cast<float>(i + 1);
        }
        for (int i = 0; i < head_dim * seq_kv; i++) {
            pk[i] = static_cast<float>(i + 1) * 0.5f;
        }
        for (int i = 0; i < seq_kv * head_dim; i++) {
            pv[i] = static_cast<float>(i + 1) * 0.25f;
        }

        backend->unmap_buffer(q.buffer());
        backend->unmap_buffer(k.buffer());
        backend->unmap_buffer(v.buffer());
    }

    auto attn_op = OperatorRegistry::instance().create(
        OpType::ScaledDotProductAttention, BackendType::Vulkan);
    REQUIRE(attn_op != nullptr);

    OpContext ctx;
    ctx.backend = backend;
    ctx.inputs = {q, k, v};
    ctx.attrs.set("scale", 1.0 / std::sqrt(static_cast<double>(head_dim)));

    auto shapes_result = attn_op->infer_shapes(ctx);
    REQUIRE(shapes_result.ok());

    auto out_result = Tensor::allocate(shapes_result.value()[0], DataType::FP32, backend);
    REQUIRE(out_result.ok());
    ctx.outputs = {std::move(out_result).take()};

    auto exec_result = attn_op->execute(ctx);
    REQUIRE(exec_result.ok());

    auto map_out = backend->map_buffer(ctx.outputs[0].buffer());
    REQUIRE(map_out.ok());
    const float* pout = static_cast<const float*>(map_out.value());

    // CPU reference
    float scores[seq_q * seq_kv] = {};
    float probs[seq_q * seq_kv] = {};
    float expected[seq_q * head_dim] = {};

    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    for (int i = 0; i < seq_q; i++) {
        for (int j = 0; j < seq_kv; j++) {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum += static_cast<float>((i * head_dim + d + 1) * ((d * seq_kv + j + 1) * 0.5f));
            }
            scores[i * seq_kv + j] = sum * static_cast<float>(scale);
        }

        float max_val = scores[i * seq_kv];
        for (int j = 1; j < seq_kv; j++) {
            max_val = std::max(max_val, scores[i * seq_kv + j]);
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < seq_kv; j++) {
            float val = std::exp(scores[i * seq_kv + j] - max_val);
            probs[i * seq_kv + j] = val;
            sum_exp += val;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int j = 0; j < seq_kv; j++) {
            probs[i * seq_kv + j] *= inv_sum;
        }
    }

    for (int i = 0; i < seq_q; i++) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_kv; j++) {
                float v_val = static_cast<float>((j * head_dim + d + 1) * 0.25f);
                sum += probs[i * seq_kv + j] * v_val;
            }
            expected[i * head_dim + d] = sum;
        }
    }

    for (int i = 0; i < seq_q * head_dim; i++) {
        REQUIRE_THAT(pout[i], WithinAbs(expected[i], 1e-4f));
    }

    backend->unmap_buffer(ctx.outputs[0].buffer());
}

TEST_CASE("Vulkan pipeline add shader", "[vulkan][pipeline]") {
    VulkanTestFixture fixture;

    if (!fixture.is_available()) {
        SKIP("Vulkan backend not available");
    }

#ifndef GRANITE_HAS_SHADERC
    SKIP("Shaderc not available for GLSL compilation");
#endif

    auto* backend = fixture.backend();

    const char* shader_source = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint size;
} pc;

layout(binding = 0) readonly buffer A { float data_a[]; };
layout(binding = 1) readonly buffer B { float data_b[]; };
layout(binding = 2) writeonly buffer D { float data_d[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.size) return;
    data_d[i] = data_a[i] + data_b[i];
}
)";

    std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "granite_add_test.comp";
    {
        std::ofstream out(temp_path);
        out << shader_source;
    }

    PipelineDesc pipeline_desc;
    pipeline_desc.shader_source = temp_path.string();
    pipeline_desc.entry_point = "main";

    auto pipeline_result = backend->create_pipeline(pipeline_desc);
    REQUIRE(pipeline_result.ok());
    PipelineHandle pipeline = pipeline_result.value();

    const uint32_t size = 256;
    auto input_a = random_vector(size);
    auto input_b = random_vector(size);

    BufferDesc buf_desc;
    buf_desc.size = size * sizeof(float);
    buf_desc.memory_type = MemoryType::Shared;

    auto a_buf = backend->create_buffer(buf_desc);
    auto b_buf = backend->create_buffer(buf_desc);
    auto out_buf = backend->create_buffer(buf_desc);
    REQUIRE(a_buf.ok());
    REQUIRE(b_buf.ok());
    REQUIRE(out_buf.ok());

    REQUIRE(backend->write_buffer(a_buf.value(), input_a.data(), buf_desc.size).ok());
    REQUIRE(backend->write_buffer(b_buf.value(), input_b.data(), buf_desc.size).ok());

    REQUIRE(backend->begin_commands().ok());
    REQUIRE(backend->bind_pipeline(pipeline).ok());
    REQUIRE(backend->bind_buffer(0, a_buf.value()).ok());
    REQUIRE(backend->bind_buffer(1, b_buf.value()).ok());
    REQUIRE(backend->bind_buffer(2, out_buf.value()).ok());
    REQUIRE(backend->set_push_constants(&size, sizeof(size)).ok());

    uint32_t groups_x = (size + 64 - 1) / 64;
    REQUIRE(backend->dispatch(groups_x, 1, 1).ok());
    REQUIRE(backend->end_commands().ok());
    REQUIRE(backend->submit().ok());
    REQUIRE(backend->wait_for_completion().ok());

    std::vector<float> output(size);
    REQUIRE(backend->read_buffer(out_buf.value(), output.data(), buf_desc.size).ok());

    for (uint32_t i = 0; i < size; i++) {
        REQUIRE_THAT(output[i], WithinAbs(input_a[i] + input_b[i], 1e-6f));
    }

    backend->destroy_pipeline(pipeline);
    backend->destroy_buffer(a_buf.value());
    backend->destroy_buffer(b_buf.value());
    backend->destroy_buffer(out_buf.value());

    std::error_code ec;
    std::filesystem::remove(temp_path, ec);
}

#else  // !GRANITE_HAS_VULKAN

TEST_CASE("Vulkan backend not available", "[vulkan]") {
    SKIP("Vulkan support not compiled in");
}

#endif  // GRANITE_HAS_VULKAN
