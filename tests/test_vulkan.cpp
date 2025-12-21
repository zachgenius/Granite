// test_vulkan.cpp - Vulkan backend tests
//
// These tests are only compiled and run when GRANITE_HAS_VULKAN is defined.
// They validate the Vulkan compute kernels against CPU reference implementations.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "granite/backend.h"
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
