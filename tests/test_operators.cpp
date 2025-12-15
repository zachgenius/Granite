#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <granite/granite.h>

#include <cmath>
#include <vector>
#include <numeric>

using namespace granite;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// =============================================================================
// Test Helpers
// =============================================================================

class TestFixture {
public:
    TestFixture() {
        backend = create_backend(BackendType::CPU);
        REQUIRE(backend != nullptr);
        auto result = backend->initialize();
        REQUIRE(result.ok());
    }

    ~TestFixture() {
        backend->shutdown();
    }

    // Create a tensor and fill with data
    Tensor create_tensor(const std::vector<int64_t>& shape,
                        const std::vector<float>& data) {
        auto result = Tensor::allocate(shape, DataType::FP32, backend.get());
        REQUIRE(result.ok());
        auto tensor = std::move(result).take();

        auto map_result = backend->map_buffer(tensor.buffer());
        REQUIRE(map_result.ok());
        float* ptr = static_cast<float*>(map_result.value());
        std::copy(data.begin(), data.end(), ptr);
        backend->unmap_buffer(tensor.buffer());

        return tensor;
    }

    // Create tensor with sequential values
    Tensor create_sequential(const std::vector<int64_t>& shape, float start = 0.0f) {
        size_t n = 1;
        for (auto s : shape) n *= s;
        std::vector<float> data(n);
        std::iota(data.begin(), data.end(), start);
        return create_tensor(shape, data);
    }

    // Create tensor filled with a constant
    Tensor create_constant(const std::vector<int64_t>& shape, float value) {
        size_t n = 1;
        for (auto s : shape) n *= s;
        std::vector<float> data(n, value);
        return create_tensor(shape, data);
    }

    // Read tensor data back
    std::vector<float> read_tensor(const Tensor& tensor) {
        auto map_result = backend->map_buffer(tensor.buffer());
        REQUIRE(map_result.ok());
        const float* ptr = static_cast<const float*>(map_result.value());
        std::vector<float> data(ptr, ptr + tensor.numel());
        backend->unmap_buffer(tensor.buffer());
        return data;
    }

    // Execute an operator
    template<typename... Args>
    Result<void> execute_op(OpType op, std::vector<Tensor> inputs,
                           std::vector<Tensor>& outputs, Args&&... attr_args) {
        auto& registry = OperatorRegistry::instance();
        auto op_impl = registry.create(op, BackendType::CPU);
        if (!op_impl) {
            return Error(ErrorCode::NotImplemented, "Operator not found");
        }

        OpContext ctx;
        ctx.backend = backend.get();
        ctx.inputs = std::move(inputs);
        ctx.outputs = std::move(outputs);

        auto result = op_impl->execute(ctx);
        outputs = std::move(ctx.outputs);
        return result;
    }

    std::unique_ptr<IComputeBackend> backend;
};

// =============================================================================
// Binary Operator Tests
// =============================================================================

TEST_CASE("CPU Add operator", "[operators][cpu][binary]") {
    TestFixture fixture;

    SECTION("basic add") {
        auto a = fixture.create_tensor({4}, {1.0f, 2.0f, 3.0f, 4.0f});
        auto b = fixture.create_tensor({4}, {5.0f, 6.0f, 7.0f, 8.0f});
        auto out = fixture.create_constant({4}, 0.0f);

        std::vector<Tensor> outputs = {std::move(out)};
        auto result = fixture.execute_op(OpType::Add, {std::move(a), std::move(b)}, outputs);
        REQUIRE(result.ok());

        auto data = fixture.read_tensor(outputs[0]);
        REQUIRE(data[0] == 6.0f);
        REQUIRE(data[1] == 8.0f);
        REQUIRE(data[2] == 10.0f);
        REQUIRE(data[3] == 12.0f);
    }

    SECTION("2D tensor add") {
        auto a = fixture.create_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
        auto b = fixture.create_tensor({2, 3}, {6, 5, 4, 3, 2, 1});
        auto out = fixture.create_constant({2, 3}, 0.0f);

        std::vector<Tensor> outputs = {std::move(out)};
        auto result = fixture.execute_op(OpType::Add, {std::move(a), std::move(b)}, outputs);
        REQUIRE(result.ok());

        auto data = fixture.read_tensor(outputs[0]);
        for (auto v : data) {
            REQUIRE(v == 7.0f);
        }
    }
}

TEST_CASE("CPU Sub operator", "[operators][cpu][binary]") {
    TestFixture fixture;

    auto a = fixture.create_tensor({4}, {10.0f, 20.0f, 30.0f, 40.0f});
    auto b = fixture.create_tensor({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto out = fixture.create_constant({4}, 0.0f);

    std::vector<Tensor> outputs = {std::move(out)};
    auto result = fixture.execute_op(OpType::Sub, {std::move(a), std::move(b)}, outputs);
    REQUIRE(result.ok());

    auto data = fixture.read_tensor(outputs[0]);
    REQUIRE(data[0] == 9.0f);
    REQUIRE(data[1] == 18.0f);
    REQUIRE(data[2] == 27.0f);
    REQUIRE(data[3] == 36.0f);
}

TEST_CASE("CPU Mul operator", "[operators][cpu][binary]") {
    TestFixture fixture;

    auto a = fixture.create_tensor({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = fixture.create_tensor({4}, {2.0f, 3.0f, 4.0f, 5.0f});
    auto out = fixture.create_constant({4}, 0.0f);

    std::vector<Tensor> outputs = {std::move(out)};
    auto result = fixture.execute_op(OpType::Mul, {std::move(a), std::move(b)}, outputs);
    REQUIRE(result.ok());

    auto data = fixture.read_tensor(outputs[0]);
    REQUIRE(data[0] == 2.0f);
    REQUIRE(data[1] == 6.0f);
    REQUIRE(data[2] == 12.0f);
    REQUIRE(data[3] == 20.0f);
}

TEST_CASE("CPU Div operator", "[operators][cpu][binary]") {
    TestFixture fixture;

    auto a = fixture.create_tensor({4}, {10.0f, 20.0f, 30.0f, 40.0f});
    auto b = fixture.create_tensor({4}, {2.0f, 4.0f, 5.0f, 8.0f});
    auto out = fixture.create_constant({4}, 0.0f);

    std::vector<Tensor> outputs = {std::move(out)};
    auto result = fixture.execute_op(OpType::Div, {std::move(a), std::move(b)}, outputs);
    REQUIRE(result.ok());

    auto data = fixture.read_tensor(outputs[0]);
    REQUIRE(data[0] == 5.0f);
    REQUIRE(data[1] == 5.0f);
    REQUIRE(data[2] == 6.0f);
    REQUIRE(data[3] == 5.0f);
}

// =============================================================================
// Unary Operator Tests
// =============================================================================

TEST_CASE("CPU ReLU operator", "[operators][cpu][unary]") {
    TestFixture fixture;

    auto x = fixture.create_tensor({6}, {-3.0f, -1.0f, 0.0f, 0.5f, 1.0f, 3.0f});
    auto out = fixture.create_constant({6}, 0.0f);

    std::vector<Tensor> outputs = {std::move(out)};
    auto result = fixture.execute_op(OpType::ReLU, {std::move(x)}, outputs);
    REQUIRE(result.ok());

    auto data = fixture.read_tensor(outputs[0]);
    REQUIRE(data[0] == 0.0f);
    REQUIRE(data[1] == 0.0f);
    REQUIRE(data[2] == 0.0f);
    REQUIRE(data[3] == 0.5f);
    REQUIRE(data[4] == 1.0f);
    REQUIRE(data[5] == 3.0f);
}

TEST_CASE("CPU Sigmoid operator", "[operators][cpu][unary]") {
    TestFixture fixture;

    auto x = fixture.create_tensor({3}, {0.0f, -10.0f, 10.0f});
    auto out = fixture.create_constant({3}, 0.0f);

    std::vector<Tensor> outputs = {std::move(out)};
    auto result = fixture.execute_op(OpType::Sigmoid, {std::move(x)}, outputs);
    REQUIRE(result.ok());

    auto data = fixture.read_tensor(outputs[0]);
    REQUIRE_THAT(data[0], WithinAbs(0.5f, 1e-5f));          // sigmoid(0) = 0.5
    REQUIRE_THAT(data[1], WithinAbs(0.0f, 1e-4f));          // sigmoid(-10) ~ 0
    REQUIRE_THAT(data[2], WithinAbs(1.0f, 1e-4f));          // sigmoid(10) ~ 1
}

TEST_CASE("CPU Tanh operator", "[operators][cpu][unary]") {
    TestFixture fixture;

    auto x = fixture.create_tensor({3}, {0.0f, -1.0f, 1.0f});
    auto out = fixture.create_constant({3}, 0.0f);

    std::vector<Tensor> outputs = {std::move(out)};
    auto result = fixture.execute_op(OpType::Tanh, {std::move(x)}, outputs);
    REQUIRE(result.ok());

    auto data = fixture.read_tensor(outputs[0]);
    REQUIRE_THAT(data[0], WithinAbs(0.0f, 1e-5f));                       // tanh(0) = 0
    REQUIRE_THAT(data[1], WithinAbs(std::tanh(-1.0f), 1e-5f));
    REQUIRE_THAT(data[2], WithinAbs(std::tanh(1.0f), 1e-5f));
}

TEST_CASE("CPU GELU operator", "[operators][cpu][unary]") {
    TestFixture fixture;

    auto x = fixture.create_tensor({4}, {-2.0f, 0.0f, 1.0f, 2.0f});
    auto out = fixture.create_constant({4}, 0.0f);

    std::vector<Tensor> outputs = {std::move(out)};
    auto result = fixture.execute_op(OpType::GELU, {std::move(x)}, outputs);
    REQUIRE(result.ok());

    auto data = fixture.read_tensor(outputs[0]);
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    REQUIRE_THAT(data[1], WithinAbs(0.0f, 1e-5f));   // GELU(0) = 0
    REQUIRE(data[3] > data[2]);                       // GELU is monotonic for positive values
}

TEST_CASE("CPU SiLU operator", "[operators][cpu][unary]") {
    TestFixture fixture;

    auto x = fixture.create_tensor({3}, {0.0f, 1.0f, -1.0f});
    auto out = fixture.create_constant({3}, 0.0f);

    std::vector<Tensor> outputs = {std::move(out)};
    auto result = fixture.execute_op(OpType::SiLU, {std::move(x)}, outputs);
    REQUIRE(result.ok());

    auto data = fixture.read_tensor(outputs[0]);
    // SiLU = x * sigmoid(x)
    REQUIRE_THAT(data[0], WithinAbs(0.0f, 1e-5f));   // SiLU(0) = 0
    float silu_1 = 1.0f / (1.0f + std::exp(-1.0f));  // sigmoid(1)
    REQUIRE_THAT(data[1], WithinAbs(silu_1, 1e-5f));
}

// =============================================================================
// Softmax Tests
// =============================================================================

TEST_CASE("CPU Softmax operator", "[operators][cpu][softmax]") {
    TestFixture fixture;

    SECTION("basic softmax") {
        auto x = fixture.create_tensor({4}, {1.0f, 2.0f, 3.0f, 4.0f});
        auto out = fixture.create_constant({4}, 0.0f);

        auto& registry = OperatorRegistry::instance();
        auto op = registry.create(OpType::Softmax, BackendType::CPU);
        REQUIRE(op != nullptr);

        OpContext ctx;
        ctx.backend = fixture.backend.get();
        ctx.inputs = {std::move(x)};
        ctx.outputs = {std::move(out)};
        ctx.attrs.set("axis", int64_t(-1));

        auto result = op->execute(ctx);
        REQUIRE(result.ok());

        auto data = fixture.read_tensor(ctx.outputs[0]);

        // Sum should be 1
        float sum = 0.0f;
        for (auto v : data) sum += v;
        REQUIRE_THAT(sum, WithinAbs(1.0f, 1e-5f));

        // Values should be positive
        for (auto v : data) {
            REQUIRE(v > 0.0f);
        }

        // Higher input should give higher probability
        REQUIRE(data[3] > data[2]);
        REQUIRE(data[2] > data[1]);
        REQUIRE(data[1] > data[0]);
    }

    SECTION("2D softmax along last axis") {
        auto x = fixture.create_tensor({2, 4}, {1, 2, 3, 4, 4, 3, 2, 1});
        auto out = fixture.create_constant({2, 4}, 0.0f);

        auto& registry = OperatorRegistry::instance();
        auto op = registry.create(OpType::Softmax, BackendType::CPU);
        REQUIRE(op != nullptr);

        OpContext ctx;
        ctx.backend = fixture.backend.get();
        ctx.inputs = {std::move(x)};
        ctx.outputs = {std::move(out)};
        ctx.attrs.set("axis", int64_t(-1));

        auto result = op->execute(ctx);
        REQUIRE(result.ok());

        auto data = fixture.read_tensor(ctx.outputs[0]);

        // Each row should sum to 1
        float sum1 = data[0] + data[1] + data[2] + data[3];
        float sum2 = data[4] + data[5] + data[6] + data[7];
        REQUIRE_THAT(sum1, WithinAbs(1.0f, 1e-5f));
        REQUIRE_THAT(sum2, WithinAbs(1.0f, 1e-5f));
    }
}

// =============================================================================
// MatMul Tests
// =============================================================================

TEST_CASE("CPU MatMul operator", "[operators][cpu][matmul]") {
    TestFixture fixture;

    SECTION("2x2 identity multiplication") {
        // A @ I = A
        auto a = fixture.create_tensor({2, 2}, {1, 2, 3, 4});
        auto identity = fixture.create_tensor({2, 2}, {1, 0, 0, 1});
        auto out = fixture.create_constant({2, 2}, 0.0f);

        std::vector<Tensor> outputs = {std::move(out)};
        auto result = fixture.execute_op(OpType::MatMul, {std::move(a), std::move(identity)}, outputs);
        REQUIRE(result.ok());

        auto data = fixture.read_tensor(outputs[0]);
        REQUIRE(data[0] == 1.0f);
        REQUIRE(data[1] == 2.0f);
        REQUIRE(data[2] == 3.0f);
        REQUIRE(data[3] == 4.0f);
    }

    SECTION("2x3 @ 3x2 matmul") {
        // [[1, 2, 3], [4, 5, 6]] @ [[1, 2], [3, 4], [5, 6]]
        // = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
        // = [[22, 28], [49, 64]]
        auto a = fixture.create_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
        auto b = fixture.create_tensor({3, 2}, {1, 2, 3, 4, 5, 6});
        auto out = fixture.create_constant({2, 2}, 0.0f);

        std::vector<Tensor> outputs = {std::move(out)};
        auto result = fixture.execute_op(OpType::MatMul, {std::move(a), std::move(b)}, outputs);
        REQUIRE(result.ok());

        auto data = fixture.read_tensor(outputs[0]);
        REQUIRE(data[0] == 22.0f);
        REQUIRE(data[1] == 28.0f);
        REQUIRE(data[2] == 49.0f);
        REQUIRE(data[3] == 64.0f);
    }

    SECTION("vector-matrix multiplication") {
        // [1, 2, 3] @ [[1], [2], [3]] = [14]
        auto a = fixture.create_tensor({1, 3}, {1, 2, 3});
        auto b = fixture.create_tensor({3, 1}, {1, 2, 3});
        auto out = fixture.create_constant({1, 1}, 0.0f);

        std::vector<Tensor> outputs = {std::move(out)};
        auto result = fixture.execute_op(OpType::MatMul, {std::move(a), std::move(b)}, outputs);
        REQUIRE(result.ok());

        auto data = fixture.read_tensor(outputs[0]);
        REQUIRE(data[0] == 14.0f);  // 1*1 + 2*2 + 3*3
    }
}

// =============================================================================
// LayerNorm Tests
// =============================================================================

TEST_CASE("CPU LayerNorm operator", "[operators][cpu][norm]") {
    TestFixture fixture;

    SECTION("basic layer norm") {
        // Input with known statistics
        auto x = fixture.create_tensor({2, 4}, {0, 1, 2, 3, 4, 5, 6, 7});
        auto weight = fixture.create_constant({4}, 1.0f);
        auto bias = fixture.create_constant({4}, 0.0f);
        auto out = fixture.create_constant({2, 4}, 0.0f);

        auto& registry = OperatorRegistry::instance();
        auto op = registry.create(OpType::LayerNorm, BackendType::CPU);
        REQUIRE(op != nullptr);

        OpContext ctx;
        ctx.backend = fixture.backend.get();
        ctx.inputs = {std::move(x), std::move(weight), std::move(bias)};
        ctx.outputs = {std::move(out)};
        ctx.attrs.set("eps", 1e-5);

        auto result = op->execute(ctx);
        REQUIRE(result.ok());

        auto data = fixture.read_tensor(ctx.outputs[0]);

        // Each row should have mean ~0 and std ~1
        // Row 1: [0, 1, 2, 3] mean=1.5
        // Row 2: [4, 5, 6, 7] mean=5.5
        float mean1 = (data[0] + data[1] + data[2] + data[3]) / 4.0f;
        float mean2 = (data[4] + data[5] + data[6] + data[7]) / 4.0f;
        REQUIRE_THAT(mean1, WithinAbs(0.0f, 1e-5f));
        REQUIRE_THAT(mean2, WithinAbs(0.0f, 1e-5f));
    }

    SECTION("layer norm with affine transform") {
        auto x = fixture.create_tensor({1, 4}, {0, 1, 2, 3});
        auto weight = fixture.create_tensor({4}, {2, 2, 2, 2});
        auto bias = fixture.create_tensor({4}, {1, 1, 1, 1});
        auto out = fixture.create_constant({1, 4}, 0.0f);

        auto& registry = OperatorRegistry::instance();
        auto op = registry.create(OpType::LayerNorm, BackendType::CPU);
        REQUIRE(op != nullptr);

        OpContext ctx;
        ctx.backend = fixture.backend.get();
        ctx.inputs = {std::move(x), std::move(weight), std::move(bias)};
        ctx.outputs = {std::move(out)};
        ctx.attrs.set("eps", 1e-5);

        auto result = op->execute(ctx);
        REQUIRE(result.ok());

        auto data = fixture.read_tensor(ctx.outputs[0]);

        // Output = normalized * weight + bias
        // Normalized should be symmetric around 0
        // With weight=2, bias=1, output should be symmetric around 1
        float mean = (data[0] + data[1] + data[2] + data[3]) / 4.0f;
        REQUIRE_THAT(mean, WithinAbs(1.0f, 1e-5f));
    }
}

// =============================================================================
// RMSNorm Tests
// =============================================================================

TEST_CASE("CPU RMSNorm operator", "[operators][cpu][norm]") {
    TestFixture fixture;

    SECTION("basic rms norm") {
        auto x = fixture.create_tensor({2, 4}, {1, 1, 1, 1, 2, 2, 2, 2});
        auto weight = fixture.create_constant({4}, 1.0f);
        auto out = fixture.create_constant({2, 4}, 0.0f);

        auto& registry = OperatorRegistry::instance();
        auto op = registry.create(OpType::RMSNorm, BackendType::CPU);
        REQUIRE(op != nullptr);

        OpContext ctx;
        ctx.backend = fixture.backend.get();
        ctx.inputs = {std::move(x), std::move(weight)};
        ctx.outputs = {std::move(out)};
        ctx.attrs.set("eps", 1e-5);

        auto result = op->execute(ctx);
        REQUIRE(result.ok());

        auto data = fixture.read_tensor(ctx.outputs[0]);

        // For x = [1, 1, 1, 1], rms = sqrt(mean([1,1,1,1])) = 1
        // normalized = x / rms = [1, 1, 1, 1]
        REQUIRE_THAT(data[0], WithinAbs(1.0f, 1e-4f));
        REQUIRE_THAT(data[1], WithinAbs(1.0f, 1e-4f));

        // For x = [2, 2, 2, 2], rms = sqrt(mean([4,4,4,4])) = 2
        // normalized = x / rms = [1, 1, 1, 1]
        REQUIRE_THAT(data[4], WithinAbs(1.0f, 1e-4f));
        REQUIRE_THAT(data[5], WithinAbs(1.0f, 1e-4f));
    }

    SECTION("rms norm with varying values") {
        auto x = fixture.create_tensor({1, 4}, {1, 2, 3, 4});
        auto weight = fixture.create_constant({4}, 1.0f);
        auto out = fixture.create_constant({1, 4}, 0.0f);

        auto& registry = OperatorRegistry::instance();
        auto op = registry.create(OpType::RMSNorm, BackendType::CPU);
        REQUIRE(op != nullptr);

        OpContext ctx;
        ctx.backend = fixture.backend.get();
        ctx.inputs = {std::move(x), std::move(weight)};
        ctx.outputs = {std::move(out)};
        ctx.attrs.set("eps", 1e-5);

        auto result = op->execute(ctx);
        REQUIRE(result.ok());

        auto data = fixture.read_tensor(ctx.outputs[0]);

        // rms = sqrt((1 + 4 + 9 + 16)/4) = sqrt(7.5) ~ 2.738
        float expected_rms = std::sqrt(7.5f);
        REQUIRE_THAT(data[0], WithinAbs(1.0f / expected_rms, 1e-4f));
        REQUIRE_THAT(data[1], WithinAbs(2.0f / expected_rms, 1e-4f));
        REQUIRE_THAT(data[2], WithinAbs(3.0f / expected_rms, 1e-4f));
        REQUIRE_THAT(data[3], WithinAbs(4.0f / expected_rms, 1e-4f));
    }
}

// =============================================================================
// Operator Registry Tests
// =============================================================================

TEST_CASE("Operator registry", "[operators][registry]") {
    // Ensure operators are initialized
    initialize_operators();

    auto& registry = OperatorRegistry::instance();

    SECTION("CPU operators are registered") {
        REQUIRE(registry.has_implementation(OpType::Add, BackendType::CPU));
        REQUIRE(registry.has_implementation(OpType::Sub, BackendType::CPU));
        REQUIRE(registry.has_implementation(OpType::Mul, BackendType::CPU));
        REQUIRE(registry.has_implementation(OpType::Div, BackendType::CPU));
        REQUIRE(registry.has_implementation(OpType::ReLU, BackendType::CPU));
        REQUIRE(registry.has_implementation(OpType::GELU, BackendType::CPU));
        REQUIRE(registry.has_implementation(OpType::SiLU, BackendType::CPU));
        REQUIRE(registry.has_implementation(OpType::Sigmoid, BackendType::CPU));
        REQUIRE(registry.has_implementation(OpType::Tanh, BackendType::CPU));
        REQUIRE(registry.has_implementation(OpType::Softmax, BackendType::CPU));
        REQUIRE(registry.has_implementation(OpType::MatMul, BackendType::CPU));
        REQUIRE(registry.has_implementation(OpType::LayerNorm, BackendType::CPU));
        REQUIRE(registry.has_implementation(OpType::RMSNorm, BackendType::CPU));
    }

    SECTION("operator creation") {
        auto add_op = registry.create(OpType::Add, BackendType::CPU);
        REQUIRE(add_op != nullptr);
        REQUIRE(add_op->type() == OpType::Add);

        auto matmul_op = registry.create(OpType::MatMul, BackendType::CPU);
        REQUIRE(matmul_op != nullptr);
        REQUIRE(matmul_op->type() == OpType::MatMul);
    }
}
