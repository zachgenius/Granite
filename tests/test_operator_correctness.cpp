#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <granite/granite.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace granite;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

std::vector<float> random_vector(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<float> data(size);
    for (auto& v : data) {
        v = dist(gen);
    }
    return data;
}

Tensor create_tensor(IComputeBackend* backend,
                     const std::vector<int64_t>& shape,
                     const std::vector<float>& data) {
    auto result = Tensor::allocate(shape, DataType::FP32, backend);
    REQUIRE(result.ok());
    auto tensor = std::move(result).take();

    auto map_result = backend->map_buffer(tensor.buffer());
    REQUIRE(map_result.ok());
    auto* ptr = static_cast<float*>(map_result.value());
    std::copy(data.begin(), data.end(), ptr);
    backend->unmap_buffer(tensor.buffer());
    return tensor;
}

std::vector<float> read_tensor(IComputeBackend* backend, const Tensor& tensor) {
    auto map_result = backend->map_buffer(tensor.buffer());
    REQUIRE(map_result.ok());
    const auto* ptr = static_cast<const float*>(map_result.value());
    std::vector<float> data(ptr, ptr + tensor.numel());
    backend->unmap_buffer(tensor.buffer());
    return data;
}

std::vector<float> ref_matmul(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int64_t M, int64_t K, int64_t N) {
    std::vector<float> out(static_cast<size_t>(M * N), 0.0f);
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += a[m * K + k] * b[k * N + n];
            }
            out[m * N + n] = sum;
        }
    }
    return out;
}

std::vector<float> ref_softmax_rows(const std::vector<float>& x,
                                    int64_t rows, int64_t cols) {
    std::vector<float> out(static_cast<size_t>(rows * cols), 0.0f);
    for (int64_t r = 0; r < rows; r++) {
        const float* row = x.data() + r * cols;
        float max_val = row[0];
        for (int64_t c = 1; c < cols; c++) {
            max_val = std::max(max_val, row[c]);
        }
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; c++) {
            float v = std::exp(row[c] - max_val);
            out[r * cols + c] = v;
            sum += v;
        }
        for (int64_t c = 0; c < cols; c++) {
            out[r * cols + c] /= sum;
        }
    }
    return out;
}

std::vector<float> ref_rms_norm(const std::vector<float>& x,
                                const std::vector<float>& w,
                                float eps) {
    float sum_sq = 0.0f;
    for (float v : x) {
        sum_sq += v * v;
    }
    float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(x.size()) + eps);
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        out[i] = x[i] * inv_rms * w[i];
    }
    return out;
}

Result<void> run_op(IOperator* op_impl, OpContext& ctx) {
    auto validate = op_impl->validate(ctx);
    if (!validate.ok()) {
        return validate;
    }
    return op_impl->execute(ctx);
}

}  // namespace

TEST_CASE("Operator correctness CPU", "[operators][correctness][cpu]") {
    auto backend = create_backend(BackendType::CPU);
    REQUIRE(backend != nullptr);
    REQUIRE(backend->initialize().ok());

    auto& registry = OperatorRegistry::instance();

    SECTION("MatMul FP32") {
        const int64_t M = 2;
        const int64_t K = 3;
        const int64_t N = 4;
        auto a_data = random_vector(static_cast<size_t>(M * K));
        auto b_data = random_vector(static_cast<size_t>(K * N));
        auto ref = ref_matmul(a_data, b_data, M, K, N);

        auto a = create_tensor(backend.get(), {M, K}, a_data);
        auto b = create_tensor(backend.get(), {K, N}, b_data);
        auto out = Tensor::allocate({M, N}, DataType::FP32, backend.get()).take();

        auto op_impl = registry.create(OpType::MatMul, BackendType::CPU);
        REQUIRE(op_impl != nullptr);

        OpContext ctx;
        ctx.backend = backend.get();
        ctx.inputs = {std::move(a), std::move(b)};
        ctx.outputs = {std::move(out)};

        REQUIRE(run_op(op_impl.get(), ctx).ok());

        auto out_data = read_tensor(backend.get(), ctx.outputs[0]);
        for (size_t i = 0; i < out_data.size(); i++) {
            REQUIRE_THAT(out_data[i], WithinRel(ref[i], 1e-5f));
        }
    }

    SECTION("Softmax last-axis") {
        const int64_t rows = 2;
        const int64_t cols = 5;
        auto x_data = random_vector(static_cast<size_t>(rows * cols));
        auto ref = ref_softmax_rows(x_data, rows, cols);

        auto x = create_tensor(backend.get(), {rows, cols}, x_data);
        auto out = Tensor::allocate({rows, cols}, DataType::FP32, backend.get()).take();

        auto op_impl = registry.create(OpType::Softmax, BackendType::CPU);
        REQUIRE(op_impl != nullptr);

        OpContext ctx;
        ctx.backend = backend.get();
        ctx.inputs = {std::move(x)};
        ctx.outputs = {std::move(out)};
        ctx.attrs.set("axis", static_cast<int64_t>(1));

        REQUIRE(run_op(op_impl.get(), ctx).ok());

        auto out_data = read_tensor(backend.get(), ctx.outputs[0]);
        for (size_t i = 0; i < out_data.size(); i++) {
            REQUIRE_THAT(out_data[i], WithinAbs(ref[i], 1e-5f));
        }
    }

    SECTION("RMSNorm 1D") {
        const int64_t size = 16;
        auto x_data = random_vector(static_cast<size_t>(size));
        auto w_data = random_vector(static_cast<size_t>(size));
        float eps = 1e-5f;
        auto ref = ref_rms_norm(x_data, w_data, eps);

        auto x = create_tensor(backend.get(), {size}, x_data);
        auto w = create_tensor(backend.get(), {size}, w_data);
        auto out = Tensor::allocate({size}, DataType::FP32, backend.get()).take();

        auto op_impl = registry.create(OpType::RMSNorm, BackendType::CPU);
        REQUIRE(op_impl != nullptr);

        OpContext ctx;
        ctx.backend = backend.get();
        ctx.inputs = {std::move(x), std::move(w)};
        ctx.outputs = {std::move(out)};
        ctx.attrs.set("eps", static_cast<double>(eps));

        REQUIRE(run_op(op_impl.get(), ctx).ok());

        auto out_data = read_tensor(backend.get(), ctx.outputs[0]);
        for (size_t i = 0; i < out_data.size(); i++) {
            REQUIRE_THAT(out_data[i], WithinRel(ref[i], 1e-5f));
        }
    }

    backend->shutdown();
}

#ifdef GRANITE_HAS_VULKAN
TEST_CASE("Operator correctness Vulkan", "[operators][correctness][vulkan]") {
    auto backend = create_backend(BackendType::Vulkan);
    if (!backend) {
        SKIP("Vulkan backend not available");
    }
    if (!backend->initialize().ok()) {
        SKIP("Vulkan backend failed to initialize");
    }

    auto& registry = OperatorRegistry::instance();

    SECTION("MatMul FP32") {
        const int64_t M = 2;
        const int64_t K = 3;
        const int64_t N = 4;
        auto a_data = random_vector(static_cast<size_t>(M * K));
        auto b_data = random_vector(static_cast<size_t>(K * N));
        auto ref = ref_matmul(a_data, b_data, M, K, N);

        auto a = create_tensor(backend.get(), {M, K}, a_data);
        auto b = create_tensor(backend.get(), {K, N}, b_data);
        auto out = Tensor::allocate({M, N}, DataType::FP32, backend.get()).take();

        auto op_impl = registry.create(OpType::MatMul, BackendType::Vulkan);
        if (!op_impl) {
            SKIP("Vulkan MatMul not available");
        }

        OpContext ctx;
        ctx.backend = backend.get();
        ctx.inputs = {std::move(a), std::move(b)};
        ctx.outputs = {std::move(out)};

        REQUIRE(run_op(op_impl.get(), ctx).ok());
        auto out_data = read_tensor(backend.get(), ctx.outputs[0]);
        for (size_t i = 0; i < out_data.size(); i++) {
            REQUIRE_THAT(out_data[i], WithinRel(ref[i], 1e-4f));
        }
    }

    SECTION("Softmax last-axis") {
        const int64_t rows = 2;
        const int64_t cols = 5;
        auto x_data = random_vector(static_cast<size_t>(rows * cols));
        auto ref = ref_softmax_rows(x_data, rows, cols);

        auto x = create_tensor(backend.get(), {rows, cols}, x_data);
        auto out = Tensor::allocate({rows, cols}, DataType::FP32, backend.get()).take();

        auto op_impl = registry.create(OpType::Softmax, BackendType::Vulkan);
        if (!op_impl) {
            SKIP("Vulkan Softmax not available");
        }

        OpContext ctx;
        ctx.backend = backend.get();
        ctx.inputs = {std::move(x)};
        ctx.outputs = {std::move(out)};
        ctx.attrs.set("axis", static_cast<int64_t>(1));

        REQUIRE(run_op(op_impl.get(), ctx).ok());
        auto out_data = read_tensor(backend.get(), ctx.outputs[0]);
        for (size_t i = 0; i < out_data.size(); i++) {
            REQUIRE_THAT(out_data[i], WithinAbs(ref[i], 1e-4f));
        }
    }

    SECTION("RMSNorm 1D") {
        const int64_t size = 16;
        auto x_data = random_vector(static_cast<size_t>(size));
        auto w_data = random_vector(static_cast<size_t>(size));
        float eps = 1e-5f;
        auto ref = ref_rms_norm(x_data, w_data, eps);

        auto x = create_tensor(backend.get(), {size}, x_data);
        auto w = create_tensor(backend.get(), {size}, w_data);
        auto out = Tensor::allocate({size}, DataType::FP32, backend.get()).take();

        auto op_impl = registry.create(OpType::RMSNorm, BackendType::Vulkan);
        if (!op_impl) {
            SKIP("Vulkan RMSNorm not available");
        }

        OpContext ctx;
        ctx.backend = backend.get();
        ctx.inputs = {std::move(x), std::move(w)};
        ctx.outputs = {std::move(out)};
        ctx.attrs.set("eps", static_cast<double>(eps));

        REQUIRE(run_op(op_impl.get(), ctx).ok());
        auto out_data = read_tensor(backend.get(), ctx.outputs[0]);
        for (size_t i = 0; i < out_data.size(); i++) {
            REQUIRE_THAT(out_data[i], WithinRel(ref[i], 1e-4f));
        }
    }

    backend->shutdown();
}
#endif
