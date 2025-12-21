#include <catch2/catch_test_macros.hpp>
#include <granite/granite.h>

#include <cstdlib>
#include <cstring>
#include <vector>

using namespace granite;

TEST_CASE("Operator stress loop (CPU)", "[stress][cpu]") {
    auto backend = create_backend(BackendType::CPU);
    REQUIRE(backend != nullptr);
    REQUIRE(backend->initialize().ok());

    const char* env_iters = std::getenv("GRANITE_STRESS_ITERS");
    int iters = env_iters ? std::max(1, std::atoi(env_iters)) : 200;

    auto& registry = OperatorRegistry::instance();
    auto add_impl = registry.create(OpType::Add, BackendType::CPU);
    auto mul_impl = registry.create(OpType::Mul, BackendType::CPU);
    REQUIRE(add_impl != nullptr);
    REQUIRE(mul_impl != nullptr);

    std::vector<int64_t> shape = {256};
    std::vector<float> a_data(256, 1.5f);
    std::vector<float> b_data(256, 2.0f);

    auto a = Tensor::allocate(shape, DataType::FP32, backend.get()).take();
    auto b = Tensor::allocate(shape, DataType::FP32, backend.get()).take();
    auto out = Tensor::allocate(shape, DataType::FP32, backend.get()).take();

    auto map_a = backend->map_buffer(a.buffer());
    auto map_b = backend->map_buffer(b.buffer());
    REQUIRE(map_a.ok());
    REQUIRE(map_b.ok());
    std::memcpy(map_a.value(), a_data.data(), a_data.size() * sizeof(float));
    std::memcpy(map_b.value(), b_data.data(), b_data.size() * sizeof(float));
    backend->unmap_buffer(a.buffer());
    backend->unmap_buffer(b.buffer());

    for (int i = 0; i < iters; i++) {
        OpContext ctx;
        ctx.backend = backend.get();
        ctx.inputs = {a, b};
        ctx.outputs = {out};
        REQUIRE(add_impl->execute(ctx).ok());

        ctx.inputs = {ctx.outputs[0], b};
        ctx.outputs = {out};
        REQUIRE(mul_impl->execute(ctx).ok());
    }

    backend->shutdown();
}
