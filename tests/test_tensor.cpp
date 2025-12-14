#include <catch2/catch_test_macros.hpp>
#include <granite/granite.h>

using namespace granite;

TEST_CASE("Tensor shape utilities", "[tensor]") {
    SECTION("shape_numel") {
        std::vector<int64_t> shape1 = {2, 3, 4};
        REQUIRE(shape_numel(shape1) == 24);

        std::vector<int64_t> shape2 = {1};
        REQUIRE(shape_numel(shape2) == 1);

        std::vector<int64_t> empty_shape = {};
        REQUIRE(shape_numel(empty_shape) == 0);
    }

    SECTION("shapes_broadcastable") {
        std::vector<int64_t> a = {2, 3, 4};
        std::vector<int64_t> b = {1, 3, 4};
        REQUIRE(shapes_broadcastable(a, b) == true);

        std::vector<int64_t> c = {4};
        REQUIRE(shapes_broadcastable(a, c) == true);

        std::vector<int64_t> d = {2, 3, 5};
        REQUIRE(shapes_broadcastable(a, d) == false);
    }

    SECTION("broadcast_shapes") {
        std::vector<int64_t> a = {2, 1, 4};
        std::vector<int64_t> b = {3, 1};
        auto result = broadcast_shapes(a, b);

        REQUIRE(result.size() == 3);
        REQUIRE(result[0] == 2);
        REQUIRE(result[1] == 3);
        REQUIRE(result[2] == 4);
    }
}

TEST_CASE("Tensor allocation with CPU backend", "[tensor][cpu]") {
    auto backend = create_backend(BackendType::CPU);
    REQUIRE(backend != nullptr);

    auto init_result = backend->initialize();
    REQUIRE(init_result.ok());

    SECTION("Basic allocation") {
        std::vector<int64_t> shape = {32, 64};
        auto tensor_result = Tensor::allocate(shape, DataType::FP32, backend.get());

        REQUIRE(tensor_result.ok());

        auto& tensor = tensor_result.value();
        REQUIRE(tensor.ndim() == 2);
        REQUIRE(tensor.size(0) == 32);
        REQUIRE(tensor.size(1) == 64);
        REQUIRE(tensor.numel() == 32 * 64);
        REQUIRE(tensor.dtype() == DataType::FP32);
        REQUIRE(tensor.size_bytes() == 32 * 64 * 4);
        REQUIRE(tensor.is_contiguous() == true);
    }

    SECTION("Empty tensor") {
        Tensor empty_tensor;
        REQUIRE(empty_tensor.empty() == true);
        REQUIRE(empty_tensor.valid() == false);
    }

    backend->shutdown();
}

TEST_CASE("Tensor view operations", "[tensor][cpu]") {
    auto backend = create_backend(BackendType::CPU);
    REQUIRE(backend != nullptr);
    backend->initialize();

    std::vector<int64_t> shape = {2, 3, 4};
    auto tensor_result = Tensor::allocate(shape, DataType::FP32, backend.get());
    REQUIRE(tensor_result.ok());
    auto tensor = std::move(tensor_result).take();

    SECTION("reshape view") {
        std::vector<int64_t> new_shape = {6, 4};
        auto view_result = tensor.view(new_shape);
        REQUIRE(view_result.ok());

        auto& view = view_result.value();
        REQUIRE(view.numel() == tensor.numel());
        REQUIRE(view.size(0) == 6);
        REQUIRE(view.size(1) == 4);
    }

    SECTION("invalid reshape") {
        std::vector<int64_t> bad_shape = {5, 5};  // 25 != 24
        auto view_result = tensor.view(bad_shape);
        REQUIRE(view_result.ok() == false);
        REQUIRE(view_result.error().code() == ErrorCode::InvalidShape);
    }

    SECTION("squeeze") {
        std::vector<int64_t> shape_with_ones = {1, 3, 1, 4};
        auto t = Tensor::allocate(shape_with_ones, DataType::FP32, backend.get());
        REQUIRE(t.ok());

        auto squeezed = t.value().squeeze();
        REQUIRE(squeezed.ndim() == 2);
        REQUIRE(squeezed.size(0) == 3);
        REQUIRE(squeezed.size(1) == 4);
    }

    SECTION("unsqueeze") {
        auto unsqueeze_result = tensor.unsqueeze(0);
        REQUIRE(unsqueeze_result.ok());
        REQUIRE(unsqueeze_result.value().ndim() == 4);
        REQUIRE(unsqueeze_result.value().size(0) == 1);
    }

    backend->shutdown();
}

TEST_CASE("Error propagation", "[error]") {
    SECTION("Result with value") {
        Result<int> result(42);
        REQUIRE(result.ok() == true);
        REQUIRE(result.value() == 42);
    }

    SECTION("Result with error") {
        Result<int> result(Error(ErrorCode::InvalidArgument, "test error"));
        REQUIRE(result.ok() == false);
        REQUIRE(result.error().code() == ErrorCode::InvalidArgument);
    }

    SECTION("Result<void>") {
        Result<void> success;
        REQUIRE(success.ok() == true);

        Result<void> failure(ErrorCode::InternalError);
        REQUIRE(failure.ok() == false);
    }

    SECTION("value_or") {
        Result<int> success(42);
        REQUIRE(success.value_or(0) == 42);

        Result<int> failure(ErrorCode::InvalidArgument);
        REQUIRE(failure.value_or(0) == 0);
    }
}
