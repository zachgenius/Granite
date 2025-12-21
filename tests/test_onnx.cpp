#include <catch2/catch_test_macros.hpp>

#include "granite/onnx.h"

#include <chrono>
#include <array>
#include <filesystem>
#include <fstream>

#if defined(GRANITE_HAS_ONNX)
#include "onnx.pb.h"

namespace granite {
namespace {

std::filesystem::path write_temp_model(const onnx::ModelProto& model) {
    auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("granite_onnx_test_" + std::to_string(stamp) + ".onnx");
    std::ofstream output(path, std::ios::binary);
    REQUIRE(output.good());
    REQUIRE(model.SerializeToOstream(&output));
    return path;
}

onnx::ModelProto build_transpose_model(const std::array<int64_t, 4>& input_shape,
                                       const std::array<int64_t, 4>& perm) {
    onnx::ModelProto model;
    model.set_ir_version(onnx::IR_VERSION);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("transpose_layout_test");

    auto* input = graph->add_input();
    input->set_name("x");
    auto* input_type = input->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    auto* input_dims = input_type->mutable_shape();
    for (int64_t dim : input_shape) {
        input_dims->add_dim()->set_dim_value(dim);
    }

    std::array<int64_t, 4> output_shape = {
        input_shape[perm[0]],
        input_shape[perm[1]],
        input_shape[perm[2]],
        input_shape[perm[3]],
    };

    auto* output = graph->add_output();
    output->set_name("y");
    auto* output_type = output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    auto* output_dims = output_type->mutable_shape();
    for (int64_t dim : output_shape) {
        output_dims->add_dim()->set_dim_value(dim);
    }

    auto* node = graph->add_node();
    node->set_op_type("Transpose");
    node->add_input("x");
    node->add_output("y");

    auto* attr = node->add_attribute();
    attr->set_name("perm");
    attr->set_type(onnx::AttributeProto_AttributeType_INTS);
    for (int64_t dim : perm) {
        attr->add_ints(dim);
    }

    return model;
}

const TensorDesc* find_tensor(const Graph& graph, const std::string& name) {
    for (const auto& tensor : graph.tensors()) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

}  // namespace
}  // namespace granite

TEST_CASE("ONNX transpose infers NCHW to NHWC layout") {
    auto model_proto = granite::build_transpose_model({1, 3, 4, 5}, {0, 2, 3, 1});
    auto path = granite::write_temp_model(model_proto);

    auto result = granite::load_onnx_model(path.string());
    std::filesystem::remove(path);
    REQUIRE(result.ok());

    const auto& graph = result.value().graph;
    const auto* input = granite::find_tensor(graph, "x");
    const auto* output = granite::find_tensor(graph, "y");
    REQUIRE(input);
    REQUIRE(output);
    CHECK(input->layout == granite::MemoryLayout::NCHW);
    CHECK(output->layout == granite::MemoryLayout::NHWC);
}

TEST_CASE("ONNX transpose infers NHWC to NCHW layout") {
    auto model_proto = granite::build_transpose_model({1, 4, 5, 3}, {0, 3, 1, 2});
    auto path = granite::write_temp_model(model_proto);

    auto result = granite::load_onnx_model(path.string());
    std::filesystem::remove(path);
    REQUIRE(result.ok());

    const auto& graph = result.value().graph;
    const auto* input = granite::find_tensor(graph, "x");
    const auto* output = granite::find_tensor(graph, "y");
    REQUIRE(input);
    REQUIRE(output);
    CHECK(input->layout == granite::MemoryLayout::NHWC);
    CHECK(output->layout == granite::MemoryLayout::NCHW);
}

TEST_CASE("ONNX loader keeps 2D tensors row-major") {
    onnx::ModelProto model;
    model.set_ir_version(onnx::IR_VERSION);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("row_major_test");

    auto* input = graph->add_input();
    input->set_name("x");
    auto* input_type = input->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    auto* input_dims = input_type->mutable_shape();
    input_dims->add_dim()->set_dim_value(1);
    input_dims->add_dim()->set_dim_value(8);

    auto* output = graph->add_output();
    output->set_name("y");
    auto* output_type = output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    auto* output_dims = output_type->mutable_shape();
    output_dims->add_dim()->set_dim_value(1);
    output_dims->add_dim()->set_dim_value(8);

    auto* node = graph->add_node();
    node->set_op_type("Add");
    node->add_input("x");
    node->add_input("x");
    node->add_output("y");

    auto path = granite::write_temp_model(model);
    auto result = granite::load_onnx_model(path.string());
    std::filesystem::remove(path);
    REQUIRE(result.ok());

    const auto& graph_result = result.value().graph;
    const auto* input_tensor = granite::find_tensor(graph_result, "x");
    const auto* output_tensor = granite::find_tensor(graph_result, "y");
    REQUIRE(input_tensor);
    REQUIRE(output_tensor);
    CHECK(input_tensor->layout == granite::MemoryLayout::RowMajor);
    CHECK(output_tensor->layout == granite::MemoryLayout::RowMajor);
}

#else

TEST_CASE("ONNX loader disabled") {
    SUCCEED();
}

#endif
