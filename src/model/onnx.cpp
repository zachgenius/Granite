#include "granite/onnx.h"

#if defined(GRANITE_HAS_ONNX)

#include "granite/error.h"

#include "onnx.pb.h"

#include <array>
#include <filesystem>
#include <fstream>

namespace granite {
namespace {

Result<DataType> onnx_dtype_to_granite(int32_t onnx_dtype) {
    switch (onnx_dtype) {
        case onnx::TensorProto_DataType_FLOAT:
            return DataType::FP32;
        case onnx::TensorProto_DataType_FLOAT16:
            return DataType::FP16;
        case onnx::TensorProto_DataType_BFLOAT16:
            return DataType::BF16;
        case onnx::TensorProto_DataType_INT32:
            return DataType::INT32;
        case onnx::TensorProto_DataType_INT64:
            return DataType::INT64;
        case onnx::TensorProto_DataType_INT16:
            return DataType::INT16;
        case onnx::TensorProto_DataType_INT8:
            return DataType::INT8;
        case onnx::TensorProto_DataType_UINT8:
            return DataType::UINT8;
        case onnx::TensorProto_DataType_BOOL:
            return DataType::BOOL;
        case onnx::TensorProto_DataType_INT4:
            return DataType::INT4;
        case onnx::TensorProto_DataType_UINT4:
            return DataType::UINT4;
        default:
            break;
    }
    return Error(ErrorCode::UnsupportedModel,
                 "Unsupported ONNX tensor data type");
}

std::vector<int64_t> shape_from_tensor_proto(const onnx::TensorProto& tensor) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(tensor.dims_size()));
    for (int i = 0; i < tensor.dims_size(); ++i) {
        shape.push_back(tensor.dims(i));
    }
    return shape;
}

std::vector<int64_t> shape_from_value_info(const onnx::ValueInfoProto& value_info) {
    std::vector<int64_t> shape;
    if (!value_info.has_type() || !value_info.type().has_tensor_type()) {
        return shape;
    }
    const auto& tensor_type = value_info.type().tensor_type();
    if (!tensor_type.has_shape()) {
        return shape;
    }
    const auto& onnx_shape = tensor_type.shape();
    shape.reserve(static_cast<size_t>(onnx_shape.dim_size()));
    for (int i = 0; i < onnx_shape.dim_size(); ++i) {
        const auto& dim = onnx_shape.dim(i);
        if (dim.has_dim_value()) {
            shape.push_back(dim.dim_value());
        } else {
            shape.push_back(-1);
        }
    }
    return shape;
}

Result<DataType> dtype_from_value_info(const onnx::ValueInfoProto& value_info) {
    if (!value_info.has_type() || !value_info.type().has_tensor_type()) {
        return Error(ErrorCode::InvalidFormat, "Missing tensor type in ValueInfo");
    }
    const auto& tensor_type = value_info.type().tensor_type();
    if (tensor_type.elem_type() == onnx::TensorProto_DataType_UNDEFINED) {
        return Error(ErrorCode::InvalidFormat, "Missing elem_type in ValueInfo");
    }
    return onnx_dtype_to_granite(tensor_type.elem_type());
}

size_t shape_numel_checked(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    size_t total = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            return 0;
        }
        total *= static_cast<size_t>(dim);
    }
    return total;
}

MemoryLayout default_layout_for_shape(const std::vector<int64_t>& shape) {
    if (shape.size() == 4) {
        return MemoryLayout::NCHW;
    }
    return MemoryLayout::RowMajor;
}

bool apply_layout_from_transpose(const std::vector<int64_t>& perm,
                                 TensorDesc& input_desc,
                                 TensorDesc& output_desc) {
    if (perm.size() != 4) {
        return false;
    }
    const std::array<int64_t, 4> nchw_to_nhwc = {0, 2, 3, 1};
    const std::array<int64_t, 4> nhwc_to_nchw = {0, 3, 1, 2};
    bool is_nchw_to_nhwc = std::equal(perm.begin(), perm.end(), nchw_to_nhwc.begin());
    bool is_nhwc_to_nchw = std::equal(perm.begin(), perm.end(), nhwc_to_nchw.begin());

    if (is_nchw_to_nhwc) {
        if (input_desc.layout == MemoryLayout::RowMajor ||
            input_desc.layout == MemoryLayout::NCHW) {
            input_desc.layout = MemoryLayout::NCHW;
        }
        if (input_desc.layout == MemoryLayout::NCHW &&
            (output_desc.layout == MemoryLayout::RowMajor ||
             output_desc.layout == MemoryLayout::NCHW)) {
            output_desc.layout = MemoryLayout::NHWC;
        }
        return true;
    }
    if (is_nhwc_to_nchw) {
        if (input_desc.layout == MemoryLayout::RowMajor ||
            input_desc.layout == MemoryLayout::NCHW) {
            input_desc.layout = MemoryLayout::NHWC;
        }
        if (input_desc.layout == MemoryLayout::NHWC &&
            (output_desc.layout == MemoryLayout::RowMajor ||
             output_desc.layout == MemoryLayout::NHWC)) {
            output_desc.layout = MemoryLayout::NCHW;
        }
        return true;
    }
    return false;
}

Result<std::vector<uint8_t>> tensor_data_from_proto(const onnx::TensorProto& tensor,
                                                   DataType dtype,
                                                   const std::vector<int64_t>& shape,
                                                   const std::filesystem::path& base_dir) {
    const size_t numel = shape_numel_checked(shape);
    if (numel == 0) {
        return std::vector<uint8_t>{};
    }
    const size_t elem_size = dtype_size(dtype);
    const size_t expected_size = numel * elem_size;

    const auto& raw = tensor.raw_data();
    if (!raw.empty()) {
        if (raw.size() < expected_size) {
            return Error(ErrorCode::InvalidFormat, "ONNX raw_data is too small");
        }
        return std::vector<uint8_t>(raw.begin(), raw.begin() + expected_size);
    }

    if (tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL ||
        tensor.external_data_size() > 0) {
        std::string location;
        size_t offset = 0;
        size_t length = 0;
        for (const auto& entry : tensor.external_data()) {
            if (entry.key() == "location") {
                location = entry.value();
            } else if (entry.key() == "offset") {
                offset = static_cast<size_t>(std::stoull(entry.value()));
            } else if (entry.key() == "length") {
                length = static_cast<size_t>(std::stoull(entry.value()));
            }
        }
        if (location.empty()) {
            return Error(ErrorCode::InvalidFormat, "ONNX external_data missing location");
        }

        std::filesystem::path data_path = base_dir / location;
        std::ifstream data_file(data_path, std::ios::binary);
        if (!data_file) {
            return Error(ErrorCode::FileNotFound, "Missing ONNX external data file: " + data_path.string());
        }
        if (length == 0) {
            length = expected_size;
        }
        if (length < expected_size) {
            return Error(ErrorCode::InvalidFormat, "ONNX external_data length too small");
        }

        data_file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        std::vector<uint8_t> data(expected_size, 0);
        data_file.read(reinterpret_cast<char*>(data.data()),
                       static_cast<std::streamsize>(expected_size));
        if (!data_file) {
            return Error(ErrorCode::IOError, "Failed to read ONNX external data");
        }
        return data;
    }

    std::vector<uint8_t> data(expected_size, 0);

    switch (dtype) {
        case DataType::FP32: {
            if (tensor.float_data_size() != static_cast<int>(numel)) {
                return Error(ErrorCode::InvalidFormat, "Missing float_data for FP32 tensor");
            }
            auto* out = reinterpret_cast<float*>(data.data());
            for (size_t i = 0; i < numel; ++i) {
                out[i] = tensor.float_data(static_cast<int>(i));
            }
            break;
        }
        case DataType::FP16:
        case DataType::BF16: {
            if (tensor.int32_data_size() != static_cast<int>(numel)) {
                return Error(ErrorCode::InvalidFormat, "Missing int32_data for FP16/BF16 tensor");
            }
            auto* out = reinterpret_cast<uint16_t*>(data.data());
            for (size_t i = 0; i < numel; ++i) {
                out[i] = static_cast<uint16_t>(tensor.int32_data(static_cast<int>(i)));
            }
            break;
        }
        case DataType::INT32: {
            if (tensor.int32_data_size() != static_cast<int>(numel)) {
                return Error(ErrorCode::InvalidFormat, "Missing int32_data for INT32 tensor");
            }
            auto* out = reinterpret_cast<int32_t*>(data.data());
            for (size_t i = 0; i < numel; ++i) {
                out[i] = static_cast<int32_t>(tensor.int32_data(static_cast<int>(i)));
            }
            break;
        }
        case DataType::INT64: {
            if (tensor.int64_data_size() != static_cast<int>(numel)) {
                return Error(ErrorCode::InvalidFormat, "Missing int64_data for INT64 tensor");
            }
            auto* out = reinterpret_cast<int64_t*>(data.data());
            for (size_t i = 0; i < numel; ++i) {
                out[i] = static_cast<int64_t>(tensor.int64_data(static_cast<int>(i)));
            }
            break;
        }
        case DataType::INT16: {
            if (tensor.int32_data_size() != static_cast<int>(numel)) {
                return Error(ErrorCode::InvalidFormat, "Missing int32_data for INT16 tensor");
            }
            auto* out = reinterpret_cast<int16_t*>(data.data());
            for (size_t i = 0; i < numel; ++i) {
                out[i] = static_cast<int16_t>(tensor.int32_data(static_cast<int>(i)));
            }
            break;
        }
        case DataType::INT8: {
            if (tensor.int32_data_size() != static_cast<int>(numel)) {
                return Error(ErrorCode::InvalidFormat, "Missing int32_data for INT8 tensor");
            }
            auto* out = reinterpret_cast<int8_t*>(data.data());
            for (size_t i = 0; i < numel; ++i) {
                out[i] = static_cast<int8_t>(tensor.int32_data(static_cast<int>(i)));
            }
            break;
        }
        case DataType::UINT8: {
            if (tensor.int32_data_size() != static_cast<int>(numel)) {
                return Error(ErrorCode::InvalidFormat, "Missing int32_data for UINT8 tensor");
            }
            auto* out = reinterpret_cast<uint8_t*>(data.data());
            for (size_t i = 0; i < numel; ++i) {
                out[i] = static_cast<uint8_t>(tensor.int32_data(static_cast<int>(i)));
            }
            break;
        }
        case DataType::BOOL: {
            if (tensor.int32_data_size() != static_cast<int>(numel)) {
                return Error(ErrorCode::InvalidFormat, "Missing int32_data for BOOL tensor");
            }
            auto* out = reinterpret_cast<uint8_t*>(data.data());
            for (size_t i = 0; i < numel; ++i) {
                out[i] = static_cast<uint8_t>(tensor.int32_data(static_cast<int>(i)) != 0);
            }
            break;
        }
        case DataType::INT4:
        case DataType::UINT4:
            return Error(ErrorCode::InvalidFormat, "Missing raw_data for 4-bit tensor");
        default:
            return Error(ErrorCode::UnsupportedModel, "Unsupported ONNX tensor data type");
    }

    return data;
}

Result<OpType> map_onnx_op(const std::string& op_type) {
    if (op_type == "Add") return OpType::Add;
    if (op_type == "Sub") return OpType::Sub;
    if (op_type == "Mul") return OpType::Mul;
    if (op_type == "Div") return OpType::Div;
    if (op_type == "MatMul") return OpType::MatMul;
    if (op_type == "Gemm") return OpType::MatMul;
    if (op_type == "Relu") return OpType::ReLU;
    if (op_type == "Gelu") return OpType::GELU;
    if (op_type == "Sigmoid") return OpType::Sigmoid;
    if (op_type == "Tanh") return OpType::Tanh;
    if (op_type == "Softmax") return OpType::Softmax;
    if (op_type == "Reshape") return OpType::Reshape;
    if (op_type == "Transpose") return OpType::Transpose;
    if (op_type == "Concat") return OpType::Concat;
    if (op_type == "Split") return OpType::Split;
    if (op_type == "Slice") return OpType::Slice;
    if (op_type == "Gather") return OpType::Gather;
    if (op_type == "LayerNormalization") return OpType::LayerNorm;
    if (op_type == "RMSNormalization") return OpType::RMSNorm;

    return Error(ErrorCode::MissingOperator,
                 "Unsupported ONNX op: " + op_type);
}

Attributes parse_attributes(const onnx::NodeProto& node) {
    Attributes attrs;
    for (const auto& attr : node.attribute()) {
        switch (attr.type()) {
            case onnx::AttributeProto_AttributeType_INT:
                attrs.set(attr.name(), static_cast<int64_t>(attr.i()));
                break;
            case onnx::AttributeProto_AttributeType_FLOAT:
                attrs.set(attr.name(), static_cast<double>(attr.f()));
                break;
            case onnx::AttributeProto_AttributeType_STRING:
                attrs.set(attr.name(), attr.s());
                break;
            case onnx::AttributeProto_AttributeType_INTS: {
                std::vector<int64_t> vals;
                vals.reserve(static_cast<size_t>(attr.ints_size()));
                for (int i = 0; i < attr.ints_size(); ++i) {
                    vals.push_back(attr.ints(i));
                }
                attrs.set(attr.name(), std::move(vals));
                break;
            }
            case onnx::AttributeProto_AttributeType_FLOATS: {
                std::vector<double> vals;
                vals.reserve(static_cast<size_t>(attr.floats_size()));
                for (int i = 0; i < attr.floats_size(); ++i) {
                    vals.push_back(attr.floats(i));
                }
                attrs.set(attr.name(), std::move(vals));
                break;
            }
            default:
                break;
        }
    }
    return attrs;
}

TensorId ensure_tensor(Graph& graph,
                       std::unordered_map<std::string, TensorId>& tensor_ids,
                       const std::string& name,
                       const std::vector<int64_t>& shape,
                       DataType dtype,
                       MemoryLayout layout = MemoryLayout::RowMajor,
                       bool layout_set = false) {
    auto apply_layout = [&](TensorDesc& desc) {
        if (layout_set) {
            if (desc.layout == MemoryLayout::RowMajor) {
                desc.layout = layout;
            }
            return;
        }
        if (desc.layout == MemoryLayout::RowMajor) {
            MemoryLayout default_layout = default_layout_for_shape(shape);
            if (default_layout != MemoryLayout::RowMajor) {
                desc.layout = default_layout;
            }
        }
    };

    auto it = tensor_ids.find(name);
    if (it != tensor_ids.end()) {
        auto& desc = graph.tensor(it->second);
        if (desc.shape.empty() && !shape.empty()) {
            desc.shape = shape;
        }
        if (desc.dtype == DataType::FP32 && dtype != DataType::FP32) {
            desc.dtype = dtype;
        }
        apply_layout(desc);
        return it->second;
    }
    TensorDesc desc = TensorDesc::create(name, shape, dtype);
    apply_layout(desc);
    TensorId id = graph.add_tensor(std::move(desc));
    tensor_ids.emplace(name, id);
    return id;
}

}  // namespace

size_t OnnxTensor::numel() const {
    return shape_numel_checked(shape);
}

std::string OnnxModel::summary() const {
    return "ONNX Model: " + model_name +
        " (opset " + std::to_string(opset_version) +
        ", " + std::to_string(graph.num_nodes()) + " nodes, " +
        std::to_string(graph.num_tensors()) + " tensors, " +
        std::to_string(initializers.size()) + " initializers)";
}

Result<OnnxModel> load_onnx_model(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return Error(ErrorCode::FileNotFound, "ONNX file not found: " + path);
    }

    onnx::ModelProto model_proto;
    if (!model_proto.ParseFromIstream(&input)) {
        return Error(ErrorCode::InvalidFormat, "Failed to parse ONNX model");
    }

    OnnxModel model;
    model.model_name = model_proto.graph().name();
    model.producer_name = model_proto.producer_name();
    model.domain = model_proto.domain();
    model.model_version = model_proto.model_version();
    model.ir_version = model_proto.ir_version();

    const std::filesystem::path base_dir = std::filesystem::path(path).parent_path();

    for (const auto& opset : model_proto.opset_import()) {
        if (opset.domain().empty() || opset.domain() == "ai.onnx") {
            model.opset_version = opset.version();
            break;
        }
    }

    Graph graph(model.model_name.empty() ? "onnx_model" : model.model_name);
    std::unordered_map<std::string, TensorId> tensor_ids;

    for (const auto& initializer : model_proto.graph().initializer()) {
        auto dtype_result = onnx_dtype_to_granite(initializer.data_type());
        if (!dtype_result.ok()) {
            return dtype_result.error();
        }
        auto shape = shape_from_tensor_proto(initializer);
        auto data_result = tensor_data_from_proto(initializer, dtype_result.value(), shape, base_dir);
        if (!data_result.ok()) {
            return data_result.error();
        }

        OnnxTensor tensor;
        tensor.name = initializer.name();
        tensor.dtype = dtype_result.value();
        tensor.shape = std::move(shape);
        tensor.data = std::move(data_result).take();
        model.initializers.emplace(tensor.name, std::move(tensor));
    }

    std::vector<TensorId> input_ids;
    for (const auto& input : model_proto.graph().input()) {
        if (model.initializers.count(input.name()) > 0) {
            continue;
        }
        auto dtype_result = dtype_from_value_info(input);
        if (!dtype_result.ok()) {
            return dtype_result.error();
        }
        auto shape = shape_from_value_info(input);
        TensorId id = ensure_tensor(graph, tensor_ids, input.name(), shape, dtype_result.value());
        input_ids.push_back(id);
        model.graph_inputs.push_back(input.name());
    }

    for (const auto& output : model_proto.graph().output()) {
        auto dtype_result = dtype_from_value_info(output);
        if (!dtype_result.ok()) {
            return dtype_result.error();
        }
        auto shape = shape_from_value_info(output);
        ensure_tensor(graph, tensor_ids, output.name(), shape, dtype_result.value());
        model.graph_outputs.push_back(output.name());
    }

    for (const auto& value_info : model_proto.graph().value_info()) {
        if (value_info.name().empty()) {
            continue;
        }
        auto dtype_result = dtype_from_value_info(value_info);
        if (!dtype_result.ok()) {
            return dtype_result.error();
        }
        auto shape = shape_from_value_info(value_info);
        ensure_tensor(graph, tensor_ids, value_info.name(), shape, dtype_result.value());
    }

    for (const auto& initializer : model.initializers) {
        ensure_tensor(graph, tensor_ids, initializer.first, initializer.second.shape, initializer.second.dtype);
    }

    for (const auto& node : model_proto.graph().node()) {
        auto op_result = map_onnx_op(node.op_type());
        OpType op = OpType::Custom;
        Attributes attrs = parse_attributes(node);
        if (op_result.ok()) {
            op = op_result.value();
        } else {
            attrs.set("onnx_op_type", node.op_type());
            if (!node.domain().empty()) {
                attrs.set("onnx_domain", node.domain());
            }
        }

        std::vector<TensorId> inputs;
        inputs.reserve(static_cast<size_t>(node.input_size()));
        for (int i = 0; i < node.input_size(); ++i) {
            const auto& name = node.input(i);
            if (name.empty()) {
                continue;
            }
            inputs.push_back(ensure_tensor(graph, tensor_ids, name, {}, DataType::FP32));
        }

        std::vector<TensorId> outputs;
        outputs.reserve(static_cast<size_t>(node.output_size()));
        for (int i = 0; i < node.output_size(); ++i) {
            const auto& name = node.output(i);
            outputs.push_back(ensure_tensor(graph, tensor_ids, name, {}, DataType::FP32));
        }

        NodeId node_id = graph.add_node(op, std::move(inputs), std::move(outputs), std::move(attrs));

        if (op == OpType::Transpose) {
            const auto& node = graph.node(node_id);
            if (!node.inputs.empty() && !node.outputs.empty()) {
                auto perm = node.attrs.get<std::vector<int64_t>>("perm", {});
                if (!perm.empty()) {
                    auto& input_desc = graph.tensor(node.inputs[0]);
                    auto& output_desc = graph.tensor(node.outputs[0]);
                    (void)apply_layout_from_transpose(perm, input_desc, output_desc);
                }
            }
        }
    }

    graph.set_inputs(std::move(input_ids));
    std::vector<TensorId> output_ids;
    output_ids.reserve(model.graph_outputs.size());
    for (const auto& name : model.graph_outputs) {
        auto it = tensor_ids.find(name);
        if (it != tensor_ids.end()) {
            output_ids.push_back(it->second);
        }
    }
    graph.set_outputs(std::move(output_ids));

    model.graph = std::move(graph);
    return model;
}

}  // namespace granite

#endif  // GRANITE_HAS_ONNX
