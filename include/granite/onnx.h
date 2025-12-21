#pragma once

#include "granite/error.h"
#include "granite/graph.h"
#include "granite/types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace granite {

struct OnnxTensor {
    std::string name;
    DataType dtype = DataType::FP32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;

    [[nodiscard]] size_t numel() const;
    [[nodiscard]] size_t size_bytes() const { return data.size(); }
};

struct OnnxModel {
    std::string model_name;
    std::string producer_name;
    std::string domain;
    int64_t model_version = 0;
    int64_t ir_version = 0;
    int64_t opset_version = 0;

    Graph graph;
    std::unordered_map<std::string, OnnxTensor> initializers;
    std::vector<std::string> graph_inputs;
    std::vector<std::string> graph_outputs;

    [[nodiscard]] std::string summary() const;
};

#if defined(GRANITE_HAS_ONNX)
Result<OnnxModel> load_onnx_model(const std::string& path);
#else
inline Result<OnnxModel> load_onnx_model(const std::string& path) {
    (void)path;
    return Error(ErrorCode::NotImplemented,
                 "ONNX loader is disabled. Rebuild with GRANITE_ENABLE_ONNX=ON.");
}
#endif

}  // namespace granite
