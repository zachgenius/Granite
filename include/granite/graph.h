#pragma once

#include "granite/types.h"
#include "granite/error.h"
#include "granite/operators.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>
#include <functional>

namespace granite {

// Forward declarations
class IComputeBackend;
class Tensor;

// =============================================================================
// Tensor Descriptor (metadata for graph tensors)
// =============================================================================

struct TensorDesc {
    std::string name;
    std::vector<int64_t> shape;
    DataType dtype = DataType::FP32;
    MemoryLayout layout = MemoryLayout::RowMajor;

    // Computed properties
    [[nodiscard]] size_t numel() const;
    [[nodiscard]] size_t size_bytes() const;

    // Factory methods
    static TensorDesc create(const std::string& name,
                            std::vector<int64_t> shape,
                            DataType dtype = DataType::FP32);
};

// =============================================================================
// Graph Node
// =============================================================================

using NodeId = uint32_t;
using TensorId = uint32_t;

constexpr NodeId INVALID_NODE_ID = UINT32_MAX;
constexpr TensorId INVALID_TENSOR_ID = UINT32_MAX;

struct Node {
    NodeId id = INVALID_NODE_ID;
    OpType op = OpType::Custom;
    std::string name;

    // Connections (indices into graph's tensor list)
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;

    // Operator attributes
    Attributes attrs;

    // Execution metadata
    bool is_fused = false;          // Part of a fused kernel
    NodeId fused_into = INVALID_NODE_ID;  // Which node this was fused into
};

// =============================================================================
// Graph (DAG representation)
// =============================================================================

class Graph {
public:
    Graph() = default;
    explicit Graph(const std::string& name);

    // Construction
    [[nodiscard]] TensorId add_tensor(TensorDesc desc);
    [[nodiscard]] NodeId add_node(OpType op,
                                  std::vector<TensorId> inputs,
                                  std::vector<TensorId> outputs,
                                  Attributes attrs = {});

    // Mark graph inputs/outputs
    void set_inputs(std::vector<TensorId> inputs);
    void set_outputs(std::vector<TensorId> outputs);

    // Accessors
    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] size_t num_nodes() const { return nodes_.size(); }
    [[nodiscard]] size_t num_tensors() const { return tensors_.size(); }

    [[nodiscard]] const Node& node(NodeId id) const;
    [[nodiscard]] Node& node(NodeId id);
    [[nodiscard]] const TensorDesc& tensor(TensorId id) const;
    [[nodiscard]] TensorDesc& tensor(TensorId id);

    [[nodiscard]] const std::vector<TensorId>& inputs() const { return input_tensors_; }
    [[nodiscard]] const std::vector<TensorId>& outputs() const { return output_tensors_; }

    // Iteration
    [[nodiscard]] const std::vector<Node>& nodes() const { return nodes_; }
    [[nodiscard]] const std::vector<TensorDesc>& tensors() const { return tensors_; }

    // Analysis
    [[nodiscard]] Result<std::vector<NodeId>> topological_sort() const;
    [[nodiscard]] Result<void> validate() const;
    [[nodiscard]] bool has_cycle() const;

    // Get nodes that produce/consume a tensor
    [[nodiscard]] NodeId producer(TensorId tensor_id) const;
    [[nodiscard]] std::vector<NodeId> consumers(TensorId tensor_id) const;

    // Debug
    [[nodiscard]] std::string to_string() const;
    void dump() const;

private:
    std::string name_ = "unnamed";
    std::vector<Node> nodes_;
    std::vector<TensorDesc> tensors_;
    std::vector<TensorId> input_tensors_;
    std::vector<TensorId> output_tensors_;

    // Lookup tables for fast access
    std::unordered_map<TensorId, NodeId> tensor_producer_;
    std::unordered_map<TensorId, std::vector<NodeId>> tensor_consumers_;
};

// =============================================================================
// Graph Builder (fluent API for easier graph construction)
// =============================================================================

class GraphBuilder {
public:
    explicit GraphBuilder(const std::string& name = "graph");

    // Add input tensor
    TensorId input(const std::string& name,
                   std::vector<int64_t> shape,
                   DataType dtype = DataType::FP32);

    // Add operations (returns output tensor ID)
    TensorId add(TensorId a, TensorId b, const std::string& name = "");
    TensorId sub(TensorId a, TensorId b, const std::string& name = "");
    TensorId mul(TensorId a, TensorId b, const std::string& name = "");
    TensorId div(TensorId a, TensorId b, const std::string& name = "");
    TensorId matmul(TensorId a, TensorId b, const std::string& name = "");

    TensorId relu(TensorId x, const std::string& name = "");
    TensorId gelu(TensorId x, const std::string& name = "");
    TensorId silu(TensorId x, const std::string& name = "");
    TensorId softmax(TensorId x, int axis = -1, const std::string& name = "");

    TensorId layer_norm(TensorId x, TensorId weight, TensorId bias,
                        float eps = 1e-5f, const std::string& name = "");
    TensorId rms_norm(TensorId x, TensorId weight,
                      float eps = 1e-5f, const std::string& name = "");

    // Mark outputs and build
    void mark_output(TensorId tensor);
    [[nodiscard]] Result<Graph> build();

private:
    Graph graph_;
    std::vector<TensorId> outputs_;
    uint32_t tensor_counter_ = 0;
    uint32_t node_counter_ = 0;

    TensorId add_intermediate(const std::string& name,
                             const std::vector<int64_t>& shape,
                             DataType dtype);
    std::string generate_name(const std::string& prefix);
};

// =============================================================================
// Execution Plan (result of graph compilation)
// =============================================================================

struct ExecutionStep {
    NodeId node_id;
    OpType op;

    // Buffer bindings for this step
    std::vector<BufferHandle> input_buffers;
    std::vector<BufferHandle> output_buffers;

    // The operator instance to execute
    IOperator* op_impl = nullptr;
};

class ExecutionPlan {
public:
    ExecutionPlan() = default;

    void add_step(ExecutionStep step);
    void set_input_buffers(std::vector<BufferHandle> buffers);
    void set_output_buffers(std::vector<BufferHandle> buffers);

    [[nodiscard]] const std::vector<ExecutionStep>& steps() const { return steps_; }
    [[nodiscard]] const std::vector<BufferHandle>& input_buffers() const { return input_buffers_; }
    [[nodiscard]] const std::vector<BufferHandle>& output_buffers() const { return output_buffers_; }

    [[nodiscard]] size_t num_steps() const { return steps_.size(); }
    [[nodiscard]] size_t total_memory_bytes() const { return total_memory_; }

    void set_total_memory(size_t bytes) { total_memory_ = bytes; }

private:
    std::vector<ExecutionStep> steps_;
    std::vector<BufferHandle> input_buffers_;
    std::vector<BufferHandle> output_buffers_;
    size_t total_memory_ = 0;
};

// =============================================================================
// Graph Executor
// =============================================================================

class GraphExecutor {
public:
    explicit GraphExecutor(IComputeBackend* backend);
    ~GraphExecutor();

    // Compile graph into execution plan
    [[nodiscard]] Result<void> compile(const Graph& graph);

    // Execute with provided input tensors
    [[nodiscard]] Result<std::vector<Tensor>> execute(
        const std::vector<Tensor>& inputs);

    // Get the compiled plan (for inspection)
    [[nodiscard]] const ExecutionPlan* plan() const { return plan_.get(); }

    // Memory usage
    [[nodiscard]] size_t memory_usage() const;

private:
    IComputeBackend* backend_;
    std::unique_ptr<ExecutionPlan> plan_;
    std::vector<std::unique_ptr<IOperator>> operators_;

    // Internal buffers for intermediate tensors
    std::unordered_map<TensorId, BufferHandle> tensor_buffers_;
};

}  // namespace granite
