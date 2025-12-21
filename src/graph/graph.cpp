#include "granite/graph.h"
#include "granite/log.h"

#include <algorithm>
#include <queue>
#include <sstream>
#include <stack>

namespace granite {

// =============================================================================
// TensorDesc
// =============================================================================

size_t TensorDesc::numel() const {
    // Empty shape = scalar tensor with 1 element
    if (shape.empty()) return 1;
    size_t n = 1;
    for (auto s : shape) n *= static_cast<size_t>(s);
    return n;
}

size_t TensorDesc::size_bytes() const {
    return numel() * dtype_size(dtype);
}

TensorDesc TensorDesc::create(const std::string& name,
                              std::vector<int64_t> shape,
                              DataType dtype) {
    TensorDesc desc;
    desc.name = name;
    desc.shape = std::move(shape);
    desc.dtype = dtype;
    desc.layout = MemoryLayout::RowMajor;
    return desc;
}

// =============================================================================
// Graph
// =============================================================================

Graph::Graph(const std::string& name) : name_(name) {}

TensorId Graph::add_tensor(TensorDesc desc) {
    TensorId id = static_cast<TensorId>(tensors_.size());
    tensors_.push_back(std::move(desc));
    return id;
}

NodeId Graph::add_node(OpType op,
                       std::vector<TensorId> inputs,
                       std::vector<TensorId> outputs,
                       Attributes attrs) {
    NodeId id = static_cast<NodeId>(nodes_.size());

    Node node;
    node.id = id;
    node.op = op;
    node.name = fmt::format("{}_{}", op_type_name(op), id);
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    node.attrs = std::move(attrs);

    // Update producer/consumer mappings
    for (TensorId out : node.outputs) {
        tensor_producer_[out] = id;
    }
    for (TensorId in : node.inputs) {
        tensor_consumers_[in].push_back(id);
    }

    nodes_.push_back(std::move(node));
    return id;
}

void Graph::set_inputs(std::vector<TensorId> inputs) {
    input_tensors_ = std::move(inputs);
}

void Graph::set_outputs(std::vector<TensorId> outputs) {
    output_tensors_ = std::move(outputs);
}

const Node& Graph::node(NodeId id) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("Invalid node ID");
    }
    return nodes_[id];
}

Node& Graph::node(NodeId id) {
    if (id >= nodes_.size()) {
        throw std::out_of_range("Invalid node ID");
    }
    return nodes_[id];
}

const TensorDesc& Graph::tensor(TensorId id) const {
    if (id >= tensors_.size()) {
        throw std::out_of_range("Invalid tensor ID");
    }
    return tensors_[id];
}

TensorDesc& Graph::tensor(TensorId id) {
    if (id >= tensors_.size()) {
        throw std::out_of_range("Invalid tensor ID");
    }
    return tensors_[id];
}

NodeId Graph::producer(TensorId tensor_id) const {
    auto it = tensor_producer_.find(tensor_id);
    return (it != tensor_producer_.end()) ? it->second : INVALID_NODE_ID;
}

std::vector<NodeId> Graph::consumers(TensorId tensor_id) const {
    auto it = tensor_consumers_.find(tensor_id);
    return (it != tensor_consumers_.end()) ? it->second : std::vector<NodeId>{};
}

Result<std::vector<NodeId>> Graph::topological_sort() const {
    if (nodes_.empty()) {
        return std::vector<NodeId>{};
    }

    // Kahn's algorithm
    std::vector<NodeId> sorted;
    sorted.reserve(nodes_.size());

    // Count incoming edges (dependencies) for each node
    std::vector<size_t> in_degree(nodes_.size(), 0);
    std::vector<std::vector<NodeId>> adjacency(nodes_.size());

    // Build adjacency list based on tensor dependencies
    for (const auto& node : nodes_) {
        for (TensorId input : node.inputs) {
            NodeId producer_id = producer(input);
            if (producer_id != INVALID_NODE_ID && producer_id != node.id) {
                adjacency[producer_id].push_back(node.id);
                in_degree[node.id]++;
            }
        }
    }

    // Find all nodes with no dependencies
    std::queue<NodeId> ready;
    for (NodeId i = 0; i < nodes_.size(); i++) {
        if (in_degree[i] == 0) {
            ready.push(i);
        }
    }

    // Process nodes in topological order
    while (!ready.empty()) {
        NodeId current = ready.front();
        ready.pop();
        sorted.push_back(current);

        for (NodeId successor : adjacency[current]) {
            in_degree[successor]--;
            if (in_degree[successor] == 0) {
                ready.push(successor);
            }
        }
    }

    // Check for cycles
    if (sorted.size() != nodes_.size()) {
        return Error(ErrorCode::InvalidGraph, "Graph contains a cycle");
    }

    return sorted;
}

bool Graph::has_cycle() const {
    auto result = topological_sort();
    return !result.ok();
}

Result<void> Graph::validate() const {
    // Check for empty graph
    if (nodes_.empty()) {
        return Error(ErrorCode::InvalidGraph, "Graph has no nodes");
    }

    // Check that all inputs are defined
    if (input_tensors_.empty()) {
        return Error(ErrorCode::InvalidGraph, "Graph has no input tensors");
    }

    // Check that all outputs are defined
    if (output_tensors_.empty()) {
        return Error(ErrorCode::InvalidGraph, "Graph has no output tensors");
    }

    // Check for cycles
    auto sorted = topological_sort();
    if (!sorted.ok()) {
        return sorted.error();
    }

    // Check that all tensor IDs are valid
    for (const auto& node : nodes_) {
        for (TensorId id : node.inputs) {
            if (id >= tensors_.size()) {
                return Error(ErrorCode::InvalidGraph,
                    fmt::format("Node {} references invalid input tensor {}", node.id, id));
            }
        }
        for (TensorId id : node.outputs) {
            if (id >= tensors_.size()) {
                return Error(ErrorCode::InvalidGraph,
                    fmt::format("Node {} references invalid output tensor {}", node.id, id));
            }
        }
    }

    // Check that output tensors are actually produced
    for (TensorId out : output_tensors_) {
        if (producer(out) == INVALID_NODE_ID) {
            // Could be a graph input that's also an output (passthrough)
            bool is_input = std::find(input_tensors_.begin(), input_tensors_.end(), out)
                           != input_tensors_.end();
            if (!is_input) {
                return Error(ErrorCode::InvalidGraph,
                    fmt::format("Output tensor {} has no producer", out));
            }
        }
    }

    return {};
}

std::string Graph::to_string() const {
    std::ostringstream ss;
    ss << "Graph: " << name_ << "\n";
    ss << "  Tensors: " << tensors_.size() << "\n";
    ss << "  Nodes: " << nodes_.size() << "\n";
    ss << "  Inputs: [";
    for (size_t i = 0; i < input_tensors_.size(); i++) {
        if (i > 0) ss << ", ";
        ss << tensors_[input_tensors_[i]].name;
    }
    ss << "]\n";
    ss << "  Outputs: [";
    for (size_t i = 0; i < output_tensors_.size(); i++) {
        if (i > 0) ss << ", ";
        ss << tensors_[output_tensors_[i]].name;
    }
    ss << "]\n\n";

    ss << "  Nodes:\n";
    for (const auto& node : nodes_) {
        ss << "    [" << node.id << "] " << op_type_name(node.op) << ": ";
        ss << "(";
        for (size_t i = 0; i < node.inputs.size(); i++) {
            if (i > 0) ss << ", ";
            ss << tensors_[node.inputs[i]].name;
        }
        ss << ") -> (";
        for (size_t i = 0; i < node.outputs.size(); i++) {
            if (i > 0) ss << ", ";
            ss << tensors_[node.outputs[i]].name;
        }
        ss << ")\n";
    }

    return ss.str();
}

void Graph::dump() const {
    GRANITE_LOG_INFO("{}", to_string());
}

// =============================================================================
// GraphBuilder
// =============================================================================

GraphBuilder::GraphBuilder(const std::string& name) : graph_(name) {}

std::string GraphBuilder::generate_name(const std::string& prefix) {
    return fmt::format("{}_{}", prefix, tensor_counter_++);
}

TensorId GraphBuilder::add_intermediate(const std::string& name,
                                        const std::vector<int64_t>& shape,
                                        DataType dtype) {
    std::string tensor_name = name.empty() ? generate_name("tensor") : name;
    return graph_.add_tensor(TensorDesc::create(tensor_name, shape, dtype));
}

TensorId GraphBuilder::input(const std::string& name,
                             std::vector<int64_t> shape,
                             DataType dtype) {
    TensorId id = graph_.add_tensor(TensorDesc::create(name, std::move(shape), dtype));
    // Input tensors are tracked separately; set_inputs called in build()
    return id;
}

TensorId GraphBuilder::add(TensorId a, TensorId b, const std::string& name) {
    const auto& shape_a = graph_.tensor(a).shape;
    auto out = add_intermediate(name.empty() ? generate_name("add") : name,
                               shape_a, graph_.tensor(a).dtype);
    (void)graph_.add_node(OpType::Add, {a, b}, {out});
    return out;
}

TensorId GraphBuilder::sub(TensorId a, TensorId b, const std::string& name) {
    const auto& shape_a = graph_.tensor(a).shape;
    auto out = add_intermediate(name.empty() ? generate_name("sub") : name,
                               shape_a, graph_.tensor(a).dtype);
    (void)graph_.add_node(OpType::Sub, {a, b}, {out});
    return out;
}

TensorId GraphBuilder::mul(TensorId a, TensorId b, const std::string& name) {
    const auto& shape_a = graph_.tensor(a).shape;
    auto out = add_intermediate(name.empty() ? generate_name("mul") : name,
                               shape_a, graph_.tensor(a).dtype);
    (void)graph_.add_node(OpType::Mul, {a, b}, {out});
    return out;
}

TensorId GraphBuilder::div(TensorId a, TensorId b, const std::string& name) {
    const auto& shape_a = graph_.tensor(a).shape;
    auto out = add_intermediate(name.empty() ? generate_name("div") : name,
                               shape_a, graph_.tensor(a).dtype);
    (void)graph_.add_node(OpType::Div, {a, b}, {out});
    return out;
}

TensorId GraphBuilder::matmul(TensorId a, TensorId b, const std::string& name) {
    const auto& shape_a = graph_.tensor(a).shape;
    const auto& shape_b = graph_.tensor(b).shape;

    // Output shape: [M, N] for [M, K] x [K, N]
    std::vector<int64_t> out_shape;
    if (shape_a.size() >= 2 && shape_b.size() >= 2) {
        out_shape = {shape_a[shape_a.size() - 2], shape_b[shape_b.size() - 1]};
    }

    auto out = add_intermediate(name.empty() ? generate_name("matmul") : name,
                               out_shape, graph_.tensor(a).dtype);
    (void)graph_.add_node(OpType::MatMul, {a, b}, {out});
    return out;
}

TensorId GraphBuilder::relu(TensorId x, const std::string& name) {
    const auto& shape = graph_.tensor(x).shape;
    auto out = add_intermediate(name.empty() ? generate_name("relu") : name,
                               shape, graph_.tensor(x).dtype);
    (void)graph_.add_node(OpType::ReLU, {x}, {out});
    return out;
}

TensorId GraphBuilder::gelu(TensorId x, const std::string& name) {
    const auto& shape = graph_.tensor(x).shape;
    auto out = add_intermediate(name.empty() ? generate_name("gelu") : name,
                               shape, graph_.tensor(x).dtype);
    (void)graph_.add_node(OpType::GELU, {x}, {out});
    return out;
}

TensorId GraphBuilder::silu(TensorId x, const std::string& name) {
    const auto& shape = graph_.tensor(x).shape;
    auto out = add_intermediate(name.empty() ? generate_name("silu") : name,
                               shape, graph_.tensor(x).dtype);
    (void)graph_.add_node(OpType::SiLU, {x}, {out});
    return out;
}

TensorId GraphBuilder::softmax(TensorId x, int axis, const std::string& name) {
    const auto& shape = graph_.tensor(x).shape;
    auto out = add_intermediate(name.empty() ? generate_name("softmax") : name,
                               shape, graph_.tensor(x).dtype);

    Attributes attrs;
    attrs.set("axis", static_cast<int64_t>(axis));
    (void)graph_.add_node(OpType::Softmax, {x}, {out}, std::move(attrs));
    return out;
}

TensorId GraphBuilder::layer_norm(TensorId x, TensorId weight, TensorId bias,
                                  float eps, const std::string& name) {
    const auto& shape = graph_.tensor(x).shape;
    auto out = add_intermediate(name.empty() ? generate_name("layer_norm") : name,
                               shape, graph_.tensor(x).dtype);

    Attributes attrs;
    attrs.set("eps", static_cast<double>(eps));
    (void)graph_.add_node(OpType::LayerNorm, {x, weight, bias}, {out}, std::move(attrs));
    return out;
}

TensorId GraphBuilder::rms_norm(TensorId x, TensorId weight,
                                float eps, const std::string& name) {
    const auto& shape = graph_.tensor(x).shape;
    auto out = add_intermediate(name.empty() ? generate_name("rms_norm") : name,
                               shape, graph_.tensor(x).dtype);

    Attributes attrs;
    attrs.set("eps", static_cast<double>(eps));
    (void)graph_.add_node(OpType::RMSNorm, {x, weight}, {out}, std::move(attrs));
    return out;
}

void GraphBuilder::mark_output(TensorId tensor) {
    outputs_.push_back(tensor);
}

Result<Graph> GraphBuilder::build() {
    if (outputs_.empty()) {
        return Error(ErrorCode::InvalidGraph, "No output tensors marked");
    }

    // Collect input tensors (tensors with no producer)
    std::vector<TensorId> inputs;
    for (TensorId i = 0; i < graph_.num_tensors(); i++) {
        if (graph_.producer(i) == INVALID_NODE_ID) {
            inputs.push_back(i);
        }
    }

    if (inputs.empty()) {
        return Error(ErrorCode::InvalidGraph, "No input tensors found");
    }

    graph_.set_inputs(std::move(inputs));
    graph_.set_outputs(outputs_);

    // Validate the graph
    auto validate_result = graph_.validate();
    if (!validate_result.ok()) {
        return validate_result.error();
    }

    return std::move(graph_);
}

// =============================================================================
// ExecutionPlan
// =============================================================================

void ExecutionPlan::add_step(ExecutionStep step) {
    steps_.push_back(std::move(step));
}

void ExecutionPlan::set_input_buffers(std::vector<BufferHandle> buffers) {
    input_buffers_ = std::move(buffers);
}

void ExecutionPlan::set_output_buffers(std::vector<BufferHandle> buffers) {
    output_buffers_ = std::move(buffers);
}

// =============================================================================
// GraphExecutor
// =============================================================================

GraphExecutor::GraphExecutor(IComputeBackend* backend)
    : backend_(backend) {}

GraphExecutor::~GraphExecutor() {
    // Release any allocated buffers
    for (auto& [tensor_id, buffer] : tensor_buffers_) {
        if (buffer.id != 0) {
            backend_->destroy_buffer(buffer);
        }
    }
}

Result<void> GraphExecutor::compile(const Graph& graph) {
    // Validate graph first
    GRANITE_TRY(graph.validate());

    // Get execution order
    auto sort_result = graph.topological_sort();
    if (!sort_result.ok()) {
        return sort_result.error();
    }
    auto order = std::move(sort_result).take();

    plan_ = std::make_unique<ExecutionPlan>();
    operators_.clear();
    tensor_buffers_.clear();

    auto& registry = OperatorRegistry::instance();
    BackendType backend_type = backend_->get_type();

    // Allocate buffers for all tensors
    size_t total_memory = 0;
    for (TensorId i = 0; i < graph.num_tensors(); i++) {
        const auto& tensor_desc = graph.tensor(i);
        size_t size = tensor_desc.size_bytes();

        auto buffer_result = backend_->create_buffer(BufferDesc::shared(size));
        if (!buffer_result.ok()) {
            return buffer_result.error();
        }
        tensor_buffers_[i] = buffer_result.value();
        total_memory += size;
    }

    plan_->set_total_memory(total_memory);

    // Create execution steps
    for (NodeId node_id : order) {
        const auto& node = graph.node(node_id);

        // Skip fused nodes
        if (node.is_fused) continue;

        // Create operator
        auto op = registry.create(node.op, backend_type);
        if (!op) {
            return Error(ErrorCode::OperatorNotImplemented,
                fmt::format("{} not implemented for {}",
                           op_type_name(node.op), backend_name(backend_type)));
        }

        ExecutionStep step;
        step.node_id = node_id;
        step.op = node.op;

        // Map tensor IDs to buffers
        for (TensorId in : node.inputs) {
            step.input_buffers.push_back(tensor_buffers_[in]);
        }
        for (TensorId out : node.outputs) {
            step.output_buffers.push_back(tensor_buffers_[out]);
        }

        step.op_impl = op.get();
        operators_.push_back(std::move(op));

        plan_->add_step(std::move(step));
    }

    // Set graph input/output buffers
    std::vector<BufferHandle> input_buffers;
    for (TensorId id : graph.inputs()) {
        input_buffers.push_back(tensor_buffers_[id]);
    }
    plan_->set_input_buffers(std::move(input_buffers));

    std::vector<BufferHandle> output_buffers;
    for (TensorId id : graph.outputs()) {
        output_buffers.push_back(tensor_buffers_[id]);
    }
    plan_->set_output_buffers(std::move(output_buffers));


    return {};
}

Result<std::vector<Tensor>> GraphExecutor::execute(const std::vector<Tensor>& inputs) {
    if (!plan_) {
        return Error(ErrorCode::InvalidState, "Graph not compiled");
    }

    // TODO: Copy input data to input buffers
    // TODO: Execute each step
    // TODO: Create output tensors from output buffers

    return Error(ErrorCode::NotImplemented, "GraphExecutor::execute not yet implemented");
}

size_t GraphExecutor::memory_usage() const {
    return plan_ ? plan_->total_memory_bytes() : 0;
}

}  // namespace granite
