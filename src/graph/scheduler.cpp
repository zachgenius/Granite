#include "granite/scheduler.h"
#include "granite/backend.h"
#include "granite/tensor.h"
#include "granite/log.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace granite {

// =============================================================================
// CompiledPlan Implementation
// =============================================================================

size_t CompiledPlan::total_memory_required() const {
    size_t total = 0;
    for (const auto& alloc : allocations_) {
        total += alloc.size;
    }
    return total;
}

size_t CompiledPlan::peak_memory_required() const {
    return peak_memory_;
}

std::string CompiledPlan::to_string() const {
    std::ostringstream ss;
    ss << "CompiledPlan:\n";
    ss << "  Operations: " << ops_.size() << "\n";
    ss << "  Buffers: " << allocations_.size() << "\n";
    ss << "  Total memory: " << total_memory_required() << " bytes\n";
    ss << "  Peak memory: " << peak_memory_ << " bytes\n\n";

    ss << "  Operations:\n";
    for (size_t i = 0; i < ops_.size(); i++) {
        const auto& op = ops_[i];
        ss << "    [" << i << "] " << op_type_name(op.op_type);
        ss << " (" << op.name << ")\n";
        ss << "        inputs: [";
        for (size_t j = 0; j < op.input_indices.size(); j++) {
            if (j > 0) ss << ", ";
            ss << op.input_indices[j];
        }
        ss << "] -> outputs: [";
        for (size_t j = 0; j < op.output_indices.size(); j++) {
            if (j > 0) ss << ", ";
            ss << op.output_indices[j];
        }
        ss << "]\n";
    }

    ss << "\n  Buffers:\n";
    for (size_t i = 0; i < allocations_.size(); i++) {
        const auto& alloc = allocations_[i];
        ss << "    [" << i << "] " << alloc.size << " bytes";
        ss << " (live: " << alloc.first_use << "-" << alloc.last_use << ")";
        if (alloc.alias_group >= 0) {
            ss << " [alias group " << alloc.alias_group << "]";
        }
        ss << "\n";
    }

    return ss.str();
}

void CompiledPlan::dump() const {
    GRANITE_LOG_INFO("{}", to_string());
}

void CompiledPlan::add_operation(ScheduledOp op) {
    ops_.push_back(std::move(op));
}

void CompiledPlan::add_allocation(BufferAllocation alloc) {
    allocations_.push_back(std::move(alloc));
}

void CompiledPlan::set_input_indices(std::vector<size_t> indices) {
    input_indices_ = std::move(indices);
}

void CompiledPlan::set_output_indices(std::vector<size_t> indices) {
    output_indices_ = std::move(indices);
}

// =============================================================================
// Scheduler Implementation
// =============================================================================

Scheduler::Scheduler(const SchedulerConfig& config)
    : config_(config)
{
}

Result<CompiledPlan> Scheduler::compile(const Graph& graph) {
    // Validate graph first
    auto validation_result = graph.validate();
    if (!validation_result.ok()) {
        return validation_result.error();
    }

    // Get execution order
    auto order_result = compute_execution_order(graph);
    if (!order_result.ok()) {
        return order_result.error();
    }
    auto order = std::move(order_result).take();

    // Compute buffer allocations
    auto alloc_result = compute_allocations(graph, order);
    if (!alloc_result.ok()) {
        return alloc_result.error();
    }
    auto allocations = std::move(alloc_result).take();

    // Compute liveness for each allocation
    compute_liveness(allocations, graph, order);

    // Compute aliasing opportunities
    if (config_.enable_aliasing) {
        compute_aliasing(allocations);
    }

    // Build the plan
    CompiledPlan plan;

    // Add allocations
    for (auto& alloc : allocations) {
        plan.add_allocation(std::move(alloc));
    }

    // Build tensor ID to allocation index map
    std::unordered_map<TensorId, size_t> tensor_to_alloc;
    for (size_t i = 0; i < plan.allocations().size(); i++) {
        tensor_to_alloc[plan.allocations()[i].tensor_id] = i;
    }

    // Build scheduled operations
    for (size_t step = 0; step < order.size(); step++) {
        NodeId node_id = order[step];
        const Node& node = graph.node(node_id);

        ScheduledOp op;
        op.node_id = node_id;
        op.op_type = node.op;
        op.name = node.name;
        op.attrs = node.attrs;

        // Map tensor IDs to allocation indices
        for (TensorId input : node.inputs) {
            auto it = tensor_to_alloc.find(input);
            if (it != tensor_to_alloc.end()) {
                op.input_indices.push_back(it->second);
            }
        }

        for (TensorId output : node.outputs) {
            auto it = tensor_to_alloc.find(output);
            if (it != tensor_to_alloc.end()) {
                op.output_indices.push_back(it->second);
            }
        }

        // Compute dependencies (previous ops that produce our inputs)
        std::unordered_set<size_t> deps;
        for (TensorId input : node.inputs) {
            NodeId producer = graph.producer(input);
            if (producer != INVALID_NODE_ID) {
                // Find this producer's position in the execution order
                for (size_t i = 0; i < step; i++) {
                    if (order[i] == producer) {
                        deps.insert(i);
                        break;
                    }
                }
            }
        }
        op.dependencies.assign(deps.begin(), deps.end());
        std::sort(op.dependencies.begin(), op.dependencies.end());

        plan.add_operation(std::move(op));
    }

    // Set input/output indices
    std::vector<size_t> input_indices;
    for (TensorId input : graph.inputs()) {
        auto it = tensor_to_alloc.find(input);
        if (it != tensor_to_alloc.end()) {
            input_indices.push_back(it->second);
        }
    }
    plan.set_input_indices(std::move(input_indices));

    std::vector<size_t> output_indices;
    for (TensorId output : graph.outputs()) {
        auto it = tensor_to_alloc.find(output);
        if (it != tensor_to_alloc.end()) {
            output_indices.push_back(it->second);
        }
    }
    plan.set_output_indices(std::move(output_indices));

    // Compute peak memory
    size_t peak = 0;
    for (uint32_t step = 0; step < static_cast<uint32_t>(order.size()); step++) {
        size_t live = 0;
        for (const auto& alloc : plan.allocations()) {
            if (step >= alloc.first_use && step <= alloc.last_use) {
                live += alloc.size;
            }
        }
        peak = std::max(peak, live);
    }
    plan.set_peak_memory(peak);

    GRANITE_LOG_DEBUG("Compiled plan: {} ops, {} buffers, {} peak bytes",
                      plan.num_operations(), plan.num_buffers(),
                      plan.peak_memory_required());

    return plan;
}

Result<std::vector<NodeId>> Scheduler::compute_execution_order(const Graph& graph) {
    return graph.topological_sort();
}

Result<std::vector<BufferAllocation>> Scheduler::compute_allocations(
    const Graph& graph,
    const std::vector<NodeId>& order)
{
    std::vector<BufferAllocation> allocations;
    allocations.reserve(graph.num_tensors());

    for (size_t i = 0; i < graph.num_tensors(); i++) {
        TensorId tid = static_cast<TensorId>(i);
        const auto& desc = graph.tensor(tid);

        BufferAllocation alloc;
        alloc.tensor_id = tid;
        alloc.size = desc.size_bytes();

        // Determine memory type based on usage
        // Graph inputs: Shared (for CPU upload)
        // Graph outputs: Shared (for CPU readback)
        // Intermediates: Device (GPU-only)
        bool is_input = std::find(graph.inputs().begin(), graph.inputs().end(), tid)
                        != graph.inputs().end();
        bool is_output = std::find(graph.outputs().begin(), graph.outputs().end(), tid)
                         != graph.outputs().end();

        if (is_input || is_output) {
            alloc.memory_type = MemoryType::Shared;
        } else {
            alloc.memory_type = MemoryType::Device;
        }

        allocations.push_back(alloc);
    }

    return allocations;
}

void Scheduler::compute_liveness(
    std::vector<BufferAllocation>& allocations,
    const Graph& graph,
    const std::vector<NodeId>& order)
{
    // Initialize liveness to invalid
    for (auto& alloc : allocations) {
        alloc.first_use = UINT32_MAX;
        alloc.last_use = 0;
    }

    // Scan through execution order
    for (size_t step = 0; step < order.size(); step++) {
        const Node& node = graph.node(order[step]);

        // Update liveness for inputs
        for (TensorId input : node.inputs) {
            auto& alloc = allocations[input];
            alloc.first_use = std::min(alloc.first_use, static_cast<uint32_t>(step));
            alloc.last_use = std::max(alloc.last_use, static_cast<uint32_t>(step));
        }

        // Update liveness for outputs
        for (TensorId output : node.outputs) {
            auto& alloc = allocations[output];
            alloc.first_use = std::min(alloc.first_use, static_cast<uint32_t>(step));
            alloc.last_use = std::max(alloc.last_use, static_cast<uint32_t>(step));
        }
    }

    // Graph inputs are live from step 0
    for (TensorId input : graph.inputs()) {
        allocations[input].first_use = 0;
    }

    // Graph outputs are live until the end
    for (TensorId output : graph.outputs()) {
        allocations[output].last_use = static_cast<uint32_t>(order.size());
    }
}

void Scheduler::compute_aliasing(std::vector<BufferAllocation>& allocations) {
    // Build intervals for intermediate buffers (not inputs/outputs)
    std::vector<LivenessInterval> intervals;

    for (size_t i = 0; i < allocations.size(); i++) {
        auto& alloc = allocations[i];
        // Only alias device buffers (intermediates)
        if (alloc.memory_type == MemoryType::Device) {
            intervals.push_back({
                static_cast<uint32_t>(i),  // buffer_id = allocation index
                alloc.first_use,
                alloc.last_use,
                alloc.size
            });
        }
    }

    if (intervals.empty()) return;

    // Compute aliasing groups
    auto groups = compute_aliasing_groups(intervals);

    // Assign alias groups
    // Note: groups contain buffer_id values which are the actual allocation indices
    int32_t group_id = 0;
    for (const auto& group : groups) {
        for (uint32_t alloc_idx : group) {
            // alloc_idx is the actual allocation index (from interval.buffer_id)
            allocations[alloc_idx].alias_group = group_id;
        }
        group_id++;
    }
}

// =============================================================================
// RuntimeExecutor Implementation
// =============================================================================

RuntimeExecutor::RuntimeExecutor(IComputeBackend* backend, MemoryManager* memory_manager)
    : backend_(backend)
    , memory_manager_(memory_manager)
{
}

RuntimeExecutor::~RuntimeExecutor() = default;

Result<std::vector<Tensor>> RuntimeExecutor::execute(
    const CompiledPlan& plan,
    const std::vector<Tensor>& inputs)
{
    // Validate inputs
    if (inputs.size() != plan.input_indices().size()) {
        GRANITE_FAIL(ErrorCode::InvalidArgument,
                     fmt::format("Expected {} inputs, got {}",
                                 plan.input_indices().size(), inputs.size()));
    }

    // Allocate buffers
    auto alloc_result = allocate_buffers(plan, inputs);
    if (!alloc_result.ok()) {
        return alloc_result.error();
    }
    auto buffers = std::move(alloc_result).take();

    // Copy input data to buffers
    for (size_t i = 0; i < inputs.size(); i++) {
        size_t alloc_idx = plan.input_indices()[i];
        const auto& input_tensor = inputs[i];

        // Get the input buffer handle directly
        BufferHandle src_buffer = input_tensor.buffer();
        if (!src_buffer.valid()) {
            memory_manager_->release_plan(buffers);
            GRANITE_FAIL(ErrorCode::InvalidArgument, "Input tensor has invalid buffer");
        }

        // Copy from input tensor's buffer to execution buffer
        auto copy_result = backend_->copy_buffer(
            src_buffer,
            buffers[alloc_idx],
            input_tensor.size_bytes());

        if (!copy_result.ok()) {
            memory_manager_->release_plan(buffers);
            return copy_result.error();
        }
    }

    // Execute operations
    for (const auto& op : plan.operations()) {
        auto exec_result = execute_op(op, buffers);
        if (!exec_result.ok()) {
            memory_manager_->release_plan(buffers);
            return exec_result.error();
        }
    }

    // Wait for completion
    auto wait_result = backend_->wait_for_completion();
    if (!wait_result.ok()) {
        memory_manager_->release_plan(buffers);
        return wait_result.error();
    }

    // Read output data
    std::vector<Tensor> outputs;
    outputs.reserve(plan.output_indices().size());

    for (size_t alloc_idx : plan.output_indices()) {
        const auto& alloc = plan.allocations()[alloc_idx];

        // Create output tensor
        // Note: We'd need to get shape/dtype from the plan
        // For now, create a raw buffer
        std::vector<uint8_t> data(alloc.size);

        auto read_result = backend_->read_buffer(
            buffers[alloc_idx],
            data.data(),
            alloc.size);

        if (!read_result.ok()) {
            memory_manager_->release_plan(buffers);
            return read_result.error();
        }

        // Create tensor from data
        // This is a simplified version - in practice we'd use the tensor metadata
        Tensor output;
        // TODO: Properly construct output tensor with shape/dtype
        outputs.push_back(std::move(output));
    }

    // Track memory usage
    last_execution_memory_ = plan.peak_memory_required();

    // Release buffers (return to pool)
    memory_manager_->release_plan(buffers);

    return outputs;
}

Result<std::vector<BufferHandle>> RuntimeExecutor::allocate_buffers(
    const CompiledPlan& plan,
    const std::vector<Tensor>& inputs)
{
    // Build buffer requests
    std::vector<BufferRequest> requests;
    requests.reserve(plan.allocations().size());

    for (const auto& alloc : plan.allocations()) {
        BufferRequest req;
        req.size = alloc.size;
        req.memory_type = alloc.memory_type;
        req.first_use = alloc.first_use;
        req.last_use = alloc.last_use;
        req.allow_aliasing = (alloc.alias_group >= 0);
        requests.push_back(req);
    }

    // Use memory manager to allocate with potential aliasing
    return memory_manager_->plan_allocations(requests);
}

Result<void> RuntimeExecutor::execute_op(
    const ScheduledOp& op,
    const std::vector<BufferHandle>& buffers)
{
    // This is a placeholder - actual implementation would:
    // 1. Get operator from registry
    // 2. Bind input/output buffers
    // 3. Execute the operator

    GRANITE_LOG_TRACE("Executing op: {} ({})", op.name, op_type_name(op.op_type));

    // For now, just return success
    // Real implementation would dispatch to appropriate operator
    return {};
}

}  // namespace granite
