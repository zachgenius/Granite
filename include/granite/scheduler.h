#pragma once

#include "granite/types.h"
#include "granite/error.h"
#include "granite/graph.h"
#include "granite/memory.h"

#include <memory>
#include <vector>
#include <unordered_map>

namespace granite {

// Forward declarations
class IComputeBackend;

// =============================================================================
// Scheduler Configuration
// =============================================================================

struct SchedulerConfig {
    // Memory budget (0 = unlimited)
    size_t memory_budget = 0;

    // Enable buffer aliasing for intermediate tensors
    bool enable_aliasing = true;

    // Enable operator fusion (future)
    bool enable_fusion = false;

    // Prefetch buffers for upcoming operations
    bool enable_prefetch = false;
};

// =============================================================================
// Scheduled Operation
// =============================================================================

struct ScheduledOp {
    NodeId node_id = INVALID_NODE_ID;
    OpType op_type = OpType::Custom;
    std::string name;

    // Buffer indices for inputs/outputs (into plan's buffer list)
    std::vector<size_t> input_indices;
    std::vector<size_t> output_indices;

    // Operator attributes
    Attributes attrs;

    // Dependencies (other scheduled ops that must complete first)
    std::vector<size_t> dependencies;
};

// =============================================================================
// Buffer Allocation Info
// =============================================================================

struct BufferAllocation {
    TensorId tensor_id = INVALID_TENSOR_ID;
    size_t size = 0;
    MemoryType memory_type = MemoryType::Device;

    // Liveness info
    uint32_t first_use = 0;
    uint32_t last_use = 0;

    // Aliasing (which other allocations share physical memory)
    int32_t alias_group = -1;  // -1 = no aliasing

    // The actual buffer handle (set during execution)
    BufferHandle handle;
};

// =============================================================================
// Compiled Execution Plan
// =============================================================================

class CompiledPlan {
public:
    CompiledPlan() = default;

    // Scheduled operations in execution order
    [[nodiscard]] const std::vector<ScheduledOp>& operations() const { return ops_; }

    // Buffer allocations needed
    [[nodiscard]] const std::vector<BufferAllocation>& allocations() const { return allocations_; }

    // Input/output buffer indices
    [[nodiscard]] const std::vector<size_t>& input_indices() const { return input_indices_; }
    [[nodiscard]] const std::vector<size_t>& output_indices() const { return output_indices_; }

    // Statistics
    [[nodiscard]] size_t num_operations() const { return ops_.size(); }
    [[nodiscard]] size_t num_buffers() const { return allocations_.size(); }
    [[nodiscard]] size_t total_memory_required() const;
    [[nodiscard]] size_t peak_memory_required() const;

    // Debug output
    [[nodiscard]] std::string to_string() const;
    void dump() const;

    // Builder methods (used by Scheduler)
    void add_operation(ScheduledOp op);
    void add_allocation(BufferAllocation alloc);
    void set_input_indices(std::vector<size_t> indices);
    void set_output_indices(std::vector<size_t> indices);
    void set_peak_memory(size_t bytes) { peak_memory_ = bytes; }

private:
    std::vector<ScheduledOp> ops_;
    std::vector<BufferAllocation> allocations_;
    std::vector<size_t> input_indices_;
    std::vector<size_t> output_indices_;
    size_t peak_memory_ = 0;
};

// =============================================================================
// Scheduler
// =============================================================================

class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config = {});

    /// Compile a graph into an execution plan
    [[nodiscard]] Result<CompiledPlan> compile(const Graph& graph);

    /// Get/set configuration
    [[nodiscard]] const SchedulerConfig& config() const { return config_; }
    void set_config(const SchedulerConfig& config) { config_ = config; }

private:
    SchedulerConfig config_;

    // Internal compilation stages
    Result<std::vector<NodeId>> compute_execution_order(const Graph& graph);
    Result<std::vector<BufferAllocation>> compute_allocations(
        const Graph& graph,
        const std::vector<NodeId>& order);
    void compute_liveness(
        std::vector<BufferAllocation>& allocations,
        const Graph& graph,
        const std::vector<NodeId>& order);
    void compute_aliasing(std::vector<BufferAllocation>& allocations);
};

// =============================================================================
// Runtime Executor
// =============================================================================

class RuntimeExecutor {
public:
    RuntimeExecutor(IComputeBackend* backend, MemoryManager* memory_manager);
    ~RuntimeExecutor();

    /// Execute a compiled plan with the given inputs
    [[nodiscard]] Result<std::vector<Tensor>> execute(
        const CompiledPlan& plan,
        const std::vector<Tensor>& inputs);

    /// Get memory usage of last execution
    [[nodiscard]] size_t last_execution_memory() const { return last_execution_memory_; }

private:
    IComputeBackend* backend_;
    MemoryManager* memory_manager_;
    size_t last_execution_memory_ = 0;

    // Allocate buffers according to plan
    Result<std::vector<BufferHandle>> allocate_buffers(
        const CompiledPlan& plan,
        const std::vector<Tensor>& inputs);

    // Execute a single operation
    Result<void> execute_op(
        const ScheduledOp& op,
        const std::vector<BufferHandle>& buffers);
};

}  // namespace granite
