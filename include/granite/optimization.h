#pragma once

#include "granite/graph.h"
#include "granite/error.h"

#include <memory>
#include <string>
#include <vector>

namespace granite {

// =============================================================================
// Optimization Pass Interface
// =============================================================================

class OptimizationPass {
public:
    virtual ~OptimizationPass() = default;

    /// Get the name of this pass
    [[nodiscard]] virtual const char* name() const = 0;

    /// Run the optimization pass on the graph
    /// Returns true if the graph was modified
    [[nodiscard]] virtual Result<bool> run(Graph& graph) = 0;
};

// =============================================================================
// Built-in Optimization Passes
// =============================================================================

/// Dead code elimination - removes nodes whose outputs are never used
class DeadCodeEliminationPass : public OptimizationPass {
public:
    [[nodiscard]] const char* name() const override { return "DeadCodeElimination"; }
    [[nodiscard]] Result<bool> run(Graph& graph) override;
};

/// Constant folding - evaluates ops with constant inputs at compile time
/// (Currently a placeholder - requires runtime constant tracking)
class ConstantFoldingPass : public OptimizationPass {
public:
    [[nodiscard]] const char* name() const override { return "ConstantFolding"; }
    [[nodiscard]] Result<bool> run(Graph& graph) override;
};

/// Operator fusion - combines sequences of ops into fused kernels
/// Supported fusions:
/// - MatMul + Add (bias) -> FusedMatMulBias
/// - Linear ops (Add/Mul) + Activation -> FusedActivation
class OperatorFusionPass : public OptimizationPass {
public:
    [[nodiscard]] const char* name() const override { return "OperatorFusion"; }
    [[nodiscard]] Result<bool> run(Graph& graph) override;

private:
    bool try_fuse_matmul_bias(Graph& graph);
    bool try_fuse_linear_activation(Graph& graph);
};

/// Memory layout optimization - reorders operations for better cache utilization
class LayoutOptimizationPass : public OptimizationPass {
public:
    [[nodiscard]] const char* name() const override { return "LayoutOptimization"; }
    [[nodiscard]] Result<bool> run(Graph& graph) override;
};

// =============================================================================
// Optimization Pipeline
// =============================================================================

enum class OptimizationLevel {
    None,       // No optimizations
    Basic,      // Dead code elimination only
    Standard,   // DCE + constant folding
    Aggressive  // All optimizations including fusion
};

class OptimizationPipeline {
public:
    /// Create a default pipeline for the given optimization level
    static OptimizationPipeline create(OptimizationLevel level);

    /// Add a pass to the pipeline
    void add_pass(std::unique_ptr<OptimizationPass> pass);

    /// Run all passes on the graph
    /// Returns the number of passes that modified the graph
    [[nodiscard]] Result<int> run(Graph& graph);

    /// Run passes iteratively until no more changes (fixed-point)
    /// Returns total number of modifications
    [[nodiscard]] Result<int> run_until_fixed_point(Graph& graph, int max_iterations = 10);

    /// Get pass count
    [[nodiscard]] size_t num_passes() const { return passes_.size(); }

private:
    std::vector<std::unique_ptr<OptimizationPass>> passes_;
};

// =============================================================================
// Utility Functions
// =============================================================================

/// Check if a node is a no-op (output same as input)
bool is_identity_op(const Node& node);

/// Check if two ops can be fused together
bool can_fuse_ops(OpType first, OpType second);

/// Get the fused op type for a pair of ops
OpType get_fused_op_type(OpType first, OpType second);

}  // namespace granite
