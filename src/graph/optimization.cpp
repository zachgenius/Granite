#include "granite/optimization.h"
#include "granite/log.h"

#include <unordered_set>
#include <algorithm>

namespace granite {

// =============================================================================
// Dead Code Elimination Pass
// =============================================================================

Result<bool> DeadCodeEliminationPass::run(Graph& graph) {
    // Find all tensors that are actually used (reachable from outputs)
    std::unordered_set<TensorId> used_tensors;

    // Start with graph outputs
    for (TensorId output : graph.outputs()) {
        used_tensors.insert(output);
    }

    // Work backwards to find all required tensors
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& node : graph.nodes()) {
            // Check if any output is used
            bool has_used_output = false;
            for (TensorId output : node.outputs) {
                if (used_tensors.count(output) > 0) {
                    has_used_output = true;
                    break;
                }
            }

            // If this node produces used outputs, mark its inputs as used
            if (has_used_output) {
                for (TensorId input : node.inputs) {
                    if (used_tensors.count(input) == 0) {
                        used_tensors.insert(input);
                        changed = true;
                    }
                }
            }
        }
    }

    // Count dead nodes (nodes whose outputs are not used)
    int dead_count = 0;
    for (const auto& node : graph.nodes()) {
        bool all_outputs_dead = true;
        for (TensorId output : node.outputs) {
            if (used_tensors.count(output) > 0) {
                all_outputs_dead = false;
                break;
            }
        }
        if (all_outputs_dead && !node.outputs.empty()) {
            dead_count++;
        }
    }

    // Note: Actually removing nodes from the graph would require rebuilding it
    // For now, we just mark nodes as fused (disabled) and log the potential savings
    // A full implementation would create a new graph without the dead nodes

    if (dead_count > 0) {
        GRANITE_LOG_INFO("DCE: Found {} dead nodes (not removed in current implementation)",
                        dead_count);
    }

    return dead_count > 0;
}

// =============================================================================
// Constant Folding Pass
// =============================================================================

Result<bool> ConstantFoldingPass::run(Graph& graph) {
    // This is a placeholder implementation
    // Full implementation would:
    // 1. Track which tensors are constants (initialized with data)
    // 2. For nodes with all constant inputs, evaluate at compile time
    // 3. Replace the node with a constant tensor

    // For now, we just identify potential candidates
    int candidates = 0;

    for (const auto& node : graph.nodes()) {
        // Check if this is a compute op with small constant inputs
        // Common patterns:
        // - Reshape with constant shape
        // - Slice with constant indices
        // - Arithmetic with scalar constants

        // Currently no constants tracked in the graph, so this is a no-op
        (void)node;  // Suppress unused warning
    }

    if (candidates > 0) {
    }

    return false;  // No modifications in placeholder
}

// =============================================================================
// Operator Fusion Pass
// =============================================================================

Result<bool> OperatorFusionPass::run(Graph& graph) {
    bool modified = false;

    // Try different fusion patterns
    modified |= try_fuse_matmul_bias(graph);
    modified |= try_fuse_linear_activation(graph);

    return modified;
}

bool OperatorFusionPass::try_fuse_matmul_bias(Graph& graph) {
    // Look for pattern: MatMul -> Add (where Add's second input is bias)
    int fused = 0;

    for (size_t i = 0; i < graph.nodes().size(); i++) {
        Node& matmul_node = graph.node(static_cast<NodeId>(i));
        if (matmul_node.is_fused) continue;
        if (matmul_node.op != OpType::MatMul) continue;
        if (matmul_node.outputs.empty()) continue;

        TensorId matmul_output = matmul_node.outputs[0];

        // Find consumers of matmul output
        auto consumers = graph.consumers(matmul_output);
        if (consumers.size() != 1) continue;  // Must be single consumer

        Node& add_node = graph.node(consumers[0]);
        if (add_node.is_fused) continue;
        if (add_node.op != OpType::Add) continue;

        // Verify the matmul output is one of the Add inputs
        bool found = false;
        TensorId bias_tensor = INVALID_TENSOR_ID;
        for (TensorId input : add_node.inputs) {
            if (input == matmul_output) {
                found = true;
            } else {
                bias_tensor = input;
            }
        }

        if (!found || bias_tensor == INVALID_TENSOR_ID) continue;

        // Mark for fusion
        add_node.is_fused = true;
        add_node.fused_into = matmul_node.id;

        // Update matmul node to indicate it handles bias
        matmul_node.attrs.set("has_bias", true);
        matmul_node.attrs.set("bias_tensor", static_cast<int64_t>(bias_tensor));

        fused++;
    }

    if (fused > 0) {
        GRANITE_LOG_INFO("OperatorFusion: Fused {} MatMul+Bias patterns", fused);
    }

    return fused > 0;
}

bool OperatorFusionPass::try_fuse_linear_activation(Graph& graph) {
    // Look for patterns: Linear op -> Activation
    // Linear ops: Add, Mul, Sub, Div
    // Activations: ReLU, GELU, SiLU, Sigmoid, Tanh

    auto is_linear_op = [](OpType op) {
        return op == OpType::Add || op == OpType::Mul ||
               op == OpType::Sub || op == OpType::Div;
    };

    auto is_activation = [](OpType op) {
        return op == OpType::ReLU || op == OpType::GELU ||
               op == OpType::SiLU || op == OpType::Sigmoid ||
               op == OpType::Tanh;
    };

    int fused = 0;

    for (size_t i = 0; i < graph.nodes().size(); i++) {
        Node& linear_node = graph.node(static_cast<NodeId>(i));
        if (linear_node.is_fused) continue;
        if (!is_linear_op(linear_node.op)) continue;
        if (linear_node.outputs.empty()) continue;

        TensorId linear_output = linear_node.outputs[0];

        // Find consumers
        auto consumers = graph.consumers(linear_output);
        if (consumers.size() != 1) continue;

        Node& act_node = graph.node(consumers[0]);
        if (act_node.is_fused) continue;
        if (!is_activation(act_node.op)) continue;

        // Mark activation for fusion
        act_node.is_fused = true;
        act_node.fused_into = linear_node.id;

        // Store fused activation type
        linear_node.attrs.set("fused_activation", static_cast<int64_t>(act_node.op));

        fused++;
    }

    if (fused > 0) {
        GRANITE_LOG_INFO("OperatorFusion: Fused {} Linear+Activation patterns", fused);
    }

    return fused > 0;
}

// =============================================================================
// Layout Optimization Pass
// =============================================================================

Result<bool> LayoutOptimizationPass::run(Graph& graph) {
    // This is a placeholder for memory layout optimization
    // Full implementation would:
    // 1. Analyze data access patterns
    // 2. Determine optimal memory layouts (NCHW vs NHWC)
    // 3. Insert layout transformation nodes where needed

    // For now, this is a no-op
    (void)graph;
    return false;
}

// =============================================================================
// Optimization Pipeline
// =============================================================================

OptimizationPipeline OptimizationPipeline::create(OptimizationLevel level) {
    OptimizationPipeline pipeline;

    switch (level) {
        case OptimizationLevel::None:
            break;

        case OptimizationLevel::Basic:
            pipeline.add_pass(std::make_unique<DeadCodeEliminationPass>());
            break;

        case OptimizationLevel::Standard:
            pipeline.add_pass(std::make_unique<DeadCodeEliminationPass>());
            pipeline.add_pass(std::make_unique<ConstantFoldingPass>());
            break;

        case OptimizationLevel::Aggressive:
            pipeline.add_pass(std::make_unique<DeadCodeEliminationPass>());
            pipeline.add_pass(std::make_unique<ConstantFoldingPass>());
            pipeline.add_pass(std::make_unique<OperatorFusionPass>());
            pipeline.add_pass(std::make_unique<LayoutOptimizationPass>());
            break;
    }

    return pipeline;
}

void OptimizationPipeline::add_pass(std::unique_ptr<OptimizationPass> pass) {
    passes_.push_back(std::move(pass));
}

Result<int> OptimizationPipeline::run(Graph& graph) {
    int modifications = 0;

    for (auto& pass : passes_) {

        auto result = pass->run(graph);
        if (!result.ok()) {
            return result.error();
        }

        if (result.value()) {
            modifications++;
        }
    }

    return modifications;
}

Result<int> OptimizationPipeline::run_until_fixed_point(Graph& graph, int max_iterations) {
    int total_modifications = 0;

    for (int i = 0; i < max_iterations; i++) {
        auto result = run(graph);
        if (!result.ok()) {
            return result.error();
        }

        int mods = result.value();
        total_modifications += mods;

        if (mods == 0) {
            break;
        }
    }

    return total_modifications;
}

// =============================================================================
// Utility Functions
// =============================================================================

bool is_identity_op(const Node& node) {
    // Check for ops that are no-ops in certain conditions
    // e.g., Reshape to same shape, Add with zero, Mul with one

    // Currently just checks for basic patterns
    return false;  // Placeholder
}

bool can_fuse_ops(OpType first, OpType second) {
    // MatMul + Add (bias)
    if (first == OpType::MatMul && second == OpType::Add) return true;

    // Linear ops + Activation
    bool first_linear = (first == OpType::Add || first == OpType::Mul ||
                         first == OpType::Sub || first == OpType::Div);
    bool second_activation = (second == OpType::ReLU || second == OpType::GELU ||
                             second == OpType::SiLU || second == OpType::Sigmoid ||
                             second == OpType::Tanh);

    if (first_linear && second_activation) return true;

    return false;
}

OpType get_fused_op_type(OpType first, OpType second) {
    // This would return a special fused op type
    // For now, we just return the first op and mark fusion in attributes
    (void)second;
    return first;
}

}  // namespace granite
