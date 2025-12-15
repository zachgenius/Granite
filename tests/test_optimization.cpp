#include <catch2/catch_test_macros.hpp>
#include <granite/granite.h>

using namespace granite;

TEST_CASE("OptimizationPipeline creation", "[optimization]") {
    SECTION("create with different levels") {
        auto none = OptimizationPipeline::create(OptimizationLevel::None);
        REQUIRE(none.num_passes() == 0);

        auto basic = OptimizationPipeline::create(OptimizationLevel::Basic);
        REQUIRE(basic.num_passes() == 1);

        auto standard = OptimizationPipeline::create(OptimizationLevel::Standard);
        REQUIRE(standard.num_passes() == 2);

        auto aggressive = OptimizationPipeline::create(OptimizationLevel::Aggressive);
        REQUIRE(aggressive.num_passes() == 4);
    }

    SECTION("add custom pass") {
        OptimizationPipeline pipeline;
        REQUIRE(pipeline.num_passes() == 0);

        pipeline.add_pass(std::make_unique<DeadCodeEliminationPass>());
        REQUIRE(pipeline.num_passes() == 1);

        pipeline.add_pass(std::make_unique<ConstantFoldingPass>());
        REQUIRE(pipeline.num_passes() == 2);
    }
}

TEST_CASE("OptimizationPipeline run", "[optimization]") {
    SECTION("run on simple graph") {
        GraphBuilder builder("simple");
        auto a = builder.input("a", {2, 2}, DataType::FP32);
        auto b = builder.relu(a, "b");
        builder.mark_output(b);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        auto pipeline = OptimizationPipeline::create(OptimizationLevel::Basic);
        auto result = pipeline.run(graph);
        REQUIRE(result.ok());
    }

    SECTION("run until fixed point") {
        GraphBuilder builder("fixedpoint");
        auto a = builder.input("a", {4, 4}, DataType::FP32);
        auto b = builder.relu(a, "b");
        auto c = builder.gelu(b, "c");
        builder.mark_output(c);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        auto pipeline = OptimizationPipeline::create(OptimizationLevel::Aggressive);
        auto result = pipeline.run_until_fixed_point(graph);
        REQUIRE(result.ok());
    }
}

TEST_CASE("Dead Code Elimination", "[optimization]") {
    SECTION("identifies unused nodes") {
        // Build a graph where some outputs aren't used
        Graph graph("dce_test");

        auto t0 = graph.add_tensor(TensorDesc::create("input", {2, 2}, DataType::FP32));
        auto t1 = graph.add_tensor(TensorDesc::create("used", {2, 2}, DataType::FP32));
        auto t2 = graph.add_tensor(TensorDesc::create("unused", {2, 2}, DataType::FP32));
        auto t3 = graph.add_tensor(TensorDesc::create("output", {2, 2}, DataType::FP32));

        // used path: input -> relu -> output
        (void)graph.add_node(OpType::ReLU, {t0}, {t1});
        (void)graph.add_node(OpType::GELU, {t1}, {t3});

        // unused path: input -> unused
        (void)graph.add_node(OpType::SiLU, {t0}, {t2});

        graph.set_inputs({t0});
        graph.set_outputs({t3});  // Only t3 is used, t2 is dead

        DeadCodeEliminationPass dce;
        auto result = dce.run(graph);
        REQUIRE(result.ok());
        // Should detect dead code
        REQUIRE(result.value() == true);
    }

    SECTION("no dead code in linear graph") {
        GraphBuilder builder("linear");
        auto x = builder.input("x", {2, 2}, DataType::FP32);
        auto y = builder.relu(x, "y");
        auto z = builder.gelu(y, "z");
        builder.mark_output(z);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        DeadCodeEliminationPass dce;
        auto result = dce.run(graph);
        REQUIRE(result.ok());
        REQUIRE(result.value() == false);  // No dead code
    }
}

TEST_CASE("Operator Fusion", "[optimization]") {
    SECTION("fuses MatMul + Add") {
        Graph graph("matmul_bias");

        auto a = graph.add_tensor(TensorDesc::create("a", {32, 64}, DataType::FP32));
        auto b = graph.add_tensor(TensorDesc::create("b", {64, 128}, DataType::FP32));
        auto bias = graph.add_tensor(TensorDesc::create("bias", {128}, DataType::FP32));
        auto mm_out = graph.add_tensor(TensorDesc::create("mm_out", {32, 128}, DataType::FP32));
        auto out = graph.add_tensor(TensorDesc::create("output", {32, 128}, DataType::FP32));

        auto mm_node = graph.add_node(OpType::MatMul, {a, b}, {mm_out});
        (void)graph.add_node(OpType::Add, {mm_out, bias}, {out});

        graph.set_inputs({a, b, bias});
        graph.set_outputs({out});

        OperatorFusionPass fusion;
        auto result = fusion.run(graph);
        REQUIRE(result.ok());
        REQUIRE(result.value() == true);

        // Check that fusion happened
        REQUIRE(graph.node(mm_node).attrs.get<bool>("has_bias") == true);
    }

    SECTION("fuses Linear + Activation") {
        Graph graph("linear_act");

        auto a = graph.add_tensor(TensorDesc::create("a", {2, 2}, DataType::FP32));
        auto b = graph.add_tensor(TensorDesc::create("b", {2, 2}, DataType::FP32));
        auto sum = graph.add_tensor(TensorDesc::create("sum", {2, 2}, DataType::FP32));
        auto out = graph.add_tensor(TensorDesc::create("output", {2, 2}, DataType::FP32));

        (void)graph.add_node(OpType::Add, {a, b}, {sum});
        (void)graph.add_node(OpType::ReLU, {sum}, {out});

        graph.set_inputs({a, b});
        graph.set_outputs({out});

        OperatorFusionPass fusion;
        auto result = fusion.run(graph);
        REQUIRE(result.ok());
        REQUIRE(result.value() == true);
    }

    SECTION("does not fuse incompatible ops") {
        Graph graph("no_fuse");

        auto a = graph.add_tensor(TensorDesc::create("a", {2, 2}, DataType::FP32));
        auto r1 = graph.add_tensor(TensorDesc::create("r1", {2, 2}, DataType::FP32));
        auto r2 = graph.add_tensor(TensorDesc::create("r2", {2, 2}, DataType::FP32));

        // ReLU -> GELU is not a fusable pattern
        (void)graph.add_node(OpType::ReLU, {a}, {r1});
        (void)graph.add_node(OpType::GELU, {r1}, {r2});

        graph.set_inputs({a});
        graph.set_outputs({r2});

        OperatorFusionPass fusion;
        auto result = fusion.run(graph);
        REQUIRE(result.ok());
        REQUIRE(result.value() == false);
    }
}

TEST_CASE("Constant Folding", "[optimization]") {
    SECTION("placeholder returns false") {
        GraphBuilder builder("const_fold");
        auto a = builder.input("a", {2, 2}, DataType::FP32);
        auto b = builder.relu(a, "b");
        builder.mark_output(b);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        ConstantFoldingPass fold;
        auto result = fold.run(graph);
        REQUIRE(result.ok());
        // Placeholder implementation returns false
        REQUIRE(result.value() == false);
    }
}

TEST_CASE("Utility functions", "[optimization]") {
    SECTION("can_fuse_ops") {
        // MatMul + Add can fuse
        REQUIRE(can_fuse_ops(OpType::MatMul, OpType::Add) == true);

        // Linear + Activation can fuse
        REQUIRE(can_fuse_ops(OpType::Add, OpType::ReLU) == true);
        REQUIRE(can_fuse_ops(OpType::Mul, OpType::GELU) == true);
        REQUIRE(can_fuse_ops(OpType::Sub, OpType::SiLU) == true);
        REQUIRE(can_fuse_ops(OpType::Div, OpType::Sigmoid) == true);

        // Activation + Activation cannot fuse
        REQUIRE(can_fuse_ops(OpType::ReLU, OpType::GELU) == false);

        // MatMul + ReLU cannot fuse (no bias)
        REQUIRE(can_fuse_ops(OpType::MatMul, OpType::ReLU) == false);
    }
}
