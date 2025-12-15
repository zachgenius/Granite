#include <catch2/catch_test_macros.hpp>
#include <granite/granite.h>

using namespace granite;

TEST_CASE("TensorDesc", "[graph]") {
    SECTION("create and numel") {
        auto desc = TensorDesc::create("test", {2, 3, 4}, DataType::FP32);
        REQUIRE(desc.name == "test");
        REQUIRE(desc.shape == std::vector<int64_t>{2, 3, 4});
        REQUIRE(desc.dtype == DataType::FP32);
        REQUIRE(desc.numel() == 24);
    }

    SECTION("size_bytes for different dtypes") {
        auto fp32 = TensorDesc::create("fp32", {10}, DataType::FP32);
        REQUIRE(fp32.size_bytes() == 40);  // 10 * 4 bytes

        auto fp16 = TensorDesc::create("fp16", {10}, DataType::FP16);
        REQUIRE(fp16.size_bytes() == 20);  // 10 * 2 bytes

        auto int8 = TensorDesc::create("int8", {10}, DataType::INT8);
        REQUIRE(int8.size_bytes() == 10);  // 10 * 1 byte
    }

    SECTION("scalar tensor") {
        auto scalar = TensorDesc::create("scalar", {}, DataType::FP32);
        REQUIRE(scalar.numel() == 1);
        REQUIRE(scalar.size_bytes() == 4);
    }
}

TEST_CASE("Graph construction", "[graph]") {
    Graph graph("test_graph");

    SECTION("empty graph") {
        REQUIRE(graph.name() == "test_graph");
        REQUIRE(graph.num_nodes() == 0);
        REQUIRE(graph.num_tensors() == 0);
    }

    SECTION("add tensors") {
        auto t0 = graph.add_tensor(TensorDesc::create("input_a", {2, 3}, DataType::FP32));
        auto t1 = graph.add_tensor(TensorDesc::create("input_b", {2, 3}, DataType::FP32));

        REQUIRE(graph.num_tensors() == 2);
        REQUIRE(t0 == 0);
        REQUIRE(t1 == 1);
        REQUIRE(graph.tensor(t0).name == "input_a");
        REQUIRE(graph.tensor(t1).name == "input_b");
    }

    SECTION("add nodes") {
        auto t0 = graph.add_tensor(TensorDesc::create("a", {2, 3}, DataType::FP32));
        auto t1 = graph.add_tensor(TensorDesc::create("b", {2, 3}, DataType::FP32));
        auto t2 = graph.add_tensor(TensorDesc::create("c", {2, 3}, DataType::FP32));

        auto n0 = graph.add_node(OpType::Add, {t0, t1}, {t2});

        REQUIRE(graph.num_nodes() == 1);
        REQUIRE(n0 == 0);
        REQUIRE(graph.node(n0).op == OpType::Add);
        REQUIRE(graph.node(n0).inputs.size() == 2);
        REQUIRE(graph.node(n0).outputs.size() == 1);
    }

    SECTION("producer and consumer tracking") {
        auto t0 = graph.add_tensor(TensorDesc::create("a", {2, 3}, DataType::FP32));
        auto t1 = graph.add_tensor(TensorDesc::create("b", {2, 3}, DataType::FP32));
        auto t2 = graph.add_tensor(TensorDesc::create("c", {2, 3}, DataType::FP32));
        auto t3 = graph.add_tensor(TensorDesc::create("d", {2, 3}, DataType::FP32));

        auto n0 = graph.add_node(OpType::Add, {t0, t1}, {t2});
        auto n1 = graph.add_node(OpType::ReLU, {t2}, {t3});

        // t2 is produced by n0, consumed by n1
        REQUIRE(graph.producer(t2) == n0);
        auto consumers = graph.consumers(t2);
        REQUIRE(consumers.size() == 1);
        REQUIRE(consumers[0] == n1);

        // Input tensors have no producer
        REQUIRE(graph.producer(t0) == INVALID_NODE_ID);
    }
}

TEST_CASE("Graph topological sort", "[graph]") {
    SECTION("linear chain") {
        Graph graph("linear");
        auto t0 = graph.add_tensor(TensorDesc::create("input", {2, 3}, DataType::FP32));
        auto t1 = graph.add_tensor(TensorDesc::create("hidden1", {2, 3}, DataType::FP32));
        auto t2 = graph.add_tensor(TensorDesc::create("hidden2", {2, 3}, DataType::FP32));
        auto t3 = graph.add_tensor(TensorDesc::create("output", {2, 3}, DataType::FP32));

        auto n0 = graph.add_node(OpType::ReLU, {t0}, {t1});
        auto n1 = graph.add_node(OpType::ReLU, {t1}, {t2});
        auto n2 = graph.add_node(OpType::ReLU, {t2}, {t3});

        graph.set_inputs({t0});
        graph.set_outputs({t3});

        auto result = graph.topological_sort();
        REQUIRE(result.ok());

        auto order = result.value();
        REQUIRE(order.size() == 3);
        REQUIRE(order[0] == n0);
        REQUIRE(order[1] == n1);
        REQUIRE(order[2] == n2);
    }

    SECTION("diamond pattern") {
        // input -> (node0, node1) -> node2 -> output
        Graph graph("diamond");
        auto input = graph.add_tensor(TensorDesc::create("input", {2, 3}, DataType::FP32));
        auto left = graph.add_tensor(TensorDesc::create("left", {2, 3}, DataType::FP32));
        auto right = graph.add_tensor(TensorDesc::create("right", {2, 3}, DataType::FP32));
        auto output = graph.add_tensor(TensorDesc::create("output", {2, 3}, DataType::FP32));

        auto n0 = graph.add_node(OpType::ReLU, {input}, {left});
        auto n1 = graph.add_node(OpType::ReLU, {input}, {right});
        auto n2 = graph.add_node(OpType::Add, {left, right}, {output});

        graph.set_inputs({input});
        graph.set_outputs({output});

        auto result = graph.topological_sort();
        REQUIRE(result.ok());

        auto order = result.value();
        REQUIRE(order.size() == 3);
        // n0 and n1 can be in either order, but n2 must be last
        REQUIRE(order[2] == n2);
        REQUIRE((order[0] == n0 || order[0] == n1));
        REQUIRE((order[1] == n0 || order[1] == n1));
    }

    SECTION("empty graph") {
        Graph graph("empty");
        auto result = graph.topological_sort();
        REQUIRE(result.ok());
        REQUIRE(result.value().empty());
    }
}

TEST_CASE("Graph cycle detection", "[graph]") {
    // Note: The current graph API doesn't allow creating cycles easily
    // since tensors are created before nodes. However, we test that
    // valid graphs don't report cycles.

    SECTION("linear graph has no cycle") {
        Graph graph("linear");
        auto t0 = graph.add_tensor(TensorDesc::create("t0", {2, 3}, DataType::FP32));
        auto t1 = graph.add_tensor(TensorDesc::create("t1", {2, 3}, DataType::FP32));
        auto t2 = graph.add_tensor(TensorDesc::create("t2", {2, 3}, DataType::FP32));

        (void)graph.add_node(OpType::ReLU, {t0}, {t1});
        (void)graph.add_node(OpType::ReLU, {t1}, {t2});

        graph.set_inputs({t0});
        graph.set_outputs({t2});

        REQUIRE(!graph.has_cycle());
    }
}

TEST_CASE("Graph validation", "[graph]") {
    SECTION("valid graph passes validation") {
        Graph graph("valid");
        auto t0 = graph.add_tensor(TensorDesc::create("input", {2, 3}, DataType::FP32));
        auto t1 = graph.add_tensor(TensorDesc::create("output", {2, 3}, DataType::FP32));

        (void)graph.add_node(OpType::ReLU, {t0}, {t1});
        graph.set_inputs({t0});
        graph.set_outputs({t1});

        auto result = graph.validate();
        REQUIRE(result.ok());
    }

    SECTION("graph without inputs fails validation") {
        Graph graph("no_inputs");
        auto t0 = graph.add_tensor(TensorDesc::create("a", {2, 3}, DataType::FP32));
        auto t1 = graph.add_tensor(TensorDesc::create("b", {2, 3}, DataType::FP32));

        (void)graph.add_node(OpType::ReLU, {t0}, {t1});
        graph.set_outputs({t1});
        // No inputs set

        auto result = graph.validate();
        REQUIRE(!result.ok());
        REQUIRE(result.error().code() == ErrorCode::InvalidGraph);
    }

    SECTION("graph without outputs fails validation") {
        Graph graph("no_outputs");
        auto t0 = graph.add_tensor(TensorDesc::create("a", {2, 3}, DataType::FP32));
        auto t1 = graph.add_tensor(TensorDesc::create("b", {2, 3}, DataType::FP32));

        (void)graph.add_node(OpType::ReLU, {t0}, {t1});
        graph.set_inputs({t0});
        // No outputs set

        auto result = graph.validate();
        REQUIRE(!result.ok());
        REQUIRE(result.error().code() == ErrorCode::InvalidGraph);
    }
}

TEST_CASE("GraphBuilder", "[graph]") {
    SECTION("simple add operation") {
        GraphBuilder builder("add_graph");

        auto a = builder.input("a", {2, 3}, DataType::FP32);
        auto b = builder.input("b", {2, 3}, DataType::FP32);
        auto c = builder.add(a, b, "sum");
        builder.mark_output(c);

        auto result = builder.build();
        REQUIRE(result.ok());

        auto graph = result.take();
        REQUIRE(graph.num_nodes() == 1);
        REQUIRE(graph.num_tensors() == 3);
        REQUIRE(graph.inputs().size() == 2);
        REQUIRE(graph.outputs().size() == 1);
    }

    SECTION("chained operations") {
        GraphBuilder builder("chain");

        auto x = builder.input("x", {4, 4}, DataType::FP32);
        auto r1 = builder.relu(x, "relu1");
        auto r2 = builder.gelu(r1, "gelu1");
        auto r3 = builder.silu(r2, "silu1");
        builder.mark_output(r3);

        auto result = builder.build();
        REQUIRE(result.ok());

        auto graph = result.take();
        REQUIRE(graph.num_nodes() == 3);
        REQUIRE(graph.nodes()[0].op == OpType::ReLU);
        REQUIRE(graph.nodes()[1].op == OpType::GELU);
        REQUIRE(graph.nodes()[2].op == OpType::SiLU);
    }

    SECTION("binary operations") {
        GraphBuilder builder("binary_ops");

        auto a = builder.input("a", {2, 2}, DataType::FP32);
        auto b = builder.input("b", {2, 2}, DataType::FP32);

        auto sum = builder.add(a, b);
        auto diff = builder.sub(a, b);
        auto prod = builder.mul(sum, diff);
        auto quot = builder.div(prod, b);
        builder.mark_output(quot);

        auto result = builder.build();
        REQUIRE(result.ok());

        auto graph = result.take();
        REQUIRE(graph.num_nodes() == 4);
    }

    SECTION("matmul operation") {
        GraphBuilder builder("matmul");

        auto a = builder.input("a", {32, 64}, DataType::FP32);
        auto b = builder.input("b", {64, 128}, DataType::FP32);
        auto c = builder.matmul(a, b, "output");
        builder.mark_output(c);

        auto result = builder.build();
        REQUIRE(result.ok());

        auto graph = result.take();
        REQUIRE(graph.num_nodes() == 1);
        REQUIRE(graph.nodes()[0].op == OpType::MatMul);
    }

    SECTION("softmax with axis") {
        GraphBuilder builder("softmax");

        auto x = builder.input("x", {2, 10}, DataType::FP32);
        auto y = builder.softmax(x, -1, "probs");
        builder.mark_output(y);

        auto result = builder.build();
        REQUIRE(result.ok());

        auto graph = result.take();
        REQUIRE(graph.num_nodes() == 1);
        REQUIRE(graph.nodes()[0].op == OpType::Softmax);
        // Check axis attribute
        auto& attrs = graph.nodes()[0].attrs;
        REQUIRE(attrs.get<int64_t>("axis") == -1);
    }

    SECTION("layer norm") {
        GraphBuilder builder("layer_norm");

        auto x = builder.input("x", {2, 512}, DataType::FP32);
        auto weight = builder.input("weight", {512}, DataType::FP32);
        auto bias = builder.input("bias", {512}, DataType::FP32);
        auto y = builder.layer_norm(x, weight, bias, 1e-5f, "ln");
        builder.mark_output(y);

        auto result = builder.build();
        REQUIRE(result.ok());

        auto graph = result.take();
        REQUIRE(graph.num_nodes() == 1);
        REQUIRE(graph.nodes()[0].op == OpType::LayerNorm);
    }

    SECTION("rms norm") {
        GraphBuilder builder("rms_norm");

        auto x = builder.input("x", {2, 512}, DataType::FP32);
        auto weight = builder.input("weight", {512}, DataType::FP32);
        auto y = builder.rms_norm(x, weight, 1e-5f, "rms");
        builder.mark_output(y);

        auto result = builder.build();
        REQUIRE(result.ok());

        auto graph = result.take();
        REQUIRE(graph.num_nodes() == 1);
        REQUIRE(graph.nodes()[0].op == OpType::RMSNorm);
    }
}

TEST_CASE("Graph to_string", "[graph]") {
    GraphBuilder builder("simple");
    auto a = builder.input("input", {2, 3}, DataType::FP32);
    auto b = builder.relu(a, "output");
    builder.mark_output(b);

    auto result = builder.build();
    REQUIRE(result.ok());

    auto graph = result.take();
    auto str = graph.to_string();

    // Should contain graph name and some structure info
    REQUIRE(str.find("simple") != std::string::npos);
    REQUIRE(str.find("Nodes") != std::string::npos);
}

TEST_CASE("ExecutionPlan", "[graph]") {
    ExecutionPlan plan;

    SECTION("empty plan") {
        REQUIRE(plan.num_steps() == 0);
        REQUIRE(plan.steps().empty());
    }

    SECTION("add steps") {
        ExecutionStep step1;
        step1.node_id = 0;
        step1.op = OpType::ReLU;
        plan.add_step(step1);

        ExecutionStep step2;
        step2.node_id = 1;
        step2.op = OpType::Add;
        plan.add_step(step2);

        REQUIRE(plan.num_steps() == 2);
        REQUIRE(plan.steps()[0].op == OpType::ReLU);
        REQUIRE(plan.steps()[1].op == OpType::Add);
    }

    SECTION("set memory usage") {
        plan.set_total_memory(1024 * 1024);  // 1MB
        REQUIRE(plan.total_memory_bytes() == 1024 * 1024);
    }
}
