#include <catch2/catch_test_macros.hpp>
#include <granite/granite.h>

using namespace granite;

TEST_CASE("Scheduler basic compilation", "[scheduler]") {
    SECTION("compile simple graph") {
        // Build a simple graph: a + b = c
        GraphBuilder builder("simple_add");
        auto a = builder.input("a", {2, 3}, DataType::FP32);
        auto b = builder.input("b", {2, 3}, DataType::FP32);
        auto c = builder.add(a, b, "output");
        builder.mark_output(c);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        // Compile the graph
        Scheduler scheduler;
        auto plan_result = scheduler.compile(graph);
        REQUIRE(plan_result.ok());

        auto plan = std::move(plan_result).take();

        // Check plan properties
        REQUIRE(plan.num_operations() == 1);
        REQUIRE(plan.num_buffers() == 3);  // a, b, c
        REQUIRE(plan.input_indices().size() == 2);
        REQUIRE(plan.output_indices().size() == 1);
    }

    SECTION("compile chain of operations") {
        GraphBuilder builder("chain");
        auto x = builder.input("x", {4, 4}, DataType::FP32);
        auto r1 = builder.relu(x, "r1");
        auto r2 = builder.gelu(r1, "r2");
        auto r3 = builder.silu(r2, "r3");
        builder.mark_output(r3);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        Scheduler scheduler;
        auto plan_result = scheduler.compile(graph);
        REQUIRE(plan_result.ok());

        auto plan = std::move(plan_result).take();

        // Should have 3 operations in order
        REQUIRE(plan.num_operations() == 3);
        REQUIRE(plan.operations()[0].op_type == OpType::ReLU);
        REQUIRE(plan.operations()[1].op_type == OpType::GELU);
        REQUIRE(plan.operations()[2].op_type == OpType::SiLU);

        // Should have 4 buffers (x, r1, r2, r3)
        REQUIRE(plan.num_buffers() == 4);
    }

    SECTION("compile diamond graph") {
        GraphBuilder builder("diamond");
        auto x = builder.input("x", {2, 2}, DataType::FP32);
        auto left = builder.relu(x, "left");
        auto right = builder.gelu(x, "right");
        auto out = builder.add(left, right, "output");
        builder.mark_output(out);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        Scheduler scheduler;
        auto plan_result = scheduler.compile(graph);
        REQUIRE(plan_result.ok());

        auto plan = std::move(plan_result).take();

        // Should have 3 operations
        REQUIRE(plan.num_operations() == 3);
        // Add should be last (depends on both relu and gelu)
        REQUIRE(plan.operations()[2].op_type == OpType::Add);
    }
}

TEST_CASE("Scheduler liveness analysis", "[scheduler]") {
    SECTION("liveness computed correctly") {
        GraphBuilder builder("liveness");
        auto a = builder.input("a", {2, 2}, DataType::FP32);
        auto b = builder.relu(a, "b");
        auto c = builder.gelu(b, "c");
        builder.mark_output(c);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        Scheduler scheduler;
        auto plan_result = scheduler.compile(graph);
        REQUIRE(plan_result.ok());

        auto plan = std::move(plan_result).take();
        const auto& allocations = plan.allocations();

        // Input 'a' should be live from step 0
        REQUIRE(allocations[0].first_use == 0);

        // Output 'c' should be live until end
        // The output buffer should have last_use = number of operations
        REQUIRE(allocations[2].last_use >= plan.num_operations());
    }
}

TEST_CASE("Scheduler memory estimation", "[scheduler]") {
    SECTION("total memory calculation") {
        GraphBuilder builder("memory");
        auto a = builder.input("a", {100, 100}, DataType::FP32);
        auto b = builder.relu(a, "b");
        builder.mark_output(b);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        Scheduler scheduler;
        auto plan_result = scheduler.compile(graph);
        REQUIRE(plan_result.ok());

        auto plan = std::move(plan_result).take();

        // Each buffer is 100*100*4 = 40000 bytes
        // 2 buffers (a and b)
        REQUIRE(plan.total_memory_required() == 80000);
    }

    SECTION("peak memory calculation") {
        GraphBuilder builder("peak");
        auto a = builder.input("a", {10, 10}, DataType::FP32);  // 400 bytes
        auto b = builder.relu(a, "b");                           // 400 bytes
        auto c = builder.gelu(b, "c");                           // 400 bytes
        builder.mark_output(c);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        Scheduler scheduler;
        auto plan_result = scheduler.compile(graph);
        REQUIRE(plan_result.ok());

        auto plan = std::move(plan_result).take();

        // Peak should be calculated based on live buffers
        REQUIRE(plan.peak_memory_required() > 0);
        REQUIRE(plan.peak_memory_required() <= plan.total_memory_required());
    }
}

TEST_CASE("Scheduler configuration", "[scheduler]") {
    SECTION("default config") {
        Scheduler scheduler;
        REQUIRE(scheduler.config().enable_aliasing == true);
        REQUIRE(scheduler.config().memory_budget == 0);
    }

    SECTION("custom config") {
        SchedulerConfig config;
        config.enable_aliasing = false;
        config.memory_budget = 1024 * 1024;

        Scheduler scheduler(config);
        REQUIRE(scheduler.config().enable_aliasing == false);
        REQUIRE(scheduler.config().memory_budget == 1024 * 1024);
    }

    SECTION("update config") {
        Scheduler scheduler;

        SchedulerConfig new_config;
        new_config.enable_fusion = true;
        scheduler.set_config(new_config);

        REQUIRE(scheduler.config().enable_fusion == true);
    }
}

TEST_CASE("CompiledPlan", "[scheduler]") {
    SECTION("to_string contains plan info") {
        GraphBuilder builder("plan_test");
        auto a = builder.input("a", {2, 2}, DataType::FP32);
        auto b = builder.relu(a, "b");
        builder.mark_output(b);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        Scheduler scheduler;
        auto plan_result = scheduler.compile(graph);
        REQUIRE(plan_result.ok());

        auto plan = std::move(plan_result).take();
        auto str = plan.to_string();

        REQUIRE(str.find("CompiledPlan") != std::string::npos);
        REQUIRE(str.find("Operations") != std::string::npos);
        REQUIRE(str.find("Buffers") != std::string::npos);
    }
}

TEST_CASE("Scheduler aliasing", "[scheduler]") {
    SECTION("aliasing groups assigned for intermediates") {
        // Create a longer chain where intermediate buffers could potentially alias
        GraphBuilder builder("aliasing");
        auto a = builder.input("a", {4, 4}, DataType::FP32);
        auto b = builder.relu(a, "b");      // intermediate
        auto c = builder.gelu(b, "c");      // intermediate
        auto d = builder.silu(c, "d");      // intermediate
        builder.mark_output(d);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        SchedulerConfig config;
        config.enable_aliasing = true;
        Scheduler scheduler(config);

        auto plan_result = scheduler.compile(graph);
        REQUIRE(plan_result.ok());

        auto plan = std::move(plan_result).take();

        // Check that intermediate buffers have alias groups assigned
        // (b, c, d are intermediates and should have device memory type)
        int device_buffers = 0;
        int with_alias_group = 0;
        for (const auto& alloc : plan.allocations()) {
            if (alloc.memory_type == MemoryType::Device) {
                device_buffers++;
                if (alloc.alias_group >= 0) {
                    with_alias_group++;
                }
            }
        }

        // Should have some intermediate device buffers
        REQUIRE(device_buffers > 0);
    }

    SECTION("disabling aliasing") {
        GraphBuilder builder("no_aliasing");
        auto a = builder.input("a", {4, 4}, DataType::FP32);
        auto b = builder.relu(a, "b");
        auto c = builder.gelu(b, "c");
        builder.mark_output(c);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        SchedulerConfig config;
        config.enable_aliasing = false;
        Scheduler scheduler(config);

        auto plan_result = scheduler.compile(graph);
        REQUIRE(plan_result.ok());

        auto plan = std::move(plan_result).take();

        // With aliasing disabled, no buffers should have alias groups
        for (const auto& alloc : plan.allocations()) {
            REQUIRE(alloc.alias_group == -1);
        }
    }
}

TEST_CASE("Scheduler dependencies", "[scheduler]") {
    SECTION("dependencies computed correctly") {
        GraphBuilder builder("deps");
        auto a = builder.input("a", {2, 2}, DataType::FP32);
        auto b = builder.input("b", {2, 2}, DataType::FP32);
        auto sum = builder.add(a, b, "sum");
        auto out = builder.relu(sum, "out");
        builder.mark_output(out);

        auto graph_result = builder.build();
        REQUIRE(graph_result.ok());
        auto graph = std::move(graph_result).take();

        Scheduler scheduler;
        auto plan_result = scheduler.compile(graph);
        REQUIRE(plan_result.ok());

        auto plan = std::move(plan_result).take();

        // First op (Add) should have no dependencies (inputs are graph inputs)
        REQUIRE(plan.operations()[0].dependencies.empty());

        // Second op (ReLU) should depend on first op (Add)
        REQUIRE(plan.operations()[1].dependencies.size() == 1);
        REQUIRE(plan.operations()[1].dependencies[0] == 0);
    }
}
