// Benchmark for Tree Attention and Continuous Batching
// Tests the new features implemented for speculative decoding

#include "granite/granite.h"
#include "granite/llm.h"
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace granite;

// Helper to measure time
template<typename F>
double measure_ms(F&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void benchmark_tree_attention(const std::string& model_path) {
    std::cout << "\n=== Tree Attention Benchmark ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;

    // Create backend
    auto backend = create_default_backend();
    if (!backend) {
        std::cerr << "Failed to create backend" << std::endl;
        return;
    }
    auto init_result = backend->initialize();
    if (!init_result.ok()) {
        std::cerr << "Failed to initialize backend: " << init_result.error().message() << std::endl;
        return;
    }

    // Load model
    auto model_result = TransformerModel::load(model_path, backend.get());
    if (!model_result.ok()) {
        std::cerr << "Failed to load model: " << model_result.error().message() << std::endl;
        return;
    }
    auto model = std::move(model_result).take();

    // Allocate KV cache
    auto kv_result = KVCache::allocate(model.config(), 512, backend.get());
    if (!kv_result.ok()) {
        std::cerr << "Failed to allocate KV cache" << std::endl;
        return;
    }
    auto kv_cache = std::move(kv_result).take();

    // Test different tree sizes
    std::vector<std::pair<int, int>> tree_configs = {
        {2, 2},  // width=2, depth=2 -> 7 nodes
        {2, 3},  // width=2, depth=3 -> 15 nodes
        {2, 4},  // width=2, depth=4 -> 31 nodes
        {3, 2},  // width=3, depth=2 -> 13 nodes
        {3, 3},  // width=3, depth=3 -> 40 nodes
    };

    std::cout << "\nTree Forward Performance:" << std::endl;
    std::cout << std::setw(10) << "Width" << std::setw(10) << "Depth"
              << std::setw(10) << "Nodes" << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Throughput" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (auto [width, depth] : tree_configs) {
        // Build a sample tree
        SpeculationTree tree;

        // Build tree structure
        int root = tree.add_node(1, -1);  // Root token
        std::vector<int> current_level = {root};

        for (int d = 0; d < depth && tree.size() < 64; d++) {
            std::vector<int> next_level;
            for (int parent : current_level) {
                for (int w = 0; w < width && tree.size() < 64; w++) {
                    int child = tree.add_node(100 + d * width + w, parent);
                    next_level.push_back(child);
                }
            }
            current_level = std::move(next_level);
        }

        int num_nodes = static_cast<int>(tree.size());
        auto tokens = tree.flatten_tokens();
        auto parents = tree.get_parent_indices();

        // Clear cache for fresh test
        kv_cache.clear();

        // Warm up
        auto warmup = model.forward_tree(tokens, parents, &kv_cache, 0);
        if (!warmup.ok()) {
            std::cout << "forward_tree failed: " << warmup.error().message() << std::endl;
            continue;
        }

        // Benchmark
        kv_cache.clear();
        const int num_iters = 5;
        double total_time = 0;

        for (int i = 0; i < num_iters; i++) {
            kv_cache.clear();
            double time = measure_ms([&]() {
                auto result = model.forward_tree(tokens, parents, &kv_cache, 0);
            });
            total_time += time;
        }

        double avg_time = total_time / num_iters;
        double throughput = num_nodes / (avg_time / 1000.0);

        std::cout << std::setw(10) << width << std::setw(10) << depth
                  << std::setw(10) << num_nodes << std::setw(15) << std::fixed
                  << std::setprecision(2) << avg_time
                  << std::setw(15) << std::setprecision(1) << throughput << " tok/s" << std::endl;
    }

    // Compare with regular forward_batch
    std::cout << "\nComparison: forward_tree vs forward_batch (same token count):" << std::endl;

    std::vector<int> test_sizes = {7, 15, 31};
    for (int size : test_sizes) {
        // Build tree
        SpeculationTree tree;
        int root = tree.add_node(1, -1);
        std::vector<int> current_level = {root};
        while (tree.size() < static_cast<size_t>(size)) {
            std::vector<int> next_level;
            for (int parent : current_level) {
                if (tree.size() >= static_cast<size_t>(size)) break;
                int child = tree.add_node(100, parent);
                next_level.push_back(child);
                if (tree.size() >= static_cast<size_t>(size)) break;
                child = tree.add_node(101, parent);
                next_level.push_back(child);
            }
            current_level = std::move(next_level);
        }

        auto tokens = tree.flatten_tokens();
        auto parents = tree.get_parent_indices();

        // Benchmark forward_tree
        kv_cache.clear();
        double tree_time = measure_ms([&]() {
            for (int i = 0; i < 3; i++) {
                kv_cache.clear();
                model.forward_tree(tokens, parents, &kv_cache, 0);
            }
        }) / 3.0;

        // Benchmark forward_batch (linear sequence)
        std::vector<int32_t> linear_tokens(size, 100);
        kv_cache.clear();
        double batch_time = measure_ms([&]() {
            for (int i = 0; i < 3; i++) {
                kv_cache.clear();
                model.forward_batch(linear_tokens, &kv_cache, 0);
            }
        }) / 3.0;

        std::cout << "  " << size << " tokens: tree=" << std::fixed << std::setprecision(2)
                  << tree_time << "ms, batch=" << batch_time << "ms, ratio="
                  << std::setprecision(2) << (tree_time / batch_time) << "x" << std::endl;
    }
}

void benchmark_continuous_batching(const std::string& model_path) {
    std::cout << "\n=== Continuous Batching Benchmark ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;

    // Create backend and load model
    auto backend = create_default_backend();
    if (!backend) {
        std::cerr << "Failed to create backend" << std::endl;
        return;
    }
    backend->initialize();

    auto model_result = TransformerModel::load(model_path, backend.get());
    if (!model_result.ok()) {
        std::cerr << "Failed to load model: " << model_result.error().message() << std::endl;
        return;
    }
    auto model = std::move(model_result).take();

    // Load tokenizer
    auto gguf_result = GGUFFile::open(model_path);
    if (!gguf_result.ok()) {
        std::cerr << "Failed to open GGUF" << std::endl;
        return;
    }
    auto tok_result = Tokenizer::from_gguf(gguf_result.value());
    if (!tok_result.ok()) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return;
    }
    auto tokenizer = std::move(tok_result).take();

    // Test KVCachePool
    std::cout << "\nKVCachePool Performance:" << std::endl;

    std::vector<int> pool_sizes = {2, 4, 8};
    for (int pool_size : pool_sizes) {
        KVCachePool pool;

        double alloc_time = measure_ms([&]() {
            pool.allocate(pool_size, model.config(), 256, backend.get());
        });

        // Test acquire/release cycle
        double cycle_time = measure_ms([&]() {
            for (int i = 0; i < 1000; i++) {
                int slot = pool.acquire_slot();
                if (slot >= 0) pool.release_slot(slot);
            }
        });

        std::cout << "  Pool size " << pool_size << ": alloc=" << std::fixed
                  << std::setprecision(2) << alloc_time << "ms, "
                  << "1000 acquire/release=" << cycle_time << "ms" << std::endl;
    }

    // Test BatchScheduler
    std::cout << "\nBatchScheduler Performance:" << std::endl;

    BatchScheduler scheduler;
    auto init_result = scheduler.initialize(&model, &tokenizer, 4, 128);
    if (!init_result.ok()) {
        std::cerr << "Failed to init scheduler: " << init_result.error().message() << std::endl;
        return;
    }

    // Submit multiple requests
    std::vector<std::string> prompts = {
        "Hello",
        "The capital of France is",
        "In machine learning",
        "What is"
    };

    int tokens_generated = 0;
    auto callback = [&](const std::string& tok) {
        tokens_generated++;
        return tokens_generated < 10;  // Limit to 10 tokens per request
    };

    // Submit all requests
    double submit_time = measure_ms([&]() {
        for (const auto& prompt : prompts) {
            GenerationConfig config;
            config.max_tokens = 10;
            scheduler.submit(prompt, config, callback);
        }
    });

    std::cout << "  Submit " << prompts.size() << " requests: " << std::fixed
              << std::setprecision(2) << submit_time << "ms" << std::endl;

    // Process until done
    int total_steps = 0;
    double process_time = measure_ms([&]() {
        while (scheduler.has_pending()) {
            scheduler.step();
            total_steps++;
            if (total_steps > 100) break;  // Safety limit
        }
    });

    auto completed = scheduler.take_completed();
    std::cout << "  Process " << total_steps << " steps: " << process_time << "ms" << std::endl;
    std::cout << "  Completed: " << completed.size() << " requests" << std::endl;
    std::cout << "  Tokens generated: " << tokens_generated << std::endl;

    if (total_steps > 0) {
        double throughput = tokens_generated / (process_time / 1000.0);
        std::cout << "  Aggregate throughput: " << std::setprecision(1) << throughput << " tok/s" << std::endl;
    }
}

int main(int argc, char** argv) {
    std::string model_path;

    if (argc > 1) {
        model_path = argv[1];
    } else {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>\n";
        return 1;
    }

    std::cout << "Tree Attention & Continuous Batching Benchmark" << std::endl;
    std::cout << "=============================================" << std::endl;

    benchmark_tree_attention(model_path);
    benchmark_continuous_batching(model_path);

    return 0;
}
