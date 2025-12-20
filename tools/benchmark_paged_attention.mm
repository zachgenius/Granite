// benchmark_paged_attention.mm - Compare Paged vs Contiguous KV Cache Performance
//
// This benchmark measures decode performance with:
// - Contiguous KV cache (existing implementation)
// - Paged KV cache (new PagedAttention implementation)
// - Optional comparison with llama.cpp
//
// Usage:
//   ./benchmark_paged_attention <model.gguf> [options]
//
// Options:
//   --context <n>       Context length to test (default: 512)
//   --decode <n>        Number of decode tokens (default: 64)
//   --block-size <n>    Paged attention block size (default: 16)
//   --runs <n>          Number of benchmark runs (default: 5)
//   --llama-bench <path>  Path to llama-bench binary (optional)

#include <granite/granite.h>
#ifdef GRANITE_HAS_METAL
#include <granite/metal_compute.h>
#endif
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>

using namespace granite;
using Clock = std::chrono::high_resolution_clock;

struct BenchConfig {
    std::string model_path;
    std::string llama_bench_path;
    int context_len = 512;
    int decode_tokens = 64;
    int block_size = 16;
    int runs = 5;
    int warmup = 2;
    bool verbose = false;
};

struct BenchResult {
    std::string name;
    double decode_tok_per_sec = 0;
    double time_per_token_ms = 0;
    double total_time_ms = 0;
    int tokens_generated = 0;
};

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_result(const BenchResult& r) {
    std::cout << std::left << std::setw(25) << r.name
              << std::right << std::setw(12) << std::fixed << std::setprecision(1)
              << r.decode_tok_per_sec << " tok/s"
              << std::setw(12) << std::setprecision(2) << r.time_per_token_ms << " ms/tok"
              << "\n";
}

BenchConfig parse_args(int argc, char** argv) {
    BenchConfig config;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [options]\n";
        std::cerr << "\nOptions:\n";
        std::cerr << "  --context <n>       Context length to test (default: 512)\n";
        std::cerr << "  --decode <n>        Number of decode tokens (default: 64)\n";
        std::cerr << "  --block-size <n>    Paged attention block size (default: 16)\n";
        std::cerr << "  --runs <n>          Number of benchmark runs (default: 5)\n";
        std::cerr << "  --llama-bench <path>  Path to llama-bench binary\n";
        std::cerr << "  --verbose           Verbose output\n";
        exit(1);
    }

    config.model_path = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--context" && i + 1 < argc) {
            config.context_len = std::stoi(argv[++i]);
        } else if (arg == "--decode" && i + 1 < argc) {
            config.decode_tokens = std::stoi(argv[++i]);
        } else if (arg == "--block-size" && i + 1 < argc) {
            config.block_size = std::stoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            config.runs = std::stoi(argv[++i]);
        } else if (arg == "--llama-bench" && i + 1 < argc) {
            config.llama_bench_path = argv[++i];
        } else if (arg == "--verbose") {
            config.verbose = true;
        }
    }

    return config;
}

// Run llama-bench for comparison
BenchResult run_llama_bench(const BenchConfig& config) {
    BenchResult result;
    result.name = "llama.cpp";

    if (config.llama_bench_path.empty()) {
        return result;
    }

    // Build llama-bench command
    std::ostringstream cmd;
    cmd << config.llama_bench_path
        << " -m " << config.model_path
        << " -p 0"  // No prompt processing
        << " -n " << config.decode_tokens
        << " -ngl 99"  // Offload all layers to GPU
        << " -r " << config.runs
        << " 2>&1";

    // Run and capture output
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        std::cerr << "Failed to run llama-bench\n";
        return result;
    }

    char buffer[256];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }
    pclose(pipe);

    // Parse output for decode speed (look for "t/s" or "tok/s")
    // llama-bench output format: "... | 258.45 tok/s |..."
    size_t pos = output.find("tok/s");
    if (pos != std::string::npos) {
        // Find the number before "tok/s"
        size_t end = pos;
        while (end > 0 && output[end-1] == ' ') end--;
        size_t start = end;
        while (start > 0 && (isdigit(output[start-1]) || output[start-1] == '.')) start--;
        if (start < end) {
            result.decode_tok_per_sec = std::stod(output.substr(start, end - start));
            result.time_per_token_ms = 1000.0 / result.decode_tok_per_sec;
        }
    }

    return result;
}

int main(int argc, char** argv) {
    auto config = parse_args(argc, argv);

    print_header("Paged Attention Benchmark");
    std::cout << "Model: " << config.model_path << "\n";
    std::cout << "Context: " << config.context_len << " tokens\n";
    std::cout << "Decode: " << config.decode_tokens << " tokens\n";
    std::cout << "Block size: " << config.block_size << " (paged mode)\n";
    std::cout << "Runs: " << config.runs << "\n";

    // Create backend
    auto backend = create_backend(BackendType::Metal);
    if (!backend) {
        std::cerr << "Failed to create Metal backend\n";
        return 1;
    }
    backend->initialize();

    std::cout << "\nLoading model...\n";

    // Load model
    auto model_result = TransformerModel::load(config.model_path, backend.get());
    if (!model_result.ok()) {
        std::cerr << "Failed to load model: " << model_result.error().message() << "\n";
        return 1;
    }
    auto model = std::move(model_result).take();

    const auto& mc = model.config();
    std::cout << "Model: " << mc.num_layers << " layers, "
              << mc.hidden_dim << " hidden, " << mc.num_heads << " heads\n";

    // Allocate KV caches
    int max_seq = config.context_len + config.decode_tokens + 64;

    // Contiguous cache
    auto kv_result = KVCache::allocate(mc, max_seq, backend.get());
    if (!kv_result.ok()) {
        std::cerr << "Failed to allocate KV cache: " << kv_result.error().message() << "\n";
        return 1;
    }
    auto kv_cache = std::move(kv_result).take();

    // GPU cache (contiguous)
    auto gpu_cache_result = model.allocate_gpu_kv_cache(max_seq);
    if (!gpu_cache_result.ok()) {
        std::cerr << "Failed to allocate GPU KV cache: " << gpu_cache_result.error().message() << "\n";
        return 1;
    }

    std::cout << "GPU KV cache allocated\n";

    // Create sample tokens for prefill
    std::vector<int32_t> prefill_tokens(config.context_len);
    for (int i = 0; i < config.context_len; i++) {
        prefill_tokens[i] = 1 + (i % 1000);  // Avoid token 0 (often padding)
    }

    // Results storage
    std::vector<BenchResult> results;

    // Get MetalCompute for profiling
#ifdef GRANITE_HAS_METAL
    granite::MetalCompute* gpu = granite::get_metal_compute();
#else
    void* gpu = nullptr;
#endif

    // =========================================================================
    // Benchmark 1: Contiguous KV Cache
    // =========================================================================
    print_header("Contiguous KV Cache (Baseline)");

    {
        BenchResult result;
        result.name = "Contiguous";

        std::vector<double> times;
        times.reserve(config.runs);

        for (int run = 0; run < config.warmup + config.runs; run++) {
            // Reset cache
            kv_cache.clear();
            if (model.gpu_kv_cache()) {
                model.gpu_kv_cache()->current_len = 0;
            }

            // Prefill
            auto prefill_result = model.forward_batch(prefill_tokens, &kv_cache, 0);
            if (!prefill_result.ok()) {
                std::cerr << "Prefill failed: " << prefill_result.error().message() << "\n";
                continue;
            }

            // Sync CPU->GPU cache
            auto sync_result = model.sync_cpu_to_gpu_kv_cache(&kv_cache);
            if (!sync_result.ok()) {
                std::cerr << "Sync failed: " << sync_result.error().message() << "\n";
                continue;
            }

            // Decode
#ifdef GRANITE_HAS_METAL
            if (gpu && run == config.warmup) {
                gpu->enable_profiling(true);
                gpu->reset_profiling_stats();
            }
#endif
            auto start = Clock::now();
            int32_t last_token = prefill_tokens.back();

            for (int i = 0; i < config.decode_tokens; i++) {
                auto decode_result = model.forward_single(last_token, kv_cache);
                if (!decode_result.ok()) {
                    std::cerr << "Decode failed: " << decode_result.error().message() << "\n";
                    break;
                }
                // Get next token (greedy for benchmark - just use argmax)
                auto logits = std::move(decode_result).take();
                auto* data = static_cast<const float*>(backend->map_buffer(logits.buffer()).value());
                int vocab_size = mc.vocab_size;
                float max_val = data[0];
                int max_idx = 0;
                for (int v = 1; v < vocab_size; v++) {
                    if (data[v] > max_val) {
                        max_val = data[v];
                        max_idx = v;
                    }
                }
                backend->unmap_buffer(logits.buffer());
                last_token = max_idx;
            }

            auto end = Clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

#ifdef GRANITE_HAS_METAL
            if (gpu && run == config.warmup) {
                gpu->enable_profiling(false);
                uint64_t dispatches, syncs, cmd_buffers;
                double sync_time_ms;
                gpu->get_profiling_stats(dispatches, syncs, sync_time_ms, cmd_buffers);
                std::cout << "  Profiling: " << dispatches << " dispatches, "
                          << syncs << " syncs (" << std::fixed << std::setprecision(2)
                          << sync_time_ms << " ms), " << cmd_buffers << " cmd buffers\n";
            }
#endif

            if (run >= config.warmup) {
                times.push_back(elapsed_ms);
                if (config.verbose) {
                    std::cout << "  Run " << (run - config.warmup + 1) << ": "
                              << std::fixed << std::setprecision(1) << elapsed_ms << " ms\n";
                }
            }
        }

        if (!times.empty()) {
            std::sort(times.begin(), times.end());
            double median_ms = times[times.size() / 2];
            result.total_time_ms = median_ms;
            result.tokens_generated = config.decode_tokens;
            result.decode_tok_per_sec = config.decode_tokens / (median_ms / 1000.0);
            result.time_per_token_ms = median_ms / config.decode_tokens;
        }

        print_result(result);
        results.push_back(result);
    }

    // =========================================================================
    // Benchmark 2: Paged KV Cache
    // =========================================================================
    print_header("Paged KV Cache (PagedAttention)");

    {
        // Allocate paged cache
        auto paged_result = model.allocate_paged_kv_cache(max_seq, config.block_size);
        if (!paged_result.ok()) {
            std::cerr << "Failed to allocate paged KV cache: " << paged_result.error().message() << "\n";
        } else {
            BenchResult result;
            result.name = "Paged (block=" + std::to_string(config.block_size) + ")";

            std::vector<double> times;
            times.reserve(config.runs);

            for (int run = 0; run < config.warmup + config.runs; run++) {
                // Reset cache
                model.clear_paged_cache();
                kv_cache.clear();

                // For paged attention, we need to build up the cache token by token
                // (or use a paged prefill path if available)
                // For now, we'll just measure decode starting from empty cache
                // since the paged path is optimized for decode

                // Pre-fill the paged cache by running decode tokens
                int32_t token = prefill_tokens[0];
                for (int i = 0; i < config.context_len; i++) {
                    auto decode_result = model.forward_single(token, kv_cache);
                    if (!decode_result.ok()) {
                        std::cerr << "Paged prefill failed at " << i << ": "
                                  << decode_result.error().message() << "\n";
                        break;
                    }
                    token = prefill_tokens[std::min(i + 1, (int)prefill_tokens.size() - 1)];
                }

                // Now measure decode
#ifdef GRANITE_HAS_METAL
                if (gpu && run == config.warmup) {
                    gpu->enable_profiling(true);
                    gpu->reset_profiling_stats();
                }
#endif
                auto start = Clock::now();
                int32_t last_token = prefill_tokens.back();

                for (int i = 0; i < config.decode_tokens; i++) {
                    auto decode_result = model.forward_single(last_token, kv_cache);
                    if (!decode_result.ok()) {
                        std::cerr << "Paged decode failed: " << decode_result.error().message() << "\n";
                        break;
                    }
                    auto logits = std::move(decode_result).take();
                    auto* data = static_cast<const float*>(backend->map_buffer(logits.buffer()).value());
                    int vocab_size = mc.vocab_size;
                    float max_val = data[0];
                    int max_idx = 0;
                    for (int v = 1; v < vocab_size; v++) {
                        if (data[v] > max_val) {
                            max_val = data[v];
                            max_idx = v;
                        }
                    }
                    backend->unmap_buffer(logits.buffer());
                    last_token = max_idx;
                }

                auto end = Clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

#ifdef GRANITE_HAS_METAL
                if (gpu && run == config.warmup) {
                    gpu->enable_profiling(false);
                    uint64_t dispatches, syncs, cmd_buffers;
                    double sync_time_ms;
                    gpu->get_profiling_stats(dispatches, syncs, sync_time_ms, cmd_buffers);
                    std::cout << "  Profiling: " << dispatches << " dispatches, "
                              << syncs << " syncs (" << std::fixed << std::setprecision(2)
                              << sync_time_ms << " ms), " << cmd_buffers << " cmd buffers\n";
                }
#endif

                if (run >= config.warmup) {
                    times.push_back(elapsed_ms);
                    if (config.verbose) {
                        std::cout << "  Run " << (run - config.warmup + 1) << ": "
                                  << std::fixed << std::setprecision(1) << elapsed_ms << " ms\n";
                    }
                }
            }

            if (!times.empty()) {
                std::sort(times.begin(), times.end());
                double median_ms = times[times.size() / 2];
                result.total_time_ms = median_ms;
                result.tokens_generated = config.decode_tokens;
                result.decode_tok_per_sec = config.decode_tokens / (median_ms / 1000.0);
                result.time_per_token_ms = median_ms / config.decode_tokens;
            }

            print_result(result);
            results.push_back(result);
        }
    }

    // =========================================================================
    // Benchmark 3: llama.cpp (if available)
    // =========================================================================
    if (!config.llama_bench_path.empty()) {
        print_header("llama.cpp Comparison");
        auto llama_result = run_llama_bench(config);
        if (llama_result.decode_tok_per_sec > 0) {
            print_result(llama_result);
            results.push_back(llama_result);
        } else {
            std::cout << "Could not parse llama-bench output\n";
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================
    print_header("Summary");

    std::cout << std::left << std::setw(25) << "Implementation"
              << std::right << std::setw(15) << "Decode (tok/s)"
              << std::setw(15) << "ms/token"
              << std::setw(15) << "vs Baseline"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    double baseline = results.empty() ? 0 : results[0].decode_tok_per_sec;

    for (const auto& r : results) {
        double speedup = baseline > 0 ? r.decode_tok_per_sec / baseline : 0;
        std::cout << std::left << std::setw(25) << r.name
                  << std::right << std::setw(15) << std::fixed << std::setprecision(1)
                  << r.decode_tok_per_sec
                  << std::setw(15) << std::setprecision(2) << r.time_per_token_ms
                  << std::setw(14) << std::setprecision(2) << speedup << "x"
                  << "\n";
    }

    std::cout << "\n";

    return 0;
}
