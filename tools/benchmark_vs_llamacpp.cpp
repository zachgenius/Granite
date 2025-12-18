// benchmark_vs_llamacpp.cpp - Compare Granite performance with llama.cpp
//
// This benchmark measures:
// - Prefill throughput (prompt processing)
// - Decode throughput (token generation)
// - Memory usage
// - First token latency
//
// Usage:
//   ./granite_bench_vs_llamacpp <model.gguf> [--llama-bench <path>] [options]
//
// Options:
//   --prompt-len <n>   Prompt length in tokens (default: 128)
//   --gen-len <n>      Number of tokens to generate (default: 128)
//   --batch-size <n>   Batch size for prefill (default: 512)
//   --runs <n>         Number of benchmark runs (default: 3)
//   --warmup <n>       Number of warmup runs (default: 1)
//   --llama-bench <path>  Path to llama-bench binary
//   --compare-only     Only run comparison (skip Granite bench)

#include <granite/granite.h>
#ifdef GRANITE_HAS_METAL
#include <granite/metal_compute.h>
#endif
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>

using namespace granite;

struct BenchmarkConfig {
    std::string model_path;
    std::string llama_bench_path;
    int prompt_len = 128;
    int gen_len = 128;
    int batch_size = 512;
    int runs = 3;
    int warmup = 1;
    bool compare_only = false;
    bool verbose = false;
};

struct BenchmarkResult {
    std::string name;
    double prefill_tok_per_sec = 0;
    double decode_tok_per_sec = 0;
    double decode_raw_tok_per_sec = 0;  // Raw forward pass without sampling
    double time_to_first_token_ms = 0;
    double total_time_ms = 0;
    double memory_mb = 0;
    int prompt_tokens = 0;
    int generated_tokens = 0;
};

// Parse command line arguments
BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig config;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  --prompt-len <n>      Prompt length in tokens (default: 128)\n";
        std::cerr << "  --gen-len <n>         Number of tokens to generate (default: 128)\n";
        std::cerr << "  --batch-size <n>      Batch size for prefill (default: 512)\n";
        std::cerr << "  --runs <n>            Number of benchmark runs (default: 3)\n";
        std::cerr << "  --warmup <n>          Number of warmup runs (default: 1)\n";
        std::cerr << "  --llama-bench <path>  Path to llama-bench binary\n";
        std::cerr << "  --compare-only        Only run llama.cpp comparison\n";
        std::cerr << "  --verbose             Verbose output\n";
        exit(1);
    }

    config.model_path = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--prompt-len" && i + 1 < argc) {
            config.prompt_len = std::stoi(argv[++i]);
        } else if (arg == "--gen-len" && i + 1 < argc) {
            config.gen_len = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            config.runs = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            config.warmup = std::stoi(argv[++i]);
        } else if (arg == "--llama-bench" && i + 1 < argc) {
            config.llama_bench_path = argv[++i];
        } else if (arg == "--compare-only") {
            config.compare_only = true;
        } else if (arg == "--verbose") {
            config.verbose = true;
        }
    }

    return config;
}

// Calculate statistics
struct Stats {
    double mean;
    double stddev;
    double min;
    double max;
};

Stats calculate_stats(const std::vector<double>& values) {
    Stats stats;
    if (values.empty()) {
        return {0, 0, 0, 0};
    }

    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    stats.min = *std::min_element(values.begin(), values.end());
    stats.max = *std::max_element(values.begin(), values.end());

    double sq_sum = 0;
    for (double v : values) {
        sq_sum += (v - stats.mean) * (v - stats.mean);
    }
    stats.stddev = std::sqrt(sq_sum / values.size());

    return stats;
}

// Run Granite benchmark
BenchmarkResult run_granite_benchmark(const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.name = "Granite";
    result.prompt_tokens = config.prompt_len;
    result.generated_tokens = config.gen_len;

    std::cout << "\n=== Running Granite Benchmark ===\n";
    std::cout << "Model: " << config.model_path << "\n";
    std::cout << "Prompt length: " << config.prompt_len << " tokens\n";
    std::cout << "Generation length: " << config.gen_len << " tokens\n";
    std::cout << "Runs: " << config.runs << " (warmup: " << config.warmup << ")\n\n";

    // Create backend
    auto backend = create_default_backend();
    if (!backend) {
        std::cerr << "Failed to create backend\n";
        return result;
    }
    backend->initialize();

    // Load model
    auto model_result = TransformerModel::load(config.model_path, backend.get());
    if (!model_result.ok()) {
        std::cerr << "Failed to load model: " << model_result.error().message() << "\n";
        return result;
    }
    auto model = std::move(model_result).take();

    // Create KV cache
    auto cache_result = KVCache::allocate(model.config(), config.prompt_len + config.gen_len + 64, backend.get());
    if (!cache_result.ok()) {
        std::cerr << "Failed to allocate KV cache\n";
        return result;
    }
    auto kv_cache = std::move(cache_result).take();

    // Allocate GPU KV cache for Metal acceleration
#ifdef GRANITE_HAS_METAL
    auto gpu_cache_result = model.allocate_gpu_kv_cache(config.prompt_len + config.gen_len + 64);
    if (gpu_cache_result.ok()) {
        std::cout << "GPU KV cache allocated - Metal acceleration enabled\n";
    } else {
        std::cout << "GPU KV cache allocation failed: " << gpu_cache_result.error().message() << "\n";
        std::cout << "Falling back to CPU inference\n";
    }
#endif

    // Create dummy prompt tokens
    std::vector<int32_t> prompt_tokens(config.prompt_len, 1);  // Token ID 1 repeated

    std::vector<double> prefill_times;
    std::vector<double> decode_times;
    std::vector<double> decode_raw_times;  // Raw forward without sampling
    std::vector<double> ttft_times;

    // Enable Metal profiling if available
#ifdef GRANITE_HAS_METAL
    auto* gpu = get_metal_compute();
    if (gpu && gpu->is_initialized()) {
        gpu->enable_profiling(true);
    }
#endif

    int total_runs = config.warmup + config.runs;
    for (int run = 0; run < total_runs; run++) {
        bool is_warmup = run < config.warmup;
        if (config.verbose) {
            std::cout << (is_warmup ? "[Warmup " : "[Run ")
                      << (is_warmup ? run + 1 : run - config.warmup + 1)
                      << "] ";
        }

        // Reset both CPU and GPU KV caches
        kv_cache.clear();
#ifdef GRANITE_HAS_METAL
        if (model.gpu_kv_cache()) {
            model.gpu_kv_cache()->clear();
        }
#endif

        // Prefill phase
        auto prefill_start = std::chrono::high_resolution_clock::now();
        auto prefill_result = model.forward_batch(prompt_tokens, &kv_cache, 0);
        auto prefill_end = std::chrono::high_resolution_clock::now();

        if (!prefill_result.ok()) {
            std::cerr << "Prefill failed: " << prefill_result.error().message() << "\n";
            continue;
        }

        double prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

        // Get last token for decode
        auto& logits = prefill_result.value();
        auto map_result = backend->map_buffer(logits.buffer());
        if (!map_result.ok()) {
            std::cerr << "Failed to map logits buffer\n";
            continue;
        }
        const float* logits_data = static_cast<const float*>(map_result.value());
        int vocab_size = static_cast<int>(logits.size(2));

        // Argmax for next token
        int32_t next_token = 0;
        float max_val = logits_data[0];
        for (int v = 1; v < vocab_size; v++) {
            if (logits_data[v] > max_val) {
                max_val = logits_data[v];
                next_token = v;
            }
        }
        backend->unmap_buffer(logits.buffer());

        // Decode phase - measure raw forward time separately from full pipeline
        double total_forward_ms = 0;
        auto decode_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < config.gen_len; i++) {
            auto fwd_start = std::chrono::high_resolution_clock::now();
            auto decode_result = model.forward_single(next_token, kv_cache);
            auto fwd_end = std::chrono::high_resolution_clock::now();
            total_forward_ms += std::chrono::duration<double, std::milli>(fwd_end - fwd_start).count();

            if (!decode_result.ok()) {
                std::cerr << "Decode failed at step " << i << "\n";
                break;
            }

            auto& dec_logits = decode_result.value();
            auto dec_map = backend->map_buffer(dec_logits.buffer());
            if (!dec_map.ok()) continue;
            const float* dec_data = static_cast<const float*>(dec_map.value());

            // Argmax
            next_token = 0;
            max_val = dec_data[0];
            for (int v = 1; v < vocab_size; v++) {
                if (dec_data[v] > max_val) {
                    max_val = dec_data[v];
                    next_token = v;
                }
            }
            backend->unmap_buffer(dec_logits.buffer());
        }
        auto decode_end = std::chrono::high_resolution_clock::now();

        double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

        if (!is_warmup) {
            prefill_times.push_back(prefill_ms);
            decode_times.push_back(decode_ms);
            decode_raw_times.push_back(total_forward_ms);
            ttft_times.push_back(prefill_ms);  // Time to first token = prefill time
        }

        if (config.verbose) {
            double pp_tok_s = config.prompt_len / (prefill_ms / 1000.0);
            double tg_tok_s = config.gen_len / (decode_ms / 1000.0);
            double tg_raw_tok_s = config.gen_len / (total_forward_ms / 1000.0);
            std::cout << "pp=" << std::fixed << std::setprecision(1) << pp_tok_s
                      << " tok/s, tg=" << tg_tok_s << " tok/s (raw=" << tg_raw_tok_s << ")\n";
        }
    }

    // Calculate results
    auto prefill_stats = calculate_stats(prefill_times);
    auto decode_stats = calculate_stats(decode_times);
    auto decode_raw_stats = calculate_stats(decode_raw_times);
    auto ttft_stats = calculate_stats(ttft_times);

    result.prefill_tok_per_sec = config.prompt_len / (prefill_stats.mean / 1000.0);
    result.decode_tok_per_sec = config.gen_len / (decode_stats.mean / 1000.0);
    result.decode_raw_tok_per_sec = config.gen_len / (decode_raw_stats.mean / 1000.0);
    result.time_to_first_token_ms = ttft_stats.mean;
    result.total_time_ms = prefill_stats.mean + decode_stats.mean;

    // Print results
    std::cout << "\nGranite Results:\n";
    std::cout << "  Prefill:      " << std::fixed << std::setprecision(2)
              << result.prefill_tok_per_sec << " tok/s\n";
    std::cout << "  Decode (e2e): " << std::fixed << std::setprecision(2)
              << result.decode_tok_per_sec << " tok/s"
              << " (includes buffer map + CPU argmax)\n";
    std::cout << "  Decode (raw): " << std::fixed << std::setprecision(2)
              << result.decode_raw_tok_per_sec << " tok/s"
              << " (GPU forward only)\n";
    std::cout << "  TTFT:         " << std::fixed << std::setprecision(2)
              << result.time_to_first_token_ms << " ms\n";

    // Print Metal profiling stats if available
#ifdef GRANITE_HAS_METAL
    if (gpu && gpu->is_initialized()) {
        uint64_t dispatches, syncs, cmd_buffers;
        double sync_time_ms;
        gpu->get_profiling_stats(dispatches, syncs, sync_time_ms, cmd_buffers);

        int total_tokens = config.prompt_len + config.gen_len * config.runs;
        double dispatches_per_token = static_cast<double>(dispatches) / total_tokens;
        double syncs_per_token = static_cast<double>(syncs) / total_tokens;

        std::cout << "\n  Metal Profiling:\n";
        std::cout << "    Total dispatches:   " << dispatches
                  << " (" << std::fixed << std::setprecision(1) << dispatches_per_token << " per token)\n";
        std::cout << "    Total syncs:        " << syncs
                  << " (" << std::fixed << std::setprecision(2) << syncs_per_token << " per token)\n";
        std::cout << "    Sync time:          " << std::fixed << std::setprecision(2) << sync_time_ms << " ms\n";
        std::cout << "    Cmd buffers:        " << cmd_buffers << "\n";

        gpu->enable_profiling(false);
    }
#endif

    return result;
}

// Parse llama-bench output
BenchmarkResult parse_llama_bench_output(const std::string& output, const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.name = "llama.cpp";
    result.prompt_tokens = config.prompt_len;
    result.generated_tokens = config.gen_len;

    // Parse lines like:
    // | llama 1B Q4_K | ... | pp128 | 3921.72 ± 130.45 |
    // | llama 1B Q4_K | ... | tg128 |  256.74 ±   1.26 |
    // Note: pp/tg column may have leading spaces: "|           pp128 |"
    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line)) {
        // Look for pp (prompt processing) results - check if line contains "pp" followed by digits
        bool is_pp = line.find("pp") != std::string::npos &&
                     line.find("pp") < line.rfind("|", line.rfind("|") - 1);
        bool is_tg = line.find("tg") != std::string::npos &&
                     line.find("tg") < line.rfind("|", line.rfind("|") - 1);

        if (is_pp || is_tg) {
            // Find the tok/s value in the last column (between last two |)
            size_t pos = line.rfind("|");
            if (pos != std::string::npos && pos > 0) {
                size_t prev_pos = line.rfind("|", pos - 1);
                if (prev_pos != std::string::npos) {
                    std::string val_str = line.substr(prev_pos + 1, pos - prev_pos - 1);
                    // Extract number before ±
                    size_t pm_pos = val_str.find("±");
                    if (pm_pos != std::string::npos) {
                        val_str = val_str.substr(0, pm_pos);
                    }
                    // Trim whitespace
                    size_t start = val_str.find_first_not_of(" \t");
                    size_t end = val_str.find_last_not_of(" \t");
                    if (start != std::string::npos && end != std::string::npos) {
                        val_str = val_str.substr(start, end - start + 1);
                    }
                    try {
                        double value = std::stod(val_str);
                        if (is_pp) {
                            result.prefill_tok_per_sec = value;
                        } else if (is_tg) {
                            result.decode_tok_per_sec = value;
                        }
                    } catch (...) {}
                }
            }
        }
    }

    return result;
}

// Run llama-bench
BenchmarkResult run_llama_bench(const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.name = "llama.cpp";

    if (config.llama_bench_path.empty()) {
        std::cout << "\n=== Skipping llama.cpp benchmark (no --llama-bench path) ===\n";
        return result;
    }

    std::cout << "\n=== Running llama.cpp Benchmark ===\n";
    std::cout << "Binary: " << config.llama_bench_path << "\n";

    // Build command
    std::ostringstream cmd;
    cmd << config.llama_bench_path
        << " -m " << config.model_path
        << " -p " << config.prompt_len
        << " -n " << config.gen_len
        << " -r " << config.runs
        << " -ngl 99"  // Offload all layers to GPU
        << " 2>&1";

    std::cout << "Command: " << cmd.str() << "\n\n";

    // Run llama-bench and capture output
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        std::cerr << "Failed to run llama-bench\n";
        return result;
    }

    std::string output;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
        std::cout << buffer;  // Echo output
    }

    int status = pclose(pipe);
    if (status != 0) {
        std::cerr << "llama-bench exited with status " << status << "\n";
    }

    // Parse output
    result = parse_llama_bench_output(output, config);

    std::cout << "\nllama.cpp Results:\n";
    std::cout << "  Prefill: " << std::fixed << std::setprecision(2)
              << result.prefill_tok_per_sec << " tok/s\n";
    std::cout << "  Decode:  " << std::fixed << std::setprecision(2)
              << result.decode_tok_per_sec << " tok/s\n";

    return result;
}

// Print comparison table
void print_comparison(const BenchmarkResult& granite, const BenchmarkResult& llama) {
    std::cout << "\n";
    std::cout << "╔═════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                     Performance Comparison                          ║\n";
    std::cout << "╠════════════════════╦═════════════════╦═════════════════╦════════════╣\n";
    std::cout << "║     Metric         ║     Granite     ║    llama.cpp    ║    Ratio   ║\n";
    std::cout << "╠════════════════════╬═════════════════╬═════════════════╬════════════╣\n";

    auto print_row = [](const char* name, double granite_val, double llama_val, const char* unit) {
        double ratio = (llama_val > 0) ? granite_val / llama_val : 0;
        std::cout << "║ " << std::setw(18) << std::left << name << " ║ "
                  << std::setw(12) << std::right << std::fixed << std::setprecision(2) << granite_val
                  << " " << std::setw(2) << unit << " ║ "
                  << std::setw(12) << std::right << std::fixed << std::setprecision(2) << llama_val
                  << " " << std::setw(2) << unit << " ║ "
                  << std::setw(8) << std::right << std::fixed << std::setprecision(2) << ratio << "x ║\n";
    };

    print_row("Prefill (tok/s)", granite.prefill_tok_per_sec, llama.prefill_tok_per_sec, "");
    print_row("Decode e2e (tok/s)", granite.decode_tok_per_sec, llama.decode_tok_per_sec, "");
    print_row("Decode raw (tok/s)", granite.decode_raw_tok_per_sec, llama.decode_tok_per_sec, "");

    std::cout << "╚════════════════════╩═════════════════╩═════════════════╩════════════╝\n";
    std::cout << "\nNote: 'Decode e2e' includes buffer map/unmap + CPU argmax overhead.\n";
    std::cout << "      'Decode raw' measures GPU forward pass time only.\n";

    // Summary using raw decode for fairer comparison
    double prefill_ratio = 0, decode_ratio = 0;
    if (llama.prefill_tok_per_sec > 0 && granite.prefill_tok_per_sec > 0) {
        prefill_ratio = granite.prefill_tok_per_sec / llama.prefill_tok_per_sec;
    }
    if (llama.decode_tok_per_sec > 0 && granite.decode_raw_tok_per_sec > 0) {
        decode_ratio = granite.decode_raw_tok_per_sec / llama.decode_tok_per_sec;
    }

    if (prefill_ratio > 0 || decode_ratio > 0) {
        std::cout << "\nSummary:\n";
        if (prefill_ratio > 0) {
            std::cout << "  Prefill: Granite is " << std::fixed << std::setprecision(2)
                      << prefill_ratio << "x " << (prefill_ratio >= 1.0 ? "faster" : "slower") << "\n";
        }
        if (decode_ratio > 0) {
            std::cout << "  Decode:  Granite is " << std::fixed << std::setprecision(2)
                      << decode_ratio << "x " << (decode_ratio >= 1.0 ? "faster" : "slower")
                      << " (comparing raw forward to llama.cpp)\n";
        }
    }
}

int main(int argc, char** argv) {
    auto config = parse_args(argc, argv);

    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Granite vs llama.cpp Performance Benchmark             ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    BenchmarkResult granite_result, llama_result;

    // Run Granite benchmark
    if (!config.compare_only) {
        granite_result = run_granite_benchmark(config);
    }

    // Run llama.cpp benchmark
    llama_result = run_llama_bench(config);

    // Print comparison
    if (!config.compare_only && !config.llama_bench_path.empty()) {
        print_comparison(granite_result, llama_result);
    }

    return 0;
}
