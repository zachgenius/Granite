// Detailed benchmark matching llama.cpp methodology
// Measures prefill (pp) and decode (tg) performance

#include <granite/granite.h>

#ifdef GRANITE_HAS_METAL
#include <Metal/Metal.hpp>
#include <granite/metal_compute.h>
#endif

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace granite;
using Clock = std::chrono::high_resolution_clock;

template<typename Func>
double benchmark_median_ms(Func&& f, int warmup, int iterations) {
    for (int i = 0; i < warmup; i++) f();

    std::vector<double> times;
    times.reserve(iterations);
    for (int i = 0; i < iterations; i++) {
        auto start = Clock::now();
        f();
        auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(times.begin(), times.end());
    return times[iterations / 2];
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [warmup] [iters]\n";
        return 1;
    }

    const char* model_path = argv[1];
    int warmup = argc > 2 ? std::atoi(argv[2]) : 2;
    int iters = argc > 3 ? std::atoi(argv[3]) : 5;

    // Suppress most logging
    granite::init_logging(spdlog::level::warn);

    std::cout << "============================================================\n";
    std::cout << "  Granite Detailed Benchmark (llama.cpp compatible format)\n";
    std::cout << "============================================================\n\n";

    // Use LLMRunner which handles all initialization properly
    auto runner_result = LLMRunner::load(model_path);
    if (!runner_result.ok()) {
        std::cerr << "Failed to load model: " << runner_result.error().message() << "\n";
        return 1;
    }
    auto runner = std::move(runner_result).take();

    auto& config = runner->config();
    std::cout << "Model: " << config.num_layers << " layers, "
              << config.hidden_dim << " hidden, "
              << config.num_heads << "/" << config.num_kv_heads << " heads\n";

#ifdef GRANITE_HAS_METAL
    auto* gpu = get_metal_compute();
    if (gpu && gpu->device()) {
        std::cout << "Device: " << gpu->device()->name()->utf8String() << "\n";
    }
#endif

    std::cout << "Warmup: " << warmup << ", Iterations: " << iters << "\n\n";

    std::vector<int> prefill_sizes = {32, 128, 256, 512};
    std::vector<int> decode_sizes = {32, 64, 128};

    std::cout << "| test           |     tokens |    time (ms) |         t/s |\n";
    std::cout << "| -------------- | ---------: | -----------: | ----------: |\n";

    // ========================================
    // Prefill benchmark (using generate with max_tokens=0)
    // ========================================
    for (int pp : prefill_sizes) {
        // Create a prompt with pp tokens worth of text
        // Approximate: each token is ~4 chars
        std::string prompt(pp * 4, 'x');

        auto prefill_fn = [&]() {
            runner->reset();
            // Generate 0 new tokens = just prefill
            GenerationConfig gen_config;
            gen_config.max_tokens = 0;
            (void)runner->generate(prompt, gen_config);
#ifdef GRANITE_HAS_METAL
            if (gpu) gpu->sync();
#endif
        };

        double time_ms = benchmark_median_ms(prefill_fn, warmup, iters);
        double tps = (pp * 1000.0) / time_ms;

        std::cout << "| pp" << std::setw(12) << pp << " | "
                  << std::setw(10) << pp << " | "
                  << std::setw(12) << std::fixed << std::setprecision(2) << time_ms << " | "
                  << std::setw(11) << std::setprecision(2) << tps << " |\n";
    }

    // ========================================
    // Decode benchmark
    // ========================================
    for (int tg : decode_sizes) {
        // Short prompt, then generate tg tokens
        std::string prompt = "Hello";

        auto decode_fn = [&]() {
            runner->reset();
            GenerationConfig gen_config;
            gen_config.max_tokens = tg;
            gen_config.temperature = 0.0f;  // Greedy for consistency
            (void)runner->generate(prompt, gen_config);
#ifdef GRANITE_HAS_METAL
            if (gpu) gpu->sync();
#endif
        };

        double time_ms = benchmark_median_ms(decode_fn, warmup, iters);
        double tps = (tg * 1000.0) / time_ms;

        std::cout << "| tg" << std::setw(12) << tg << " | "
                  << std::setw(10) << tg << " | "
                  << std::setw(12) << std::fixed << std::setprecision(2) << time_ms << " | "
                  << std::setw(11) << std::setprecision(2) << tps << " |\n";
    }

    std::cout << "\n";
    return 0;
}
