#include <granite/granite.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace granite;

namespace {

bool parse_backend_arg(const std::string& value, BackendType& out_type) {
    if (value == "cpu") {
        out_type = BackendType::CPU;
        return true;
    }
    if (value == "metal") {
        out_type = BackendType::Metal;
        return true;
    }
    if (value == "vulkan") {
        out_type = BackendType::Vulkan;
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model.gguf> [options]\n";
        std::cout << "  --prompt <text>       Prompt text (default: \"Hello\")\n";
        std::cout << "  --iterations <n>      Iterations (default: 50)\n";
        std::cout << "  --max-tokens <n>      Tokens per iteration (default: 16)\n";
        std::cout << "  --backend <cpu|metal|vulkan>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = "Hello";
    int iterations = 50;
    int max_tokens = 16;
    BackendType backend = BackendType::CPU;
    bool backend_set = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--backend" && i + 1 < argc) {
            BackendType parsed = BackendType::CPU;
            if (parse_backend_arg(argv[++i], parsed)) {
                backend = parsed;
                backend_set = true;
            } else {
                std::cerr << "Invalid --backend value\n";
            }
        }
    }

    Config config = Config::Balanced();
    if (backend_set) {
        config.preferred_backend = backend;
    }

    auto runner_result = LLMRunner::load(model_path, config);
    if (!runner_result.ok()) {
        std::cerr << "Failed to load model: " << runner_result.error().message() << "\n";
        return 1;
    }

    auto runner = std::move(runner_result).take();
    GenerationConfig gen_config;
    gen_config.max_tokens = max_tokens;
    gen_config.do_sample = false;

    std::cout << "Stress test: " << iterations << " iterations, " << max_tokens
              << " tokens per iteration\n";

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto result = runner->generate(prompt, gen_config);
        if (!result.ok()) {
            std::cerr << "Generation failed at iter " << i << ": "
                      << result.error().message() << "\n";
            return 1;
        }
        if ((i + 1) % 10 == 0) {
            std::cout << "Completed " << (i + 1) << " iterations\n";
        }
        runner->reset();
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "Done in " << elapsed << " seconds\n";

    return 0;
}
