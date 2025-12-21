#include <granite/granite.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cstdlib>
#include <sstream>
#include <algorithm>

using namespace granite;
using Clock = std::chrono::high_resolution_clock;

// Benchmark helper
template<typename Func>
double benchmark_ms(Func&& f, int warmup = 2, int iterations = 10,
                    std::vector<double>* samples = nullptr) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        f();
    }

    // Measure
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        auto start = Clock::now();
        f();
        auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    if (samples) {
        *samples = times;
    }

    // Return median
    std::sort(times.begin(), times.end());
    return times[iterations / 2];
}

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void benchmark_matmul(IComputeBackend* backend) {
    print_header("Matrix Multiplication Benchmark");

    std::vector<std::tuple<int64_t, int64_t, int64_t>> sizes = {
        {1, 2048, 2048},      // Single token, hidden->hidden
        {1, 2048, 5632},      // Single token, hidden->intermediate (gate/up)
        {1, 5632, 2048},      // Single token, intermediate->hidden (down)
        {1, 2048, 32000},     // Single token, hidden->vocab (output)
        {128, 2048, 2048},    // Batch of 128 tokens
    };

    std::cout << std::setw(10) << "M" << std::setw(10) << "K" << std::setw(10) << "N"
              << std::setw(15) << "Time (ms)" << std::setw(15) << "GFLOPS" << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (auto& [M, K, N] : sizes) {
        // Allocate tensors
        std::vector<int64_t> a_shape = {M, K};
        std::vector<int64_t> b_shape = {K, N};
        std::vector<int64_t> c_shape = {M, N};
        auto a_result = Tensor::allocate(a_shape, DataType::FP32, backend);
        auto b_result = Tensor::allocate(b_shape, DataType::FP32, backend);
        auto c_result = Tensor::allocate(c_shape, DataType::FP32, backend);

        if (!a_result.ok() || !b_result.ok() || !c_result.ok()) {
            std::cerr << "Failed to allocate tensors for " << M << "x" << K << "x" << N << "\n";
            continue;
        }

        auto A = std::move(a_result).take();
        auto B = std::move(b_result).take();
        auto C = std::move(c_result).take();

        // Initialize with random values
        auto map_a = backend->map_buffer(A.buffer());
        auto map_b = backend->map_buffer(B.buffer());
        if (map_a.ok() && map_b.ok()) {
            auto* a_data = static_cast<float*>(map_a.value());
            auto* b_data = static_cast<float*>(map_b.value());
            for (int i = 0; i < M * K; i++) a_data[i] = 0.01f * (i % 100);
            for (int i = 0; i < K * N; i++) b_data[i] = 0.01f * (i % 100);
            backend->unmap_buffer(A.buffer());
            backend->unmap_buffer(B.buffer());
        }

        // Benchmark CPU matmul
        auto cpu_matmul = [&]() {
            auto map_a = backend->map_buffer(A.buffer());
            auto map_b = backend->map_buffer(B.buffer());
            auto map_c = backend->map_buffer(C.buffer());

            const float* a = static_cast<const float*>(map_a.value());
            const float* b = static_cast<const float*>(map_b.value());
            float* c = static_cast<float*>(map_c.value());

            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++) {
                        sum += a[m * K + k] * b[k * N + n];
                    }
                    c[m * N + n] = sum;
                }
            }

            backend->unmap_buffer(A.buffer());
            backend->unmap_buffer(B.buffer());
            backend->unmap_buffer(C.buffer());
        };

        double time_ms = benchmark_ms(cpu_matmul, 1, 5);
        double flops = 2.0 * M * N * K;
        double gflops = (flops / 1e9) / (time_ms / 1000.0);

        std::cout << std::setw(10) << M << std::setw(10) << K << std::setw(10) << N
                  << std::setw(15) << std::fixed << std::setprecision(2) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << gflops << "\n";
    }
}

std::vector<int64_t> parse_seq_lens(const std::string& value) {
    std::vector<int64_t> seq_lens;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        char* end = nullptr;
        long len = std::strtol(token.c_str(), &end, 10);
        if (!end || *end != '\0' || len <= 0) {
            seq_lens.clear();
            return seq_lens;
        }
        seq_lens.push_back(static_cast<int64_t>(len));
    }
    return seq_lens;
}

void benchmark_inference(
    const std::string& model_path,
    IComputeBackend* backend,
    const std::vector<int64_t>& seq_lens,
    bool full_logits = false,
    uint32_t prefill_chunk_size = 0,
    bool prefill_chunk_set = false,
    size_t kv_cache_max_seq = 0,
    bool kv_cache_max_seq_set = false,
    bool enable_profiling = false) {
    print_header("LLM Inference Benchmark");

    Config runtime_config = Config::Balanced();
    if (prefill_chunk_set) {
        runtime_config.prefill_chunk_size = prefill_chunk_size;
        std::cout << "Prefill chunk size: " << runtime_config.prefill_chunk_size << "\n";
    } else if (const char* chunk_env = std::getenv("GRANITE_PREFILL_CHUNK_SIZE")) {
        char* end = nullptr;
        long chunk = std::strtol(chunk_env, &end, 10);
        if (end && *end == '\0' && chunk > 0) {
            runtime_config.prefill_chunk_size = static_cast<uint32_t>(chunk);
            std::cout << "Prefill chunk size: " << runtime_config.prefill_chunk_size << "\n";
        }
    }

    if (kv_cache_max_seq_set) {
        runtime_config.kv_cache_max_seq = kv_cache_max_seq;
        std::cout << "KV cache max seq: " << runtime_config.kv_cache_max_seq << "\n";
    } else if (const char* kv_env = std::getenv("GRANITE_KV_CACHE_MAX_SEQ")) {
        char* end = nullptr;
        long kv = std::strtol(kv_env, &end, 10);
        if (end && *end == '\0' && kv > 0) {
            runtime_config.kv_cache_max_seq = static_cast<size_t>(kv);
            kv_cache_max_seq_set = true;
            kv_cache_max_seq = runtime_config.kv_cache_max_seq;
            std::cout << "KV cache max seq: " << runtime_config.kv_cache_max_seq << "\n";
        }
    }

    if (enable_profiling) {
        runtime_config.enable_profiling = true;
        std::cout << "Profiling: enabled\n";
    }

    // Load model
    std::cout << "Loading model: " << model_path << "\n";
    auto model_result = TransformerModel::load(model_path, backend, runtime_config);
    if (!model_result.ok()) {
        std::cerr << "Failed to load model: " << model_result.error().message() << "\n";
        return;
    }
    auto model = std::move(model_result).take();

    auto& config = model.config();
    std::cout << "Model: " << config.num_layers << " layers, "
              << config.hidden_dim << " hidden, "
              << config.num_heads << " heads\n";
    std::cout << "Logits mode: " << (full_logits ? "ALL tokens" : "last token only") << "\n";

    size_t cache_max_seq = 512;
    if (kv_cache_max_seq_set) {
        cache_max_seq = kv_cache_max_seq;
    }
    if (cache_max_seq > static_cast<size_t>(config.max_seq_len)) {
        cache_max_seq = static_cast<size_t>(config.max_seq_len);
    }

    // Allocate KV cache
    auto kv_result = KVCache::allocate(config, static_cast<int>(cache_max_seq), backend);
    if (!kv_result.ok()) {
        std::cerr << "Failed to allocate KV cache\n";
        return;
    }
    auto kv_cache = std::move(kv_result).take();

    // Enable GPU mode and allocate GPU KV cache
#ifdef GRANITE_HAS_METAL
    model.set_use_gpu(true);
    // Control logits computation: full_logits=false means last-token-only (faster)
    model.set_prefill_last_token_only(!full_logits);
    auto gpu_cache_result = model.allocate_gpu_kv_cache(static_cast<int>(cache_max_seq));
    if (gpu_cache_result.ok()) {
        std::cout << "GPU KV cache allocated\n";
    } else {
        std::cout << "GPU KV cache allocation failed, using CPU cache\n";
    }
#endif

    // Benchmark prefill with different sequence lengths
    std::cout << "\nPrefill (prompt processing):\n";
    std::cout << std::setw(15) << "Seq Length" << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Tokens/sec" << "\n";
    std::cout << std::string(45, '-') << "\n";
    std::cout << std::flush;

    for (int64_t seq_len : seq_lens) {  // Match llama.cpp benchmark sizes by default
        // Create token tensor
        std::vector<int64_t> shape = {1, seq_len};
        auto ids_result = Tensor::allocate(shape, DataType::INT32, backend);
        if (!ids_result.ok()) continue;
        auto ids = std::move(ids_result).take();

        // Fill with dummy tokens
        auto map_ids = backend->map_buffer(ids.buffer());
        if (map_ids.ok()) {
            auto* ptr = static_cast<int32_t*>(map_ids.value());
            for (int i = 0; i < seq_len; i++) ptr[i] = 1;  // BOS token
            backend->unmap_buffer(ids.buffer());
        }

        // Clear cache
        kv_cache.clear();
#ifdef GRANITE_HAS_METAL
        if (model.gpu_kv_cache()) {
            model.gpu_kv_cache()->clear();
        }
#endif

        // Benchmark
        auto prefill = [&]() {
            kv_cache.clear();
#ifdef GRANITE_HAS_METAL
            if (model.gpu_kv_cache()) {
                model.gpu_kv_cache()->clear();
            }
#endif
            auto result = model.forward(ids, &kv_cache, 0);
            return result.ok();
        };

        std::vector<double> samples;
        double time_ms = benchmark_ms(prefill, 1, 3, enable_profiling ? &samples : nullptr);
        double tokens_per_sec = (seq_len / time_ms) * 1000.0;

        std::cout << std::setw(15) << seq_len
                  << std::setw(15) << std::fixed << std::setprecision(2) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(1) << tokens_per_sec
                  << "\n";
        if (enable_profiling && !samples.empty()) {
            double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
            double avg = sum / static_cast<double>(samples.size());
            double min = *std::min_element(samples.begin(), samples.end());
            double max = *std::max_element(samples.begin(), samples.end());
            std::cout << "  prefill stats (ms): avg=" << std::fixed << std::setprecision(2) << avg
                      << " min=" << min << " max=" << max << "\n";
        }
    }

    // Benchmark single-token generation (decode)
    std::cout << "\nDecode (token generation):\n";
    std::cout << std::setw(15) << "Cache Length" << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Tokens/sec" << "\n";
    std::cout << std::string(45, '-') << "\n";

    // First, do a prefill to populate the cache
    {
        kv_cache.clear();
#ifdef GRANITE_HAS_METAL
        if (model.gpu_kv_cache()) {
            model.gpu_kv_cache()->clear();
        }
#endif
        std::vector<int64_t> shape = {1, 1};
        auto ids_result = Tensor::allocate(shape, DataType::INT32, backend);
        if (ids_result.ok()) {
            auto ids = std::move(ids_result).take();
            auto map_ids = backend->map_buffer(ids.buffer());
            if (map_ids.ok()) {
                static_cast<int32_t*>(map_ids.value())[0] = 1;
                backend->unmap_buffer(ids.buffer());
            }
            (void)model.forward(ids, &kv_cache, 0);
        }
    }

    // Benchmark single token generation at different cache lengths
    for (int cache_len : {1, 16, 64, 128}) {  // Reduced from 256 for faster benchmarking
        // Get current cache length (prefer GPU cache if available)
        auto get_cache_len = [&]() -> int {
#ifdef GRANITE_HAS_METAL
            if (model.gpu_kv_cache() && model.gpu_kv_cache()->is_allocated()) {
                return model.gpu_kv_cache()->seq_len();
            }
#endif
            return kv_cache.seq_len();
        };

        // Extend cache to desired length by generating tokens
        while (get_cache_len() < cache_len) {
            auto result = model.forward_single(1, kv_cache);
            if (!result.ok()) break;
        }

        // Benchmark single token
        auto decode = [&]() {
            // Note: This modifies the cache, so results may vary
            auto result = model.forward_single(1, kv_cache);
            return result.ok();
        };

        int actual_len = get_cache_len();
        std::vector<double> samples;
        double time_ms = benchmark_ms(decode, 1, 3, enable_profiling ? &samples : nullptr);
        double tokens_per_sec = 1000.0 / time_ms;

        std::cout << std::setw(15) << actual_len
                  << std::setw(15) << std::fixed << std::setprecision(2) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(1) << tokens_per_sec << "\n";
        if (enable_profiling && !samples.empty()) {
            double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
            double avg = sum / static_cast<double>(samples.size());
            double min = *std::min_element(samples.begin(), samples.end());
            double max = *std::max_element(samples.begin(), samples.end());
            std::cout << "  decode stats (ms): avg=" << std::fixed << std::setprecision(2) << avg
                      << " min=" << min << " max=" << max << "\n";
        }
    }
}

void benchmark_memory(IComputeBackend* backend) {
    print_header("Memory Bandwidth Benchmark");

    std::vector<size_t> sizes = {
        1024 * 1024,        // 1 MB
        16 * 1024 * 1024,   // 16 MB
        64 * 1024 * 1024,   // 64 MB
        256 * 1024 * 1024,  // 256 MB
    };

    std::cout << std::setw(15) << "Size (MB)" << std::setw(15) << "Read (GB/s)"
              << std::setw(15) << "Write (GB/s)" << "\n";
    std::cout << std::string(45, '-') << "\n";

    for (size_t size : sizes) {
        BufferDesc desc;
        desc.size = size;
        desc.memory_type = MemoryType::Shared;

        auto buf_result = backend->create_buffer(desc);
        if (!buf_result.ok()) {
            std::cerr << "Failed to allocate " << size / (1024*1024) << " MB buffer\n";
            continue;
        }
        auto buf = buf_result.value();

        // Initialize
        auto map = backend->map_buffer(buf);
        if (!map.ok()) continue;
        std::memset(map.value(), 0, size);
        backend->unmap_buffer(buf);

        // Benchmark read
        volatile float sum = 0;
        auto read_bench = [&]() {
            auto map = backend->map_buffer(buf);
            const float* data = static_cast<const float*>(map.value());
            float local_sum = 0;
            for (size_t i = 0; i < size / sizeof(float); i += 16) {
                local_sum += data[i];
            }
            sum = local_sum;
            backend->unmap_buffer(buf);
        };

        double read_ms = benchmark_ms(read_bench, 2, 5);
        double read_bw = (size / 1e9) / (read_ms / 1000.0);

        // Benchmark write
        auto write_bench = [&]() {
            auto map = backend->map_buffer(buf);
            float* data = static_cast<float*>(map.value());
            for (size_t i = 0; i < size / sizeof(float); i += 16) {
                data[i] = 1.0f;
            }
            backend->unmap_buffer(buf);
        };

        double write_ms = benchmark_ms(write_bench, 2, 5);
        double write_bw = (size / 1e9) / (write_ms / 1000.0);

        std::cout << std::setw(15) << size / (1024*1024)
                  << std::setw(15) << std::fixed << std::setprecision(2) << read_bw
                  << std::setw(15) << std::fixed << std::setprecision(2) << write_bw << "\n";

        backend->destroy_buffer(buf);
    }
}

int main(int argc, char* argv[]) {
    // Respect SPDLOG_LEVEL environment variable, default to warn
    const char* log_level = std::getenv("SPDLOG_LEVEL");
    if (log_level) {
        std::string level_str(log_level);
        if (level_str == "debug") granite::init_logging(spdlog::level::debug);
        else if (level_str == "info") granite::init_logging(spdlog::level::info);
        else if (level_str == "warn") granite::init_logging(spdlog::level::warn);
        else if (level_str == "error") granite::init_logging(spdlog::level::err);
        else granite::init_logging(spdlog::level::warn);
    } else {
        granite::init_logging(spdlog::level::warn);
    }

    std::cout << "Granite Performance Benchmark\n";
    std::cout << "==============================\n";

    // Create backend
    auto backend = create_default_backend();
    if (!backend) {
        std::cerr << "Failed to create backend\n";
        return 1;
    }
    backend->initialize();

    auto caps = backend->get_capabilities();
    std::cout << "\nBackend: " << (backend->get_type() == BackendType::Metal ? "Metal" : "CPU") << "\n";
    std::cout << "Device: " << caps.name << "\n";
    std::cout << "Max buffer size: " << caps.max_buffer_size / (1024*1024*1024) << " GB\n";
    std::cout << "FP16 support: " << (caps.supports_fp16 ? "yes" : "no") << "\n";

    // Run benchmarks
    benchmark_memory(backend.get());
    benchmark_matmul(backend.get());

    // If model path provided, run inference benchmark
    if (argc > 1) {
        std::string model_path;
        bool full_logits = false;
        std::vector<int64_t> seq_lens = {32, 128, 256, 512};
        uint32_t prefill_chunk_size = 0;
        bool prefill_chunk_set = false;
        size_t kv_cache_max_seq = 0;
        bool kv_cache_max_seq_set = false;
        bool enable_profiling = false;

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--full-logits") {
                full_logits = true;
            } else if (arg == "--seq-lens" && i + 1 < argc) {
                auto parsed = parse_seq_lens(argv[++i]);
                if (!parsed.empty()) {
                    seq_lens = std::move(parsed);
                } else {
                    std::cerr << "Invalid --seq-lens value, using defaults.\n";
                }
            } else if (arg == "--prefill-chunk-size" && i + 1 < argc) {
                char* end = nullptr;
                long chunk = std::strtol(argv[++i], &end, 10);
                if (end && *end == '\0' && chunk >= 0) {
                    prefill_chunk_size = static_cast<uint32_t>(chunk);
                    prefill_chunk_set = true;
                } else {
                    std::cerr << "Invalid --prefill-chunk-size value, ignoring.\n";
                }
            } else if (arg == "--kv-cache-max-seq" && i + 1 < argc) {
                char* end = nullptr;
                long kv = std::strtol(argv[++i], &end, 10);
                if (end && *end == '\0' && kv > 0) {
                    kv_cache_max_seq = static_cast<size_t>(kv);
                    kv_cache_max_seq_set = true;
                } else {
                    std::cerr << "Invalid --kv-cache-max-seq value, ignoring.\n";
                }
            } else if (arg == "--profile") {
                enable_profiling = true;
            } else if (arg[0] != '-') {
                model_path = arg;
            }
        }

        if (!model_path.empty()) {
            benchmark_inference(model_path, backend.get(), seq_lens, full_logits,
                                prefill_chunk_size, prefill_chunk_set,
                                kv_cache_max_seq, kv_cache_max_seq_set,
                                enable_profiling);
        } else {
            std::cout << "\nUsage: " << argv[0] << " [model.gguf] [--full-logits]\n";
            std::cout << "Provide a GGUF model path to run inference benchmarks.\n";
            std::cout << "  --full-logits: Compute logits for all tokens (default: last token only)\n";
            std::cout << "  --seq-lens <a,b,c>: Override prefill prompt lengths\n";
            std::cout << "  --prefill-chunk-size <n>: Override prefill chunk size (0 disables)\n";
            std::cout << "  --kv-cache-max-seq <n>: Override KV cache max sequence length\n";
            std::cout << "  --profile: Print per-section timing stats\n";
        }
    } else {
        std::cout << "\nUsage: " << argv[0] << " [model.gguf] [--full-logits]\n";
        std::cout << "Provide a GGUF model path to run inference benchmarks.\n";
        std::cout << "  --full-logits: Compute logits for all tokens (default: last token only)\n";
        std::cout << "  --seq-lens <a,b,c>: Override prefill prompt lengths\n";
        std::cout << "  --prefill-chunk-size <n>: Override prefill chunk size (0 disables)\n";
        std::cout << "  --kv-cache-max-seq <n>: Override KV cache max sequence length\n";
        std::cout << "  --profile: Print per-section timing stats\n";
    }

    backend->shutdown();
    return 0;
}
