#include <granite/granite.h>
#ifdef GRANITE_HAS_METAL
#include <granite/metal_compute.h>
#endif
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cstdlib>

using namespace granite;
using Clock = std::chrono::high_resolution_clock;

// Benchmark helper
template<typename Func>
double benchmark_ms(Func&& f, int warmup = 2, int iterations = 10) {
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

void benchmark_inference(const std::string& model_path, IComputeBackend* backend) {
    print_header("LLM Inference Benchmark");

    // Load model
    std::cout << "Loading model: " << model_path << "\n";
    auto model_result = TransformerModel::load(model_path, backend);
    if (!model_result.ok()) {
        std::cerr << "Failed to load model: " << model_result.error().message() << "\n";
        return;
    }
    auto model = std::move(model_result).take();

    auto& config = model.config();
    std::cout << "Model: " << config.num_layers << " layers, "
              << config.hidden_dim << " hidden, "
              << config.num_heads << " heads\n";

    // Allocate KV cache
    auto kv_result = KVCache::allocate(config, 512, backend);
    if (!kv_result.ok()) {
        std::cerr << "Failed to allocate KV cache\n";
        return;
    }
    auto kv_cache = std::move(kv_result).take();

    // Enable GPU mode and allocate GPU KV cache
#ifdef GRANITE_HAS_METAL
    model.set_use_gpu(true);
    // Match llama.cpp prompt processing behavior (logits for last token only).
    model.set_prefill_last_token_only(true);
    auto gpu_cache_result = model.allocate_gpu_kv_cache(512);
    if (gpu_cache_result.ok()) {
        std::cout << "GPU KV cache allocated\n";
    } else {
        std::cout << "GPU KV cache allocation failed, using CPU cache\n";
    }
#endif

    // Benchmark prefill with different sequence lengths
    std::cerr << "[DEBUG] Starting prefill benchmark...\n" << std::flush;
    std::cout << "\nPrefill (prompt processing):\n";
    std::cout << std::setw(15) << "Seq Length" << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Tokens/sec" << "\n";
    std::cout << std::string(45, '-') << "\n";
    std::cout << std::flush;

    for (int64_t seq_len : {32, 128, 256, 512}) {  // Match llama.cpp benchmark sizes
        std::cerr << "[DEBUG] Testing prefill seq_len=" << seq_len << "...\n" << std::flush;
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
#ifdef GRANITE_HAS_METAL
        auto* gpu = get_metal_compute();
        if (gpu) {
            gpu->enable_profiling(true);
            gpu->reset_profiling_stats();
        }
#endif

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

        double time_ms = benchmark_ms(prefill, 1, 3);
        double tokens_per_sec = (seq_len / time_ms) * 1000.0;

        std::cout << std::setw(15) << seq_len
                  << std::setw(15) << std::fixed << std::setprecision(2) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(1) << tokens_per_sec;

#ifdef GRANITE_HAS_METAL
        // Print profiling stats (per single run, not averaged over benchmark iterations)
        if (gpu) {
            uint64_t dispatches, syncs, cmd_buffers;
            double sync_time;
            gpu->get_profiling_stats(dispatches, syncs, sync_time, cmd_buffers);
            // Stats are cumulative over 3 iterations, divide by 3
            std::cout << " [" << dispatches/3 << " dispatches, "
                      << syncs/3 << " syncs, "
                      << cmd_buffers/3 << " cmdbufs]";
            gpu->enable_profiling(false);
        }
#endif
        std::cout << "\n";
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
    std::cerr << "[DEBUG] Starting decode benchmark...\n" << std::flush;
    for (int cache_len : {1, 16, 64, 128}) {  // Reduced from 256 for faster benchmarking
        std::cerr << "[DEBUG] Extending cache to " << cache_len << "...\n" << std::flush;
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
        std::cerr << "[DEBUG] Benchmarking decode at len=" << get_cache_len() << "...\n" << std::flush;
        auto decode = [&]() {
            // Note: This modifies the cache, so results may vary
            auto result = model.forward_single(1, kv_cache);
            return result.ok();
        };

        int actual_len = get_cache_len();
        double time_ms = benchmark_ms(decode, 1, 3);
        double tokens_per_sec = 1000.0 / time_ms;

        std::cout << std::setw(15) << actual_len
                  << std::setw(15) << std::fixed << std::setprecision(2) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(1) << tokens_per_sec << "\n";
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
    std::cerr << "[DEBUG] Starting memory benchmark...\n" << std::flush;
    benchmark_memory(backend.get());
    std::cerr << "[DEBUG] Starting matmul benchmark...\n" << std::flush;
    benchmark_matmul(backend.get());

    // If model path provided, run inference benchmark
    if (argc > 1) {
        std::cerr << "[DEBUG] Starting inference benchmark...\n" << std::flush;
        benchmark_inference(argv[1], backend.get());
    } else {
        std::cout << "\nUsage: " << argv[0] << " [model.gguf]\n";
        std::cout << "Provide a GGUF model path to run inference benchmarks.\n";
    }

    backend->shutdown();
    return 0;
}
