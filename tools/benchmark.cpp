#include <granite/granite.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <optional>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/resource.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#ifdef GRANITE_HAS_METAL
#include <granite/metal_compute.h>
#endif

using namespace granite;
using Clock = std::chrono::high_resolution_clock;

struct DeviceProfile {
    const char* name;
    std::vector<int64_t> seq_lens;
    size_t kv_cache_max_seq = 0;
    uint32_t prefill_chunk_size = 0;
};

const std::vector<DeviceProfile>& device_profiles() {
    static const std::vector<DeviceProfile> profiles = {
        {"ios-a14", {128, 256}, 2048, 128},
        {"ios-a16", {128, 256, 512}, 4096, 128},
        {"ios-a17", {128, 256, 512}, 4096, 128},
        {"android-mali-g78", {128, 256}, 2048, 128},
        {"android-adreno-740", {128, 256, 512}, 4096, 128},
    };
    return profiles;
}

const DeviceProfile* find_device_profile(const std::string& name) {
    for (const auto& profile : device_profiles()) {
        if (name == profile.name) {
            return &profile;
        }
    }
    return nullptr;
}

void print_device_profiles() {
    std::cout << "  --device-profile <name>: Use a predefined mobile profile\n";
    std::cout << "    Profiles:";
    for (const auto& profile : device_profiles()) {
        std::cout << " " << profile.name;
    }
    std::cout << "\n";
}

std::string format_bytes(size_t bytes) {
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes);
    size_t suffix_index = 0;
    const size_t suffix_count = sizeof(suffixes) / sizeof(suffixes[0]);
    while (value >= 1024.0 && suffix_index + 1 < suffix_count) {
        value /= 1024.0;
        suffix_index++;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value << " " << suffixes[suffix_index];
    return out.str();
}

void print_model_memory_stats(const TransformerModel& model, const char* label) {
    auto stats = model.memory_stats();
    std::cout << label << "\n";
    std::cout << "  weights: " << format_bytes(stats.weights_bytes) << "\n";
    if (stats.raw_weights_bytes > 0) {
        std::cout << "  raw weights: " << format_bytes(stats.raw_weights_bytes) << "\n";
    }
    if (stats.decode_pool_bytes > 0) {
        std::cout << "  decode pool: " << format_bytes(stats.decode_pool_bytes) << "\n";
    }
    if (stats.prefill_pool_bytes > 0) {
        std::cout << "  prefill pool: " << format_bytes(stats.prefill_pool_bytes) << "\n";
    }
#ifdef GRANITE_HAS_METAL
    if (stats.gpu_kv_bytes > 0) {
        std::cout << "  gpu kv: " << format_bytes(stats.gpu_kv_bytes) << "\n";
    }
#endif
}

void print_kernel_timings(MetalCompute* gpu) {
    if (!gpu) {
        return;
    }
    auto timings = gpu->get_kernel_timing_stats();
    if (timings.empty()) {
        std::cout << "\nKernel timing: no data\n";
        return;
    }
    std::sort(timings.begin(), timings.end(),
              [](const MetalCompute::KernelTiming& a, const MetalCompute::KernelTiming& b) {
                  return a.gpu_time_ms > b.gpu_time_ms;
              });
    double total_ms = 0.0;
    for (const auto& entry : timings) {
        total_ms += entry.gpu_time_ms;
    }
    std::cout << "\nKernel timing (top 15 by GPU time):\n";
    size_t limit = std::min<size_t>(15, timings.size());
    for (size_t i = 0; i < limit; i++) {
        const auto& entry = timings[i];
        std::cout << "  " << entry.name << ": " << std::fixed << std::setprecision(2)
                  << entry.gpu_time_ms << " ms (" << entry.dispatches << " dispatches)\n";
    }
    std::cout << "  total gpu time: " << std::fixed << std::setprecision(2) << total_ms << " ms\n";
}

bool get_rss_bytes(size_t& out_bytes) {
#if defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        out_bytes = static_cast<size_t>(info.resident_size);
        return true;
    }
    return false;
#elif defined(__linux__)
    FILE* fp = std::fopen("/proc/self/statm", "r");
    if (!fp) {
        return false;
    }
    long pages = 0;
    if (std::fscanf(fp, "%*s %ld", &pages) != 1) {
        std::fclose(fp);
        return false;
    }
    std::fclose(fp);
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return false;
    }
    out_bytes = static_cast<size_t>(pages) * static_cast<size_t>(page_size);
    return true;
#else
    (void)out_bytes;
    return false;
#endif
}

bool get_max_rss_bytes(size_t& out_bytes) {
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return false;
    }
#if defined(__APPLE__)
    out_bytes = static_cast<size_t>(usage.ru_maxrss);
#else
    out_bytes = static_cast<size_t>(usage.ru_maxrss) * 1024ull;
#endif
    return true;
}

void print_rss(const char* label) {
    size_t rss = 0;
    size_t max_rss = 0;
    bool has_rss = get_rss_bytes(rss);
    bool has_max = get_max_rss_bytes(max_rss);
    std::cout << label;
    if (has_rss) {
        std::cout << " RSS: " << format_bytes(rss);
    } else {
        std::cout << " RSS: n/a";
    }
    if (has_max) {
        std::cout << ", max RSS: " << format_bytes(max_rss);
    }
    std::cout << "\n";
}

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
    bool enable_profiling = false,
    bool kernel_timing = false) {
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
    if (kernel_timing) {
        std::cout << "Kernel timing: enabled\n";
    }

    print_rss("Before load:");

    // Load model
    std::cout << "Loading model: " << model_path << "\n";
    auto model_result = TransformerModel::load(model_path, backend, runtime_config);
    if (!model_result.ok()) {
        std::cerr << "Failed to load model: " << model_result.error().message() << "\n";
        return;
    }
    auto model = std::move(model_result).take();
    print_rss("After load:");
    print_model_memory_stats(model, "Model memory (after load):");

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
    std::cout << "KV cache (CPU): " << format_bytes(kv_cache.memory_bytes()) << "\n";
    print_rss("After CPU KV:");

    // Enable GPU mode and allocate GPU KV cache
#ifdef GRANITE_HAS_METAL
    model.set_use_gpu(true);
    // Control logits computation: full_logits=false means last-token-only (faster)
    model.set_prefill_last_token_only(!full_logits);
    auto gpu_cache_result = model.allocate_gpu_kv_cache(static_cast<int>(cache_max_seq));
    if (gpu_cache_result.ok()) {
        std::cout << "GPU KV cache allocated\n";
        const auto& cfg = model.config();
        size_t elems = static_cast<size_t>(cfg.num_layers) *
                       static_cast<size_t>(cfg.num_kv_heads) *
                       static_cast<size_t>(cache_max_seq) *
                       static_cast<size_t>(cfg.head_dim);
        size_t gpu_kv_bytes = elems * sizeof(uint16_t) * 2;
        std::cout << "KV cache (GPU): " << format_bytes(gpu_kv_bytes) << "\n";
        print_rss("After GPU KV:");
    } else {
        std::cout << "GPU KV cache allocation failed, using CPU cache\n";
    }
#endif

#ifdef GRANITE_HAS_METAL
    if (enable_profiling && backend->get_type() == BackendType::Metal) {
        if (auto* gpu = get_metal_compute()) {
            gpu->reset_profiling_stats();
        }
    }
#endif

#ifdef GRANITE_HAS_METAL
    if (kernel_timing && backend->get_type() == BackendType::Metal) {
        if (auto* gpu = get_metal_compute()) {
            gpu->enable_kernel_timing(true);
            gpu->reset_kernel_timing();
        }
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

#ifdef GRANITE_HAS_METAL
        if (enable_profiling && backend->get_type() == BackendType::Metal) {
            if (auto* gpu = get_metal_compute()) {
                gpu->reset_profiling_stats();
            }
        }
#endif
        std::vector<double> samples;
        double time_ms = benchmark_ms(prefill, 1, 3, enable_profiling ? &samples : nullptr);
        double tokens_per_sec = (seq_len / time_ms) * 1000.0;

        std::cout << std::setw(15) << seq_len
                  << std::setw(15) << std::fixed << std::setprecision(2) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(1) << tokens_per_sec
                  << "\n";
#ifdef GRANITE_HAS_METAL
        if (enable_profiling && backend->get_type() == BackendType::Metal) {
            if (auto* gpu = get_metal_compute()) {
                uint64_t dispatches = 0;
                uint64_t syncs = 0;
                double sync_time_ms = 0.0;
                uint64_t cmd_buffers = 0;
                double gpu_time_ms = 0.0;
                uint64_t gpu_timed_buffers = 0;
                gpu->get_profiling_stats(dispatches, syncs, sync_time_ms, cmd_buffers,
                                         gpu_time_ms, gpu_timed_buffers);
                if (gpu_timed_buffers > 0) {
                    std::cout << "  gpu time: " << std::fixed << std::setprecision(2)
                              << gpu_time_ms << " ms (" << gpu_timed_buffers << " buffers)\n";
                }
            }
        }
#endif
        if (enable_profiling && !samples.empty()) {
            double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
            double avg = sum / static_cast<double>(samples.size());
            double min = *std::min_element(samples.begin(), samples.end());
            double max = *std::max_element(samples.begin(), samples.end());
            std::cout << "  prefill stats (ms): avg=" << std::fixed << std::setprecision(2) << avg
                      << " min=" << min << " max=" << max << "\n";
        }
    }

    print_model_memory_stats(model, "Model memory (after prefill):");

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
#ifdef GRANITE_HAS_METAL
        if (enable_profiling && backend->get_type() == BackendType::Metal) {
            if (auto* gpu = get_metal_compute()) {
                gpu->reset_profiling_stats();
            }
        }
#endif
        std::vector<double> samples;
        double time_ms = benchmark_ms(decode, 1, 3, enable_profiling ? &samples : nullptr);
        double tokens_per_sec = 1000.0 / time_ms;

        std::cout << std::setw(15) << actual_len
                  << std::setw(15) << std::fixed << std::setprecision(2) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(1) << tokens_per_sec << "\n";
#ifdef GRANITE_HAS_METAL
        if (enable_profiling && backend->get_type() == BackendType::Metal) {
            if (auto* gpu = get_metal_compute()) {
                uint64_t dispatches = 0;
                uint64_t syncs = 0;
                double sync_time_ms = 0.0;
                uint64_t cmd_buffers = 0;
                double gpu_time_ms = 0.0;
                uint64_t gpu_timed_buffers = 0;
                gpu->get_profiling_stats(dispatches, syncs, sync_time_ms, cmd_buffers,
                                         gpu_time_ms, gpu_timed_buffers);
                if (gpu_timed_buffers > 0) {
                    std::cout << "  gpu time: " << std::fixed << std::setprecision(2)
                              << gpu_time_ms << " ms (" << gpu_timed_buffers << " buffers)\n";
                }
            }
        }
#endif
        if (enable_profiling && !samples.empty()) {
            double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
            double avg = sum / static_cast<double>(samples.size());
            double min = *std::min_element(samples.begin(), samples.end());
            double max = *std::max_element(samples.begin(), samples.end());
            std::cout << "  decode stats (ms): avg=" << std::fixed << std::setprecision(2) << avg
                      << " min=" << min << " max=" << max << "\n";
        }
    }

    print_model_memory_stats(model, "Model memory (after decode):");

#ifdef GRANITE_HAS_METAL
    if (enable_profiling && backend->get_type() == BackendType::Metal) {
        if (auto* gpu = get_metal_compute()) {
            uint64_t dispatches = 0;
            uint64_t syncs = 0;
            double sync_time_ms = 0.0;
            uint64_t cmd_buffers = 0;
            double gpu_time_ms = 0.0;
            uint64_t gpu_timed_buffers = 0;
            gpu->get_profiling_stats(dispatches, syncs, sync_time_ms, cmd_buffers,
                                     gpu_time_ms, gpu_timed_buffers);
            std::cout << "\nGPU profiling:\n";
            std::cout << "  dispatches: " << dispatches << "\n";
            std::cout << "  command buffers: " << cmd_buffers << "\n";
            std::cout << "  syncs: " << syncs << " (" << std::fixed << std::setprecision(2)
                      << sync_time_ms << " ms)\n";
            if (gpu_timed_buffers > 0) {
                std::cout << "  gpu time: " << std::fixed << std::setprecision(2)
                          << gpu_time_ms << " ms (" << gpu_timed_buffers << " buffers)\n";
            } else {
                std::cout << "  gpu time: unavailable (no timing data)\n";
            }
        }
    }
#endif

#ifdef GRANITE_HAS_METAL
    if (kernel_timing && backend->get_type() == BackendType::Metal) {
        print_kernel_timings(get_metal_compute());
    }
#endif
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
    BackendType preferred_backend = BackendType::CPU;
    bool preferred_backend_set = false;
    if (const char* backend_env = std::getenv("GRANITE_BACKEND")) {
        BackendType parsed = BackendType::CPU;
        if (parse_backend_arg(backend_env, parsed)) {
            preferred_backend = parsed;
            preferred_backend_set = true;
        } else {
            std::cerr << "Invalid GRANITE_BACKEND value, using default.\n";
        }
    }
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            BackendType parsed = BackendType::CPU;
            if (parse_backend_arg(argv[++i], parsed)) {
                preferred_backend = parsed;
                preferred_backend_set = true;
            } else {
                std::cerr << "Invalid --backend value, using default.\n";
            }
        }
    }

    std::unique_ptr<IComputeBackend> backend;
    if (preferred_backend_set) {
        backend = create_backend(preferred_backend);
        if (!backend) {
            std::cerr << "Preferred backend unavailable, falling back to default.\n";
            backend = create_default_backend();
        }
    } else {
        backend = create_default_backend();
    }
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
        bool seq_lens_set = false;
        uint32_t prefill_chunk_size = 0;
        bool prefill_chunk_set = false;
        size_t kv_cache_max_seq = 0;
        bool kv_cache_max_seq_set = false;
        bool enable_profiling = false;
        bool kernel_timing = false;
        std::string device_profile_name;
        bool device_profile_set = false;
        BackendType run_backend = preferred_backend;
        bool run_backend_set = preferred_backend_set;

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--full-logits") {
                full_logits = true;
            } else if (arg == "--seq-lens" && i + 1 < argc) {
                auto parsed = parse_seq_lens(argv[++i]);
                if (!parsed.empty()) {
                    seq_lens = std::move(parsed);
                    seq_lens_set = true;
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
            } else if (arg == "--kernel-timing") {
                enable_profiling = true;
                kernel_timing = true;
            } else if (arg == "--device-profile" && i + 1 < argc) {
                device_profile_name = argv[++i];
                device_profile_set = true;
            } else if (arg == "--backend" && i + 1 < argc) {
                BackendType parsed = BackendType::CPU;
                if (parse_backend_arg(argv[++i], parsed)) {
                    run_backend = parsed;
                    run_backend_set = true;
                } else {
                    std::cerr << "Invalid --backend value, using default.\n";
                }
            } else if (arg[0] != '-') {
                model_path = arg;
            }
        }

        if (run_backend_set && run_backend != backend->get_type()) {
            std::cerr << "Requested backend differs from active backend; using active backend.\n";
        }

        if (device_profile_set) {
            const auto* profile = find_device_profile(device_profile_name);
            if (!profile) {
                std::cerr << "Unknown device profile '" << device_profile_name << "'.\n";
            } else {
                if (!seq_lens_set) {
                    seq_lens = profile->seq_lens;
                }
                if (!kv_cache_max_seq_set && profile->kv_cache_max_seq > 0) {
                    kv_cache_max_seq = profile->kv_cache_max_seq;
                    kv_cache_max_seq_set = true;
                }
                if (!prefill_chunk_set && profile->prefill_chunk_size > 0) {
                    prefill_chunk_size = profile->prefill_chunk_size;
                    prefill_chunk_set = true;
                }
                std::cout << "Device profile: " << profile->name << "\n";
            }
        }

        if (!model_path.empty()) {
            benchmark_inference(model_path, backend.get(), seq_lens, full_logits,
                                prefill_chunk_size, prefill_chunk_set,
                                kv_cache_max_seq, kv_cache_max_seq_set,
                                enable_profiling, kernel_timing);
        } else {
            std::cout << "\nUsage: " << argv[0] << " [model.gguf] [--full-logits]\n";
            std::cout << "Provide a GGUF model path to run inference benchmarks.\n";
            std::cout << "  --full-logits: Compute logits for all tokens (default: last token only)\n";
            std::cout << "  --seq-lens <a,b,c>: Override prefill prompt lengths\n";
            std::cout << "  --prefill-chunk-size <n>: Override prefill chunk size (0 disables)\n";
            std::cout << "  --kv-cache-max-seq <n>: Override KV cache max sequence length\n";
            std::cout << "  --profile: Print per-section timing stats\n";
            std::cout << "  --kernel-timing: Collect per-kernel GPU timing stats (Metal only)\n";
            print_device_profiles();
            std::cout << "  --backend <cpu|metal|vulkan>: Force backend for benchmarking\n";
            std::cout << "  GRANITE_BACKEND=cpu|metal|vulkan (env override)\n";
        }
    } else {
        std::cout << "\nUsage: " << argv[0] << " [model.gguf] [--full-logits]\n";
        std::cout << "Provide a GGUF model path to run inference benchmarks.\n";
        std::cout << "  --full-logits: Compute logits for all tokens (default: last token only)\n";
        std::cout << "  --seq-lens <a,b,c>: Override prefill prompt lengths\n";
        std::cout << "  --prefill-chunk-size <n>: Override prefill chunk size (0 disables)\n";
        std::cout << "  --kv-cache-max-seq <n>: Override KV cache max sequence length\n";
        std::cout << "  --profile: Print per-section timing stats\n";
        std::cout << "  --kernel-timing: Collect per-kernel GPU timing stats (Metal only)\n";
        print_device_profiles();
        std::cout << "  --backend <cpu|metal|vulkan>: Force backend for benchmarking\n";
        std::cout << "  GRANITE_BACKEND=cpu|metal|vulkan (env override)\n";
    }

    backend->shutdown();
    return 0;
}
