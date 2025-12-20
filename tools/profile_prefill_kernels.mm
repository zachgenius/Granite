// Profile individual prefill kernels to identify performance bottlenecks
// Usage: ./profile_prefill_kernels <model.gguf> [seq_len]

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include "granite/granite.h"
#include "granite/metal_compute.h"
#include "backend/metal/kernels/metal_shaders.h"

using namespace granite;

struct KernelTiming {
    const char* name;
    double total_ms;
    int count;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [seq_len]" << std::endl;
        return 1;
    }

    int seq_len = (argc > 2) ? std::atoi(argv[2]) : 128;

    std::cout << "Prefill Kernel Profiler\n";
    std::cout << "=======================\n";
    std::cout << "Sequence length: " << seq_len << "\n\n";

    // Initialize Metal
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to create Metal device" << std::endl;
        return 1;
    }

    MTL::CommandQueue* queue = device->newCommandQueue();

    // Compile shaders
    NS::Error* error = nullptr;
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    NS::String* source = NS::String::string(METAL_SHADER_SOURCE.c_str(), NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(source, options, &error);
    if (!library) {
        std::cerr << "Failed to compile shaders" << std::endl;
        return 1;
    }

    // Get pipelines
    auto get_pipeline = [&](const char* name) -> MTL::ComputePipelineState* {
        NS::String* fn_name = NS::String::string(name, NS::UTF8StringEncoding);
        MTL::Function* fn = library->newFunction(fn_name);
        if (!fn) return nullptr;
        MTL::ComputePipelineState* pso = device->newComputePipelineState(fn, &error);
        fn->release();
        return pso;
    };

    // Get matmul pipeline with function constants (no bounds checking for aligned dims)
    auto get_matmul_pipeline = [&]() -> MTL::ComputePipelineState* {
        constexpr uint32_t FC_MUL_MM = 100;
        MTL::FunctionConstantValues* fc = MTL::FunctionConstantValues::alloc()->init();
        bool bc_inp = false, bc_out = false;
        fc->setConstantValue(&bc_inp, MTL::DataTypeBool, FC_MUL_MM + 0);
        fc->setConstantValue(&bc_out, MTL::DataTypeBool, FC_MUL_MM + 1);

        NS::String* fn_name = NS::String::string("matmul_q4k_simdgroup", NS::UTF8StringEncoding);
        MTL::Function* fn = library->newFunction(fn_name, fc, &error);
        fc->release();
        if (!fn) return nullptr;
        MTL::ComputePipelineState* pso = device->newComputePipelineState(fn, &error);
        fn->release();
        return pso;
    };

    // Model dimensions (TinyLlama 1.1B)
    const uint32_t hidden_dim = 2048;
    const uint32_t num_heads = 32;
    const uint32_t num_kv_heads = 4;
    const uint32_t head_dim = 64;
    const uint32_t q_dim = num_heads * head_dim;
    const uint32_t kv_dim = num_kv_heads * head_dim;
    const uint32_t intermediate_dim = 5632;
    const uint32_t vocab_size = 32000;
    const uint32_t max_seq = 512;
    const uint32_t M = seq_len;
    const float eps = 1e-5f;
    const float rope_theta = 10000.0f;
    const float attn_scale = 1.0f / std::sqrt(float(head_dim));

    // Q4_K block size
    const uint32_t Q4K_BLOCK_SIZE = 256;
    const uint32_t Q4K_BYTES_PER_BLOCK = 144;  // 12 + 2 + 128 + 2 bytes

    // Allocate buffers
    auto alloc_buf = [&](size_t size) {
        return device->newBuffer(size, MTL::ResourceStorageModeShared);
    };

    // Input/output buffers
    MTL::Buffer* hidden_buf = alloc_buf(M * hidden_dim * sizeof(float));
    MTL::Buffer* attn_input_buf = alloc_buf(M * hidden_dim * sizeof(float));
    MTL::Buffer* q_buf = alloc_buf(M * q_dim * sizeof(float));
    MTL::Buffer* k_buf = alloc_buf(M * kv_dim * sizeof(float));
    MTL::Buffer* v_buf = alloc_buf(M * kv_dim * sizeof(float));
    MTL::Buffer* attn_out_buf = alloc_buf(M * q_dim * sizeof(float));
    MTL::Buffer* post_attn_buf = alloc_buf(M * hidden_dim * sizeof(float));
    MTL::Buffer* ffn_input_buf = alloc_buf(M * hidden_dim * sizeof(float));
    MTL::Buffer* gate_buf = alloc_buf(M * intermediate_dim * sizeof(float));
    MTL::Buffer* up_buf = alloc_buf(M * intermediate_dim * sizeof(float));

    // KV cache (FP16)
    MTL::Buffer* k_cache_buf = alloc_buf(num_kv_heads * max_seq * head_dim * sizeof(uint16_t));
    MTL::Buffer* v_cache_buf = alloc_buf(num_kv_heads * max_seq * head_dim * sizeof(uint16_t));

    // Weight buffers (Q4_K format)
    auto q4k_size = [](uint32_t rows, uint32_t cols) -> size_t {
        uint32_t num_blocks = (rows * cols + 255) / 256;
        return num_blocks * 144;
    };

    MTL::Buffer* wq_buf = alloc_buf(q4k_size(q_dim, hidden_dim));
    MTL::Buffer* wk_buf = alloc_buf(q4k_size(kv_dim, hidden_dim));
    MTL::Buffer* wv_buf = alloc_buf(q4k_size(kv_dim, hidden_dim));
    MTL::Buffer* wo_buf = alloc_buf(q4k_size(hidden_dim, q_dim));
    MTL::Buffer* wgate_buf = alloc_buf(q4k_size(intermediate_dim, hidden_dim));
    MTL::Buffer* wup_buf = alloc_buf(q4k_size(intermediate_dim, hidden_dim));
    MTL::Buffer* wdown_buf = alloc_buf(q4k_size(hidden_dim, intermediate_dim));
    MTL::Buffer* norm_buf = alloc_buf(hidden_dim * sizeof(uint16_t));  // FP16 norm weights

    // Initialize with random data
    auto* hidden_ptr = static_cast<float*>(hidden_buf->contents());
    for (uint32_t i = 0; i < M * hidden_dim; i++) {
        hidden_ptr[i] = (float(rand()) / RAND_MAX - 0.5f) * 0.1f;
    }

    // Simdgroup matmul constants
    const uint32_t NR0 = 64;
    const uint32_t NR1 = 32;
    const size_t SHMEM_SIZE = 8192;

    // Flash attention constants
    const uint32_t Q_TILE = 8;
    const uint32_t K_TILE = 32;
    const uint32_t DK = 64;
    const size_t ATTN_SHMEM = Q_TILE * DK * 2 + Q_TILE * DK * 2 + Q_TILE * K_TILE * 2;

    // Get pipelines
    auto* pipe_rms_batch_f16 = get_pipeline("rms_norm_batch_f16");
    auto* pipe_matmul_q4k = get_matmul_pipeline();
    auto* pipe_rope = get_pipeline("rope_multihead");
    auto* pipe_kv_append = get_pipeline("kv_cache_append_f16");
    auto* pipe_attn = get_pipeline("flash_attention_prefill");
    auto* pipe_add = get_pipeline("elementwise_add");
    auto* pipe_silu_mul = get_pipeline("silu_mul");

    if (!pipe_rms_batch_f16 || !pipe_matmul_q4k || !pipe_rope ||
        !pipe_kv_append || !pipe_attn || !pipe_add || !pipe_silu_mul) {
        std::cerr << "Failed to create one or more pipelines" << std::endl;
        return 1;
    }

    // Timing helper
    auto benchmark_kernel = [&](const char* name, int iterations, auto dispatch_fn) -> double {
        // Warmup
        for (int i = 0; i < 3; i++) {
            MTL::CommandBuffer* cmd = queue->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
            dispatch_fn(enc);
            enc->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            MTL::CommandBuffer* cmd = queue->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
            dispatch_fn(enc);
            enc->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    };

    int num_layers = 22;  // TinyLlama
    int iterations = 20;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "| Kernel                  | Per-call (ms) | Per-layer (ms) | Total 22L (ms) | % of total |\n";
    std::cout << "|-------------------------|---------------|----------------|----------------|------------|\n";

    double total_time = 0;
    std::vector<std::pair<const char*, double>> timings;

    // 1. RMSNorm (attention)
    double rms_time = benchmark_kernel("rms_norm_batch_f16", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pipe_rms_batch_f16);
        enc->setBuffer(hidden_buf, 0, 0);
        enc->setBuffer(norm_buf, 0, 1);
        enc->setBuffer(attn_input_buf, 0, 2);
        enc->setBytes(&M, 4, 3);
        enc->setBytes(&hidden_dim, 4, 4);
        enc->setBytes(&eps, 4, 5);
        enc->dispatchThreadgroups(MTL::Size::Make(M, 1, 1), MTL::Size::Make(256, 1, 1));
    });
    timings.push_back({"RMSNorm (attn+ffn)", rms_time * 2});  // 2 per layer

    // 2. Q projection
    double q_proj_time = benchmark_kernel("Q projection", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pipe_matmul_q4k);
        enc->setBuffer(attn_input_buf, 0, 0);
        enc->setBuffer(wq_buf, 0, 1);
        enc->setBuffer(q_buf, 0, 2);
        enc->setBytes(&M, 4, 3);
        enc->setBytes(&hidden_dim, 4, 4);
        enc->setBytes(&q_dim, 4, 5);
        enc->setThreadgroupMemoryLength(SHMEM_SIZE, 0);
        enc->dispatchThreadgroups(MTL::Size::Make((M + NR1 - 1) / NR1, (q_dim + NR0 - 1) / NR0, 1),
                                  MTL::Size::Make(128, 1, 1));
    });
    timings.push_back({"Q projection", q_proj_time});

    // 3. K projection
    double k_proj_time = benchmark_kernel("K projection", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pipe_matmul_q4k);
        enc->setBuffer(attn_input_buf, 0, 0);
        enc->setBuffer(wk_buf, 0, 1);
        enc->setBuffer(k_buf, 0, 2);
        enc->setBytes(&M, 4, 3);
        enc->setBytes(&hidden_dim, 4, 4);
        enc->setBytes(&kv_dim, 4, 5);
        enc->setThreadgroupMemoryLength(SHMEM_SIZE, 0);
        enc->dispatchThreadgroups(MTL::Size::Make((M + NR1 - 1) / NR1, (kv_dim + NR0 - 1) / NR0, 1),
                                  MTL::Size::Make(128, 1, 1));
    });
    timings.push_back({"K projection", k_proj_time});

    // 4. V projection
    double v_proj_time = benchmark_kernel("V projection", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pipe_matmul_q4k);
        enc->setBuffer(attn_input_buf, 0, 0);
        enc->setBuffer(wv_buf, 0, 1);
        enc->setBuffer(v_buf, 0, 2);
        enc->setBytes(&M, 4, 3);
        enc->setBytes(&hidden_dim, 4, 4);
        enc->setBytes(&kv_dim, 4, 5);
        enc->setThreadgroupMemoryLength(SHMEM_SIZE, 0);
        enc->dispatchThreadgroups(MTL::Size::Make((M + NR1 - 1) / NR1, (kv_dim + NR0 - 1) / NR0, 1),
                                  MTL::Size::Make(128, 1, 1));
    });
    timings.push_back({"V projection", v_proj_time});

    // 5. RoPE
    double rope_time = benchmark_kernel("RoPE", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        uint32_t start_pos = 0;
        enc->setComputePipelineState(pipe_rope);
        enc->setBuffer(q_buf, 0, 0);
        enc->setBuffer(k_buf, 0, 1);
        enc->setBytes(&num_heads, 4, 2);
        enc->setBytes(&num_kv_heads, 4, 3);
        enc->setBytes(&M, 4, 4);
        enc->setBytes(&head_dim, 4, 5);
        enc->setBytes(&start_pos, 4, 6);
        enc->setBytes(&rope_theta, 4, 7);
        enc->dispatchThreads(MTL::Size::Make(head_dim / 2, M, num_heads + num_kv_heads),
                            MTL::Size::Make(32, 1, 1));
    });
    timings.push_back({"RoPE", rope_time});

    // 6. KV cache append
    double kv_append_time = benchmark_kernel("KV append", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        uint32_t curr_len = 0;
        enc->setComputePipelineState(pipe_kv_append);
        enc->setBuffer(k_buf, 0, 0);
        enc->setBuffer(k_cache_buf, 0, 1);
        enc->setBytes(&num_kv_heads, 4, 2);
        enc->setBytes(&head_dim, 4, 3);
        enc->setBytes(&curr_len, 4, 4);
        enc->setBytes(&M, 4, 5);
        enc->setBytes(&max_seq, 4, 6);
        enc->dispatchThreads(MTL::Size::Make(head_dim, M, num_kv_heads), MTL::Size::Make(32, 1, 1));

        enc->setBuffer(v_buf, 0, 0);
        enc->setBuffer(v_cache_buf, 0, 1);
        enc->dispatchThreads(MTL::Size::Make(head_dim, M, num_kv_heads), MTL::Size::Make(32, 1, 1));
    });
    timings.push_back({"KV append", kv_append_time});

    // 7. Flash attention
    double attn_time = benchmark_kernel("Flash attention", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        uint32_t start_pos = 0;
        enc->setComputePipelineState(pipe_attn);
        enc->setBuffer(q_buf, 0, 0);
        enc->setBuffer(k_cache_buf, 0, 1);
        enc->setBuffer(v_cache_buf, 0, 2);
        enc->setBuffer(attn_out_buf, 0, 3);
        enc->setBytes(&num_heads, 4, 4);
        enc->setBytes(&num_kv_heads, 4, 5);
        enc->setBytes(&M, 4, 6);
        enc->setBytes(&M, 4, 7);
        enc->setBytes(&head_dim, 4, 8);
        enc->setBytes(&attn_scale, 4, 9);
        enc->setBytes(&start_pos, 4, 10);
        enc->setBytes(&max_seq, 4, 11);
        enc->setThreadgroupMemoryLength(ATTN_SHMEM, 0);
        uint32_t num_q_blocks = (M + Q_TILE - 1) / Q_TILE;
        enc->dispatchThreadgroups(MTL::Size::Make(num_heads, num_q_blocks, 1), MTL::Size::Make(128, 1, 1));
    });
    timings.push_back({"Flash attention", attn_time});

    // 8. Output projection (Wo)
    double wo_time = benchmark_kernel("Wo projection", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pipe_matmul_q4k);
        enc->setBuffer(attn_out_buf, 0, 0);
        enc->setBuffer(wo_buf, 0, 1);
        enc->setBuffer(attn_input_buf, 0, 2);
        enc->setBytes(&M, 4, 3);
        enc->setBytes(&q_dim, 4, 4);
        enc->setBytes(&hidden_dim, 4, 5);
        enc->setThreadgroupMemoryLength(SHMEM_SIZE, 0);
        enc->dispatchThreadgroups(MTL::Size::Make((M + NR1 - 1) / NR1, (hidden_dim + NR0 - 1) / NR0, 1),
                                  MTL::Size::Make(128, 1, 1));
    });
    timings.push_back({"Wo projection", wo_time});

    // 9. Residual add (attn)
    double add_time = benchmark_kernel("Residual add", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        uint32_t count = M * hidden_dim;
        enc->setComputePipelineState(pipe_add);
        enc->setBuffer(hidden_buf, 0, 0);
        enc->setBuffer(attn_input_buf, 0, 1);
        enc->setBuffer(post_attn_buf, 0, 2);
        enc->setBytes(&count, 4, 3);
        enc->dispatchThreads(MTL::Size::Make(count, 1, 1), MTL::Size::Make(256, 1, 1));
    });
    timings.push_back({"Residual add (x2)", add_time * 2});  // 2 per layer

    // 10. Gate projection
    double gate_time = benchmark_kernel("Gate projection", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pipe_matmul_q4k);
        enc->setBuffer(ffn_input_buf, 0, 0);
        enc->setBuffer(wgate_buf, 0, 1);
        enc->setBuffer(gate_buf, 0, 2);
        enc->setBytes(&M, 4, 3);
        enc->setBytes(&hidden_dim, 4, 4);
        enc->setBytes(&intermediate_dim, 4, 5);
        enc->setThreadgroupMemoryLength(SHMEM_SIZE, 0);
        enc->dispatchThreadgroups(MTL::Size::Make((M + NR1 - 1) / NR1, (intermediate_dim + NR0 - 1) / NR0, 1),
                                  MTL::Size::Make(128, 1, 1));
    });
    timings.push_back({"Gate projection", gate_time});

    // 11. Up projection
    double up_time = benchmark_kernel("Up projection", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pipe_matmul_q4k);
        enc->setBuffer(ffn_input_buf, 0, 0);
        enc->setBuffer(wup_buf, 0, 1);
        enc->setBuffer(up_buf, 0, 2);
        enc->setBytes(&M, 4, 3);
        enc->setBytes(&hidden_dim, 4, 4);
        enc->setBytes(&intermediate_dim, 4, 5);
        enc->setThreadgroupMemoryLength(SHMEM_SIZE, 0);
        enc->dispatchThreadgroups(MTL::Size::Make((M + NR1 - 1) / NR1, (intermediate_dim + NR0 - 1) / NR0, 1),
                                  MTL::Size::Make(128, 1, 1));
    });
    timings.push_back({"Up projection", up_time});

    // 12. SiLU + Mul
    double silu_time = benchmark_kernel("SiLU + Mul", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        uint32_t count = M * intermediate_dim;
        enc->setComputePipelineState(pipe_silu_mul);
        enc->setBuffer(gate_buf, 0, 0);
        enc->setBuffer(up_buf, 0, 1);
        enc->setBuffer(gate_buf, 0, 2);
        enc->setBytes(&count, 4, 3);
        enc->dispatchThreads(MTL::Size::Make(count, 1, 1), MTL::Size::Make(256, 1, 1));
    });
    timings.push_back({"SiLU + Mul", silu_time});

    // 13. Down projection
    double down_time = benchmark_kernel("Down projection", iterations, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pipe_matmul_q4k);
        enc->setBuffer(gate_buf, 0, 0);
        enc->setBuffer(wdown_buf, 0, 1);
        enc->setBuffer(ffn_input_buf, 0, 2);
        enc->setBytes(&M, 4, 3);
        enc->setBytes(&intermediate_dim, 4, 4);
        enc->setBytes(&hidden_dim, 4, 5);
        enc->setThreadgroupMemoryLength(SHMEM_SIZE, 0);
        enc->dispatchThreadgroups(MTL::Size::Make((M + NR1 - 1) / NR1, (hidden_dim + NR0 - 1) / NR0, 1),
                                  MTL::Size::Make(128, 1, 1));
    });
    timings.push_back({"Down projection", down_time});

    // Calculate totals
    for (const auto& [name, time] : timings) {
        total_time += time * num_layers;
    }

    // Print results
    for (const auto& [name, time] : timings) {
        double layer_total = time * num_layers;
        double pct = (layer_total / total_time) * 100;
        std::cout << "| " << std::left << std::setw(23) << name << " | "
                  << std::right << std::setw(13) << time << " | "
                  << std::setw(14) << time << " | "
                  << std::setw(14) << layer_total << " | "
                  << std::setw(9) << pct << "% |\n";
    }

    std::cout << "|-------------------------|---------------|----------------|----------------|------------|\n";
    std::cout << "| " << std::left << std::setw(23) << "TOTAL" << " | "
              << std::right << std::setw(13) << "-" << " | "
              << std::setw(14) << "-" << " | "
              << std::setw(14) << total_time << " | "
              << std::setw(9) << "100" << "% |\n";

    // Summary by category
    std::cout << "\n=== Summary by Category ===\n";
    double matmul_total = (q_proj_time + k_proj_time + v_proj_time + wo_time +
                          gate_time + up_time + down_time) * num_layers;
    double attn_total = attn_time * num_layers;
    double other_total = total_time - matmul_total - attn_total;

    std::cout << "MatMul (Q4K):     " << std::setw(8) << matmul_total << " ms ("
              << std::setw(5) << (matmul_total/total_time*100) << "%)\n";
    std::cout << "Flash Attention:  " << std::setw(8) << attn_total << " ms ("
              << std::setw(5) << (attn_total/total_time*100) << "%)\n";
    std::cout << "Other (RMS/RoPE): " << std::setw(8) << other_total << " ms ("
              << std::setw(5) << (other_total/total_time*100) << "%)\n";

    // Cleanup
    hidden_buf->release();
    attn_input_buf->release();
    q_buf->release();
    k_buf->release();
    v_buf->release();
    attn_out_buf->release();
    post_attn_buf->release();
    ffn_input_buf->release();
    gate_buf->release();
    up_buf->release();
    k_cache_buf->release();
    v_cache_buf->release();
    wq_buf->release();
    wk_buf->release();
    wv_buf->release();
    wo_buf->release();
    wgate_buf->release();
    wup_buf->release();
    wdown_buf->release();
    norm_buf->release();
    library->release();
    queue->release();
    device->release();

    return 0;
}
