// MetalCompute - High-level GPU compute interface for LLM inference
// Manages shader compilation, pipeline states, and command buffer batching

#include "granite/metal_compute.h"
#include "granite/log.h"

#ifdef GRANITE_HAS_METAL

// Note: NS_PRIVATE_IMPLEMENTATION and MTL_PRIVATE_IMPLEMENTATION are defined
// in metal_backend.mm to avoid duplicate symbols
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>

namespace granite {

// Embedded shader source
static const char* QUANTIZED_MATMUL_SHADER = R"(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint QK_K = 256;

struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

inline void get_scale_min_k4(int j, const device uint8_t* q, thread uint8_t& sc, thread uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// SIMD-optimized matvec for Q4_K
// Each SIMD group (32 threads) processes one output row together
kernel void matvec_q4k(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one output row
    uint row = tgid;
    if (row >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    // Each thread in the SIMD group handles different blocks
    float local_sum = 0.0f;

    // 32 threads, each handles different K blocks
    for (uint kb = simd_lane; kb < num_blocks_k; kb += 32) {
        const device block_q4_K* block = &weights[row * num_blocks_k + kb];
        float d = float(block->d);
        float dmin = float(block->dmin);
        const device uint8_t* scales = block->scales;
        const device uint8_t* qs = block->qs;

        float block_sum = 0.0f;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is, scales, sc1, m1);
            get_scale_min_k4(is + 1, scales, sc2, m2);

            float d1 = d * float(sc1);
            float dm1 = dmin * float(m1);
            float d2 = d * float(sc2);
            float dm2 = dmin * float(m2);

            uint base_idx = kb * QK_K + j;

            // Unrolled inner loops for better performance
            for (int l = 0; l < 32; l += 4) {
                block_sum += x[base_idx + l] * (d1 * float(qs[l] & 0xF) - dm1);
                block_sum += x[base_idx + l + 1] * (d1 * float(qs[l + 1] & 0xF) - dm1);
                block_sum += x[base_idx + l + 2] * (d1 * float(qs[l + 2] & 0xF) - dm1);
                block_sum += x[base_idx + l + 3] * (d1 * float(qs[l + 3] & 0xF) - dm1);
            }

            for (int l = 0; l < 32; l += 4) {
                block_sum += x[base_idx + 32 + l] * (d2 * float(qs[l] >> 4) - dm2);
                block_sum += x[base_idx + 32 + l + 1] * (d2 * float(qs[l + 1] >> 4) - dm2);
                block_sum += x[base_idx + 32 + l + 2] * (d2 * float(qs[l + 2] >> 4) - dm2);
                block_sum += x[base_idx + 32 + l + 3] * (d2 * float(qs[l + 3] >> 4) - dm2);
            }

            qs += 32;
            is += 2;
        }
        local_sum += block_sum;
    }

    // SIMD reduction - sum across all 32 lanes
    float sum = simd_sum(local_sum);

    // Lane 0 writes the result
    if (simd_lane == 0) {
        y[row] = sum;
    }
}

kernel void matmul_q4k(
    device const float* X          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* Y                [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    float sum = 0.0f;

    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q4_K* block = &weights[col * num_blocks_k + kb];
        float d = float(block->d);
        float dmin = float(block->dmin);
        const device uint8_t* scales = block->scales;
        const device uint8_t* qs = block->qs;

        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is, scales, sc1, m1);
            get_scale_min_k4(is + 1, scales, sc2, m2);

            float d1 = d * float(sc1);
            float dm1 = dmin * float(m1);
            float d2 = d * float(sc2);
            float dm2 = dmin * float(m2);

            uint base_idx = kb * QK_K + j;

            for (int l = 0; l < 32; l++) {
                float w = d1 * float(qs[l] & 0xF) - dm1;
                sum += X[row * K + base_idx + l] * w;
            }

            for (int l = 0; l < 32; l++) {
                float w = d2 * float(qs[l] >> 4) - dm2;
                sum += X[row * K + base_idx + 32 + l] * w;
            }

            qs += 32;
            is += 2;
        }
    }

    Y[row * N + col] = sum;
}

// SIMD-optimized matvec for FP16 weights
// Each SIMD group (32 threads) processes one output row together
kernel void matvec_f16(
    device const float* x          [[buffer(0)]],
    device const half* W           [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]]
) {
    uint row = tgid;
    if (row >= N) return;

    device const half* w_row = W + row * K;

    // Each thread handles K/32 elements
    float local_sum = 0.0f;
    for (uint i = simd_lane; i < K; i += 32) {
        local_sum += x[i] * float(w_row[i]);
    }

    // SIMD reduction
    float sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        y[row] = sum;
    }
}

kernel void matvec_f32(
    device const float* x          [[buffer(0)]],
    device const float* W          [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        sum += x[i] * W[gid * K + i];
    }
    y[gid] = sum;
}

// Optimized RMS norm with parallel reduction
kernel void rms_norm(
    device const float* x          [[buffer(0)]],
    device const float* weight     [[buffer(1)]],
    device float* out              [[buffer(2)]],
    constant uint& size            [[buffer(3)]],
    constant float& eps            [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    // Shared memory for reduction
    threadgroup float shared_sum[256];

    // Each thread sums multiple elements
    float local_sum = 0.0f;
    for (uint i = tid; i < size; i += tg_size) {
        float val = x[i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 has final sum
    threadgroup float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrt(shared_sum[0] / float(size) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread normalizes and scales its elements
    for (uint i = gid; i < size; i += tg_size) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// RMS norm with FP16 weights
kernel void rms_norm_f16(
    device const float* x          [[buffer(0)]],
    device const half* weight      [[buffer(1)]],
    device float* out              [[buffer(2)]],
    constant uint& size            [[buffer(3)]],
    constant float& eps            [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];

    float local_sum = 0.0f;
    for (uint i = tid; i < size; i += tg_size) {
        float val = x[i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrt(shared_sum[0] / float(size) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = gid; i < size; i += tg_size) {
        out[i] = x[i] * inv_rms * float(weight[i]);
    }
}

kernel void silu(
    device float* x                [[buffer(0)]],
    constant uint& size            [[buffer(1)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float val = x[gid];
    x[gid] = val / (1.0f + exp(-val));
}

kernel void elementwise_mul(
    device const float* a          [[buffer(0)]],
    device const float* b          [[buffer(1)]],
    device float* c                [[buffer(2)]],
    constant uint& size            [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    c[gid] = a[gid] * b[gid];
}

kernel void rope(
    device float* x                [[buffer(0)]],
    constant uint& seq_len         [[buffer(1)]],
    constant uint& head_dim        [[buffer(2)]],
    constant uint& start_pos       [[buffer(3)]],
    constant float& freq_base      [[buffer(4)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint pos = gid.y;
    uint dim = gid.x;

    if (pos >= seq_len || dim >= head_dim / 2) return;

    float freq = 1.0f / pow(freq_base, float(dim * 2) / float(head_dim));
    float theta = float(start_pos + pos) * freq;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    uint idx = pos * head_dim + dim * 2;
    float x0 = x[idx];
    float x1 = x[idx + 1];

    x[idx] = x0 * cos_theta - x1 * sin_theta;
    x[idx + 1] = x0 * sin_theta + x1 * cos_theta;
}

// Embedding lookup: gather rows from FP16 embedding table
// token_ids: [num_tokens] int32
// embeddings: [vocab_size, hidden_dim] half
// output: [num_tokens, hidden_dim] float
kernel void embedding_lookup(
    device const int* token_ids    [[buffer(0)]],
    device const half* embeddings  [[buffer(1)]],
    device float* output           [[buffer(2)]],
    constant uint& hidden_dim      [[buffer(3)]],
    constant uint& vocab_size      [[buffer(4)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;  // Which token
    uint dim = gid.x;        // Which dimension

    if (dim >= hidden_dim) return;

    int token_id = token_ids[token_idx];
    // Clamp to valid range
    if (token_id < 0) token_id = 0;
    if (token_id >= int(vocab_size)) token_id = 0;

    // Gather from embedding table (row-major: [vocab_size, hidden_dim])
    output[token_idx * hidden_dim + dim] = float(embeddings[token_id * hidden_dim + dim]);
}

kernel void elementwise_add(
    device const float* a          [[buffer(0)]],
    device const float* b          [[buffer(1)]],
    device float* c                [[buffer(2)]],
    constant uint& size            [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    c[gid] = a[gid] + b[gid];
}

// RoPE for multi-head attention
// Q shape: [seq_len, num_heads_q * head_dim], K shape: [seq_len, num_heads_k * head_dim]
kernel void rope_multihead(
    device float* q                [[buffer(0)]],
    device float* k                [[buffer(1)]],
    constant uint& num_heads_q     [[buffer(2)]],
    constant uint& num_heads_k     [[buffer(3)]],
    constant uint& seq_len         [[buffer(4)]],
    constant uint& head_dim        [[buffer(5)]],
    constant uint& start_pos       [[buffer(6)]],
    constant float& freq_base      [[buffer(7)]],
    uint3 gid                      [[thread_position_in_grid]]
) {
    uint pos = gid.z;       // Position in sequence
    uint head = gid.y;      // Head index
    uint dim = gid.x;       // Dimension pair index (0 to head_dim/2-1)

    if (pos >= seq_len || dim >= head_dim / 2) return;

    float freq = 1.0f / pow(freq_base, float(dim * 2) / float(head_dim));
    float theta = float(start_pos + pos) * freq;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    // Apply RoPE to Q (all heads)
    if (head < num_heads_q) {
        uint q_idx = pos * num_heads_q * head_dim + head * head_dim + dim * 2;
        float q0 = q[q_idx];
        float q1 = q[q_idx + 1];
        q[q_idx] = q0 * cos_theta - q1 * sin_theta;
        q[q_idx + 1] = q0 * sin_theta + q1 * cos_theta;
    }

    // Apply RoPE to K (KV heads)
    if (head < num_heads_k) {
        uint k_idx = pos * num_heads_k * head_dim + head * head_dim + dim * 2;
        float k0 = k[k_idx];
        float k1 = k[k_idx + 1];
        k[k_idx] = k0 * cos_theta - k1 * sin_theta;
        k[k_idx + 1] = k0 * sin_theta + k1 * cos_theta;
    }
}

// Softmax over rows (in-place)
// x shape: [M, N], softmax over N dimension
kernel void softmax_row(
    device float* x                [[buffer(0)]],
    constant uint& M               [[buffer(1)]],
    constant uint& N               [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= M) return;

    device float* row = x + gid * N;

    // Find max
    float max_val = row[0];
    for (uint i = 1; i < N; i++) {
        max_val = max(max_val, row[i]);
    }

    // Exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < N; i++) {
        row[i] = exp(row[i] - max_val);
        sum += row[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < N; i++) {
        row[i] *= inv_sum;
    }
}

// Single-head attention for decode (seq_q = 1)
// Q: [1, head_dim], K: [seq_kv, head_dim], V: [seq_kv, head_dim]
// output: [1, head_dim]
kernel void attention_decode(
    device const float* Q          [[buffer(0)]],
    device const float* K          [[buffer(1)]],
    device const float* V          [[buffer(2)]],
    device float* output           [[buffer(3)]],
    constant uint& seq_kv          [[buffer(4)]],
    constant uint& head_dim        [[buffer(5)]],
    constant float& scale          [[buffer(6)]],
    uint gid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    // Each thread computes one output dimension
    if (gid >= head_dim) return;

    // Compute attention scores: Q @ K^T
    threadgroup float scores[1024];  // Max seq_kv = 1024
    threadgroup float max_score;
    threadgroup float sum_exp;

    // Thread 0 computes all scores and softmax
    if (lid == 0) {
        float local_max = -INFINITY;
        for (uint k = 0; k < seq_kv; k++) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                dot += Q[d] * K[k * head_dim + d];
            }
            scores[k] = dot * scale;
            local_max = max(local_max, scores[k]);
        }
        max_score = local_max;

        // Softmax
        float local_sum = 0.0f;
        for (uint k = 0; k < seq_kv; k++) {
            scores[k] = exp(scores[k] - max_score);
            local_sum += scores[k];
        }
        sum_exp = local_sum;

        for (uint k = 0; k < seq_kv; k++) {
            scores[k] /= sum_exp;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes one output dimension: sum(scores[k] * V[k, dim])
    float out_val = 0.0f;
    for (uint k = 0; k < seq_kv; k++) {
        out_val += scores[k] * V[k * head_dim + gid];
    }
    output[gid] = out_val;
}

// KV cache append kernel
// Copies new K/V to the cache at the current position
// cache: [num_heads, max_seq, head_dim], new_kv: [num_heads, new_len, head_dim]
kernel void kv_cache_append(
    device const float* new_kv     [[buffer(0)]],
    device float* cache            [[buffer(1)]],
    constant uint& num_heads       [[buffer(2)]],
    constant uint& head_dim        [[buffer(3)]],
    constant uint& current_len     [[buffer(4)]],
    constant uint& new_len         [[buffer(5)]],
    constant uint& max_seq         [[buffer(6)]],
    uint3 gid                      [[thread_position_in_grid]]
) {
    uint h = gid.z;   // Head index
    uint s = gid.y;   // Position in new sequence
    uint d = gid.x;   // Dimension

    if (h >= num_heads || s >= new_len || d >= head_dim) return;

    // Source: [num_heads, new_len, head_dim]
    uint src_idx = h * new_len * head_dim + s * head_dim + d;
    // Dest: [num_heads, max_seq, head_dim]
    uint dst_idx = h * max_seq * head_dim + (current_len + s) * head_dim + d;

    cache[dst_idx] = new_kv[src_idx];
}

// Multi-head attention kernel for decode (seq_q = 1)
// Q: [num_heads, 1, head_dim]
// K: [num_kv_heads, seq_kv, head_dim]
// V: [num_kv_heads, seq_kv, head_dim]
// output: [num_heads, 1, head_dim]
// Each threadgroup handles one head
kernel void multihead_attention_decode(
    device const float* Q          [[buffer(0)]],
    device const float* K          [[buffer(1)]],
    device const float* V          [[buffer(2)]],
    device float* output           [[buffer(3)]],
    constant uint& num_heads       [[buffer(4)]],
    constant uint& num_kv_heads    [[buffer(5)]],
    constant uint& seq_kv          [[buffer(6)]],
    constant uint& head_dim        [[buffer(7)]],
    constant float& scale          [[buffer(8)]],
    uint head_idx                  [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    if (head_idx >= num_heads) return;

    // GQA: map Q head to KV head
    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;

    // Pointers to this head's data
    device const float* q_head = Q + head_idx * head_dim;
    device const float* k_head = K + kv_head * seq_kv * head_dim;
    device const float* v_head = V + kv_head * seq_kv * head_dim;
    device float* out_head = output + head_idx * head_dim;

    // Shared memory for attention scores
    threadgroup float scores[2048];  // Max seq_kv = 2048
    threadgroup float shared_max;
    threadgroup float shared_sum;

    // Step 1: Compute attention scores (Q @ K^T) in parallel
    // Each thread handles multiple K positions
    float local_max = -INFINITY;
    for (uint k = tid; k < seq_kv; k += tg_size) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_head[k * head_dim + d];
        }
        scores[k] = dot * scale;
        local_max = max(local_max, scores[k]);
    }

    // Reduce to find global max
    threadgroup float thread_maxes[256];
    thread_maxes[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simple reduction for max
    if (tid == 0) {
        float global_max = thread_maxes[0];
        for (uint i = 1; i < min(tg_size, 256u); i++) {
            global_max = max(global_max, thread_maxes[i]);
        }
        shared_max = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (uint k = tid; k < seq_kv; k += tg_size) {
        scores[k] = exp(scores[k] - shared_max);
        local_sum += scores[k];
    }

    // Reduce to find global sum
    threadgroup float thread_sums[256];
    thread_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float global_sum = 0.0f;
        for (uint i = 0; i < min(tg_size, 256u); i++) {
            global_sum += thread_sums[i];
        }
        shared_sum = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Normalize scores
    float inv_sum = 1.0f / shared_sum;
    for (uint k = tid; k < seq_kv; k += tg_size) {
        scores[k] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Compute output = scores @ V
    // Each thread handles multiple output dimensions
    for (uint d = tid; d < head_dim; d += tg_size) {
        float out_val = 0.0f;
        for (uint k = 0; k < seq_kv; k++) {
            out_val += scores[k] * v_head[k * head_dim + d];
        }
        out_head[d] = out_val;
    }
}
)";


// Implementation class
class MetalCompute::Impl {
public:
    Impl() = default;
    ~Impl() { shutdown(); }

    Result<void> initialize(MTL::Device* device) {
        if (initialized_) return {};

        device_ = device;
        if (!device_) {
            return Error(ErrorCode::BackendNotInitialized, "No Metal device");
        }

        command_queue_ = device_->newCommandQueue();
        if (!command_queue_) {
            return Error(ErrorCode::InternalError, "Failed to create command queue");
        }

        auto compile_result = compile_shaders();
        if (!compile_result.ok()) {
            return compile_result.error();
        }

        initialized_ = true;
        GRANITE_LOG_INFO("MetalCompute initialized");
        return {};
    }

    void shutdown() {
        if (!initialized_) return;

        sync();

        for (auto& [name, pipeline] : pipelines_) {
            pipeline->release();
        }
        pipelines_.clear();

        if (command_queue_) {
            command_queue_->release();
            command_queue_ = nullptr;
        }

        initialized_ = false;
    }

    void sync() {
        if (current_command_buffer_) {
            if (current_encoder_) {
                current_encoder_->endEncoding();
                current_encoder_ = nullptr;
            }
            current_command_buffer_->commit();
            current_command_buffer_->waitUntilCompleted();
            current_command_buffer_->release();
            current_command_buffer_ = nullptr;
        }
    }

    void commit() {
        if (current_command_buffer_) {
            if (current_encoder_) {
                current_encoder_->endEncoding();
                current_encoder_ = nullptr;
            }
            current_command_buffer_->commit();
            current_command_buffer_ = nullptr;
        }
    }

    bool is_initialized() const { return initialized_; }
    MTL::Device* device() const { return device_; }

    MTL::ComputeCommandEncoder* get_encoder() {
        if (!current_encoder_) {
            current_command_buffer_ = command_queue_->commandBuffer();
            current_encoder_ = current_command_buffer_->computeCommandEncoder();
        }
        return current_encoder_;
    }

    MTL::ComputePipelineState* get_pipeline(const std::string& name) {
        auto it = pipelines_.find(name);
        return (it != pipelines_.end()) ? it->second : nullptr;
    }

    MTL::Buffer* create_buffer(size_t size, bool shared) {
        MTL::ResourceOptions options = shared ?
            MTL::ResourceStorageModeShared :
            MTL::ResourceStorageModePrivate;
        return device_->newBuffer(size, options);
    }

private:
    Result<void> compile_shaders() {
        NS::Error* error = nullptr;
        MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();

        NS::String* source = NS::String::string(QUANTIZED_MATMUL_SHADER, NS::UTF8StringEncoding);
        MTL::Library* library = device_->newLibrary(source, options, &error);
        options->release();

        if (!library) {
            std::string msg = "Shader compilation failed";
            if (error) {
                msg = error->localizedDescription()->utf8String();
            }
            return Error(ErrorCode::ShaderCompilationFailed, msg);
        }

        std::vector<std::string> kernels = {
            "matvec_q4k", "matmul_q4k", "matvec_f16", "matvec_f32",
            "rms_norm", "rms_norm_f16", "silu", "elementwise_mul", "rope",
            "elementwise_add", "rope_multihead", "softmax_row", "attention_decode",
            "kv_cache_append", "multihead_attention_decode", "embedding_lookup"
        };

        for (const auto& name : kernels) {
            NS::String* func_name = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
            MTL::Function* func = library->newFunction(func_name);

            if (!func) {
                GRANITE_LOG_WARN("Function '{}' not found in shader library", name);
                continue;
            }

            MTL::ComputePipelineState* pipeline = device_->newComputePipelineState(func, &error);
            func->release();

            if (!pipeline) {
                GRANITE_LOG_WARN("Failed to create pipeline for '{}'", name);
                continue;
            }

            pipelines_[name] = pipeline;
            GRANITE_LOG_DEBUG("Created Metal pipeline: {}", name);
        }

        library->release();
        return {};
    }

    bool initialized_ = false;
    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* command_queue_ = nullptr;
    MTL::CommandBuffer* current_command_buffer_ = nullptr;
    MTL::ComputeCommandEncoder* current_encoder_ = nullptr;
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines_;
};

// MetalCompute public interface implementation
MetalCompute::MetalCompute() : impl_(new Impl()) {}
MetalCompute::~MetalCompute() { delete impl_; }

Result<void> MetalCompute::initialize(MTL::Device* device) {
    return impl_->initialize(device);
}

void MetalCompute::shutdown() { impl_->shutdown(); }
void MetalCompute::sync() { impl_->sync(); }
void MetalCompute::commit() { impl_->commit(); }
bool MetalCompute::is_initialized() const { return impl_->is_initialized(); }
MTL::Device* MetalCompute::device() const { return impl_->device(); }

MTL::Buffer* MetalCompute::create_buffer(size_t size, bool shared) {
    return impl_->create_buffer(size, shared);
}

Result<void> MetalCompute::matvec_q4k(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_q4k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_q4k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // Each threadgroup (32 threads = 1 SIMD group) handles one output row
    // N threadgroups total, one per output dimension
    MTL::Size grid_size = MTL::Size::Make(N, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32, 1, 1);  // One SIMD group
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matmul_q4k(
    MTL::Buffer* X, MTL::Buffer* W, MTL::Buffer* Y,
    uint32_t M, uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matmul_q4k");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matmul_q4k pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(X, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(Y, 0, 2);
    encoder->setBytes(&M, sizeof(M), 3);
    encoder->setBytes(&K, sizeof(K), 4);
    encoder->setBytes(&N, sizeof(N), 5);

    MTL::Size grid_size = MTL::Size::Make(N, M, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(16, 16, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::matvec_f16(
    MTL::Buffer* x, MTL::Buffer* W, MTL::Buffer* y,
    uint32_t K, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("matvec_f16");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "matvec_f16 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);
    encoder->setBytes(&K, sizeof(K), 3);
    encoder->setBytes(&N, sizeof(N), 4);

    // Each threadgroup (32 threads = 1 SIMD group) handles one output row
    MTL::Size grid_size = MTL::Size::Make(N, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(32, 1, 1);  // One SIMD group
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm(
    MTL::Buffer* x, MTL::Buffer* weight, MTL::Buffer* out,
    uint32_t size, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(weight, 0, 1);
    encoder->setBuffer(out, 0, 2);
    encoder->setBytes(&size, sizeof(size), 3);
    encoder->setBytes(&eps, sizeof(eps), 4);

    // Use single threadgroup for reduction
    uint32_t tg_size = std::min((uint32_t)256, size);
    MTL::Size grid_size = MTL::Size::Make(tg_size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(tg_size, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rms_norm_f16(
    MTL::Buffer* x, MTL::Buffer* weight, MTL::Buffer* out,
    uint32_t size, float eps)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rms_norm_f16");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rms_norm_f16 pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(weight, 0, 1);
    encoder->setBuffer(out, 0, 2);
    encoder->setBytes(&size, sizeof(size), 3);
    encoder->setBytes(&eps, sizeof(eps), 4);

    // Use single threadgroup for reduction
    uint32_t tg_size = std::min((uint32_t)256, size);
    MTL::Size grid_size = MTL::Size::Make(tg_size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(tg_size, 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::silu(MTL::Buffer* x, uint32_t size) {
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("silu");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "silu pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBytes(&size, sizeof(size), 1);

    MTL::Size grid_size = MTL::Size::Make(size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(std::min((uint32_t)256, size), 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::elementwise_mul(
    MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* c, uint32_t size)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("elementwise_mul");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "elementwise_mul pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(a, 0, 0);
    encoder->setBuffer(b, 0, 1);
    encoder->setBuffer(c, 0, 2);
    encoder->setBytes(&size, sizeof(size), 3);

    MTL::Size grid_size = MTL::Size::Make(size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(std::min((uint32_t)256, size), 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rope(
    MTL::Buffer* x, uint32_t seq_len, uint32_t head_dim,
    uint32_t start_pos, float freq_base)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rope");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rope pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBytes(&seq_len, sizeof(seq_len), 1);
    encoder->setBytes(&head_dim, sizeof(head_dim), 2);
    encoder->setBytes(&start_pos, sizeof(start_pos), 3);
    encoder->setBytes(&freq_base, sizeof(freq_base), 4);

    MTL::Size grid_size = MTL::Size::Make(head_dim / 2, seq_len, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(16, 16, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::elementwise_add(
    MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* c, uint32_t size)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("elementwise_add");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "elementwise_add pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(a, 0, 0);
    encoder->setBuffer(b, 0, 1);
    encoder->setBuffer(c, 0, 2);
    encoder->setBytes(&size, sizeof(size), 3);

    MTL::Size grid_size = MTL::Size::Make(size, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(std::min((uint32_t)256, size), 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::rope_multihead(
    MTL::Buffer* q, MTL::Buffer* k,
    uint32_t num_heads_q, uint32_t num_heads_k,
    uint32_t seq_len, uint32_t head_dim,
    uint32_t start_pos, float freq_base)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("rope_multihead");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "rope_multihead pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(q, 0, 0);
    encoder->setBuffer(k, 0, 1);
    encoder->setBytes(&num_heads_q, sizeof(num_heads_q), 2);
    encoder->setBytes(&num_heads_k, sizeof(num_heads_k), 3);
    encoder->setBytes(&seq_len, sizeof(seq_len), 4);
    encoder->setBytes(&head_dim, sizeof(head_dim), 5);
    encoder->setBytes(&start_pos, sizeof(start_pos), 6);
    encoder->setBytes(&freq_base, sizeof(freq_base), 7);

    uint32_t max_heads = std::max(num_heads_q, num_heads_k);
    MTL::Size grid_size = MTL::Size::Make(head_dim / 2, max_heads, seq_len);
    MTL::Size threadgroup_size = MTL::Size::Make(
        std::min((uint32_t)32, head_dim / 2),
        std::min((uint32_t)4, max_heads),
        std::min((uint32_t)2, seq_len)
    );
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::softmax(
    MTL::Buffer* x, uint32_t M, uint32_t N)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("softmax_row");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "softmax_row pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(x, 0, 0);
    encoder->setBytes(&M, sizeof(M), 1);
    encoder->setBytes(&N, sizeof(N), 2);

    MTL::Size grid_size = MTL::Size::Make(M, 1, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(std::min((uint32_t)256, M), 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::attention_single_head(
    MTL::Buffer* Q, MTL::Buffer* K, MTL::Buffer* V, MTL::Buffer* output,
    uint32_t seq_q, uint32_t seq_kv, uint32_t head_dim,
    uint32_t start_pos, float scale)
{
    // For decode (seq_q == 1), use optimized attention_decode kernel
    if (seq_q == 1) {
        auto* encoder = impl_->get_encoder();
        auto* pipeline = impl_->get_pipeline("attention_decode");
        if (!pipeline) {
            return Error(ErrorCode::InternalError, "attention_decode pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(Q, 0, 0);
        encoder->setBuffer(K, 0, 1);
        encoder->setBuffer(V, 0, 2);
        encoder->setBuffer(output, 0, 3);
        encoder->setBytes(&seq_kv, sizeof(seq_kv), 4);
        encoder->setBytes(&head_dim, sizeof(head_dim), 5);
        encoder->setBytes(&scale, sizeof(scale), 6);

        // Use one threadgroup for the whole head
        MTL::Size grid_size = MTL::Size::Make(head_dim, 1, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(head_dim, 1, 1);
        encoder->dispatchThreads(grid_size, threadgroup_size);

        return {};
    }

    // For prefill (seq_q > 1), we would need a different kernel
    // For now, return error (caller should fall back to CPU)
    return Error(ErrorCode::NotImplemented, "GPU prefill attention not implemented");
}

std::pair<MTL::Buffer*, MTL::Buffer*> MetalCompute::create_kv_cache(
    uint32_t num_kv_heads,
    uint32_t max_seq_len,
    uint32_t head_dim)
{
    size_t size = num_kv_heads * max_seq_len * head_dim * sizeof(float);
    MTL::Buffer* k_cache = impl_->create_buffer(size, true);
    MTL::Buffer* v_cache = impl_->create_buffer(size, true);
    return {k_cache, v_cache};
}

Result<void> MetalCompute::kv_cache_append(
    MTL::Buffer* cache_k,
    MTL::Buffer* cache_v,
    MTL::Buffer* new_k,
    MTL::Buffer* new_v,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t current_len,
    uint32_t new_len,
    uint32_t max_seq_len)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("kv_cache_append");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "kv_cache_append pipeline not found");
    }

    // Append K
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(new_k, 0, 0);
    encoder->setBuffer(cache_k, 0, 1);
    encoder->setBytes(&num_kv_heads, sizeof(num_kv_heads), 2);
    encoder->setBytes(&head_dim, sizeof(head_dim), 3);
    encoder->setBytes(&current_len, sizeof(current_len), 4);
    encoder->setBytes(&new_len, sizeof(new_len), 5);
    encoder->setBytes(&max_seq_len, sizeof(max_seq_len), 6);

    MTL::Size grid_size = MTL::Size::Make(head_dim, new_len, num_kv_heads);
    MTL::Size threadgroup_size = MTL::Size::Make(
        std::min((uint32_t)64, head_dim),
        1,
        1
    );
    encoder->dispatchThreads(grid_size, threadgroup_size);

    // Append V
    encoder->setBuffer(new_v, 0, 0);
    encoder->setBuffer(cache_v, 0, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

Result<void> MetalCompute::multihead_attention(
    MTL::Buffer* Q,
    MTL::Buffer* K,
    MTL::Buffer* V,
    MTL::Buffer* output,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t seq_q,
    uint32_t seq_kv,
    uint32_t head_dim,
    float scale)
{
    // For decode (seq_q == 1), use multihead_attention_decode kernel
    if (seq_q == 1) {
        auto* encoder = impl_->get_encoder();
        auto* pipeline = impl_->get_pipeline("multihead_attention_decode");
        if (!pipeline) {
            return Error(ErrorCode::InternalError, "multihead_attention_decode pipeline not found");
        }

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(Q, 0, 0);
        encoder->setBuffer(K, 0, 1);
        encoder->setBuffer(V, 0, 2);
        encoder->setBuffer(output, 0, 3);
        encoder->setBytes(&num_heads, sizeof(num_heads), 4);
        encoder->setBytes(&num_kv_heads, sizeof(num_kv_heads), 5);
        encoder->setBytes(&seq_kv, sizeof(seq_kv), 6);
        encoder->setBytes(&head_dim, sizeof(head_dim), 7);
        encoder->setBytes(&scale, sizeof(scale), 8);

        // One threadgroup per head, with enough threads for parallel work
        uint32_t threads_per_group = std::min((uint32_t)128, std::max((uint32_t)32, head_dim));
        MTL::Size grid_size = MTL::Size::Make(num_heads, 1, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(threads_per_group, 1, 1);
        encoder->dispatchThreadgroups(grid_size, threadgroup_size);

        return {};
    }

    return Error(ErrorCode::NotImplemented, "GPU prefill multihead attention not implemented");
}

Result<void> MetalCompute::embedding_lookup(
    MTL::Buffer* token_ids,
    MTL::Buffer* embeddings,
    MTL::Buffer* output,
    uint32_t num_tokens,
    uint32_t hidden_dim,
    uint32_t vocab_size)
{
    auto* encoder = impl_->get_encoder();
    auto* pipeline = impl_->get_pipeline("embedding_lookup");
    if (!pipeline) {
        return Error(ErrorCode::InternalError, "embedding_lookup pipeline not found");
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(token_ids, 0, 0);
    encoder->setBuffer(embeddings, 0, 1);
    encoder->setBuffer(output, 0, 2);
    encoder->setBytes(&hidden_dim, sizeof(hidden_dim), 3);
    encoder->setBytes(&vocab_size, sizeof(vocab_size), 4);

    // 2D grid: [hidden_dim, num_tokens]
    MTL::Size grid_size = MTL::Size::Make(hidden_dim, num_tokens, 1);
    MTL::Size threadgroup_size = MTL::Size::Make(std::min((uint32_t)256, hidden_dim), 1, 1);
    encoder->dispatchThreads(grid_size, threadgroup_size);

    return {};
}

// Global singleton
static std::unique_ptr<MetalCompute> g_metal_compute;
static std::once_flag g_metal_compute_init;

MetalCompute* get_metal_compute() {
    std::call_once(g_metal_compute_init, []() {
        g_metal_compute = std::make_unique<MetalCompute>();

        MTL::Device* device = MTL::CreateSystemDefaultDevice();
        if (device) {
            auto result = g_metal_compute->initialize(device);
            if (!result.ok()) {
                GRANITE_LOG_ERROR("Failed to initialize MetalCompute: {}",
                                 result.error().message());
                g_metal_compute.reset();
            }
        }
    });

    return g_metal_compute.get();
}

}  // namespace granite

#endif  // GRANITE_HAS_METAL
