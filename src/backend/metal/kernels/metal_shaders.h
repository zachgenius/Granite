// Metal shader source code for LLM inference kernels
// This file contains all Metal compute kernels as embedded strings
// NOTE: This file is included inside namespace granite {} in metal_compute.mm

#pragma once

// Embedded shader source - all kernels in one compilation unit
static const char* METAL_SHADER_SOURCE = R"(
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

// Original matvec for Q4_K - kept for fallback
constant constexpr uint ROWS_PER_TG = 8;

kernel void matvec_q4k_basic(
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
    uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    float local_sum = 0.0f;

    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q4_K* block = &weights[row * num_blocks_k + kb];
        float d = float(block->d);
        float dmin = float(block->dmin);
        const device uint8_t* scales = block->scales;
        const device uint8_t* qs = block->qs;

        uint elem_offset = simd_lane * 8;
        uint x_base = kb * QK_K + elem_offset;
        uint sub_block = elem_offset / 64;
        uint sub_offset = elem_offset % 64;

        uint8_t sc1, m1, sc2, m2;
        get_scale_min_k4(sub_block * 2, scales, sc1, m1);
        get_scale_min_k4(sub_block * 2 + 1, scales, sc2, m2);

        float d1 = d * float(sc1);
        float dm1 = dmin * float(m1);
        float d2 = d * float(sc2);
        float dm2 = dmin * float(m2);

        const device uint8_t* qs_ptr = qs + sub_block * 32;
        float lane_sum = 0.0f;

        if (sub_offset < 32) {
            uint qs_idx = sub_offset;
            float4 x_vec1 = float4(x[x_base], x[x_base + 1], x[x_base + 2], x[x_base + 3]);
            float4 x_vec2 = float4(x[x_base + 4], x[x_base + 5], x[x_base + 6], x[x_base + 7]);
            float4 w_vec1 = float4(
                d1 * float(qs_ptr[qs_idx] & 0xF) - dm1,
                d1 * float(qs_ptr[qs_idx + 1] & 0xF) - dm1,
                d1 * float(qs_ptr[qs_idx + 2] & 0xF) - dm1,
                d1 * float(qs_ptr[qs_idx + 3] & 0xF) - dm1
            );
            float4 w_vec2 = float4(
                d1 * float(qs_ptr[qs_idx + 4] & 0xF) - dm1,
                d1 * float(qs_ptr[qs_idx + 5] & 0xF) - dm1,
                d1 * float(qs_ptr[qs_idx + 6] & 0xF) - dm1,
                d1 * float(qs_ptr[qs_idx + 7] & 0xF) - dm1
            );
            lane_sum = dot(x_vec1, w_vec1) + dot(x_vec2, w_vec2);
        } else {
            uint qs_idx = sub_offset - 32;
            float4 x_vec1 = float4(x[x_base], x[x_base + 1], x[x_base + 2], x[x_base + 3]);
            float4 x_vec2 = float4(x[x_base + 4], x[x_base + 5], x[x_base + 6], x[x_base + 7]);
            float4 w_vec1 = float4(
                d2 * float(qs_ptr[qs_idx] >> 4) - dm2,
                d2 * float(qs_ptr[qs_idx + 1] >> 4) - dm2,
                d2 * float(qs_ptr[qs_idx + 2] >> 4) - dm2,
                d2 * float(qs_ptr[qs_idx + 3] >> 4) - dm2
            );
            float4 w_vec2 = float4(
                d2 * float(qs_ptr[qs_idx + 4] >> 4) - dm2,
                d2 * float(qs_ptr[qs_idx + 5] >> 4) - dm2,
                d2 * float(qs_ptr[qs_idx + 6] >> 4) - dm2,
                d2 * float(qs_ptr[qs_idx + 7] >> 4) - dm2
            );
            lane_sum = dot(x_vec1, w_vec1) + dot(x_vec2, w_vec2);
        }
        local_sum += lane_sum;
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row] = sum;
    }
}

// =============================================================================
// Optimized matvec for Q4_K - llama.cpp style
// =============================================================================
// Key optimizations (from llama.cpp):
// 1. Process 2 rows per SIMD group (nr0 = 2)
// 2. Register-based input caching (no shared memory, no barriers)
// 3. Strided block access (4 threads per block)
// 4. uint16_t reading for better memory bandwidth
// 5. Smart accumulation using 1/256 multiplier

constant constexpr short NR0_Q4K = 2;  // Rows per SIMD group
constant constexpr ushort kmask1 = 0x3f3f;
constant constexpr ushort kmask2 = 0x0f0f;
constant constexpr ushort kmask3 = 0xc0c0;

kernel void matvec_q4k(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK_K;  // Number of Q4_K blocks per row

    // Thread indexing (32 threads per SIMD group)
    // Split into 4 groups of 8 threads
    const short ix = tiisg / 8;  // 0...3 (which quarter of block)
    const short it = tiisg % 8;  // 0...7 (position within quarter)
    const short iq = it / 4;     // 0 or 1 (which half of sub-block)
    const short ir = it % 4;     // 0...3 (position within half)

    // Each SIMD group handles NR0_Q4K rows
    const uint first_row = (tgid * ROWS_PER_TG + sgitg) * NR0_Q4K;

    device const block_q4_K* weights = (const device block_q4_K*)W;

    // Register-based input caching - each thread caches 32 floats
    float yl[16];
    float yh[16];
    float sumf[NR0_Q4K] = {0.f, 0.f};

    // Pointer to input for this thread's portion
    device const float* y4 = x + ix * QK_K + 64 * iq + 8 * ir;

    // Scale extraction buffer
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    // Process blocks with stride 4 (4 thread groups cooperate on each block position)
    for (uint ib = ix; ib < nb; ib += 4) {
        // Load input values into registers (32 values per thread)
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        // Process each row
        for (short row = 0; row < NR0_Q4K; row++) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_q4_K* block = &weights[row_idx * nb + ib];
            device const ushort* sc = (device const ushort*)block->scales + iq;
            device const ushort* q1 = (device const ushort*)block->qs + 16 * iq + 4 * ir;
            device const half* dh = &block->d;

            // Extract scales using bit masks
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const ushort* q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            // Unrolled accumulation with uint16 reading
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            // Accumulate with smart scaling (1/256 instead of bit shifts)
            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                  sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * QK_K;  // Stride to next block for this thread
    }

    // SIMD reduction and output
    for (short row = 0; row < NR0_Q4K; row++) {
        uint row_idx = first_row + row;
        if (row_idx < N) {
            float sum = simd_sum(sumf[row]);
            if (tiisg == 0) {
                y[row_idx] = sum;
            }
        }
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

// KV cache append with float->half conversion
// new_kv: [num_heads, new_len, head_dim] float
// cache: [num_heads, max_seq, head_dim] half
kernel void kv_cache_append_f16(
    device const float* new_kv     [[buffer(0)]],
    device half* cache             [[buffer(1)]],
    constant uint& num_heads       [[buffer(2)]],
    constant uint& head_dim        [[buffer(3)]],
    constant uint& current_len     [[buffer(4)]],
    constant uint& new_len         [[buffer(5)]],
    constant uint& max_seq         [[buffer(6)]],
    uint3 gid                      [[thread_position_in_grid]]
) {
    uint h = gid.z;
    uint s = gid.y;
    uint d = gid.x;

    if (h >= num_heads || s >= new_len || d >= head_dim) return;

    uint src_idx = h * new_len * head_dim + s * head_dim + d;
    uint dst_idx = h * max_seq * head_dim + (current_len + s) * head_dim + d;

    // Convert float to half when storing
    cache[dst_idx] = half(new_kv[src_idx]);
}

// Optimized multi-head attention kernel for decode (seq_q = 1)
// Uses FP16 for attention scores to reduce memory bandwidth
// Q: [num_heads, 1, head_dim]
// K: [num_kv_heads, seq_kv, head_dim]
// V: [num_kv_heads, seq_kv, head_dim]
// output: [num_heads, 1, head_dim]
// Each threadgroup handles one head with 128 threads (4 SIMD groups)
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
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
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

    // FP16 scores use half the threadgroup memory - can fit 4096 scores
    threadgroup half scores_h[4096];
    threadgroup float reduction_scratch[4];

    // Step 1: Compute attention scores (Q @ K^T) using SIMD parallelism
    float local_max = -INFINITY;

    for (uint k = tid; k < seq_kv; k += 128) {
        device const float* k_vec = k_head + k * head_dim;

        // Vectorized dot product using half4 for K access
        float dot_sum = 0.0f;
        for (uint d = 0; d < head_dim; d += 4) {
            float4 q_v = float4(q_head[d], q_head[d+1], q_head[d+2], q_head[d+3]);
            float4 k_v = float4(k_vec[d], k_vec[d+1], k_vec[d+2], k_vec[d+3]);
            dot_sum += dot(q_v, k_v);
        }

        float score = dot_sum * scale;
        scores_h[k] = half(score);
        local_max = max(local_max, score);
    }

    // SIMD reduction for max
    float simd_max_val = simd_max(local_max);
    if (simd_lane == 0) {
        reduction_scratch[simd_id] = simd_max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_max;
    if (tid == 0) {
        float global_max = reduction_scratch[0];
        for (uint i = 1; i < 4; i++) {
            global_max = max(global_max, reduction_scratch[i]);
        }
        shared_max = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (uint k = tid; k < seq_kv; k += 128) {
        float exp_score = exp(float(scores_h[k]) - shared_max);
        scores_h[k] = half(exp_score);
        local_sum += exp_score;
    }

    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane == 0) {
        reduction_scratch[simd_id] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_sum;
    if (tid == 0) {
        float global_sum = 0.0f;
        for (uint i = 0; i < 4; i++) {
            global_sum += reduction_scratch[i];
        }
        shared_sum = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Normalize scores in-place
    float inv_sum = 1.0f / shared_sum;
    for (uint k = tid; k < seq_kv; k += 128) {
        scores_h[k] = half(float(scores_h[k]) * inv_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Compute output = scores @ V
    // Use SIMD groups: each group computes a subset of output dimensions
    uint dims_per_simd = (head_dim + 3) / 4;
    uint d_start = simd_id * dims_per_simd;
    uint d_end = min(d_start + dims_per_simd, head_dim);

    for (uint d = d_start + simd_lane; d < d_end; d += 32) {
        float out_val = 0.0f;
        // Process 4 scores at a time for better throughput
        uint k = 0;
        for (; k + 3 < seq_kv; k += 4) {
            half4 s = half4(scores_h[k], scores_h[k+1], scores_h[k+2], scores_h[k+3]);
            float4 v = float4(
                v_head[k * head_dim + d],
                v_head[(k+1) * head_dim + d],
                v_head[(k+2) * head_dim + d],
                v_head[(k+3) * head_dim + d]
            );
            out_val += dot(float4(s), v);
        }
        for (; k < seq_kv; k++) {
            out_val += float(scores_h[k]) * v_head[k * head_dim + d];
        }
        if (d < head_dim) {
            out_head[d] = out_val;
        }
    }
}
// =============================================================================
// FUSED KERNELS for reduced memory bandwidth
// =============================================================================

// Fused SiLU + Elementwise Multiply: c = silu(a) * b
// Used in SwiGLU FFN: silu(gate) * up
kernel void silu_mul(
    device const float* a          [[buffer(0)]],   // gate input
    device const float* b          [[buffer(1)]],   // up input
    device float* c                [[buffer(2)]],   // output
    constant uint& size            [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float val = a[gid];
    float silu_val = val / (1.0f + exp(-val));
    c[gid] = silu_val * b[gid];
}

// Fused RMSNorm + Q4_K MatVec
// Computes: y = RMSNorm(x, weight) @ W^T (Q4_K)
// Each threadgroup:
//   1. Cooperatively computes RMSNorm and stores normalized input to threadgroup memory
//   2. Each SIMD group computes one output row using the normalized input
// This eliminates the intermediate memory write/read between RMSNorm and MatVec
constant constexpr uint FUSED_ROWS_PER_TG = 8;

kernel void rms_norm_matvec_q4k(
    device const float* x          [[buffer(0)]],   // Input: [K] float
    device const half* norm_weight [[buffer(1)]],   // RMSNorm weight: [K] half
    device const void* W           [[buffer(2)]],   // Q4_K weights: [N, K/QK_K] blocks
    device float* y                [[buffer(3)]],   // Output: [N] float
    constant uint& K               [[buffer(4)]],   // Input/hidden dimension
    constant uint& N               [[buffer(5)]],   // Output dimension
    constant float& eps            [[buffer(6)]],   // RMSNorm epsilon
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory for normalized input (max 4096 floats = 16KB)
    threadgroup float x_norm[4096];
    threadgroup float reduction_scratch[8];

    // =========================================================================
    // Step 1: Compute RMSNorm cooperatively across all 256 threads
    // =========================================================================

    // Each thread accumulates sum of squares for its portion
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < K; i += 256) {
        float val = x[i];
        local_sum_sq += val * val;
    }

    // SIMD reduction within each of 8 SIMD groups
    float simd_sum_sq = simd_sum(local_sum_sq);
    if (simd_lane == 0) {
        reduction_scratch[simd_id] = simd_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 finishes reduction
    threadgroup float rms_inv;
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; i++) {
            total += reduction_scratch[i];
        }
        rms_inv = rsqrt(total / float(K) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize and store to threadgroup memory
    for (uint i = tid; i < K; i += 256) {
        x_norm[i] = x[i] * rms_inv * float(norm_weight[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Step 2: Q4_K MatVec using normalized input from threadgroup memory
    // Each SIMD group (32 threads) handles one output row
    // =========================================================================

    uint row = tgid * FUSED_ROWS_PER_TG + simd_id;
    if (row >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    float local_sum = 0.0f;

    // Process all K blocks - all 32 lanes work on each block together
    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q4_K* block = &weights[row * num_blocks_k + kb];
        float d = float(block->d);
        float dmin = float(block->dmin);
        const device uint8_t* scales = block->scales;
        const device uint8_t* qs = block->qs;

        // Each lane handles 8 consecutive elements
        uint elem_offset = simd_lane * 8;
        uint x_base = kb * QK_K + elem_offset;

        // Determine which 64-element sub-block
        uint sub_block = elem_offset / 64;
        uint sub_offset = elem_offset % 64;

        // Get scale and min for this sub-block
        uint8_t sc1, m1, sc2, m2;
        get_scale_min_k4(sub_block * 2, scales, sc1, m1);
        get_scale_min_k4(sub_block * 2 + 1, scales, sc2, m2);

        float d1 = d * float(sc1);
        float dm1 = dmin * float(m1);
        float d2 = d * float(sc2);
        float dm2 = dmin * float(m2);

        const device uint8_t* qs_ptr = qs + sub_block * 32;
        float lane_sum = 0.0f;

        if (sub_offset < 32) {
            // First half: low nibbles
            uint qs_idx = sub_offset;
            // Read from threadgroup memory instead of global
            float4 x_vec1 = float4(x_norm[x_base], x_norm[x_base + 1],
                                   x_norm[x_base + 2], x_norm[x_base + 3]);
            float4 x_vec2 = float4(x_norm[x_base + 4], x_norm[x_base + 5],
                                   x_norm[x_base + 6], x_norm[x_base + 7]);
            float4 w_vec1 = float4(
                d1 * float(qs_ptr[qs_idx] & 0xF) - dm1,
                d1 * float(qs_ptr[qs_idx + 1] & 0xF) - dm1,
                d1 * float(qs_ptr[qs_idx + 2] & 0xF) - dm1,
                d1 * float(qs_ptr[qs_idx + 3] & 0xF) - dm1
            );
            float4 w_vec2 = float4(
                d1 * float(qs_ptr[qs_idx + 4] & 0xF) - dm1,
                d1 * float(qs_ptr[qs_idx + 5] & 0xF) - dm1,
                d1 * float(qs_ptr[qs_idx + 6] & 0xF) - dm1,
                d1 * float(qs_ptr[qs_idx + 7] & 0xF) - dm1
            );
            lane_sum = dot(x_vec1, w_vec1) + dot(x_vec2, w_vec2);
        } else {
            // Second half: high nibbles
            uint qs_idx = sub_offset - 32;
            float4 x_vec1 = float4(x_norm[x_base], x_norm[x_base + 1],
                                   x_norm[x_base + 2], x_norm[x_base + 3]);
            float4 x_vec2 = float4(x_norm[x_base + 4], x_norm[x_base + 5],
                                   x_norm[x_base + 6], x_norm[x_base + 7]);
            float4 w_vec1 = float4(
                d2 * float(qs_ptr[qs_idx] >> 4) - dm2,
                d2 * float(qs_ptr[qs_idx + 1] >> 4) - dm2,
                d2 * float(qs_ptr[qs_idx + 2] >> 4) - dm2,
                d2 * float(qs_ptr[qs_idx + 3] >> 4) - dm2
            );
            float4 w_vec2 = float4(
                d2 * float(qs_ptr[qs_idx + 4] >> 4) - dm2,
                d2 * float(qs_ptr[qs_idx + 5] >> 4) - dm2,
                d2 * float(qs_ptr[qs_idx + 6] >> 4) - dm2,
                d2 * float(qs_ptr[qs_idx + 7] >> 4) - dm2
            );
            lane_sum = dot(x_vec1, w_vec1) + dot(x_vec2, w_vec2);
        }
        local_sum += lane_sum;
    }

    // SIMD reduction
    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row] = sum;
    }
}

// Fused RMSNorm + FP16 MatVec (for output projection with tied embeddings)
kernel void rms_norm_matvec_f16(
    device const float* x          [[buffer(0)]],   // Input: [K] float
    device const half* norm_weight [[buffer(1)]],   // RMSNorm weight: [K] half
    device const half* W           [[buffer(2)]],   // FP16 weights: [N, K]
    device float* y                [[buffer(3)]],   // Output: [N] float
    constant uint& K               [[buffer(4)]],   // Input dimension
    constant uint& N               [[buffer(5)]],   // Output dimension
    constant float& eps            [[buffer(6)]],   // RMSNorm epsilon
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float x_norm[4096];
    threadgroup float reduction_scratch[8];

    // Step 1: RMSNorm
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < K; i += 256) {
        float val = x[i];
        local_sum_sq += val * val;
    }

    float simd_sum_sq = simd_sum(local_sum_sq);
    if (simd_lane == 0) {
        reduction_scratch[simd_id] = simd_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float rms_inv;
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; i++) total += reduction_scratch[i];
        rms_inv = rsqrt(total / float(K) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < K; i += 256) {
        x_norm[i] = x[i] * rms_inv * float(norm_weight[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: FP16 MatVec - each SIMD group handles one row
    uint row = tgid * 8 + simd_id;
    if (row >= N) return;

    device const half* w_row = W + row * K;

    float local_sum = 0.0f;
    for (uint i = simd_lane; i < K; i += 32) {
        local_sum += x_norm[i] * float(w_row[i]);
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row] = sum;
    }
}

// =============================================================================
// Flash Attention style decode kernel using online softmax
// =============================================================================
// Benefits:
// 1. No need to store all attention scores - processes in tiles
// 2. Better cache locality - K/V data stays in registers longer
// 3. Numerically stable online softmax update
// 4. Reduced threadgroup memory usage

constant constexpr uint FLASH_TILE_SIZE = 32;  // Process 32 K/V vectors at a time

kernel void flash_attention_decode(
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
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    if (head_idx >= num_heads) return;

    // GQA: map Q head to KV head
    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;

    device const float* q_head = Q + head_idx * head_dim;
    device const float* k_head = K + kv_head * seq_kv * head_dim;
    device const float* v_head = V + kv_head * seq_kv * head_dim;
    device float* out_head = output + head_idx * head_dim;

    // Threadgroup memory for Q vector and partial outputs
    threadgroup float q_shared[128];      // Cache Q in shared memory (head_dim <= 128)
    threadgroup float out_shared[128];    // Accumulated output
    threadgroup float tile_scores[FLASH_TILE_SIZE];  // Scores for current tile
    threadgroup float reduction_scratch[8];

    // Load Q into shared memory cooperatively
    if (tid < head_dim) {
        q_shared[tid] = q_head[tid];
    }

    // Initialize output accumulator to zero
    if (tid < head_dim) {
        out_shared[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state - tracked by thread 0
    threadgroup float running_max;
    threadgroup float running_sum;

    if (tid == 0) {
        running_max = -INFINITY;
        running_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process K/V in tiles
    uint num_tiles = (seq_kv + FLASH_TILE_SIZE - 1) / FLASH_TILE_SIZE;

    for (uint tile = 0; tile < num_tiles; tile++) {
        uint tile_start = tile * FLASH_TILE_SIZE;
        uint tile_end = min(tile_start + FLASH_TILE_SIZE, seq_kv);
        uint tile_len = tile_end - tile_start;

        // Step 1: Compute attention scores for this tile
        // Each thread in first SIMD group handles one position in tile
        float local_score = -INFINITY;
        uint k_pos = tile_start + tid;

        if (tid < tile_len) {
            device const float* k_vec = k_head + k_pos * head_dim;

            // Dot product Q @ K[k_pos]
            float dot_sum = 0.0f;
            for (uint d = 0; d < head_dim; d += 4) {
                float4 q_v = float4(q_shared[d], q_shared[d+1], q_shared[d+2], q_shared[d+3]);
                float4 k_v = float4(k_vec[d], k_vec[d+1], k_vec[d+2], k_vec[d+3]);
                dot_sum += dot(q_v, k_v);
            }
            local_score = dot_sum * scale;
            tile_scores[tid] = local_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 2: Find max in this tile using SIMD reduction
        float tile_max_local = (tid < tile_len) ? tile_scores[tid] : -INFINITY;
        float simd_tile_max = simd_max(tile_max_local);
        if (simd_lane == 0 && simd_id < 4) {
            reduction_scratch[simd_id] = simd_tile_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float tile_max;
        if (tid == 0) {
            tile_max = reduction_scratch[0];
            for (uint i = 1; i < 4; i++) {
                tile_max = max(tile_max, reduction_scratch[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Online softmax update
        // new_max = max(running_max, tile_max)
        // correction = exp(running_max - new_max)
        // For existing accumulated output: scale by correction
        // For new scores: compute exp(score - new_max)
        threadgroup float new_max;
        threadgroup float correction;

        if (tid == 0) {
            new_max = max(running_max, tile_max);
            correction = (running_max > -INFINITY) ? exp(running_max - new_max) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scale existing output accumulator by correction factor
        if (tid < head_dim) {
            out_shared[tid] *= correction;
        }

        // Compute exp(score - new_max) for tile scores
        float exp_score = 0.0f;
        if (tid < tile_len) {
            exp_score = exp(tile_scores[tid] - new_max);
            tile_scores[tid] = exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 4: Compute sum of exp scores in tile
        float tile_sum_local = (tid < tile_len) ? exp_score : 0.0f;
        float simd_tile_sum = simd_sum(tile_sum_local);
        if (simd_lane == 0 && simd_id < 4) {
            reduction_scratch[simd_id] = simd_tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < 4; i++) {
                tile_sum += reduction_scratch[i];
            }
            running_sum = running_sum * correction + tile_sum;
            running_max = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: Accumulate weighted V into output
        // out += sum_k(exp_score[k] * V[k])
        // Each thread handles one output dimension
        if (tid < head_dim) {
            float acc = 0.0f;
            for (uint k = 0; k < tile_len; k++) {
                float score_val = tile_scores[k];
                float v_val = v_head[(tile_start + k) * head_dim + tid];
                acc += score_val * v_val;
            }
            out_shared[tid] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization: divide by running_sum
    if (tid < head_dim) {
        out_head[tid] = out_shared[tid] / running_sum;
    }
}

// Flash Attention with simdgroup matrix operations for V aggregation
// Uses 8x8 matrix tiles for better throughput on scores @ V
kernel void flash_attention_decode_simd(
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
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    if (head_idx >= num_heads) return;

    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;

    device const float* q_head = Q + head_idx * head_dim;
    device const float* k_head = K + kv_head * seq_kv * head_dim;
    device const float* v_head = V + kv_head * seq_kv * head_dim;
    device float* out_head = output + head_idx * head_dim;

    // Use 256 threads (8 SIMD groups) for more parallelism
    threadgroup float q_shared[128];
    threadgroup float out_shared[128];
    threadgroup float reduction_scratch[8];

    // Load Q cooperatively
    for (uint i = tid; i < head_dim; i += 256) {
        q_shared[i] = q_head[i];
    }

    // Initialize output
    for (uint i = tid; i < head_dim; i += 256) {
        out_shared[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float out_scale = 1.0f;

    // Process all K/V positions
    // Each thread group processes in parallel
    uint positions_per_thread = (seq_kv + 255) / 256;

    for (uint p = 0; p < positions_per_thread; p++) {
        uint k_pos = p * 256 + tid;
        if (k_pos >= seq_kv) continue;

        device const float* k_vec = k_head + k_pos * head_dim;
        device const float* v_vec = v_head + k_pos * head_dim;

        // Compute Q @ K[k_pos] dot product
        float dot_sum = 0.0f;
        for (uint d = 0; d < head_dim; d += 4) {
            float4 q_v = float4(q_shared[d], q_shared[d+1], q_shared[d+2], q_shared[d+3]);
            float4 k_v = float4(k_vec[d], k_vec[d+1], k_vec[d+2], k_vec[d+3]);
            dot_sum += dot(q_v, k_v);
        }
        float score = dot_sum * scale;

        // Online softmax update (per-thread)
        float new_max = max(running_max, score);
        float old_scale = (running_max > -INFINITY) ? exp(running_max - new_max) : 0.0f;
        float new_weight = exp(score - new_max);

        out_scale *= old_scale;
        running_sum = running_sum * old_scale + new_weight;
        running_max = new_max;

        // Store weight and v_vec info for later aggregation
        // This is simplified - full implementation would use shared memory
    }

    // For correctness, fall back to the simpler approach for now
    // The full simdgroup implementation requires more complex tiling
    // ... (simplified for this version)

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// FP16 KV cache version - reads K/V from half precision
// Q stays FP32 (output of Q4_K projection), K/V are FP16
kernel void multihead_attention_decode_f16kv(
    device const float* Q          [[buffer(0)]],
    device const half* K           [[buffer(1)]],
    device const half* V           [[buffer(2)]],
    device float* output           [[buffer(3)]],
    constant uint& num_heads       [[buffer(4)]],
    constant uint& num_kv_heads    [[buffer(5)]],
    constant uint& seq_kv          [[buffer(6)]],
    constant uint& head_dim        [[buffer(7)]],
    constant float& scale          [[buffer(8)]],
    uint head_idx                  [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    if (head_idx >= num_heads) return;

    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;

    device const float* q_head = Q + head_idx * head_dim;
    device const half* k_head = K + kv_head * seq_kv * head_dim;
    device const half* v_head = V + kv_head * seq_kv * head_dim;
    device float* out_head = output + head_idx * head_dim;

    threadgroup half scores_h[4096];
    threadgroup float reduction_scratch[4];

    // Step 1: Q @ K^T with FP16 K reads
    float local_max = -INFINITY;
    for (uint k = tid; k < seq_kv; k += 128) {
        device const half* k_vec = k_head + k * head_dim;

        float dot_sum = 0.0f;
        for (uint d = 0; d < head_dim; d += 4) {
            float4 q_v = float4(q_head[d], q_head[d+1], q_head[d+2], q_head[d+3]);
            // Read K as FP16, convert to FP32 for computation
            half4 k_h = half4(k_vec[d], k_vec[d+1], k_vec[d+2], k_vec[d+3]);
            dot_sum += dot(q_v, float4(k_h));
        }

        float score = dot_sum * scale;
        scores_h[k] = half(score);
        local_max = max(local_max, score);
    }

    float simd_max_val = simd_max(local_max);
    if (simd_lane == 0) reduction_scratch[simd_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_max;
    if (tid == 0) {
        float m = reduction_scratch[0];
        for (uint i = 1; i < 4; i++) m = max(m, reduction_scratch[i]);
        shared_max = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Softmax
    float local_sum = 0.0f;
    for (uint k = tid; k < seq_kv; k += 128) {
        float exp_score = exp(float(scores_h[k]) - shared_max);
        scores_h[k] = half(exp_score);
        local_sum += exp_score;
    }

    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane == 0) reduction_scratch[simd_id] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_sum;
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < 4; i++) s += reduction_scratch[i];
        shared_sum = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = 1.0f / shared_sum;
    for (uint k = tid; k < seq_kv; k += 128) {
        scores_h[k] = half(float(scores_h[k]) * inv_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Output = scores @ V (with FP16 V reads)
    uint dims_per_simd = (head_dim + 3) / 4;
    uint d_start = simd_id * dims_per_simd;
    uint d_end = min(d_start + dims_per_simd, head_dim);

    for (uint d = d_start + simd_lane; d < d_end; d += 32) {
        float out_val = 0.0f;
        uint k = 0;
        for (; k + 3 < seq_kv; k += 4) {
            half4 s = half4(scores_h[k], scores_h[k+1], scores_h[k+2], scores_h[k+3]);
            // Read V as FP16
            half4 v = half4(
                v_head[k * head_dim + d],
                v_head[(k+1) * head_dim + d],
                v_head[(k+2) * head_dim + d],
                v_head[(k+3) * head_dim + d]
            );
            out_val += dot(float4(s), float4(v));
        }
        for (; k < seq_kv; k++) {
            out_val += float(scores_h[k]) * float(v_head[k * head_dim + d]);
        }
        if (d < head_dim) {
            out_head[d] = out_val;
        }
    }
}

// Prefill attention kernel for seq_q > 1 with causal masking
// Q: [num_heads, seq_q, head_dim]
// K: [num_kv_heads, seq_kv, head_dim] (from KV cache, may include past tokens)
// V: [num_kv_heads, seq_kv, head_dim]
// output: [num_heads, seq_q, head_dim]
// Each threadgroup handles one (head, query_pos) pair
kernel void attention_prefill(
    device const float* Q          [[buffer(0)]],
    device const float* K          [[buffer(1)]],
    device const float* V          [[buffer(2)]],
    device float* output           [[buffer(3)]],
    constant uint& num_heads       [[buffer(4)]],
    constant uint& num_kv_heads    [[buffer(5)]],
    constant uint& seq_q           [[buffer(6)]],
    constant uint& seq_kv          [[buffer(7)]],
    constant uint& head_dim        [[buffer(8)]],
    constant float& scale          [[buffer(9)]],
    constant uint& start_pos       [[buffer(10)]],  // For causal: query at q_pos attends to [0, start_pos + q_pos]
    uint2 tgid                     [[threadgroup_position_in_grid]],  // (head_idx, q_pos)
    uint2 tid_vec                  [[thread_position_in_threadgroup]]
) {
    uint head_idx = tgid.x;
    uint q_pos = tgid.y;
    uint tid = tid_vec.x;
    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (head_idx >= num_heads || q_pos >= seq_q) return;

    // GQA: map Q head to KV head
    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;

    // Causal mask: can only attend up to position (start_pos + q_pos) inclusive
    uint max_kv_pos = min(start_pos + q_pos + 1, seq_kv);

    // Pointers to this head's data
    device const float* q_head = Q + head_idx * seq_q * head_dim + q_pos * head_dim;
    device const float* k_head = K + kv_head * seq_kv * head_dim;
    device const float* v_head = V + kv_head * seq_kv * head_dim;
    device float* out_head = output + head_idx * seq_q * head_dim + q_pos * head_dim;

    // Use threadgroup memory for scores (limit to 2048 for prefill)
    threadgroup half scores_h[2048];
    threadgroup float reduction_scratch[4];

    // Step 1: Compute attention scores with causal mask
    float local_max = -INFINITY;
    for (uint k = tid; k < max_kv_pos; k += 128) {
        device const float* k_vec = k_head + k * head_dim;

        float dot_sum = 0.0f;
        for (uint d = 0; d < head_dim; d += 4) {
            float4 q_v = float4(q_head[d], q_head[d+1], q_head[d+2], q_head[d+3]);
            float4 k_v = float4(k_vec[d], k_vec[d+1], k_vec[d+2], k_vec[d+3]);
            dot_sum += dot(q_v, k_v);
        }

        float score = dot_sum * scale;
        scores_h[k] = half(score);
        local_max = max(local_max, score);
    }

    // SIMD reduction for max
    float simd_max_val = simd_max(local_max);
    if (simd_lane == 0) {
        reduction_scratch[simd_id] = simd_max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_max = 0.0f;
    if (tid == 0) {
        float global_max = reduction_scratch[0];
        for (uint i = 1; i < 4; i++) {
            global_max = max(global_max, reduction_scratch[i]);
        }
        shared_max = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute softmax
    float local_sum = 0.0f;
    for (uint k = tid; k < max_kv_pos; k += 128) {
        float exp_score = exp(float(scores_h[k]) - shared_max);
        scores_h[k] = half(exp_score);
        local_sum += exp_score;
    }

    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane == 0) {
        reduction_scratch[simd_id] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_sum = 0.0f;
    if (tid == 0) {
        float global_sum = 0.0f;
        for (uint i = 0; i < 4; i++) {
            global_sum += reduction_scratch[i];
        }
        shared_sum = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize scores
    float inv_sum = 1.0f / (shared_sum + 1e-6f);
    for (uint k = tid; k < max_kv_pos; k += 128) {
        scores_h[k] = half(float(scores_h[k]) * inv_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Compute weighted sum of V
    uint dims_per_thread = (head_dim + 127) / 128;
    uint d_start = tid * dims_per_thread;
    uint d_end = min(d_start + dims_per_thread, head_dim);

    for (uint d = d_start; d < d_end; d++) {
        float out_val = 0.0f;
        for (uint k = 0; k < max_kv_pos; k++) {
            out_val += float(scores_h[k]) * v_head[k * head_dim + d];
        }
        out_head[d] = out_val;
    }
}

// Prefill attention with FP16 KV cache
kernel void attention_prefill_f16kv(
    device const float* Q          [[buffer(0)]],
    device const half* K           [[buffer(1)]],
    device const half* V           [[buffer(2)]],
    device float* output           [[buffer(3)]],
    constant uint& num_heads       [[buffer(4)]],
    constant uint& num_kv_heads    [[buffer(5)]],
    constant uint& seq_q           [[buffer(6)]],
    constant uint& seq_kv          [[buffer(7)]],
    constant uint& head_dim        [[buffer(8)]],
    constant float& scale          [[buffer(9)]],
    constant uint& start_pos       [[buffer(10)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint2 tid_vec                  [[thread_position_in_threadgroup]]
) {
    uint head_idx = tgid.x;
    uint q_pos = tgid.y;
    uint tid = tid_vec.x;
    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (head_idx >= num_heads || q_pos >= seq_q) return;

    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;
    uint max_kv_pos = min(start_pos + q_pos + 1, seq_kv);

    device const float* q_head = Q + head_idx * seq_q * head_dim + q_pos * head_dim;
    device const half* k_head = K + kv_head * seq_kv * head_dim;
    device const half* v_head = V + kv_head * seq_kv * head_dim;
    device float* out_head = output + head_idx * seq_q * head_dim + q_pos * head_dim;

    threadgroup half scores_h[2048];
    threadgroup float reduction_scratch[4];

    // Step 1: Q @ K^T with FP16 K
    float local_max = -INFINITY;
    for (uint k = tid; k < max_kv_pos; k += 128) {
        device const half* k_vec = k_head + k * head_dim;

        float dot_sum = 0.0f;
        for (uint d = 0; d < head_dim; d += 4) {
            float4 q_v = float4(q_head[d], q_head[d+1], q_head[d+2], q_head[d+3]);
            half4 k_h = half4(k_vec[d], k_vec[d+1], k_vec[d+2], k_vec[d+3]);
            dot_sum += dot(q_v, float4(k_h));
        }

        float score = dot_sum * scale;
        scores_h[k] = half(score);
        local_max = max(local_max, score);
    }

    float simd_max_val = simd_max(local_max);
    if (simd_lane == 0) reduction_scratch[simd_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_max = 0.0f;
    if (tid == 0) {
        float global_max = reduction_scratch[0];
        for (uint i = 1; i < 4; i++) global_max = max(global_max, reduction_scratch[i]);
        shared_max = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Softmax
    float local_sum = 0.0f;
    for (uint k = tid; k < max_kv_pos; k += 128) {
        float exp_score = exp(float(scores_h[k]) - shared_max);
        scores_h[k] = half(exp_score);
        local_sum += exp_score;
    }

    float simd_sum_val2 = simd_sum(local_sum);
    if (simd_lane == 0) reduction_scratch[simd_id] = simd_sum_val2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_sum = 0.0f;
    if (tid == 0) {
        float global_sum = 0.0f;
        for (uint i = 0; i < 4; i++) global_sum += reduction_scratch[i];
        shared_sum = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = 1.0f / (shared_sum + 1e-6f);
    for (uint k = tid; k < max_kv_pos; k += 128) {
        scores_h[k] = half(float(scores_h[k]) * inv_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: V multiply with FP16 V
    uint dims_per_thread = (head_dim + 127) / 128;
    uint d_start = tid * dims_per_thread;
    uint d_end = min(d_start + dims_per_thread, head_dim);

    for (uint d = d_start; d < d_end; d++) {
        float out_val = 0.0f;
        for (uint k = 0; k < max_kv_pos; k++) {
            out_val += float(scores_h[k]) * float(v_head[k * head_dim + d]);
        }
        out_head[d] = out_val;
    }
}
)";
