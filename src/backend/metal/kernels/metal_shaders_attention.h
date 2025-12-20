// =============================================================================
// Metal Shader Attention Kernels
// =============================================================================
// Attention kernels: decode attention, KV cache operations, flash attention,
// tree attention, paged attention, and fused RMSNorm+MatVec variants.
// =============================================================================

#pragma once

static const char* METAL_SHADER_ATTENTION = R"(
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

// KV cache append kernel with layout transpose
// Copies new K/V to the cache at the current position
// new_kv: [new_len, num_heads, head_dim] (sequence-major from matmul output)
// cache: [num_heads, max_seq, head_dim] (head-major for attention kernels)
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

    // Source layout: [new_len, num_heads, head_dim] (sequence-major from matmul output)
    // Dest layout:   [num_heads, max_seq, head_dim] (head-major for attention kernels)
    uint src_idx = s * num_heads * head_dim + h * head_dim + d;
    uint dst_idx = h * max_seq * head_dim + (current_len + s) * head_dim + d;

    cache[dst_idx] = new_kv[src_idx];
}

// KV cache append with float->half conversion and layout transpose
// new_kv: [new_len, num_heads, head_dim] float (sequence-major from matmul)
// cache: [num_heads, max_seq, head_dim] half (head-major for attention)
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

    // Source layout: [new_len, num_heads, head_dim] (sequence-major from matmul output)
    // Dest layout:   [num_heads, max_seq, head_dim] (head-major for attention kernels)
    uint src_idx = s * num_heads * head_dim + h * head_dim + d;
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

// Half-precision fused SiLU + multiply for bandwidth-efficient prefill
kernel void silu_mul_half(
    device const half* a           [[buffer(0)]],   // gate input
    device const half* b           [[buffer(1)]],   // up input
    device half* c                 [[buffer(2)]],   // output
    constant uint& size            [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float val = float(a[gid]);
    float silu_val = val / (1.0f + exp(-val));
    c[gid] = half(silu_val * float(b[gid]));
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
// Fused RMSNorm + Dual Q4_K MatVec (Gate + Up projections in one kernel)
// =============================================================================
// Optimization: Computes RMSNorm once, then performs two matrix-vector products
// for gate and up projections simultaneously. Eliminates redundant RMSNorm.
//
// Current FFN path (2 kernel calls, RMSNorm computed twice):
//   rms_norm_matvec_q4k(x, norm, Wg, gate)  // RMSNorm + gate projection
//   rms_norm_matvec_q4k(x, norm, Wu, up)    // RMSNorm + up projection (redundant!)
//
// Fused path (1 kernel call, RMSNorm computed once):
//   rms_norm_dual_matvec_q4k(x, norm, Wg, Wu, gate, up)
//
constant constexpr uint DUAL_ROWS_PER_TG = 4;  // 4 SIMD groups × 2 outputs = 8 output rows

kernel void rms_norm_dual_matvec_q4k(
    device const float* x          [[buffer(0)]],   // Input: [K] float
    device const half* norm_weight [[buffer(1)]],   // RMSNorm weight: [K] half
    device const void* W_gate      [[buffer(2)]],   // Gate Q4_K weights: [N, K/QK_K] blocks
    device const void* W_up        [[buffer(3)]],   // Up Q4_K weights: [N, K/QK_K] blocks
    device float* y_gate           [[buffer(4)]],   // Gate output: [N] float
    device float* y_up             [[buffer(5)]],   // Up output: [N] float
    constant uint& K               [[buffer(6)]],   // Input/hidden dimension
    constant uint& N               [[buffer(7)]],   // Output dimension (intermediate_dim)
    constant float& eps            [[buffer(8)]],   // RMSNorm epsilon
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory for normalized input (max 4096 floats = 16KB)
    threadgroup float x_norm[4096];
    threadgroup float reduction_scratch[8];

    // =========================================================================
    // Step 1: Compute RMSNorm cooperatively across all threads (ONCE!)
    // =========================================================================

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
    // Step 2: Dual Q4_K MatVec - each SIMD group computes one gate and one up row
    // =========================================================================

    uint row = tgid * DUAL_ROWS_PER_TG + simd_id;
    if (row >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights_gate = (const device block_q4_K*)W_gate;
    const device block_q4_K* weights_up = (const device block_q4_K*)W_up;

    float sum_gate = 0.0f;
    float sum_up = 0.0f;

    // Process all K blocks - same normalized input for both gate and up
    for (uint kb = 0; kb < num_blocks_k; kb++) {
        // Load from normalized input (shared by both projections)
        uint elem_offset = simd_lane * 8;
        uint x_base = kb * QK_K + elem_offset;
        uint sub_block = elem_offset / 64;
        uint sub_offset = elem_offset % 64;

        // Read normalized input once
        float4 x_vec1 = float4(x_norm[x_base], x_norm[x_base + 1],
                               x_norm[x_base + 2], x_norm[x_base + 3]);
        float4 x_vec2 = float4(x_norm[x_base + 4], x_norm[x_base + 5],
                               x_norm[x_base + 6], x_norm[x_base + 7]);

        // === Gate projection ===
        {
            const device block_q4_K* block = &weights_gate[row * num_blocks_k + kb];
            float d = float(block->d);
            float dmin = float(block->dmin);
            const device uint8_t* scales = block->scales;
            const device uint8_t* qs = block->qs;

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
            sum_gate += lane_sum;
        }

        // === Up projection (same normalized input) ===
        {
            const device block_q4_K* block = &weights_up[row * num_blocks_k + kb];
            float d = float(block->d);
            float dmin = float(block->dmin);
            const device uint8_t* scales = block->scales;
            const device uint8_t* qs = block->qs;

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
            sum_up += lane_sum;
        }
    }

    // SIMD reduction for both outputs
    float gate_result = simd_sum(sum_gate);
    float up_result = simd_sum(sum_up);

    if (simd_lane == 0) {
        y_gate[row] = gate_result;
        y_up[row] = up_result;
    }
}

// =============================================================================
// Fused MatVec + Residual Add (Down projection + residual in one kernel)
// =============================================================================
// Optimization: Combines down projection with residual addition
// Current: down_proj(x, W) -> y; elementwise_add(residual, y) -> out
// Fused: matvec_residual(x, W, residual) -> out (= residual + x @ W)

kernel void matvec_residual_q4k(
    device const float* x          [[buffer(0)]],   // Input: [K] float
    device const void* W           [[buffer(1)]],   // Q4_K weights: [N, K/QK_K] blocks
    device const float* residual   [[buffer(2)]],   // Residual: [N] float
    device float* y                [[buffer(3)]],   // Output: [N] float (= residual + x @ W)
    constant uint& K               [[buffer(4)]],   // Input dimension
    constant uint& N               [[buffer(5)]],   // Output dimension
    uint tgid                      [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    const uint row = tgid * FUSED_ROWS_PER_TG + simd_id;
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

        float4 x_vec1 = float4(x[x_base], x[x_base + 1], x[x_base + 2], x[x_base + 3]);
        float4 x_vec2 = float4(x[x_base + 4], x[x_base + 5], x[x_base + 6], x[x_base + 7]);

        float lane_sum = 0.0f;
        if (sub_offset < 32) {
            uint qs_idx = sub_offset;
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

    // SIMD reduction and add residual
    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row] = residual[row] + sum;  // Fused residual add!
    }
}

// =============================================================================
// Fused MatVec + Residual Add for Q3_K (Down projection + residual)
// =============================================================================
kernel void matvec_residual_q3k(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device const float* residual   [[buffer(2)]],
    device float* y                [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK_K;

    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short ip  = tid / 4;
    const short il  = 2 * ((tid % 4) / 2);
    const short ir  = tid % 2;
    const short l0  = 8 * ir;

    const ushort4 mm[4] = {
        ushort4(0x0001, 0x0100, 0x0002, 0x0200),
        ushort4(0x0004, 0x0400, 0x0008, 0x0800),
        ushort4(0x0010, 0x1000, 0x0020, 0x2000),
        ushort4(0x0040, 0x4000, 0x0080, 0x8000)
    };

    const int4 qm[2] = {
        int4(0x0003, 0x0300, 0x000c, 0x0c00),
        int4(0x0030, 0x3000, 0x00c0, 0xc000)
    };

    const ushort4 hm = mm[2 * ip + il / 2];
    const short shift = 2 * il;
    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    const ushort s_shift1 = 4 * ip;
    const ushort s_shift2 = s_shift1 + il;

    const short q_offset = 32 * ip + l0;
    const short y_offset = 128 * ip + 32 * il + l0;

    const uint first_row = (tgid * Q3K_ROWS_PER_TG + sgitg) * NR0_Q3K;

    device const block_q3_K* weights = (const device block_q3_K*)W;

    float yl[32];
    float sumf1[NR0_Q3K] = {0.f, 0.f};
    float sumf2[NR0_Q3K] = {0.f, 0.f};

    device const float* y1_base = x + ix * QK_K + y_offset;

    for (uint i = ix; i < nb; i += 4) {
        device const float* y1 = y1_base + (i / 4) * 4 * QK_K;
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0];
            yl[l + 8] = y1[l + 16];
            yl[l + 16] = y1[l + 32];
            yl[l + 24] = y1[l + 48];
        }

        for (short row = 0; row < NR0_Q3K; ++row) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_q3_K* block = &weights[row_idx * nb + i];
            device const ushort* q = (device const ushort*)(block->qs + q_offset);
            device const ushort* h = (device const ushort*)(block->hmask + l0);
            device const ushort* a = (device const ushort*)(block->scales);

            const float d_all = float(block->d);

            uint32_t scales32, aux32;
            thread ushort* scales16 = (thread ushort*)&scales32;
            thread const int8_t* scales = (thread const int8_t*)&scales32;

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il + 0];
            scales16[1] = a[il + 1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l / 2];
                s1 += yl[l + 0] * (qs & qm[il / 2][0]);
                s2 += yl[l + 1] * (qs & qm[il / 2][1]);
                s3 += ((h[l / 2] & hm[0]) ? 0.f : yl[l + 0]) + ((h[l / 2] & hm[1]) ? 0.f : yl[l + 1]);
                s4 += yl[l + 16] * (qs & qm[il / 2][2]);
                s5 += yl[l + 17] * (qs & qm[il / 2][3]);
                s6 += ((h[l / 2] & hm[2]) ? 0.f : yl[l + 16]) + ((h[l / 2] & hm[3]) ? 0.f : yl[l + 17]);
            }
            float d1 = d_all * (s1 + 1.f / 256.f * s2 - s3 * v1);
            float d2 = d_all * (s4 + 1.f / 256.f * s5 - s6 * v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l / 2 + 8];
                s1 += yl[l + 8] * (qs & qm[il / 2][0]);
                s2 += yl[l + 9] * (qs & qm[il / 2][1]);
                s3 += ((h[l / 2 + 8] & hm[0]) ? 0.f : yl[l + 8]) + ((h[l / 2 + 8] & hm[1]) ? 0.f : yl[l + 9]);
                s4 += yl[l + 24] * (qs & qm[il / 2][2]);
                s5 += yl[l + 25] * (qs & qm[il / 2][3]);
                s6 += ((h[l / 2 + 8] & hm[2]) ? 0.f : yl[l + 24]) + ((h[l / 2 + 8] & hm[3]) ? 0.f : yl[l + 25]);
            }
            d1 = d_all * (s1 + 1.f / 256.f * s2 - s3 * v1);
            d2 = d_all * (s4 + 1.f / 256.f * s5 - s6 * v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);
        }
    }

    // SIMD reduction with fused residual add
    for (short row = 0; row < NR0_Q3K; ++row) {
        uint row_idx = first_row + row;
        if (row_idx < N) {
            float sum = simd_sum(sumf1[row]) + simd_sum(sumf2[row]);
            if (tiisg == 0) {
                y[row_idx] = residual[row_idx] + sum;  // Fused residual!
            }
        }
    }
}

// =============================================================================
// Fused MatVec + Residual Add for Q2_K (Down projection + residual)
// =============================================================================
kernel void matvec_residual_q2k(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device const float* residual   [[buffer(2)]],
    device float* y                [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK_K;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    const uint first_row = (tgid * Q2K_ROWS_PER_TG + sgitg) * NR0_Q2K;

    device const block_q2_K* weights = (const device block_q2_K*)W;

    float yl[32];
    float sumf[NR0_Q2K] = {0.f, 0.f};

    device const float* y4 = x + ix * QK_K + 128 * iq + 8 * ir;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        for (short row = 0; row < NR0_Q2K; row++) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_q2_K* block = &weights[row_idx * nb + ib];
            device const uchar* sc = block->scales + 8 * iq + is;
            device const ushort* qs = (device const ushort*)block->qs + 16 * iq + 4 * ir;
            device const half* dh = &block->d;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }

            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;

            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                 sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
        }

        y4 += 4 * QK_K;
    }

    // SIMD reduction with fused residual add
    for (short row = 0; row < NR0_Q2K; ++row) {
        uint row_idx = first_row + row;
        if (row_idx < N) {
            float sum = simd_sum(sumf[row]);
            if (tiisg == 0) {
                y[row_idx] = residual[row_idx] + sum;  // Fused residual!
            }
        }
    }
}

// =============================================================================
// Fused RMSNorm + Dual Q3_K MatVec (Gate + Up projections in one kernel)
// =============================================================================
// Eliminates redundant RMSNorm: compute once, use for both gate and up
kernel void rms_norm_dual_matvec_q3k(
    device const float* x          [[buffer(0)]],
    device const half* norm_weight [[buffer(1)]],
    device const void* W_gate      [[buffer(2)]],
    device const void* W_up        [[buffer(3)]],
    device float* y_gate           [[buffer(4)]],
    device float* y_up             [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    constant uint& N               [[buffer(7)]],
    constant float& eps            [[buffer(8)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float x_norm[4096];
    threadgroup float reduction_scratch[8];

    // Step 1: RMSNorm (computed ONCE for both projections)
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
        for (uint i = 0; i < 8; i++) {
            total += reduction_scratch[i];
        }
        rms_inv = rsqrt(total / float(K) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < K; i += 256) {
        x_norm[i] = x[i] * rms_inv * float(norm_weight[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Dual Q3_K MatVec
    uint row = tgid * 8 + simd_id;
    if (row >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q3_K* weights_gate = (const device block_q3_K*)W_gate;
    const device block_q3_K* weights_up = (const device block_q3_K*)W_up;

    float sum_gate = 0.0f;
    float sum_up = 0.0f;

    for (uint kb = simd_lane; kb < num_blocks_k; kb += 32) {
        uint base_idx = kb * QK_K;

        // Gate projection
        {
            const device block_q3_K* block = &weights_gate[row * num_blocks_k + kb];
            float d = float(block->d);
            float acc = 0.0f;

            for (uint i = 0; i < 256; i++) {
                uint qs_byte = i / 4;
                uint qs_shift = (i % 4) * 2;
                uint low2 = (block->qs[qs_byte] >> qs_shift) & 0x3;

                uint hm_byte = i / 8;
                uint hm_bit = i % 8;
                uint high1 = (block->hmask[hm_byte] >> hm_bit) & 0x1;

                int q3 = int(low2 | (high1 << 2)) - 4;

                uint sub_block = i / 32;
                int8_t scale;
                if (sub_block < 4) {
                    uint8_t sc_low = block->scales[sub_block] & 0x0F;
                    uint8_t sc_high = (block->scales[8 + sub_block / 2] >> (4 * (sub_block % 2))) & 0x03;
                    scale = int8_t((sc_low | (sc_high << 4))) - 32;
                } else {
                    uint8_t sc_low = (block->scales[sub_block - 4] >> 4) & 0x0F;
                    uint8_t sc_high = (block->scales[10 + (sub_block - 4) / 2] >> (4 * ((sub_block - 4) % 2))) & 0x03;
                    scale = int8_t((sc_low | (sc_high << 4))) - 32;
                }

                acc += x_norm[base_idx + i] * (d * float(scale) * float(q3));
            }
            sum_gate += acc;
        }

        // Up projection (same normalized input)
        {
            const device block_q3_K* block = &weights_up[row * num_blocks_k + kb];
            float d = float(block->d);
            float acc = 0.0f;

            for (uint i = 0; i < 256; i++) {
                uint qs_byte = i / 4;
                uint qs_shift = (i % 4) * 2;
                uint low2 = (block->qs[qs_byte] >> qs_shift) & 0x3;

                uint hm_byte = i / 8;
                uint hm_bit = i % 8;
                uint high1 = (block->hmask[hm_byte] >> hm_bit) & 0x1;

                int q3 = int(low2 | (high1 << 2)) - 4;

                uint sub_block = i / 32;
                int8_t scale;
                if (sub_block < 4) {
                    uint8_t sc_low = block->scales[sub_block] & 0x0F;
                    uint8_t sc_high = (block->scales[8 + sub_block / 2] >> (4 * (sub_block % 2))) & 0x03;
                    scale = int8_t((sc_low | (sc_high << 4))) - 32;
                } else {
                    uint8_t sc_low = (block->scales[sub_block - 4] >> 4) & 0x0F;
                    uint8_t sc_high = (block->scales[10 + (sub_block - 4) / 2] >> (4 * ((sub_block - 4) % 2))) & 0x03;
                    scale = int8_t((sc_low | (sc_high << 4))) - 32;
                }

                acc += x_norm[base_idx + i] * (d * float(scale) * float(q3));
            }
            sum_up += acc;
        }
    }

    float gate_result = simd_sum(sum_gate);
    float up_result = simd_sum(sum_up);

    if (simd_lane == 0) {
        y_gate[row] = gate_result;
        y_up[row] = up_result;
    }
}

// =============================================================================
// Fused RMSNorm + Dual Q2_K MatVec (Gate + Up projections in one kernel)
// =============================================================================
// Eliminates redundant RMSNorm: compute once, use for both gate and up
kernel void rms_norm_dual_matvec_q2k(
    device const float* x          [[buffer(0)]],
    device const half* norm_weight [[buffer(1)]],
    device const void* W_gate      [[buffer(2)]],
    device const void* W_up        [[buffer(3)]],
    device float* y_gate           [[buffer(4)]],
    device float* y_up             [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    constant uint& N               [[buffer(7)]],
    constant float& eps            [[buffer(8)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float x_norm[4096];
    threadgroup float reduction_scratch[8];

    // Step 1: RMSNorm (computed ONCE for both projections)
    float local_ss = 0.0f;
    for (uint i = tid; i < K; i += 256) {
        float val = x[i];
        local_ss += val * val;
    }

    float simd_ss = simd_sum(local_ss);
    if (simd_lane == 0) {
        reduction_scratch[simd_id] = simd_ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total_ss = 0.0f;
        for (uint i = 0; i < 8; i++) {
            total_ss += reduction_scratch[i];
        }
        reduction_scratch[0] = rsqrt(total_ss / float(K) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = reduction_scratch[0];

    for (uint i = tid; i < K; i += 256) {
        x_norm[i] = x[i] * scale * float(norm_weight[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Dual Q2_K MatVec
    uint row = tgid * 8 + simd_id;
    if (row >= N) return;

    const uint nb = K / QK_K;
    const device block_q2_K* weights_gate = (const device block_q2_K*)W_gate;
    const device block_q2_K* weights_up = (const device block_q2_K*)W_up;

    float sum_gate = 0.0f;
    float sum_up = 0.0f;

    for (uint kb = simd_lane; kb < nb; kb += 32) {
        uint base_idx = kb * QK_K;

        // Gate projection
        {
            const device block_q2_K* block = &weights_gate[row * nb + kb];
            float d = float(block->d);
            float dmin = float(block->dmin);
            float acc = 0.0f;

            for (uint j = 0; j < 16; j++) {
                uint8_t sc_byte = block->scales[j];
                float scale_val = d * float(sc_byte & 0xF);
                float min_val = dmin * float(sc_byte >> 4);

                for (uint i = 0; i < 16; i++) {
                    uint elem = j * 16 + i;
                    uint qs_idx = elem / 4;
                    uint qs_shift = (elem % 4) * 2;
                    int q2 = int((block->qs[qs_idx] >> qs_shift) & 0x3);

                    acc += x_norm[base_idx + elem] * (scale_val * float(q2) - min_val);
                }
            }
            sum_gate += acc;
        }

        // Up projection (same normalized input)
        {
            const device block_q2_K* block = &weights_up[row * nb + kb];
            float d = float(block->d);
            float dmin = float(block->dmin);
            float acc = 0.0f;

            for (uint j = 0; j < 16; j++) {
                uint8_t sc_byte = block->scales[j];
                float scale_val = d * float(sc_byte & 0xF);
                float min_val = dmin * float(sc_byte >> 4);

                for (uint i = 0; i < 16; i++) {
                    uint elem = j * 16 + i;
                    uint qs_idx = elem / 4;
                    uint qs_shift = (elem % 4) * 2;
                    int q2 = int((block->qs[qs_idx] >> qs_shift) & 0x3);

                    acc += x_norm[base_idx + elem] * (scale_val * float(q2) - min_val);
                }
            }
            sum_up += acc;
        }
    }

    float gate_result = simd_sum(sum_gate);
    float up_result = simd_sum(sum_up);

    if (simd_lane == 0) {
        y_gate[row] = gate_result;
        y_up[row] = up_result;
    }
}

// =============================================================================
// MULTI-HEAD ATTENTION DECODE (FP16 KV Cache) - Fallback
// =============================================================================
//
// Simple attention kernel used as fallback when simdgroup kernels unavailable.
// For optimal performance, use simdgroup_flash_attention_decode_f16kv_d64/d128.
//
// =============================================================================
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

// =============================================================================
// Flash Attention Prefill - Simdgroup Matrix Operations (llama.cpp style)
// =============================================================================
// Uses simdgroup 8x8 matrix multiply for efficient Q*K^T computation.
// Processes Q_TILE query positions and K_TILE KV positions per threadgroup.
//
// Key insight from llama.cpp:
// - Use simdgroup_multiply_accumulate for 8x8 blocked Q*K^T
// - Each simdgroup handles Q_TILE/NSG query positions
// - Loop over K in blocks of K_TILE, with each simdgroup handling K_TILE/NSG
//
// Grid: (num_heads, ceil(seq_q / Q_TILE), 1)
// Threadgroup: 128 threads (4 simdgroups)
//
// Template: DK=64 (head_dim), Q_TILE=8, K_TILE=32, NSG=4

constant constexpr uint FA_DK = 64;      // head dimension (compile-time for TinyLlama)
constant constexpr uint FA_Q_TILE = 8;   // queries per threadgroup
constant constexpr uint FA_K_TILE = 32;  // KV positions per iteration
constant constexpr uint FA_NSG = 4;      // simdgroups per threadgroup
constant constexpr uint FA_NW = 32;      // SIMD width

kernel void flash_attention_prefill(
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
    constant uint& max_seq         [[buffer(11)]],  // KV cache stride (max sequence length)
    threadgroup half* shmem        [[threadgroup(0)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    const uint head_idx = tgid.x;
    const uint q_block = tgid.y;
    const uint q_base = q_block * FA_Q_TILE;

    if (head_idx >= num_heads) return;

    const uint heads_per_kv = num_heads / num_kv_heads;
    const uint kv_head = head_idx / heads_per_kv;

    // Reduced threadgroup memory - no V buffer needed!
    // sq[Q_TILE * DK] - Q data (half)
    // so[Q_TILE * DK] - output accumulator (half)
    // ss[Q_TILE * K_TILE] - attention scores (half for simdgroup ops)
    threadgroup half* sq = shmem;                                    // 8 * 64 = 512 halfs
    threadgroup half* so = shmem + FA_Q_TILE * FA_DK;                // 8 * 64 = 512 halfs
    threadgroup half* ss = shmem + 2 * FA_Q_TILE * FA_DK;            // 8 * 32 = 256 halfs

    // Pointers to global memory
    // Q layout: [seq_q, num_heads, head_dim] (sequence-major from matmul output)
    // K/V layout: [num_kv_heads, max_seq, head_dim] (head-major in cache, use max_seq for stride!)
    // output layout: [seq_q, num_heads, head_dim] (sequence-major)
    device const half* k_ptr = K + kv_head * max_seq * head_dim;
    device const half* v_ptr = V + kv_head * max_seq * head_dim;

    // Load Q into shared memory (convert float to half)
    // Q is in [seq_q, num_heads, head_dim] layout
    #pragma clang loop unroll(full)
    for (uint i = tid; i < FA_Q_TILE * FA_DK; i += FA_NSG * FA_NW) {
        uint q_row = i / FA_DK;
        uint q_col = i % FA_DK;
        uint global_q_pos = q_base + q_row;

        // Index into [seq_q, num_heads, head_dim]: global_q_pos * (num_heads * head_dim) + head_idx * head_dim + q_col
        uint q_idx = global_q_pos * num_heads * head_dim + head_idx * head_dim + q_col;
        sq[q_row * FA_DK + q_col] = (global_q_pos < seq_q && q_col < head_dim) ?
            half(Q[q_idx]) : half(0.0f);
    }

    // Zero output accumulator
    #pragma clang loop unroll(full)
    for (uint i = tid; i < FA_Q_TILE * FA_DK; i += FA_NSG * FA_NW) {
        so[i] = half(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-query running statistics for online softmax
    constexpr uint NQ = FA_Q_TILE / FA_NSG;  // 2 queries per simdgroup
    float M[NQ];  // running max
    float S[NQ];  // running sum
    #pragma clang loop unroll(full)
    for (uint i = 0; i < NQ; i++) {
        M[i] = -INFINITY;
        S[i] = 0.0f;
    }

    // Loop over K/V positions in blocks of K_TILE
    for (uint k_base = 0; k_base < seq_kv; k_base += FA_K_TILE) {
        // Compute Q * K^T using simdgroup matrix operations
        device const half* pk = k_ptr + (k_base + simd_id * 8) * head_dim;

        // Compute 8x8 Q*K^T block using simdgroup matrix multiply
        simdgroup_half8x8 mqk = simdgroup_half8x8(0);

        // Iterate over head_dim in chunks of 8 - UNROLLED
        #pragma clang loop unroll(full)
        for (uint d = 0; d < FA_DK; d += 8) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;

            simdgroup_load(mq, sq + d, FA_DK);
            simdgroup_load(mk, pk + d, head_dim, 0, true);  // transpose

            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }

        // Store Q*K^T scores to shared memory (half)
        simdgroup_store(mqk, ss + simd_id * 8, FA_K_TILE, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax for each query - compute in float, store as half
        #pragma clang loop unroll(full)
        for (uint jj = 0; jj < NQ; jj++) {
            uint j = jj * FA_NSG + simd_id;
            uint global_q_pos = q_base + j;

            if (global_q_pos >= seq_q) continue;

            // Causal mask
            uint max_kv = min(start_pos + global_q_pos + 1, seq_kv);

            // Find max for this query (read half, compute float)
            float m_prev = M[jj];
            float m_new = m_prev;

            for (uint k = simd_lane; k < FA_K_TILE; k += FA_NW) {
                uint global_k_pos = k_base + k;
                float score = (global_k_pos < max_kv) ?
                    float(ss[j * FA_K_TILE + k]) * scale : -INFINITY;
                m_new = max(m_new, score);
            }
            m_new = simd_max(m_new);

            // Rescale previous accumulator
            float rescale = exp(m_prev - m_new);
            S[jj] *= rescale;

            // Compute exp(score - m_new) and store DIRECTLY as half
            float s_local = 0.0f;
            for (uint k = simd_lane; k < FA_K_TILE; k += FA_NW) {
                uint global_k_pos = k_base + k;
                float score = (global_k_pos < max_kv) ?
                    float(ss[j * FA_K_TILE + k]) * scale : -INFINITY;
                float exp_score = exp(score - m_new);
                ss[j * FA_K_TILE + k] = half(exp_score);  // Store directly as half!
                s_local += exp_score;
            }
            s_local = simd_sum(s_local);
            S[jj] += s_local;
            M[jj] = m_new;

            // Rescale output
            for (uint d = simd_lane; d < FA_DK; d += FA_NW) {
                so[j * FA_DK + d] = half(float(so[j * FA_DK + d]) * rescale);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S * V using simdgroup ops - load V DIRECTLY from global memory!
        // No extra conversion needed - scores already stored as half
        {
            const uint d_start = simd_id * 16;

            // Load current output blocks
            simdgroup_half8x8 mo0, mo1;
            simdgroup_load(mo0, so + d_start, FA_DK);
            simdgroup_load(mo1, so + d_start + 8, FA_DK);

            // Load V directly from global and accumulate
            device const half* pv = v_ptr + k_base * head_dim;

            #pragma clang loop unroll(full)
            for (uint kb = 0; kb < FA_K_TILE / 8; kb++) {
                simdgroup_half8x8 ms;
                simdgroup_load(ms, ss + kb * 8, FA_K_TILE);

                // Load V blocks directly from global memory
                simdgroup_half8x8 mv0, mv1;
                device const half* pv_row = pv + kb * 8 * head_dim + d_start;
                simdgroup_load(mv0, pv_row, head_dim);
                simdgroup_load(mv1, pv_row + 8, head_dim);

                simdgroup_multiply_accumulate(mo0, ms, mv0, mo0);
                simdgroup_multiply_accumulate(mo1, ms, mv1, mo1);
            }

            simdgroup_store(mo0, so + d_start, FA_DK);
            simdgroup_store(mo1, so + d_start + 8, FA_DK);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write to global memory
    // Output layout: [seq_q, num_heads, head_dim] (sequence-major)
    #pragma clang loop unroll(full)
    for (uint jj = 0; jj < NQ; jj++) {
        uint j = jj * FA_NSG + simd_id;
        uint global_q_pos = q_base + j;

        if (global_q_pos >= seq_q) continue;

        float inv_sum = 1.0f / (S[jj] + 1e-6f);

        for (uint d = simd_lane; d < FA_DK; d += FA_NW) {
            // Index into [seq_q, num_heads, head_dim]: global_q_pos * (num_heads * head_dim) + head_idx * head_dim + d
            uint out_idx = global_q_pos * num_heads * head_dim + head_idx * head_dim + d;
            output[out_idx] = float(so[j * FA_DK + d]) * inv_sum;
        }
    }
}

// =============================================================================
// Tree Attention Kernel - For Speculative Decoding with Tree-Based Verification
// =============================================================================
//
// Tree attention allows each node in a speculation tree to attend only to its
// ancestors, enabling efficient parallel verification of multiple speculation paths.
//
// Key difference from causal attention:
// - Causal: position i attends to positions [0, i]
// - Tree: node i attends to its ancestors + KV cache positions
//
// The parent_indices buffer encodes the tree structure:
// - parent_indices[i] = parent node of node i (-1 for root)
// - Ancestor mask is computed by walking up the parent chain

kernel void attention_tree_f16kv(
    device const float* Q              [[buffer(0)]],   // [num_heads, num_nodes, head_dim]
    device const half* K_cache         [[buffer(1)]],   // [num_kv_heads, cache_len, head_dim]
    device const half* V_cache         [[buffer(2)]],   // [num_kv_heads, cache_len, head_dim]
    device const half* K_tree          [[buffer(3)]],   // [num_kv_heads, num_nodes, head_dim]
    device const half* V_tree          [[buffer(4)]],   // [num_kv_heads, num_nodes, head_dim]
    device const int* parent_indices   [[buffer(5)]],   // [num_nodes] parent index for each node
    device float* output               [[buffer(6)]],   // [num_heads, num_nodes, head_dim]
    constant uint& num_heads           [[buffer(7)]],
    constant uint& num_kv_heads        [[buffer(8)]],
    constant uint& num_nodes           [[buffer(9)]],   // Number of tree nodes
    constant uint& cache_len           [[buffer(10)]],  // Length of KV cache (past context)
    constant uint& head_dim            [[buffer(11)]],
    constant float& scale              [[buffer(12)]],
    uint2 tgid                         [[threadgroup_position_in_grid]],  // (head_idx, node_idx)
    uint2 tid_vec                      [[thread_position_in_threadgroup]]
) {
    uint head_idx = tgid.x;
    uint node_idx = tgid.y;
    uint tid = tid_vec.x;
    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (head_idx >= num_heads || node_idx >= num_nodes) return;

    // GQA: map Q head to KV head
    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;

    // Pointers to this head's data
    device const float* q_head = Q + head_idx * num_nodes * head_dim + node_idx * head_dim;
    device const half* k_cache_head = K_cache + kv_head * cache_len * head_dim;
    device const half* v_cache_head = V_cache + kv_head * cache_len * head_dim;
    device const half* k_tree_head = K_tree + kv_head * num_nodes * head_dim;
    device const half* v_tree_head = V_tree + kv_head * num_nodes * head_dim;
    device float* out_head = output + head_idx * num_nodes * head_dim + node_idx * head_dim;

    // Build ancestor mask for this node
    // ancestor_mask[i] = true if tree node i is an ancestor of current node
    threadgroup bool ancestor_mask[64];  // Max 64 tree nodes
    threadgroup half scores_cache[2048]; // For cache positions
    threadgroup half scores_tree[64];    // For tree positions
    threadgroup float reduction_scratch[4];

    // Initialize ancestor mask (cooperatively)
    if (tid < num_nodes) {
        ancestor_mask[tid] = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 walks up parent chain to find ancestors
    if (tid == 0) {
        // Current node always attends to itself
        ancestor_mask[node_idx] = true;

        // Walk up parent chain
        int current = parent_indices[node_idx];
        while (current >= 0 && current < (int)num_nodes) {
            ancestor_mask[current] = true;
            current = parent_indices[current];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Total attention length = cache_len + num ancestors in tree
    uint total_kv_len = cache_len + num_nodes;

    // Step 1: Compute attention scores
    // First: scores for KV cache positions (attend to all of them)
    float local_max = -INFINITY;

    for (uint k = tid; k < cache_len; k += 128) {
        device const half* k_vec = k_cache_head + k * head_dim;

        float dot_sum = 0.0f;
        for (uint d = 0; d < head_dim; d += 4) {
            float4 q_v = float4(q_head[d], q_head[d+1], q_head[d+2], q_head[d+3]);
            half4 k_h = half4(k_vec[d], k_vec[d+1], k_vec[d+2], k_vec[d+3]);
            dot_sum += dot(q_v, float4(k_h));
        }

        float score = dot_sum * scale;
        scores_cache[k] = half(score);
        local_max = max(local_max, score);
    }

    // Then: scores for tree positions (only ancestors)
    for (uint k = tid; k < num_nodes; k += 128) {
        if (ancestor_mask[k]) {
            device const half* k_vec = k_tree_head + k * head_dim;

            float dot_sum = 0.0f;
            for (uint d = 0; d < head_dim; d += 4) {
                float4 q_v = float4(q_head[d], q_head[d+1], q_head[d+2], q_head[d+3]);
                half4 k_h = half4(k_vec[d], k_vec[d+1], k_vec[d+2], k_vec[d+3]);
                dot_sum += dot(q_v, float4(k_h));
            }

            float score = dot_sum * scale;
            scores_tree[k] = half(score);
            local_max = max(local_max, score);
        } else {
            scores_tree[k] = half(-INFINITY);  // Masked out
        }
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

    // Step 2: Softmax
    float local_sum = 0.0f;

    for (uint k = tid; k < cache_len; k += 128) {
        float exp_score = exp(float(scores_cache[k]) - shared_max);
        scores_cache[k] = half(exp_score);
        local_sum += exp_score;
    }

    for (uint k = tid; k < num_nodes; k += 128) {
        if (ancestor_mask[k]) {
            float exp_score = exp(float(scores_tree[k]) - shared_max);
            scores_tree[k] = half(exp_score);
            local_sum += exp_score;
        }
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

    // Normalize
    float inv_sum = 1.0f / (shared_sum + 1e-6f);
    for (uint k = tid; k < cache_len; k += 128) {
        scores_cache[k] = half(float(scores_cache[k]) * inv_sum);
    }
    for (uint k = tid; k < num_nodes; k += 128) {
        scores_tree[k] = half(float(scores_tree[k]) * inv_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Compute weighted sum of V
    uint dims_per_thread = (head_dim + 127) / 128;
    uint d_start = tid * dims_per_thread;
    uint d_end = min(d_start + dims_per_thread, head_dim);

    for (uint d = d_start; d < d_end; d++) {
        float out_val = 0.0f;

        // Contribution from cache
        for (uint k = 0; k < cache_len; k++) {
            out_val += float(scores_cache[k]) * float(v_cache_head[k * head_dim + d]);
        }

        // Contribution from tree (only ancestors)
        for (uint k = 0; k < num_nodes; k++) {
            if (ancestor_mask[k]) {
                out_val += float(scores_tree[k]) * float(v_tree_head[k * head_dim + d]);
            }
        }

        out_head[d] = out_val;
    }
}

// Simpler tree attention kernel when there's no prior KV cache
// (Fresh generation without context)
kernel void attention_tree_nocontext_f16kv(
    device const float* Q              [[buffer(0)]],   // [num_heads, num_nodes, head_dim]
    device const half* K_tree          [[buffer(1)]],   // [num_kv_heads, num_nodes, head_dim]
    device const half* V_tree          [[buffer(2)]],   // [num_kv_heads, num_nodes, head_dim]
    device const int* parent_indices   [[buffer(3)]],   // [num_nodes] parent index for each node
    device float* output               [[buffer(4)]],   // [num_heads, num_nodes, head_dim]
    constant uint& num_heads           [[buffer(5)]],
    constant uint& num_kv_heads        [[buffer(6)]],
    constant uint& num_nodes           [[buffer(7)]],
    constant uint& head_dim            [[buffer(8)]],
    constant float& scale              [[buffer(9)]],
    uint2 tgid                         [[threadgroup_position_in_grid]],
    uint2 tid_vec                      [[thread_position_in_threadgroup]]
) {
    uint head_idx = tgid.x;
    uint node_idx = tgid.y;
    uint tid = tid_vec.x;
    uint simd_lane = tid % 32;
    uint simd_id = tid / 32;

    if (head_idx >= num_heads || node_idx >= num_nodes) return;

    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;

    device const float* q_head = Q + head_idx * num_nodes * head_dim + node_idx * head_dim;
    device const half* k_tree_head = K_tree + kv_head * num_nodes * head_dim;
    device const half* v_tree_head = V_tree + kv_head * num_nodes * head_dim;
    device float* out_head = output + head_idx * num_nodes * head_dim + node_idx * head_dim;

    threadgroup bool ancestor_mask[64];
    threadgroup half scores[64];
    threadgroup float reduction_scratch[4];

    // Initialize mask
    if (tid < num_nodes) {
        ancestor_mask[tid] = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Build ancestor mask
    if (tid == 0) {
        ancestor_mask[node_idx] = true;
        int current = parent_indices[node_idx];
        while (current >= 0 && current < (int)num_nodes) {
            ancestor_mask[current] = true;
            current = parent_indices[current];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute scores
    float local_max = -INFINITY;
    for (uint k = tid; k < num_nodes; k += 128) {
        if (ancestor_mask[k]) {
            device const half* k_vec = k_tree_head + k * head_dim;
            float dot_sum = 0.0f;
            for (uint d = 0; d < head_dim; d += 4) {
                float4 q_v = float4(q_head[d], q_head[d+1], q_head[d+2], q_head[d+3]);
                half4 k_h = half4(k_vec[d], k_vec[d+1], k_vec[d+2], k_vec[d+3]);
                dot_sum += dot(q_v, float4(k_h));
            }
            float score = dot_sum * scale;
            scores[k] = half(score);
            local_max = max(local_max, score);
        } else {
            scores[k] = half(-INFINITY);
        }
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

    // Softmax
    float local_sum = 0.0f;
    for (uint k = tid; k < num_nodes; k += 128) {
        if (ancestor_mask[k]) {
            float exp_score = exp(float(scores[k]) - shared_max);
            scores[k] = half(exp_score);
            local_sum += exp_score;
        }
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

    float inv_sum = 1.0f / (shared_sum + 1e-6f);
    for (uint k = tid; k < num_nodes; k += 128) {
        scores[k] = half(float(scores[k]) * inv_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Output
    uint dims_per_thread = (head_dim + 127) / 128;
    uint d_start = tid * dims_per_thread;
    uint d_end = min(d_start + dims_per_thread, head_dim);

    for (uint d = d_start; d < d_end; d++) {
        float out_val = 0.0f;
        for (uint k = 0; k < num_nodes; k++) {
            if (ancestor_mask[k]) {
                out_val += float(scores[k]) * float(v_tree_head[k * head_dim + d]);
            }
        }
        out_head[d] = out_val;
    }
}

// =============================================================================
// Paged Attention Kernels
// =============================================================================
// These kernels support PagedAttention where K/V are stored in scattered physical
// blocks rather than contiguous memory. A block table maps logical positions to
// physical block indices.
//
// Memory layout:
//   K_cache/V_cache: [num_blocks * block_size, num_kv_heads, head_dim] (half)
//   block_table: [num_logical_blocks] - maps logical block -> physical block
//
// For a token at position `pos`:
//   logical_block = pos / block_size
//   block_offset = pos % block_size
//   physical_block = block_table[logical_block]
//   physical_pos = physical_block * block_size + block_offset

// Paged attention decode kernel for single query (seq_q = 1)
// Q: [num_heads, head_dim]
// K_cache: [num_blocks * block_size, num_kv_heads, head_dim] (half)
// V_cache: [num_blocks * block_size, num_kv_heads, head_dim] (half)
// block_table: [num_logical_blocks]
// output: [num_heads, head_dim]
// Each threadgroup handles one head with 128 threads (4 SIMD groups)
kernel void paged_attention_decode(
    device const float* Q              [[buffer(0)]],
    device const half* K_cache         [[buffer(1)]],
    device const half* V_cache         [[buffer(2)]],
    device const int* block_table      [[buffer(3)]],
    device float* output               [[buffer(4)]],
    constant uint& num_heads           [[buffer(5)]],
    constant uint& num_kv_heads        [[buffer(6)]],
    constant uint& seq_len             [[buffer(7)]],
    constant uint& head_dim            [[buffer(8)]],
    constant uint& block_size          [[buffer(9)]],
    constant uint& num_kv_heads_stride [[buffer(10)]],  // = num_blocks * block_size
    constant float& scale              [[buffer(11)]],
    uint head_idx                      [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (head_idx >= num_heads) return;

    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;

    device const float* q_head = Q + head_idx * head_dim;
    device float* out_head = output + head_idx * head_dim;

    // K/V layout: [physical_pos, num_kv_heads, head_dim]
    // To access K[kv_head, physical_pos, d]: K[physical_pos * num_kv_heads * head_dim + kv_head * head_dim + d]

    threadgroup half scores_h[4096];
    threadgroup float reduction_scratch[4];

    // Step 1: Q @ K^T with scattered K reads
    float local_max = -INFINITY;
    for (uint k = tid; k < seq_len; k += 128) {
        // Map logical position to physical position
        uint logical_block = k / block_size;
        uint block_offset = k % block_size;
        int physical_block = block_table[logical_block];
        uint physical_pos = physical_block * block_size + block_offset;

        // K at [physical_pos, kv_head, :]
        device const half* k_vec = K_cache + physical_pos * num_kv_heads * head_dim + kv_head * head_dim;

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

    threadgroup float shared_max;
    if (tid == 0) {
        float m = reduction_scratch[0];
        for (uint i = 1; i < 4; i++) m = max(m, reduction_scratch[i]);
        shared_max = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Softmax
    float local_sum = 0.0f;
    for (uint k = tid; k < seq_len; k += 128) {
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

    float inv_sum = 1.0f / (shared_sum + 1e-6f);
    for (uint k = tid; k < seq_len; k += 128) {
        scores_h[k] = half(float(scores_h[k]) * inv_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Output = scores @ V (with scattered V reads)
    uint dims_per_simd = (head_dim + 3) / 4;
    uint d_start = simd_id * dims_per_simd;
    uint d_end = min(d_start + dims_per_simd, head_dim);

    for (uint d = d_start + simd_lane; d < d_end; d += 32) {
        float out_val = 0.0f;
        for (uint k = 0; k < seq_len; k++) {
            // Map logical position to physical position
            uint logical_block = k / block_size;
            uint block_offset = k % block_size;
            int physical_block = block_table[logical_block];
            uint physical_pos = physical_block * block_size + block_offset;

            // V at [physical_pos, kv_head, d]
            half v_val = V_cache[physical_pos * num_kv_heads * head_dim + kv_head * head_dim + d];
            out_val += float(scores_h[k]) * float(v_val);
        }
        if (d < head_dim) {
            out_head[d] = out_val;
        }
    }
}

// Paged attention append kernel - copies new K/V to paged cache
// new_k/new_v: [num_kv_heads, new_len, head_dim] (float)
// K_cache/V_cache: [num_blocks * block_size, num_kv_heads, head_dim] (half)
// block_table: [num_logical_blocks]
// Appends new K/V starting at position `start_pos`
kernel void paged_kv_cache_append(
    device const float* new_k          [[buffer(0)]],
    device const float* new_v          [[buffer(1)]],
    device half* K_cache               [[buffer(2)]],
    device half* V_cache               [[buffer(3)]],
    device const int* block_table      [[buffer(4)]],
    constant uint& num_kv_heads        [[buffer(5)]],
    constant uint& head_dim            [[buffer(6)]],
    constant uint& start_pos           [[buffer(7)]],
    constant uint& new_len             [[buffer(8)]],
    constant uint& block_size          [[buffer(9)]],
    uint3 gid                          [[thread_position_in_grid]]
) {
    uint h = gid.z;   // KV head index
    uint s = gid.y;   // Position in new sequence (0 to new_len-1)
    uint d = gid.x;   // Dimension

    if (h >= num_kv_heads || s >= new_len || d >= head_dim) return;

    // Source index: new_k/v is [num_kv_heads, new_len, head_dim]
    uint src_idx = h * new_len * head_dim + s * head_dim + d;

    // Destination: map (start_pos + s) to physical position
    uint logical_pos = start_pos + s;
    uint logical_block = logical_pos / block_size;
    uint block_offset = logical_pos % block_size;
    int physical_block = block_table[logical_block];
    uint physical_pos = physical_block * block_size + block_offset;

    // Dest index: K/V_cache is [physical_pos, num_kv_heads, head_dim]
    uint dst_idx = physical_pos * num_kv_heads * head_dim + h * head_dim + d;

    K_cache[dst_idx] = half(new_k[src_idx]);
    V_cache[dst_idx] = half(new_v[src_idx]);
}

// Batched paged attention decode - multiple sequences in one kernel launch
// Handles multiple independent sequences, each with its own block table
// Q: [batch_size, num_heads, head_dim]
// K_cache/V_cache: [num_blocks * block_size, num_kv_heads, head_dim] (half) - shared pool
// block_tables: [batch_size, max_blocks_per_seq] - each row is a sequence's block table
// seq_lens: [batch_size] - actual sequence length for each batch entry
// output: [batch_size, num_heads, head_dim]
// Each threadgroup handles one (batch, head) pair
kernel void batched_paged_attention_decode(
    device const float* Q              [[buffer(0)]],
    device const half* K_cache         [[buffer(1)]],
    device const half* V_cache         [[buffer(2)]],
    device const int* block_tables     [[buffer(3)]],
    device const int* seq_lens         [[buffer(4)]],
    device float* output               [[buffer(5)]],
    constant uint& batch_size          [[buffer(6)]],
    constant uint& num_heads           [[buffer(7)]],
    constant uint& num_kv_heads        [[buffer(8)]],
    constant uint& head_dim            [[buffer(9)]],
    constant uint& block_size          [[buffer(10)]],
    constant uint& max_blocks_per_seq  [[buffer(11)]],
    constant float& scale              [[buffer(12)]],
    uint2 tgid                         [[threadgroup_position_in_grid]],
    uint2 tid_2d                       [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    uint tid = tid_2d.x;  // Use x component for 1D indexing
    uint batch_idx = tgid.y;
    uint head_idx = tgid.x;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;
    uint seq_len = seq_lens[batch_idx];

    if (seq_len == 0) return;

    device const float* q_head = Q + batch_idx * num_heads * head_dim + head_idx * head_dim;
    device float* out_head = output + batch_idx * num_heads * head_dim + head_idx * head_dim;
    device const int* block_table = block_tables + batch_idx * max_blocks_per_seq;

    threadgroup half scores_h[4096];
    threadgroup float reduction_scratch[4];

    // Step 1: Q @ K^T
    float local_max = -INFINITY;
    for (uint k = tid; k < seq_len; k += 128) {
        uint logical_block = k / block_size;
        uint block_offset = k % block_size;
        int physical_block = block_table[logical_block];
        uint physical_pos = physical_block * block_size + block_offset;

        device const half* k_vec = K_cache + physical_pos * num_kv_heads * head_dim + kv_head * head_dim;

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

    threadgroup float shared_max;
    if (tid == 0) {
        float m = reduction_scratch[0];
        for (uint i = 1; i < 4; i++) m = max(m, reduction_scratch[i]);
        shared_max = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Softmax
    float local_sum = 0.0f;
    for (uint k = tid; k < seq_len; k += 128) {
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

    float inv_sum = 1.0f / (shared_sum + 1e-6f);
    for (uint k = tid; k < seq_len; k += 128) {
        scores_h[k] = half(float(scores_h[k]) * inv_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Output = scores @ V
    uint dims_per_simd = (head_dim + 3) / 4;
    uint d_start = simd_id * dims_per_simd;
    uint d_end = min(d_start + dims_per_simd, head_dim);

    for (uint d = d_start + simd_lane; d < d_end; d += 32) {
        float out_val = 0.0f;
        for (uint k = 0; k < seq_len; k++) {
            uint logical_block = k / block_size;
            uint block_offset = k % block_size;
            int physical_block = block_table[logical_block];
            uint physical_pos = physical_block * block_size + block_offset;

            half v_val = V_cache[physical_pos * num_kv_heads * head_dim + kv_head * head_dim + d];
            out_val += float(scores_h[k]) * float(v_val);
        }
        if (d < head_dim) {
            out_head[d] = out_val;
        }
    }
}

// =============================================================================
// SIMDGROUP FLASH ATTENTION - Vectorized Tiled Implementation
// =============================================================================
//
// This kernel uses Metal simdgroup_matrix operations for hardware-accelerated
// 8x8 matrix multiplies, following the approach from llama.cpp.
//
// Key differences from legacy attention:
// - Uses simdgroup_matrix<half, 8, 8> for Q@K^T and S@V
// - Tiled KV processing with online softmax
// - Better memory access patterns via simdgroup loads
//
// Configuration:
// - 128 threads per threadgroup (4 simdgroups x 32 threads)
// - KV tile size: 32 positions (fits well in shared memory)
// - One threadgroup per attention head
//
// =============================================================================

// Tile size for KV positions processed per iteration
constant constexpr uint FLASH_KV_TILE = 32;

// =============================================================================
// Vectorized Flash Attention Decode - head_dim=64 specialization
// Tiled approach with vectorized loads using float4/half4
// =============================================================================
kernel void simdgroup_flash_attention_decode_f16kv_d64(
    device const float* Q          [[buffer(0)]],   // [num_heads, 64] FP32
    device const half* K           [[buffer(1)]],   // [num_kv_heads, seq_kv, 64] FP16
    device const half* V           [[buffer(2)]],   // [num_kv_heads, seq_kv, 64] FP16
    device float* output           [[buffer(3)]],   // [num_heads, 64] FP32
    constant uint& num_heads       [[buffer(4)]],
    constant uint& num_kv_heads    [[buffer(5)]],
    constant uint& seq_kv          [[buffer(6)]],
    constant uint& head_dim        [[buffer(7)]],   // Must be 64
    constant float& scale          [[buffer(8)]],
    uint head_idx                  [[threadgroup_position_in_grid]],
    ushort tiisg                   [[thread_index_in_simdgroup]],
    ushort sgitg                   [[simdgroup_index_in_threadgroup]]
) {
    if (head_idx >= num_heads) return;

    // Constants for head_dim=64
    constexpr uint HD = 64;
    constexpr uint HD4 = HD / 4;  // 16 float4s
    constexpr uint TILE = FLASH_KV_TILE;  // 32 positions per tile
    constexpr uint NSG = 4;  // Number of simdgroups
    constexpr uint NW = 32;  // Threads per simdgroup

    // GQA: map query head to KV head
    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;

    // Pointers (vectorized)
    device const float4* q4 = (device const float4*)(Q + head_idx * HD);
    device const half4* k4_base = (device const half4*)(K + kv_head * seq_kv * HD);
    device const half4* v4_base = (device const half4*)(V + kv_head * seq_kv * HD);
    device float* out_head = output + head_idx * HD;

    // Shared memory
    threadgroup float4 sq4[HD4];      // Q vector
    threadgroup half4 skv4[TILE * HD4]; // K/V tile buffer
    threadgroup float ss[TILE];       // Attention scores
    threadgroup float4 so4[HD4];      // Output accumulator
    threadgroup float sg_scratch[NSG]; // For cross-simdgroup reductions

    // Load Q into shared memory (vectorized)
    uint tid = tiisg + sgitg * NW;
    if (tid < HD4) {
        sq4[tid] = q4[tid];
    }

    // Initialize output accumulator
    if (tid < HD4) {
        so4[tid] = float4(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float running_max = -FLT_MAX / 2;
    float running_sum = 0.0f;

    // Process KV cache in tiles
    uint num_tiles = (seq_kv + TILE - 1) / TILE;

    for (uint tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        uint tile_start = tile_idx * TILE;
        uint tile_len = min(TILE, seq_kv - tile_start);

        // =====================================================================
        // Step 1: Load K tile into shared memory (vectorized, coalesced)
        // =====================================================================
        device const half4* k4_tile = k4_base + tile_start * HD4;

        // 128 threads, tile_len * HD4 = 32 * 16 = 512 half4s to load
        // Each thread loads 4 half4s
        for (uint i = tid; i < tile_len * HD4; i += 128) {
            skv4[i] = k4_tile[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // Step 2: Compute Q @ K^T for all K positions in tile
        // =====================================================================
        // Each simdgroup handles a subset of K positions
        uint k_per_sg = (tile_len + NSG - 1) / NSG;
        uint k_start_sg = sgitg * k_per_sg;
        uint k_end_sg = min(k_start_sg + k_per_sg, tile_len);

        for (uint k = k_start_sg; k < k_end_sg; k++) {
            threadgroup half4* k4_vec = skv4 + k * HD4;

            // Vectorized dot product: threads 0-15 each handle one float4
            float partial_dot = 0.0f;
            if (tiisg < HD4) {
                float4 q_vec = sq4[tiisg];
                float4 k_vec = float4(k4_vec[tiisg]);
                partial_dot = dot(q_vec, k_vec);
            }

            // Sum across SIMD group
            float dot_sum = simd_sum(partial_dot);

            // Thread 0 writes the score
            if (tiisg == 0) {
                ss[k] = dot_sum * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // Step 3: Online softmax update
        // =====================================================================
        // Find max in this tile
        float local_max = -FLT_MAX / 2;
        for (uint k = tiisg; k < tile_len; k += NW) {
            local_max = max(local_max, ss[k]);
        }
        float tile_max = simd_max(local_max);

        // Cross-simdgroup max reduction
        if (tiisg == 0) sg_scratch[sgitg] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0 && tiisg == 0) {
            float m = sg_scratch[0];
            for (uint i = 1; i < NSG; i++) m = max(m, sg_scratch[i]);
            sg_scratch[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max = sg_scratch[0];

        // Update running max and correction factor
        float new_max = max(running_max, tile_max);
        float correction = (running_max > -FLT_MAX / 4) ? exp(running_max - new_max) : 0.0f;

        // Scale previous output accumulator
        if (tid < HD4) {
            so4[tid] *= correction;
        }

        // Compute exp(score - new_max) and sum
        float local_sum = 0.0f;
        for (uint k = tiisg; k < tile_len; k += NW) {
            float exp_score = exp(ss[k] - new_max);
            ss[k] = exp_score;
            local_sum += exp_score;
        }
        float tile_sum = simd_sum(local_sum);

        // Cross-simdgroup sum reduction
        if (tiisg == 0) sg_scratch[sgitg] = tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0 && tiisg == 0) {
            float s = 0.0f;
            for (uint i = 0; i < NSG; i++) s += sg_scratch[i];
            sg_scratch[0] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_sum = sg_scratch[0];

        running_sum = running_sum * correction + tile_sum;
        running_max = new_max;

        // =====================================================================
        // Step 4: Load V tile and compute S @ V
        // =====================================================================
        device const half4* v4_tile = v4_base + tile_start * HD4;
        for (uint i = tid; i < tile_len * HD4; i += 128) {
            skv4[i] = v4_tile[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate: output += attention_weights @ V
        if (tid < HD4) {
            float4 accum = float4(0.0f);
            for (uint k = 0; k < tile_len; k++) {
                float weight = ss[k];
                float4 v_vec = float4(skv4[k * HD4 + tid]);
                accum += weight * v_vec;
            }
            so4[tid] += accum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =====================================================================
    // Step 5: Normalize and write output
    // =====================================================================
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    if (tid < HD4) {
        float4 result = so4[tid] * inv_sum;
        out_head[tid * 4 + 0] = result.x;
        out_head[tid * 4 + 1] = result.y;
        out_head[tid * 4 + 2] = result.z;
        out_head[tid * 4 + 3] = result.w;
    }
}

// =============================================================================
// Vectorized Flash Attention Decode - head_dim=128 specialization
// Tiled approach with vectorized loads using float4/half4
// =============================================================================
kernel void simdgroup_flash_attention_decode_f16kv_d128(
    device const float* Q          [[buffer(0)]],   // [num_heads, 128] FP32
    device const half* K           [[buffer(1)]],   // [num_kv_heads, seq_kv, 128] FP16
    device const half* V           [[buffer(2)]],   // [num_kv_heads, seq_kv, 128] FP16
    device float* output           [[buffer(3)]],   // [num_heads, 128] FP32
    constant uint& num_heads       [[buffer(4)]],
    constant uint& num_kv_heads    [[buffer(5)]],
    constant uint& seq_kv          [[buffer(6)]],
    constant uint& head_dim        [[buffer(7)]],   // Must be 128
    constant float& scale          [[buffer(8)]],
    uint head_idx                  [[threadgroup_position_in_grid]],
    ushort tiisg                   [[thread_index_in_simdgroup]],
    ushort sgitg                   [[simdgroup_index_in_threadgroup]]
) {
    if (head_idx >= num_heads) return;

    // Constants for head_dim=128
    constexpr uint HD = 128;
    constexpr uint HD4 = HD / 4;  // 32 float4s
    constexpr uint TILE = FLASH_KV_TILE;  // 32 positions per tile
    constexpr uint NSG = 4;
    constexpr uint NW = 32;

    // GQA: map query head to KV head
    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;

    // Pointers (vectorized)
    device const float4* q4 = (device const float4*)(Q + head_idx * HD);
    device const half4* k4_base = (device const half4*)(K + kv_head * seq_kv * HD);
    device const half4* v4_base = (device const half4*)(V + kv_head * seq_kv * HD);
    device float* out_head = output + head_idx * HD;

    // Shared memory
    threadgroup float4 sq4[HD4];      // Q vector
    threadgroup half4 skv4[TILE * HD4]; // K/V tile buffer
    threadgroup float ss[TILE];       // Attention scores
    threadgroup float4 so4[HD4];      // Output accumulator
    threadgroup float sg_scratch[NSG]; // For cross-simdgroup reductions

    // Load Q into shared memory (vectorized)
    uint tid = tiisg + sgitg * NW;
    if (tid < HD4) {
        sq4[tid] = q4[tid];
    }

    // Initialize output accumulator
    if (tid < HD4) {
        so4[tid] = float4(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float running_max = -FLT_MAX / 2;
    float running_sum = 0.0f;

    // Process KV cache in tiles
    uint num_tiles = (seq_kv + TILE - 1) / TILE;

    for (uint tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        uint tile_start = tile_idx * TILE;
        uint tile_len = min(TILE, seq_kv - tile_start);

        // =====================================================================
        // Step 1: Load K tile into shared memory (vectorized, coalesced)
        // =====================================================================
        device const half4* k4_tile = k4_base + tile_start * HD4;

        // 128 threads, tile_len * HD4 = 32 * 32 = 1024 half4s to load
        for (uint i = tid; i < tile_len * HD4; i += 128) {
            skv4[i] = k4_tile[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // Step 2: Compute Q @ K^T for all K positions in tile
        // =====================================================================
        uint k_per_sg = (tile_len + NSG - 1) / NSG;
        uint k_start_sg = sgitg * k_per_sg;
        uint k_end_sg = min(k_start_sg + k_per_sg, tile_len);

        for (uint k = k_start_sg; k < k_end_sg; k++) {
            threadgroup half4* k4_vec = skv4 + k * HD4;

            // Vectorized dot product: 32 threads, 32 float4s -> 1 per thread
            float partial_dot = 0.0f;
            if (tiisg < HD4) {
                float4 q_vec = sq4[tiisg];
                float4 k_vec = float4(k4_vec[tiisg]);
                partial_dot = dot(q_vec, k_vec);
            }

            float dot_sum = simd_sum(partial_dot);

            if (tiisg == 0) {
                ss[k] = dot_sum * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // Step 3: Online softmax update
        // =====================================================================
        float local_max = -FLT_MAX / 2;
        for (uint k = tiisg; k < tile_len; k += NW) {
            local_max = max(local_max, ss[k]);
        }
        float tile_max = simd_max(local_max);

        if (tiisg == 0) sg_scratch[sgitg] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0 && tiisg == 0) {
            float m = sg_scratch[0];
            for (uint i = 1; i < NSG; i++) m = max(m, sg_scratch[i]);
            sg_scratch[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max = sg_scratch[0];

        float new_max = max(running_max, tile_max);
        float correction = (running_max > -FLT_MAX / 4) ? exp(running_max - new_max) : 0.0f;

        if (tid < HD4) {
            so4[tid] *= correction;
        }

        float local_sum = 0.0f;
        for (uint k = tiisg; k < tile_len; k += NW) {
            float exp_score = exp(ss[k] - new_max);
            ss[k] = exp_score;
            local_sum += exp_score;
        }
        float tile_sum = simd_sum(local_sum);

        if (tiisg == 0) sg_scratch[sgitg] = tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0 && tiisg == 0) {
            float s = 0.0f;
            for (uint i = 0; i < NSG; i++) s += sg_scratch[i];
            sg_scratch[0] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_sum = sg_scratch[0];

        running_sum = running_sum * correction + tile_sum;
        running_max = new_max;

        // =====================================================================
        // Step 4: Load V tile and compute S @ V
        // =====================================================================
        device const half4* v4_tile = v4_base + tile_start * HD4;
        for (uint i = tid; i < tile_len * HD4; i += 128) {
            skv4[i] = v4_tile[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < HD4) {
            float4 accum = float4(0.0f);
            for (uint k = 0; k < tile_len; k++) {
                float weight = ss[k];
                float4 v_vec = float4(skv4[k * HD4 + tid]);
                accum += weight * v_vec;
            }
            so4[tid] += accum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =====================================================================
    // Step 5: Normalize and write output
    // =====================================================================
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    if (tid < HD4) {
        float4 result = so4[tid] * inv_sum;
        out_head[tid * 4 + 0] = result.x;
        out_head[tid * 4 + 1] = result.y;
        out_head[tid * 4 + 2] = result.z;
        out_head[tid * 4 + 3] = result.w;
    }
}

// =============================================================================
// llama.cpp-style Flash Attention Decode Kernel
// =============================================================================
//
// This kernel is adapted from llama.cpp's kernel_flash_attn_ext_vec_impl.
// Key optimizations:
//   1. NE (elements per thread) - Each SIMD lane handles multiple K positions
//   2. Hierarchical SIMD reduction using simd_shuffle_down
//   3. Vectorized Q*K^T with coalesced memory access
//   4. Online softmax with running max/sum
//
// Memory layout assumptions:
//   - Q: [num_heads, head_dim] contiguous FP32
//   - K: [num_kv_heads, seq_kv, head_dim] contiguous FP16
//   - V: [num_kv_heads, seq_kv, head_dim] contiguous FP16
//
// Threading:
//   - 1 threadgroup per query head
//   - 4 simdgroups (128 threads) per threadgroup
//   - Each simdgroup processes different tiles of KV cache
//
// =============================================================================

// Tile size: number of K positions processed per iteration
constant constexpr uint FA_TILE_SIZE = 32;

// Helper: hierarchical SIMD reduction for dot products
// Reduces NE adjacent lanes using simd_shuffle_down
template<uint NE>
inline float simd_reduce_add(float val, ushort tiisg) {
    // Hierarchical reduction based on NE (elements per thread)
    if (NE <= 16) val += simd_shuffle_down(val, 16);
    if (NE <= 8)  val += simd_shuffle_down(val, 8);
    if (NE <= 4)  val += simd_shuffle_down(val, 4);
    if (NE <= 2)  val += simd_shuffle_down(val, 2);
    if (NE <= 1)  val += simd_shuffle_down(val, 1);
    return val;
}

// Helper: broadcast reduced value to all lanes in a group
template<uint NL>
inline float simd_broadcast(float val, ushort lane_group) {
    return simd_shuffle(val, lane_group * (32 / (32 / NL)));
}

// =============================================================================
// Flash Attention Decode - head_dim=64 with NE/NL parallelism (llama.cpp style)
// =============================================================================
//
// Key optimization: NE/NL parallelism
// - NL = 16 threads cooperate on each K dot product
// - NE = 2 K positions processed in parallel per simdgroup
// - This doubles throughput compared to sequential K processing
//
kernel void flash_attention_decode_d64(
    device const float* Q          [[buffer(0)]],   // [num_heads, 64] FP32
    device const half* K           [[buffer(1)]],   // [num_kv_heads, seq_kv, 64] FP16
    device const half* V           [[buffer(2)]],   // [num_kv_heads, seq_kv, 64] FP16
    device float* output           [[buffer(3)]],   // [num_heads, 64] FP32
    constant uint& num_heads       [[buffer(4)]],
    constant uint& num_kv_heads    [[buffer(5)]],
    constant uint& seq_kv          [[buffer(6)]],
    constant uint& head_dim        [[buffer(7)]],   // Must be 64
    constant float& scale          [[buffer(8)]],
    uint3 tgpig                    [[threadgroup_position_in_grid]],
    ushort tiisg                   [[thread_index_in_simdgroup]],
    ushort sgitg                   [[simdgroup_index_in_threadgroup]]
) {
    const uint head_idx = tgpig.x;
    if (head_idx >= num_heads) return;

    // Constants
    constexpr uint DK = 64;
    constexpr uint DK4 = DK / 4;     // 16 float4s per head
    constexpr uint C = FA_TILE_SIZE; // 32 K positions per tile
    constexpr uint NW = 32;          // Threads per simdgroup
    constexpr uint NSG = 4;          // Simdgroups per threadgroup

    // NE/NL parallelism: process NE K positions in parallel
    // For d64: 16 threads (NL) per K, 2 K positions (NE) in parallel
    constexpr uint NL = DK4;         // 16 threads per K dot product
    constexpr uint NE = NW / NL;     // 2 K positions in parallel

    // Thread role within simdgroup
    const ushort tx = tiisg % NL;    // 0-15: which float4 element
    const ushort ty = tiisg / NL;    // 0-1: which K position (of NE)

    // GQA: map query head to KV head
    const uint heads_per_kv = num_heads / num_kv_heads;
    const uint kv_head = head_idx / heads_per_kv;

    // Base pointers
    device const float4* q4 = (device const float4*)(Q + head_idx * DK);
    device const half4* k4_base = (device const half4*)(K + kv_head * seq_kv * DK);
    device const half4* v4_base = (device const half4*)(V + kv_head * seq_kv * DK);
    device float4* out4 = (device float4*)(output + head_idx * DK);

    // Shared memory
    threadgroup float4 sq4[DK4];         // Query vector (16 float4s)
    threadgroup float ss[NSG][C];        // Attention scores per simdgroup (was shared bug!)
    threadgroup float4 so4[NSG * DK4];   // Output accumulators per simdgroup
    threadgroup float sm_s[NSG];         // Running sum per simdgroup
    threadgroup float sm_m[NSG];         // Running max per simdgroup

    const uint tid = tiisg + sgitg * NW;

    // Load Q into shared memory
    if (tid < DK4) {
        sq4[tid] = q4[tid];
    }

    // Initialize per-simdgroup output accumulator
    threadgroup float4* my_so4 = so4 + sgitg * DK4;
    if (tiisg < DK4) {
        my_so4[tiisg] = float4(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float S = 0.0f;
    float M = -FLT_MAX / 2;

    // Process KV cache in tiles
    for (uint ic0 = sgitg; ic0 * C < seq_kv; ic0 += NSG) {
        const uint ic = ic0 * C;
        const uint tile_len = min(C, seq_kv - ic);

        // =====================================================================
        // Q * K^T with NE/NL parallelism
        // =====================================================================
        device const half4* pk4 = k4_base + ic * DK4;

        // Process C K positions, NE at a time
        for (uint kk = 0; kk < tile_len; kk += NE) {
            // Each thread handles one K position (ty) and one float4 element (tx)
            const uint k_idx = kk + ty;
            float score = 0.0f;

            if (k_idx < tile_len) {
                device const half4* k_row = pk4 + k_idx * DK4;

                // Each thread computes dot(Q[tx], K[k_idx][tx])
                float4 q_vec = sq4[tx];
                float4 k_vec = float4(k_row[tx]);
                float partial = dot(q_vec, k_vec);

                // Reduce across NL threads using simd_shuffle_down
                // This sums the 16 partial dot products
                partial += simd_shuffle_down(partial, 8);
                partial += simd_shuffle_down(partial, 4);
                partial += simd_shuffle_down(partial, 2);
                partial += simd_shuffle_down(partial, 1);

                score = partial * scale;
            }

            // Thread 0 and 16 (tx=0 for each ty) write their scores
            if (tx == 0 && k_idx < tile_len) {
                ss[sgitg][k_idx] = score;
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // Online softmax update
        // =====================================================================
        const float old_M = M;

        // Find max in this tile
        float local_max = -FLT_MAX / 2;
        for (uint k = tiisg; k < tile_len; k += NW) {
            local_max = max(local_max, ss[sgitg][k]);
        }
        float tile_max = simd_max(local_max);
        M = max(M, tile_max);

        const float ms = exp(old_M - M);

        // Compute exp and sum
        float local_sum = 0.0f;
        for (uint k = tiisg; k < tile_len; k += NW) {
            float vs = exp(ss[sgitg][k] - M);
            ss[sgitg][k] = vs;
            local_sum += vs;
        }
        float tile_sum = simd_sum(local_sum);
        S = S * ms + tile_sum;

        // Scale previous output
        if (tiisg < DK4) {
            my_so4[tiisg] *= ms;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // O += P * V with NE/NL parallelism
        // =====================================================================
        device const half4* pv4 = v4_base + ic * DK4;

        // Each thread accumulates for its assigned output element (tx)
        float4 accum = float4(0.0f);

        // Process C K positions, NE at a time
        for (uint kk = 0; kk < tile_len; kk += NE) {
            const uint k_idx = kk + ty;
            if (k_idx < tile_len) {
                float weight = ss[sgitg][k_idx];
                float4 v_vec = float4(pv4[k_idx * DK4 + tx]);
                accum += v_vec * weight;
            }
        }

        // Reduce across ty (NE dimension) - threads 0-15 and 16-31 need to combine
        // simd_shuffle_down by NL brings thread 16's value to thread 0, etc.
        accum += simd_shuffle_down(accum, NL);

        // Only ty=0 threads (0-15) have the complete sum
        if (ty == 0) {
            my_so4[tx] += accum;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =====================================================================
    // Cross-simdgroup reduction
    // =====================================================================
    if (tiisg == 0) {
        sm_s[sgitg] = S;
        sm_m[sgitg] = M;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        // Find global max
        float final_M = sm_m[0];
        for (uint sg = 1; sg < NSG; ++sg) {
            final_M = max(final_M, sm_m[sg]);
        }

        // Compute combined sum
        float final_S = 0.0f;
        for (uint sg = 0; sg < NSG; ++sg) {
            final_S += sm_s[sg] * exp(sm_m[sg] - final_M);
        }

        // Combine output vectors
        if (tiisg < DK4) {
            float4 final_o = float4(0.0f);
            for (uint sg = 0; sg < NSG; ++sg) {
                float scale_sg = exp(sm_m[sg] - final_M);
                final_o += so4[sg * DK4 + tiisg] * scale_sg;
            }

            float inv_S = (final_S > 0.0f) ? 1.0f / final_S : 0.0f;
            out4[tiisg] = final_o * inv_S;
        }
    }
}

// =============================================================================
// Flash Attention Decode - head_dim=128, llama.cpp style
// =============================================================================
kernel void flash_attention_decode_d128(
    device const float* Q          [[buffer(0)]],   // [num_heads, 128] FP32
    device const half* K           [[buffer(1)]],   // [num_kv_heads, seq_kv, 128] FP16
    device const half* V           [[buffer(2)]],   // [num_kv_heads, seq_kv, 128] FP16
    device float* output           [[buffer(3)]],   // [num_heads, 128] FP32
    constant uint& num_heads       [[buffer(4)]],
    constant uint& num_kv_heads    [[buffer(5)]],
    constant uint& seq_kv          [[buffer(6)]],
    constant uint& head_dim        [[buffer(7)]],   // Must be 128
    constant float& scale          [[buffer(8)]],
    uint3 tgpig                    [[threadgroup_position_in_grid]],
    ushort tiisg                   [[thread_index_in_simdgroup]],
    ushort sgitg                   [[simdgroup_index_in_threadgroup]]
) {
    const uint head_idx = tgpig.x;
    if (head_idx >= num_heads) return;

    // Constants for head_dim=128
    constexpr uint DK = 128;
    constexpr uint DK4 = DK / 4;     // 32 float4s per head
    constexpr uint C = FA_TILE_SIZE; // 32 K positions per tile
    constexpr uint NW = 32;          // Threads per simdgroup
    constexpr uint NSG = 4;          // Simdgroups per threadgroup

    // For DK=128: NL=32, NE=1 (all 32 threads work on each K)
    constexpr uint NE = 1;
    constexpr uint NL = NW / NE;     // 32 threads per K dot product

    // GQA: map query head to KV head
    const uint heads_per_kv = num_heads / num_kv_heads;
    const uint kv_head = head_idx / heads_per_kv;

    // Base pointers
    device const float4* q4 = (device const float4*)(Q + head_idx * DK);
    device const half4* k4_base = (device const half4*)(K + kv_head * seq_kv * DK);
    device const half4* v4_base = (device const half4*)(V + kv_head * seq_kv * DK);
    device float4* out4 = (device float4*)(output + head_idx * DK);

    // Shared memory
    threadgroup float4 sq4[DK4];
    threadgroup float ss[NSG][C];   // Per-simdgroup scores (was shared bug!)
    threadgroup float4 so4[NSG * DK4];
    threadgroup float sm_s[NSG];
    threadgroup float sm_m[NSG];

    const uint tid = tiisg + sgitg * NW;

    // Load Q into shared memory
    if (tid < DK4) {
        sq4[tid] = q4[tid];
    }
    // Need second load for DK4=32 with 128 threads
    if (tid >= DK4 && tid < DK4 * 2) {
        // Already loaded above
    }

    // Initialize per-simdgroup output accumulator
    threadgroup float4* my_so4 = so4 + sgitg * DK4;
    for (ushort i = tiisg; i < DK4; i += NW) {
        my_so4[i] = float4(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float S = 0.0f;
    float M = -FLT_MAX / 2;

    // Process KV cache in tiles
    for (uint ic0 = sgitg; ic0 * C < seq_kv; ic0 += NSG) {
        const uint ic = ic0 * C;
        const uint tile_len = min(C, seq_kv - ic);

        // =====================================================================
        // Q * K^T
        // =====================================================================
        device const half4* pk4 = k4_base + ic * DK4;

        // For NE=1, each iteration handles one K position
        // All 32 threads cooperate on the dot product
        for (uint k_idx = 0; k_idx < tile_len; ++k_idx) {
            device const half4* k_row = pk4 + k_idx * DK4;

            // Each thread handles DK4/NW = 1 float4
            float partial = 0.0f;
            #pragma unroll
            for (uint ii = 0; ii < DK4 / NW; ++ii) {
                const uint idx = ii * NW + tiisg;
                float4 q_vec = sq4[idx];
                float4 k_vec = float4(k_row[idx]);
                partial += dot(q_vec, k_vec);
            }

            // Full SIMD reduction
            float score = simd_sum(partial) * scale;

            if (tiisg == 0) {
                ss[sgitg][k_idx] = score;
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // Online softmax
        // =====================================================================
        {
            const float m = M;

            float local_max = -FLT_MAX / 2;
            for (uint k = tiisg; k < tile_len; k += NW) {
                local_max = max(local_max, ss[sgitg][k]);
            }
            float tile_max = simd_max(local_max);

            M = max(M, tile_max);
            const float ms = exp(m - M);

            float local_sum = 0.0f;
            for (uint k = tiisg; k < tile_len; k += NW) {
                float vs = exp(ss[sgitg][k] - M);
                ss[sgitg][k] = vs;
                local_sum += vs;
            }
            float tile_sum = simd_sum(local_sum);

            S = S * ms + tile_sum;

            for (ushort i = tiisg; i < DK4; i += NW) {
                my_so4[i] *= ms;
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // O += P * V
        // =====================================================================
        {
            device const half4* pv4 = v4_base + ic * DK4;

            for (uint k_idx = 0; k_idx < tile_len; ++k_idx) {
                const float weight = ss[sgitg][k_idx];
                device const half4* v_row = pv4 + k_idx * DK4;

                // Each thread handles DK4/NW float4s
                #pragma unroll
                for (uint ii = 0; ii < DK4 / NW; ++ii) {
                    const uint idx = ii * NW + tiisg;
                    float4 v_vec = float4(v_row[idx]);
                    my_so4[idx] += v_vec * weight;
                }
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =====================================================================
    // Cross-simdgroup reduction
    // =====================================================================
    if (tiisg == 0) {
        sm_s[sgitg] = S;
        sm_m[sgitg] = M;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads in simdgroup 0 do the final reduction and output
    if (sgitg == 0) {
        // Find global max across all simdgroups
        float final_M = sm_m[0];
        for (uint sg = 1; sg < NSG; ++sg) {
            final_M = max(final_M, sm_m[sg]);
        }

        // Compute combined sum with proper scaling
        float final_S = 0.0f;
        for (uint sg = 0; sg < NSG; ++sg) {
            final_S += sm_s[sg] * exp(sm_m[sg] - final_M);
        }

        // Combine output vectors - each thread handles multiple float4s
        for (uint i = tiisg; i < DK4; i += NW) {
            float4 final_o = float4(0.0f);
            for (uint sg = 0; sg < NSG; ++sg) {
                float scale_sg = exp(sm_m[sg] - final_M);
                final_o += so4[sg * DK4 + i] * scale_sg;
            }

            // Normalize and write output
            float inv_S = (final_S > 0.0f) ? 1.0f / final_S : 0.0f;
            out4[i] = final_o * inv_S;
        }
    }
}

// =============================================================================
// PAGED FLASH ATTENTION - Long Context Support with Block Tables
// =============================================================================
//
// This kernel extends flash attention to support arbitrary sequence lengths
// using paged KV cache with block tables. Uses online softmax to avoid
// materializing the full attention matrix.
//
// Key features:
// - No O(seq_len) threadgroup memory requirement
// - Supports sequences > 4096 tokens (limited only by total KV blocks)
// - Uses block table for scattered KV block lookups
// - Online softmax with running max/sum for numerical stability
//
// Memory layout:
//   K_cache/V_cache: [num_blocks * block_size, num_kv_heads, head_dim] (half)
//   block_table: [num_logical_blocks] int32
//
// =============================================================================

// Tile size for KV positions per iteration (fits in shared memory)
constant constexpr uint PAGED_FLASH_TILE = 32;

kernel void paged_flash_attention_decode_d64(
    device const float* Q              [[buffer(0)]],   // [num_heads, 64] FP32
    device const half* K_cache         [[buffer(1)]],   // [num_blocks * block_size, num_kv_heads, head_dim] FP16
    device const half* V_cache         [[buffer(2)]],   // [num_blocks * block_size, num_kv_heads, head_dim] FP16
    device const int* block_table      [[buffer(3)]],   // [num_logical_blocks]
    device float* output               [[buffer(4)]],   // [num_heads, 64] FP32
    constant uint& num_heads           [[buffer(5)]],
    constant uint& num_kv_heads        [[buffer(6)]],
    constant uint& seq_len             [[buffer(7)]],   // Total sequence length
    constant uint& head_dim            [[buffer(8)]],   // Must be 64
    constant uint& block_size          [[buffer(9)]],   // Tokens per block (e.g., 16)
    constant float& scale              [[buffer(10)]],
    uint head_idx                      [[threadgroup_position_in_grid]],
    ushort tiisg                       [[thread_index_in_simdgroup]],
    ushort sgitg                       [[simdgroup_index_in_threadgroup]]
) {
    if (head_idx >= num_heads) return;

    // Constants for head_dim=64
    constexpr uint HD = 64;
    constexpr uint HD4 = HD / 4;  // 16 float4s
    constexpr uint TILE = PAGED_FLASH_TILE;  // 32 positions per tile
    constexpr uint NSG = 4;  // Number of simdgroups
    constexpr uint NW = 32;  // Threads per simdgroup

    // GQA: map query head to KV head
    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;

    // Pointers
    device const float4* q4 = (device const float4*)(Q + head_idx * HD);
    device float* out_head = output + head_idx * HD;

    // KV stride for indexing: cache[physical_pos, kv_head, d]
    uint kv_stride = num_kv_heads * HD;

    // Shared memory
    threadgroup float4 sq4[HD4];        // Q vector (16 float4s = 64 floats)
    threadgroup half4 skv4[TILE * HD4]; // K/V tile buffer (32 * 16 half4s)
    threadgroup float ss[TILE];         // Attention scores for current tile
    threadgroup float4 so4[HD4];        // Output accumulator
    threadgroup float sg_scratch[NSG];  // For cross-simdgroup reductions

    // Load Q into shared memory
    uint tid = tiisg + sgitg * NW;
    if (tid < HD4) {
        sq4[tid] = q4[tid];
    }

    // Initialize output accumulator
    if (tid < HD4) {
        so4[tid] = float4(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float running_max = -FLT_MAX / 2;
    float running_sum = 0.0f;

    // Process KV cache in tiles
    uint num_tiles = (seq_len + TILE - 1) / TILE;

    for (uint tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        uint tile_start = tile_idx * TILE;
        uint tile_len = min(TILE, seq_len - tile_start);

        // =====================================================================
        // Step 1: Load K tile with block table lookups
        // =====================================================================
        // Each thread loads one K vector's worth of data
        for (uint i = tid; i < tile_len; i += 128) {
            uint logical_pos = tile_start + i;

            // Block table lookup
            uint logical_block = logical_pos / block_size;
            uint block_offset = logical_pos % block_size;
            int physical_block = block_table[logical_block];
            uint physical_pos = physical_block * block_size + block_offset;

            // K at [physical_pos, kv_head, :]
            device const half4* k4_vec = (device const half4*)(K_cache + physical_pos * kv_stride + kv_head * HD);

            // Copy to shared memory
            for (uint d = 0; d < HD4; d++) {
                skv4[i * HD4 + d] = k4_vec[d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // Step 2: Compute Q @ K^T for all K positions in tile
        // =====================================================================
        uint k_per_sg = (tile_len + NSG - 1) / NSG;
        uint k_start_sg = sgitg * k_per_sg;
        uint k_end_sg = min(k_start_sg + k_per_sg, tile_len);

        for (uint k = k_start_sg; k < k_end_sg; k++) {
            threadgroup half4* k4_vec = skv4 + k * HD4;

            // Vectorized dot product
            float partial_dot = 0.0f;
            if (tiisg < HD4) {
                float4 q_vec = sq4[tiisg];
                float4 k_vec = float4(k4_vec[tiisg]);
                partial_dot = dot(q_vec, k_vec);
            }

            // Sum across SIMD group
            float dot_sum = simd_sum(partial_dot);

            // Thread 0 writes the score
            if (tiisg == 0) {
                ss[k] = dot_sum * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // Step 3: Online softmax update
        // =====================================================================
        // Find max in this tile
        float local_max = -FLT_MAX / 2;
        for (uint k = tiisg; k < tile_len; k += NW) {
            local_max = max(local_max, ss[k]);
        }
        float tile_max = simd_max(local_max);

        // Cross-simdgroup max reduction
        if (tiisg == 0) sg_scratch[sgitg] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0 && tiisg == 0) {
            float m = sg_scratch[0];
            for (uint i = 1; i < NSG; i++) m = max(m, sg_scratch[i]);
            sg_scratch[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max = sg_scratch[0];

        // Update running max and correction factor
        float new_max = max(running_max, tile_max);
        float correction = (running_max > -FLT_MAX / 4) ? exp(running_max - new_max) : 0.0f;

        // Scale previous output accumulator
        if (tid < HD4) {
            so4[tid] *= correction;
        }

        // Compute exp(score - new_max) and sum
        float local_sum = 0.0f;
        for (uint k = tiisg; k < tile_len; k += NW) {
            float exp_score = exp(ss[k] - new_max);
            ss[k] = exp_score;
            local_sum += exp_score;
        }
        float tile_sum = simd_sum(local_sum);

        // Cross-simdgroup sum reduction
        if (tiisg == 0) sg_scratch[sgitg] = tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0 && tiisg == 0) {
            float s = 0.0f;
            for (uint i = 0; i < NSG; i++) s += sg_scratch[i];
            sg_scratch[0] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_sum = sg_scratch[0];

        running_sum = running_sum * correction + tile_sum;
        running_max = new_max;

        // =====================================================================
        // Step 4: Load V tile with block table lookups and compute S @ V
        // =====================================================================
        // Load V tile (same positions as K)
        for (uint i = tid; i < tile_len; i += 128) {
            uint logical_pos = tile_start + i;

            // Block table lookup (same as K)
            uint logical_block = logical_pos / block_size;
            uint block_offset = logical_pos % block_size;
            int physical_block = block_table[logical_block];
            uint physical_pos = physical_block * block_size + block_offset;

            // V at [physical_pos, kv_head, :]
            device const half4* v4_vec = (device const half4*)(V_cache + physical_pos * kv_stride + kv_head * HD);

            // Copy to shared memory
            for (uint d = 0; d < HD4; d++) {
                skv4[i * HD4 + d] = v4_vec[d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate: output += attention_weights @ V
        if (tid < HD4) {
            float4 accum = float4(0.0f);
            for (uint k = 0; k < tile_len; k++) {
                float weight = ss[k];
                float4 v_vec = float4(skv4[k * HD4 + tid]);
                accum += weight * v_vec;
            }
            so4[tid] += accum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =====================================================================
    // Step 5: Normalize and write output
    // =====================================================================
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    if (tid < HD4) {
        float4 result = so4[tid] * inv_sum;
        out_head[tid * 4 + 0] = result.x;
        out_head[tid * 4 + 1] = result.y;
        out_head[tid * 4 + 2] = result.z;
        out_head[tid * 4 + 3] = result.w;
    }
}

// =============================================================================
// PAGED FLASH ATTENTION - head_dim=128 specialization
// =============================================================================

kernel void paged_flash_attention_decode_d128(
    device const float* Q              [[buffer(0)]],   // [num_heads, 128] FP32
    device const half* K_cache         [[buffer(1)]],   // [num_blocks * block_size, num_kv_heads, head_dim] FP16
    device const half* V_cache         [[buffer(2)]],   // [num_blocks * block_size, num_kv_heads, head_dim] FP16
    device const int* block_table      [[buffer(3)]],   // [num_logical_blocks]
    device float* output               [[buffer(4)]],   // [num_heads, 128] FP32
    constant uint& num_heads           [[buffer(5)]],
    constant uint& num_kv_heads        [[buffer(6)]],
    constant uint& seq_len             [[buffer(7)]],   // Total sequence length
    constant uint& head_dim            [[buffer(8)]],   // Must be 128
    constant uint& block_size          [[buffer(9)]],   // Tokens per block
    constant float& scale              [[buffer(10)]],
    uint head_idx                      [[threadgroup_position_in_grid]],
    ushort tiisg                       [[thread_index_in_simdgroup]],
    ushort sgitg                       [[simdgroup_index_in_threadgroup]]
) {
    if (head_idx >= num_heads) return;

    // Constants for head_dim=128
    constexpr uint HD = 128;
    constexpr uint HD4 = HD / 4;  // 32 float4s
    constexpr uint TILE = PAGED_FLASH_TILE;  // 32 positions per tile
    constexpr uint NSG = 4;  // Number of simdgroups
    constexpr uint NW = 32;  // Threads per simdgroup

    // GQA: map query head to KV head
    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head = head_idx / heads_per_kv;

    // Pointers
    device const float4* q4 = (device const float4*)(Q + head_idx * HD);
    device float* out_head = output + head_idx * HD;

    // KV stride for indexing
    uint kv_stride = num_kv_heads * HD;

    // Shared memory
    threadgroup float4 sq4[HD4];        // Q vector (32 float4s = 128 floats)
    threadgroup half4 skv4[TILE * HD4]; // K/V tile buffer (32 * 32 = 1024 half4s)
    threadgroup float ss[TILE];         // Attention scores
    threadgroup float4 so4[HD4];        // Output accumulator
    threadgroup float sg_scratch[NSG];  // For reductions

    // Load Q into shared memory
    uint tid = tiisg + sgitg * NW;
    if (tid < HD4) {
        sq4[tid] = q4[tid];
    }

    // Initialize output accumulator
    if (tid < HD4) {
        so4[tid] = float4(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float running_max = -FLT_MAX / 2;
    float running_sum = 0.0f;

    // Process KV cache in tiles
    uint num_tiles = (seq_len + TILE - 1) / TILE;

    for (uint tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        uint tile_start = tile_idx * TILE;
        uint tile_len = min(TILE, seq_len - tile_start);

        // Step 1: Load K tile with block table lookups
        for (uint i = tid; i < tile_len; i += 128) {
            uint logical_pos = tile_start + i;
            uint logical_block = logical_pos / block_size;
            uint block_offset = logical_pos % block_size;
            int physical_block = block_table[logical_block];
            uint physical_pos = physical_block * block_size + block_offset;

            device const half4* k4_vec = (device const half4*)(K_cache + physical_pos * kv_stride + kv_head * HD);
            for (uint d = 0; d < HD4; d++) {
                skv4[i * HD4 + d] = k4_vec[d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 2: Compute Q @ K^T
        uint k_per_sg = (tile_len + NSG - 1) / NSG;
        uint k_start_sg = sgitg * k_per_sg;
        uint k_end_sg = min(k_start_sg + k_per_sg, tile_len);

        for (uint k = k_start_sg; k < k_end_sg; k++) {
            threadgroup half4* k4_vec = skv4 + k * HD4;

            float partial_dot = 0.0f;
            if (tiisg < HD4) {
                float4 q_vec = sq4[tiisg];
                float4 k_vec = float4(k4_vec[tiisg]);
                partial_dot = dot(q_vec, k_vec);
            }

            float dot_sum = simd_sum(partial_dot);
            if (tiisg == 0) {
                ss[k] = dot_sum * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Online softmax update
        float local_max = -FLT_MAX / 2;
        for (uint k = tiisg; k < tile_len; k += NW) {
            local_max = max(local_max, ss[k]);
        }
        float tile_max = simd_max(local_max);

        if (tiisg == 0) sg_scratch[sgitg] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0 && tiisg == 0) {
            float m = sg_scratch[0];
            for (uint i = 1; i < NSG; i++) m = max(m, sg_scratch[i]);
            sg_scratch[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max = sg_scratch[0];

        float new_max = max(running_max, tile_max);
        float correction = (running_max > -FLT_MAX / 4) ? exp(running_max - new_max) : 0.0f;

        if (tid < HD4) {
            so4[tid] *= correction;
        }

        float local_sum = 0.0f;
        for (uint k = tiisg; k < tile_len; k += NW) {
            float exp_score = exp(ss[k] - new_max);
            ss[k] = exp_score;
            local_sum += exp_score;
        }
        float tile_sum = simd_sum(local_sum);

        if (tiisg == 0) sg_scratch[sgitg] = tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0 && tiisg == 0) {
            float s = 0.0f;
            for (uint i = 0; i < NSG; i++) s += sg_scratch[i];
            sg_scratch[0] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_sum = sg_scratch[0];

        running_sum = running_sum * correction + tile_sum;
        running_max = new_max;

        // Step 4: Load V tile and compute S @ V
        for (uint i = tid; i < tile_len; i += 128) {
            uint logical_pos = tile_start + i;
            uint logical_block = logical_pos / block_size;
            uint block_offset = logical_pos % block_size;
            int physical_block = block_table[logical_block];
            uint physical_pos = physical_block * block_size + block_offset;

            device const half4* v4_vec = (device const half4*)(V_cache + physical_pos * kv_stride + kv_head * HD);
            for (uint d = 0; d < HD4; d++) {
                skv4[i * HD4 + d] = v4_vec[d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < HD4) {
            float4 accum = float4(0.0f);
            for (uint k = 0; k < tile_len; k++) {
                float weight = ss[k];
                float4 v_vec = float4(skv4[k * HD4 + tid]);
                accum += weight * v_vec;
            }
            so4[tid] += accum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 5: Normalize and write output
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    if (tid < HD4) {
        float4 result = so4[tid] * inv_sum;
        out_head[tid * 4 + 0] = result.x;
        out_head[tid * 4 + 1] = result.y;
        out_head[tid * 4 + 2] = result.z;
        out_head[tid * 4 + 3] = result.w;
    }
}
)";
