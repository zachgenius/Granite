// =============================================================================
// Metal Shader Utility Kernels
// =============================================================================
// Utility kernels: RMSNorm, SiLU, RoPE, embedding lookup, element-wise ops,
// and softmax.
// =============================================================================

#pragma once

static const char* METAL_SHADER_UTIL = R"(
// Fast RMS norm using simd_sum for reduction (matches llama.cpp approach)
kernel void rms_norm(
    device const float* x          [[buffer(0)]],
    device const float* weight     [[buffer(1)]],
    device float* out              [[buffer(2)]],
    constant uint& size            [[buffer(3)]],
    constant float& eps            [[buffer(4)]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]]
) {
    // Shared memory for inter-simdgroup reduction (only need 8 slots for 256 threads)
    threadgroup float shared_sum[8];

    // Each thread sums multiple elements
    float local_sum = 0.0f;
    for (uint i = tid; i < size; i += tg_size) {
        float val = x[i];
        local_sum += val * val;
    }

    // Fast simd reduction within each simdgroup (32 threads)
    local_sum = simd_sum(local_sum);

    // First thread of each simdgroup stores partial sum
    if (simd_lane == 0) {
        shared_sum[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup reduces the partial sums
    float final_sum = 0.0f;
    if (simd_group == 0 && simd_lane < (tg_size / 32)) {
        final_sum = shared_sum[simd_lane];
        final_sum = simd_sum(final_sum);
    }

    // Thread 0 computes inv_rms
    threadgroup float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrt(final_sum / float(size) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread normalizes and scales its elements
    for (uint i = tid; i < size; i += tg_size) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// RMS norm with FP16 weights - using simd_sum
kernel void rms_norm_f16(
    device const float* x          [[buffer(0)]],
    device const half* weight      [[buffer(1)]],
    device float* out              [[buffer(2)]],
    constant uint& size            [[buffer(3)]],
    constant float& eps            [[buffer(4)]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[8];

    float local_sum = 0.0f;
    for (uint i = tid; i < size; i += tg_size) {
        float val = x[i];
        local_sum += val * val;
    }

    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        shared_sum[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float final_sum = 0.0f;
    if (simd_group == 0 && simd_lane < (tg_size / 32)) {
        final_sum = shared_sum[simd_lane];
        final_sum = simd_sum(final_sum);
    }

    threadgroup float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrt(final_sum / float(size) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < size; i += tg_size) {
        out[i] = x[i] * inv_rms * float(weight[i]);
    }
}

// Batched RMS norm with FP32 weights - processes M tokens of size N each
// x: [M, N] input, out: [M, N] output, weight: [N] (broadcast)
// One threadgroup per token (row) - using simd_sum for fast reduction
kernel void rms_norm_batch(
    device const float* x          [[buffer(0)]],
    device const float* weight     [[buffer(1)]],
    device float* out              [[buffer(2)]],
    constant uint& M               [[buffer(3)]],  // Batch size (num tokens)
    constant uint& N               [[buffer(4)]],  // Hidden dim
    constant float& eps            [[buffer(5)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]]
) {
    if (tg_idx >= M) return;

    // Pointer to this token's data
    device const float* x_row = x + tg_idx * N;
    device float* out_row = out + tg_idx * N;

    threadgroup float shared_sum[8];

    // Each thread sums multiple elements with float4 vectorized loads
    float local_sum = 0.0f;
    uint i = tid * 4;
    for (; i + 3 < N; i += tg_size * 4) {
        float4 val = *((device const float4*)(x_row + i));
        local_sum += dot(val, val);
    }
    // Handle remaining elements
    for (; i < N; i += tg_size) {
        float val = x_row[i];
        local_sum += val * val;
    }

    // Fast simd reduction
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        shared_sum[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float final_sum = 0.0f;
    if (simd_group == 0 && simd_lane < (tg_size / 32)) {
        final_sum = shared_sum[simd_lane];
        final_sum = simd_sum(final_sum);
    }

    // Thread 0 computes inv_rms
    threadgroup float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrt(final_sum / float(N) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread normalizes and scales its elements
    for (uint j = tid; j < N; j += tg_size) {
        out_row[j] = x_row[j] * inv_rms * weight[j];
    }
}

// Batched RMS norm with FP16 weights - processes M tokens of size N each
// x: [M, N] input, out: [M, N] output, weight: [N] (broadcast)
// One threadgroup per token (row) - using simd_sum for fast reduction
kernel void rms_norm_batch_f16(
    device const float* x          [[buffer(0)]],
    device const half* weight      [[buffer(1)]],
    device float* out              [[buffer(2)]],
    constant uint& M               [[buffer(3)]],  // Batch size (num tokens)
    constant uint& N               [[buffer(4)]],  // Hidden dim
    constant float& eps            [[buffer(5)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]]
) {
    if (tg_idx >= M) return;

    device const float* x_row = x + tg_idx * N;
    device float* out_row = out + tg_idx * N;

    threadgroup float shared_sum[8];

    // Vectorized accumulation
    float local_sum = 0.0f;
    uint i = tid * 4;
    for (; i + 3 < N; i += tg_size * 4) {
        float4 val = *((device const float4*)(x_row + i));
        local_sum += dot(val, val);
    }
    for (; i < N; i += tg_size) {
        float val = x_row[i];
        local_sum += val * val;
    }

    // Fast simd reduction
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        shared_sum[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float final_sum = 0.0f;
    if (simd_group == 0 && simd_lane < (tg_size / 32)) {
        final_sum = shared_sum[simd_lane];
        final_sum = simd_sum(final_sum);
    }

    threadgroup float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrt(final_sum / float(N) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < N; j += tg_size) {
        out_row[j] = x_row[j] * inv_rms * float(weight[j]);
    }
}

// Batched RMS norm with float input, half output (for f16 matmul input)
// x: [M, N] float input, out: [M, N] half output, weight: [N] float (broadcast)
// This eliminates a separate conversion kernel before f16 matmul - using simd_sum
kernel void rms_norm_batch_f32_to_f16(
    device const float* x          [[buffer(0)]],
    device const float* weight     [[buffer(1)]],
    device half* out               [[buffer(2)]],
    constant uint& M               [[buffer(3)]],  // Batch size (num tokens)
    constant uint& N               [[buffer(4)]],  // Hidden dim
    constant float& eps            [[buffer(5)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]]
) {
    if (tg_idx >= M) return;

    device const float* x_row = x + tg_idx * N;
    device half* out_row = out + tg_idx * N;

    threadgroup float shared_sum[8];

    float local_sum = 0.0f;
    uint i = tid * 4;
    for (; i + 3 < N; i += tg_size * 4) {
        float4 val = *((device const float4*)(x_row + i));
        local_sum += dot(val, val);
    }
    for (; i < N; i += tg_size) {
        float val = x_row[i];
        local_sum += val * val;
    }

    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        shared_sum[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float final_sum = 0.0f;
    if (simd_group == 0 && simd_lane < (tg_size / 32)) {
        final_sum = shared_sum[simd_lane];
        final_sum = simd_sum(final_sum);
    }

    threadgroup float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrt(final_sum / float(N) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < N; j += tg_size) {
        out_row[j] = half(x_row[j] * inv_rms * weight[j]);
    }
}

// Same as above but with FP16 weights - using simd_sum
kernel void rms_norm_batch_f16w_to_f16(
    device const float* x          [[buffer(0)]],
    device const half* weight      [[buffer(1)]],
    device half* out               [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    constant float& eps            [[buffer(5)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]]
) {
    if (tg_idx >= M) return;

    device const float* x_row = x + tg_idx * N;
    device half* out_row = out + tg_idx * N;

    threadgroup float shared_sum[8];

    float local_sum = 0.0f;
    uint i = tid * 4;
    for (; i + 3 < N; i += tg_size * 4) {
        float4 val = *((device const float4*)(x_row + i));
        local_sum += dot(val, val);
    }
    for (; i < N; i += tg_size) {
        float val = x_row[i];
        local_sum += val * val;
    }

    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        shared_sum[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float final_sum = 0.0f;
    if (simd_group == 0 && simd_lane < (tg_size / 32)) {
        final_sum = shared_sum[simd_lane];
        final_sum = simd_sum(final_sum);
    }

    threadgroup float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrt(final_sum / float(N) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < N; j += tg_size) {
        out_row[j] = half(x_row[j] * inv_rms * float(weight[j]));
    }
}

// Batched RMS norm with half precision I/O - for bandwidth-efficient prefill
// x: [M, N] half input, out: [M, N] half output, weight: [N] half (broadcast)
// Computation done in float for accuracy, I/O in half for bandwidth - using simd_sum
kernel void rms_norm_batch_half(
    device const half* x           [[buffer(0)]],
    device const half* weight      [[buffer(1)]],
    device half* out               [[buffer(2)]],
    constant uint& M               [[buffer(3)]],  // Batch size (num tokens)
    constant uint& N               [[buffer(4)]],  // Hidden dim
    constant float& eps            [[buffer(5)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]]
) {
    if (tg_idx >= M) return;

    device const half* x_row = x + tg_idx * N;
    device half* out_row = out + tg_idx * N;

    threadgroup float shared_sum[8];

    // Vectorized half4 loads for better bandwidth
    float local_sum = 0.0f;
    uint i = tid * 4;
    for (; i + 3 < N; i += tg_size * 4) {
        half4 h4 = *((device const half4*)(x_row + i));
        float4 val = float4(h4);
        local_sum += dot(val, val);
    }
    for (; i < N; i += tg_size) {
        float val = float(x_row[i]);
        local_sum += val * val;
    }

    // Fast simd reduction
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        shared_sum[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float final_sum = 0.0f;
    if (simd_group == 0 && simd_lane < (tg_size / 32)) {
        final_sum = shared_sum[simd_lane];
        final_sum = simd_sum(final_sum);
    }

    threadgroup float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrt(final_sum / float(N) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread normalizes and scales its elements (output in half)
    for (uint j = tid; j < N; j += tg_size) {
        out_row[j] = half(float(x_row[j]) * inv_rms * float(weight[j]));
    }
}

// Batched RMS norm with half precision I/O and FP32 weights - using simd_sum
kernel void rms_norm_batch_half_f32w(
    device const half* x           [[buffer(0)]],
    device const float* weight     [[buffer(1)]],
    device half* out               [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    constant float& eps            [[buffer(5)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]]
) {
    if (tg_idx >= M) return;

    device const half* x_row = x + tg_idx * N;
    device half* out_row = out + tg_idx * N;

    threadgroup float shared_sum[8];

    float local_sum = 0.0f;
    uint i = tid * 4;
    for (; i + 3 < N; i += tg_size * 4) {
        half4 h4 = *((device const half4*)(x_row + i));
        float4 val = float4(h4);
        local_sum += dot(val, val);
    }
    for (; i < N; i += tg_size) {
        float val = float(x_row[i]);
        local_sum += val * val;
    }

    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        shared_sum[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float final_sum = 0.0f;
    if (simd_group == 0 && simd_lane < (tg_size / 32)) {
        final_sum = shared_sum[simd_lane];
        final_sum = simd_sum(final_sum);
    }

    threadgroup float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrt(final_sum / float(N) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < N; j += tg_size) {
        out_row[j] = half(float(x_row[j]) * inv_rms * weight[j]);
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
    // Process 4 elements per thread for better memory bandwidth
    uint idx = gid * 4;
    if (idx + 3 < size) {
        float4 va = *((device const float4*)(a + idx));
        float4 vb = *((device const float4*)(b + idx));
        *((device float4*)(c + idx)) = va * vb;
    } else if (idx < size) {
        // Handle remaining elements
        for (uint i = idx; i < size; i++) {
            c[i] = a[i] * b[i];
        }
    }
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
    // Process 4 elements per thread for better memory bandwidth
    uint idx = gid * 4;
    if (idx + 3 < size) {
        float4 va = *((device const float4*)(a + idx));
        float4 vb = *((device const float4*)(b + idx));
        *((device float4*)(c + idx)) = va + vb;
    } else if (idx < size) {
        // Handle remaining elements
        for (uint i = idx; i < size; i++) {
            c[i] = a[i] + b[i];
        }
    }
}

// Half-precision elementwise add for bandwidth-efficient prefill
kernel void elementwise_add_half(
    device const half* a           [[buffer(0)]],
    device const half* b           [[buffer(1)]],
    device half* c                 [[buffer(2)]],
    constant uint& size            [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    c[gid] = a[gid] + b[gid];
}

// Float-to-half conversion for prefill pipeline
kernel void convert_f32_to_f16(
    device const float* src        [[buffer(0)]],
    device half* dst               [[buffer(1)]],
    constant uint& size            [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    dst[gid] = half(src[gid]);
}

// Half-to-float conversion
kernel void convert_f16_to_f32(
    device const half* src         [[buffer(0)]],
    device float* dst              [[buffer(1)]],
    constant uint& size            [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    dst[gid] = float(src[gid]);
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

// Softmax over rows (in-place) — simd-reduced, 256 threads per row
// x shape: [M, N], softmax over N dimension
// One threadgroup (256 threads) collaborates on each row for high throughput.
kernel void softmax_row(
    device float* x                [[buffer(0)]],
    constant uint& M               [[buffer(1)]],
    constant uint& N               [[buffer(2)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]]
) {
    if (tg_idx >= M) return;

    device float* row = x + tg_idx * N;

    threadgroup float shared_val[8];

    // 1. Parallel max via simd_max + threadgroup reduction
    float local_max = -INFINITY;
    for (uint i = tid; i < N; i += tg_size) {
        local_max = max(local_max, row[i]);
    }
    local_max = simd_max(local_max);
    if (simd_lane == 0) shared_val[simd_group] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < (tg_size / 32)) {
        local_max = shared_val[simd_lane];
        local_max = simd_max(local_max);
    }
    threadgroup float row_max;
    if (tid == 0) row_max = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Parallel exp(x - max) and sum via simd_sum + threadgroup reduction
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float e = exp(row[i] - row_max);
        row[i] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) shared_val[simd_group] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float final_sum = 0.0f;
    if (simd_group == 0 && simd_lane < (tg_size / 32)) {
        final_sum = shared_val[simd_lane];
        final_sum = simd_sum(final_sum);
    }
    threadgroup float row_sum;
    if (tid == 0) row_sum = final_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Parallel normalize
    float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < N; i += tg_size) {
        row[i] *= inv_sum;
    }
}

)";
