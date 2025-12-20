// =============================================================================
// Metal Shader Utility Kernels
// =============================================================================
// Utility kernels: RMSNorm, SiLU, RoPE, embedding lookup, element-wise ops,
// and softmax.
// =============================================================================

#pragma once

static const char* METAL_SHADER_UTIL = R"(
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

// Batched RMS norm with FP32 weights - processes M tokens of size N each
// x: [M, N] input, out: [M, N] output, weight: [N] (broadcast)
// One threadgroup per token (row)
kernel void rms_norm_batch(
    device const float* x          [[buffer(0)]],
    device const float* weight     [[buffer(1)]],
    device float* out              [[buffer(2)]],
    constant uint& M               [[buffer(3)]],  // Batch size (num tokens)
    constant uint& N               [[buffer(4)]],  // Hidden dim
    constant float& eps            [[buffer(5)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    if (tg_idx >= M) return;

    // Pointer to this token's data
    device const float* x_row = x + tg_idx * N;
    device float* out_row = out + tg_idx * N;

    threadgroup float shared_sum[256];

    // Each thread sums multiple elements
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float val = x_row[i];
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

    // Thread 0 computes inv_rms
    threadgroup float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrt(shared_sum[0] / float(N) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread normalizes and scales its elements
    for (uint i = tid; i < N; i += tg_size) {
        out_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

// Batched RMS norm with FP16 weights - processes M tokens of size N each
// x: [M, N] input, out: [M, N] output, weight: [N] (broadcast)
// One threadgroup per token (row)
kernel void rms_norm_batch_f16(
    device const float* x          [[buffer(0)]],
    device const half* weight      [[buffer(1)]],
    device float* out              [[buffer(2)]],
    constant uint& M               [[buffer(3)]],  // Batch size (num tokens)
    constant uint& N               [[buffer(4)]],  // Hidden dim
    constant float& eps            [[buffer(5)]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    if (tg_idx >= M) return;

    // Pointer to this token's data
    device const float* x_row = x + tg_idx * N;
    device float* out_row = out + tg_idx * N;

    threadgroup float shared_sum[256];

    // Each thread sums multiple elements
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float val = x_row[i];
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

    // Thread 0 computes inv_rms
    threadgroup float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrt(shared_sum[0] / float(N) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread normalizes and scales its elements
    for (uint i = tid; i < N; i += tg_size) {
        out_row[i] = x_row[i] * inv_rms * float(weight[i]);
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

)";
