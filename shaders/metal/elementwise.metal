#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Element-wise Binary Operations
// =============================================================================

kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = a[id] + b[id];
    }
}

kernel void add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = a[id] + b[id];
    }
}

kernel void sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = a[id] - b[id];
    }
}

kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = a[id] * b[id];
    }
}

kernel void mul_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = a[id] * b[id];
    }
}

kernel void div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = a[id] / b[id];
    }
}

// =============================================================================
// Activation Functions
// =============================================================================

kernel void relu_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = max(x[id], 0.0f);
    }
}

kernel void relu_f16(
    device const half* x [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = max(x[id], half(0.0h));
    }
}

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        float val = x[id];
        float x3 = val * val * val;
        float inner = 0.7978845608f * (val + 0.044715f * x3);  // sqrt(2/pi) ≈ 0.7978845608
        out[id] = 0.5f * val * (1.0f + tanh(inner));
    }
}

kernel void gelu_f16(
    device const half* x [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        float val = float(x[id]);
        float x3 = val * val * val;
        float inner = 0.7978845608f * (val + 0.044715f * x3);
        out[id] = half(0.5f * val * (1.0f + tanh(inner)));
    }
}

// SiLU (Swish): x * sigmoid(x)
kernel void silu_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        float val = x[id];
        out[id] = val / (1.0f + exp(-val));
    }
}

kernel void silu_f16(
    device const half* x [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        float val = float(x[id]);
        out[id] = half(val / (1.0f + exp(-val)));
    }
}

kernel void sigmoid_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = 1.0f / (1.0f + exp(-x[id]));
    }
}

kernel void tanh_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = tanh(x[id]);
    }
}

// =============================================================================
// Softmax (along last dimension)
// =============================================================================

// Two-pass softmax for numerical stability
// Pass 1: Find max value per row
kernel void softmax_max_f32(
    device const float* x [[buffer(0)]],
    device float* max_vals [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row < rows) {
        float max_val = -INFINITY;
        for (uint i = 0; i < cols; i++) {
            max_val = max(max_val, x[row * cols + i]);
        }
        max_vals[row] = max_val;
    }
}

// Pass 2: Compute exp(x - max) and sum
kernel void softmax_exp_sum_f32(
    device const float* x [[buffer(0)]],
    device const float* max_vals [[buffer(1)]],
    device float* exp_vals [[buffer(2)]],
    device float* sum_vals [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint row [[thread_position_in_grid]])
{
    if (row < rows) {
        float max_val = max_vals[row];
        float sum = 0.0f;
        for (uint i = 0; i < cols; i++) {
            float e = exp(x[row * cols + i] - max_val);
            exp_vals[row * cols + i] = e;
            sum += e;
        }
        sum_vals[row] = sum;
    }
}

// Pass 3: Normalize
kernel void softmax_norm_f32(
    device const float* exp_vals [[buffer(0)]],
    device const float* sum_vals [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint row = pos.y;
    uint col = pos.x;
    if (row < rows && col < cols) {
        out[row * cols + col] = exp_vals[row * cols + col] / sum_vals[row];
    }
}

// Fused single-pass softmax (for smaller dimensions)
kernel void softmax_fused_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row < rows) {
        // Find max
        float max_val = -INFINITY;
        for (uint i = 0; i < cols; i++) {
            max_val = max(max_val, x[row * cols + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (uint i = 0; i < cols; i++) {
            float e = exp(x[row * cols + i] - max_val);
            out[row * cols + i] = e;
            sum += e;
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (uint i = 0; i < cols; i++) {
            out[row * cols + i] *= inv_sum;
        }
    }
}
