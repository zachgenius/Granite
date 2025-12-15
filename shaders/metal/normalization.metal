#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Layer Normalization
// =============================================================================

// LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
// Applied over the last dimension

kernel void layer_norm_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& num_rows [[buffer(4)]],
    constant uint& norm_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= num_rows) return;

    uint offset = row * norm_size;

    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < norm_size; i++) {
        sum += x[offset + i];
    }
    float mean = sum / float(norm_size);

    // Compute variance
    float var_sum = 0.0f;
    for (uint i = 0; i < norm_size; i++) {
        float diff = x[offset + i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / float(norm_size);
    float inv_std = rsqrt(var + eps);

    // Normalize and apply affine transform
    for (uint i = 0; i < norm_size; i++) {
        float normalized = (x[offset + i] - mean) * inv_std;
        out[offset + i] = normalized * weight[i] + bias[i];
    }
}

kernel void layer_norm_f16(
    device const half* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant uint& num_rows [[buffer(4)]],
    constant uint& norm_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= num_rows) return;

    uint offset = row * norm_size;

    // Compute in FP32 for precision
    float sum = 0.0f;
    for (uint i = 0; i < norm_size; i++) {
        sum += float(x[offset + i]);
    }
    float mean = sum / float(norm_size);

    float var_sum = 0.0f;
    for (uint i = 0; i < norm_size; i++) {
        float diff = float(x[offset + i]) - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / float(norm_size);
    float inv_std = rsqrt(var + eps);

    for (uint i = 0; i < norm_size; i++) {
        float normalized = (float(x[offset + i]) - mean) * inv_std;
        out[offset + i] = half(normalized * float(weight[i]) + float(bias[i]));
    }
}

// =============================================================================
// RMS Normalization (used in LLaMA, etc.)
// =============================================================================

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight

kernel void rms_norm_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& norm_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= num_rows) return;

    uint offset = row * norm_size;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = 0; i < norm_size; i++) {
        float val = x[offset + i];
        sum_sq += val * val;
    }

    // RMS = sqrt(mean(x^2))
    float rms = sqrt(sum_sq / float(norm_size) + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and apply weight
    for (uint i = 0; i < norm_size; i++) {
        out[offset + i] = x[offset + i] * inv_rms * weight[i];
    }
}

kernel void rms_norm_f16(
    device const half* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& norm_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= num_rows) return;

    uint offset = row * norm_size;

    // Compute in FP32
    float sum_sq = 0.0f;
    for (uint i = 0; i < norm_size; i++) {
        float val = float(x[offset + i]);
        sum_sq += val * val;
    }

    float rms = sqrt(sum_sq / float(norm_size) + eps);
    float inv_rms = 1.0f / rms;

    for (uint i = 0; i < norm_size; i++) {
        out[offset + i] = half(float(x[offset + i]) * inv_rms * float(weight[i]));
    }
}

// =============================================================================
// Optimized LayerNorm with SIMD groups (for larger dimensions)
// =============================================================================

kernel void layer_norm_simd_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& num_rows [[buffer(4)]],
    constant uint& norm_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= num_rows) return;

    uint offset = row * norm_size;
    uint simd_size = 32;  // Apple GPUs have 32-wide SIMD

    // Each thread handles multiple elements
    float local_sum = 0.0f;
    for (uint i = lane; i < norm_size; i += simd_size) {
        local_sum += x[offset + i];
    }

    // Reduce within SIMD group
    float sum = simd_sum(local_sum);
    float mean = sum / float(norm_size);

    // Compute variance
    float local_var = 0.0f;
    for (uint i = lane; i < norm_size; i += simd_size) {
        float diff = x[offset + i] - mean;
        local_var += diff * diff;
    }
    float var = simd_sum(local_var) / float(norm_size);
    float inv_std = rsqrt(var + eps);

    // Normalize
    for (uint i = lane; i < norm_size; i += simd_size) {
        float normalized = (x[offset + i] - mean) * inv_std;
        out[offset + i] = normalized * weight[i] + bias[i];
    }
}

kernel void rms_norm_simd_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& norm_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (row >= num_rows) return;

    uint offset = row * norm_size;
    uint simd_size = 32;

    float local_sum_sq = 0.0f;
    for (uint i = lane; i < norm_size; i += simd_size) {
        float val = x[offset + i];
        local_sum_sq += val * val;
    }

    float sum_sq = simd_sum(local_sum_sq);
    float inv_rms = rsqrt(sum_sq / float(norm_size) + eps);

    for (uint i = lane; i < norm_size; i += simd_size) {
        out[offset + i] = x[offset + i] * inv_rms * weight[i];
    }
}
