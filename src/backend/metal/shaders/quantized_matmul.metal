// Granite Quantized Matrix Multiplication Metal Shaders
// Fused dequantization + matmul for maximum GPU performance

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Q4_K Format Constants
// =============================================================================
// Q4_K: 256 elements per super-block, 144 bytes per block
// Layout:
//   - d: fp16 scale (2 bytes)
//   - dmin: fp16 min (2 bytes)
//   - scales: 12 bytes (8 6-bit scales + 8 6-bit mins, packed)
//   - qs: 128 bytes (4-bit quantized values)

constant constexpr uint QK_K = 256;           // Elements per super-block
constant constexpr uint Q4_K_BLOCK_SIZE = 144; // Bytes per block

// Q4_K block structure (must match GGML layout exactly)
struct block_q4_K {
    half d;              // delta (scale)
    half dmin;           // min
    uint8_t scales[12];  // 8 6-bit scales + 8 6-bit mins
    uint8_t qs[128];     // 4-bit quantized values (256 elements, 2 per byte)
};

// =============================================================================
// Helper Functions
// =============================================================================

// Extract 6-bit scale and min values from packed format
// Matches GGML's get_scale_min_k4
inline void get_scale_min_k4(int j, const device uint8_t* q, thread uint8_t& sc, thread uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Dequantize a single Q4_K block to float (256 elements)
inline void dequantize_q4_k_block(
    const device block_q4_K* block,
    thread float* output  // Must be 256 floats
) {
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

        // Low nibble: 32 elements
        for (int l = 0; l < 32; l++) {
            output[j + l] = d1 * float(qs[l] & 0xF) - dm1;
        }
        // High nibble: 32 elements
        for (int l = 0; l < 32; l++) {
            output[j + 32 + l] = d2 * float(qs[l] >> 4) - dm2;
        }

        qs += 32;
        is += 2;
    }
}

// =============================================================================
// Q4_K Matrix-Vector Multiplication (Single Token Decode)
// =============================================================================
// Computes: y = x @ W^T where W is Q4_K quantized
// x: [1, K] float
// W: [N, K] Q4_K quantized (stored as [N, K/256] blocks)
// y: [1, N] float

kernel void matvec_q4k(
    device const float* x          [[buffer(0)]],  // Input [K]
    device const void* W           [[buffer(1)]],  // Weights [N, K/QK_K] Q4_K blocks
    device float* y                [[buffer(2)]],  // Output [N]
    constant uint& K               [[buffer(3)]],  // Input dimension
    constant uint& N               [[buffer(4)]],  // Output dimension
    uint gid                       [[thread_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    if (gid >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    float sum = 0.0f;

    // Each thread computes one output element
    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q4_K* block = &weights[gid * num_blocks_k + kb];

        float d = float(block->d);
        float dmin = float(block->dmin);
        const device uint8_t* scales = block->scales;
        const device uint8_t* qs = block->qs;

        // Process 256 elements per block
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

            // Accumulate dot product for low nibble (32 elements)
            for (int l = 0; l < 32; l++) {
                float w = d1 * float(qs[l] & 0xF) - dm1;
                sum += x[base_idx + l] * w;
            }

            // Accumulate dot product for high nibble (32 elements)
            for (int l = 0; l < 32; l++) {
                float w = d2 * float(qs[l] >> 4) - dm2;
                sum += x[base_idx + 32 + l] * w;
            }

            qs += 32;
            is += 2;
        }
    }

    y[gid] = sum;
}

// =============================================================================
// Q4_K Matrix Multiplication (Batched - Prefill)
// =============================================================================
// Computes: Y = X @ W^T where W is Q4_K quantized
// X: [M, K] float
// W: [N, K] Q4_K quantized
// Y: [M, N] float

// Tile sizes for efficient GPU utilization
constant constexpr uint TILE_M = 4;   // Rows per threadgroup
constant constexpr uint TILE_N = 32;  // Cols per threadgroup

kernel void matmul_q4k(
    device const float* X          [[buffer(0)]],  // Input [M, K]
    device const void* W           [[buffer(1)]],  // Weights [N, K/QK_K] Q4_K blocks
    device float* Y                [[buffer(2)]],  // Output [M, N]
    constant uint& M               [[buffer(3)]],  // Batch size (num tokens)
    constant uint& K               [[buffer(4)]],  // Input dimension
    constant uint& N               [[buffer(5)]],  // Output dimension
    uint2 gid                      [[thread_position_in_grid]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint2 tid                      [[thread_position_in_threadgroup]]
) {
    uint row = gid.y;  // Which input row (token)
    uint col = gid.x;  // Which output column (hidden dim)

    if (row >= M || col >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    float sum = 0.0f;

    // Compute dot product for this (row, col) pair
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

            // Low nibble
            for (int l = 0; l < 32; l++) {
                float w = d1 * float(qs[l] & 0xF) - dm1;
                sum += X[row * K + base_idx + l] * w;
            }

            // High nibble
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

// =============================================================================
// Optimized Q4_K MatVec with SIMD Groups
// =============================================================================
// Uses simdgroup operations for faster reduction

kernel void matvec_q4k_simd(
    device const float* x          [[buffer(0)]],  // Input [K]
    device const void* W           [[buffer(1)]],  // Weights [N, K/QK_K] Q4_K blocks
    device float* y                [[buffer(2)]],  // Output [N]
    constant uint& K               [[buffer(3)]],  // Input dimension
    constant uint& N               [[buffer(4)]],  // Output dimension
    uint gid                       [[thread_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]]
) {
    // Each simdgroup (32 threads) processes one output row
    uint row = gid / 32;
    if (row >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    float partial_sum = 0.0f;

    // Distribute blocks across simd lanes
    uint blocks_per_lane = (num_blocks_k + 31) / 32;
    uint start_block = simd_lane * blocks_per_lane;
    uint end_block = min(start_block + blocks_per_lane, num_blocks_k);

    for (uint kb = start_block; kb < end_block; kb++) {
        const device block_q4_K* block = &weights[row * num_blocks_k + kb];

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
                partial_sum += x[base_idx + l] * w;
            }

            for (int l = 0; l < 32; l++) {
                float w = d2 * float(qs[l] >> 4) - dm2;
                partial_sum += x[base_idx + 32 + l] * w;
            }

            qs += 32;
            is += 2;
        }
    }

    // Reduce across simd group
    float sum = simd_sum(partial_sum);

    // Lane 0 writes result
    if (simd_lane == 0) {
        y[row] = sum;
    }
}

// =============================================================================
// Q8_0 Matrix-Vector Multiplication
// =============================================================================
// Q8_0: 32 elements per block, simpler format
// - d: fp16 scale (2 bytes)
// - qs: 32 int8 values (32 bytes)

struct block_q8_0 {
    half d;          // delta
    int8_t qs[32];   // quants
};

kernel void matvec_q8_0(
    device const float* x          [[buffer(0)]],  // Input [K]
    device const void* W           [[buffer(1)]],  // Weights [N, K/32] Q8_0 blocks
    device float* y                [[buffer(2)]],  // Output [N]
    constant uint& K               [[buffer(3)]],  // Input dimension
    constant uint& N               [[buffer(4)]],  // Output dimension
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= N) return;

    const uint QK8_0 = 32;
    const uint num_blocks_k = K / QK8_0;
    const device block_q8_0* weights = (const device block_q8_0*)W;

    float sum = 0.0f;

    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q8_0* block = &weights[gid * num_blocks_k + kb];
        float d = float(block->d);

        uint base_idx = kb * QK8_0;

        for (int i = 0; i < 32; i++) {
            float w = d * float(block->qs[i]);
            sum += x[base_idx + i] * w;
        }
    }

    y[gid] = sum;
}

// =============================================================================
// FP16 Matrix-Vector Multiplication (for embedding lookup results)
// =============================================================================

kernel void matvec_f16(
    device const float* x          [[buffer(0)]],  // Input [K]
    device const half* W           [[buffer(1)]],  // Weights [N, K]
    device float* y                [[buffer(2)]],  // Output [N]
    constant uint& K               [[buffer(3)]],  // Input dimension
    constant uint& N               [[buffer(4)]],  // Output dimension
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= N) return;

    float sum = 0.0f;

    for (uint i = 0; i < K; i++) {
        sum += x[i] * float(W[gid * K + i]);
    }

    y[gid] = sum;
}

// =============================================================================
// FP32 Matrix-Vector Multiplication
// =============================================================================

kernel void matvec_f32(
    device const float* x          [[buffer(0)]],  // Input [K]
    device const float* W          [[buffer(1)]],  // Weights [N, K]
    device float* y                [[buffer(2)]],  // Output [N]
    constant uint& K               [[buffer(3)]],  // Input dimension
    constant uint& N               [[buffer(4)]],  // Output dimension
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= N) return;

    float sum = 0.0f;

    for (uint i = 0; i < K; i++) {
        sum += x[i] * W[gid * K + i];
    }

    y[gid] = sum;
}
