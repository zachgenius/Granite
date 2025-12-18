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
// SIMD-Optimized Q4_K Matrix Multiplication (Prefill)
// =============================================================================
// Uses simdgroup operations for cooperative tiled computation
// Each simdgroup computes a 4x4 output tile cooperatively
// Much more efficient than scalar per-element computation
//
// Computes: Y = X @ W^T where W is Q4_K quantized
// X: [M, K] float
// W: [N, K] Q4_K quantized
// Y: [M, N] float

// Tile configuration for SIMD matmul
constant constexpr uint SIMD_TILE_M = 4;   // Output rows per simdgroup
constant constexpr uint SIMD_TILE_N = 4;   // Output cols per simdgroup
constant constexpr uint SIMD_SIZE = 32;    // Threads per simdgroup

kernel void matmul_q4k_simd(
    device const float* X          [[buffer(0)]],  // Input [M, K]
    device const void* W           [[buffer(1)]],  // Weights [N, K/QK_K] Q4_K blocks
    device float* Y                [[buffer(2)]],  // Output [M, N]
    constant uint& M               [[buffer(3)]],  // Batch size (num tokens)
    constant uint& K               [[buffer(4)]],  // Input dimension
    constant uint& N               [[buffer(5)]],  // Output dimension
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_gid                  [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles SIMD_TILE_M x (SIMD_TILE_N * num_simdgroups) output tile
    // Each simdgroup handles SIMD_TILE_M x SIMD_TILE_N output tile
    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    // Tile position in output matrix
    // tgid.y = row tile index, tgid.x = col tile index
    uint tile_row = tgid.y * SIMD_TILE_M;
    uint tile_col = (tgid.x * 8 + simd_gid) * SIMD_TILE_N;  // 8 simdgroups per TG

    // Early exit if out of bounds
    if (tile_row >= M || tile_col >= N) return;

    // Each thread in simdgroup accumulates partial results for its assigned (row, col) pairs
    // Lane assignment: lane 0-3 handle row 0, lane 4-7 handle row 1, etc.
    // Within each group of 4 lanes, each lane handles one column
    uint local_row = simd_lane / 8;           // 0-3 (which row within tile)
    uint local_col = simd_lane % 4;           // 0-3 (which col within tile)
    uint local_k_offset = (simd_lane / 4) % 2; // For K-dimension parallelism

    // Actual output position
    uint out_row = tile_row + local_row;
    uint out_col = tile_col + local_col;

    // Skip if this thread's output is out of bounds
    bool valid = (out_row < M) && (out_col < N);

    float sum = 0.0f;

    if (valid) {
        // Process all K blocks for this (row, col) pair
        // Each pair of lanes processes alternating blocks for better parallelism
        for (uint kb = local_k_offset; kb < num_blocks_k; kb += 2) {
            const device block_q4_K* block = &weights[out_col * num_blocks_k + kb];

            float d = float(block->d);
            float dmin = float(block->dmin);
            const device uint8_t* scales = block->scales;
            const device uint8_t* qs = block->qs;

            uint base_k = kb * QK_K;

            // Process 256 elements in this block
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(is, scales, sc1, m1);
                get_scale_min_k4(is + 1, scales, sc2, m2);

                float d1 = d * float(sc1);
                float dm1 = dmin * float(m1);
                float d2 = d * float(sc2);
                float dm2 = dmin * float(m2);

                uint x_base = out_row * K + base_k + j;

                // Vectorized accumulation using float4
                for (int l = 0; l < 32; l += 4) {
                    float4 x_vec = float4(X[x_base + l], X[x_base + l + 1],
                                          X[x_base + l + 2], X[x_base + l + 3]);
                    float4 w_vec = float4(
                        d1 * float(qs[l] & 0xF) - dm1,
                        d1 * float(qs[l + 1] & 0xF) - dm1,
                        d1 * float(qs[l + 2] & 0xF) - dm1,
                        d1 * float(qs[l + 3] & 0xF) - dm1
                    );
                    sum += dot(x_vec, w_vec);
                }

                for (int l = 0; l < 32; l += 4) {
                    float4 x_vec = float4(X[x_base + 32 + l], X[x_base + 32 + l + 1],
                                          X[x_base + 32 + l + 2], X[x_base + 32 + l + 3]);
                    float4 w_vec = float4(
                        d2 * float(qs[l] >> 4) - dm2,
                        d2 * float(qs[l + 1] >> 4) - dm2,
                        d2 * float(qs[l + 2] >> 4) - dm2,
                        d2 * float(qs[l + 3] >> 4) - dm2
                    );
                    sum += dot(x_vec, w_vec);
                }

                qs += 32;
                is += 2;
            }
        }

        // Reduce partial sums from lanes processing same (row, col) with different K blocks
        // Lanes are paired: 0&4, 1&5, 2&6, 3&7 for row 0, etc.
        float partner_sum = simd_shuffle_xor(sum, 4);
        sum += partner_sum;
    }

    // Write output (only lanes 0-3, 8-11, 16-19, 24-27 write)
    if (valid && (local_k_offset == 0)) {
        Y[out_row * N + out_col] = sum;
    }
}

// =============================================================================
// Highly Optimized Q4_K Matmul with Cooperative Loading
// =============================================================================
// Each SIMD group cooperatively loads input tiles into registers
// Uses wave-level parallelism for maximum throughput

kernel void matmul_q4k_simd_v2(
    device const float* X          [[buffer(0)]],  // Input [M, K]
    device const void* W           [[buffer(1)]],  // Weights [N, K/QK_K] Q4_K blocks
    device float* Y                [[buffer(2)]],  // Output [M, N]
    constant uint& M               [[buffer(3)]],  // Batch size (num tokens)
    constant uint& K               [[buffer(4)]],  // Input dimension
    constant uint& N               [[buffer(5)]],  // Output dimension
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint2 tid                      [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_gid                  [[simdgroup_index_in_threadgroup]]
) {
    // Configuration: 8 simdgroups per threadgroup
    // Each simdgroup computes 2 rows x 8 cols of output
    const uint ROWS_PER_SIMD = 2;
    const uint COLS_PER_SIMD = 8;

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    // This simdgroup's output tile position
    uint base_row = tgid.y * 16 + (simd_gid / 4) * ROWS_PER_SIMD;  // 2 row groups
    uint base_col = tgid.x * 32 + (simd_gid % 4) * COLS_PER_SIMD;  // 4 col groups

    // Each lane computes one element
    uint local_row = simd_lane / 16;     // 0 or 1
    uint local_col = simd_lane % 8;      // 0-7 (lanes 8-15 also map to cols 0-7)
    uint k_parallel = (simd_lane / 8) % 2; // For K parallelism within same output

    uint out_row = base_row + local_row;
    uint out_col = base_col + local_col;

    bool valid = (out_row < M) && (out_col < N);

    float sum = 0.0f;

    if (valid) {
        // Each pair of threads (lane and lane+8) processes different K blocks
        for (uint kb = k_parallel; kb < num_blocks_k; kb += 2) {
            const device block_q4_K* block = &weights[out_col * num_blocks_k + kb];

            float d = float(block->d);
            float dmin = float(block->dmin);
            const device uint8_t* scales = block->scales;
            const device uint8_t* qs = block->qs;

            uint x_row_base = out_row * K + kb * QK_K;

            // Process 256 elements with vectorized loads
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(is, scales, sc1, m1);
                get_scale_min_k4(is + 1, scales, sc2, m2);

                float d1 = d * float(sc1);
                float dm1 = dmin * float(m1);
                float d2 = d * float(sc2);
                float dm2 = dmin * float(m2);

                // Process low nibble (32 elements) with float4 vectorization
                #pragma unroll
                for (int l = 0; l < 32; l += 4) {
                    uint idx = x_row_base + j + l;
                    float4 xv = float4(X[idx], X[idx+1], X[idx+2], X[idx+3]);
                    float4 wv = float4(
                        d1 * float(qs[l] & 0xF) - dm1,
                        d1 * float(qs[l+1] & 0xF) - dm1,
                        d1 * float(qs[l+2] & 0xF) - dm1,
                        d1 * float(qs[l+3] & 0xF) - dm1
                    );
                    sum += dot(xv, wv);
                }

                // Process high nibble (32 elements)
                #pragma unroll
                for (int l = 0; l < 32; l += 4) {
                    uint idx = x_row_base + j + 32 + l;
                    float4 xv = float4(X[idx], X[idx+1], X[idx+2], X[idx+3]);
                    float4 wv = float4(
                        d2 * float(qs[l] >> 4) - dm2,
                        d2 * float(qs[l+1] >> 4) - dm2,
                        d2 * float(qs[l+2] >> 4) - dm2,
                        d2 * float(qs[l+3] >> 4) - dm2
                    );
                    sum += dot(xv, wv);
                }

                qs += 32;
                is += 2;
            }
        }

        // Reduce across K-parallel lanes (lane XOR 8)
        float partner = simd_shuffle_xor(sum, 8);
        sum += partner;
    }

    // Only first set of lanes write (k_parallel == 0)
    if (valid && k_parallel == 0) {
        Y[out_row * N + out_col] = sum;
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

// =============================================================================
// SIMDGROUP MATRIX Q4_K Matmul (Optimized for Prefill)
// =============================================================================
// Uses simdgroup_half8x8 matrices and simdgroup_multiply_accumulate for
// efficient tiled matrix multiplication. This approach can close the 25%
// performance gap with llama.cpp.
//
// Algorithm:
// 1. Dequantize Q4_K blocks into half precision tiles in threadgroup memory
// 2. Use simdgroup_load to load 8x8 tiles into simdgroup matrices
// 3. Use simdgroup_multiply_accumulate for efficient matmul
//
// Tile sizes: 64 rows x 32 cols output per threadgroup
// Each simdgroup handles 32x16 output elements
// =============================================================================

// Helper to extract 6-bit scale and min for dequantization into tiles
inline half2 get_scale_min_k4_h(int j, int k, device const uchar * q) {
    return j < 4 ? half2{half(q[j+0+k] & 63), half(q[j+4+k] & 63)}
                 : half2{half((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                         half((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// Dequantize 16 elements of Q4_K into a 4x4 half matrix
// il selects which 16-element chunk within the 256-element block (0..15)
inline void dequantize_q4_k_to_half4x4(
    device const block_q4_K * xb,
    short il,
    thread half4x4 & reg
) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const half2 sc = get_scale_min_k4_h(is, il/2, xb->scales);
    const half d   = il < 2 ? xb->d : xb->d / 16.h;
    const half min = xb->dmin;
    const half dl = d * sc[0];
    const half ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * half(q[i] & mask) - ml;
    }
}

// Configuration for simdgroup matmul kernel
constant constexpr uint SGMM_NR0 = 64;   // Output rows per threadgroup
constant constexpr uint SGMM_NR1 = 32;   // Output cols per threadgroup
constant constexpr uint SGMM_NK  = 32;   // K-dimension per iteration

kernel void matmul_q4k_simdgroup(
    device const float* X          [[buffer(0)]],  // Input [M, K]
    device const void* W           [[buffer(1)]],  // Weights [N, K/QK_K] Q4_K blocks
    device float* Y                [[buffer(2)]],  // Output [M, N]
    constant uint& M               [[buffer(3)]],  // Batch size (num tokens)
    constant uint& K               [[buffer(4)]],  // Input dimension
    constant uint& N               [[buffer(5)]],  // Output dimension
    threadgroup char* shmem        [[threadgroup(0)]],  // Shared memory (8192 bytes)
    uint3 tgpig                    [[threadgroup_position_in_grid]],
    ushort tiitg                   [[thread_index_in_threadgroup]],
    ushort sgitg                   [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory layout:
    // sa: 64 x 32 half for A tiles (dequantized weights)  = 4096 bytes
    // sb: 32 x 32 half for B tiles (input activations)    = 2048 bytes
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    // Output tile position
    const uint r0 = tgpig.y * SGMM_NR0;  // Output row start
    const uint r1 = tgpig.x * SGMM_NR1;  // Output col start (which N dimension)

    // Bounds check
    const short nr0 = min((uint)SGMM_NR0, M - r0);
    const short nr1 = min((uint)SGMM_NR1, N - r1);

    // Thread's loading position
    const short NL0 = SGMM_NK / 16;  // 2 - number of 16-element chunks per thread
    const short NL1 = SGMM_NK / 8;   // 4 - for loading input

    const short lr0 = min((short)(tiitg/NL0), (short)(nr0 - 1));  // Which output row
    const short lr1 = min((short)(tiitg/NL1), (short)(nr1 - 1));  // Which output col

    const short il0 = tiitg % NL0;  // Which 16-element chunk within block

    // Simdgroup accumulator matrices (8 x 8x8 = 64 output elements per simdgroup)
    simdgroup_half8x8 ma[4];   // Weight tiles
    simdgroup_half8x8 mb[2];   // Input tiles
    simdgroup_float8x8 mc[8];  // Accumulators (in float for precision)

    // Initialize accumulators to zero
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = il0;

    // Process K dimension in chunks of SGMM_NK
    for (uint loop_k = 0; loop_k < K; loop_k += SGMM_NK) {
        // Determine which Q4_K block we're in
        uint block_idx = (loop_k + 16 * il) / QK_K;
        uint within_block = ((loop_k + 16 * il) % QK_K) / 16;  // Which 16-element within block

        // Load and dequantize weights into threadgroup memory
        // Each thread loads one 4x4 tile (16 elements)
        if (lr0 < nr0 && block_idx < num_blocks_k) {
            // Weight matrix: W[N, K] - for output row (r1 + lr0), we need weight row
            // Actually for Y = X @ W^T, we need W row corresponding to output dimension
            device const block_q4_K* w_block = &weights[(r1 + lr0) * num_blocks_k + block_idx];

            half4x4 temp_a;
            dequantize_q4_k_to_half4x4(w_block, within_block, temp_a);

            // Store to threadgroup memory in a layout suitable for simdgroup_load
            // We want 8x8 tiles for simdgroup operations
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i/8;
                const short sy = lr0 / 8;
                const short lx = lr0 % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;

                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i/4][i%4];
            }
        }

        // Load input activations into threadgroup memory
        // X[M, K] - for batch index (r0 + lr1), load K elements
        if (lr1 < nr1 && loop_k + 8 * (tiitg % NL1) < K) {
            const short iy = 8 * (tiitg % NL1);
            device const float* x_ptr = X + (r0 + lr1) * K + loop_k + iy;

            // Load 8 elements and convert to half
            for (short i = 0; i < 8; ++i) {
                const short sx = tiitg % NL1;
                const short sy = lr1 / 8;
                const short lx = i;
                const short ly = lr1 % 8;
                const short ib = 4 * sx + sy;

                *(sb + 64 * ib + 8 * ly + lx) = loop_k + iy + i < K ?
                    half(x_ptr[i]) : half(0);
            }
        }

        // Advance within Q4_K block
        il = (il + 2 < 16) ? il + 2 : il % 2;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Perform simdgroup matrix multiply-accumulate
        // Each simdgroup computes a 32x16 output tile
        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < SGMM_NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            // Load 4 x 8x8 weight tiles
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            // Load 2 x 8x8 input tiles
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            // Multiply-accumulate: C += B * A^T
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // Write results to global memory
    device float* C = Y + (r0 + 32 * (sgitg & 1)) + (r1 + 16 * (sgitg >> 1)) * M;

    // Check bounds and store
    if (r0 + 32 * (sgitg & 1) + 32 <= M && r1 + 16 * (sgitg >> 1) + 16 <= N) {
        // Full tile - can store directly
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * M * (i / 4), M, 0, false);
        }
    } else {
        // Partial tile - need bounds checking
        // Use threadgroup memory as intermediate
        threadgroup float * sc = (threadgroup float *)(shmem);

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], sc + 64 * i, 8, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread copies its element if in bounds
        const short local_row = tiitg % 32;
        const short local_col = tiitg / 32;
        const uint out_row = r0 + 32 * (sgitg & 1) + local_row;
        const uint out_col = r1 + 16 * (sgitg >> 1) + local_col;

        if (out_row < M && out_col < N && local_col < 16) {
            Y[out_row + out_col * M] = sc[local_row + 8 * local_col];
        }
    }
}
