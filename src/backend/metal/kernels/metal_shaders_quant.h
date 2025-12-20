// =============================================================================
// Metal Shader Quantized Kernels
// =============================================================================
// Quantized matrix-vector (decode) and matrix-matrix (prefill) kernels for all
// supported quantization formats: Q4_K, Q8_0, Q4_0, IQ4_NL, IQ4_XS, IQ3_S,
// Q6_K, Q3_K, Q2_K, Q5_K, F16.
// =============================================================================

#pragma once

static const char* METAL_SHADER_QUANT = R"(
// =============================================================================
// OPTIMIZED MATVEC FOR Q4_K (llama.cpp style)
// =============================================================================
//
// This is the primary decode kernel for Q4_K quantized models. It implements
// y = x @ W^T where W is Q4_K quantized [N, K] and x is FP32 [K].
//
// THREAD INDEXING (32 threads per SIMD group):
// -----------------------------------------------------------------------------
//   tiisg = 0..31   (thread index in SIMD group)
//   ix = tiisg / 8  (0..3) - which quarter of the 256-element super-block
//   it = tiisg % 8  (0..7) - position within quarter
//   iq = it / 4     (0..1) - which half of 64-element sub-block
//   ir = it % 4     (0..3) - position within 16-element chunk
//
// MEMORY ACCESS PATTERN:
// -----------------------------------------------------------------------------
//   - Each SIMD group processes 2 output rows (NR0_Q4K = 2)
//   - 4 threads cooperate on each Q4_K super-block (stride of 4)
//   - Input values are cached in registers (32 floats per thread)
//   - Weights are read as uint16 for better memory bandwidth
//   - No threadgroup memory or barriers (register-only computation)
//
// PERFORMANCE CHARACTERISTICS:
// -----------------------------------------------------------------------------
//   - Memory bound: ~144 bytes weight + 1024 bytes input per 256 MACs
//   - Arithmetic intensity: ~0.22 FLOP/byte (very low)
//   - Target efficiency: >50% of theoretical bandwidth
//   - Threadgroup: 64 threads (2 SIMD groups) for better occupancy
//
// SCALE EXTRACTION:
// -----------------------------------------------------------------------------
//   The Q4_K scales are packed in a complex 6-bit format. We use bitmasks
//   to extract them efficiently:
//   - kmask1 (0x3f3f): Extract low 6 bits from each byte
//   - kmask2 (0x0f0f): Extract low 4 bits from each byte
//   - kmask3 (0xc0c0): Extract high 2 bits from each byte
//
// =============================================================================

constant constexpr short NR0_Q4K = 2;  // Rows per SIMD group
constant constexpr uint ROWS_PER_TG = 2;  // SIMD groups per threadgroup for Q4_K

// Bitmasks for extracting 6-bit scales from packed format
constant constexpr ushort kmask1 = 0x3f3f;  // Low 6 bits of each byte
constant constexpr ushort kmask2 = 0x0f0f;  // Low 4 bits of each byte
constant constexpr ushort kmask3 = 0xc0c0;  // High 2 bits of each byte

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

// Fused Q/K/V projection kernel - processes all 3 attention projections in single dispatch
// This reduces kernel launch overhead from 3 dispatches to 1
kernel void fused_qkv_matvec_q4k(
    device const float* x           [[buffer(0)]],   // Input hidden state [hidden_dim]
    device const void* Wq           [[buffer(1)]],   // Q weight [q_dim, hidden_dim]
    device const void* Wk           [[buffer(2)]],   // K weight [kv_dim, hidden_dim]
    device const void* Wv           [[buffer(3)]],   // V weight [kv_dim, hidden_dim]
    device float* yq                [[buffer(4)]],   // Q output [q_dim]
    device float* yk                [[buffer(5)]],   // K output [kv_dim]
    device float* yv                [[buffer(6)]],   // V output [kv_dim]
    constant uint& K                [[buffer(7)]],   // Input dimension (hidden_dim)
    constant uint& Nq               [[buffer(8)]],   // Q output dimension (num_heads * head_dim)
    constant uint& Nkv              [[buffer(9)]],   // KV output dimension (num_kv_heads * head_dim)
    constant uint& q_threadgroups   [[buffer(10)]],  // Number of threadgroups for Q
    constant uint& kv_threadgroups  [[buffer(11)]],  // Number of threadgroups for K (and V)
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tiisg                      [[thread_index_in_simdgroup]],
    uint sgitg                      [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK_K;

    // Determine which output (Q=0, K=1, V=2) and local row offset
    uint which_output;
    uint local_tgid;
    uint N;
    device const void* W;
    device float* y;

    if (tgid < q_threadgroups) {
        which_output = 0;
        local_tgid = tgid;
        N = Nq;
        W = Wq;
        y = yq;
    } else if (tgid < q_threadgroups + kv_threadgroups) {
        which_output = 1;
        local_tgid = tgid - q_threadgroups;
        N = Nkv;
        W = Wk;
        y = yk;
    } else {
        which_output = 2;
        local_tgid = tgid - q_threadgroups - kv_threadgroups;
        N = Nkv;
        W = Wv;
        y = yv;
    }

    // Thread indexing (same as matvec_q4k)
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    // Each SIMD group handles NR0_Q4K rows
    const uint first_row = (local_tgid * ROWS_PER_TG + sgitg) * NR0_Q4K;

    device const block_q4_K* weights = (const device block_q4_K*)W;

    // Register-based input caching
    float yl[16];
    float yh[16];
    float sumf[NR0_Q4K] = {0.f, 0.f};

    device const float* y4 = x + ix * QK_K + 64 * iq + 8 * ir;

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        for (short row = 0; row < NR0_Q4K; row++) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_q4_K* block = &weights[row_idx * nb + ib];
            device const ushort* sc = (device const ushort*)block->scales + iq;
            device const ushort* q1 = (device const ushort*)block->qs + 16 * iq + 4 * ir;
            device const half* dh = &block->d;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const ushort* q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

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

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                  sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * QK_K;
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

// =============================================================================
// Vectorized Q4_K Matrix Multiplication (Prefill)
// =============================================================================
// Same 1-thread-per-output model as original, but with float4 vectorized X access
// This is a safe optimization that doesn't change parallelization

kernel void matmul_q4k_vec(
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
        uint base_idx = row * K + kb * QK_K;

        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is, scales, sc1, m1);
            get_scale_min_k4(is + 1, scales, sc2, m2);

            float d1 = d * float(sc1);
            float dm1 = dmin * float(m1);
            float d2 = d * float(sc2);
            float dm2 = dmin * float(m2);

            uint x_off = base_idx + j;

            // Vectorized low nibble (4 elements at a time)
            for (int l = 0; l < 32; l += 4) {
                float4 xv = float4(X[x_off+l], X[x_off+l+1], X[x_off+l+2], X[x_off+l+3]);
                float4 wv = float4(d1*float(qs[l]&0xF)-dm1, d1*float(qs[l+1]&0xF)-dm1,
                                   d1*float(qs[l+2]&0xF)-dm1, d1*float(qs[l+3]&0xF)-dm1);
                sum += dot(xv, wv);
            }

            // Vectorized high nibble (4 elements at a time)
            for (int l = 0; l < 32; l += 4) {
                float4 xv = float4(X[x_off+32+l], X[x_off+32+l+1], X[x_off+32+l+2], X[x_off+32+l+3]);
                float4 wv = float4(d2*float(qs[l]>>4)-dm2, d2*float(qs[l+1]>>4)-dm2,
                                   d2*float(qs[l+2]>>4)-dm2, d2*float(qs[l+3]>>4)-dm2);
                sum += dot(xv, wv);
            }

            qs += 32;
            is += 2;
        }
    }

    Y[row * N + col] = sum;
}

// =============================================================================
// Register-Tiled Q4_K Matrix Multiplication (Prefill)
// =============================================================================
// Each thread computes 2 output rows, amortizing weight loading overhead
// Also uses float4 vectorized X access

kernel void matmul_q4k_tiled(
    device const float* X          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* Y                [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    // Each thread handles 2 consecutive rows
    uint row0 = gid.y * 2;
    uint row1 = row0 + 1;
    uint col = gid.x;

    if (row0 >= M || col >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    // Register tiling: accumulate 2 rows at once
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    bool valid1 = (row1 < M);

    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q4_K* block = &weights[col * num_blocks_k + kb];
        float d = float(block->d);
        float dmin = float(block->dmin);
        const device uint8_t* scales = block->scales;
        const device uint8_t* qs = block->qs;
        uint k_base = kb * QK_K;

        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is, scales, sc1, m1);
            get_scale_min_k4(is + 1, scales, sc2, m2);

            float d1 = d * float(sc1);
            float dm1 = dmin * float(m1);
            float d2 = d * float(sc2);
            float dm2 = dmin * float(m2);

            uint x0_off = row0 * K + k_base + j;
            uint x1_off = row1 * K + k_base + j;

            // Low nibble - 4 at a time
            for (int l = 0; l < 32; l += 4) {
                uint8_t q0 = qs[l], q1 = qs[l+1], q2 = qs[l+2], q3 = qs[l+3];
                float4 wv = float4(d1*float(q0&0xF)-dm1, d1*float(q1&0xF)-dm1,
                                   d1*float(q2&0xF)-dm1, d1*float(q3&0xF)-dm1);

                float4 xv0 = float4(X[x0_off+l], X[x0_off+l+1], X[x0_off+l+2], X[x0_off+l+3]);
                sum0 += dot(xv0, wv);

                if (valid1) {
                    float4 xv1 = float4(X[x1_off+l], X[x1_off+l+1], X[x1_off+l+2], X[x1_off+l+3]);
                    sum1 += dot(xv1, wv);
                }
            }

            // High nibble - 4 at a time
            for (int l = 0; l < 32; l += 4) {
                uint8_t q0 = qs[l], q1 = qs[l+1], q2 = qs[l+2], q3 = qs[l+3];
                float4 wv = float4(d2*float(q0>>4)-dm2, d2*float(q1>>4)-dm2,
                                   d2*float(q2>>4)-dm2, d2*float(q3>>4)-dm2);

                float4 xv0 = float4(X[x0_off+32+l], X[x0_off+32+l+1], X[x0_off+32+l+2], X[x0_off+32+l+3]);
                sum0 += dot(xv0, wv);

                if (valid1) {
                    float4 xv1 = float4(X[x1_off+32+l], X[x1_off+32+l+1], X[x1_off+32+l+2], X[x1_off+32+l+3]);
                    sum1 += dot(xv1, wv);
                }
            }

            qs += 32;
            is += 2;
        }
    }

    Y[row0 * N + col] = sum0;
    if (valid1) {
        Y[row1 * N + col] = sum1;
    }
}

// =============================================================================
// SIMD Group Matrix Q4_K Matmul (Prefill - K-Parallel)
// =============================================================================
// Simple approach: 4 rows per simdgroup with 8-way K parallelism
// Each simdgroup handles 4 rows x 1 column

kernel void matmul_q4k_simd(
    device const float* X          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* Y                [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    uint2 tgpig                    [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]]
) {
    // Each simdgroup (threadgroup) handles 4 rows x 1 column
    // tgpig.x = column index, tgpig.y = row group index
    uint row_base = tgpig.y * 4;
    uint col = tgpig.x;

    if (col >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    // Each thread accumulates for one row (lanes 0-7 -> row 0, etc.)
    uint local_row = simd_lane / 8;
    uint k_lane = simd_lane % 8;  // 8-way K parallelism

    uint row = row_base + local_row;
    float sum = 0.0f;

    if (row < M) {
        for (uint kb = 0; kb < num_blocks_k; kb++) {
            const device block_q4_K* block = &weights[col * num_blocks_k + kb];
            float d = float(block->d);
            float dmin = float(block->dmin);
            const device uint8_t* scales = block->scales;
            const device uint8_t* qs = block->qs;

            // Each lane processes 32 elements of the 256-element block
            // k_lane 0: elements 0-31, k_lane 1: 32-63, etc.
            uint elem_start = k_lane * 32;

            // Determine scale index based on which 64-element group we're in
            int is = (elem_start / 64) * 2;
            if (elem_start % 64 >= 32) is++;

            uint8_t sc1, m1;
            get_scale_min_k4(is, scales, sc1, m1);
            float d1 = d * float(sc1);
            float dm1 = dmin * float(m1);

            // qs base: each 64-element group uses 32 bytes
            uint qs_base = (elem_start / 64) * 32;
            bool high = (elem_start % 64) >= 32;

            for (uint l = 0; l < 32; l += 4) {
                uint k_idx = kb * QK_K + elem_start + l;
                if (k_idx + 3 < K) {
                    float4 xv = float4(X[row * K + k_idx], X[row * K + k_idx + 1],
                                       X[row * K + k_idx + 2], X[row * K + k_idx + 3]);
                    uint8_t q0 = qs[qs_base + l];
                    uint8_t q1 = qs[qs_base + l + 1];
                    uint8_t q2 = qs[qs_base + l + 2];
                    uint8_t q3 = qs[qs_base + l + 3];

                    float4 wv;
                    if (high) {
                        wv = float4(d1 * float(q0 >> 4) - dm1, d1 * float(q1 >> 4) - dm1,
                                    d1 * float(q2 >> 4) - dm1, d1 * float(q3 >> 4) - dm1);
                    } else {
                        wv = float4(d1 * float(q0 & 0xF) - dm1, d1 * float(q1 & 0xF) - dm1,
                                    d1 * float(q2 & 0xF) - dm1, d1 * float(q3 & 0xF) - dm1);
                    }
                    sum += dot(xv, wv);
                }
            }
        }
    }

    // Reduce across the 8 K-parallel lanes for each row
    for (uint offset = 4; offset > 0; offset /= 2) {
        sum += simd_shuffle_xor(sum, offset);
    }

    // Lane 0, 8, 16, 24 write the result for their respective rows
    if (k_lane == 0 && row < M) {
        Y[row * N + col] = sum;
    }
}

// =============================================================================
// SIMDGROUP MATRIX Q4_K Matmul (Optimized for Prefill)
// =============================================================================
// Uses simdgroup_half8x8 matrices and simdgroup_multiply_accumulate for
// efficient tiled matrix multiplication following llama.cpp's approach.
//
// Computes: Y[M, N] = X[M, K] @ W[N, K]^T
//
// Algorithm:
// 1. Dequantize Q4_K weight blocks into half precision in threadgroup memory (sa)
// 2. Load input activations into threadgroup memory (sb)
// 3. Use simdgroup matrices for tiled multiply-accumulate
// 4. Write results with correct row-major layout
//
// Tile sizes: 64 output-N x 32 output-M per threadgroup
// Each simdgroup handles 32x16 output elements
// 4 simdgroups (128 threads) per threadgroup
// =============================================================================

// Helper to extract 6-bit scale and min for dequantization
inline half2 get_scale_min_k4_h(int j, int k, device const uint8_t* q) {
    return j < 4 ? half2{half(q[j+0+k] & 63), half(q[j+4+k] & 63)}
                 : half2{half((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                         half((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// Dequantize 16 elements of Q4_K into a 4x4 half matrix
// il selects which 16-element chunk within the 256-element block (0..15)
inline void dequantize_q4_k_to_half4x4(
    device const block_q4_K* xb,
    short il,
    thread half4x4& reg
) {
    device const uint8_t* q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const half2 sc = get_scale_min_k4_h(is, il/2, xb->scales);
    const half d   = il < 2 ? xb->d : xb->d / half(16);
    const half min = xb->dmin;
    const half dl = d * sc[0];
    const half ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * half(q[i] & mask) - ml;
    }
}

// Configuration - matches llama.cpp kernel_mul_mm
constant constexpr uint SGMM_NR0 = 64;   // Weight rows (N output) per threadgroup
constant constexpr uint SGMM_NR1 = 32;   // Input rows (M output) per threadgroup
constant constexpr uint SGMM_NK  = 32;   // K-dimension per iteration

kernel void matmul_q4k_simdgroup(
    device const float* X          [[buffer(0)]],  // Input [M, K] row-major
    device const void* W           [[buffer(1)]],  // Weights [N, K/256] Q4_K blocks
    device float* Y                [[buffer(2)]],  // Output [M, N] row-major
    constant uint& M               [[buffer(3)]],  // Batch size (num tokens)
    constant uint& K               [[buffer(4)]],  // Input dimension
    constant uint& N               [[buffer(5)]],  // Output dimension
    threadgroup char* shmem        [[threadgroup(0)]],  // Shared memory (8192 bytes)
    uint3 tgpig                    [[threadgroup_position_in_grid]],
    ushort tiitg                   [[thread_index_in_threadgroup]],
    ushort sgitg                   [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory layout (following llama.cpp):
    // sa: NR0 x NK half for weight tiles (64 x 32 = 4096 bytes)
    // sb: NR1 x NK half for input tiles (32 x 32 = 2048 bytes)
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    // Grid assignment (following llama.cpp exactly):
    // r0 = weight rows (N output dimension)
    // r1 = input rows (M output dimension)
    const uint r0 = tgpig.y * SGMM_NR0;  // N dimension start
    const uint r1 = tgpig.x * SGMM_NR1;  // M dimension start

    // Bounds
    const short nr0 = min((uint)SGMM_NR0, N - r0);  // Actual N rows to process
    const short nr1 = min((uint)SGMM_NR1, M - r1);  // Actual M rows to process

    // Thread assignment for loading
    const short NL0 = SGMM_NK / 16;  // 2 chunks per thread for weights
    const short NL1 = SGMM_NK / 8;   // 4 chunks per thread for inputs

    // Which weight row (0..63) and input row (0..31) this thread loads
    const short lr0 = min((short)(tiitg / NL0), (short)(nr0 - 1));  // Weight row
    const short lr1 = min((short)(tiitg / NL1), (short)(nr1 - 1));  // Input row

    const short il0 = tiitg % NL0;  // Which 16-elem chunk within Q4_K block

    // Simdgroup accumulator matrices
    simdgroup_half8x8 ma[4];   // Weight tiles
    simdgroup_half8x8 mb[2];   // Input tiles
    simdgroup_float8x8 mc[8];  // Accumulators

    // Initialize accumulators
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    const short iy = 8 * (tiitg % NL1);  // K offset for input loading

    // Pointer-based iteration matching llama.cpp pattern
    // nl = number of 16-element chunks per Q4_K block (256/16 = 16)
    constexpr short nl = QK_K / 16;  // 16 for Q4_K
    short il = il0;  // Position within Q4_K block (0..15)

    // Initial weight pointer for this thread's row
    device const block_q4_K* x = (lr0 < nr0) ?
        &weights[(r0 + lr0) * num_blocks_k + il0/nl] : nullptr;

    // Input pointer for this thread's row
    device const float* y = (lr1 < nr1) ? X + (r1 + lr1) * K + iy : nullptr;

    // Process K dimension in chunks
    for (uint loop_k = 0; loop_k < K; loop_k += SGMM_NK) {
        // Load and dequantize weights into sa
        half4x4 temp_a;
        if (x != nullptr) {
            dequantize_q4_k_to_half4x4(x, il, temp_a);
        } else {
            temp_a = half4x4(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Store weight tile - exactly matching llama.cpp pattern
        // Uses FOR_UNROLL and pointer arithmetic (NOT array indexing which is slower)
        #pragma clang loop unroll(full)
        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (short)(tiitg/NL0)/8;
            const short lx = (short)(tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
        }

        // Load input into sb - llama.cpp style with half2x4 cast
        if (y != nullptr) {
            const short sx = tiitg % NL1;       // K block index
            const short sy = lr1 / 8;           // M block index
            const short ly = lr1 % 8;           // Within M block
            const short ib = 4 * sx + sy;

            // Single vector load with type cast (matches llama.cpp pattern)
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) = half2x4(*(device const float2x4*)y);
        }

        // Advance pointers for next iteration (matching llama.cpp exactly)
        // il advances by 2 each iteration (we load 32 elements = 2 chunks of 16)
        il = (il + 2 < nl) ? il + 2 : il % 2;
        // x advances when we cross a block boundary (when il wraps to < 2)
        x = (il < 2 && x != nullptr) ? x + 1 : x;
        // y advances by NK elements each iteration
        y = (y != nullptr) ? y + SGMM_NK : nullptr;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Simdgroup matrix multiply-accumulate
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);  // Weight tiles
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);  // Input tiles

        #pragma clang loop unroll(full)
        for (short ik = 0; ik < SGMM_NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma clang loop unroll(full)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // Write results to global memory
    // Output Y[M, N] row-major: Y[m, n] at address m * N + n
    // This simdgroup computed output for:
    //   N rows: r0 + 32*(sgitg%2) .. r0 + 32*(sgitg%2) + 31
    //   M rows: r1 + 16*(sgitg/2) .. r1 + 16*(sgitg/2) + 15
    // But we need to transpose because mc = input @ weight gives [M_tile, N_tile]

    const uint sg_n = r0 + 32 * (sgitg % 2);   // N start for this simdgroup
    const uint sg_m = r1 + 16 * (sgitg / 2);   // M start for this simdgroup

    // The mc matrices contain [M_local, N_local] data
    // mc[i] where i/4 selects M block, i%4 selects N block
    // So mc[0..3] are M rows 0-7 with N cols 0-7, 8-15, 16-23, 24-31
    //    mc[4..7] are M rows 8-15 with N cols 0-7, 8-15, 16-23, 24-31

    if (sg_m + 16 <= M && sg_n + 32 <= N) {
        // Full tile - store with row-major layout (matching llama.cpp)
        device float* C = Y + sg_m * N + sg_n;

        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*N*(i/4), N, 0, false);
        }
    } else {
        // Partial tile - use threadgroup memory as intermediate
        threadgroup float* sc = (threadgroup float*)(shmem);

        // Store all mc matrices to threadgroup memory with stride 8
        simdgroup_store(mc[0], sc + 64 * 0, 8, 0, false);
        simdgroup_store(mc[1], sc + 64 * 1, 8, 0, false);
        simdgroup_store(mc[2], sc + 64 * 2, 8, 0, false);
        simdgroup_store(mc[3], sc + 64 * 3, 8, 0, false);
        simdgroup_store(mc[4], sc + 64 * 4, 8, 0, false);
        simdgroup_store(mc[5], sc + 64 * 5, 8, 0, false);
        simdgroup_store(mc[6], sc + 64 * 6, 8, 0, false);
        simdgroup_store(mc[7], sc + 64 * 7, 8, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread copies elements that are in bounds
        // 128 threads, need to cover 16x32 = 512 elements
        for (uint idx = tiitg; idx < 16 * 32; idx += 128) {
            const uint local_m = idx / 32;  // 0..15
            const uint local_n = idx % 32;  // 0..31
            const uint out_m = sg_m + local_m;
            const uint out_n = sg_n + local_n;

            if (out_m < M && out_n < N) {
                // Map to sc index
                const uint m_block = local_m / 8;    // 0 or 1
                const uint n_block = local_n / 8;    // 0..3
                const uint in_m = local_m % 8;
                const uint in_n = local_n % 8;
                const uint tile_idx = m_block * 4 + n_block;

                Y[out_m * N + out_n] = sc[tile_idx * 64 + in_m * 8 + in_n];
            }
        }
    }
}

// =============================================================================
// FAST SIMDGROUP MATRIX Q4_K Matmul (Fully Aligned Dimensions)
// =============================================================================
// Same as matmul_q4k_simdgroup but ALL conditionals removed.
// REQUIRES: M % 32 == 0 && N % 64 == 0 && K % 32 == 0
// This eliminates ALL bounds checking and conditional branches.
// =============================================================================

kernel void matmul_q4k_simdgroup_fast(
    device const float* X          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* Y                [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    threadgroup char* shmem        [[threadgroup(0)]],
    uint3 tgpig                    [[threadgroup_position_in_grid]],
    ushort tiitg                   [[thread_index_in_threadgroup]],
    ushort sgitg                   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    const uint num_blocks_k = K / QK_K;
    const device block_q4_K* weights = (const device block_q4_K*)W;

    const uint r0 = tgpig.y * SGMM_NR0;
    const uint r1 = tgpig.x * SGMM_NR1;

    // No bounds checking - dimensions are aligned
    constexpr short NL0 = SGMM_NK / 16;
    constexpr short NL1 = SGMM_NK / 8;

    const short lr0 = tiitg / NL0;  // No min() needed
    const short lr1 = tiitg / NL1;  // No min() needed
    const short il0 = tiitg % NL0;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    const short iy = 8 * (tiitg % NL1);

    constexpr short nl = QK_K / 16;
    short il = il0;

    // Direct pointer assignment - no nullptr check
    device const block_q4_K* x = &weights[(r0 + lr0) * num_blocks_k + il0/nl];
    device const float* y = X + (r1 + lr1) * K + iy;

    for (uint loop_k = 0; loop_k < K; loop_k += SGMM_NK) {
        // No nullptr check - always dequantize
        half4x4 temp_a;
        dequantize_q4_k_to_half4x4(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (short)(tiitg/NL0)/8;
            const short lx = (short)(tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
        }

        // No nullptr check - always load
        {
            const short sx = tiitg % NL1;
            const short sy = lr1 / 8;
            const short ly = lr1 % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) = half2x4(*(device const float2x4*)y);
        }

        // Simplified pointer advance - no nullptr checks
        il = (il + 2 < nl) ? il + 2 : il % 2;
        x = (il < 2) ? x + 1 : x;
        y = y + SGMM_NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

        #pragma clang loop unroll(full)
        for (short ik = 0; ik < SGMM_NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma clang loop unroll(full)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // Direct store - no bounds checking
    const uint sg_n = r0 + 32 * (sgitg % 2);
    const uint sg_m = r1 + 16 * (sgitg / 2);
    device float* C = Y + sg_m * N + sg_n;

    #pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], C + 8*(i%4) + 8*N*(i/4), N, 0, false);
    }
}

// =============================================================================
// Q8_0 Matrix-Vector Multiplication (Single Token Decode)
// =============================================================================
// Optimized using same techniques as Q4_K:
// - SIMD groups for parallel reduction
// - Register-based input caching
// - 2 rows per SIMD group

constant constexpr short NR0_Q8_0 = 2;  // Rows per SIMD group
constant constexpr uint Q8_0_ROWS_PER_TG = 8;  // SIMD groups per threadgroup

kernel void matvec_q8_0(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK8_0;  // Number of Q8_0 blocks per row

    // Each SIMD group handles NR0_Q8_0 rows
    const uint first_row = (tgid * Q8_0_ROWS_PER_TG + sgitg) * NR0_Q8_0;

    device const block_q8_0* weights = (const device block_q8_0*)W;

    float sumf[NR0_Q8_0] = {0.f, 0.f};

    // Thread indexing: 32 threads per SIMD group
    // Each thread processes every 32nd block
    for (uint ib = tiisg; ib < nb; ib += 32) {
        // Cache input values in registers
        float xl[32];
        uint x_base = ib * QK8_0;
        for (int i = 0; i < 32; i++) {
            xl[i] = x[x_base + i];
        }

        // Process each row
        for (short row = 0; row < NR0_Q8_0; row++) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_q8_0* block = &weights[row_idx * nb + ib];
            float d = float(block->d);

            // Vectorized accumulation
            float acc = 0.f;
            for (int i = 0; i < 32; i += 4) {
                acc += xl[i+0] * float(block->qs[i+0]);
                acc += xl[i+1] * float(block->qs[i+1]);
                acc += xl[i+2] * float(block->qs[i+2]);
                acc += xl[i+3] * float(block->qs[i+3]);
            }
            sumf[row] += d * acc;
        }
    }

    // SIMD reduction and output
    for (short row = 0; row < NR0_Q8_0; row++) {
        uint row_idx = first_row + row;
        if (row_idx < N) {
            float sum = simd_sum(sumf[row]);
            if (tiisg == 0) {
                y[row_idx] = sum;
            }
        }
    }
}

// Q8_0 Matrix Multiplication (Batched - Prefill)
kernel void matmul_q8_0(
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

    const uint num_blocks_k = K / QK8_0;
    const device block_q8_0* weights = (const device block_q8_0*)W;

    float sum = 0.0f;

    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q8_0* block = &weights[col * num_blocks_k + kb];
        float d = float(block->d);

        uint base_idx = kb * QK8_0;

        for (int i = 0; i < 32; i++) {
            float w = d * float(block->qs[i]);
            sum += X[row * K + base_idx + i] * w;
        }
    }

    Y[row * N + col] = sum;
}

// Fused RMSNorm + Q8_0 MatVec
// Computes: y = RMSNorm(x, weight) @ W^T (Q8_0)
kernel void rms_norm_matvec_q8_0(
    device const float* x          [[buffer(0)]],
    device const half* norm_weight [[buffer(1)]],
    device const void* W           [[buffer(2)]],
    device float* y                [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
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

    // Step 2: Q8_0 MatVec
    uint row = tgid * 8 + simd_id;
    if (row >= N) return;

    const uint num_blocks_k = K / QK8_0;
    const device block_q8_0* weights = (const device block_q8_0*)W;

    float local_sum = 0.0f;

    for (uint kb = simd_lane; kb < num_blocks_k; kb += 32) {
        const device block_q8_0* block = &weights[row * num_blocks_k + kb];
        float d = float(block->d);

        uint base_idx = kb * QK8_0;
        float acc = 0.f;
        for (int i = 0; i < 32; i++) {
            acc += x_norm[base_idx + i] * float(block->qs[i]);
        }
        local_sum += d * acc;
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row] = sum;
    }
}

// =============================================================================
// Q4_0 Matrix-Vector Multiplication (Single Token Decode)
// =============================================================================
// Q4_0: 32 elements per block, 18 bytes (half d + 16 bytes qs)
// Dequantization: w = d * (q - 8) where q is 4-bit unsigned (0-15)

constant constexpr uint QK4_0 = 32;  // Q4_0 block size
constant constexpr short NR0_Q4_0 = 2;  // Rows per SIMD group
constant constexpr uint Q4_0_ROWS_PER_TG = 8;  // SIMD groups per threadgroup

kernel void matvec_q4_0(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK4_0;  // Number of Q4_0 blocks per row

    // Each SIMD group handles NR0_Q4_0 rows
    const uint first_row = (tgid * Q4_0_ROWS_PER_TG + sgitg) * NR0_Q4_0;

    device const block_q4_0* weights = (const device block_q4_0*)W;

    float sumf[NR0_Q4_0] = {0.f, 0.f};

    // Thread indexing: 32 threads per SIMD group
    // Each thread processes every 32nd block
    for (uint ib = tiisg; ib < nb; ib += 32) {
        // Cache input values in registers
        float xl[32];
        uint x_base = ib * QK4_0;
        for (int i = 0; i < 32; i++) {
            xl[i] = x[x_base + i];
        }

        // Process each row
        for (short row = 0; row < NR0_Q4_0; row++) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_q4_0* block = &weights[row_idx * nb + ib];
            float d = float(block->d);

            // Dequantize and accumulate
            // Q4_0: 2 values per byte, low nibble first
            float acc = 0.f;
            for (int i = 0; i < 16; i++) {
                uint8_t qbyte = block->qs[i];
                int q0 = (qbyte & 0xF) - 8;  // low nibble
                int q1 = (qbyte >> 4) - 8;   // high nibble
                acc += xl[i*2 + 0] * float(q0);
                acc += xl[i*2 + 1] * float(q1);
            }
            sumf[row] += d * acc;
        }
    }

    // SIMD reduction and output
    for (short row = 0; row < NR0_Q4_0; row++) {
        uint row_idx = first_row + row;
        if (row_idx < N) {
            float sum = simd_sum(sumf[row]);
            if (tiisg == 0) {
                y[row_idx] = sum;
            }
        }
    }
}

// Q4_0 Matrix Multiplication (Batched - Prefill)
kernel void matmul_q4_0(
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

    const uint num_blocks_k = K / QK4_0;
    const device block_q4_0* weights = (const device block_q4_0*)W;

    float sum = 0.0f;

    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q4_0* block = &weights[col * num_blocks_k + kb];
        float d = float(block->d);

        uint base_idx = kb * QK4_0;

        for (int i = 0; i < 16; i++) {
            uint8_t qbyte = block->qs[i];
            int q0 = (qbyte & 0xF) - 8;
            int q1 = (qbyte >> 4) - 8;
            sum += d * float(q0) * X[row * K + base_idx + i*2 + 0];
            sum += d * float(q1) * X[row * K + base_idx + i*2 + 1];
        }
    }

    Y[row * N + col] = sum;
}

// Fused RMSNorm + Q4_0 MatVec
kernel void rms_norm_matvec_q4_0(
    device const float* x          [[buffer(0)]],
    device const half* norm_weight [[buffer(1)]],
    device const void* W           [[buffer(2)]],
    device float* y                [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
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

    // Step 2: Q4_0 MatVec
    uint row = tgid * 8 + simd_id;
    if (row >= N) return;

    const uint num_blocks_k = K / QK4_0;
    const device block_q4_0* weights = (const device block_q4_0*)W;

    float local_sum = 0.0f;

    for (uint kb = simd_lane; kb < num_blocks_k; kb += 32) {
        const device block_q4_0* block = &weights[row * num_blocks_k + kb];
        float d = float(block->d);

        uint base_idx = kb * QK4_0;
        float acc = 0.f;
        for (int i = 0; i < 16; i++) {
            uint8_t qbyte = block->qs[i];
            int q0 = (qbyte & 0xF) - 8;
            int q1 = (qbyte >> 4) - 8;
            acc += x_norm[base_idx + i*2 + 0] * float(q0);
            acc += x_norm[base_idx + i*2 + 1] * float(q1);
        }
        local_sum += d * acc;
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row] = sum;
    }
}

// =============================================================================
// IQ4_NL Matrix-Vector Multiplication (Non-linear 4-bit I-quant)
// =============================================================================
// IQ4_NL: 32 elements per block, 18 bytes (half d + 16 bytes qs)
// Uses non-linear lookup table for better quality at 4-bit
// Dequantization: w = d * kvalues_iq4nl[q] where q is 4-bit index

// IQ4_NL lookup table - non-linearly distributed values
constant int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// IQ4_NL block - same structure as Q4_0 but different dequantization
struct block_iq4_nl {
    half d;           // scale
    uint8_t qs[16];   // 4-bit quants (2 per byte), uses lookup table
};

constant constexpr uint QK_IQ4_NL = 32;  // IQ4_NL block size
constant constexpr short NR0_IQ4_NL = 2;  // Rows per SIMD group
constant constexpr uint IQ4_NL_ROWS_PER_TG = 8;  // SIMD groups per threadgroup

kernel void matvec_iq4_nl(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK_IQ4_NL;  // Number of IQ4_NL blocks per row

    // Each SIMD group handles NR0_IQ4_NL rows
    const uint first_row = (tgid * IQ4_NL_ROWS_PER_TG + sgitg) * NR0_IQ4_NL;

    device const block_iq4_nl* weights = (const device block_iq4_nl*)W;

    float sumf[NR0_IQ4_NL] = {0.f, 0.f};

    // Thread indexing: 32 threads per SIMD group
    // Each thread processes every 32nd block
    for (uint ib = tiisg; ib < nb; ib += 32) {
        // Cache input values in registers
        float xl[32];
        uint x_base = ib * QK_IQ4_NL;
        for (int i = 0; i < 32; i++) {
            xl[i] = x[x_base + i];
        }

        // Process each row
        for (short row = 0; row < NR0_IQ4_NL; row++) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_iq4_nl* block = &weights[row_idx * nb + ib];
            float d = float(block->d);

            // Dequantize using lookup table and accumulate
            // IQ4_NL: 2 values per byte, low nibble first
            float acc = 0.f;
            for (int i = 0; i < 16; i++) {
                uint8_t qbyte = block->qs[i];
                int q0 = kvalues_iq4nl[qbyte & 0xF];  // lookup for low nibble
                int q1 = kvalues_iq4nl[qbyte >> 4];   // lookup for high nibble
                acc += xl[i*2 + 0] * float(q0);
                acc += xl[i*2 + 1] * float(q1);
            }
            sumf[row] += d * acc;
        }
    }

    // SIMD reduction and output
    for (short row = 0; row < NR0_IQ4_NL; row++) {
        uint row_idx = first_row + row;
        if (row_idx < N) {
            float sum = simd_sum(sumf[row]);
            if (tiisg == 0) {
                y[row_idx] = sum;
            }
        }
    }
}

// IQ4_NL Matrix Multiplication (Batched - Prefill)
kernel void matmul_iq4_nl(
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

    const uint num_blocks_k = K / QK_IQ4_NL;
    const device block_iq4_nl* weights = (const device block_iq4_nl*)W;

    float sum = 0.0f;

    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_iq4_nl* block = &weights[col * num_blocks_k + kb];
        float d = float(block->d);

        uint base_idx = kb * QK_IQ4_NL;

        for (int i = 0; i < 16; i++) {
            uint8_t qbyte = block->qs[i];
            int q0 = kvalues_iq4nl[qbyte & 0xF];
            int q1 = kvalues_iq4nl[qbyte >> 4];
            sum += d * float(q0) * X[row * K + base_idx + i*2 + 0];
            sum += d * float(q1) * X[row * K + base_idx + i*2 + 1];
        }
    }

    Y[row * N + col] = sum;
}

// Fused RMSNorm + IQ4_NL MatVec
kernel void rms_norm_matvec_iq4_nl(
    device const float* x          [[buffer(0)]],
    device const half* norm_weight [[buffer(1)]],
    device const void* W           [[buffer(2)]],
    device float* y                [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
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

    // Step 2: IQ4_NL MatVec
    uint row = tgid * 8 + simd_id;
    if (row >= N) return;

    const uint num_blocks_k = K / QK_IQ4_NL;
    const device block_iq4_nl* weights = (const device block_iq4_nl*)W;

    float local_sum = 0.0f;

    for (uint kb = simd_lane; kb < num_blocks_k; kb += 32) {
        const device block_iq4_nl* block = &weights[row * num_blocks_k + kb];
        float d = float(block->d);

        uint base_idx = kb * QK_IQ4_NL;
        float acc = 0.f;
        for (int i = 0; i < 16; i++) {
            uint8_t qbyte = block->qs[i];
            int q0 = kvalues_iq4nl[qbyte & 0xF];
            int q1 = kvalues_iq4nl[qbyte >> 4];
            acc += x_norm[base_idx + i*2 + 0] * float(q0);
            acc += x_norm[base_idx + i*2 + 1] * float(q1);
        }
        local_sum += d * acc;
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row] = sum;
    }
}

// =============================================================================
// IQ4_XS Matrix-Vector Multiplication (Single Token Decode)
// =============================================================================
// IQ4_XS: 256 elements per super-block, 136 bytes
// Layout: d (half) + scales_h (uint16) + scales_l[4] + qs[128]
// Uses same lookup table as IQ4_NL but with per-sub-block 6-bit scales

struct block_iq4_xs {
    half d;              // super-block scale
    uint16_t scales_h;   // high 2 bits of scales
    uint8_t scales_l[4]; // low 4 bits of scales
    uint8_t qs[128];     // 4-bit quants (uses lookup table)
};

constant constexpr uint QK_IQ4_XS = 256;  // IQ4_XS block size
constant constexpr short NR0_IQ4_XS = 2;  // Rows per SIMD group
constant constexpr uint IQ4_XS_ROWS_PER_TG = 8;  // SIMD groups per threadgroup

kernel void matvec_iq4_xs(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK_IQ4_XS;  // Number of IQ4_XS blocks per row

    // Each SIMD group handles NR0_IQ4_XS rows
    const uint first_row = (tgid * IQ4_XS_ROWS_PER_TG + sgitg) * NR0_IQ4_XS;

    device const block_iq4_xs* weights = (const device block_iq4_xs*)W;

    float sumf[NR0_IQ4_XS] = {0.f, 0.f};

    // Thread indexing for IQ4_XS (256-element super-blocks)
    // Each thread handles 32 elements (1 sub-block) per iteration
    const short ix = tiisg / 16;  // 0 or 1 (which half of super-block)
    const short it = tiisg % 16;  // 0...15
    const short ib = it / 2;      // sub-block index within half (0...7)
    const short il = it % 2;      // which 16 elements within sub-block

    // Process all blocks
    for (uint ibl = ix; ibl < nb; ibl += 2) {
        // Calculate base position in input
        uint x_base = ibl * QK_IQ4_XS + ib * 32 + il * 16;

        // Cache 16 input values (covers half of a 32-element sub-block)
        float xl[16];
        for (int i = 0; i < 16; i++) {
            xl[i] = x[x_base + i];
        }

        // Process each row
        for (short row = 0; row < NR0_IQ4_XS; row++) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_iq4_xs& block = weights[row_idx * nb + ibl];
            float d = float(block.d);

            // Extract 6-bit scale for this sub-block
            int ls = ((block.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                     (((block.scales_h >> (2 * ib)) & 3) << 4);
            float scale = d * float(ls - 32);

            // Dequantize and accumulate using lookup table
            float acc = 0.f;
            for (int i = 0; i < 8; i++) {
                uint8_t qbyte = block.qs[ib * 16 + il * 8 + i];
                int q0 = kvalues_iq4nl[qbyte & 0xF];
                int q1 = kvalues_iq4nl[qbyte >> 4];
                acc += xl[i * 2 + 0] * float(q0);
                acc += xl[i * 2 + 1] * float(q1);
            }
            sumf[row] += scale * acc;
        }
    }

    // SIMD reduction and output
    for (short row = 0; row < NR0_IQ4_XS; row++) {
        uint row_idx = first_row + row;
        if (row_idx < N) {
            float sum = simd_sum(sumf[row]);
            if (tiisg == 0) {
                y[row_idx] = sum;
            }
        }
    }
}

// IQ4_XS Matrix Multiplication (Batched - Prefill) - Optimized with SIMD groups
// Each threadgroup processes one input row and COLS_PER_TG output columns
// Each SIMD group handles 2 output columns with K-reduction cooperation
constant constexpr uint IQ4_XS_MATMUL_COLS_PER_SIMD = 2;  // Columns per SIMD group
constant constexpr uint IQ4_XS_MATMUL_SIMD_GROUPS = 8;    // SIMD groups per threadgroup

kernel void matmul_iq4_xs(
    device const float* X          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* Y                [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    // Row is determined by threadgroup y position
    const uint row = tgid.y;
    if (row >= M) return;

    // Each SIMD group handles 2 columns
    const uint col_base = (tgid.x * IQ4_XS_MATMUL_SIMD_GROUPS + sgitg) * IQ4_XS_MATMUL_COLS_PER_SIMD;

    const uint nb = K / QK_IQ4_XS;  // Number of super-blocks per row
    const device block_iq4_xs* weights = (const device block_iq4_xs*)W;

    // Each thread accumulates for 2 columns
    float sumf[IQ4_XS_MATMUL_COLS_PER_SIMD] = {0.f, 0.f};

    // Thread indexing for IQ4_XS (256-element super-blocks)
    // 32 threads cooperate: each handles different sub-blocks
    const short ix = tiisg / 16;  // 0 or 1 (which half of super-block)
    const short it = tiisg % 16;
    const short ib = it / 2;      // sub-block index within half (0...7)
    const short il = it % 2;      // which 16 elements within sub-block

    // Process all blocks with strided access
    for (uint ibl = ix; ibl < nb; ibl += 2) {
        // Calculate base position in input
        uint x_base = row * K + ibl * QK_IQ4_XS + ib * 32 + il * 16;

        // Cache 16 input values in registers
        float xl[16];
        for (int i = 0; i < 16; i++) {
            xl[i] = X[x_base + i];
        }

        // Process each output column
        for (short c = 0; c < IQ4_XS_MATMUL_COLS_PER_SIMD; c++) {
            uint col = col_base + c;
            if (col >= N) continue;

            device const block_iq4_xs& block = weights[col * nb + ibl];
            float d = float(block.d);

            // Extract 6-bit scale for this sub-block
            int ls = ((block.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                     (((block.scales_h >> (2 * ib)) & 3) << 4);
            float scale = d * float(ls - 32);

            // Dequantize and accumulate using lookup table
            float acc = 0.f;
            for (int i = 0; i < 8; i++) {
                uint8_t qbyte = block.qs[ib * 16 + il * 8 + i];
                int q0 = kvalues_iq4nl[qbyte & 0xF];
                int q1 = kvalues_iq4nl[qbyte >> 4];
                acc += xl[i * 2 + 0] * float(q0);
                acc += xl[i * 2 + 1] * float(q1);
            }
            sumf[c] += scale * acc;
        }
    }

    // SIMD reduction and output
    for (short c = 0; c < IQ4_XS_MATMUL_COLS_PER_SIMD; c++) {
        uint col = col_base + c;
        if (col < N) {
            float sum = simd_sum(sumf[c]);
            if (tiisg == 0) {
                Y[row * N + col] = sum;
            }
        }
    }
}

// Fused RMSNorm + IQ4_XS MatVec
kernel void rms_norm_matvec_iq4_xs(
    device const float* x          [[buffer(0)]],
    device const half* norm_weight [[buffer(1)]],
    device const void* W           [[buffer(2)]],
    device float* y                [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
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

    // Step 2: IQ4_XS MatVec
    uint row = tgid * 8 + simd_id;
    if (row >= N) return;

    const uint num_blocks_k = K / QK_IQ4_XS;
    const device block_iq4_xs* weights = (const device block_iq4_xs*)W;

    float local_sum = 0.0f;

    for (uint kb = simd_lane; kb < num_blocks_k; kb += 32) {
        const device block_iq4_xs& block = weights[row * num_blocks_k + kb];
        float d = float(block.d);

        uint base_idx = kb * QK_IQ4_XS;

        // Process 8 sub-blocks of 32 elements each
        for (int ib32 = 0; ib32 < 8; ib32++) {
            // Extract 6-bit scale
            int ls = ((block.scales_l[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf) |
                     (((block.scales_h >> (2 * ib32)) & 3) << 4);
            float scale = d * float(ls - 32);

            float acc = 0.f;
            for (int i = 0; i < 16; i++) {
                uint8_t qbyte = block.qs[ib32 * 16 + i];
                int q0 = kvalues_iq4nl[qbyte & 0xF];
                int q1 = kvalues_iq4nl[qbyte >> 4];
                acc += x_norm[base_idx + ib32 * 32 + i * 2 + 0] * float(q0);
                acc += x_norm[base_idx + ib32 * 32 + i * 2 + 1] * float(q1);
            }
            local_sum += scale * acc;
        }
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row] = sum;
    }
}

// =============================================================================
// IQ3_S Matrix-Vector Multiplication (Single Token Decode)
// =============================================================================
// IQ3_S: 256 elements per super-block, 110 bytes
// Layout: d (half) + qs[64] + qh[8] + signs[32] + scales[4]
// Uses iq3s_grid[512] lookup table for 3-bit values

struct block_iq3_s {
    half d;              // super-block scale
    uint8_t qs[64];      // 8-bit grid indices (low bits)
    uint8_t qh[8];       // high bits for grid indices
    uint8_t signs[32];   // sign bits
    uint8_t scales[4];   // 4-bit scales per 64 elements
};

constant constexpr uint QK_IQ3_S = 256;  // IQ3_S block size
constant constexpr short NR0_IQ3_S = 2;  // Rows per SIMD group
constant constexpr uint IQ3_S_ROWS_PER_TG = 8;  // SIMD groups per threadgroup

// IQ3_S lookup table (512 entries)
constant uint32_t iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
};

// Sign bitmask for IQ3_S
constant uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

kernel void matvec_iq3_s(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK_IQ3_S;  // Number of IQ3_S blocks per row

    // Each SIMD group handles NR0_IQ3_S rows
    const uint first_row = (tgid * IQ3_S_ROWS_PER_TG + sgitg) * NR0_IQ3_S;

    device const block_iq3_s* weights = (const device block_iq3_s*)W;

    float sumf[NR0_IQ3_S] = {0.f, 0.f};

    // Thread indexing for IQ3_S (256-element super-blocks)
    // Same pattern as IQ4_XS: each thread handles 16 elements from a sub-block half
    const short ix = tiisg / 16;  // 0 or 1 (which half of super-block to start - strided)
    const short it = tiisg % 16;
    const short ib = it / 2;      // sub-block index within half (0...7)
    const short il = it % 2;      // which 16-element half within sub-block

    // Process blocks with strided access (ix=0 or 1, step by 2)
    for (uint ibl = ix; ibl < nb; ibl += 2) {
        // Calculate base position in input
        uint x_base = ibl * QK_IQ3_S + ib * 32 + il * 16;

        // Cache 16 input values in registers
        float xl[16];
        for (int i = 0; i < 16; i++) {
            xl[i] = x[x_base + i];
        }

        // Process each row
        for (short row = 0; row < NR0_IQ3_S; row++) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_iq3_s& block = weights[row_idx * nb + ibl];
            float d = float(block.d);

            // Extract 4-bit scale for this sub-block: 1 + 2 * scale_value
            int scale_val = (block.scales[ib / 2] >> (4 * (ib % 2))) & 0xf;
            float dl = d * float(1 + 2 * scale_val);

            // Get pointers for this half
            device const uint8_t* qs_ptr = block.qs + 8 * ib + 4 * il;
            device const uint8_t* signs_ptr = block.signs + 4 * ib + 2 * il;
            uint8_t qh_byte = block.qh[ib] >> (4 * il);

            // Process 4 grid entries × 4 elements each = 16 elements
            float acc = 0.f;
            for (int j = 0; j < 4; j++) {
                int grid_idx = qs_ptr[j] | (((qh_byte >> (3 - j)) & 1) << 8);
                uint32_t grid_val = iq3s_grid[grid_idx];
                uint8_t sign_byte = signs_ptr[j / 2];

                // Unrolled inner loop for 4 elements
                for (int k = 0; k < 4; k++) {
                    uint8_t grid_byte = (grid_val >> (8 * k)) & 0xFF;
                    int sign_bit_idx = (j % 2) * 4 + k;
                    float sign = (sign_byte & kmask_iq2xs[sign_bit_idx]) ? -1.0f : 1.0f;
                    acc += xl[j * 4 + k] * float(grid_byte) * sign;
                }
            }
            sumf[row] += dl * acc;
        }
    }

    // SIMD reduction and output
    for (short row = 0; row < NR0_IQ3_S; row++) {
        uint row_idx = first_row + row;
        if (row_idx < N) {
            float sum = simd_sum(sumf[row]);
            if (tiisg == 0) {
                y[row_idx] = sum;
            }
        }
    }
}

// IQ3_S Matrix Multiplication (Batched - Prefill) - Optimized with SIMD groups
// Each threadgroup processes one input row and multiple output columns
// Each SIMD group handles 2 output columns with K-reduction cooperation
constant constexpr uint IQ3_S_MATMUL_COLS_PER_SIMD = 2;
constant constexpr uint IQ3_S_MATMUL_SIMD_GROUPS = 8;

kernel void matmul_iq3_s(
    device const float* X          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* Y                [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint row = tgid.y;
    if (row >= M) return;

    const uint col_base = (tgid.x * IQ3_S_MATMUL_SIMD_GROUPS + sgitg) * IQ3_S_MATMUL_COLS_PER_SIMD;

    const uint nb = K / QK_IQ3_S;
    const device block_iq3_s* weights = (const device block_iq3_s*)W;

    float sumf[IQ3_S_MATMUL_COLS_PER_SIMD] = {0.f, 0.f};

    // Thread indexing: 32 threads cooperate on K dimension
    // Each thread handles a different sub-block position
    const short ix = tiisg / 16;   // 0 or 1 (which half of super-block)
    const short it = tiisg % 16;
    const short ib = it / 2;       // sub-block index (0...7)
    const short il = it % 2;       // which 16 elements within sub-block

    // Process all blocks with strided access
    for (uint ibl = ix; ibl < nb; ibl += 2) {
        // Calculate base position in input
        uint x_base = row * K + ibl * QK_IQ3_S + ib * 32 + il * 16;

        // Cache 16 input values in registers
        float xl[16];
        for (int i = 0; i < 16; i++) {
            xl[i] = X[x_base + i];
        }

        // Process each output column
        for (short c = 0; c < IQ3_S_MATMUL_COLS_PER_SIMD; c++) {
            uint col = col_base + c;
            if (col >= N) continue;

            device const block_iq3_s& block = weights[col * nb + ibl];
            float d = float(block.d);

            // Extract 4-bit scale
            int scale_val = (block.scales[ib / 2] >> (4 * (ib % 2))) & 0xf;
            float dl = d * float(1 + 2 * scale_val);

            // Get pointers for this half
            device const uint8_t* qs_ptr = block.qs + 8 * ib + 4 * il;
            device const uint8_t* signs_ptr = block.signs + 4 * ib + 2 * il;
            uint8_t qh_byte = block.qh[ib] >> (4 * il);

            // Accumulate 16 elements
            float acc = 0.f;
            for (int j = 0; j < 4; j++) {
                int grid_idx = qs_ptr[j] | (((qh_byte >> (3 - j)) & 1) << 8);
                uint32_t grid_val = iq3s_grid[grid_idx];
                uint8_t sign_byte = signs_ptr[j / 2];

                for (int k = 0; k < 4; k++) {
                    uint8_t grid_byte = (grid_val >> (8 * k)) & 0xFF;
                    int sign_bit_idx = (j % 2) * 4 + k;
                    float sign = (sign_byte & kmask_iq2xs[sign_bit_idx]) ? -1.0f : 1.0f;
                    acc += xl[j * 4 + k] * float(grid_byte) * sign;
                }
            }
            sumf[c] += dl * acc;
        }
    }

    // SIMD reduction and output
    for (short c = 0; c < IQ3_S_MATMUL_COLS_PER_SIMD; c++) {
        uint col = col_base + c;
        if (col < N) {
            float sum = simd_sum(sumf[c]);
            if (tiisg == 0) {
                Y[row * N + col] = sum;
            }
        }
    }
}

// Fused RMSNorm + IQ3_S MatVec
kernel void rms_norm_matvec_iq3_s(
    device const float* x          [[buffer(0)]],
    device const half* norm_weight [[buffer(1)]],
    device const void* W           [[buffer(2)]],
    device float* y                [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
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

    // Step 2: IQ3_S MatVec
    uint row_out = tgid * 8 + simd_id;
    if (row_out >= N) return;

    const uint num_blocks_k = K / QK_IQ3_S;
    const device block_iq3_s* weights = (const device block_iq3_s*)W;

    float local_sum = 0.0f;

    for (uint kb = simd_lane; kb < num_blocks_k; kb += 32) {
        const device block_iq3_s& block = weights[row_out * num_blocks_k + kb];
        float d = float(block.d);

        uint base_idx = kb * QK_IQ3_S;

        // Process 8 sub-blocks of 32 elements each
        for (int ib32 = 0; ib32 < 8; ib32++) {
            // Extract 4-bit scale
            int scale_val = (block.scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf;
            float dl = d * float(1 + 2 * scale_val);

            // Process 2 halves of 16 elements each
            for (int il = 0; il < 2; il++) {
                device const uint8_t* qs_ptr = block.qs + 8 * ib32 + 4 * il;
                device const uint8_t* signs_ptr = block.signs + 4 * ib32 + 2 * il;
                uint8_t qh_byte = block.qh[ib32] >> (4 * il);

                // Process 4 grid entries of 4 elements each
                for (int j = 0; j < 4; j++) {
                    int grid_idx = qs_ptr[j] | (((qh_byte >> (3 - j)) & 1) << 8);
                    uint32_t grid_val = iq3s_grid[grid_idx];
                    uint8_t sign_byte = signs_ptr[j / 2];

                    for (int k = 0; k < 4; k++) {
                        // Extract byte from grid_val using bit shifts
                        uint8_t grid_byte = (grid_val >> (8 * k)) & 0xFF;
                        int sign_bit_idx = (j % 2) * 4 + k;
                        float sign = (sign_byte & kmask_iq2xs[sign_bit_idx]) ? -1.0f : 1.0f;
                        uint idx = base_idx + ib32 * 32 + il * 16 + j * 4 + k;
                        local_sum += dl * float(grid_byte) * sign * x_norm[idx];
                    }
                }
            }
        }
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row_out] = sum;
    }
}

// =============================================================================
// Q6_K Matrix-Vector Multiplication (Single Token Decode)
// =============================================================================
// Q6_K: 256 elements per super-block, 210 bytes
// Layout: ql[128] (lower 4 bits) + qh[64] (upper 2 bits) + scales[16] + d
// Dequantization: w = d * scales[j] * (q6 - 32) where q6 is 6-bit value

constant constexpr short NR0_Q6K = 2;  // Rows per SIMD group
constant constexpr uint Q6K_ROWS_PER_TG = 8;  // SIMD groups per threadgroup

kernel void matvec_q6_k(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK_K;  // Number of Q6_K blocks per row

    // Each SIMD group handles NR0_Q6K rows
    const uint first_row = (tgid * Q6K_ROWS_PER_TG + sgitg) * NR0_Q6K;

    device const block_q6_K* weights = (const device block_q6_K*)W;

    float sumf[NR0_Q6K] = {0.f, 0.f};

    // Thread indexing: 32 threads per SIMD group
    // Each thread processes 8 elements per block (256/32 = 8)
    // We'll have each thread handle a contiguous 8-element chunk
    const uint elem_per_thread = 8;
    const uint thread_offset = tiisg * elem_per_thread;

    for (uint ib = 0; ib < nb; ib++) {
        // Cache input values for this block
        float xl[8];
        uint x_base = ib * QK_K + thread_offset;
        for (int i = 0; i < 8; i++) {
            xl[i] = x[x_base + i];
        }

        // Process each row
        for (short row = 0; row < NR0_Q6K; row++) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_q6_K* block = &weights[row_idx * nb + ib];
            float d = float(block->d);

            // Dequantize and accumulate for this thread's 8 elements
            float acc = 0.f;
            for (uint i = 0; i < 8; i++) {
                uint elem_idx = thread_offset + i;

                // Extract lower 4 bits from ql (2 values per byte)
                uint ql_byte = elem_idx / 2;
                uint ql_shift = (elem_idx % 2) * 4;
                uint low4 = (block->ql[ql_byte] >> ql_shift) & 0xF;

                // Extract upper 2 bits from qh (4 values per byte)
                uint qh_byte = elem_idx / 4;
                uint qh_shift = (elem_idx % 4) * 2;
                uint high2 = (block->qh[qh_byte] >> qh_shift) & 0x3;

                // Combine to get 6-bit value and convert to signed
                int q6 = int(low4 | (high2 << 4)) - 32;

                // Get scale for this 16-element sub-block
                uint scale_idx = elem_idx / 16;
                float scale = float(block->scales[scale_idx]);

                acc += xl[i] * (d * scale * float(q6));
            }
            sumf[row] += acc;
        }
    }

    // SIMD reduction and output
    for (short row = 0; row < NR0_Q6K; row++) {
        uint row_idx = first_row + row;
        if (row_idx < N) {
            float sum = simd_sum(sumf[row]);
            if (tiisg == 0) {
                y[row_idx] = sum;
            }
        }
    }
}

// Q6_K Matrix Multiplication (Batched - Prefill)
kernel void matmul_q6_k(
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
    const device block_q6_K* weights = (const device block_q6_K*)W;

    float sum = 0.0f;

    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q6_K* block = &weights[col * num_blocks_k + kb];
        float d = float(block->d);

        uint base_idx = kb * QK_K;

        // Process all 256 elements in the block
        for (uint i = 0; i < 256; i++) {
            // Extract lower 4 bits from ql
            uint ql_byte = i / 2;
            uint ql_shift = (i % 2) * 4;
            uint low4 = (block->ql[ql_byte] >> ql_shift) & 0xF;

            // Extract upper 2 bits from qh
            uint qh_byte = i / 4;
            uint qh_shift = (i % 4) * 2;
            uint high2 = (block->qh[qh_byte] >> qh_shift) & 0x3;

            // Combine and convert to signed
            int q6 = int(low4 | (high2 << 4)) - 32;

            // Get scale
            uint scale_idx = i / 16;
            float scale = float(block->scales[scale_idx]);

            sum += d * scale * float(q6) * X[row * K + base_idx + i];
        }
    }

    Y[row * N + col] = sum;
}

// Fused RMSNorm + Q6_K MatVec
kernel void rms_norm_matvec_q6_k(
    device const float* x          [[buffer(0)]],
    device const half* norm_weight [[buffer(1)]],
    device const void* W           [[buffer(2)]],
    device float* y                [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
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

    // Step 2: Q6_K MatVec
    uint row = tgid * 8 + simd_id;
    if (row >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q6_K* weights = (const device block_q6_K*)W;

    float local_sum = 0.0f;

    // Each SIMD lane processes blocks with stride
    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q6_K* block = &weights[row * num_blocks_k + kb];
        float d = float(block->d);

        uint base_idx = kb * QK_K;

        // Each thread handles 8 elements (256/32 = 8)
        const uint elem_per_thread = 8;
        uint thread_offset = simd_lane * elem_per_thread;

        float acc = 0.f;
        for (uint i = 0; i < elem_per_thread; i++) {
            uint elem_idx = thread_offset + i;

            // Extract lower 4 bits from ql
            uint ql_byte = elem_idx / 2;
            uint ql_shift = (elem_idx % 2) * 4;
            uint low4 = (block->ql[ql_byte] >> ql_shift) & 0xF;

            // Extract upper 2 bits from qh
            uint qh_byte = elem_idx / 4;
            uint qh_shift = (elem_idx % 4) * 2;
            uint high2 = (block->qh[qh_byte] >> qh_shift) & 0x3;

            // Combine and convert to signed
            int q6 = int(low4 | (high2 << 4)) - 32;

            // Get scale
            uint scale_idx = elem_idx / 16;
            float scale = float(block->scales[scale_idx]);

            acc += x_norm[base_idx + elem_idx] * (d * scale * float(q6));
        }
        local_sum += acc;
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row] = sum;
    }
}

// =============================================================================
// Q3_K Matrix-Vector Multiplication (Single Token Decode)
// =============================================================================
// Q3_K: 256 elements per super-block, 110 bytes
// Layout: hmask[32] + qs[64] + scales[12] + d[2]
// Each weight = 2 low bits from qs + 1 high bit from hmask
// Dequantization: w = d * (scales[j] - 32) * (q3 - 4)

constant constexpr short NR0_Q3K = 2;  // Rows per SIMD group
constant constexpr uint Q3K_ROWS_PER_TG = 8;  // SIMD groups per threadgroup

// Optimized Q3_K MatVec - follows llama.cpp style approach
kernel void matvec_q3_k(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK_K;  // Number of Q3_K blocks per row

    // Thread indexing (32 threads per SIMD group)
    // Based on llama.cpp: tid=tiisg/4, ix=tiisg%4
    const short tid = tiisg / 4;  // 0...7
    const short ix  = tiisg % 4;  // 0...3 (strided access)
    const short ip  = tid / 4;    // 0 or 1 (which half of block)
    const short il  = 2 * ((tid % 4) / 2);  // 0 or 2
    const short ir  = tid % 2;
    const short l0  = 8 * ir;

    // Masks for high bit extraction (based on ip and il)
    // Each combination of ip,il gives different mask patterns
    const ushort4 mm[4] = {
        ushort4(0x0001, 0x0100, 0x0002, 0x0200),  // ip = 0, il = 0
        ushort4(0x0004, 0x0400, 0x0008, 0x0800),  // ip = 0, il = 2
        ushort4(0x0010, 0x1000, 0x0020, 0x2000),  // ip = 1, il = 0
        ushort4(0x0040, 0x4000, 0x0080, 0x8000)   // ip = 1, il = 2
    };

    // Masks for low 2-bit extraction
    const int4 qm[2] = {
        int4(0x0003, 0x0300, 0x000c, 0x0c00),
        int4(0x0030, 0x3000, 0x00c0, 0xc000)
    };

    const ushort4 hm = mm[2 * ip + il / 2];
    const short shift = 2 * il;

    // v1/v2 multipliers for high bit contribution
    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    // Scale extraction helpers
    const ushort s_shift1 = 4 * ip;
    const ushort s_shift2 = s_shift1 + il;

    const short q_offset = 32 * ip + l0;
    const short y_offset = 128 * ip + 32 * il + l0;

    // Each SIMD group handles NR0_Q3K rows
    const uint first_row = (tgid * Q3K_ROWS_PER_TG + sgitg) * NR0_Q3K;

    device const block_q3_K* weights = (const device block_q3_K*)W;

    // Register-based input caching - each thread caches 32 floats
    float yl[32];
    float sumf1[NR0_Q3K] = {0.f, 0.f};
    float sumf2[NR0_Q3K] = {0.f, 0.f};

    // Process blocks with strided access (ix=0..3, step by 4)
    device const float* y1_base = x + ix * QK_K + y_offset;

    for (uint i = ix; i < nb; i += 4) {
        // Cache input values
        device const float* y1 = y1_base + (i / 4) * 4 * QK_K;
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0];
            yl[l + 8] = y1[l + 16];
            yl[l + 16] = y1[l + 32];
            yl[l + 24] = y1[l + 48];
        }

        // Process each row
        for (short row = 0; row < NR0_Q3K; ++row) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_q3_K* block = &weights[row_idx * nb + i];
            device const ushort* q = (device const ushort*)(block->qs + q_offset);
            device const ushort* h = (device const ushort*)(block->hmask + l0);
            device const ushort* a = (device const ushort*)(block->scales);

            const float d_all = float(block->d);

            // Extract scales using llama.cpp method
            uint32_t scales32, aux32;
            thread ushort* scales16 = (thread ushort*)&scales32;
            thread const int8_t* scales = (thread const int8_t*)&scales32;

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il + 0];
            scales16[1] = a[il + 1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            // First half of sub-block
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

            // Second half of sub-block
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

    // SIMD reduction and output
    for (short row = 0; row < NR0_Q3K; ++row) {
        uint row_idx = first_row + row;
        if (row_idx < N) {
            const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / float(1 << shift);
            float sum = simd_sum(sumf);
            if (tiisg == 0) {
                y[row_idx] = sum;
            }
        }
    }
}

// Q3_K Matrix Multiplication (Batched - Prefill)
// Simple implementation for batched prefill
kernel void matmul_q3_k(
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
    const device block_q3_K* weights = (const device block_q3_K*)W;

    float sum = 0.0f;

    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q3_K* block = &weights[col * num_blocks_k + kb];
        float d = float(block->d);

        uint base_idx = kb * QK_K;

        // Process all 256 elements in the block
        for (uint i = 0; i < 256; i++) {
            // Extract low 2 bits from qs (4 values per byte)
            uint qs_byte = i / 4;
            uint qs_shift = (i % 4) * 2;
            uint low2 = (block->qs[qs_byte] >> qs_shift) & 0x3;

            // Extract high bit from hmask (8 values per byte)
            uint hm_byte = i / 8;
            uint hm_bit = i % 8;
            uint high1 = (block->hmask[hm_byte] >> hm_bit) & 0x1;

            // Combine to get 3-bit value and convert to signed
            int q3 = int(low2 | (high1 << 2)) - 4;

            // Get scale for this 32-element sub-block (8 sub-blocks total)
            // Q3_K has 6-bit scales packed in 12 bytes
            uint sub_block = i / 32;  // 0-7
            int8_t scale;

            // Scale extraction is complex for Q3_K
            // Lower 4 bits come from scales[0..3] or scales[2..5]
            // Upper 2 bits come from scales[8..11]
            if (sub_block < 4) {
                uint8_t sc_low = block->scales[sub_block] & 0x0F;
                uint8_t sc_high = (block->scales[8 + sub_block / 2] >> (4 * (sub_block % 2))) & 0x03;
                scale = int8_t((sc_low | (sc_high << 4))) - 32;
            } else {
                uint8_t sc_low = (block->scales[sub_block - 4] >> 4) & 0x0F;
                uint8_t sc_high = (block->scales[10 + (sub_block - 4) / 2] >> (4 * ((sub_block - 4) % 2))) & 0x03;
                scale = int8_t((sc_low | (sc_high << 4))) - 32;
            }

            sum += X[row * K + base_idx + i] * (d * float(scale) * float(q3));
        }
    }

    Y[row * N + col] = sum;
}

// Fused RMSNorm + Q3_K MatVec
kernel void rms_norm_matvec_q3_k(
    device const float* x          [[buffer(0)]],
    device const half* norm_weight [[buffer(1)]],
    device const void* W           [[buffer(2)]],
    device float* y                [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
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

    // Step 2: Q3_K MatVec
    uint row = tgid * 8 + simd_id;
    if (row >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q3_K* weights = (const device block_q3_K*)W;

    float local_sum = 0.0f;

    for (uint kb = simd_lane; kb < num_blocks_k; kb += 32) {
        const device block_q3_K* block = &weights[row * num_blocks_k + kb];
        float d = float(block->d);

        uint base_idx = kb * QK_K;
        float acc = 0.0f;

        // Process 8 elements per iteration (256/32 = 8 elements per thread per block)
        for (uint i = 0; i < 256; i++) {
            // Extract low 2 bits from qs
            uint qs_byte = i / 4;
            uint qs_shift = (i % 4) * 2;
            uint low2 = (block->qs[qs_byte] >> qs_shift) & 0x3;

            // Extract high bit from hmask
            uint hm_byte = i / 8;
            uint hm_bit = i % 8;
            uint high1 = (block->hmask[hm_byte] >> hm_bit) & 0x1;

            // Combine to get 3-bit value
            int q3 = int(low2 | (high1 << 2)) - 4;

            // Get scale
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
        local_sum += acc;
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row] = sum;
    }
}

// =============================================================================
// Q2_K Matrix-Vector Multiplication (Single Token Decode)
// =============================================================================
// Q2_K: 256 elements per super-block, 84 bytes
// Layout: scales[16] + qs[64] + d[2] + dmin[2]
// Dequantization: w = d * (scale & 0xF) * q2 - dmin * (scale >> 4)

constant constexpr short NR0_Q2K = 2;  // Rows per SIMD group
constant constexpr uint Q2K_ROWS_PER_TG = 8;  // SIMD groups per threadgroup

kernel void matvec_q2_k(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK_K;  // Number of Q2_K blocks per row

    // Thread indexing (32 threads per SIMD group)
    const short ix = tiisg / 8;  // 0...3
    const short it = tiisg % 8;  // 0...7
    const short iq = it / 4;     // 0 or 1
    const short ir = it % 4;     // 0...3
    const short is = (8 * ir) / 16;  // 0 or 1

    // Each SIMD group handles NR0_Q2K rows
    const uint first_row = (tgid * Q2K_ROWS_PER_TG + sgitg) * NR0_Q2K;

    device const block_q2_K* weights = (const device block_q2_K*)W;

    // Register-based input caching
    float yl[32];
    float sumf[NR0_Q2K] = {0.f, 0.f};

    // Pointer to input for this thread's portion
    device const float* y4 = x + ix * QK_K + 128 * iq + 8 * ir;

    // Process blocks with stride 4
    for (uint ib = ix; ib < nb; ib += 4) {
        // Load input values and compute sumy for min subtraction
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        // Process each row
        for (short row = 0; row < NR0_Q2K; row++) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_q2_K* block = &weights[row_idx * nb + ib];
            device const uchar* sc = block->scales + 8 * iq + is;
            device const ushort* qs = (device const ushort*)block->qs + 16 * iq + 4 * ir;
            device const half* dh = &block->d;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            // Extract 2-bit values using masks
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

    // SIMD reduction and output
    for (short row = 0; row < NR0_Q2K; ++row) {
        uint row_idx = first_row + row;
        if (row_idx < N) {
            float sum = simd_sum(sumf[row]);
            if (tiisg == 0) {
                y[row_idx] = sum;
            }
        }
    }
}

// Q2_K Matrix Multiplication (Batched - Prefill)
kernel void matmul_q2_k(
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
    const device block_q2_K* weights = (const device block_q2_K*)W;

    float sum = 0.0f;

    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q2_K* block = &weights[col * num_blocks_k + kb];
        float d = float(block->d);
        float dmin = float(block->dmin);

        uint base_idx = kb * QK_K;

        // Process 256 elements in the block
        // Q2_K has 16 sub-blocks of 16 elements each
        for (uint j = 0; j < 16; j++) {
            // Get scale and min for this sub-block
            uint8_t sc_byte = block->scales[j];
            float scale = d * float(sc_byte & 0xF);
            float min_val = dmin * float(sc_byte >> 4);

            // Process 16 elements in this sub-block
            for (uint i = 0; i < 16; i++) {
                uint elem = j * 16 + i;
                uint qs_idx = elem / 4;
                uint qs_shift = (elem % 4) * 2;
                int q2 = int((block->qs[qs_idx] >> qs_shift) & 0x3);

                sum += X[row * K + base_idx + elem] * (scale * float(q2) - min_val);
            }
        }
    }

    Y[row * N + col] = sum;
}

// Fused RMSNorm + Q2_K MatVec
kernel void rms_norm_matvec_q2_k(
    device const float* x          [[buffer(0)]],
    device const half* norm_weight [[buffer(1)]],
    device const void* W           [[buffer(2)]],
    device float* y                [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float x_norm[4096];
    threadgroup float reduction_scratch[8];

    // Step 1: Compute RMS norm collaboratively
    float local_ss = 0.0f;
    for (uint i = tid; i < K; i += 256) {
        float val = x[i];
        local_ss += val * val;
    }

    // SIMD reduction within each simdgroup
    float simd_ss = simd_sum(local_ss);

    // Write simdgroup results to shared memory
    if (simd_lane == 0) {
        reduction_scratch[simd_id] = simd_ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by thread 0
    if (tid == 0) {
        float total_ss = 0.0f;
        for (uint i = 0; i < 8; i++) {
            total_ss += reduction_scratch[i];
        }
        reduction_scratch[0] = rsqrt(total_ss / float(K) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = reduction_scratch[0];

    // Step 2: Apply normalization and store
    for (uint i = tid; i < K; i += 256) {
        x_norm[i] = x[i] * scale * float(norm_weight[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Compute MatVec
    uint row = tgid * 8 + simd_id;
    if (row >= N) return;

    const uint nb = K / QK_K;
    const device block_q2_K* weights = (const device block_q2_K*)W;

    float local_sum = 0.0f;

    for (uint kb = simd_lane; kb < nb; kb += 32) {
        const device block_q2_K* block = &weights[row * nb + kb];
        float d = float(block->d);
        float dmin = float(block->dmin);
        uint base_idx = kb * QK_K;

        float acc = 0.0f;

        // Process all 256 elements
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
        local_sum += acc;
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row] = sum;
    }
}

// =============================================================================
// Q5_K Matrix-Vector Multiplication (Single Token Decode)
// =============================================================================
// Q5_K: 256 elements per super-block, 176 bytes
// Layout: d[2] + dmin[2] + scales[12] + qh[32] + qs[128]
// Dequantization: w = d * scale * q5 - dmin * min
// where q5 = (qs low/high nibble) + (qh bit) * 16

constant constexpr short NR0_Q5K = 2;  // Rows per SIMD group
constant constexpr uint Q5K_ROWS_PER_TG = 8;  // SIMD groups per threadgroup

// Optimized Q5_K MatVec - follows Q4_K llama.cpp-style approach
kernel void matvec_q5_k(
    device const float* x          [[buffer(0)]],
    device const void* W           [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tiisg                     [[thread_index_in_simdgroup]],
    uint sgitg                     [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = K / QK_K;  // Number of Q5_K blocks per row

    // Thread indexing (32 threads per SIMD group)
    // Split into 4 groups of 8 threads (like Q4_K)
    const short ix = tiisg / 8;  // 0...3 (which quarter of block)
    const short it = tiisg % 8;  // 0...7 (position within quarter)
    const short iq = it / 4;     // 0 or 1 (which half of sub-block)
    const short ir = it % 4;     // 0...3 (position within half)

    // Each SIMD group handles NR0_Q5K rows
    const uint first_row = (tgid * Q5K_ROWS_PER_TG + sgitg) * NR0_Q5K;

    device const block_q5_K* weights = (const device block_q5_K*)W;

    // Register-based input caching - each thread caches 32 floats
    float yl[16];
    float yh[16];
    float sumf[NR0_Q5K] = {0.f, 0.f};

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
        for (short row = 0; row < NR0_Q5K; row++) {
            uint row_idx = first_row + row;
            if (row_idx >= N) continue;

            device const block_q5_K* block = &weights[row_idx * nb + ib];
            device const ushort* sc = (device const ushort*)block->scales + iq;
            device const ushort* q1 = (device const ushort*)block->qs + 16 * iq + 4 * ir;
            device const uchar* qh = block->qh;
            device const half* dh = &block->d;

            // Extract scales using bit masks (same as Q4_K)
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const ushort* q2 = q1 + 32;

            // Get qh base offsets for this sub-block
            // Each sub-block of 64 elements uses 8 bytes of qh
            uint qh_base = 64 * iq + 8 * ir;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            // Unrolled accumulation - accumulate q values with high bits
            for (short i = 0; i < 4; ++i) {
                // Low nibbles (first 32 elements of sub-block)
                uint lo0 = q1[i] & 0x000F;
                uint lo1 = (q1[i] & 0x0F00) >> 8;
                uint lo2 = (q1[i] & 0x00F0) >> 4;
                uint lo3 = (q1[i] & 0xF000) >> 12;

                // High bits for low nibbles - elements 0-31 in sub-block
                uint elem0 = 2*i + 0;
                uint elem1 = 2*i + 1;
                uint elem2 = 2*i + 8;
                uint elem3 = 2*i + 9;

                uint h0 = (qh[(qh_base + elem0) / 8] >> ((qh_base + elem0) % 8)) & 1;
                uint h1 = (qh[(qh_base + elem1) / 8] >> ((qh_base + elem1) % 8)) & 1;
                uint h2 = (qh[(qh_base + elem2) / 8] >> ((qh_base + elem2) % 8)) & 1;
                uint h3 = (qh[(qh_base + elem3) / 8] >> ((qh_base + elem3) % 8)) & 1;

                acc1[0] += yl[2*i + 0] * float(lo0 + h0 * 16);
                acc1[1] += yl[2*i + 1] * float(lo1 + h1 * 16);
                acc1[2] += yl[2*i + 8] * float(lo2 + h2 * 16);
                acc1[3] += yl[2*i + 9] * float(lo3 + h3 * 16);

                // High nibbles (second 32 elements of sub-block)
                lo0 = q2[i] & 0x000F;
                lo1 = (q2[i] & 0x0F00) >> 8;
                lo2 = (q2[i] & 0x00F0) >> 4;
                lo3 = (q2[i] & 0xF000) >> 12;

                elem0 = 2*i + 32;
                elem1 = 2*i + 33;
                elem2 = 2*i + 40;
                elem3 = 2*i + 41;

                h0 = (qh[(qh_base + elem0) / 8] >> ((qh_base + elem0) % 8)) & 1;
                h1 = (qh[(qh_base + elem1) / 8] >> ((qh_base + elem1) % 8)) & 1;
                h2 = (qh[(qh_base + elem2) / 8] >> ((qh_base + elem2) % 8)) & 1;
                h3 = (qh[(qh_base + elem3) / 8] >> ((qh_base + elem3) % 8)) & 1;

                acc2[0] += yh[2*i + 0] * float(lo0 + h0 * 16);
                acc2[1] += yh[2*i + 1] * float(lo1 + h1 * 16);
                acc2[2] += yh[2*i + 8] * float(lo2 + h2 * 16);
                acc2[3] += yh[2*i + 9] * float(lo3 + h3 * 16);
            }

            // Accumulate with scales
            sumf[row] += dh[0] * (acc1[0] * sc8[0] + acc1[1] * sc8[0] +
                                  acc1[2] * sc8[1] + acc1[3] * sc8[1] +
                                  acc2[0] * sc8[4] + acc2[1] * sc8[4] +
                                  acc2[2] * sc8[5] + acc2[3] * sc8[5]) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                  sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * QK_K;  // Stride to next block for this thread
    }

    // SIMD reduction and output
    for (short row = 0; row < NR0_Q5K; row++) {
        uint row_idx = first_row + row;
        if (row_idx < N) {
            float sum = simd_sum(sumf[row]);
            if (tiisg == 0) {
                y[row_idx] = sum;
            }
        }
    }
}

// Q5_K Matrix Multiplication (Batched - Prefill)
kernel void matmul_q5_k(
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
    const device block_q5_K* weights = (const device block_q5_K*)W;

    float sum = 0.0f;

    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q5_K* block = &weights[col * num_blocks_k + kb];
        float d = float(block->d);
        float dmin = float(block->dmin);

        uint base_idx = kb * QK_K;

        // Process all 256 elements in the block
        for (uint i = 0; i < 256; i++) {
            // Get sub-block and extract scale/min
            uint sub_block = i / 32;
            uint8_t sc, m;
            get_scale_min_k4(sub_block, block->scales, sc, m);

            // Get low 4 bits from qs
            uint qs_byte = i / 2;
            uint low4 = (i % 2 == 0) ?
                (block->qs[qs_byte] & 0xF) :
                (block->qs[qs_byte] >> 4);

            // Get high bit from qh
            uint qh_byte = i / 8;
            uint qh_bit = (i % 8);
            uint high_bit = (block->qh[qh_byte] >> qh_bit) & 1;

            // Combine to get 5-bit value
            uint q5 = low4 + high_bit * 16;

            // Dequantize
            float w = d * float(sc) * float(q5) - dmin * float(m);
            sum += w * X[row * K + base_idx + i];
        }
    }

    Y[row * N + col] = sum;
}

// Fused RMSNorm + Q5_K MatVec
kernel void rms_norm_matvec_q5_k(
    device const float* x          [[buffer(0)]],
    device const half* norm_weight [[buffer(1)]],
    device const void* W           [[buffer(2)]],
    device float* y                [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
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

    // Step 2: Q5_K MatVec
    uint row = tgid * 8 + simd_id;
    if (row >= N) return;

    const uint num_blocks_k = K / QK_K;
    const device block_q5_K* weights = (const device block_q5_K*)W;

    float local_sum = 0.0f;

    // Each SIMD lane processes blocks
    for (uint kb = 0; kb < num_blocks_k; kb++) {
        const device block_q5_K* block = &weights[row * num_blocks_k + kb];
        float d = float(block->d);
        float dmin = float(block->dmin);

        uint base_idx = kb * QK_K;

        // Each thread handles 8 elements (256/32 = 8)
        const uint elem_per_thread = 8;
        uint thread_offset = simd_lane * elem_per_thread;

        float acc = 0.f;
        for (uint i = 0; i < elem_per_thread; i++) {
            uint elem_idx = thread_offset + i;

            // Get sub-block and extract scale/min
            uint sub_block = elem_idx / 32;
            uint8_t sc, m;
            get_scale_min_k4(sub_block, block->scales, sc, m);

            // Get low 4 bits from qs
            uint qs_byte = elem_idx / 2;
            uint low4 = (elem_idx % 2 == 0) ?
                (block->qs[qs_byte] & 0xF) :
                (block->qs[qs_byte] >> 4);

            // Get high bit from qh
            uint qh_byte = elem_idx / 8;
            uint qh_bit = (elem_idx % 8);
            uint high_bit = (block->qh[qh_byte] >> qh_bit) & 1;

            // Combine to get 5-bit value
            uint q5 = low4 + high_bit * 16;

            // Dequantize
            float w = d * float(sc) * float(q5) - dmin * float(m);
            acc += x_norm[base_idx + elem_idx] * w;
        }
        local_sum += acc;
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        y[row] = sum;
    }
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

// Batched FP16 matmul: Y = X @ W^T where X is [M,K] float, W is [N,K] half
// Each thread computes one output element
kernel void matmul_f16(
    device const float* X          [[buffer(0)]],
    device const half* W           [[buffer(1)]],
    device float* Y                [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint row = gid.y;  // token index
    uint col = gid.x;  // output dimension index

    if (row >= M || col >= N) return;

    device const float* x_row = X + row * K;
    device const half* w_row = W + col * K;

    // Vectorized accumulation using float4
    float sum = 0.0f;
    uint k = 0;

    // Process 4 elements at a time
    for (; k + 3 < K; k += 4) {
        float4 x_vec = float4(x_row[k], x_row[k+1], x_row[k+2], x_row[k+3]);
        float4 w_vec = float4(w_row[k], w_row[k+1], w_row[k+2], w_row[k+3]);
        sum += dot(x_vec, w_vec);
    }

    // Handle remaining elements
    for (; k < K; k++) {
        sum += x_row[k] * float(w_row[k]);
    }

    Y[row * N + col] = sum;
}

)";
