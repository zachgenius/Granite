// =============================================================================
// Metal Shader Source for LLM Inference Kernels
// =============================================================================
//
// This file contains all Metal compute kernels for Granite's GPU-accelerated
// LLM inference. Kernels are embedded as a string and compiled at runtime.
//
// NOTE: This file is included inside namespace granite {} in metal_compute.mm
//
// =============================================================================
// TABLE OF CONTENTS
// =============================================================================
//
// 1. QUANTIZATION FORMATS
//    - block_q4_K:  4-bit (256 elem/block, 144 bytes) - high quality K-quants
//    - block_q8_0:  8-bit (32 elem/block, 34 bytes) - simple 8-bit
//    - block_q4_0:  4-bit (32 elem/block, 18 bytes) - legacy 4-bit
//    - block_iq4_nl: 4-bit (32 elem/block, 18 bytes) - non-linear i-quants
//    - block_iq4_xs: 4-bit (256 elem/block, 136 bytes) - extended i-quants
//    - block_q6_K:  6-bit (256 elem/block, 210 bytes) - high precision
//    - block_q5_K:  5-bit (256 elem/block, 176 bytes) - balanced quality/size
//    - block_q3_K:  3-bit (256 elem/block, 110 bytes) - aggressive compression
//    - block_q2_K:  2-bit (256 elem/block, 84 bytes) - maximum compression
//    - block_iq3_s: 3-bit (256 elem/block) - importance-based i-quants
//
// 2. MATRIX-VECTOR KERNELS (DECODE - Single Token)
//    - matvec_q4k:      Q4_K decode (SIMD-optimized, llama.cpp style)
//    - matvec_q8_0:     Q8_0 decode
//    - matvec_q4_0:     Q4_0 decode
//    - matvec_iq4_nl:   IQ4_NL decode (lookup table)
//    - matvec_iq4_xs:   IQ4_XS decode (super-block scales)
//    - matvec_q6k:      Q6_K decode
//    - matvec_q5k:      Q5_K decode
//    - matvec_q3k:      Q3_K decode
//    - matvec_q2k:      Q2_K decode
//    - matvec_iq3_s:    IQ3_S decode
//    - matvec_residual_*: Fused matvec + residual addition
//
// 3. MATRIX-MATRIX KERNELS (PREFILL - Batched)
//    - matmul_q4k:      Q4_K batched multiply
//    - matmul_q8_0:     Q8_0 batched multiply
//    - matmul_q4_0:     Q4_0 batched multiply
//    - matmul_iq4_nl:   IQ4_NL batched multiply
//    - matmul_iq4_xs:   IQ4_XS batched multiply (SIMD-optimized)
//    - matmul_iq3_s:    IQ3_S batched multiply (SIMD-optimized)
//
// 4. FUSED KERNELS
//    - fused_qkv_matvec_q4k:  Q+K+V projections in single dispatch
//    - rms_norm_matvec_*:     RMSNorm + MatVec fusion
//
// 5. ATTENTION KERNELS
//    - attention_decode:              Single query attention (FP32 KV)
//    - multihead_attention_decode:    Multi-head decode (FP32 KV)
//    - multihead_attention_decode_f16kv: Multi-head decode (FP16 KV, fallback)
//    - simdgroup_flash_attention_decode_f16kv_d64/d128: High-perf decode (FP16 KV)
//    - flash_attention_decode_d64/d128: llama.cpp-style decode (NEW - highest perf)
//    - attention_prefill:             Batched attention (FP32 KV)
//    - attention_prefill_f16kv:       Batched attention (FP16 KV)
//    - attention_tree_f16kv:          Tree attention for speculative decoding
//    - paged_attention_decode:        Paged KV cache attention
//    - batched_paged_attention_decode: Batched paged attention
//
// 6. UTILITY KERNELS
//    - copy_to_kv_cache:     Copy Q/K projections to KV cache
//    - silu_elementwise:     SiLU activation
//    - add_vectors:          Vector addition
//    - rope_f32/rope_f16:    Rotary position embeddings
//    - rms_norm:             RMS normalization
//    - softmax:              Softmax activation
//
// =============================================================================
// PERFORMANCE NOTES
// =============================================================================
//
// Memory Bandwidth:
//   - M3 Max theoretical: ~400 GB/s
//   - Good efficiency target: > 50% of theoretical
//   - Matvec kernels are memory-bound (low arithmetic intensity)
//
// Threading Model:
//   - SIMD width: 32 threads (Apple Silicon)
//   - Most kernels use 2-8 SIMD groups per threadgroup (64-256 threads)
//   - Q4_K kernels: 2 SIMD groups (64 threads) matches llama.cpp
//
// Quantization Trade-offs:
//   - Q4_K: Best quality/size for 4-bit (144 bytes/256 elems = 4.5 bits/elem)
//   - Q8_0: Simple, fast, good quality (34 bytes/32 elems = 8.5 bits/elem)
//   - IQ4_*: Better quality than Q4 at same size via non-linear LUT
//   - Q2_K: Maximum compression (84 bytes/256 elems = 2.6 bits/elem)
//
// =============================================================================

#pragma once

// Embedded shader source - all kernels in one compilation unit
static const char* METAL_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// CONSTANTS AND QUANTIZATION BLOCK DEFINITIONS
// =============================================================================

// Super-block size for K-quants (Q4_K, Q6_K, Q5_K, Q3_K, Q2_K)
constant constexpr uint QK_K = 256;

// Metal doesn't have _Pragma in the same way, use [[ ]] attribute hints instead
// Note: Metal uses unroll hints via [[unroll]] or just relies on compiler optimization

// Simple quant block sizes
constant constexpr uint QK8_0 = 32;  // Q8_0 block size

// -----------------------------------------------------------------------------
// Q4_K: 256 elements per super-block, 144 bytes
// -----------------------------------------------------------------------------
// High-quality 4-bit quantization with sub-block scales and mins.
// Memory layout: d[2] + dmin[2] + scales[12] + qs[128] = 144 bytes
//
// Dequantization formula (for 64-element sub-block j):
//   w[i] = d * sc[j] * q[i] - dmin * m[j]
// where sc/m are 6-bit scales/mins packed in the scales array.
//
// The 256 elements are split into 8 groups of 32, with pairs sharing scales:
//   - Groups 0,1 use scales[0..1]
//   - Groups 2,3 use scales[2..3]
//   - etc.
// -----------------------------------------------------------------------------
struct block_q4_K {
    half d;             // super-block scale for quantized scales
    half dmin;          // super-block scale for quantized mins
    uint8_t scales[12]; // 6-bit scales and mins (packed)
    uint8_t qs[128];    // 4-bit quants: 256 values / 2 per byte
};

// -----------------------------------------------------------------------------
// Q8_0: 32 elements per block, 34 bytes
// -----------------------------------------------------------------------------
// Simple 8-bit quantization: w[i] = d * qs[i]
// Highest quality of the standard formats, ~8.5 bits per weight.
// Used for activations and some weight tensors where quality matters.
// -----------------------------------------------------------------------------
struct block_q8_0 {
    half d;           // scale factor
    int8_t qs[32];    // quantized values (-128 to 127)
};

// -----------------------------------------------------------------------------
// Q4_0: 32 elements per block, 18 bytes
// -----------------------------------------------------------------------------
// Legacy 4-bit format with simple symmetric quantization.
// Dequantization: w = d * (q - 8), where q is 4-bit unsigned (0-15)
// Less accurate than Q4_K but simpler and compatible with older models.
// -----------------------------------------------------------------------------
struct block_q4_0 {
    half d;           // scale factor
    uint8_t qs[16];   // 4-bit quants packed 2 per byte (low nibble first)
};

// -----------------------------------------------------------------------------
// Q6_K: 256 elements per super-block, 210 bytes
// -----------------------------------------------------------------------------
// 6-bit quantization with 8-bit per-16-element scales.
// Layout: ql[128] + qh[64] + scales[16] + d[2]
//
// Each 6-bit value is split: 4 low bits in ql, 2 high bits in qh.
// Dequantization: w = d * scale[i/16] * ((ql_part | (qh_part << 4)) - 32)
//
// Higher precision than Q4_K, useful when quality is critical.
// ~6.6 bits per weight effective.
// -----------------------------------------------------------------------------
struct block_q6_K {
    uint8_t ql[128];    // lower 4 bits of 6-bit quants (2 per byte)
    uint8_t qh[64];     // upper 2 bits of 6-bit quants (4 per byte)
    int8_t scales[16];  // 8-bit scale per 16 elements
    half d;             // super-block scale
};

// -----------------------------------------------------------------------------
// Q5_K: 256 elements per super-block, 176 bytes
// -----------------------------------------------------------------------------
// 5-bit quantization with 6-bit sub-block scales/mins.
// Layout: d[2] + dmin[2] + scales[12] + qh[32] + qs[128]
//
// Each 5-bit value is split: 4 low bits in qs, 1 high bit in qh.
// Dequantization: w = d * scale * ((q_low | (q_high << 4))) - dmin * min
//
// Good balance between quality and compression (~5.5 bits per weight).
// -----------------------------------------------------------------------------
struct block_q5_K {
    half d;             // super-block scale for quantized scales
    half dmin;          // super-block scale for quantized mins
    uint8_t scales[12]; // 6-bit scales and mins (packed)
    uint8_t qh[32];     // high bit of quants (1 bit per element)
    uint8_t qs[128];    // low 4 bits of quants (2 per byte)
};

// -----------------------------------------------------------------------------
// Q3_K: 256 elements per super-block, 110 bytes
// -----------------------------------------------------------------------------
// 3-bit quantization for aggressive compression (~3.4 bits per weight).
// Layout: hmask[32] + qs[64] + scales[12] + d[2]
//
// Each weight = 2 low bits from qs + 1 high bit from hmask.
// Uses 6-bit scales like Q4_K/Q5_K.
//
// Noticeable quality loss vs Q4_K but 25% smaller.
// -----------------------------------------------------------------------------
struct block_q3_K {
    uint8_t hmask[32];  // high bit of 3-bit quants (1 per element, bit-packed)
    uint8_t qs[64];     // low 2 bits of 3-bit quants (4 per byte)
    uint8_t scales[12]; // 6-bit scales (packed)
    half d;             // super-block scale
};

// -----------------------------------------------------------------------------
// Q2_K: 256 elements per super-block, 84 bytes
// -----------------------------------------------------------------------------
// 2-bit quantization for maximum compression (~2.6 bits per weight).
// Layout: scales[16] + qs[64] + d[2] + dmin[2]
//
// Dequantization: w = d * (scale & 0xF) * q2 - dmin * (scale >> 4)
// Uses 4-bit scales and mins packed together.
//
// Significant quality loss - use only when model size is critical.
// -----------------------------------------------------------------------------
struct block_q2_K {
    uint8_t scales[16]; // 4-bit scales (low nibble) and mins (high nibble)
    uint8_t qs[64];     // 2-bit quants (4 per byte)
    half d;             // super-block scale
    half dmin;          // super-block min
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

// Helper to extract 6-bit scale and min for dequantization into tiles
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

    // Process K dimension in chunks
    for (uint loop_k = 0; loop_k < K; loop_k += SGMM_NK) {
        // Each weight-loading thread loads 16 elements at K position: loop_k + 16*il0
        // Thread il0=0 loads first 16 elements, il0=1 loads second 16 elements
        // This covers all SGMM_NK=32 elements per iteration
        uint k_position = loop_k + 16 * il0;
        uint block_idx = k_position / QK_K;
        uint within_block = (k_position % QK_K) / 16;

        // Load and dequantize weights into sa
        // Weight W[n, k] for output dimension n = r0 + lr0
        half4x4 temp_a;
        if (lr0 < nr0 && block_idx < num_blocks_k) {
            device const block_q4_K* w_block = &weights[(r0 + lr0) * num_blocks_k + block_idx];
            dequantize_q4_k_to_half4x4(w_block, within_block, temp_a);
        } else {
            temp_a = half4x4(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Store weight tile with layout matching simdgroup_load expectations
        // Pre-compute loop-invariant values
        const short sy_w = lr0 / 8;               // N block index (0..7 for NR0=64)
        const short lx_w = lr0 % 8;               // Within N block

        // Unrolled stores with pre-computed indices
        // First 8 elements (i=0..7): sx = 2*il0, varying ly = i%8
        {
            const short ib0 = 8 * (2 * il0) + sy_w;
            threadgroup half* sa_dst = sa + 64 * ib0 + lx_w;
            sa_dst[0]  = temp_a[0][0];  // i=0, ly=0
            sa_dst[8]  = temp_a[0][1];  // i=1, ly=1
            sa_dst[16] = temp_a[0][2];  // i=2, ly=2
            sa_dst[24] = temp_a[0][3];  // i=3, ly=3
            sa_dst[32] = temp_a[1][0];  // i=4, ly=4
            sa_dst[40] = temp_a[1][1];  // i=5, ly=5
            sa_dst[48] = temp_a[1][2];  // i=6, ly=6
            sa_dst[56] = temp_a[1][3];  // i=7, ly=7
        }
        // Second 8 elements (i=8..15): sx = 2*il0+1, varying ly = i%8
        {
            const short ib1 = 8 * (2 * il0 + 1) + sy_w;
            threadgroup half* sa_dst = sa + 64 * ib1 + lx_w;
            sa_dst[0]  = temp_a[2][0];  // i=8,  ly=0
            sa_dst[8]  = temp_a[2][1];  // i=9,  ly=1
            sa_dst[16] = temp_a[2][2];  // i=10, ly=2
            sa_dst[24] = temp_a[2][3];  // i=11, ly=3
            sa_dst[32] = temp_a[3][0];  // i=12, ly=4
            sa_dst[40] = temp_a[3][1];  // i=13, ly=5
            sa_dst[48] = temp_a[3][2];  // i=14, ly=6
            sa_dst[56] = temp_a[3][3];  // i=15, ly=7
        }

        // Load input into sb - optimized with vectorized loads
        // Input X[m, k] for batch index m = r1 + lr1
        if (lr1 < nr1) {
            // Pre-compute loop-invariant values
            const short sx = tiitg % NL1;       // K block index
            const short sy = lr1 / 8;           // M block index
            const short ly = lr1 % 8;           // Within M block
            const short ib = 4 * sx + sy;
            threadgroup half* sb_dst = sb + 64 * ib + 8 * ly;

            device const float* x_ptr = X + (r1 + lr1) * K + loop_k + iy;

            // Vectorized load and convert: read 8 floats as 2x float4, convert to half
            if (loop_k + iy + 8 <= K) {
                // Full vector load - no bounds checking needed
                float4 v0 = *((device const float4*)(x_ptr));
                float4 v1 = *((device const float4*)(x_ptr + 4));
                half4 h0 = half4(v0);
                half4 h1 = half4(v1);
                *((threadgroup half4*)(sb_dst)) = h0;
                *((threadgroup half4*)(sb_dst + 4)) = h1;
            } else {
                // Partial load with bounds checking
                for (short i = 0; i < 8; ++i) {
                    sb_dst[i] = (loop_k + iy + i < K) ? half(x_ptr[i]) : half(0);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Simdgroup matrix multiply-accumulate
        // Each simdgroup computes 32x16 output: 32 N rows x 16 M rows
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);  // Weight tiles
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);  // Input tiles

        #pragma clang loop unroll(full)
        for (short ik = 0; ik < SGMM_NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            // Load 4 weight 8x8 tiles (32 N rows)
            simdgroup_load(ma[0], lsma + 64 * 0, 8, 0, false);
            simdgroup_load(ma[1], lsma + 64 * 1, 8, 0, false);
            simdgroup_load(ma[2], lsma + 64 * 2, 8, 0, false);
            simdgroup_load(ma[3], lsma + 64 * 3, 8, 0, false);

            simdgroup_barrier(mem_flags::mem_none);

            // Load 2 input 8x8 tiles (16 M rows)
            simdgroup_load(mb[0], lsmb + 64 * 0, 8, 0, false);
            simdgroup_load(mb[1], lsmb + 64 * 1, 8, 0, false);

            simdgroup_barrier(mem_flags::mem_none);

            // Compute: mc[i] = mb[i/4] @ ma[i%4]
            // Result: 8 matrices of 8x8 covering 16x32 output
            simdgroup_multiply_accumulate(mc[0], mb[0], ma[0], mc[0]);
            simdgroup_multiply_accumulate(mc[1], mb[0], ma[1], mc[1]);
            simdgroup_multiply_accumulate(mc[2], mb[0], ma[2], mc[2]);
            simdgroup_multiply_accumulate(mc[3], mb[0], ma[3], mc[3]);
            simdgroup_multiply_accumulate(mc[4], mb[1], ma[0], mc[4]);
            simdgroup_multiply_accumulate(mc[5], mb[1], ma[1], mc[5]);
            simdgroup_multiply_accumulate(mc[6], mb[1], ma[2], mc[6]);
            simdgroup_multiply_accumulate(mc[7], mb[1], ma[3], mc[7]);

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
        // Full tile - store with row-major layout
        // Y[sg_m + m_off, sg_n + n_off] = mc[...][m_local][n_local]
        device float* C = Y + sg_m * N + sg_n;

        // Unrolled stores: i/4 gives M block, i%4 gives N block
        simdgroup_store(mc[0], C + 0 * N + 0,  N, 0, false);
        simdgroup_store(mc[1], C + 0 * N + 8,  N, 0, false);
        simdgroup_store(mc[2], C + 0 * N + 16, N, 0, false);
        simdgroup_store(mc[3], C + 0 * N + 24, N, 0, false);
        simdgroup_store(mc[4], C + 8 * N + 0,  N, 0, false);
        simdgroup_store(mc[5], C + 8 * N + 8,  N, 0, false);
        simdgroup_store(mc[6], C + 8 * N + 16, N, 0, false);
        simdgroup_store(mc[7], C + 8 * N + 24, N, 0, false);
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

// SIMD-optimized batched FP16 matmul for large K dimension
// Each simdgroup handles one output row of one token
kernel void matmul_f16_simd(
    device const float* X          [[buffer(0)]],
    device const half* W           [[buffer(1)]],
    device float* Y                [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& K               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]]
) {
    uint row = tgid.y;  // token index
    uint col = tgid.x;  // output dimension index

    if (row >= M || col >= N) return;

    device const float* x_row = X + row * K;
    device const half* w_row = W + col * K;

    // Each thread in simdgroup processes K/32 elements
    float local_sum = 0.0f;
    for (uint i = simd_lane; i < K; i += 32) {
        local_sum += x_row[i] * float(w_row[i]);
    }

    // SIMD reduction
    float sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        Y[row * N + col] = sum;
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
    device const float* q_ptr = Q + head_idx * seq_q * head_dim;
    device const half* k_ptr = K + kv_head * seq_kv * head_dim;
    device const half* v_ptr = V + kv_head * seq_kv * head_dim;
    device float* out_ptr = output + head_idx * seq_q * head_dim;

    // Load Q into shared memory (convert float to half)
    #pragma clang loop unroll(full)
    for (uint i = tid; i < FA_Q_TILE * FA_DK; i += FA_NSG * FA_NW) {
        uint q_row = i / FA_DK;
        uint q_col = i % FA_DK;
        uint global_q_pos = q_base + q_row;

        sq[q_row * FA_DK + q_col] = (global_q_pos < seq_q && q_col < head_dim) ?
            half(q_ptr[global_q_pos * head_dim + q_col]) : half(0.0f);
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
    #pragma clang loop unroll(full)
    for (uint jj = 0; jj < NQ; jj++) {
        uint j = jj * FA_NSG + simd_id;
        uint global_q_pos = q_base + j;

        if (global_q_pos >= seq_q) continue;

        float inv_sum = 1.0f / (S[jj] + 1e-6f);

        for (uint d = simd_lane; d < FA_DK; d += FA_NW) {
            out_ptr[global_q_pos * head_dim + d] = float(so[j * FA_DK + d]) * inv_sum;
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
