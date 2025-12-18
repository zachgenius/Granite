// kernel_types.h - Common types and structures for Metal kernels
// This file is included at the start of the combined shader source
#pragma once

// Common type definitions and structures for all Metal kernels
static const char* KERNEL_TYPES_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Quantization Constants
// =============================================================================

constant constexpr uint QK_K = 256;   // K-quant super-block size
constant constexpr uint QK8_0 = 32;   // Q8_0 block size
constant constexpr uint QK4_0 = 32;   // Q4_0 block size

// =============================================================================
// Quantized Block Structures
// =============================================================================

// Q4_K: 256 elements per super-block, 144 bytes
// 4-bit quantization with 6-bit sub-block scales/mins
struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

// Q8_0: 32 elements per block, 34 bytes
// Simple format: FP16 scale + 32 int8 values
struct block_q8_0 {
    half d;           // scale
    int8_t qs[32];    // quantized values
};

// Q4_0: 32 elements per block, 18 bytes
// Legacy format: FP16 scale + 16 bytes (4-bit quants, 2 per byte)
// Dequantization: w = d * (q - 8) where q is 0-15
struct block_q4_0 {
    half d;           // scale
    uint8_t qs[16];   // 4-bit quants (2 per byte)
};

// Q6_K: 256 elements per super-block, 210 bytes
// 6-bit quantization with 8-bit sub-block scales
struct block_q6_K {
    uint8_t ql[128];    // lower 4 bits of 6-bit quants (2 per byte)
    uint8_t qh[64];     // upper 2 bits of 6-bit quants (4 per byte)
    int8_t scales[16];  // 8-bit scale per 16 elements
    half d;             // super-block scale
};

// Q5_K: 256 elements per super-block, 176 bytes
// 5-bit quantization with 6-bit sub-block scales/mins
struct block_q5_K {
    half d;             // super-block scale for quantized scales
    half dmin;          // super-block scale for quantized mins
    uint8_t scales[12]; // scales and mins, quantized with 6 bits
    uint8_t qh[32];     // high bit of quants (1 bit per element)
    uint8_t qs[128];    // low 4 bits of quants (2 per byte)
};

// Q3_K: 256 elements per super-block, 110 bytes
// 3-bit quantization with 6-bit sub-block scales
struct block_q3_K {
    uint8_t hmask[32];  // high bit of 3-bit quants (1 bit per element)
    uint8_t qs[64];     // low 2 bits of 3-bit quants (4 per byte)
    uint8_t scales[12]; // scales, quantized with 6 bits
    half d;             // super-block scale
};

// Q2_K: 256 elements per super-block, 84 bytes
// 2-bit quantization with 4-bit sub-block scales/mins
struct block_q2_K {
    uint8_t scales[16]; // 4-bit scales (low nibble) and mins (high nibble)
    uint8_t qs[64];     // 2-bit quants (4 per byte)
    half d;             // super-block scale
    half dmin;          // super-block min
};

// =============================================================================
// Helper Functions
// =============================================================================

inline void get_scale_min_k4(int j, const device uint8_t* q, thread uint8_t& sc, thread uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
)";
