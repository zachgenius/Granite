// =============================================================================
// Metal Shader Common Definitions
// =============================================================================
// Constants, quantization block structures, and helper functions shared by all kernels.
// This file is concatenated into METAL_SHADER_SOURCE at compile time.
// =============================================================================

#pragma once

// Common shader source - constants and block definitions
static const char* METAL_SHADER_COMMON = R"(
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// CONSTANTS AND QUANTIZATION BLOCK DEFINITIONS
// =============================================================================

// Super-block size for K-quants (Q4_K, Q6_K, Q5_K, Q3_K, Q2_K)
constant constexpr uint QK_K = 256;

// Simple quant block sizes
constant constexpr uint QK8_0 = 32;  // Q8_0 block size

// -----------------------------------------------------------------------------
// Q4_K: 256 elements per super-block, 144 bytes
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
struct block_q8_0 {
    half d;           // scale factor
    int8_t qs[32];    // quantized values (-128 to 127)
};

// -----------------------------------------------------------------------------
// Q4_0: 32 elements per block, 18 bytes
// -----------------------------------------------------------------------------
struct block_q4_0 {
    half d;           // scale factor
    uint8_t qs[16];   // 4-bit quants packed 2 per byte (low nibble first)
};

// -----------------------------------------------------------------------------
// Q6_K: 256 elements per super-block, 210 bytes
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
struct block_q3_K {
    uint8_t hmask[32];  // high bit of 3-bit quants (1 per element, bit-packed)
    uint8_t qs[64];     // low 2 bits of 3-bit quants (4 per byte)
    uint8_t scales[12]; // 6-bit scales (packed)
    half d;             // super-block scale
};

// -----------------------------------------------------------------------------
// Q2_K: 256 elements per super-block, 84 bytes
// -----------------------------------------------------------------------------
struct block_q2_K {
    uint8_t scales[16]; // 4-bit scales (low nibble) and mins (high nibble)
    uint8_t qs[64];     // 2-bit quants (4 per byte)
    half d;             // super-block scale
    half dmin;          // super-block min
};

// Helper function for K4 scale extraction
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
