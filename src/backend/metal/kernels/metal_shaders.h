// =============================================================================
// Metal Shader Source for LLM Inference Kernels
// =============================================================================
//
// This file concatenates all Metal compute kernels for Granite's GPU-accelerated
// LLM inference. Kernels are embedded as strings and compiled at runtime.
//
// The shaders are split into logical modules for maintainability:
//   - metal_shaders_common.h:    Block structures, constants, helpers (~100 lines)
//   - metal_shaders_quant.h:     Quantized matmul/matvec kernels (~3200 lines)
//   - metal_shaders_util.h:      RMSNorm, RoPE, SiLU, etc (~350 lines)
//   - metal_shaders_attention.h: Attention kernels (~3500 lines)
//
// =============================================================================
// KERNEL INDEX
// =============================================================================
//
// QUANTIZED MATVEC (Decode):
//   matvec_q4k, matvec_q8_0, matvec_q4_0, matvec_iq4_nl, matvec_iq4_xs,
//   matvec_iq3_s, matvec_q6_k, matvec_q3_k, matvec_q2_k, matvec_q5_k, matvec_f16
//
// QUANTIZED MATMUL (Prefill):
//   matmul_q4k, matmul_q4k_simdgroup, matmul_q8_0, matmul_q4_0, matmul_iq4_nl,
//   matmul_iq4_xs, matmul_iq3_s, matmul_q6_k, matmul_q3_k, matmul_q2_k,
//   matmul_q5_k, matmul_f16
//
// UTILITY:
//   rms_norm, rms_norm_f16, rms_norm_batch, rms_norm_batch_f16,
//   silu, silu_mul, elementwise_mul, elementwise_add,
//   rope, rope_multihead, embedding_lookup, softmax_row
//
// ATTENTION:
//   attention_decode, multihead_attention_decode, multihead_attention_decode_f16kv,
//   flash_attention_prefill, flash_attention_decode_d64, flash_attention_decode_d128,
//   simdgroup_flash_attention_decode_f16kv_d64, simdgroup_flash_attention_decode_f16kv_d128,
//   attention_tree_f16kv, attention_tree_nocontext_f16kv,
//   paged_attention_decode, paged_kv_cache_append, batched_paged_attention_decode,
//   paged_flash_attention_decode_d64, paged_flash_attention_decode_d128,
//   kv_cache_append, kv_cache_append_f16
//
// FUSED:
//   fused_qkv_matvec_q4k, rms_norm_matvec_*, rms_norm_dual_matvec_*, matvec_residual_*
//
// =============================================================================

#pragma once

#include "metal_shaders_common.h"
#include "metal_shaders_quant.h"
#include "metal_shaders_util.h"
#include "metal_shaders_attention.h"

#include <string>

// Concatenate all shader sources into a single compilation unit
static const std::string METAL_SHADER_SOURCE =
    std::string(METAL_SHADER_COMMON) +
    std::string(METAL_SHADER_QUANT) +
    std::string(METAL_SHADER_UTIL) +
    std::string(METAL_SHADER_ATTENTION);
