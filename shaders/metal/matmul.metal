#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Matrix Multiplication Kernels
// =============================================================================

// Naive MatMul: C[M,N] = A[M,K] @ B[K,N]
kernel void matmul_naive_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint row = pos.y;
    uint col = pos.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (uint k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled MatMul for better cache utilization
// Tile size: 16x16
#define TILE_SIZE 16

kernel void matmul_tiled_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]])
{
    // Shared memory for tiles
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    uint row = gid.y;
    uint col = gid.x;
    uint local_row = tid.y;
    uint local_col = tid.x;

    float sum = 0.0f;

    // Loop over tiles
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        // Load tile of A into shared memory
        uint a_col = t * TILE_SIZE + local_col;
        if (row < M && a_col < K) {
            As[local_row][local_col] = A[row * K + a_col];
        } else {
            As[local_row][local_col] = 0.0f;
        }

        // Load tile of B into shared memory
        uint b_row = t * TILE_SIZE + local_row;
        if (b_row < K && col < N) {
            Bs[local_row][local_col] = B[b_row * N + col];
        } else {
            Bs[local_row][local_col] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[local_row][k] * Bs[k][local_col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// FP16 tiled MatMul
kernel void matmul_tiled_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]])
{
    threadgroup half As[TILE_SIZE][TILE_SIZE];
    threadgroup half Bs[TILE_SIZE][TILE_SIZE];

    uint row = gid.y;
    uint col = gid.x;
    uint local_row = tid.y;
    uint local_col = tid.x;

    float sum = 0.0f;  // Accumulate in FP32 for precision

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + local_col;
        if (row < M && a_col < K) {
            As[local_row][local_col] = A[row * K + a_col];
        } else {
            As[local_row][local_col] = half(0.0h);
        }

        uint b_row = t * TILE_SIZE + local_row;
        if (b_row < K && col < N) {
            Bs[local_row][local_col] = B[b_row * N + col];
        } else {
            Bs[local_row][local_col] = half(0.0h);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(As[local_row][k]) * float(Bs[k][local_col]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = half(sum);
    }
}

// Transposed B: C[M,N] = A[M,K] @ B^T[N,K]
kernel void matmul_transb_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],  // B is stored as [N, K]
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    uint row = gid.y;
    uint col = gid.x;
    uint local_row = tid.y;
    uint local_col = tid.x;

    float sum = 0.0f;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + local_col;
        if (row < M && a_col < K) {
            As[local_row][local_col] = A[row * K + a_col];
        } else {
            As[local_row][local_col] = 0.0f;
        }

        // B is transposed, so B[col, k] instead of B[k, col]
        uint b_k = t * TILE_SIZE + local_row;
        if (col < N && b_k < K) {
            Bs[local_row][local_col] = B[col * K + b_k];
        } else {
            Bs[local_row][local_col] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[local_row][k] * Bs[k][local_col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Batched MatMul: C[B,M,N] = A[B,M,K] @ B[B,K,N]
kernel void batched_matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (b >= batch || row >= M || col >= N) return;

    uint a_offset = b * M * K;
    uint b_offset = b * K * N;
    uint c_offset = b * M * N;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[a_offset + row * K + k] * B[b_offset + k * N + col];
    }
    C[c_offset + row * N + col] = sum;
}
