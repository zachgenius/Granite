// GPU Q4_K Matmul Benchmark - compares f32 vs f16 I/O kernels
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

// Include the kernel source
#include "backend/metal/kernels/metal_shaders.h"

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "No Metal device\n";
            return 1;
        }

        std::cout << "GPU Q4_K Matmul Benchmark - F32 vs F16 I/O\n";
        std::cout << "==========================================\n";
        std::cout << "Device: " << [device.name UTF8String] << "\n\n";

        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Compile kernels
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;

        NSString* source = @(METAL_SHADER_SOURCE.c_str());
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:options
                                                        error:&error];
        if (!library) {
            std::cerr << "Failed to compile: " << [[error localizedDescription] UTF8String] << "\n";
            return 1;
        }

        // Get kernels
        id<MTLFunction> fast_f32_func = [library newFunctionWithName:@"matmul_q4k_simdgroup_fast"];
        id<MTLFunction> fast_f16_func = [library newFunctionWithName:@"matmul_q4k_simdgroup_fast_f16"];

        if (!fast_f32_func) {
            std::cerr << "matmul_q4k_simdgroup_fast not found\n";
            return 1;
        }
        if (!fast_f16_func) {
            std::cerr << "matmul_q4k_simdgroup_fast_f16 not found\n";
            return 1;
        }

        id<MTLComputePipelineState> f32_pipeline = [device newComputePipelineStateWithFunction:fast_f32_func error:&error];
        id<MTLComputePipelineState> f16_pipeline = [device newComputePipelineStateWithFunction:fast_f16_func error:&error];

        if (!f32_pipeline || !f16_pipeline) {
            std::cerr << "Pipeline failed: " << [[error localizedDescription] UTF8String] << "\n";
            return 1;
        }

        // Test cases matching TinyLlama prefill (must be aligned for fast kernels)
        struct TestCase { uint32_t M, K, N; const char* name; };
        std::vector<TestCase> tests = {
            {32, 2048, 2048, "Attn Q/K/V pp32"},
            {64, 2048, 2048, "Attn Q/K/V pp64"},
            {128, 2048, 2048, "Attn Q/K/V pp128"},
            {256, 2048, 2048, "Attn Q/K/V pp256"},
            {512, 2048, 2048, "Attn Q/K/V pp512"},
            {32, 2048, 5632, "FFN gate/up pp32"},
            {128, 2048, 5632, "FFN gate/up pp128"},
            {512, 2048, 5632, "FFN gate/up pp512"},
            {32, 5632, 2048, "FFN down pp32"},
            {128, 5632, 2048, "FFN down pp128"},
        };

        constexpr uint32_t NR0 = 64, NR1 = 32;

        auto bench_kernel = [&](id<MTLComputePipelineState> pipeline,
                                id<MTLBuffer> X_buf, id<MTLBuffer> W_buf, id<MTLBuffer> Y_buf,
                                uint32_t M, uint32_t K, uint32_t N) -> double {
            uint32_t num_m_tiles = (M + NR1 - 1) / NR1;
            uint32_t num_n_tiles = (N + NR0 - 1) / NR0;

            // Warmup
            for (int w = 0; w < 5; w++) {
                id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:pipeline];
                [enc setBuffer:X_buf offset:0 atIndex:0];
                [enc setBuffer:W_buf offset:0 atIndex:1];
                [enc setBuffer:Y_buf offset:0 atIndex:2];
                [enc setBytes:&M length:sizeof(M) atIndex:3];
                [enc setBytes:&K length:sizeof(K) atIndex:4];
                [enc setBytes:&N length:sizeof(N) atIndex:5];
                [enc setThreadgroupMemoryLength:8192 atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(num_m_tiles, num_n_tiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];
            }

            // Benchmark
            int runs = 50;
            auto start = std::chrono::high_resolution_clock::now();

            for (int r = 0; r < runs; r++) {
                id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:pipeline];
                [enc setBuffer:X_buf offset:0 atIndex:0];
                [enc setBuffer:W_buf offset:0 atIndex:1];
                [enc setBuffer:Y_buf offset:0 atIndex:2];
                [enc setBytes:&M length:sizeof(M) atIndex:3];
                [enc setBytes:&K length:sizeof(K) atIndex:4];
                [enc setBytes:&N length:sizeof(N) atIndex:5];
                [enc setThreadgroupMemoryLength:8192 atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(num_m_tiles, num_n_tiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];
            }

            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::micro>(end - start).count() / runs;
        };

        std::cout << std::setw(22) << "Test"
                  << std::setw(8) << "M"
                  << std::setw(8) << "K"
                  << std::setw(8) << "N"
                  << std::setw(11) << "F32(us)"
                  << std::setw(11) << "F16(us)"
                  << std::setw(9) << "Speedup" << "\n";
        std::cout << std::string(77, '-') << "\n";

        double total_f32_time = 0, total_f16_time = 0;

        for (const auto& tc : tests) {
            uint32_t M = tc.M, K = tc.K, N = tc.N;

            // Q4_K: 256 elements per block, 144 bytes per block
            size_t num_blocks = (size_t(K) * N + 255) / 256;
            size_t W_size = num_blocks * 144;
            size_t X_size_f32 = size_t(M) * K * sizeof(float);
            size_t X_size_f16 = size_t(M) * K * sizeof(uint16_t);  // half
            size_t Y_size_f32 = size_t(M) * N * sizeof(float);
            size_t Y_size_f16 = size_t(M) * N * sizeof(uint16_t);  // half

            // F32 buffers
            id<MTLBuffer> X_buf_f32 = [device newBufferWithLength:X_size_f32 options:MTLResourceStorageModeShared];
            id<MTLBuffer> W_buf = [device newBufferWithLength:W_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> Y_buf_f32 = [device newBufferWithLength:Y_size_f32 options:MTLResourceStorageModeShared];

            // F16 input kernel uses half input, float output (llama.cpp style)
            id<MTLBuffer> X_buf_f16 = [device newBufferWithLength:X_size_f16 options:MTLResourceStorageModeShared];
            id<MTLBuffer> Y_buf_f16 = [device newBufferWithLength:Y_size_f32 options:MTLResourceStorageModeShared];  // Float output!

            // Initialize with test data
            float* X_f32 = (float*)[X_buf_f32 contents];
            uint16_t* X_f16 = (uint16_t*)[X_buf_f16 contents];
            uint8_t* W_ptr = (uint8_t*)[W_buf contents];

            for (size_t i = 0; i < size_t(M) * K; i++) {
                float val = 0.01f * (i % 100);
                X_f32[i] = val;
                // Convert to half (simple approximation for benchmark)
                uint32_t f = *(uint32_t*)&val;
                uint16_t h = ((f >> 16) & 0x8000) | (((f >> 13) - (127 - 15) << 10) & 0x7c00) | ((f >> 13) & 0x03ff);
                X_f16[i] = h;
            }
            for (size_t i = 0; i < W_size; i++) W_ptr[i] = rand() % 256;

            // Benchmark both kernels
            double us_f32 = bench_kernel(f32_pipeline, X_buf_f32, W_buf, Y_buf_f32, M, K, N);
            double us_f16 = bench_kernel(f16_pipeline, X_buf_f16, W_buf, Y_buf_f16, M, K, N);

            double speedup = us_f32 / us_f16;
            total_f32_time += us_f32;
            total_f16_time += us_f16;

            std::cout << std::setw(22) << tc.name
                      << std::setw(8) << M
                      << std::setw(8) << K
                      << std::setw(8) << N
                      << std::setw(11) << std::fixed << std::setprecision(1) << us_f32
                      << std::setw(11) << std::setprecision(1) << us_f16
                      << std::setw(8) << std::setprecision(2) << speedup << "x\n";
        }

        std::cout << std::string(77, '-') << "\n";
        std::cout << std::setw(54) << "Total:"
                  << std::setw(11) << std::fixed << std::setprecision(1) << total_f32_time
                  << std::setw(11) << std::setprecision(1) << total_f16_time
                  << std::setw(8) << std::setprecision(2) << (total_f32_time / total_f16_time) << "x\n";

        std::cout << "\nBandwidth savings with F16 I/O:\n";
        std::cout << "  F32: X[M*K*4] + Y[M*N*4] = " << (2*512*2048*4 + 2*512*5632*4)/1e6 << " MB per FFN layer\n";
        std::cout << "  F16: X[M*K*2] + Y[M*N*2] = " << (2*512*2048*2 + 2*512*5632*2)/1e6 << " MB per FFN layer\n";
        std::cout << "  Reduction: 50%\n";
    }
    return 0;
}
