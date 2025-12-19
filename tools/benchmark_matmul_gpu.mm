// GPU Q4_K Matmul Benchmark - isolates kernel performance
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
        
        std::cout << "GPU Q4_K Matmul Benchmark\n";
        std::cout << "=========================\n";
        std::cout << "Device: " << [device.name UTF8String] << "\n\n";
        
        id<MTLCommandQueue> queue = [device newCommandQueue];
        
        // Compile kernels
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;
        
        NSString* source = @(METAL_SHADER_SOURCE);
        id<MTLLibrary> library = [device newLibraryWithSource:source 
                                                      options:options 
                                                        error:&error];
        if (!library) {
            std::cerr << "Failed to compile: " << [[error localizedDescription] UTF8String] << "\n";
            return 1;
        }
        
        // Get kernels
        id<MTLFunction> simdgroup_func = [library newFunctionWithName:@"matmul_q4k_simdgroup"];
        id<MTLFunction> tiled_func = [library newFunctionWithName:@"matmul_q4k_tiled"];
        
        if (!simdgroup_func) {
            std::cerr << "matmul_q4k_simdgroup not found\n";
            return 1;
        }
        
        id<MTLComputePipelineState> simdgroup_pipeline = [device newComputePipelineStateWithFunction:simdgroup_func error:&error];
        id<MTLComputePipelineState> tiled_pipeline = tiled_func ? 
            [device newComputePipelineStateWithFunction:tiled_func error:&error] : nil;
        
        if (!simdgroup_pipeline) {
            std::cerr << "Pipeline failed: " << [[error localizedDescription] UTF8String] << "\n";
            return 1;
        }
        
        // Test cases matching TinyLlama prefill
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
        
        std::cout << std::setw(22) << "Test" 
                  << std::setw(8) << "M" 
                  << std::setw(8) << "K"
                  << std::setw(8) << "N"
                  << std::setw(12) << "Time(us)"
                  << std::setw(12) << "TFLOPS"
                  << std::setw(12) << "GB/s" << "\n";
        std::cout << std::string(82, '-') << "\n";
        
        for (const auto& tc : tests) {
            uint32_t M = tc.M, K = tc.K, N = tc.N;
            
            // Q4_K: 256 elements per block, 144 bytes per block
            size_t num_blocks = (size_t(K) * N + 255) / 256;
            size_t W_size = num_blocks * 144;
            size_t X_size = size_t(M) * K * sizeof(float);
            size_t Y_size = size_t(M) * N * sizeof(float);
            
            id<MTLBuffer> X_buf = [device newBufferWithLength:X_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> W_buf = [device newBufferWithLength:W_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> Y_buf = [device newBufferWithLength:Y_size options:MTLResourceStorageModeShared];
            
            // Initialize with test data
            float* X_ptr = (float*)[X_buf contents];
            uint8_t* W_ptr = (uint8_t*)[W_buf contents];
            for (size_t i = 0; i < size_t(M) * K; i++) X_ptr[i] = 0.01f * (i % 100);
            for (size_t i = 0; i < W_size; i++) W_ptr[i] = rand() % 256;
            
            // Dispatch parameters for simdgroup kernel
            constexpr uint32_t NR0 = 64, NR1 = 32;
            uint32_t num_m_tiles = (M + NR1 - 1) / NR1;
            uint32_t num_n_tiles = (N + NR0 - 1) / NR0;
            
            // Warmup
            for (int w = 0; w < 5; w++) {
                id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:simdgroup_pipeline];
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
                [enc setComputePipelineState:simdgroup_pipeline];
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
            double us = std::chrono::duration<double, std::micro>(end - start).count() / runs;
            
            // Metrics
            double flops = 2.0 * M * K * N;
            double tflops = flops / us / 1e6;
            
            // Bandwidth: read X (M*K*4), read W (num_blocks*144), write Y (M*N*4)
            double bytes = X_size + W_size + Y_size;
            double gb_s = bytes / us / 1e3;
            
            std::cout << std::setw(22) << tc.name
                      << std::setw(8) << M
                      << std::setw(8) << K
                      << std::setw(8) << N
                      << std::setw(12) << std::fixed << std::setprecision(1) << us
                      << std::setw(12) << std::setprecision(2) << tflops
                      << std::setw(12) << std::setprecision(1) << gb_s << "\n";
        }
        
        std::cout << "\nReference: M3 Max theoretical ~15 TFLOPS FP32, ~400 GB/s bandwidth\n";
    }
    return 0;
}
