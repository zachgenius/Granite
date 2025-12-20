// GPU Q4_K Matmul Benchmark - tests function constant kernel variants
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

        std::cout << "GPU Q4_K Matmul Benchmark - Function Constants\n";
        std::cout << "================================================\n";
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

        // Create specialized kernels with function constants
        // FC_MUL_MM + 0 = FC_mm_bc_inp, FC_MUL_MM + 1 = FC_mm_bc_out
        constexpr uint32_t FC_MUL_MM = 100;

        auto create_pipeline = [&](bool bc_inp, bool bc_out) -> id<MTLComputePipelineState> {
            MTLFunctionConstantValues* fc = [[MTLFunctionConstantValues alloc] init];
            [fc setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:FC_MUL_MM + 0];
            [fc setConstantValue:&bc_out type:MTLDataTypeBool atIndex:FC_MUL_MM + 1];

            NSError* err = nil;
            id<MTLFunction> func = [library newFunctionWithName:@"matmul_q4k_simdgroup"
                                                 constantValues:fc
                                                          error:&err];
            if (!func) {
                std::cerr << "Function creation failed: " << [[err localizedDescription] UTF8String] << "\n";
                return nil;
            }

            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
            if (!pipeline) {
                std::cerr << "Pipeline failed: " << [[err localizedDescription] UTF8String] << "\n";
            }
            return pipeline;
        };

        // Create all 4 variants
        id<MTLComputePipelineState> pipe_00 = create_pipeline(false, false);  // No bounds check
        id<MTLComputePipelineState> pipe_01 = create_pipeline(false, true);   // Output bounds only
        id<MTLComputePipelineState> pipe_10 = create_pipeline(true, false);   // Input bounds only
        id<MTLComputePipelineState> pipe_11 = create_pipeline(true, true);    // Full bounds check

        if (!pipe_00 || !pipe_11) {
            std::cerr << "Failed to create required pipelines\n";
            return 1;
        }

        // Test cases matching TinyLlama prefill
        struct TestCase { uint32_t M, K, N; const char* name; bool aligned; };
        std::vector<TestCase> tests = {
            {32, 2048, 2048, "Attn Q/K/V pp32", true},
            {64, 2048, 2048, "Attn Q/K/V pp64", true},
            {128, 2048, 2048, "Attn Q/K/V pp128", true},
            {256, 2048, 2048, "Attn Q/K/V pp256", true},
            {512, 2048, 2048, "Attn Q/K/V pp512", true},
            {32, 2048, 5632, "FFN gate/up pp32", true},
            {128, 2048, 5632, "FFN gate/up pp128", true},
            {512, 2048, 5632, "FFN gate/up pp512", true},
            {32, 5632, 2048, "FFN down pp32", true},
            {128, 5632, 2048, "FFN down pp128", true},
            // Unaligned cases (require bounds checking)
            {33, 2048, 2048, "Unaligned M=33", false},
            {64, 2048, 2049, "Unaligned N=2049", false},
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
                  << std::setw(11) << "00(us)"
                  << std::setw(11) << "11(us)"
                  << std::setw(9) << "Speedup" << "\n";
        std::cout << std::string(77, '-') << "\n";

        double total_00_time = 0, total_11_time = 0;

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
            float* X = (float*)[X_buf contents];
            uint8_t* W_ptr = (uint8_t*)[W_buf contents];
            for (size_t i = 0; i < size_t(M) * K; i++) X[i] = 0.01f * (i % 100);
            for (size_t i = 0; i < W_size; i++) W_ptr[i] = rand() % 256;

            // Use appropriate kernel based on alignment
            id<MTLComputePipelineState> fast_pipe = tc.aligned ? pipe_00 : pipe_11;

            // Benchmark: compare no-bounds-check vs full-bounds-check
            double us_00 = tc.aligned ? bench_kernel(pipe_00, X_buf, W_buf, Y_buf, M, K, N) : 0;
            double us_11 = bench_kernel(pipe_11, X_buf, W_buf, Y_buf, M, K, N);

            if (tc.aligned) {
                double speedup = us_11 / us_00;
                total_00_time += us_00;
                total_11_time += us_11;

                std::cout << std::setw(22) << tc.name
                          << std::setw(8) << M
                          << std::setw(8) << K
                          << std::setw(8) << N
                          << std::setw(11) << std::fixed << std::setprecision(1) << us_00
                          << std::setw(11) << std::setprecision(1) << us_11
                          << std::setw(8) << std::setprecision(2) << speedup << "x\n";
            } else {
                std::cout << std::setw(22) << tc.name
                          << std::setw(8) << M
                          << std::setw(8) << K
                          << std::setw(8) << N
                          << std::setw(11) << "-"
                          << std::setw(11) << std::fixed << std::setprecision(1) << us_11
                          << std::setw(9) << "(N/A)\n";
            }
        }

        std::cout << std::string(77, '-') << "\n";
        std::cout << std::setw(54) << "Total (aligned):"
                  << std::setw(11) << std::fixed << std::setprecision(1) << total_00_time
                  << std::setw(11) << std::setprecision(1) << total_11_time
                  << std::setw(8) << std::setprecision(2) << (total_11_time / total_00_time) << "x\n";

        std::cout << "\nFunction constant variants:\n";
        std::cout << "  00 (bc_inp=false, bc_out=false): No bounds checking - fastest\n";
        std::cout << "  11 (bc_inp=true,  bc_out=true):  Full bounds checking - safe for any dimension\n";
    }
    return 0;
}
