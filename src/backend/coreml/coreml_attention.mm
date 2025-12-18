// CoreML/MPSGraph Attention Implementation
//
// Uses Metal Performance Shaders Graph (MPSGraph) for attention computation.
// MPSGraph can automatically target the Apple Neural Engine (ANE) for
// power-efficient inference on iOS devices.

#include "granite/coreml_attention.h"
#include "granite/log.h"

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace granite {

// =============================================================================
// Implementation Class
// =============================================================================

class CoreMLAttention::Impl {
public:
    Impl() = default;
    ~Impl() { shutdown(); }

    Result<void> initialize(MTL::Device* device) {
        if (initialized_) return {};

        // Get the Objective-C Metal device from C++ wrapper
        mtl_device_ = (__bridge id<MTLDevice>)device;
        if (!mtl_device_) {
            return Error(ErrorCode::BackendNotInitialized, "Invalid Metal device");
        }

        // Create MPSGraph
        graph_ = [[MPSGraph alloc] init];
        if (!graph_) {
            return Error(ErrorCode::InternalError, "Failed to create MPSGraph");
        }

        // Create command queue for execution
        command_queue_ = [mtl_device_ newCommandQueue];
        if (!command_queue_) {
            return Error(ErrorCode::InternalError, "Failed to create command queue");
        }

        initialized_ = true;
        GRANITE_LOG_INFO("CoreMLAttention initialized (MPSGraph backend)");

        return {};
    }

    void shutdown() {
        if (!initialized_) return;

        graph_ = nil;
        command_queue_ = nil;
        mtl_device_ = nil;
        compiled_attention_ = nil;

        initialized_ = false;
    }

    bool is_initialized() const { return initialized_; }

    static bool is_ane_available() {
        // ANE is available on A11+ (iPhone 8+) and M1+
        // Check via device capabilities
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return false;

        // Check for Apple GPU family that includes ANE
        // Apple7 (A14/M1) and later have good ANE support
        if (@available(iOS 14.0, macOS 11.0, *)) {
            if ([device supportsFamily:MTLGPUFamilyApple7]) {
                return true;
            }
        }

        // Apple6 (A13) also has ANE but with limitations
        if (@available(iOS 13.0, *)) {
            if ([device supportsFamily:MTLGPUFamilyApple6]) {
                return true;
            }
        }

        return false;
    }

    void set_compute_unit(CoreMLAttention::ComputeUnit unit) {
        compute_unit_ = unit;
    }

    CoreMLAttention::ComputeUnit get_compute_unit() const {
        return compute_unit_;
    }

    Result<void> multihead_attention(
        MTL::Buffer* Q_buf,
        MTL::Buffer* K_buf,
        MTL::Buffer* V_buf,
        MTL::Buffer* output_buf,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t seq_q,
        uint32_t seq_kv,
        uint32_t head_dim,
        float scale)
    {
        if (!initialized_) {
            return Error(ErrorCode::BackendNotInitialized, "CoreMLAttention not initialized");
        }

        // Get Objective-C buffer references
        id<MTLBuffer> Q = (__bridge id<MTLBuffer>)Q_buf;
        id<MTLBuffer> K = (__bridge id<MTLBuffer>)K_buf;
        id<MTLBuffer> V = (__bridge id<MTLBuffer>)V_buf;
        id<MTLBuffer> output = (__bridge id<MTLBuffer>)output_buf;

        // Calculate GQA head ratio
        uint32_t heads_per_kv = num_heads / num_kv_heads;

        // Process each head
        // Note: For ANE efficiency, we process heads sequentially
        // The ANE parallelizes within each matmul operation
        for (uint32_t h = 0; h < num_heads; h++) {
            uint32_t kv_h = h / heads_per_kv;

            // Calculate buffer offsets
            size_t q_offset = h * seq_q * head_dim * sizeof(float);
            size_t k_offset = kv_h * seq_kv * head_dim * sizeof(uint16_t);  // FP16
            size_t v_offset = kv_h * seq_kv * head_dim * sizeof(uint16_t);  // FP16
            size_t out_offset = h * seq_q * head_dim * sizeof(float);

            auto result = attention_single_head_impl(
                Q, K, V, output,
                q_offset, k_offset, v_offset, out_offset,
                seq_q, seq_kv, head_dim, scale
            );

            if (!result.ok()) {
                return result;
            }
        }

        return {};
    }

    Result<void> attention_single_head(
        MTL::Buffer* Q_buf,
        MTL::Buffer* K_buf,
        MTL::Buffer* V_buf,
        MTL::Buffer* output_buf,
        uint32_t seq_q,
        uint32_t seq_kv,
        uint32_t head_dim,
        float scale)
    {
        if (!initialized_) {
            return Error(ErrorCode::BackendNotInitialized, "CoreMLAttention not initialized");
        }

        id<MTLBuffer> Q = (__bridge id<MTLBuffer>)Q_buf;
        id<MTLBuffer> K = (__bridge id<MTLBuffer>)K_buf;
        id<MTLBuffer> V = (__bridge id<MTLBuffer>)V_buf;
        id<MTLBuffer> output = (__bridge id<MTLBuffer>)output_buf;

        return attention_single_head_impl(
            Q, K, V, output,
            0, 0, 0, 0,  // No offsets
            seq_q, seq_kv, head_dim, scale
        );
    }

private:
    Result<void> attention_single_head_impl(
        id<MTLBuffer> Q_buffer,
        id<MTLBuffer> K_buffer,
        id<MTLBuffer> V_buffer,
        id<MTLBuffer> output_buffer,
        size_t q_offset,
        size_t k_offset,
        size_t v_offset,
        size_t out_offset,
        uint32_t seq_q,
        uint32_t seq_kv,
        uint32_t head_dim,
        float scale)
    {
        @autoreleasepool {
            // Create a fresh graph for this attention computation
            // (MPSGraph caches compiled executables internally)
            MPSGraph* graph = [[MPSGraph alloc] init];

            // Define tensor shapes
            NSArray<NSNumber*>* q_shape = @[@(seq_q), @(head_dim)];
            NSArray<NSNumber*>* k_shape = @[@(seq_kv), @(head_dim)];
            NSArray<NSNumber*>* v_shape = @[@(seq_kv), @(head_dim)];

            // Create input placeholders
            MPSGraphTensor* Q = [graph placeholderWithShape:q_shape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"Q"];
            MPSGraphTensor* K = [graph placeholderWithShape:k_shape
                                                   dataType:MPSDataTypeFloat16
                                                       name:@"K"];
            MPSGraphTensor* V = [graph placeholderWithShape:v_shape
                                                   dataType:MPSDataTypeFloat16
                                                       name:@"V"];

            // Convert K and V from FP16 to FP32 for computation
            MPSGraphTensor* K_f32 = [graph castTensor:K
                                               toType:MPSDataTypeFloat32
                                                 name:@"K_f32"];
            MPSGraphTensor* V_f32 = [graph castTensor:V
                                               toType:MPSDataTypeFloat32
                                                 name:@"V_f32"];

            // Transpose K: [seq_kv, head_dim] -> [head_dim, seq_kv]
            MPSGraphTensor* K_T = [graph transposeTensor:K_f32
                                               dimension:0
                                           withDimension:1
                                                    name:@"K_T"];

            // Compute attention scores: Q @ K^T -> [seq_q, seq_kv]
            MPSGraphTensor* scores = [graph matrixMultiplicationWithPrimaryTensor:Q
                                                                  secondaryTensor:K_T
                                                                             name:@"scores"];

            // Scale by 1/sqrt(head_dim)
            MPSGraphTensor* scale_tensor = [graph constantWithScalar:scale
                                                            dataType:MPSDataTypeFloat32];
            MPSGraphTensor* scaled_scores = [graph multiplicationWithPrimaryTensor:scores
                                                                   secondaryTensor:scale_tensor
                                                                              name:@"scaled_scores"];

            // Apply softmax along last axis (seq_kv dimension)
            MPSGraphTensor* attn_weights = [graph softMaxWithTensor:scaled_scores
                                                               axis:-1
                                                               name:@"attn_weights"];

            // Compute output: attn_weights @ V -> [seq_q, head_dim]
            MPSGraphTensor* output = [graph matrixMultiplicationWithPrimaryTensor:attn_weights
                                                                  secondaryTensor:V_f32
                                                                             name:@"output"];

            // Create buffer views at offsets (Metal shared memory allows pointer math)
            // We use newBufferWithBytesNoCopy to create views at offsets
            size_t q_size = seq_q * head_dim * sizeof(float);
            size_t k_size = seq_kv * head_dim * sizeof(uint16_t);
            size_t v_size = seq_kv * head_dim * sizeof(uint16_t);
            size_t out_size = seq_q * head_dim * sizeof(float);

            id<MTLBuffer> q_view = Q_buffer;
            id<MTLBuffer> k_view = K_buffer;
            id<MTLBuffer> v_view = V_buffer;
            id<MTLBuffer> out_view = output_buffer;

            // If offsets are non-zero, create sub-buffer views
            if (q_offset > 0) {
                q_view = [mtl_device_ newBufferWithBytesNoCopy:(uint8_t*)[Q_buffer contents] + q_offset
                                                       length:q_size
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
            }
            if (k_offset > 0) {
                k_view = [mtl_device_ newBufferWithBytesNoCopy:(uint8_t*)[K_buffer contents] + k_offset
                                                       length:k_size
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
            }
            if (v_offset > 0) {
                v_view = [mtl_device_ newBufferWithBytesNoCopy:(uint8_t*)[V_buffer contents] + v_offset
                                                       length:v_size
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
            }
            if (out_offset > 0) {
                out_view = [mtl_device_ newBufferWithBytesNoCopy:(uint8_t*)[output_buffer contents] + out_offset
                                                         length:out_size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
            }

            // Create input data descriptors using buffer views
            MPSGraphTensorData* Q_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:q_view
                           shape:q_shape
                        dataType:MPSDataTypeFloat32];

            MPSGraphTensorData* K_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:k_view
                           shape:k_shape
                        dataType:MPSDataTypeFloat16];

            MPSGraphTensorData* V_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:v_view
                           shape:v_shape
                        dataType:MPSDataTypeFloat16];

            // Create output data descriptor
            NSArray<NSNumber*>* out_shape = @[@(seq_q), @(head_dim)];
            MPSGraphTensorData* output_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:out_view
                           shape:out_shape
                        dataType:MPSDataTypeFloat32];

            // Prepare feeds and targets
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
                Q: Q_data,
                K: K_data,
                V: V_data
            };

            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* targets = @{
                output: output_data
            };

            // Execute the graph
            // Note: MPSGraph automatically selects ANE/GPU based on operation support
            id<MTLCommandBuffer> cmdBuffer = [command_queue_ commandBuffer];

            [graph encodeToCommandBuffer:cmdBuffer
                                   feeds:feeds
                       targetOperations:nil
                         resultsDictionary:targets
                   executionDescriptor:nil];

            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];

            // Check for errors
            if (cmdBuffer.status == MTLCommandBufferStatusError) {
                NSError* error = cmdBuffer.error;
                return Error(ErrorCode::InternalError,
                    std::string("MPSGraph execution failed: ") +
                    [error.localizedDescription UTF8String]);
            }
        }

        return {};
    }

    bool initialized_ = false;
    id<MTLDevice> mtl_device_ = nil;
    id<MTLCommandQueue> command_queue_ = nil;
    MPSGraph* graph_ = nil;
    MPSGraphExecutable* compiled_attention_ = nil;
    CoreMLAttention::ComputeUnit compute_unit_ = CoreMLAttention::ComputeUnit::Auto;
};

// =============================================================================
// Public Interface
// =============================================================================

CoreMLAttention::CoreMLAttention() : impl_(new Impl()) {}
CoreMLAttention::~CoreMLAttention() { delete impl_; }

Result<void> CoreMLAttention::initialize(MTL::Device* device) {
    return impl_->initialize(device);
}

void CoreMLAttention::shutdown() {
    impl_->shutdown();
}

bool CoreMLAttention::is_initialized() const {
    return impl_->is_initialized();
}

bool CoreMLAttention::is_ane_available() {
    return Impl::is_ane_available();
}

Result<void> CoreMLAttention::multihead_attention(
    MTL::Buffer* Q,
    MTL::Buffer* K,
    MTL::Buffer* V,
    MTL::Buffer* output,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t seq_q,
    uint32_t seq_kv,
    uint32_t head_dim,
    float scale)
{
    return impl_->multihead_attention(Q, K, V, output,
        num_heads, num_kv_heads, seq_q, seq_kv, head_dim, scale);
}

Result<void> CoreMLAttention::attention_single_head(
    MTL::Buffer* Q,
    MTL::Buffer* K,
    MTL::Buffer* V,
    MTL::Buffer* output,
    uint32_t seq_q,
    uint32_t seq_kv,
    uint32_t head_dim,
    float scale)
{
    return impl_->attention_single_head(Q, K, V, output,
        seq_q, seq_kv, head_dim, scale);
}

void CoreMLAttention::set_compute_unit(ComputeUnit unit) {
    impl_->set_compute_unit(unit);
}

CoreMLAttention::ComputeUnit CoreMLAttention::get_compute_unit() const {
    return impl_->get_compute_unit();
}

// =============================================================================
// Global Instance
// =============================================================================

static CoreMLAttention* g_coreml_attention = nullptr;

CoreMLAttention* get_coreml_attention() {
    if (!g_coreml_attention) {
        g_coreml_attention = new CoreMLAttention();
    }
    return g_coreml_attention;
}

}  // namespace granite

#endif  // __APPLE__
