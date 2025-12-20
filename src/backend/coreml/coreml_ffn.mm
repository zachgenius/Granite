// CoreML/MPSGraph FFN Implementation
//
// Uses Metal Performance Shaders Graph (MPSGraph) for FFN computation.
// MPSGraph can automatically target the Apple Neural Engine (ANE) for
// power-efficient inference on iOS devices.
//
// FFN is suitable for ANE because shapes are fixed (hidden_dim × intermediate_dim).
// Pre-compilation eliminates the ~100ms graph construction overhead.

#include "granite/coreml_ffn.h"
#include "granite/log.h"

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <unordered_map>
#include <mutex>

namespace granite {

// =============================================================================
// FFN Configuration Key
// =============================================================================

struct FFNConfig {
    uint32_t batch_size;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;

    bool operator==(const FFNConfig& other) const {
        return batch_size == other.batch_size &&
               hidden_dim == other.hidden_dim &&
               intermediate_dim == other.intermediate_dim;
    }
};

struct FFNConfigHash {
    size_t operator()(const FFNConfig& config) const {
        return std::hash<uint64_t>()(
            (static_cast<uint64_t>(config.batch_size) << 48) |
            (static_cast<uint64_t>(config.hidden_dim) << 24) |
            config.intermediate_dim
        );
    }
};

// =============================================================================
// Cached FFN Executable
// =============================================================================

struct CachedFFNExecutable {
    MPSGraphExecutable* executable = nil;
    MPSGraphTensor* input_placeholder = nil;
    MPSGraphTensor* w_gate_placeholder = nil;
    MPSGraphTensor* w_up_placeholder = nil;
    MPSGraphTensor* w_down_placeholder = nil;
    MPSGraphTensor* output_tensor = nil;
};

// =============================================================================
// Implementation Class
// =============================================================================

class CoreMLFFN::Impl {
public:
    Impl() = default;
    ~Impl() { shutdown(); }

    Result<void> initialize(MTL::Device* device) {
        if (initialized_) return {};

        mtl_device_ = (__bridge id<MTLDevice>)device;
        if (!mtl_device_) {
            return Error(ErrorCode::BackendNotInitialized, "Invalid Metal device");
        }

        command_queue_ = [mtl_device_ newCommandQueue];
        if (!command_queue_) {
            return Error(ErrorCode::InternalError, "Failed to create command queue");
        }

        GRANITE_LOG_INFO("CoreMLFFN initialized (MPSGraph backend)");
        initialized_ = true;

        return {};
    }

    void shutdown() {
        if (!initialized_) return;

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            for (auto& pair : executable_cache_) {
                pair.second.executable = nil;
                pair.second.input_placeholder = nil;
                pair.second.w_gate_placeholder = nil;
                pair.second.w_up_placeholder = nil;
                pair.second.w_down_placeholder = nil;
                pair.second.output_tensor = nil;
            }
            executable_cache_.clear();
        }

        command_queue_ = nil;
        mtl_device_ = nil;
        initialized_ = false;
    }

    bool is_initialized() const { return initialized_; }

    static bool is_ane_available() {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return false;

        if (@available(iOS 14.0, macOS 11.0, *)) {
            if ([device supportsFamily:MTLGPUFamilyApple7]) {
                return true;
            }
        }

        if (@available(iOS 13.0, *)) {
            if ([device supportsFamily:MTLGPUFamilyApple6]) {
                return true;
            }
        }

        return false;
    }

    void set_power_mode(CoreMLFFN::PowerMode mode) {
        power_mode_ = mode;
    }

    CoreMLFFN::PowerMode get_power_mode() const {
        return power_mode_;
    }

    Result<void> compile_ffn(uint32_t hidden_dim, uint32_t intermediate_dim) {
        // Pre-compile for common batch sizes
        std::vector<uint32_t> batch_sizes = {1, 4, 8, 16, 32, 64, 128};

        int compiled = 0;
        for (uint32_t batch : batch_sizes) {
            auto result = compile_ffn_executable(batch, hidden_dim, intermediate_dim);
            if (result.ok()) {
                compiled++;
            }
        }

        GRANITE_LOG_INFO("Pre-compiled {} FFN executables for hidden={}, intermediate={}",
                        compiled, hidden_dim, intermediate_dim);

        return {};
    }

    Result<void> forward(
        MTL::Buffer* input_buf,
        MTL::Buffer* w_gate_buf,
        MTL::Buffer* w_up_buf,
        MTL::Buffer* w_down_buf,
        MTL::Buffer* output_buf,
        uint32_t batch_size,
        uint32_t hidden_dim,
        uint32_t intermediate_dim)
    {
        if (!initialized_) {
            return Error(ErrorCode::BackendNotInitialized, "CoreMLFFN not initialized");
        }

        // Check power mode
        if (power_mode_ == CoreMLFFN::PowerMode::Performance) {
            return Error(ErrorCode::InvalidArgument,
                "CoreMLFFN in Performance mode - use Metal GPU instead");
        }

        // Get or compile executable
        CachedFFNExecutable* cached = get_executable(batch_size, hidden_dim, intermediate_dim);
        if (!cached || !cached->executable) {
            return Error(ErrorCode::InternalError,
                "No compiled FFN executable for batch=" + std::to_string(batch_size));
        }

        @autoreleasepool {
            id<MTLBuffer> input = (__bridge id<MTLBuffer>)input_buf;
            id<MTLBuffer> w_gate = (__bridge id<MTLBuffer>)w_gate_buf;
            id<MTLBuffer> w_up = (__bridge id<MTLBuffer>)w_up_buf;
            id<MTLBuffer> w_down = (__bridge id<MTLBuffer>)w_down_buf;
            id<MTLBuffer> output = (__bridge id<MTLBuffer>)output_buf;

            // Create tensor data from buffers
            NSArray<NSNumber*>* input_shape = @[@(batch_size), @(hidden_dim)];
            NSArray<NSNumber*>* gate_shape = @[@(hidden_dim), @(intermediate_dim)];
            NSArray<NSNumber*>* up_shape = @[@(hidden_dim), @(intermediate_dim)];
            NSArray<NSNumber*>* down_shape = @[@(intermediate_dim), @(hidden_dim)];
            NSArray<NSNumber*>* output_shape = @[@(batch_size), @(hidden_dim)];

            MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:input shape:input_shape dataType:MPSDataTypeFloat32];
            MPSGraphTensorData* w_gate_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:w_gate shape:gate_shape dataType:MPSDataTypeFloat16];
            MPSGraphTensorData* w_up_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:w_up shape:up_shape dataType:MPSDataTypeFloat16];
            MPSGraphTensorData* w_down_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:w_down shape:down_shape dataType:MPSDataTypeFloat16];
            MPSGraphTensorData* output_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:output shape:output_shape dataType:MPSDataTypeFloat32];

            // Build inputsArray in the order expected by the executable's feedTensors
            // The feedTensors property returns the placeholders in the order expected by runWith:
            NSArray<MPSGraphTensor*>* feedTensors = cached->executable.feedTensors;
            NSMutableArray<MPSGraphTensorData*>* inputsArray = [[NSMutableArray alloc] initWithCapacity:feedTensors.count];

            for (MPSGraphTensor* tensor in feedTensors) {
                // Match tensors by comparing to cached placeholders
                if (tensor == cached->input_placeholder) {
                    [inputsArray addObject:input_data];
                } else if (tensor == cached->w_gate_placeholder) {
                    [inputsArray addObject:w_gate_data];
                } else if (tensor == cached->w_up_placeholder) {
                    [inputsArray addObject:w_up_data];
                } else if (tensor == cached->w_down_placeholder) {
                    [inputsArray addObject:w_down_data];
                }
            }

            NSArray<MPSGraphTensorData*>* resultsArray = @[output_data];

            // Execute synchronously
            [cached->executable runWithMTLCommandQueue:command_queue_
                                           inputsArray:inputsArray
                                          resultsArray:resultsArray
                                   executionDescriptor:nil];
        }

        return {};
    }

private:
    Result<void> compile_ffn_executable(
        uint32_t batch_size,
        uint32_t hidden_dim,
        uint32_t intermediate_dim)
    {
        FFNConfig config{batch_size, hidden_dim, intermediate_dim};

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            if (executable_cache_.find(config) != executable_cache_.end()) {
                return {};
            }
        }

        @autoreleasepool {
            MPSGraph* graph = [[MPSGraph alloc] init];

            // Define shapes
            NSArray<NSNumber*>* input_shape = @[@(batch_size), @(hidden_dim)];
            NSArray<NSNumber*>* gate_shape = @[@(hidden_dim), @(intermediate_dim)];
            NSArray<NSNumber*>* up_shape = @[@(hidden_dim), @(intermediate_dim)];
            NSArray<NSNumber*>* down_shape = @[@(intermediate_dim), @(hidden_dim)];

            // Create placeholders
            MPSGraphTensor* input = [graph placeholderWithShape:input_shape
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"input"];
            MPSGraphTensor* w_gate = [graph placeholderWithShape:gate_shape
                                                        dataType:MPSDataTypeFloat16
                                                            name:@"w_gate"];
            MPSGraphTensor* w_up = [graph placeholderWithShape:up_shape
                                                      dataType:MPSDataTypeFloat16
                                                          name:@"w_up"];
            MPSGraphTensor* w_down = [graph placeholderWithShape:down_shape
                                                        dataType:MPSDataTypeFloat16
                                                            name:@"w_down"];

            // Cast weights to FP32 for computation
            MPSGraphTensor* w_gate_f32 = [graph castTensor:w_gate
                                                    toType:MPSDataTypeFloat32
                                                      name:@"w_gate_f32"];
            MPSGraphTensor* w_up_f32 = [graph castTensor:w_up
                                                  toType:MPSDataTypeFloat32
                                                    name:@"w_up_f32"];
            MPSGraphTensor* w_down_f32 = [graph castTensor:w_down
                                                    toType:MPSDataTypeFloat32
                                                      name:@"w_down_f32"];

            // SwiGLU FFN: output = down(silu(gate(x)) * up(x))

            // Gate projection: [batch, hidden] @ [hidden, inter] = [batch, inter]
            MPSGraphTensor* gate_out = [graph matrixMultiplicationWithPrimaryTensor:input
                                                                    secondaryTensor:w_gate_f32
                                                                               name:@"gate"];

            // Up projection: [batch, hidden] @ [hidden, inter] = [batch, inter]
            MPSGraphTensor* up_out = [graph matrixMultiplicationWithPrimaryTensor:input
                                                                  secondaryTensor:w_up_f32
                                                                             name:@"up"];

            // SiLU activation on gate: silu(x) = x * sigmoid(x)
            MPSGraphTensor* gate_sigmoid = [graph sigmoidWithTensor:gate_out name:@"gate_sigmoid"];
            MPSGraphTensor* gate_silu = [graph multiplicationWithPrimaryTensor:gate_out
                                                               secondaryTensor:gate_sigmoid
                                                                          name:@"gate_silu"];

            // Elementwise multiply: gate_silu * up_out
            MPSGraphTensor* hidden = [graph multiplicationWithPrimaryTensor:gate_silu
                                                            secondaryTensor:up_out
                                                                       name:@"hidden"];

            // Down projection: [batch, inter] @ [inter, hidden] = [batch, hidden]
            MPSGraphTensor* output = [graph matrixMultiplicationWithPrimaryTensor:hidden
                                                                  secondaryTensor:w_down_f32
                                                                             name:@"output"];

            // Compile
            MPSGraphCompilationDescriptor* compileDesc = [[MPSGraphCompilationDescriptor alloc] init];
            compileDesc.optimizationLevel = MPSGraphOptimizationLevel1;

            MPSGraphShapedType* input_shaped = [[MPSGraphShapedType alloc]
                initWithShape:input_shape dataType:MPSDataTypeFloat32];
            MPSGraphShapedType* gate_shaped = [[MPSGraphShapedType alloc]
                initWithShape:gate_shape dataType:MPSDataTypeFloat16];
            MPSGraphShapedType* up_shaped = [[MPSGraphShapedType alloc]
                initWithShape:up_shape dataType:MPSDataTypeFloat16];
            MPSGraphShapedType* down_shaped = [[MPSGraphShapedType alloc]
                initWithShape:down_shape dataType:MPSDataTypeFloat16];

            NSDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds = @{
                input: input_shaped,
                w_gate: gate_shaped,
                w_up: up_shaped,
                w_down: down_shaped
            };

            MPSGraphDevice* mps_device = [MPSGraphDevice deviceWithMTLDevice:mtl_device_];

            MPSGraphExecutable* executable = [graph compileWithDevice:mps_device
                                                                feeds:feeds
                                                        targetTensors:@[output]
                                                     targetOperations:nil
                                               compilationDescriptor:compileDesc];

            if (!executable) {
                return Error(ErrorCode::InternalError,
                    "Failed to compile FFN graph for batch=" + std::to_string(batch_size));
            }

            CachedFFNExecutable cached;
            cached.executable = executable;
            cached.input_placeholder = input;
            cached.w_gate_placeholder = w_gate;
            cached.w_up_placeholder = w_up;
            cached.w_down_placeholder = w_down;
            cached.output_tensor = output;

            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                executable_cache_[config] = cached;
            }
        }

        return {};
    }

    CachedFFNExecutable* get_executable(
        uint32_t batch_size,
        uint32_t hidden_dim,
        uint32_t intermediate_dim)
    {
        FFNConfig config{batch_size, hidden_dim, intermediate_dim};

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto it = executable_cache_.find(config);
            if (it != executable_cache_.end()) {
                return &it->second;
            }
        }

        // Compile on demand
        auto result = compile_ffn_executable(batch_size, hidden_dim, intermediate_dim);
        if (!result.ok()) {
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(cache_mutex_);
        return &executable_cache_[config];
    }

    bool initialized_ = false;
    id<MTLDevice> mtl_device_ = nil;
    id<MTLCommandQueue> command_queue_ = nil;
    CoreMLFFN::PowerMode power_mode_ = CoreMLFFN::PowerMode::Balanced;

    std::mutex cache_mutex_;
    std::unordered_map<FFNConfig, CachedFFNExecutable, FFNConfigHash> executable_cache_;
};

// =============================================================================
// Public Interface
// =============================================================================

CoreMLFFN::CoreMLFFN() : impl_(new Impl()) {}
CoreMLFFN::~CoreMLFFN() { delete impl_; }

Result<void> CoreMLFFN::initialize(MTL::Device* device) {
    return impl_->initialize(device);
}

void CoreMLFFN::shutdown() {
    impl_->shutdown();
}

bool CoreMLFFN::is_initialized() const {
    return impl_->is_initialized();
}

bool CoreMLFFN::is_ane_available() {
    return Impl::is_ane_available();
}

Result<void> CoreMLFFN::compile_ffn(uint32_t hidden_dim, uint32_t intermediate_dim) {
    return impl_->compile_ffn(hidden_dim, intermediate_dim);
}

Result<void> CoreMLFFN::forward(
    MTL::Buffer* input,
    MTL::Buffer* w_gate,
    MTL::Buffer* w_up,
    MTL::Buffer* w_down,
    MTL::Buffer* output,
    uint32_t batch_size,
    uint32_t hidden_dim,
    uint32_t intermediate_dim)
{
    return impl_->forward(input, w_gate, w_up, w_down, output,
                         batch_size, hidden_dim, intermediate_dim);
}

void CoreMLFFN::set_power_mode(PowerMode mode) {
    impl_->set_power_mode(mode);
}

CoreMLFFN::PowerMode CoreMLFFN::get_power_mode() const {
    return impl_->get_power_mode();
}

// =============================================================================
// Global Instance
// =============================================================================

static CoreMLFFN* g_coreml_ffn = nullptr;

CoreMLFFN* get_coreml_ffn() {
    if (!g_coreml_ffn) {
        g_coreml_ffn = new CoreMLFFN();
    }
    return g_coreml_ffn;
}

}  // namespace granite

#endif  // __APPLE__
