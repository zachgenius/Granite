#include <granite/granite.h>
#include <iostream>

int main() {
    // Initialize logging
    granite::init_logging(spdlog::level::debug);

    GRANITE_LOG_INFO("Granite v{}", granite::version_string());

    // Create default backend
    auto backend = granite::create_default_backend();
    if (!backend) {
        GRANITE_LOG_ERROR("Failed to create backend");
        return 1;
    }

    // Initialize backend
    auto init_result = backend->initialize();
    if (!init_result.ok()) {
        GRANITE_LOG_ERROR("Failed to initialize backend: {}", init_result.error().message());
        return 1;
    }

    // Print capabilities
    auto caps = backend->get_capabilities();
    GRANITE_LOG_INFO("Device: {}", caps.name);
    GRANITE_LOG_INFO("Max buffer size: {} MB", caps.max_buffer_size / (1024 * 1024));
    GRANITE_LOG_INFO("Max threadgroup size: {}", caps.max_threadgroup_size);
    GRANITE_LOG_INFO("FP16 support: {}", caps.supports_fp16 ? "yes" : "no");

    // Allocate a tensor
    std::vector<int64_t> shape = {32, 64};
    auto tensor_result = granite::Tensor::allocate(shape, granite::DataType::FP32, backend.get());

    if (!tensor_result.ok()) {
        GRANITE_LOG_ERROR("Failed to allocate tensor: {}", tensor_result.error().message());
        return 1;
    }

    auto& tensor = tensor_result.value();
    GRANITE_LOG_INFO("Allocated tensor: shape=[{}, {}], dtype={}, size={} bytes",
                     tensor.size(0), tensor.size(1),
                     granite::dtype_name(tensor.dtype()),
                     tensor.size_bytes());

    // Cleanup
    backend->shutdown();
    GRANITE_LOG_INFO("Done!");

    return 0;
}
