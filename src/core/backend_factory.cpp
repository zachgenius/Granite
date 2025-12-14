#include "granite/backend.h"
#include "granite/log.h"

namespace granite {

// Forward declarations of backend classes
class CPUBackend;

#ifdef GRANITE_HAS_METAL
class MetalBackend;
#endif

#ifdef GRANITE_HAS_VULKAN
class VulkanBackend;
#endif

// CPUBackend is defined in cpu_backend.cpp
extern std::unique_ptr<IComputeBackend> create_cpu_backend();

#ifdef GRANITE_HAS_METAL
extern std::unique_ptr<IComputeBackend> create_metal_backend();
#endif

#ifdef GRANITE_HAS_VULKAN
extern std::unique_ptr<IComputeBackend> create_vulkan_backend();
#endif

std::unique_ptr<IComputeBackend> create_backend(BackendType type) {
    switch (type) {
        case BackendType::CPU:
#ifdef GRANITE_HAS_CPU
            return create_cpu_backend();
#else
            GRANITE_LOG_WARN("CPU backend not available");
            return nullptr;
#endif

        case BackendType::Metal:
#ifdef GRANITE_HAS_METAL
            return create_metal_backend();
#else
            GRANITE_LOG_WARN("Metal backend not available on this platform");
            return nullptr;
#endif

        case BackendType::Vulkan:
#ifdef GRANITE_HAS_VULKAN
            return create_vulkan_backend();
#else
            GRANITE_LOG_WARN("Vulkan backend not available");
            return nullptr;
#endif
    }

    return nullptr;
}

std::unique_ptr<IComputeBackend> create_default_backend() {
#ifdef GRANITE_HAS_METAL
    GRANITE_LOG_INFO("Using Metal as default backend");
    return create_backend(BackendType::Metal);
#elif defined(GRANITE_HAS_VULKAN)
    GRANITE_LOG_INFO("Using Vulkan as default backend");
    return create_backend(BackendType::Vulkan);
#elif defined(GRANITE_HAS_CPU)
    GRANITE_LOG_INFO("Using CPU as default backend");
    return create_backend(BackendType::CPU);
#else
    GRANITE_LOG_ERROR("No backend available");
    return nullptr;
#endif
}

bool is_backend_available(BackendType type) {
    switch (type) {
        case BackendType::CPU:
#ifdef GRANITE_HAS_CPU
            return true;
#else
            return false;
#endif

        case BackendType::Metal:
#ifdef GRANITE_HAS_METAL
            return true;
#else
            return false;
#endif

        case BackendType::Vulkan:
#ifdef GRANITE_HAS_VULKAN
            return true;
#else
            return false;
#endif
    }

    return false;
}

}  // namespace granite
