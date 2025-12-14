#include <granite/granite.h>
#include <iostream>

int main() {
    granite::init_logging(spdlog::level::info);

    GRANITE_LOG_INFO("Granite Embedded Inference Framework v{}", granite::version_string());
    GRANITE_LOG_INFO("Build configuration:");
#ifdef GRANITE_HAS_METAL
    GRANITE_LOG_INFO("  - Metal backend: enabled");
#else
    GRANITE_LOG_INFO("  - Metal backend: disabled");
#endif
#ifdef GRANITE_HAS_CPU
    GRANITE_LOG_INFO("  - CPU backend: enabled");
#else
    GRANITE_LOG_INFO("  - CPU backend: disabled");
#endif

    // Test backend creation
    auto backend = granite::create_default_backend();
    if (backend) {
        auto result = backend->initialize();
        if (result.ok()) {
            auto caps = backend->get_capabilities();
            GRANITE_LOG_INFO("Default backend: {} ({})",
                             granite::backend_name(backend->get_type()),
                             caps.name);
            backend->shutdown();
        }
    }

    return 0;
}
