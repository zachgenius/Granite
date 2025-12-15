#pragma once

// =============================================================================
// Granite - Embedded Inference Framework
// =============================================================================
//
// High-performance, cross-platform C++ framework for on-device machine learning
// inference, targeting Apple (Metal) and Android (Vulkan) with GPU-accelerated
// compute.
//
// Usage:
//   #include <granite/granite.h>
//
//   auto backend = granite::create_default_backend();
//   backend->initialize();
//
//   auto tensor = granite::Tensor::allocate({32, 64}, granite::DataType::FP16, backend.get());
//
// =============================================================================

#include "granite/types.h"
#include "granite/error.h"
#include "granite/backend.h"
#include "granite/tensor.h"
#include "granite/operators.h"
#include "granite/log.h"

namespace granite {

/// Get the version string
constexpr const char* version_string() {
    return "0.1.0";
}

/// Get version components
constexpr int version_major() { return GRANITE_VERSION_MAJOR; }
constexpr int version_minor() { return GRANITE_VERSION_MINOR; }
constexpr int version_patch() { return GRANITE_VERSION_PATCH; }

}  // namespace granite
