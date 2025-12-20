#pragma once

// Internal header for LLM implementation files
// Not part of public API

#include "granite/llm.h"
#include "granite/operators.h"
#include "granite/log.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>
#include <unordered_set>
#include <thread>
#include <vector>

#ifdef GRANITE_HAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

#ifdef GRANITE_HAS_OPENMP
#include <omp.h>
#endif

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#define USE_GCD 1
#endif

#ifdef GRANITE_HAS_METAL
#include <Metal/Metal.hpp>
#include "granite/metal_compute.h"
#endif

#ifdef GRANITE_HAS_COREML
#include "granite/coreml_ffn.h"
#endif

namespace granite {

// Thread-local RNG for sampling (defined in runner.cpp)
extern thread_local std::mt19937 g_rng;

}  // namespace granite
