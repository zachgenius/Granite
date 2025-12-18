#!/bin/bash
# build_linux_vulkan.sh - Build Granite with Vulkan support on Linux
#
# Prerequisites:
#   - Vulkan SDK installed (https://vulkan.lunarg.com/sdk/home)
#   - Or system Vulkan headers/libs: libvulkan-dev (Debian/Ubuntu)
#   - Optional: shaderc for runtime shader compilation
#
# Usage:
#   ./scripts/build_linux_vulkan.sh
#   ./scripts/build_linux_vulkan.sh --debug
#   ./scripts/build_linux_vulkan.sh --with-shaderc

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build-vulkan"
WITH_SHADERC=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --with-shaderc)
            WITH_SHADERC=1
            shift
            ;;
        --build-dir=*)
            BUILD_DIR="${1#*=}"
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug            Build with debug symbols"
            echo "  --with-shaderc     Enable runtime shader compilation"
            echo "  --build-dir=DIR    Build directory name (default: build-vulkan)"
            echo "  --clean            Clean build directory before building"
            echo "  --help, -h         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for Vulkan SDK
echo "Checking for Vulkan SDK..."
if command -v vulkaninfo &> /dev/null; then
    echo "  Found vulkaninfo"
    vulkaninfo --summary 2>/dev/null | head -20 || true
elif pkg-config --exists vulkan 2>/dev/null; then
    echo "  Found Vulkan via pkg-config"
else
    echo "Warning: Vulkan SDK not detected"
    echo "Install Vulkan SDK from https://vulkan.lunarg.com/sdk/home"
    echo "Or install system packages: libvulkan-dev (Debian/Ubuntu)"
fi

echo ""
echo "========================================"
echo "Granite Linux Vulkan Build"
echo "========================================"
echo "Build Type:      $BUILD_TYPE"
echo "Build Directory: $PROJECT_ROOT/$BUILD_DIR"
echo "Shaderc:         $([ $WITH_SHADERC -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
echo "========================================"

cd "$PROJECT_ROOT"

# Clean if requested
if [[ -n "$CLEAN_BUILD" && -d "$BUILD_DIR" ]]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Configure
echo ""
echo "Configuring..."

CMAKE_ARGS=(
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DGRANITE_BUILD_VULKAN=ON
    -DGRANITE_BUILD_METAL=OFF
    -DGRANITE_BUILD_CPU=ON
    -DGRANITE_BUILD_COREML=OFF
    -DGRANITE_BUILD_TESTS=ON
    -DGRANITE_BUILD_EXAMPLES=ON
)

if [[ $WITH_SHADERC -eq 1 ]]; then
    CMAKE_ARGS+=(-DGRANITE_WITH_SHADERC=ON)
fi

cmake "${CMAKE_ARGS[@]}"

# Build
echo ""
echo "Building..."
cmake --build "$BUILD_DIR" --parallel $(nproc 2>/dev/null || echo 4)

# Run tests if available
if [[ -f "$BUILD_DIR/granite_tests" ]]; then
    echo ""
    echo "Running tests..."
    "$BUILD_DIR/granite_tests" --reporter compact || true
fi

# Summary
echo ""
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo "Output files:"
ls -lh "$BUILD_DIR/libgranite.a" 2>/dev/null || echo "  (library not found)"
ls -lh "$BUILD_DIR/granite_main" 2>/dev/null || echo "  (main not found)"
ls -lh "$BUILD_DIR/granite_benchmark" 2>/dev/null || echo "  (benchmark not found)"
echo ""
echo "To run inference:"
echo "  $BUILD_DIR/granite_main /path/to/model.gguf"
