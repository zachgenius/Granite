#!/bin/bash
# build_android.sh - Cross-compile Granite for Android with Vulkan support
#
# Prerequisites:
#   - Android NDK r25+ installed
#   - Set ANDROID_NDK environment variable or pass as argument
#
# Usage:
#   ./scripts/build_android.sh                    # Uses $ANDROID_NDK
#   ./scripts/build_android.sh /path/to/ndk      # Explicit NDK path
#   ./scripts/build_android.sh --abi=x86_64      # Build for x86_64 emulator

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
BUILD_TYPE="Release"
ANDROID_ABI="arm64-v8a"
ANDROID_PLATFORM="android-26"  # Vulkan 1.0 requires API 24+, use 26 for better support
BUILD_DIR="build-android"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --abi=*)
            ANDROID_ABI="${1#*=}"
            shift
            ;;
        --platform=*)
            ANDROID_PLATFORM="${1#*=}"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
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
            echo "Usage: $0 [OPTIONS] [NDK_PATH]"
            echo ""
            echo "Options:"
            echo "  --abi=ABI          Target ABI (arm64-v8a, armeabi-v7a, x86_64, x86)"
            echo "                     Default: arm64-v8a"
            echo "  --platform=LEVEL   Android API level (android-24, android-26, etc.)"
            echo "                     Default: android-26"
            echo "  --debug            Build with debug symbols"
            echo "  --build-dir=DIR    Build directory name"
            echo "                     Default: build-android"
            echo "  --clean            Clean build directory before building"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Environment:"
            echo "  ANDROID_NDK        Path to Android NDK (can also be passed as argument)"
            exit 0
            ;;
        *)
            # Assume it's the NDK path
            ANDROID_NDK="$1"
            shift
            ;;
    esac
done

# Find Android NDK
if [[ -z "$ANDROID_NDK" ]]; then
    # Try common locations
    if [[ -d "$HOME/Android/Sdk/ndk-bundle" ]]; then
        ANDROID_NDK="$HOME/Android/Sdk/ndk-bundle"
    elif [[ -d "$HOME/Library/Android/sdk/ndk-bundle" ]]; then
        ANDROID_NDK="$HOME/Library/Android/sdk/ndk-bundle"
    elif [[ -d "/usr/local/android-ndk" ]]; then
        ANDROID_NDK="/usr/local/android-ndk"
    else
        # Try to find latest NDK in SDK
        SDK_NDK_DIR="$HOME/Android/Sdk/ndk"
        if [[ -d "$SDK_NDK_DIR" ]]; then
            ANDROID_NDK=$(ls -d "$SDK_NDK_DIR"/*/ 2>/dev/null | sort -V | tail -n1)
        fi
        SDK_NDK_DIR="$HOME/Library/Android/sdk/ndk"
        if [[ -z "$ANDROID_NDK" && -d "$SDK_NDK_DIR" ]]; then
            ANDROID_NDK=$(ls -d "$SDK_NDK_DIR"/*/ 2>/dev/null | sort -V | tail -n1)
        fi
    fi
fi

if [[ -z "$ANDROID_NDK" || ! -d "$ANDROID_NDK" ]]; then
    echo "Error: Android NDK not found"
    echo "Please set ANDROID_NDK environment variable or pass NDK path as argument"
    echo "Example: $0 /path/to/android-ndk"
    exit 1
fi

TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
if [[ ! -f "$TOOLCHAIN_FILE" ]]; then
    echo "Error: CMake toolchain file not found at $TOOLCHAIN_FILE"
    echo "Please ensure you have a valid Android NDK installation"
    exit 1
fi

echo "========================================"
echo "Granite Android Build Configuration"
echo "========================================"
echo "NDK Path:        $ANDROID_NDK"
echo "Target ABI:      $ANDROID_ABI"
echo "API Level:       $ANDROID_PLATFORM"
echo "Build Type:      $BUILD_TYPE"
echo "Build Directory: $PROJECT_ROOT/$BUILD_DIR"
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
cmake -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGRANITE_BUILD_VULKAN=ON \
    -DGRANITE_BUILD_METAL=OFF \
    -DGRANITE_BUILD_CPU=ON \
    -DGRANITE_BUILD_COREML=OFF \
    -DGRANITE_BUILD_TESTS=OFF \
    -DGRANITE_BUILD_EXAMPLES=ON

# Build
echo ""
echo "Building..."
cmake --build "$BUILD_DIR" --parallel $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Summary
echo ""
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo "Output files:"
ls -lh "$BUILD_DIR/libgranite.a" 2>/dev/null || echo "  (library not found)"
ls -lh "$BUILD_DIR/granite_main" 2>/dev/null || echo "  (main executable not found)"
ls -lh "$BUILD_DIR/granite_benchmark" 2>/dev/null || echo "  (benchmark not found)"
echo ""
echo "To run on device:"
echo "  adb push $BUILD_DIR/granite_main /data/local/tmp/"
echo "  adb push /path/to/model.gguf /data/local/tmp/"
echo "  adb shell /data/local/tmp/granite_main /data/local/tmp/model.gguf"
