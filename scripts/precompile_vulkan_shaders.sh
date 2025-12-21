#!/bin/bash
# precompile_vulkan_shaders.sh - Precompile Vulkan GLSL compute shaders to SPIR-V
#
# Usage:
#   ./scripts/precompile_vulkan_shaders.sh
#   ./scripts/precompile_vulkan_shaders.sh --out-dir=shaders/vulkan/spv
#   ./scripts/precompile_vulkan_shaders.sh --shader-dir=shaders/vulkan

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SHADER_DIR="$PROJECT_ROOT/shaders/vulkan"
OUT_DIR="$SHADER_DIR/spv"
FORCE_COMPILE=0
EXTRA_DEFINES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shader-dir=*)
            SHADER_DIR="${1#*=}"
            shift
            ;;
        --out-dir=*)
            OUT_DIR="${1#*=}"
            shift
            ;;
        --define=*)
            EXTRA_DEFINES+=("-D" "${1#*=}")
            shift
            ;;
        --force)
            FORCE_COMPILE=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--shader-dir=DIR] [--out-dir=DIR] [--define=NAME[=VALUE]] [--force]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ ! -d "$SHADER_DIR" ]]; then
    echo "Shader directory not found: $SHADER_DIR"
    exit 1
fi

if command -v glslc >/dev/null 2>&1; then
    COMPILER="glslc"
    COMPILER_ARGS=("--target-env=vulkan1.1" "-I" "$SHADER_DIR")
elif command -v glslangValidator >/dev/null 2>&1; then
    COMPILER="glslangValidator"
    COMPILER_ARGS=("-V" "-I" "$SHADER_DIR")
else
    echo "No shader compiler found (glslc or glslangValidator)."
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Precompiling shaders from $SHADER_DIR to $OUT_DIR"
echo "Using compiler: $COMPILER"

find "$SHADER_DIR" -type f -name "*.comp" | while read -r shader; do
    rel_name="$(basename "$shader")"
    out_path="$OUT_DIR/${rel_name}.spv"
    echo "  $rel_name -> $(basename "$out_path")"
    if [[ "$FORCE_COMPILE" -eq 0 ]]; then
        if rg -q "#include \\\"" "$shader"; then
            echo "  skipping (requires defines/includes): $rel_name"
            continue
        fi
        if rg -q "A_TYPE|B_TYPE|D_TYPE" "$shader"; then
            if ! rg -q "DATA_A_|DATA_B_|DATA_D_" "$shader"; then
                echo "  skipping (requires defines): $rel_name"
                continue
            fi
        fi
    fi
    if [[ ${#EXTRA_DEFINES[@]} -gt 0 ]]; then
        "$COMPILER" "${COMPILER_ARGS[@]}" "${EXTRA_DEFINES[@]}" -o "$out_path" "$shader"
    else
        "$COMPILER" "${COMPILER_ARGS[@]}" -o "$out_path" "$shader"
    fi
done

echo "Done."
