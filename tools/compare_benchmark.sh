#!/bin/bash

# Granite vs llama.cpp Benchmark Comparison Script
# Usage: ./compare_benchmark.sh <model.gguf>
# Environment variables:
#   LLAMA_CPP_DIR - Path to llama.cpp directory (required for llama.cpp comparison)

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <model.gguf>"
    echo "Environment variables:"
    echo "  LLAMA_CPP_DIR - Path to llama.cpp directory (optional)"
    exit 1
fi

MODEL_PATH="$1"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}"

echo "=========================================="
echo "Granite vs llama.cpp Benchmark Comparison"
echo "=========================================="
echo ""
echo "Model: $MODEL_PATH"
echo "Date: $(date)"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Build Granite if needed
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRANITE_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$GRANITE_DIR/build/granite_benchmark" ]; then
    echo "Building Granite..."
    cmake -B "$GRANITE_DIR/build" -DCMAKE_BUILD_TYPE=Release "$GRANITE_DIR"
    cmake --build "$GRANITE_DIR/build" --target granite_benchmark -j8
fi

# Check if llama.cpp exists and build if needed
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo ""
    echo "llama.cpp not found at $LLAMA_CPP_DIR"
    echo "Please clone llama.cpp first:"
    echo "  git clone https://github.com/ggerganov/llama.cpp.git $LLAMA_CPP_DIR"
    echo ""
    echo "Or set LLAMA_CPP_DIR environment variable to your llama.cpp directory"
    exit 1
fi

if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-bench" ]; then
    echo "Building llama.cpp..."
    cmake -B "$LLAMA_CPP_DIR/build" -DCMAKE_BUILD_TYPE=Release \
          -DGGML_METAL=ON "$LLAMA_CPP_DIR"
    cmake --build "$LLAMA_CPP_DIR/build" --target llama-bench -j8
fi

# Run benchmarks
echo ""
echo "=========================================="
echo "Running Granite Benchmark"
echo "=========================================="
echo ""

# Granite benchmark (focused on inference)
"$GRANITE_DIR/build/granite_benchmark" "$MODEL_PATH" 2>&1 | tee /tmp/granite_bench.txt

echo ""
echo "=========================================="
echo "Running llama.cpp Benchmark"
echo "=========================================="
echo ""

# llama.cpp benchmark
# -ngl 999: offload all layers to GPU (Metal on macOS)
# -p 1,16: test prompt lengths of 1 and 16 tokens
# -n 1: generate 1 token (for measuring per-token speed)
# -r 3: 3 repetitions for averaging
if [ -z "$LLAMA_CPP_DIR" ]; then
    echo "Skipping llama.cpp benchmark (LLAMA_CPP_DIR not set)"
    echo "To run llama.cpp comparison: export LLAMA_CPP_DIR=/path/to/llama.cpp"
else
    "$LLAMA_CPP_DIR/build/bin/llama-bench" \
        -m "$MODEL_PATH" \
        -p 1,16 \
        -n 1,20 \
        -ngl 999 \
        -r 3 \
        -o md 2>&1 | tee /tmp/llama_bench.txt
fi

echo ""
echo "=========================================="
echo "Comparison Summary"
echo "=========================================="
echo ""

# Extract key metrics
echo "Granite Results (from /tmp/granite_bench.txt):"
grep -A5 "Decode" /tmp/granite_bench.txt | tail -5

echo ""
echo "llama.cpp Results (from /tmp/llama_bench.txt):"
echo "(pp = prompt processing, tg = text generation)"
grep -E "^\|.*TinyLlama" /tmp/llama_bench.txt || grep -E "^\|.*model" /tmp/llama_bench.txt | head -5

echo ""
echo "=========================================="
echo "Notes:"
echo "=========================================="
echo "- Granite runs on CPU with BLAS acceleration"
echo "- llama.cpp with -ngl 999 uses Metal GPU on macOS"
echo "- For fair CPU comparison, use: llama-bench -ngl 0"
echo ""
echo "Raw results saved to:"
echo "  - /tmp/granite_bench.txt"
echo "  - /tmp/llama_bench.txt"
