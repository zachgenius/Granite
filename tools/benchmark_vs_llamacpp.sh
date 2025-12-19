#!/bin/bash
# Comprehensive benchmark: Granite vs llama.cpp
# Tests various scenarios to identify optimization opportunities

set -e

MODEL_PATH="${1:-/Users/zach/Downloads/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf}"
LLAMACPP_BENCH="${LLAMACPP_BENCH:-/Users/zach/Downloads/llama.cpp/build/bin/llama-bench}"
GRANITE_BENCH="./build/granite_benchmark"

echo "============================================================"
echo "  Granite vs llama.cpp Comprehensive Benchmark"
echo "============================================================"
echo ""
echo "Model: $MODEL_PATH"
echo "Date: $(date)"
echo ""

# Check if llama-bench exists
if [ ! -f "$LLAMACPP_BENCH" ]; then
    echo "ERROR: llama-bench not found at $LLAMACPP_BENCH"
    echo "Set LLAMACPP_BENCH env var to the correct path"
    exit 1
fi

# Build Granite if needed
if [ ! -f "$GRANITE_BENCH" ]; then
    echo "Building Granite..."
    cmake --build build --parallel --target granite_benchmark
fi

echo ""
echo "============================================================"
echo "  Scenario 1: Decode Performance (varying context length)"
echo "============================================================"
echo ""

echo "--- llama.cpp ---"
$LLAMACPP_BENCH -m "$MODEL_PATH" -p 0 -n 128 -ngl 99 2>/dev/null | grep -E "^llama|model|test"

echo ""
echo "--- Granite (will run via custom benchmark) ---"
# We'll capture this from the existing benchmark

echo ""
echo "============================================================"
echo "  Scenario 2: Prefill Performance (varying prompt length)"
echo "============================================================"
echo ""

echo "--- llama.cpp Prefill ---"
for pp in 32 64 128 256 512; do
    echo "Prompt length: $pp"
    $LLAMACPP_BENCH -m "$MODEL_PATH" -p $pp -n 0 -ngl 99 2>/dev/null | grep -E "pp\s+$pp\s" || true
done

echo ""
echo "============================================================"
echo "  Scenario 3: Combined Prefill + Decode"
echo "============================================================"
echo ""

echo "--- llama.cpp Combined ---"
$LLAMACPP_BENCH -m "$MODEL_PATH" -p 128 -n 64 -ngl 99 2>/dev/null | grep -E "^llama|pp |tg "

echo ""
echo "============================================================"
echo "  Scenario 4: Batch Size Scaling (if supported)"
echo "============================================================"
echo ""

echo "--- llama.cpp Batch Sizes ---"
for batch in 1 8 16 32; do
    echo "Batch size: $batch"
    $LLAMACPP_BENCH -m "$MODEL_PATH" -p 64 -n 32 -b $batch -ngl 99 2>/dev/null | grep -E "tg\s" | head -1 || true
done

echo ""
echo "============================================================"
echo "  Full llama-bench Output"
echo "============================================================"
echo ""
$LLAMACPP_BENCH -m "$MODEL_PATH" -p 0,32,128,512 -n 0,32,128 -ngl 99 2>/dev/null

echo ""
echo "============================================================"
echo "  Benchmark Complete"
echo "============================================================"
