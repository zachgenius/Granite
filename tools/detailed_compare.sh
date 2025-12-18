#!/bin/bash

# Detailed Granite vs llama.cpp Benchmark with Resource Monitoring
# Usage: ./detailed_compare.sh <model.gguf>
# Environment variables:
#   LLAMA_CPP_DIR - Path to llama.cpp directory (optional)

if [ -z "$1" ]; then
    echo "Usage: $0 <model.gguf>"
    echo "Environment variables:"
    echo "  LLAMA_CPP_DIR - Path to llama.cpp directory (optional)"
    exit 1
fi

MODEL_PATH="$1"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}"
GRANITE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=============================================="
echo "Detailed Performance & Resource Comparison"
echo "=============================================="
echo ""
echo "Model: $MODEL_PATH"
echo "Date: $(date)"
echo ""

# Function to monitor a process and report stats
monitor_process() {
    local name="$1"
    shift
    local cmd=("$@")
    
    echo "--- Running: $name ---"
    
    # Start the process and capture its PID
    "${cmd[@]}" &
    local pid=$!
    
    local max_mem=0
    local max_cpu=0
    local gpu_usage="N/A"
    
    # Monitor resource usage while process runs
    while kill -0 $pid 2>/dev/null; do
        # Get memory and CPU usage
        local stats=$(ps -p $pid -o rss=,pcpu= 2>/dev/null | tail -1)
        if [ -n "$stats" ]; then
            local mem=$(echo $stats | awk '{print $1}')
            local cpu=$(echo $stats | awk '{print $2}')
            
            if [ "${mem:-0}" -gt "${max_mem:-0}" ] 2>/dev/null; then
                max_mem=$mem
            fi
            if (( $(echo "${cpu:-0} > ${max_cpu:-0}" | bc -l) )); then
                max_cpu=$cpu
            fi
        fi
        sleep 0.1
    done
    
    wait $pid
    local exit_code=$?
    
    echo ""
    echo "Resource Usage Summary for $name:"
    echo "  Peak Memory: $((max_mem / 1024)) MB"
    echo "  Peak CPU: ${max_cpu}%"
    echo ""
    
    return $exit_code
}

echo ""
echo "=============================================="
echo "1. Granite Benchmark (CPU + BLAS)"
echo "=============================================="
echo ""

# Run Granite benchmark
"$GRANITE_DIR/build/granite_benchmark" "$MODEL_PATH" 2>&1 | tee /tmp/granite_detail.txt &
GRANITE_PID=$!

# Monitor Granite
MAX_MEM_G=0
MAX_CPU_G=0
while kill -0 $GRANITE_PID 2>/dev/null; do
    STATS=$(ps -p $GRANITE_PID -o rss=,pcpu= 2>/dev/null | tail -1)
    if [ -n "$STATS" ]; then
        MEM=$(echo $STATS | awk '{print $1}')
        CPU=$(echo $STATS | awk '{print $2}')
        [ "${MEM:-0}" -gt "${MAX_MEM_G:-0}" ] 2>/dev/null && MAX_MEM_G=$MEM
    fi
    sleep 0.2
done
wait $GRANITE_PID

echo ""
echo "Granite Resource Usage:"
echo "  Peak Memory: $((MAX_MEM_G / 1024)) MB"
echo ""

echo ""
echo "=============================================="
echo "2. llama.cpp Benchmark (Metal GPU)"
echo "=============================================="
echo ""

if [ -z "$LLAMA_CPP_DIR" ]; then
    echo "Skipping llama.cpp benchmark (LLAMA_CPP_DIR not set)"
    echo "To run llama.cpp comparison: export LLAMA_CPP_DIR=/path/to/llama.cpp"
    LLAMA_PID=""
else
    # Run llama.cpp with GPU
    "$LLAMA_CPP_DIR/build/bin/llama-bench" \
        -m "$MODEL_PATH" \
        -p 1,16 \
        -n 1,20 \
        -ngl 999 \
        -r 3 \
        -o md 2>&1 | tee /tmp/llama_gpu.txt &
    LLAMA_PID=$!
fi

# Monitor llama.cpp
MAX_MEM_L=0
if [ -n "$LLAMA_PID" ]; then
    while kill -0 $LLAMA_PID 2>/dev/null; do
        STATS=$(ps -p $LLAMA_PID -o rss=,pcpu= 2>/dev/null | tail -1)
        if [ -n "$STATS" ]; then
            MEM=$(echo $STATS | awk '{print $1}')
            [ "${MEM:-0}" -gt "${MAX_MEM_L:-0}" ] 2>/dev/null && MAX_MEM_L=$MEM
        fi
        sleep 0.2
    done
    wait $LLAMA_PID

    echo ""
    echo "llama.cpp (GPU) Resource Usage:"
    echo "  Peak Memory: $((MAX_MEM_L / 1024)) MB"
    echo ""
fi

echo ""
echo "=============================================="
echo "3. Performance Comparison Summary"
echo "=============================================="
echo ""

# Extract key metrics
echo "| Metric | Granite (CPU) | llama.cpp (Metal GPU) | Ratio |"
echo "|--------|---------------|----------------------|-------|"

# Extract Granite decode speed (tokens/sec at cache len 1)
GRANITE_TPS=$(grep -A5 "Decode" /tmp/granite_detail.txt | grep "^              1" | awk '{print $3}')
# Extract llama.cpp decode speed (tg1)
LLAMA_TPS=$(grep "tg1" /tmp/llama_gpu.txt | awk -F'|' '{print $7}' | awk '{print $1}')

if [ -n "$GRANITE_TPS" ] && [ -n "$LLAMA_TPS" ]; then
    RATIO=$(echo "scale=1; $LLAMA_TPS / $GRANITE_TPS" | bc 2>/dev/null || echo "N/A")
    echo "| Decode (tok/s) | $GRANITE_TPS | $LLAMA_TPS | ${RATIO}x |"
fi

# Extract prefill speed
GRANITE_PP=$(grep -A5 "Prefill" /tmp/granite_detail.txt | grep "^             16" | awk '{print $3}')
LLAMA_PP=$(grep "pp16" /tmp/llama_gpu.txt | awk -F'|' '{print $7}' | awk '{print $1}')

if [ -n "$GRANITE_PP" ] && [ -n "$LLAMA_PP" ]; then
    RATIO=$(echo "scale=1; $LLAMA_PP / $GRANITE_PP" | bc 2>/dev/null || echo "N/A")
    echo "| Prefill 16 (tok/s) | $GRANITE_PP | $LLAMA_PP | ${RATIO}x |"
fi

echo "| Peak Memory (MB) | $((MAX_MEM_G / 1024)) | $((MAX_MEM_L / 1024)) | - |"

echo ""
echo "=============================================="
echo "Key Observations:"
echo "=============================================="
echo ""
echo "1. llama.cpp uses Metal GPU compute shaders for all operations"
echo "2. Granite currently uses CPU + Accelerate BLAS (Metal kernels exist but not integrated)"
echo "3. Both frameworks use unified memory on Apple Silicon"
echo ""
echo "To achieve parity, Granite needs to:"
echo "  - Integrate Metal compute shaders into TransformerModel::forward()"
echo "  - Implement quantized (Q4_K) GPU kernels"
echo "  - Use async GPU command submission with batching"
