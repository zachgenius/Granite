#!/bin/bash
# run_leak_check.sh - Run leak checks for granite_tests (best-effort).
#
# Usage:
#   ./scripts/run_leak_check.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_BIN="$ROOT_DIR/build/granite_tests"

if [[ ! -x "$TEST_BIN" ]]; then
    echo "Test binary not found: $TEST_BIN"
    echo "Build first: cmake --build build --parallel"
    exit 1
fi

if [[ "${GRANITE_LEAK_CHECK_CMD:-}" != "" ]]; then
    echo "Using custom leak check command: $GRANITE_LEAK_CHECK_CMD"
    eval "$GRANITE_LEAK_CHECK_CMD"
    exit $?
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v leaks >/dev/null 2>&1; then
        echo "Running leaks on macOS..."
        leaks -atExit -- "$TEST_BIN"
        exit $?
    fi
fi

if command -v valgrind >/dev/null 2>&1; then
    echo "Running valgrind leak check..."
    valgrind --leak-check=full --errors-for-leak-kinds=definite "$TEST_BIN"
    exit $?
fi

echo "No leak checker found. Install 'leaks' (macOS) or 'valgrind' (Linux)."
exit 0
