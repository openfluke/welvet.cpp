#!/usr/bin/env bash
# Host build (Linux/macOS/Windows via native toolchain).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
echo "built: $BUILD"
echo "  lib:  $BUILD/libwelvet.so (or .dylib/.dll)"
echo "  tool: $BUILD/welvet_dense_fwd"
echo "  ex:   $BUILD/example_dense_forward"
