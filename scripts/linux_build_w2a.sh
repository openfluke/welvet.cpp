#!/usr/bin/env bash
# Linux: build welvet.cpp, then run w2a Go↔C++ validation.
#
# Usage:
#   ./scripts/linux_build_w2a.sh              # build + go_v_cpp (default)
#   ./scripts/linux_build_w2a.sh --all        # build + entire w2a go test ./...
#   ./scripts/linux_build_w2a.sh --go-v-cpp   # same as default
#   BUILD_DIR=/tmp/welvet-build ./scripts/linux_build_w2a.sh
#
# Expects sibling checkout: ../w2a (override with W2A_ROOT).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
W2A_ROOT="${W2A_ROOT:-$ROOT/../w2a}"
MODE="go_v_cpp"

for arg in "$@"; do
  case "$arg" in
    --all) MODE="all" ;;
    --go-v-cpp) MODE="go_v_cpp" ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "warning: this script is intended for Linux (got $(uname -s)); continuing anyway" >&2
fi

if [[ ! -d "$W2A_ROOT" ]]; then
  echo "error: w2a not found at $W2A_ROOT (set W2A_ROOT)" >&2
  exit 1
fi

echo "==> building welvet.cpp → $BUILD"
bash "$ROOT/scripts/build.sh"
TOOL="$BUILD/welvet_dense_fwd"
if [[ ! -x "$TOOL" ]]; then
  echo "error: missing tool $TOOL" >&2
  exit 1
fi
echo "    version: $($TOOL --version)"

export WELVET_DENSE_FWD="$TOOL"

cd "$W2A_ROOT"
case "$MODE" in
  go_v_cpp)
    echo "==> w2a go_v_cpp (Go Welvet ↔ welvet.cpp)"
    go test ./tests/go_v_cpp/ -v -count=1
    ;;
  all)
    echo "==> w2a full suite (go test ./...)"
    go test ./... -count=1
    ;;
esac

echo "==> ok ($MODE)"
