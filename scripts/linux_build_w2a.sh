#!/usr/bin/env bash
# Linux: build welvet.cpp, then run w2a Go↔C++ validation.
#
# Usage:
#   ./scripts/linux_build_w2a.sh              # build + full go_v_cpp suite
#   ./scripts/linux_build_w2a.sh --mha        # MHA only
#   ./scripts/linux_build_w2a.sh --swiglu     # SwiGLU only
#   ./scripts/linux_build_w2a.sh --rmsnorm    # RMSNorm only
#   ./scripts/linux_build_w2a.sh --layernorm  # LayerNorm only
#   ./scripts/linux_build_w2a.sh --cnn        # CNN1/2/3 only
#   ./scripts/linux_build_w2a.sh --convt      # ConvT1/2/3 only
#   ./scripts/linux_build_w2a.sh --rnn        # RNN only
#   ./scripts/linux_build_w2a.sh --lstm       # LSTM only
#   ./scripts/linux_build_w2a.sh --embedding  # Embedding only
#   ./scripts/linux_build_w2a.sh --softmax    # Softmax all Kind variants
#   ./scripts/linux_build_w2a.sh --kmeans     # KMeans soft assignment
#   ./scripts/linux_build_w2a.sh --sequential # Sequential Dense chain
#   ./scripts/linux_build_w2a.sh --residual   # Residual Dense F(x)+x
#   ./scripts/linux_build_w2a.sh --go-v-cpp   # same as default
#   ./scripts/linux_build_w2a.sh --all        # entire w2a go test ./...
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
    --mha) MODE="mha" ;;
    --swiglu) MODE="swiglu" ;;
    --rmsnorm) MODE="rmsnorm" ;;
    --layernorm) MODE="layernorm" ;;
    --cnn) MODE="cnn" ;;
    --convt) MODE="convt" ;;
    --rnn) MODE="rnn" ;;
    --lstm) MODE="lstm" ;;
    --embedding) MODE="embedding" ;;
    --softmax) MODE="softmax" ;;
    --kmeans) MODE="kmeans" ;;
    --sequential) MODE="sequential" ;;
    --residual) MODE="residual" ;;
    -h|--help)
      sed -n '2,26p' "$0"
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
    echo "==> w2a go_v_cpp full (dense + arch + transformer + cnn + convt + rnn + lstm + emb + sm + km + seq + res)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCpp$' -v -count=1
    ;;
  mha)
    echo "==> w2a go_v_cpp MHA only"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMHA$' -v -count=1
    ;;
  swiglu)
    echo "==> w2a go_v_cpp SwiGLU only"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppSwiGLU$' -v -count=1
    ;;
  rmsnorm)
    echo "==> w2a go_v_cpp RMSNorm only"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppRMSNorm$' -v -count=1
    ;;
  layernorm)
    echo "==> w2a go_v_cpp LayerNorm only"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppLayerNorm$' -v -count=1
    ;;
  cnn)
    echo "==> w2a go_v_cpp CNN1/2/3 only"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppCNN$' -v -count=1
    ;;
  convt)
    echo "==> w2a go_v_cpp ConvT1/2/3 only"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppConvT$' -v -count=1
    ;;
  rnn)
    echo "==> w2a go_v_cpp RNN only"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppRNN$' -v -count=1
    ;;
  lstm)
    echo "==> w2a go_v_cpp LSTM only"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppLSTM$' -v -count=1
    ;;
  embedding)
    echo "==> w2a go_v_cpp Embedding only"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppEmbedding$' -v -count=1
    ;;
  softmax)
    echo "==> w2a go_v_cpp Softmax all Kind"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppSoftmax$' -v -count=1
    ;;
  kmeans)
    echo "==> w2a go_v_cpp KMeans"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppKMeans$' -v -count=1
    ;;
  sequential)
    echo "==> w2a go_v_cpp Sequential"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppSequential$' -v -count=1
    ;;
  residual)
    echo "==> w2a go_v_cpp Residual"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppResidual$' -v -count=1
    ;;
  all)
    echo "==> w2a full suite (go test ./...)"
    go test ./... -count=1
    ;;
esac

echo "==> ok ($MODE)"
