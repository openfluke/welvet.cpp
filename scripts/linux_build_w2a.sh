#!/usr/bin/env bash
# Linux: build welvet.cpp, then run w2a Go↔C++ validation.
#
# Usage:
#   ./scripts/linux_build_w2a.sh              # build + full go_v_cpp suite
#   ./scripts/linux_build_w2a.sh --all-train  # ALL train sections + timing table
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
#   ./scripts/linux_build_w2a.sh --kmeans-train # KMeans NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --sequential # Sequential Dense chain
#   ./scripts/linux_build_w2a.sh --residual   # Residual Dense F(x)+x
#   ./scripts/linux_build_w2a.sh --mamba      # Mamba minimal SSM
#   ./scripts/linux_build_w2a.sh --metacognition # Metacognition Observed+rules
#   ./scripts/linux_build_w2a.sh --gdn        # GDN Gated DeltaNet (Float32)
#   ./scripts/linux_build_w2a.sh --seqmix     # seqmix Kind contract smoke
#   ./scripts/linux_build_w2a.sh --parallel   # Parallel Dense × 7 combine modes (forward)
#   ./scripts/linux_build_w2a.sh --parallel-train # Parallel Dense NormalBP + Tween families
#   ./scripts/linux_build_w2a.sh --dense-train # Dense NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --rmsnorm-train # RMSNorm NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --layernorm-train # LayerNorm NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --swiglu-train # SwiGLU NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --softmax-train # Softmax NormalBP (all Kind) leaf + volumetric
#   ./scripts/linux_build_w2a.sh --embedding-train # Embedding NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --mha-train  # MHA NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --cnn-train  # CNN1/2/3 NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --convt-train # ConvT1/2/3 NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --rnn-train  # RNN NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --lstm-train # LSTM NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --sequential-train # Sequential NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --residual-train # Residual NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --mamba-train # Mamba NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --metacognition-train # Metacognition NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --gdn-train # GDN truncated-BPTT NormalBP leaf + volumetric (f32)
#   ./scripts/linux_build_w2a.sh --stack-train # Stack NestedStack-of-Dense NormalBP leaf + volumetric
#   ./scripts/linux_build_w2a.sh --stepbp-train # Line StepBP (Stack + FamilyBP)
#   ./scripts/linux_build_w2a.sh --tween-train  # ModeTween projected local + volume
#   ./scripts/linux_build_w2a.sh --tweenchain-train  # ModeTweenChain FamilyBP @ full lr + volume
#   ./scripts/linux_build_w2a.sh --steptween-train  # Line StepTween FamilyTween @ lr*0.5 + MeshTween vol
#   ./scripts/linux_build_w2a.sh --steptweenchain-train  # Line StepTweenChain FamilyBP + MeshTweenChain vol
#   ./scripts/linux_build_w2a.sh --meshtween-train  # MeshTween mesh-clock FamilyTween + volume
#   ./scripts/linux_build_w2a.sh --meshtweenchain-train  # MeshTweenChain mesh-clock FamilyBP + volume
#   ./scripts/linux_build_w2a.sh --tweensplit-train  # ModeTweenSplit 1/N @ full lr + volume
#   ./scripts/linux_build_w2a.sh --steptweensplit-train  # Line StepTweenSplit FamilyTweenSplit + MeshTweenSplit vol
#   ./scripts/linux_build_w2a.sh --tweenalt-train  # ModeTweenAlt Split→Tween + volume
#   ./scripts/linux_build_w2a.sh --steptweenalt-train  # Line StepTweenAlt + MeshTweenAlt vol
#   ./scripts/linux_build_w2a.sh --tweensplitheadproxy-train  # HeadProxy head Jᵀgy + volume
#   ./scripts/linux_build_w2a.sh --tweensplitlinear-train     # Linear 1/N P(W̃ᵀgy) + volume
#   ./scripts/linux_build_w2a.sh --tweensplitfastproxy-train  # FastProxy head GradWOnly + volume
#   ./scripts/linux_build_w2a.sh --tweensplitlinearcache-train # LinearCache refresh/hit + volume
#   ./scripts/linux_build_w2a.sh --tweensplitheadproxyasync-train # HeadProxyAsync stale proxy + volume
#   ./scripts/linux_build_w2a.sh --tweensplitsparse-train # Sparse head+one-hidden + volume
#   ./scripts/linux_build_w2a.sh --meshtweensplit-train # MeshTweenSplit mesh-clock + TweenSplit vol
#   ./scripts/linux_build_w2a.sh --meshtweenalt-train # MeshTweenAlt mesh-clock + TweenAlt vol
#   ./scripts/linux_build_w2a.sh --meshtweensplitfastproxy-train # MeshTweenSplitFastProxy + FastProxy vol
#   ./scripts/linux_build_w2a.sh --meshtweensplitsparse-train # MeshTweenSplitSparse + Sparse vol
#   ./scripts/linux_build_w2a.sh --steptweensplitheadproxy-train # Line StepTweenSplitHeadProxy + HeadProxy vol
#   ./scripts/linux_build_w2a.sh --steptweensplitlinear-train # Line StepTweenSplitLinear + Linear vol
#   ./scripts/linux_build_w2a.sh --steptweensplitfastproxy-train # Line StepTweenSplitFastProxy + FastProxy vol
#   ./scripts/linux_build_w2a.sh --steptweensplitlinearcache-train # Line StepTweenSplitLinearCache + LinearCache vol
#   ./scripts/linux_build_w2a.sh --steptweensplitheadproxyasync-train # Line StepTweenSplitHeadProxyAsync + HeadProxyAsync vol
#   ./scripts/linux_build_w2a.sh --steptweensplitsparse-train # Line StepTweenSplitSparse + Sparse vol
#   ./scripts/linux_build_w2a.sh --freeze-train # ModeFreeze fwd+mse no apply + volume + Parallel FZB1
#   ./scripts/linux_build_w2a.sh --adversarial-train # Parallel BranchMode Adversarial
#   ./scripts/linux_build_w2a.sh --memory-train # Parallel BranchMode Memory + CamKit
#   ./scripts/linux_build_w2a.sh --shadow-train # Parallel BranchMode Shadow KD + CamKit
#   ./scripts/linux_build_w2a.sh --parallel-train # Parallel Dense NormalBP + Tween families (PNB1/PTW1/PTC1/PTS1/PTA1)
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
    --all-train) MODE="all_train" ;;
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
    --kmeans-train) MODE="kmeans_train" ;;
    --sequential) MODE="sequential" ;;
    --residual) MODE="residual" ;;
    --mamba) MODE="mamba" ;;
    --metacognition) MODE="metacognition" ;;
    --gdn) MODE="gdn" ;;
    --seqmix) MODE="seqmix" ;;
    --parallel) MODE="parallel" ;;
    --dense-train) MODE="dense_train" ;;
    --rmsnorm-train) MODE="rmsnorm_train" ;;
    --layernorm-train) MODE="layernorm_train" ;;
    --swiglu-train) MODE="swiglu_train" ;;
    --softmax-train) MODE="softmax_train" ;;
    --embedding-train) MODE="embedding_train" ;;
    --mha-train) MODE="mha_train" ;;
    --cnn-train) MODE="cnn_train" ;;
    --convt-train) MODE="convt_train" ;;
    --rnn-train) MODE="rnn_train" ;;
    --lstm-train) MODE="lstm_train" ;;
    --sequential-train) MODE="sequential_train" ;;
    --residual-train) MODE="residual_train" ;;
    --mamba-train) MODE="mamba_train" ;;
    --metacognition-train) MODE="metacognition_train" ;;
    --gdn-train) MODE="gdn_train" ;;
    --stack-train) MODE="stack_train" ;;
    --stepbp-train) MODE="stepbp_train" ;;
    --tween-train) MODE="tween_train" ;;
    --tweenchain-train) MODE="tweenchain_train" ;;
    --steptween-train) MODE="steptween_train" ;;
    --steptweenchain-train) MODE="steptweenchain_train" ;;
    --meshtween-train) MODE="meshtween_train" ;;
    --meshtweenchain-train) MODE="meshtweenchain_train" ;;
    --tweensplit-train) MODE="tweensplit_train" ;;
    --steptweensplit-train) MODE="steptweensplit_train" ;;
    --tweenalt-train) MODE="tweenalt_train" ;;
    --steptweenalt-train) MODE="steptweenalt_train" ;;
    --tweensplitheadproxy-train) MODE="tweensplitheadproxy_train" ;;
    --tweensplitlinear-train) MODE="tweensplitlinear_train" ;;
    --tweensplitfastproxy-train) MODE="tweensplitfastproxy_train" ;;
    --tweensplitlinearcache-train) MODE="tweensplitlinearcache_train" ;;
    --tweensplitheadproxyasync-train) MODE="tweensplitheadproxyasync_train" ;;
    --tweensplitsparse-train) MODE="tweensplitsparse_train" ;;
    --meshtweensplit-train) MODE="meshtweensplit_train" ;;
    --meshtweenalt-train) MODE="meshtweenalt_train" ;;
    --meshtweensplitfastproxy-train) MODE="meshtweensplitfastproxy_train" ;;
    --meshtweensplitsparse-train) MODE="meshtweensplitsparse_train" ;;
    --steptweensplitheadproxy-train) MODE="steptweensplitheadproxy_train" ;;
    --steptweensplitlinear-train) MODE="steptweensplitlinear_train" ;;
    --steptweensplitfastproxy-train) MODE="steptweensplitfastproxy_train" ;;
    --steptweensplitlinearcache-train) MODE="steptweensplitlinearcache_train" ;;
    --steptweensplitheadproxyasync-train) MODE="steptweensplitheadproxyasync_train" ;;
    --steptweensplitsparse-train) MODE="steptweensplitsparse_train" ;;
    --freeze-train) MODE="freeze_train" ;;
    --adversarial-train) MODE="adversarial_train" ;;
    --memory-train) MODE="memory_train" ;;
    --shadow-train) MODE="shadow_train" ;;
    --parallel-train) MODE="parallel_train" ;;
    -h|--help)
      sed -n '2,60p' "$0"
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
    echo "==> w2a go_v_cpp full (+ parallel)"
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
  kmeans_train)
    echo "==> w2a go_v_cpp KMeans NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppKMeansTrain$' -v -count=1
    ;;
  sequential)
    echo "==> w2a go_v_cpp Sequential"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppSequential$' -v -count=1
    ;;
  residual)
    echo "==> w2a go_v_cpp Residual"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppResidual$' -v -count=1
    ;;
  mamba)
    echo "==> w2a go_v_cpp Mamba"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMamba$' -v -count=1
    ;;
  metacognition)
    echo "==> w2a go_v_cpp Metacognition"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMetacognition$' -v -count=1
    ;;
  gdn)
    echo "==> w2a go_v_cpp GDN"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppGDN$' -v -count=1
    ;;
  seqmix)
    echo "==> w2a go_v_cpp seqmix"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppSeqmix$' -v -count=1
    ;;
  parallel)
    echo "==> w2a go_v_cpp Parallel"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppParallel$' -v -count=1
    ;;
  dense_train)
    echo "==> w2a go_v_cpp Dense NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppDenseTrain$' -v -count=1
    ;;
  rmsnorm_train)
    echo "==> w2a go_v_cpp RMSNorm NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppRMSNormTrain$' -v -count=1
    ;;
  layernorm_train)
    echo "==> w2a go_v_cpp LayerNorm NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppLayerNormTrain$' -v -count=1
    ;;
  swiglu_train)
    echo "==> w2a go_v_cpp SwiGLU NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppSwiGLUTrain$' -v -count=1
    ;;
  softmax_train)
    echo "==> w2a go_v_cpp Softmax NormalBP (all Kind; leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppSoftmaxTrain$' -v -count=1
    ;;
  embedding_train)
    echo "==> w2a go_v_cpp Embedding NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppEmbeddingTrain$' -v -count=1
    ;;
  mha_train)
    echo "==> w2a go_v_cpp MHA NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMHATrain$' -v -count=1
    ;;
  cnn_train)
    echo "==> w2a go_v_cpp CNN1/2/3 NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppCNNTrain$' -v -count=1
    ;;
  convt_train)
    echo "==> w2a go_v_cpp ConvT1/2/3 NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppConvTTrain$' -v -count=1
    ;;
  rnn_train)
    echo "==> w2a go_v_cpp RNN NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppRNNTrain$' -v -count=1
    ;;
  lstm_train)
    echo "==> w2a go_v_cpp LSTM NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppLSTMTrain$' -v -count=1
    ;;
  sequential_train)
    echo "==> w2a go_v_cpp Sequential NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppSequentialTrain$' -v -count=1
    ;;
  residual_train)
    echo "==> w2a go_v_cpp Residual NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppResidualTrain$' -v -count=1
    ;;
  mamba_train)
    echo "==> w2a go_v_cpp Mamba NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMambaTrain$' -v -count=1
    ;;
  metacognition_train)
    echo "==> w2a go_v_cpp Metacognition NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMetacognitionTrain$' -v -count=1
    ;;
  gdn_train)
    echo "==> w2a go_v_cpp GDN NormalBP (leaf + volumetric, f32)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppGDNTrain$' -v -count=1
    ;;
  stack_train)
    echo "==> w2a go_v_cpp Stack NestedStack-of-Dense NormalBP (leaf + volumetric)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStackTrain$' -v -count=1
    ;;
  stepbp_train)
    echo "==> w2a go_v_cpp StepBP Line (thin Stack + FamilyBP)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStepBPTrain$' -v -count=1
    ;;
  tween_train)
    echo "==> w2a go_v_cpp Tween train (ModeTween leaf + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppTweenTrain$' -v -count=1
    ;;
  tweenchain_train)
    echo "==> w2a go_v_cpp TweenChain train (ModeTweenChain leaf + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppTweenChainTrain$' -v -count=1
    ;;
  steptween_train)
    echo "==> w2a go_v_cpp StepTween train (Line FamilyTween + MeshTween volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStepTweenTrain$' -v -count=1
    ;;
  steptweenchain_train)
    echo "==> w2a go_v_cpp StepTweenChain train (Line FamilyBP + MeshTweenChain volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStepTweenChainTrain$' -v -count=1
    ;;
  meshtween_train)
    echo "==> w2a go_v_cpp MeshTween train (mesh-clock FamilyTween + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMeshTweenTrain$' -v -count=1
    ;;
  meshtweenchain_train)
    echo "==> w2a go_v_cpp MeshTweenChain train (mesh-clock FamilyBP + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMeshTweenChainTrain$' -v -count=1
    ;;
  tweensplit_train)
    echo "==> w2a go_v_cpp TweenSplit train (1/N gap @ full lr + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppTweenSplitTrain$' -v -count=1
    ;;
  steptweensplit_train)
    echo "==> w2a go_v_cpp StepTweenSplit train (Line FamilyTweenSplit + MeshTweenSplit volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStepTweenSplitTrain$' -v -count=1
    ;;
  tweenalt_train)
    echo "==> w2a go_v_cpp TweenAlt train (Split→Tween + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppTweenAltTrain$' -v -count=1
    ;;
  steptweenalt_train)
    echo "==> w2a go_v_cpp StepTweenAlt train (Line FamilyTweenAlt + MeshTweenAlt volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStepTweenAltTrain$' -v -count=1
    ;;
  tweensplitheadproxy_train)
    echo "==> w2a go_v_cpp TweenSplitHeadProxy train (head Jᵀgy + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppTweenSplitHeadProxyTrain$' -v -count=1
    ;;
  tweensplitlinear_train)
    echo "==> w2a go_v_cpp TweenSplitLinear train (1/N P(W̃ᵀgy) + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppTweenSplitLinearTrain$' -v -count=1
    ;;
  tweensplitfastproxy_train)
    echo "==> w2a go_v_cpp TweenSplitFastProxy train (head GradWOnly + Wᵀgy proxy + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppTweenSplitFastProxyTrain$' -v -count=1
    ;;
  tweensplitlinearcache_train)
    echo "==> w2a go_v_cpp TweenSplitLinearCache train (cached linear walk + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppTweenSplitLinearCacheTrain$' -v -count=1
    ;;
  tweensplitheadproxyasync_train)
    echo "==> w2a go_v_cpp TweenSplitHeadProxyAsync train (stale proxy hidden + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppTweenSplitHeadProxyAsyncTrain$' -v -count=1
    ;;
  tweensplitsparse_train)
    echo "==> w2a go_v_cpp TweenSplitSparse train (head GradWOnly + one hidden + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppTweenSplitSparseTrain$' -v -count=1
    ;;
  meshtweensplit_train)
    echo "==> w2a go_v_cpp MeshTweenSplit train (mesh-clock FamilyTweenSplit + TweenSplit volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMeshTweenSplitTrain$' -v -count=1
    ;;
  meshtweenalt_train)
    echo "==> w2a go_v_cpp MeshTweenAlt train (mesh-clock FamilyTweenAlt + TweenAlt volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMeshTweenAltTrain$' -v -count=1
    ;;
  meshtweensplitfastproxy_train)
    echo "==> w2a go_v_cpp MeshTweenSplitFastProxy train (mesh-clock FastProxy + FastProxy volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMeshTweenSplitFastProxyTrain$' -v -count=1
    ;;
  meshtweensplitsparse_train)
    echo "==> w2a go_v_cpp MeshTweenSplitSparse train (mesh-clock Sparse + Sparse volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMeshTweenSplitSparseTrain$' -v -count=1
    ;;
  steptweensplitheadproxy_train)
    echo "==> w2a go_v_cpp StepTweenSplitHeadProxy train (Line HeadProxy + HeadProxy volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStepTweenSplitHeadProxyTrain$' -v -count=1
    ;;
  steptweensplitlinear_train)
    echo "==> w2a go_v_cpp StepTweenSplitLinear train (Line Linear + Linear volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStepTweenSplitLinearTrain$' -v -count=1
    ;;
  steptweensplitfastproxy_train)
    echo "==> w2a go_v_cpp StepTweenSplitFastProxy train (Line FastProxy + FastProxy volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStepTweenSplitFastProxyTrain$' -v -count=1
    ;;
  steptweensplitlinearcache_train)
    echo "==> w2a go_v_cpp StepTweenSplitLinearCache train (Line LinearCache + LinearCache volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStepTweenSplitLinearCacheTrain$' -v -count=1
    ;;
  steptweensplitheadproxyasync_train)
    echo "==> w2a go_v_cpp StepTweenSplitHeadProxyAsync train (Line HeadProxyAsync + HeadProxyAsync volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStepTweenSplitHeadProxyAsyncTrain$' -v -count=1
    ;;
  steptweensplitsparse_train)
    echo "==> w2a go_v_cpp StepTweenSplitSparse train (Line Sparse + Sparse volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStepTweenSplitSparseTrain$' -v -count=1
    ;;
  freeze_train)
    echo "==> w2a go_v_cpp Freeze train (fwd+mse; no weight update + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppFreezeTrain$' -v -count=1
    ;;
  adversarial_train)
    echo "==> w2a go_v_cpp Adversarial train (Parallel BranchMode neg-LR)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppAdversarialTrain$' -v -count=1
    ;;
  memory_train)
    echo "==> w2a go_v_cpp Memory train (Parallel BranchMode surprise gate)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppMemoryTrain$' -v -count=1
    ;;
  shadow_train)
    echo "==> w2a go_v_cpp Shadow train (Parallel BranchMode KD teacher)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppShadowTrain$' -v -count=1
    ;;
  parallel_train)
    echo "==> w2a go_v_cpp Parallel NormalBP train (Dense all combines + poly PPB1 + nested + volume PNV1)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppParallelNBPTrain$|^TestGoVCppParallelPolyTrain$' -v -count=1
    echo "==> w2a go_v_cpp Parallel Tween-family train (Tween/Chain/Split/Alt + volume)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppParallelTweenTrain$' -v -count=1
    echo "==> w2a go_v_cpp Mesh* Parallel volume (MeshBP/Tween/Chain/Split/Alt/FastProxy/Sparse)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppStepBPTrain$|^TestGoVCppMeshTweenTrain$|^TestGoVCppMeshTweenChainTrain$|^TestGoVCppMeshTweenSplitTrain$|^TestGoVCppMeshTweenAltTrain$|^TestGoVCppMeshTweenSplitFastProxyTrain$|^TestGoVCppMeshTweenSplitSparseTrain$' -v -count=1
    echo "==> w2a go_v_cpp Freeze/Shadow/Adv/Memory Parallel 3D"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppFreezeTrain$|^TestGoVCppShadowTrain$|^TestGoVCppAdversarialTrain$|^TestGoVCppMemoryTrain$' -v -count=1
    ;;
  all_train)
    echo "==> w2a go_v_cpp ALL TRAIN (section table + totals)"
    go test ./tests/go_v_cpp/ -run '^TestGoVCppAllTrain$' -v -count=1 -timeout 120m
    ;;
  all)
    echo "==> w2a full suite (go test ./...)"
    go test ./... -count=1
    ;;
esac

echo "==> ok ($MODE)"
