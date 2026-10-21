#include "welvet/gaps.hpp"

#include <iostream>

namespace welvet::gaps {

namespace {

const std::vector<Entry>& build() {
  static const std::vector<Entry> k = {
      // --- Implemented (M1 slice) ---
      {"core.dtype.float32", "core", "DType Float32 name + forward gate", true},
      {"core.activate.f64", "core", "Activate/ActivateDeriv float64 (Go parity)", true},
      {"core.fp8_fp4_codecs", "core", "FP8/FP4 pack-unpack (FormatNone)", true},
      {"weights.matvec.f32_none", "weights", "FormatNone MatVec f64-acc (all dtypes)", true},
      {"weights.native.all_dtypes", "weights", "native pack/decode all 34 FormatNone dtypes", true},
      {"dense.forward.f32_cpu", "layers", "Dense ForwardCPUTiled serial FormatNone×34", true},
      {"dense.all_dtypes", "layers", "Dense FormatNone × 34 dtypes forward", true},
      {"abi.dense_forward_f32", "interop", "C ABI welvet_dense_forward_f32 (+dtype)", true},
      {"abi.arch_topology", "interop", "C ABI welvet_arch_topology + WVA1/WVF1 tool", true},
      {"abi.mha_forward_f32", "interop", "C ABI welvet_mha_forward_f32 + WVM1 tool", true},
      {"abi.mha_normalbp_f32", "interop", "C ABI welvet_mha_normalbp_f32 + WHB1/WHV1 tool", true},
      {"abi.swiglu_forward_f32", "interop", "C ABI welvet_swiglu_forward_f32 + WVS1 tool", true},
      {"abi.swiglu_normalbp_f32", "interop", "C ABI welvet_swiglu_normalbp_f32 + WSB1/WSV1 tool", true},
      {"abi.rmsnorm_forward_f32", "interop", "C ABI welvet_rmsnorm_forward_f32 + WVR1 tool", true},
      {"abi.rmsnorm_normalbp_f32", "interop", "C ABI welvet_rmsnorm_normalbp_f32 + WRB1/WRV1 tool", true},
      {"abi.layernorm_forward_f32", "interop", "C ABI welvet_layernorm_forward_f32 + WVL1 tool", true},
      {"abi.layernorm_normalbp_f32", "interop", "C ABI welvet_layernorm_normalbp_f32 + WLB1/WLV1 tool", true},
      {"abi.cnn_forward_f32", "interop", "C ABI welvet_cnn{1,2,3}_forward_f32 + WVC1/2/3 tool", true},
      {"abi.convt_forward_f32", "interop", "C ABI welvet_convt{1,2,3}_forward_f32 + WVT1/2/3 tool", true},
      {"abi.convt_normalbp_f32", "interop",
       "C ABI welvet_convt{1,2,3}_normalbp_f32 + T*B1/T*V1 tool", true},
      {"abi.rnn_forward_f32", "interop", "C ABI welvet_rnn_forward_f32 + WRN1 tool", true},
      {"abi.lstm_forward_f32", "interop", "C ABI welvet_lstm_forward_f32 + WLS1 tool", true},
      {"abi.embedding_forward_f32", "interop", "C ABI welvet_embedding_forward_f32 + WEM1 tool", true},
      {"abi.embedding_normalbp_f32", "interop", "C ABI welvet_embedding_normalbp_f32 + WEB1/WEV1 tool", true},
      {"abi.softmax_forward_f32", "interop", "C ABI welvet_softmax_forward_f32 + WSM1 tool (all Kind)", true},
      {"abi.softmax_normalbp_f32", "interop", "C ABI welvet_softmax_normalbp_f32 + WXB1/WXV1 tool (all Kind)", true},
      {"abi.kmeans_forward_f32", "interop", "C ABI welvet_kmeans_forward_f32 + WKM1 tool", true},
      {"abi.sequential_forward_f32", "interop",
       "C ABI welvet_sequential_forward_f32 + WSQ1 tool (Dense children)", true},
      {"abi.residual_forward_f32", "interop",
       "C ABI welvet_residual_forward_f32 + WRE1 tool (Dense F + skip)", true},
      {"abi.mamba_forward_f32", "interop",
       "C ABI welvet_mamba_forward_f32 + WMB1 tool (minimal SSM)", true},
      {"abi.metacognition_forward_f32", "interop",
       "C ABI welvet_metacognition_forward_f32 + WMC1 tool", true},
      {"abi.gdn_forward_f32", "interop",
       "C ABI welvet_gdn_forward_f32 + WGD1 tool (Float32 FormatNone)", true},
      {"abi.seqmix_kind", "interop",
       "C ABI welvet_seqmix_kind_name + mixer tags + WSX1 smoke", true},
      {"abi.parallel_forward_f32", "interop",
       "C ABI welvet_parallel_forward_f32 + WPL1 (7 combine modes)", true},
      {"backends.cpu", "backends", "Native CPU target available", true},
      {"core.layer_types", "core", "LayerType 0–22 + LayerMeta descriptor", true},
      {"arch.volume", "architecture", "Grid/Cell/Coord/HopOrder/remote links", true},
      {"runtime.forward.dense", "runtime", "volume Forward Dense-only hop walker", true},

      // --- Core / dtypes ---
      {"core.numeric.acts", "core", "gap? all 15 Go Numeric activation host types", false},
      {"core.tensor.generic", "core", "gap? Tensor[T] generic host/device buffers", false},

      // --- Weights / quant ---
      {"weights.quant.all_formats", "weights", "gap? all quant.Format MatVec/MatVecT", false},
      {"weights.sgd.native", "weights", "ApplySGD FormatNone native (Go parity)", true},
      {"weights.matvect.f32", "weights", "MatVecT dX path (FormatNone)", true},
      {"weights.bias.train", "weights", "gap? bias gradient + update", false},

      // --- Dense remaining ---
      {"dense.backward", "layers", "Dense NormalBP dX/dW + act deriv + ApplyGradSGD", true},
      {"dense.forward.mc", "layers", "gap? Dense multicore tiled forward", false},
      {"dense.forward.simd", "layers", "imported DotTile/DotI8 gemvSIMD (welvet-to-rust_n_cpp)", true},
      {"simd.dot_tile", "simd", "imported AVX2/NEON f32AccF64 + rowsF32/F64", true},
      {"simd.dot_i8", "simd", "imported AVX2 DotI8Tile", true},
      {"simd.packed_expand", "simd", "imported packed FP8/FP4/NF4 expand kernels", true},
      {"dense.forward.webgpu", "layers", "gap? Dense WebGPU path (native equiv)", false},
      {"dense.all_quants", "layers", "gap? Dense × all quant formats", false},

      // --- Other layers (23 kinds) ---
      {"layer.mha", "layers", "MHA ModeSelf forward (causal/bi, RoPE/none, GQA) FormatNone×34", true},
      {"layer.mha.cross", "layers", "gap? MHA ModeCross / SetContext", false},
      {"layer.mha.backward", "layers",
       "MHA ModeSelf NormalBP (RoPE/bi/GQA) FormatNone×34; dropout/QKNorm/cross gap?", true},
      {"layer.swiglu", "layers", "SwiGLU forward Gate/Up/Down + SiLU⊙ FormatNone×34", true},
      {"layer.swiglu.backward", "layers", "SwiGLU NormalBP Gate/Up/Down + SiLU⊙ grads FormatNone×34", true},
      {"layer.rmsnorm", "layers", "RMSNorm forward x̂ + γ⊙ FormatNone×34", true},
      {"layer.rmsnorm.backward", "layers", "RMSNorm NormalBP dγ/dx + ApplyGradSGD FormatNone×34", true},
      {"layer.layernorm", "layers", "LayerNorm forward x̂ + γ⊙+β FormatNone×34", true},
      {"layer.layernorm.backward", "layers", "LayerNorm NormalBP dγ/dβ/dx + ApplyGradSGD FormatNone×34", true},
      {"layer.cnn1", "layers", "CNN1 im2col+Dense forward FormatNone×34", true},
      {"layer.cnn2", "layers", "CNN2 im2col+Dense forward FormatNone×34", true},
      {"layer.cnn3", "layers", "CNN3 im2col+Dense forward FormatNone×34", true},
      {"layer.cnn.backward", "layers",
       "CNN1/2/3 NormalBP im2col+Dense bwd+col2im FormatNone×34; C1/C2/C3 B/V wires", true},
      {"abi.cnn_normalbp_f32", "interop", "C ABI welvet_cnn{1,2,3}_normalbp_f32 + C*B1/C*V1 tool",
       true},
      {"layer.convt1", "layers", "ConvT1 scatter forward FormatNone×34", true},
      {"layer.convt2", "layers", "ConvT2 scatter forward FormatNone×34", true},
      {"layer.convt3", "layers", "ConvT3 scatter forward FormatNone×34", true},
      {"layer.convt.backward", "layers",
       "ConvT1/2/3 NormalBP scatter adjoint + ApplyGradSGD FormatNone×34; T1/T2/T3 B/V wires",
       true},
      {"layer.rnn", "layers", "RNN forward IH/HH + tanh FormatNone×34", true},
      {"layer.rnn.backward", "layers",
       "RNN NormalBP BPTT + ApplyGradSGD FormatNone×34; RNB1/RNV1 wires", true},
      {"abi.rnn_normalbp_f32", "interop", "C ABI welvet_rnn_normalbp_f32 + RNB1/RNV1 tool", true},
      {"layer.lstm", "layers", "LSTM forward gates i/f/g/o FormatNone×34", true},
      {"layer.lstm.backward", "layers",
       "LSTM NormalBP BPTT + ApplyGradSGD FormatNone×34; LSB1/LSV1 wires", true},
      {"abi.lstm_normalbp_f32", "interop", "C ABI welvet_lstm_normalbp_f32 + LSB1/LSV1 tool", true},
      {"layer.embedding", "layers", "Embedding gather forward FormatNone×34", true},
      {"layer.embedding.backward", "layers", "Embedding NormalBP scatter dW + ApplyGradSGD FormatNone×34", true},
      {"layer.softmax", "layers",
       "Softmax forward all Kind (std/temp/gumbel/masked/sparse/entmax/grid/hier)", true},
      {"layer.softmax.backward", "layers", "Softmax NormalBP Jacobian dX (all Kind; weightless)", true},
      {"layer.kmeans", "layers",
       "KMeans soft assignment forward (probs|features) FormatNone×34", true},
      {"layer.kmeans.backward", "layers", "gap? KMeans soft assignment backward", false},
      {"layer.parallel", "layers",
       "Parallel Dense branches + 7 combine modes FormatNone×34", true},
      {"layer.parallel.poly", "layers",
       "gap? Parallel NewFromBranches / Nested / ResidualGraft / cameral", false},
      {"layer.parallel.backward", "layers",
       "Parallel Add/Avg splitCombineGrad + BranchModes Adv/Memory/Shadow + CamKit; ADB1/MMB1/SHB1", true},
      {"layer.sequential", "layers",
       "Sequential Dense Dim→Dim chain forward FormatNone×34", true},
      {"layer.sequential.ops", "layers",
       "gap? Sequential NewFromOps (SwiGLU/RMSNorm/LayerNorm mix)", false},
      {"layer.sequential.backward", "layers",
       "Sequential NormalBP reverse Dense chain + ApplyGradSGD FormatNone×34; SQB1/SQV1 wires",
       true},
      {"abi.sequential_normalbp_f32", "interop",
       "C ABI welvet_sequential_normalbp_f32 + SQB1/SQV1 tool", true},
      {"layer.residual", "layers",
       "Residual Dense F(x)+x forward FormatNone×34", true},
      {"layer.residual.ops", "layers",
       "gap? Residual NewFromOps (SwiGLU/RMSNorm/LayerNorm mix)", false},
      {"layer.residual.backward", "layers",
       "Residual NormalBP F reverse Dense + skip grad + ApplyGradSGD FormatNone×34; REB1/REV1 wires",
       true},
      {"abi.residual_normalbp_f32", "interop",
       "C ABI welvet_residual_normalbp_f32 + REB1/REV1 tool", true},
      {"layer.metacognition", "layers",
       "Metacognition Observed Dense + stability rules FormatNone×34", true},
      {"layer.metacognition.rules_wire", "layers",
       "gap? Metacognition custom rules on wire / multi-step fireCount", false},
      {"layer.metacognition.backward", "layers",
       "Metacognition NormalBP lastGate-scale + Observed Dense bwd + ApplyGradSGD FormatNone×34; MCB1/MCV1 wires",
       true},
      {"abi.metacognition_normalbp_f32", "interop",
       "C ABI welvet_metacognition_normalbp_f32 + MCB1/MCV1 tool", true},
      {"layer.mamba", "layers",
       "Mamba minimal selective SSM forward FormatNone×34", true},
      {"layer.mamba.full", "layers",
       "gap? Mamba-N (DState/B/C/conv1d/SiLU) beyond minimal scan", false},
      {"layer.mamba.backward", "layers",
       "Mamba NormalBP scan BPTT + In/Out proj + ALog/DSkip SGD FormatNone×34; MBB1/MBV1 wires",
       true},
      {"abi.mamba_normalbp_f32", "interop",
       "C ABI welvet_mamba_normalbp_f32 + MBB1/MBV1 tool", true},
      {"layer.gdn", "layers",
       "GDN Gated DeltaNet forward Float32 FormatNone (f32-acc GEMV)", true},
      {"layer.gdn.dtype34", "layers", "gap? GDN ×34 dtypes (Go Float32-only today)", false},
      {"layer.gdn.backward", "layers",
       "GDN truncated-BPTT NormalBP (state/conv stop-grad) f32; GDB1/GDV1 wires", true},
      {"abi.gdn_normalbp_f32", "interop",
       "C ABI welvet_gdn_normalbp_f32 + GDB1/GDV1 tool", true},
      {"layer.stack", "layers",
       "thin Stack host (ordered children + FamilyBP); NestedStack-of-Dense NormalBP signed; full cameral NewFromOps/CamSync gap?", true},
      {"layer.stack.backward", "layers",
       "Stack NestedStack-of-Dense NormalBP + ApplyGradSGD FormatNone×34; volume Step; STKB/STKV wires", true},
      {"abi.stack_normalbp_f32", "interop",
       "C ABI welvet_stack_normalbp_f32 + STKB/STKV tool", true},
      {"layer.stack.cameral", "layers", "gap? Stack NewFromOps + CamSync / BranchModes", false},
      {"runtime.line_stepbp", "runtime",
       "Line TickLine FamilyBP/…/FamilyTweenSplitSparse on pop; STB1/…/SHA1/SSP1 + welvet_stack_stepbp_dense_f32", true},
      {"abi.stack_stepbp_f32", "interop", "C ABI welvet_stack_stepbp_dense_f32 + STB1/STBO tool", true},
      {"layer.seqmix", "layers",
       "seqmix Kind/Contract helper + MHA/Mamba/GDN tags (no Forward)", true},
      {"layer.seqmix.conv_mix", "layers", "gap? seqmix KindConvMix implementor", false},

      // --- Architecture / runtime remaining ---
      {"runtime.forward", "runtime", "gap? volume Forward all layer kinds (Dense done)", false},
      {"runtime.backward", "runtime", "volume Backward Dense+RMSNorm+LayerNorm+SwiGLU + training.Step (NormalBP)", true},
      {"runtime.training", "runtime", "NormalBP volume Step + Line StepBP/StepTween/StepTweenSplit/StepTweenAlt; other TrainModes gap?", true},
      {"runtime.tween_train", "runtime", "TrainStackMSE ModeTween (soft_lr=lr*0.5)", true},
      {"runtime.tween_chain_train", "runtime", "TrainStackMSE ModeTweenChain (FamilyBP @ full lr)", true},
      {"runtime.step_mesh", "runtime",
       "MeshBP/MeshTween/MeshTweenChain: step Credit FamilyBP|FamilyTween; MBP1/MTW1/MTC1; ModeMesh* volume", true},
      {"arch.branches", "architecture", "gap? ParallelBranches/SequentialLayers combine", false},
      {"tiling.cpu", "tiling", "gap? CPU tile partition helpers", false},
      {"simd.x86_arm", "simd", "imported DotTile (x86 AVX2 + arm64 NEON stubs)", true},
      {"train.tuned_outer", "training", "gap? wire imported tuned_outer/tuned_tile/backward_native", false},

      // --- Systems / model ---
      {"systems.dna", "systems", "gap? DNA", false},
      {"systems.evolution", "systems", "gap? evolution", false},
      {"systems.tween", "systems",
       "ModeTween + MeshTween/MeshTweenSplit/Alt/FastProxy/Sparse mesh-clock; TWB1/MTW1/MTS1/MTA1/MTF1/MSS1",
       true},
      {"systems.tween_chain", "systems", "ModeTweenChain FamilyBP @ full lr; TCB1/TCBO + TCV1/TCVO + TRC1…MBTC", true},
      {"systems.tween_split", "systems",
       "ModeTweenSplit/…/HeadProxyAsync/Sparse + Line StepTweenSplit; TSB1…/HAB1/SPB1…",
       true},
      {"runtime.tween_split_train", "runtime", "TrainStackMSE ModeTweenSplit (OpenSplitTape twin)", true},
      {"runtime.tween_split_head_proxy_train", "runtime",
       "TrainStackMSE ModeTweenSplitHeadProxy (head Jᵀgy; hidden 1/(N-1) P(gx))", true},
      {"runtime.tween_split_linear_train", "runtime",
       "TrainStackMSE ModeTweenSplitLinear (1/N P(gy); Dense gx=Wᵀgy no act')", true},
      {"runtime.tween_split_fast_proxy_train", "runtime",
       "TrainStackMSE ModeTweenSplitFastProxy (head GradWOnly; hidden 1/(N-1) P(Wᵀgy))", true},
      {"runtime.tween_split_linear_cache_train", "runtime",
       "TrainStackMSE ModeTweenSplitLinearCache (cached linear; refresh every 20)", true},
      {"runtime.tween_split_head_proxy_async_train", "runtime",
       "TrainStackMSE ModeTweenSplitHeadProxyAsync (stale proxy hidden; head GradWOnly)", true},
      {"runtime.tween_split_sparse_train", "runtime",
       "TrainStackMSE ModeTweenSplitSparse (head GradWOnly; one hidden P(Wᵀgy))", true},
      {"runtime.freeze_train", "runtime",
       "TrainStackMSE ModeFreeze (fwd+mse; no ApplyGrad)", true},
      {"runtime.parallel_train", "runtime",
       "TrainMSE Parallel BranchModes Adv/Memory/Shadow + CamKit (Dense Add)", true},
      {"systems.tween_alt", "systems",
       "ModeTweenAlt + Line StepTweenAlt Split→Tween; TAB1/STA1 + TAV1/TRA1…MBTA", true},
      {"runtime.tween_alt_train", "runtime", "TrainStackMSE ModeTweenAlt (Split @ lr → Tween @ lr×0.5)", true},
      {"systems.tanhi", "systems", "gap? TANHI", false},
      {"systems.telemetry", "systems", "gap? telemetry", false},
      {"model.tokenizer", "model", "gap? tokenizer", false},
      {"model.entity", "model", "gap? ENTITY serialization", false},
      {"model.hf", "model", "gap? HuggingFace load", false},
      {"model.transformer", "model", "gap? transformer helpers", false},
      {"model.sampling", "model", "gap? sampling", false},
      {"lucy.lpd", "lucy", "gap? Lucy/LPD/Lean", false},

      // --- Vendor accel load layer ---
      {"accel.nvidia", "backends", "gap? NVIDIA CUDA load + kernels", false},
      {"accel.amd", "backends", "gap? AMD ROCm/HIP load + kernels", false},
      {"accel.intel_gpu", "backends", "gap? Intel GPU load + kernels", false},
      {"accel.intel_npu", "backends", "gap? Intel NPU load + kernels", false},
      {"accel.apple_metal", "backends", "gap? Apple Metal load + kernels", false},
      {"accel.apple_ane", "backends", "gap? Apple Neural Engine load", false},
      {"accel.qcom_gpu", "backends", "gap? Qualcomm GPU load + kernels", false},
      {"accel.qcom_npu", "backends", "gap? Qualcomm QNN NPU load", false},
      {"accel.google_tpu", "backends", "gap? Google TPU load + kernels", false},
      {"fusedgpu", "fusedgpu", "gap? fused device execution paths", false},

      // --- Platforms / delivery ---
      {"platform.linux", "platform", "Linux host build (this tree)", true},
      {"platform.macos", "platform", "gap? validated macOS artifact", false},
      {"platform.windows", "platform", "gap? validated Windows artifact", false},
      {"platform.android", "platform", "gap? NDK shared lib + JNI/CABI", false},
      {"platform.ios", "platform", "gap? iOS static/framework artifact", false},
      {"nac.go_delegate", "interop", "gap? Go Welvet NAC dispatcher", false},
      {"bindings.go", "interop", "gap? bindings/go wrapper package", false},
  };
  return k;
}

} // namespace

const std::vector<Entry>& all() { return build(); }

std::size_t count_gaps() {
  std::size_t n = 0;
  for (const auto& e : all()) {
    if (!e.implemented) {
      ++n;
    }
  }
  return n;
}

std::size_t count_done() {
  std::size_t n = 0;
  for (const auto& e : all()) {
    if (e.implemented) {
      ++n;
    }
  }
  return n;
}

void print_census(std::ostream& os) {
  os << "welvet.cpp gap census: done=" << count_done() << " gaps=" << count_gaps() << "\n";
  for (const auto& e : all()) {
    os << (e.implemented ? "  [done] " : "  [gap?] ") << e.id << " — " << e.detail << "\n";
  }
}

std::string mark(std::string_view what) {
  return std::string("gap? ") + std::string(what);
}

} // namespace welvet::gaps
