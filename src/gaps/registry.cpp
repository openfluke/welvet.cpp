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
      {"abi.swiglu_forward_f32", "interop", "C ABI welvet_swiglu_forward_f32 + WVS1 tool", true},
      {"abi.rmsnorm_forward_f32", "interop", "C ABI welvet_rmsnorm_forward_f32 + WVR1 tool", true},
      {"abi.rmsnorm_normalbp_f32", "interop", "C ABI welvet_rmsnorm_normalbp_f32 + WRB1/WRV1 tool", true},
      {"abi.layernorm_forward_f32", "interop", "C ABI welvet_layernorm_forward_f32 + WVL1 tool", true},
      {"abi.layernorm_normalbp_f32", "interop", "C ABI welvet_layernorm_normalbp_f32 + WLB1/WLV1 tool", true},
      {"abi.cnn_forward_f32", "interop", "C ABI welvet_cnn{1,2,3}_forward_f32 + WVC1/2/3 tool", true},
      {"abi.convt_forward_f32", "interop", "C ABI welvet_convt{1,2,3}_forward_f32 + WVT1/2/3 tool", true},
      {"abi.rnn_forward_f32", "interop", "C ABI welvet_rnn_forward_f32 + WRN1 tool", true},
      {"abi.lstm_forward_f32", "interop", "C ABI welvet_lstm_forward_f32 + WLS1 tool", true},
      {"abi.embedding_forward_f32", "interop", "C ABI welvet_embedding_forward_f32 + WEM1 tool", true},
      {"abi.softmax_forward_f32", "interop", "C ABI welvet_softmax_forward_f32 + WSM1 tool (all Kind)", true},
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
      {"layer.mha.backward", "layers", "gap? MHA backward + dropout/QKNorm", false},
      {"layer.swiglu", "layers", "SwiGLU forward Gate/Up/Down + SiLU⊙ FormatNone×34", true},
      {"layer.swiglu.backward", "layers", "gap? SwiGLU backward + SiLU⊙ grads", false},
      {"layer.rmsnorm", "layers", "RMSNorm forward x̂ + γ⊙ FormatNone×34", true},
      {"layer.rmsnorm.backward", "layers", "RMSNorm NormalBP dγ/dx + ApplyGradSGD FormatNone×34", true},
      {"layer.layernorm", "layers", "LayerNorm forward x̂ + γ⊙+β FormatNone×34", true},
      {"layer.layernorm.backward", "layers", "LayerNorm NormalBP dγ/dβ/dx + ApplyGradSGD FormatNone×34", true},
      {"layer.cnn1", "layers", "CNN1 im2col+Dense forward FormatNone×34", true},
      {"layer.cnn2", "layers", "CNN2 im2col+Dense forward FormatNone×34", true},
      {"layer.cnn3", "layers", "CNN3 im2col+Dense forward FormatNone×34", true},
      {"layer.cnn.backward", "layers", "gap? CNN1/2/3 backward + col2im", false},
      {"layer.convt1", "layers", "ConvT1 scatter forward FormatNone×34", true},
      {"layer.convt2", "layers", "ConvT2 scatter forward FormatNone×34", true},
      {"layer.convt3", "layers", "ConvT3 scatter forward FormatNone×34", true},
      {"layer.convt.backward", "layers", "gap? ConvT1/2/3 backward", false},
      {"layer.rnn", "layers", "RNN forward IH/HH + tanh FormatNone×34", true},
      {"layer.rnn.backward", "layers", "gap? RNN BPTT", false},
      {"layer.lstm", "layers", "LSTM forward gates i/f/g/o FormatNone×34", true},
      {"layer.lstm.backward", "layers", "gap? LSTM BPTT", false},
      {"layer.embedding", "layers", "Embedding gather forward FormatNone×34", true},
      {"layer.embedding.backward", "layers", "gap? Embedding scatter dW", false},
      {"layer.softmax", "layers",
       "Softmax forward all Kind (std/temp/gumbel/masked/sparse/entmax/grid/hier)", true},
      {"layer.softmax.backward", "layers", "gap? Softmax Jacobian backward", false},
      {"layer.kmeans", "layers",
       "KMeans soft assignment forward (probs|features) FormatNone×34", true},
      {"layer.kmeans.backward", "layers", "gap? KMeans soft assignment backward", false},
      {"layer.parallel", "layers",
       "Parallel Dense branches + 7 combine modes FormatNone×34", true},
      {"layer.parallel.poly", "layers",
       "gap? Parallel NewFromBranches / Nested / ResidualGraft / cameral", false},
      {"layer.parallel.backward", "layers", "gap? Parallel combine backward + train modes", false},
      {"layer.sequential", "layers",
       "Sequential Dense Dim→Dim chain forward FormatNone×34", true},
      {"layer.sequential.ops", "layers",
       "gap? Sequential NewFromOps (SwiGLU/RMSNorm/LayerNorm mix)", false},
      {"layer.sequential.backward", "layers", "gap? Sequential reverse chain backward", false},
      {"layer.residual", "layers",
       "Residual Dense F(x)+x forward FormatNone×34", true},
      {"layer.residual.ops", "layers",
       "gap? Residual NewFromOps (SwiGLU/RMSNorm/LayerNorm mix)", false},
      {"layer.residual.backward", "layers", "gap? Residual F backward + skip grad", false},
      {"layer.metacognition", "layers",
       "Metacognition Observed Dense + stability rules FormatNone×34", true},
      {"layer.metacognition.rules_wire", "layers",
       "gap? Metacognition custom rules on wire / multi-step fireCount", false},
      {"layer.metacognition.backward", "layers", "gap? Metacognition Observed backward", false},
      {"layer.mamba", "layers",
       "Mamba minimal selective SSM forward FormatNone×34", true},
      {"layer.mamba.full", "layers",
       "gap? Mamba-N (DState/B/C/conv1d/SiLU) beyond minimal scan", false},
      {"layer.mamba.backward", "layers", "gap? Mamba scan/projection backward", false},
      {"layer.gdn", "layers",
       "GDN Gated DeltaNet forward Float32 FormatNone (f32-acc GEMV)", true},
      {"layer.gdn.dtype34", "layers", "gap? GDN ×34 dtypes (Go Float32-only today)", false},
      {"layer.gdn.backward", "layers", "gap? GDN truncated-BPTT backward", false},
      {"layer.stack", "layers", "gap? Stack nested ops", false},
      {"layer.seqmix", "layers",
       "seqmix Kind/Contract helper + MHA/Mamba/GDN tags (no Forward)", true},
      {"layer.seqmix.conv_mix", "layers", "gap? seqmix KindConvMix implementor", false},

      // --- Architecture / runtime remaining ---
      {"runtime.forward", "runtime", "gap? volume Forward all layer kinds (Dense done)", false},
      {"runtime.backward", "runtime", "volume Backward Dense+RMSNorm+LayerNorm + training.Step (NormalBP)", true},
      {"runtime.training", "runtime", "gap? training modes / credit", false},
      {"runtime.step_mesh", "runtime", "gap? Step / Mesh scheduling", false},
      {"arch.branches", "architecture", "gap? ParallelBranches/SequentialLayers combine", false},
      {"tiling.cpu", "tiling", "gap? CPU tile partition helpers", false},
      {"simd.x86_arm", "simd", "imported DotTile (x86 AVX2 + arm64 NEON stubs)", true},
      {"train.tuned_outer", "training", "gap? wire imported tuned_outer/tuned_tile/backward_native", false},

      // --- Systems / model ---
      {"systems.dna", "systems", "gap? DNA", false},
      {"systems.evolution", "systems", "gap? evolution", false},
      {"systems.tween", "systems", "gap? Tween", false},
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
