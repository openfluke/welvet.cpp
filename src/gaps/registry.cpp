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
      {"backends.cpu", "backends", "Native CPU target available", true},
      {"core.layer_types", "core", "LayerType 0–22 + LayerMeta descriptor", true},
      {"arch.volume", "architecture", "Grid/Cell/Coord/HopOrder/remote links", true},
      {"runtime.forward.dense", "runtime", "volume Forward Dense-only hop walker", true},

      // --- Core / dtypes ---
      {"core.numeric.acts", "core", "gap? all 15 Go Numeric activation host types", false},
      {"core.tensor.generic", "core", "gap? Tensor[T] generic host/device buffers", false},

      // --- Weights / quant ---
      {"weights.quant.all_formats", "weights", "gap? all quant.Format MatVec/MatVecT", false},
      {"weights.sgd.native", "weights", "gap? ApplySGD storage-preserving updates", false},
      {"weights.matvect.f32", "weights", "gap? MatVecT dX path", false},
      {"weights.bias.train", "weights", "gap? bias gradient + update", false},

      // --- Dense remaining ---
      {"dense.backward", "layers", "gap? Dense dX/dW/db + act deriv", false},
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
      {"layer.swiglu", "layers", "gap? SwiGLU F/B", false},
      {"layer.rmsnorm", "layers", "gap? RMSNorm F/B", false},
      {"layer.layernorm", "layers", "gap? LayerNorm F/B", false},
      {"layer.cnn1", "layers", "gap? CNN1 F/B", false},
      {"layer.cnn2", "layers", "gap? CNN2 F/B", false},
      {"layer.cnn3", "layers", "gap? CNN3 F/B", false},
      {"layer.convt1", "layers", "gap? ConvTransposed1D F/B", false},
      {"layer.convt2", "layers", "gap? ConvTransposed2D F/B", false},
      {"layer.convt3", "layers", "gap? ConvTransposed3D F/B", false},
      {"layer.rnn", "layers", "gap? RNN F/B", false},
      {"layer.lstm", "layers", "gap? LSTM F/B", false},
      {"layer.embedding", "layers", "gap? Embedding gather/scatter", false},
      {"layer.softmax", "layers", "gap? Softmax F/B", false},
      {"layer.kmeans", "layers", "gap? KMeans soft assignment", false},
      {"layer.parallel", "layers", "gap? Parallel MoE combine", false},
      {"layer.sequential", "layers", "gap? Sequential compose", false},
      {"layer.residual", "layers", "gap? Residual F(x)+x", false},
      {"layer.metacognition", "layers", "gap? Metacognition", false},
      {"layer.mamba", "layers", "gap? Mamba SSM", false},
      {"layer.gdn", "layers", "gap? Gated DeltaNet", false},
      {"layer.stack", "layers", "gap? Stack nested ops", false},
      {"layer.seqmix", "layers", "gap? seqmix helper", false},

      // --- Architecture / runtime remaining ---
      {"runtime.forward", "runtime", "gap? volume Forward all layer kinds (Dense done)", false},
      {"runtime.backward", "runtime", "gap? volume Backward dispatcher", false},
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
