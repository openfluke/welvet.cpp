#include "welvet/backends.hpp"

namespace welvet::backends {

std::string_view target_name(Target t) {
  switch (t) {
  case Target::CpuNative: return "cpu_native";
  case Target::NvidiaCuda: return "nvidia_cuda";
  case Target::AmdRocm: return "amd_rocm";
  case Target::IntelGpu: return "intel_gpu";
  case Target::IntelNpu: return "intel_npu";
  case Target::AppleMetal: return "apple_metal";
  case Target::AppleNeuralEngine: return "apple_ane";
  case Target::QualcommGpu: return "qualcomm_gpu";
  case Target::QualcommNpu: return "qualcomm_npu";
  case Target::GoogleTpu: return "google_tpu";
  }
  return "unknown";
}

// Declared in per-vendor TUs.
Status cpu_load();
Status cpu_unload();
Status nvidia_load();
Status nvidia_unload();
Status amd_load();
Status amd_unload();
Status intel_load();
Status intel_unload();
Status apple_load();
Status apple_unload();
Status qualcomm_load();
Status qualcomm_unload();
Status tpu_load();
Status tpu_unload();

std::vector<Info> list() {
  return {
      {Target::CpuNative, "Native CPU", true, "serial f32 dense forward (M1)"},
      {Target::NvidiaCuda, "NVIDIA CUDA", false, "gap? load CUDA runtime + GEMV kernels"},
      {Target::AmdRocm, "AMD ROCm/HIP", false, "gap? load HIP runtime + GEMV kernels"},
      {Target::IntelGpu, "Intel GPU", false, "gap? oneAPI / Level Zero GPU path"},
      {Target::IntelNpu, "Intel NPU", false, "gap? OpenVINO / NPU compile path"},
      {Target::AppleMetal, "Apple Metal", false, "gap? Metal compute shaders"},
      {Target::AppleNeuralEngine, "Apple Neural Engine", false, "gap? Core ML / ANE path"},
      {Target::QualcommGpu, "Qualcomm GPU", false, "gap? Adreno / OpenCL path"},
      {Target::QualcommNpu, "Qualcomm NPU", false, "gap? QNN HTP path"},
      {Target::GoogleTpu, "Google TPU", false, "gap? TPU runtime / PJRT path"},
  };
}

Status load(Target t) {
  switch (t) {
  case Target::CpuNative: return cpu_load();
  case Target::NvidiaCuda: return nvidia_load();
  case Target::AmdRocm: return amd_load();
  case Target::IntelGpu:
  case Target::IntelNpu: return intel_load();
  case Target::AppleMetal:
  case Target::AppleNeuralEngine: return apple_load();
  case Target::QualcommGpu:
  case Target::QualcommNpu: return qualcomm_load();
  case Target::GoogleTpu: return tpu_load();
  }
  return Status::Gap("backends.load unknown target");
}

Status unload(Target t) {
  switch (t) {
  case Target::CpuNative: return cpu_unload();
  case Target::NvidiaCuda: return nvidia_unload();
  case Target::AmdRocm: return amd_unload();
  case Target::IntelGpu:
  case Target::IntelNpu: return intel_unload();
  case Target::AppleMetal:
  case Target::AppleNeuralEngine: return apple_unload();
  case Target::QualcommGpu:
  case Target::QualcommNpu: return qualcomm_unload();
  case Target::GoogleTpu: return tpu_unload();
  }
  return Status::Gap("backends.unload unknown target");
}

} // namespace welvet::backends
