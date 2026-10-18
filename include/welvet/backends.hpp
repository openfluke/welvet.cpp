#pragma once

#include <string_view>
#include <vector>

#include "welvet/status.hpp"

namespace welvet::backends {

enum class Target {
  CpuNative,
  NvidiaCuda,
  AmdRocm,
  IntelGpu,
  IntelNpu,
  AppleMetal,
  AppleNeuralEngine,
  QualcommGpu,
  QualcommNpu,
  GoogleTpu,
};

struct Info {
  Target target;
  const char* name;
  bool available; // runtime probe; placeholders report false
  const char* note; // often "gap? …"
};

std::vector<Info> list();
Status load(Target t);   // gap? for every non-CPU target until wired
Status unload(Target t); // gap?

std::string_view target_name(Target t);

} // namespace welvet::backends
