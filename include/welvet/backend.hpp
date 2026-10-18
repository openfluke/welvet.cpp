#pragma once

#include <cstdint>

namespace welvet {

// Matches welvet/core Backend subset used by Dense.
enum class Backend : int32_t {
  CPUTiled = 0,
  SIMD = 1,
  // gap? WebGPU / device backends
};

} // namespace welvet
