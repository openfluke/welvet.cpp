#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <vector>

namespace welvet::runtime::backward {

struct GradW {
  architecture::Coord coord;
  TensorF32 dW;
};

struct Result {
  TensorF32 grad_in;
  std::vector<GradW> grad_ws;
};

// Reverse hop over a Dense-only forward tape. Remote cells keep local gradIn
// (not re-routed to the remote source — Go v0 parity).
Status volume_dense(architecture::Grid& g, const forward::Result& fwd, const TensorF32& grad_out,
                    Result* out);

} // namespace welvet::runtime::backward
