#pragma once

#include "welvet/layer.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/weights.hpp"

namespace welvet::layernorm {

struct Config {
  int dim = 0;
  double eps = 0.0; // <=0 → 1e-5 (matches Go Validate / loom default)
};

Status validate(Config* cfg);

struct Layer {
  Config cfg{};
  WeightStore gamma; // 1×Dim
  WeightStore beta;  // 1×Dim
  Backend backend = Backend::CPUTiled;
  LayerMeta meta{};
};

Status new_configured(const Config& cfg, DType dt, const float* gamma_init, const float* beta_init,
                      Layer* out);

// Forward: pre = x̂ = (x−μ)/σ; post = γ⊙x̂ + β.
// input shape [batch,dim] or [batch,seq,dim].
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

} // namespace welvet::layernorm
