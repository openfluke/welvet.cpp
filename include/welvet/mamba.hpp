#pragma once

#include "welvet/backend.hpp"
#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/layer.hpp"
#include "welvet/seqmix.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <vector>

namespace welvet::mamba {

struct Config {
  int d_model = 0;
  int d_state = 0; // validated; unused in minimal forward
  int expand = 2;  // 0 → 2; inner = expand * d_model
  int seq_len = 0; // expected T
};

int inner_dim(const Config& cfg);
int weight_count(const Config& cfg); // InProj + OutProj + ALog + DSkip
Status validate(Config* cfg);

struct Layer {
  Config cfg{};
  dense::Layer in_proj;
  dense::Layer out_proj;
  std::vector<float> a_log;  // [Inner], always f32
  std::vector<float> d_skip; // [Inner], always f32
  Backend backend = Backend::CPUTiled;
  LayerMeta meta{};

  static constexpr seqmix::Kind kind() { return seqmix::mamba_kind(); }
};

// packed: InProj[2*Inner×D] | OutProj[D×Inner] | ALog[Inner] | DSkip[Inner] (f32).
// nullptr → zero projs + default ALog/DSkip.
Status new_configured(const Config& cfg, DType dt, const float* packed_or_null, Layer* out);

// Input must be [batch, seq, d_model] with seq == SeqLen. pre/post = OutProj [B,T,D].
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

} // namespace welvet::mamba
