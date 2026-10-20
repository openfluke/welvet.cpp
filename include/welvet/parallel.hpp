#pragma once

#include "welvet/backend.hpp"
#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/layer.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <optional>
#include <vector>

namespace welvet::parallel {

// Wire / ABI combine codes (match go_v_cpp).
enum class Combine : int32_t {
  Concat = 0,
  Add = 1,
  Avg = 2,
  Filter = 3,
  Max = 4,
  SparseK = 5,
  Disagree = 6,
};

struct Config {
  int dim = 0;
  int out_feat = 0; // per-branch; required >0 for new_configured
  int branches = 0;
  Combine combine = Combine::Concat;
  int seq_len = 0; // 0 → [B,D]; >0 → [B,T,D]
  int sparse_k = 0; // 0 → min(2, branches)
  double disagree_beta = 0; // 0 → 1
};

int out_dim(const Config& cfg);
int weight_count(const Config& cfg); // branches*OutFeat*Dim + (filter ? Branches*Dim : 0)
int sparse_k_eff(const Config& cfg);
double disagree_beta_eff(const Config& cfg);
Status validate(Config* cfg);

struct Layer {
  Config cfg{};
  std::vector<dense::Layer> branches;
  std::optional<dense::Layer> gate; // filter only
  Backend backend = Backend::CPUTiled;
  LayerMeta meta{};
};

// packed: Branches×[OutFeat×Dim] then optional Gate[Branches×Dim] if filter.
Status new_configured(const Config& cfg, DType dt, const float* packed_or_null, Layer* out);

// Input [batch,dim] or [batch,seq,dim]. pre = input copy; post = combined.
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

} // namespace welvet::parallel
