#pragma once

#include "welvet/backend.hpp"
#include "welvet/cam_kit.hpp"
#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/layer.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/train_mode.hpp"

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
  int out_feat = 0; // per-branch; required >0 for new_configured; 0 OK for new_from_branches
  int branches = 0;
  Combine combine = Combine::Concat;
  int seq_len = 0; // 0 → [B,D]; >0 → [B,T,D]
  int sparse_k = 0; // 0 → min(2, branches)
  double disagree_beta = 0; // 0 → 1
};

int out_dim(const Config& cfg);
int weight_count(const Config& cfg); // Dense NewConfigured layout only
int sparse_k_eff(const Config& cfg);
double disagree_beta_eff(const Config& cfg);
Status validate(Config* cfg);

// Polymorphic branch handle — same tagged union as stack::Child (Go branch.go kinds).
// Nested Parallel uses Kind::Parallel (OpKind 23); Stack-as-branch uses Kind::NestedStack.
using Branch = stack::Child;
using BranchKind = stack::Kind;

struct Layer {
  Config cfg{};
  std::vector<Branch> branches;
  std::optional<dense::Layer> gate; // filter only
  Backend backend = Backend::CPUTiled;
  LayerMeta meta{};

  // Cameral: per-hemisphere modes / LR scales (empty ⇒ inherit parent / 1.0).
  std::vector<TrainMode> branch_modes;
  std::vector<double> branch_lrs;
  CamKit* cam_kit = nullptr; // optional; owned externally
};

TrainMode effective_branch_mode(const Layer& l, int i, TrainMode parent);
double effective_branch_lr(const Layer& l, int i, double parent_lr);
bool has_mixed_branch_modes(const Layer& l);

// Dense-only packed init (Branches×[OutFeat×Dim] then optional Gate).
Status new_configured(const Config& cfg, DType dt, const float* packed_or_null, Layer* out);

// Polymorphic branches (Go NewFromBranches). cfg.Branches set from len when 0.
Status new_from_branches(const Config& cfg, std::vector<Branch> branches,
                         std::optional<dense::Layer> gate, Layer* out);

// Dense branch helper (nullptr when branch i is not Dense).
dense::Layer* dense_branch(Layer& l, int i);
const dense::Layer* dense_branch(const Layer& l, int i);

// Per-branch dispatch (Dense receives flat when provided).
Status branch_forward(Branch& br, const TensorF32& input, const TensorF32* flat, TensorF32* pre,
                      TensorF32* post);
Status branch_backward(Branch& br, const TensorF32& grad_out, const TensorF32& input,
                       const TensorF32* flat, const TensorF32& pre, TensorF32* grad_in,
                       TensorF32* grad_w);
Status branch_apply_grad_sgd(Branch* br, const TensorF32& dW, double lr);
int branch_grad_w_size(const Branch& br);
Status branch_export_f32(const Branch& br, float* dst);
bool branch_needs_dense_flat(const Branch& br);

// Input [batch,dim] or [batch,seq,dim] (or poly CNN/… layout). pre = input copy; post = combined.
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

int grad_w_size(const Layer& layer);

// Map combined ∂L/∂y into per-branch grads (Add/Avg/Max/SparseK/Disagree/Concat/Filter≈Add).
Status split_combine_grad(const Layer& layer, const TensorF32& grad_out,
                          const std::vector<TensorF32>& branch_outs,
                          std::vector<TensorF32>* branch_grads);

// Soft placeholder gate logits grad [batch, branches] (Go approxGateGrad).
Status approx_gate_grad(const Layer& layer, const TensorF32& grad_out, TensorF32* gate_gy);

// NormalBP: splitCombineGrad → per-kind branch Backward (+ Filter gate); grad_w = concat dWs.
Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

// Export packed Dense branch weights (NewConfigured layout). Poly: concat branch exports.
Status export_packed_f32(const Layer& layer, float* dst);

} // namespace welvet::parallel
