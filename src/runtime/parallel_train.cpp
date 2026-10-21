#include "welvet/runtime/parallel_train.hpp"

#include "welvet/runtime/training.hpp"
#include "welvet/systems/tween.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace welvet::runtime::parallel_train {
namespace {

struct DenseLeaf {
  dense::Layer* op = nullptr;
  TensorF32 input;
  TensorF32 pre;
  TensorF32 post;
};

void scale_gap(TensorF32* g, double scale) {
  if (!g) return;
  for (float& v : g->data) {
    v = static_cast<float>(static_cast<double>(v) * scale);
  }
}

Status maybe_train_gate(parallel::Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                        TrainMode parent_mode, double lr) {
  if (!layer.gate) return Status::Ok();
  const TrainMode resolved = resolve_train_mode(parent_mode, TrainMode::NormalBP);
  if (train_mode_family(resolved) == TrainFamily::Tween) return Status::Ok();
  TensorF32 gate_pre, gate_post, gate_gy;
  if (auto st = dense::forward(*layer.gate, input, &gate_pre, &gate_post); !st) return Status::Ok();
  if (auto st = parallel::approx_gate_grad(layer, grad_out, &gate_gy); !st) return st;
  TensorF32 gin, dWg;
  if (auto st = dense::backward_f32(*layer.gate, gate_gy, input, gate_pre, &gin, &dWg); !st) {
    return Status::Ok();
  }
  if (lr != 0.0 && !dWg.data.empty()) {
    return dense::apply_grad_sgd(&(*layer.gate), dWg, lr);
  }
  return Status::Ok();
}

Status apply_dense_projected(dense::Layer& d, const TensorF32& gy, const TensorF32& input,
                             const TensorF32& pre, const TensorF32& post, double soft_lr) {
  if (gy.data.empty() || input.data.empty()) return Status::Ok();
  TensorF32 proj = systems::tween::project_gap(gy, post.shape);
  if (proj.data.empty()) return Status::Ok();
  TensorF32 gin, dW;
  if (auto st = dense::backward_f32(d, proj, input, pre, &gin, &dW); !st) return st;
  if (soft_lr != 0.0 && !dW.data.empty()) {
    return dense::apply_grad_sgd(&d, dW, soft_lr);
  }
  return Status::Ok();
}

Status apply_dense_dw_only(dense::Layer& d, const TensorF32& gy, const TensorF32& input,
                           const TensorF32& pre, double lr) {
  if (gy.data.empty() || input.data.empty()) return Status::Ok();
  TensorF32 dW;
  if (auto st = dense::grad_w_only(d, gy, input, pre, &dW); !st) return st;
  if (lr != 0.0 && !dW.data.empty()) {
    return dense::apply_grad_sgd(&d, dW, lr);
  }
  return Status::Ok();
}

Status apply_dense_full_sgd(dense::Layer& d, const TensorF32& gy, const TensorF32& input,
                            const TensorF32& pre, double lr, TensorF32* gx_out) {
  TensorF32 gin, dW;
  if (auto st = dense::backward_f32(d, gy, input, pre, &gin, &dW); !st) return st;
  if (gx_out) *gx_out = std::move(gin);
  if (lr != 0.0 && !dW.data.empty()) {
    return dense::apply_grad_sgd(&d, dW, lr);
  }
  return Status::Ok();
}

Status collect_dense_leaves(parallel::Layer& layer, const TensorF32& input,
                            std::vector<DenseLeaf>* leaves) {
  if (!leaves) {
    return Status::Err(StatusCode::InvalidArgument, "parallel_train: nil leaves");
  }
  leaves->clear();
  leaves->reserve(layer.branches.size());
  for (auto& br : layer.branches) {
    if (br.kind != parallel::BranchKind::Dense) {
      return Status::Err(StatusCode::InvalidArgument,
                         "parallel_train: Tween/Split leaves require Dense branches");
    }
    DenseLeaf leaf;
    leaf.op = &br.dense_op;
    leaf.input = input;
    if (auto st = dense::forward(br.dense_op, input, &leaf.pre, &leaf.post); !st) {
      return Status::Err(st.code, std::string("parallel_train collect fwd: ") + st.message);
    }
    leaves->push_back(std::move(leaf));
  }
  return Status::Ok();
}

Status apply_split_even(std::vector<DenseLeaf>& leaves, const TensorF32& grad_out, double lr) {
  const int n = static_cast<int>(leaves.size());
  if (n == 0) return Status::Ok();
  const double scale = 1.0 / static_cast<double>(n);
  for (int i = 0; i < n; ++i) {
    auto& leaf = leaves[static_cast<std::size_t>(i)];
    if (!leaf.op || leaf.post.data.empty()) continue;
    TensorF32 gi = systems::tween::project_gap(grad_out, leaf.post.shape);
    scale_gap(&gi, scale);
    if (auto st = apply_dense_dw_only(*leaf.op, gi, leaf.input, leaf.pre, lr); !st) {
      return Status::Err(st.code, std::string("parallel_train split leaf: ") + st.message);
    }
  }
  return Status::Ok();
}

Status apply_split_head_proxy(std::vector<DenseLeaf>& leaves, const TensorF32& grad_out,
                              double lr) {
  const int n = static_cast<int>(leaves.size());
  if (n == 0) return Status::Ok();
  auto& head = leaves[static_cast<std::size_t>(n - 1)];
  if (!head.op || head.post.data.empty()) {
    return apply_split_even(leaves, grad_out, lr);
  }
  TensorF32 g_head = systems::tween::project_gap(grad_out, head.post.shape);
  TensorF32 gx;
  if (auto st = apply_dense_full_sgd(*head.op, g_head, head.input, head.pre, lr, &gx); !st) {
    return st;
  }
  if (n == 1) return Status::Ok();
  const TensorF32& src = gx.data.empty() ? grad_out : gx;
  const double scale = 1.0 / static_cast<double>(n - 1);
  for (int i = 0; i < n - 1; ++i) {
    auto& leaf = leaves[static_cast<std::size_t>(i)];
    if (!leaf.op || leaf.post.data.empty()) continue;
    TensorF32 gi = systems::tween::project_gap(src, leaf.post.shape);
    scale_gap(&gi, scale);
    if (auto st = apply_dense_full_sgd(*leaf.op, gi, leaf.input, leaf.pre, lr, nullptr); !st) {
      return st;
    }
  }
  return Status::Ok();
}

Status apply_split_linear(std::vector<DenseLeaf>& leaves, const TensorF32& grad_out, double lr) {
  const int n = static_cast<int>(leaves.size());
  if (n == 0) return Status::Ok();
  TensorF32 gy = grad_out;
  const double scale = 1.0 / static_cast<double>(n);
  for (int i = n - 1; i >= 0; --i) {
    auto& leaf = leaves[static_cast<std::size_t>(i)];
    if (leaf.op && !leaf.post.data.empty()) {
      TensorF32 gi = systems::tween::project_gap(gy, leaf.post.shape);
      scale_gap(&gi, scale);
      if (auto st = apply_dense_dw_only(*leaf.op, gi, leaf.input, leaf.pre, lr); !st) return st;
    }
    TensorF32 gx;
    if (leaf.op) {
      if (auto st = dense::linear_grad_in(*leaf.op, gy, leaf.input, &gx); !st) return st;
    }
    if (!gx.data.empty()) {
      gy = std::move(gx);
    } else if (!leaf.input.data.empty()) {
      gy = systems::tween::project_gap(gy, leaf.input.shape);
    }
  }
  return Status::Ok();
}

Status apply_split_fast_proxy(std::vector<DenseLeaf>& leaves, const TensorF32& grad_out,
                              double lr) {
  const int n = static_cast<int>(leaves.size());
  if (n == 0) return Status::Ok();
  auto& head = leaves[static_cast<std::size_t>(n - 1)];
  if (!head.op || head.post.data.empty()) {
    return apply_split_even(leaves, grad_out, lr);
  }
  TensorF32 proxy;
  if (auto st = dense::linear_grad_in(*head.op, grad_out, head.input, &proxy); !st) return st;
  TensorF32 g_head = systems::tween::project_gap(grad_out, head.post.shape);
  if (auto st = apply_dense_dw_only(*head.op, g_head, head.input, head.pre, lr); !st) return st;
  if (n == 1) return Status::Ok();
  const TensorF32& src = proxy.data.empty() ? grad_out : proxy;
  const double scale = 1.0 / static_cast<double>(n - 1);
  for (int i = 0; i < n - 1; ++i) {
    auto& leaf = leaves[static_cast<std::size_t>(i)];
    if (!leaf.op || leaf.post.data.empty()) continue;
    TensorF32 gi = systems::tween::project_gap(src, leaf.post.shape);
    scale_gap(&gi, scale);
    if (auto st = apply_dense_dw_only(*leaf.op, gi, leaf.input, leaf.pre, lr); !st) return st;
  }
  return Status::Ok();
}

Status apply_split_sparse(std::vector<DenseLeaf>& leaves, const TensorF32& grad_out, double lr) {
  // Go Sparse with accel=nil: head GradWOnly; one hidden at idx 0 (no 1/(N-1)).
  const int n = static_cast<int>(leaves.size());
  if (n == 0) return Status::Ok();
  auto& head = leaves[static_cast<std::size_t>(n - 1)];
  if (!head.op || head.post.data.empty()) {
    return apply_split_even(leaves, grad_out, lr);
  }
  TensorF32 proxy;
  if (auto st = dense::linear_grad_in(*head.op, grad_out, head.input, &proxy); !st) return st;
  TensorF32 g_head = systems::tween::project_gap(grad_out, head.post.shape);
  if (auto st = apply_dense_dw_only(*head.op, g_head, head.input, head.pre, lr); !st) return st;
  if (n == 1) return Status::Ok();
  auto& hid = leaves[0];
  if (!hid.op || hid.post.data.empty()) return Status::Ok();
  const TensorF32& src = proxy.data.empty() ? grad_out : proxy;
  TensorF32 gi = systems::tween::project_gap(src, hid.post.shape);
  return apply_dense_dw_only(*hid.op, gi, hid.input, hid.pre, lr);
}

Status apply_projected_leaves(std::vector<DenseLeaf>& leaves, const TensorF32& grad_out,
                              double soft_lr) {
  for (auto& leaf : leaves) {
    if (!leaf.op || leaf.post.data.empty()) continue;
    if (auto st = apply_dense_projected(*leaf.op, grad_out, leaf.input, leaf.pre, leaf.post,
                                        soft_lr);
        !st) {
      return st;
    }
  }
  return Status::Ok();
}

Status train_split_leaves(std::vector<DenseLeaf>& leaves, const TensorF32& gy, double lr,
                          TrainMode mode) {
  const TrainMode m = resolve_train_mode(mode, TrainMode::NormalBP);
  switch (m) {
  case TrainMode::TweenSplitHeadProxy:
  case TrainMode::StepTweenSplitHeadProxy:
    return apply_split_head_proxy(leaves, gy, lr);
  case TrainMode::TweenSplitLinear:
  case TrainMode::StepTweenSplitLinear:
  case TrainMode::TweenSplitLinearCache:
  case TrainMode::StepTweenSplitLinearCache:
    // accel=nil ⇒ always refresh linear walk (Go Dense hops).
    return apply_split_linear(leaves, gy, lr);
  case TrainMode::TweenSplitFastProxy:
  case TrainMode::MeshTweenSplitFastProxy:
  case TrainMode::StepTweenSplitFastProxy:
    return apply_split_fast_proxy(leaves, gy, lr);
  case TrainMode::TweenSplitHeadProxyAsync:
  case TrainMode::StepTweenSplitHeadProxyAsync:
    // accel=nil ⇒ no stale proxy; fresh FastProxy-shaped path.
    return apply_split_fast_proxy(leaves, gy, lr);
  case TrainMode::TweenSplitSparse:
  case TrainMode::MeshTweenSplitSparse:
  case TrainMode::StepTweenSplitSparse:
    return apply_split_sparse(leaves, gy, lr);
  default:
    return apply_split_even(leaves, gy, lr);
  }
}

Status combine_leaf_posts(const parallel::Layer& layer, const std::vector<DenseLeaf>& leaves,
                          TensorF32* post) {
  if (!post || leaves.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "parallel_train: combine empty");
  }
  *post = leaves[0].post;
  if (leaves.size() == 1) return Status::Ok();
  switch (layer.cfg.combine) {
  case parallel::Combine::Add:
    for (std::size_t i = 1; i < leaves.size(); ++i) {
      if (leaves[i].post.data.size() != post->data.size()) {
        return Status::Err(StatusCode::InvalidArgument, "parallel_train: combine shape");
      }
      for (std::size_t j = 0; j < post->data.size(); ++j) {
        post->data[j] += leaves[i].post.data[j];
      }
    }
    return Status::Ok();
  case parallel::Combine::Avg: {
    for (std::size_t i = 1; i < leaves.size(); ++i) {
      if (leaves[i].post.data.size() != post->data.size()) {
        return Status::Err(StatusCode::InvalidArgument, "parallel_train: combine shape");
      }
      for (std::size_t j = 0; j < post->data.size(); ++j) {
        post->data[j] += leaves[i].post.data[j];
      }
    }
    const float inv = static_cast<float>(1.0 / static_cast<double>(leaves.size()));
    for (float& v : post->data) v *= inv;
    return Status::Ok();
  }
  default:
    return Status::Err(StatusCode::InvalidArgument,
                       "parallel_train: split combine only Add/Avg");
  }
}

Status train_split_family_mse(parallel::Layer& layer, const TensorF32& input,
                              const TensorF32& target, TrainMode mode, double lr, double* loss) {
  // Go OpenSplitTape: one collect walk builds leaves + combined post, then TrainGap.
  std::vector<DenseLeaf> leaves;
  if (auto st = collect_dense_leaves(layer, input, &leaves); !st) return st;
  TensorF32 post;
  if (auto st = combine_leaf_posts(layer, leaves, &post); !st) return st;
  if (auto st = training::mse(post, target, loss); !st) return st;
  TensorF32 gy;
  if (auto st = training::mse_grad(post, target, &gy); !st) return st;
  if (layer.cam_kit) layer.cam_kit->note_loss(*loss);
  return train_split_leaves(leaves, gy, lr, mode);
}

Status train_alt_layer_mse(parallel::Layer& layer, const TensorF32& input, const TensorF32& target,
                           double lr, double* loss) {
  // Go trainTweenAltLayerMSE: AltTimes=1 default — Split @ lr then Tween @ lr×0.5 with
  // recomputed MSE gaps.
  const int times = 1;
  double last = 0.0;
  for (int i = 0; i < times; ++i) {
    std::vector<DenseLeaf> leaves;
    if (auto st = collect_dense_leaves(layer, input, &leaves); !st) return st;
    TensorF32 post;
    if (auto st = combine_leaf_posts(layer, leaves, &post); !st) return st;
    if (auto st = training::mse(post, target, &last); !st) return st;
    TensorF32 gy;
    if (auto st = training::mse_grad(post, target, &gy); !st) return st;
    if (layer.cam_kit) layer.cam_kit->note_loss(last);
    if (auto st = apply_split_even(leaves, gy, lr); !st) return st;

    leaves.clear();
    if (auto st = collect_dense_leaves(layer, input, &leaves); !st) return st;
    if (auto st = combine_leaf_posts(layer, leaves, &post); !st) return st;
    if (auto st = training::mse(post, target, &last); !st) return st;
    if (auto st = training::mse_grad(post, target, &gy); !st) return st;
    if (layer.cam_kit) layer.cam_kit->note_loss(last);
    if (auto st = apply_projected_leaves(leaves, gy, lr * 0.5); !st) return st;
  }
  *loss = last;
  return Status::Ok();
}

Status apply_branch_projected(parallel::Branch& br, const TensorF32& gy, const TensorF32& input,
                              const TensorF32& pre, const TensorF32& post, double soft_lr) {
  if (br.kind == parallel::BranchKind::Dense) {
    return apply_dense_projected(br.dense_op, gy, input, pre, post, soft_lr);
  }
  // Non-Dense: full BP then soft SGD (Tween poly).
  if (gy.data.empty() || input.data.empty()) return Status::Ok();
  TensorF32 proj = systems::tween::project_gap(gy, post.shape);
  if (proj.data.empty()) return Status::Ok();
  TensorF32 gin, dW;
  if (auto st = parallel::branch_backward(br, proj, input, nullptr, pre, &gin, &dW); !st) {
    return st;
  }
  if (soft_lr != 0.0 && !dW.data.empty()) {
    return parallel::branch_apply_grad_sgd(&br, dW, soft_lr);
  }
  return Status::Ok();
}

Status train_op_branch(parallel::Branch& br, const TensorF32& gy, const TensorF32& input,
                       const TensorF32& pre, const TensorF32& post, TrainMode mode, double lr) {
  const TrainMode m = resolve_train_mode(mode, TrainMode::NormalBP);
  if (train_mode_is_frozen(m)) return Status::Ok();
  if (br.kind == parallel::BranchKind::Dense) {
    dense::Layer& d = br.dense_op;
    switch (train_mode_family(m)) {
    case TrainFamily::Tween:
      return apply_dense_projected(d, gy, input, pre, post, lr * 0.5);
    case TrainFamily::TweenSplit: {
      DenseLeaf leaf{&d, input, pre, post};
      std::vector<DenseLeaf> leaves{std::move(leaf)};
      return train_split_leaves(leaves, gy, lr, m);
    }
    case TrainFamily::TweenAlt: {
      DenseLeaf leaf{&d, input, pre, post};
      std::vector<DenseLeaf> leaves{leaf};
      if (auto st = apply_split_even(leaves, gy, lr); !st) return st;
      TensorF32 p2, o2;
      if (auto st = dense::forward(d, input, &p2, &o2); !st) return st;
      return apply_dense_projected(d, gy, input, p2, o2, lr * 0.5);
    }
    case TrainFamily::BP:
    case TrainFamily::TweenChain:
    default: {
      TensorF32 gin, dW;
      if (auto st = dense::backward_f32(d, gy, input, pre, &gin, &dW); !st) return st;
      return dense::apply_grad_sgd(&d, dW, lr);
    }
    }
  }
  // Poly non-Dense: FamilyBP / TweenChain / Tween projected; Split/Alt Dense-only.
  switch (train_mode_family(m)) {
  case TrainFamily::TweenSplit:
  case TrainFamily::TweenAlt:
    return Status::Err(StatusCode::InvalidArgument,
                       "parallel_train: TweenSplit/Alt require Dense branches");
  case TrainFamily::Tween:
    return apply_branch_projected(br, gy, input, pre, post, lr * 0.5);
  case TrainFamily::BP:
  case TrainFamily::TweenChain:
  default: {
    TensorF32 gin, dW;
    if (auto st = parallel::branch_backward(br, gy, input, nullptr, pre, &gin, &dW); !st) {
      return st;
    }
    return parallel::branch_apply_grad_sgd(&br, dW, lr);
  }
  }
}

Status train_parallel_mixed(parallel::Layer& layer, const TensorF32& grad_out,
                            const TensorF32& input, TrainMode parent_mode, double lr) {
  const int nb = static_cast<int>(layer.branches.size());
  if (nb == 0) return Status::Ok();

  std::vector<TensorF32> branch_pres(static_cast<std::size_t>(nb));
  std::vector<TensorF32> branch_outs(static_cast<std::size_t>(nb));
  for (int i = 0; i < nb; ++i) {
    if (auto st = parallel::branch_forward(layer.branches[static_cast<std::size_t>(i)], input,
                                           nullptr, &branch_pres[static_cast<std::size_t>(i)],
                                           &branch_outs[static_cast<std::size_t>(i)]);
        !st) {
      return Status::Err(st.code, std::string("parallel_train branch fwd: ") + st.message);
    }
  }

  std::vector<TensorF32> branch_grads;
  if (auto st = parallel::split_combine_grad(layer, grad_out, branch_outs, &branch_grads); !st) {
    return st;
  }

  // Shadow KD
  const TensorF32* teacher = nullptr;
  for (int i = 0; i < nb; ++i) {
    if (train_mode_is_shadow(parallel::effective_branch_mode(layer, i, parent_mode))) {
      teacher = &branch_outs[static_cast<std::size_t>(i)];
      break;
    }
  }
  if (teacher) {
    double coef = 1.0;
    if (layer.cam_kit) coef = layer.cam_kit->effective_shadow_coef();
    double n = static_cast<double>(teacher->data.size());
    if (n < 1.0) n = 1.0;
    for (int i = 0; i < nb; ++i) {
      const TrainMode bm = parallel::effective_branch_mode(layer, i, parent_mode);
      if (train_mode_is_frozen(bm)) continue;
      auto& gy = branch_grads[static_cast<std::size_t>(i)];
      const auto& post = branch_outs[static_cast<std::size_t>(i)];
      if (gy.data.size() != teacher->data.size() || post.data.size() != teacher->data.size()) {
        continue;
      }
      for (std::size_t j = 0; j < gy.data.size(); ++j) {
        const double diff = static_cast<double>(post.data[j]) - static_cast<double>(teacher->data[j]);
        gy.data[j] += static_cast<float>(coef * 2.0 * diff / n);
      }
    }
  }

  const bool mem_ok = !layer.cam_kit || layer.cam_kit->memory_allows_train();
  for (int i = 0; i < nb; ++i) {
    const TrainMode bm = parallel::effective_branch_mode(layer, i, parent_mode);
    if (train_mode_is_frozen(bm)) continue;
    if (bm == TrainMode::Memory && !mem_ok) continue;
    double blr = parallel::effective_branch_lr(layer, i, lr);
    if (blr == 0.0) continue;
    if (bm == TrainMode::Adversarial) blr = -blr;

    TrainMode update_mode = bm;
    if (bm == TrainMode::Adversarial || bm == TrainMode::Memory) {
      update_mode = resolve_train_mode(parent_mode, TrainMode::NormalBP);
    }

    if (auto st = train_op_branch(layer.branches[static_cast<std::size_t>(i)],
                                  branch_grads[static_cast<std::size_t>(i)], input,
                                  branch_pres[static_cast<std::size_t>(i)],
                                  branch_outs[static_cast<std::size_t>(i)], update_mode, blr);
        !st) {
      return Status::Err(st.code, std::string("parallel_train branch: ") + st.message);
    }
  }
  return maybe_train_gate(layer, grad_out, input, parent_mode, lr);
}

// Go Train unified BP/TweenChain: Backward + ApplyGradSGD (full combine grads + Filter gate).
Status train_unified_bp(parallel::Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                        const TensorF32& pre, double lr) {
  TensorF32 gin, dW;
  if (auto st = parallel::backward_f32(layer, grad_out, input, pre, &gin, &dW); !st) return st;
  return parallel::apply_grad_sgd(&layer, dW, lr);
}

Status train_layer(parallel::Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                   const TensorF32& pre, TrainMode parent_mode, double lr) {
  const TrainMode mode = resolve_train_mode(parent_mode, TrainMode::NormalBP);
  if (train_mode_is_frozen(mode)) return Status::Ok();
  const TrainFamily fam = train_mode_family(mode);
  if (fam == TrainFamily::TweenSplit) {
    std::vector<DenseLeaf> leaves;
    if (auto st = collect_dense_leaves(layer, input, &leaves); !st) return st;
    return train_split_leaves(leaves, grad_out, lr, mode);
  }
  if (fam == TrainFamily::TweenAlt) {
    std::vector<DenseLeaf> leaves;
    if (auto st = collect_dense_leaves(layer, input, &leaves); !st) return st;
    if (auto st = apply_split_even(leaves, grad_out, lr); !st) return st;
    leaves.clear();
    if (auto st = collect_dense_leaves(layer, input, &leaves); !st) return st;
    return apply_projected_leaves(leaves, grad_out, lr * 0.5);
  }
  if (!parallel::has_mixed_branch_modes(layer) &&
      (fam == TrainFamily::BP || fam == TrainFamily::TweenChain)) {
    return train_unified_bp(layer, grad_out, input, pre, lr);
  }
  return train_parallel_mixed(layer, grad_out, input, mode, lr);
}

} // namespace

Status train_mse(parallel::Layer& layer, const TensorF32& input, const TensorF32& target,
                 TrainMode parent_mode, double lr, double* loss) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "parallel_train: nil loss");
  }
  const TrainMode mode = resolve_train_mode(parent_mode, TrainMode::NormalBP);
  const TrainFamily fam = train_mode_family(mode);

  // Go TrainMSE: Alt / Split-family open their own tapes before unified Forward→Train.
  if (fam == TrainFamily::TweenAlt) {
    return train_alt_layer_mse(layer, input, target, lr, loss);
  }
  if (fam == TrainFamily::TweenSplit) {
    return train_split_family_mse(layer, input, target, mode, lr, loss);
  }

  TensorF32 pre, post;
  if (auto st = parallel::forward(layer, input, &pre, &post); !st) {
    return Status::Err(st.code, std::string("parallel_train fwd: ") + st.message);
  }
  TensorF32 gy;
  if (auto st = training::mse(post, target, loss); !st) return st;
  if (auto st = training::mse_grad(post, target, &gy); !st) return st;
  if (layer.cam_kit) layer.cam_kit->note_loss(*loss);

  if (train_mode_is_frozen(mode)) {
    return Status::Ok();
  }
  return train_layer(layer, gy, input, pre, mode, lr);
}

} // namespace welvet::runtime::parallel_train
