#include "welvet/runtime/parallel_train.hpp"

#include "welvet/runtime/training.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace welvet::runtime::parallel_train {
namespace {

Status split_combine_grad(const parallel::Layer& layer, const TensorF32& grad_out,
                          const std::vector<TensorF32>& branch_outs,
                          std::vector<TensorF32>* branch_grads) {
  if (!branch_grads) {
    return Status::Err(StatusCode::InvalidArgument, "parallel_train: nil branch_grads");
  }
  const int nb = static_cast<int>(branch_outs.size());
  branch_grads->assign(static_cast<std::size_t>(nb), TensorF32{});
  if (grad_out.data.empty()) {
    for (int i = 0; i < nb; ++i) {
      (*branch_grads)[static_cast<std::size_t>(i)].shape = branch_outs[static_cast<std::size_t>(i)].shape;
      (*branch_grads)[static_cast<std::size_t>(i)].data.assign(
          branch_outs[static_cast<std::size_t>(i)].data.size(), 0.0f);
    }
    return Status::Ok();
  }
  switch (layer.cfg.combine) {
    case parallel::Combine::Add:
      for (int i = 0; i < nb; ++i) {
        (*branch_grads)[static_cast<std::size_t>(i)] = grad_out;
      }
      return Status::Ok();
    case parallel::Combine::Avg: {
      const float inv = static_cast<float>(1.0 / static_cast<double>(nb));
      for (int i = 0; i < nb; ++i) {
        TensorF32 g = grad_out;
        for (float& v : g.data) v *= inv;
        (*branch_grads)[static_cast<std::size_t>(i)] = std::move(g);
      }
      return Status::Ok();
    }
    default:
      return Status::Err(StatusCode::InvalidArgument,
                         "parallel_train: splitCombineGrad only Add/Avg in this ship");
  }
}

Status train_parallel_mixed(parallel::Layer& layer, const TensorF32& grad_out,
                            const TensorF32& input, TrainMode parent_mode, double lr) {
  const int nb = static_cast<int>(layer.branches.size());
  if (nb == 0) return Status::Ok();

  std::vector<TensorF32> branch_pres(static_cast<std::size_t>(nb));
  std::vector<TensorF32> branch_outs(static_cast<std::size_t>(nb));
  for (int i = 0; i < nb; ++i) {
    if (auto st = dense::forward(layer.branches[static_cast<std::size_t>(i)], input,
                                 &branch_pres[static_cast<std::size_t>(i)],
                                 &branch_outs[static_cast<std::size_t>(i)]);
        !st) {
      return Status::Err(st.code, std::string("parallel_train branch fwd: ") + st.message);
    }
  }

  std::vector<TensorF32> branch_grads;
  if (auto st = split_combine_grad(layer, grad_out, branch_outs, &branch_grads); !st) {
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

    TensorF32 gin, dW;
    if (auto st = dense::backward_f32(layer.branches[static_cast<std::size_t>(i)],
                                      branch_grads[static_cast<std::size_t>(i)], input,
                                      branch_pres[static_cast<std::size_t>(i)], &gin, &dW);
        !st) {
      return Status::Err(st.code, std::string("parallel_train branch bwd: ") + st.message);
    }
    if (auto st = dense::apply_grad_sgd(&layer.branches[static_cast<std::size_t>(i)], dW, blr); !st) {
      return st;
    }
  }
  return Status::Ok();
}

Status train_unified_bp(parallel::Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                        double lr) {
  // Unified: recompute branch posts, split grads, SGD each branch @ +lr (no KD/gate).
  const int nb = static_cast<int>(layer.branches.size());
  std::vector<TensorF32> branch_pres(static_cast<std::size_t>(nb));
  std::vector<TensorF32> branch_outs(static_cast<std::size_t>(nb));
  for (int i = 0; i < nb; ++i) {
    if (auto st = dense::forward(layer.branches[static_cast<std::size_t>(i)], input,
                                 &branch_pres[static_cast<std::size_t>(i)],
                                 &branch_outs[static_cast<std::size_t>(i)]);
        !st) {
      return st;
    }
  }
  std::vector<TensorF32> branch_grads;
  if (auto st = split_combine_grad(layer, grad_out, branch_outs, &branch_grads); !st) {
    return st;
  }
  for (int i = 0; i < nb; ++i) {
    TensorF32 gin, dW;
    if (auto st = dense::backward_f32(layer.branches[static_cast<std::size_t>(i)],
                                      branch_grads[static_cast<std::size_t>(i)], input,
                                      branch_pres[static_cast<std::size_t>(i)], &gin, &dW);
        !st) {
      return st;
    }
    if (auto st = dense::apply_grad_sgd(&layer.branches[static_cast<std::size_t>(i)], dW, lr); !st) {
      return st;
    }
  }
  return Status::Ok();
}

} // namespace

Status train_mse(parallel::Layer& layer, const TensorF32& input, const TensorF32& target,
                 TrainMode parent_mode, double lr, double* loss) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "parallel_train: nil loss");
  }
  TensorF32 pre, post;
  if (auto st = parallel::forward(layer, input, &pre, &post); !st) {
    return Status::Err(st.code, std::string("parallel_train fwd: ") + st.message);
  }
  TensorF32 gy;
  if (auto st = training::mse(post, target, loss); !st) return st;
  if (auto st = training::mse_grad(post, target, &gy); !st) return st;
  if (layer.cam_kit) layer.cam_kit->note_loss(*loss);

  const TrainMode mode = resolve_train_mode(parent_mode, TrainMode::NormalBP);
  if (train_mode_is_frozen(mode)) {
    return Status::Ok();
  }
  if (parallel::has_mixed_branch_modes(layer)) {
    return train_parallel_mixed(layer, gy, input, mode, lr);
  }
  return train_unified_bp(layer, gy, input, lr);
}

} // namespace welvet::runtime::parallel_train
