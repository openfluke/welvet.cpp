#include "welvet/systems/tween.hpp"

#include "welvet/runtime/backward.hpp"
#include "welvet/runtime/training.hpp"

#include "welvet/cnn1.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/convt1.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt3.hpp"
#include "welvet/dense.hpp"
#include "welvet/embedding.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/lstm.hpp"
#include "welvet/mamba.hpp"
#include "welvet/mha.hpp"
#include "welvet/residual.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/rnn.hpp"
#include "welvet/sequential.hpp"
#include "welvet/softmax.hpp"
#include "welvet/stack.hpp"
#include "welvet/swiglu.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace welvet::systems::tween {
namespace {

Status apply_one(architecture::Cell* cell, const TensorF32& dW, double lr) {
  if (!cell || dW.data.empty() || lr == 0.0) {
    return Status::Ok();
  }
  switch (cell->op_kind) {
  case architecture::OpKind::Dense:
    return dense::apply_grad_sgd(&cell->dense_op, dW, lr);
  case architecture::OpKind::RMSNorm:
    return rmsnorm::apply_grad_sgd(&cell->rmsnorm_op, dW, lr);
  case architecture::OpKind::LayerNorm:
    return layernorm::apply_grad_sgd(&cell->layernorm_op, dW, lr);
  case architecture::OpKind::SwiGLU:
    return swiglu::apply_grad_sgd(&cell->swiglu_op, dW, lr);
  case architecture::OpKind::Softmax:
    return softmax::apply_grad_sgd(&cell->softmax_op, dW, lr);
  case architecture::OpKind::Embedding:
    return embedding::apply_grad_sgd(&cell->embedding_op, dW, lr);
  case architecture::OpKind::MHA:
    return mha::apply_grad_sgd(&cell->mha_op, dW, lr);
  case architecture::OpKind::CNN1:
    return cnn1::apply_grad_sgd(&cell->cnn1_op, dW, lr);
  case architecture::OpKind::CNN2:
    return cnn2::apply_grad_sgd(&cell->cnn2_op, dW, lr);
  case architecture::OpKind::CNN3:
    return cnn3::apply_grad_sgd(&cell->cnn3_op, dW, lr);
  case architecture::OpKind::ConvT1:
    return convt1::apply_grad_sgd(&cell->convt1_op, dW, lr);
  case architecture::OpKind::ConvT2:
    return convt2::apply_grad_sgd(&cell->convt2_op, dW, lr);
  case architecture::OpKind::ConvT3:
    return convt3::apply_grad_sgd(&cell->convt3_op, dW, lr);
  case architecture::OpKind::RNN:
    return rnn::apply_grad_sgd(&cell->rnn_op, dW, lr);
  case architecture::OpKind::LSTM:
    return lstm::apply_grad_sgd(&cell->lstm_op, dW, lr);
  case architecture::OpKind::Sequential:
    return sequential::apply_grad_sgd(&cell->sequential_op, dW, lr);
  case architecture::OpKind::Residual:
    return residual::apply_grad_sgd(&cell->residual_op, dW, lr);
  case architecture::OpKind::Mamba:
    return mamba::apply_grad_sgd(&cell->mamba_op, dW, lr);
  case architecture::OpKind::Stack:
    return stack::apply_grad_sgd(&cell->stack_op, dW, lr);
  default:
    return Status::Ok();
  }
}

Status apply_dense_projected(dense::Layer& d, const TensorF32& grad_out, const TensorF32& input,
                             const TensorF32& pre, const TensorF32& post, double soft_lr) {
  if (grad_out.data.empty() || input.data.empty()) {
    return Status::Ok();
  }
  TensorF32 proj = project_gap(grad_out, post.shape);
  if (proj.data.empty()) {
    return Status::Ok();
  }
  TensorF32 g_in, dW;
  if (auto st = dense::backward_f32(d, proj, input, pre, &g_in, &dW); !st) {
    return st;
  }
  if (soft_lr != 0.0 && !dW.data.empty()) {
    if (auto st = dense::apply_grad_sgd(&d, dW, soft_lr); !st) {
      return st;
    }
  }
  return Status::Ok();
}

// Flatten [B,D] or [B,S,D] → [B*S,D] for Dense children (matches sequential::forward).
TensorF32 flatten_seq_like(const TensorF32& t) {
  if (t.shape.size() == 3) {
    TensorF32 out;
    const int b = static_cast<int>(t.shape[0]);
    const int s = static_cast<int>(t.shape[1]);
    const int d = static_cast<int>(t.shape[2]);
    out.shape = {static_cast<int64_t>(b) * s, d};
    out.data = t.data;
    return out;
  }
  return t;
}

// Go collectTweenLeaves + applyProjectedTween on Sequential Dense children.
Status apply_projected_seq(sequential::Layer& layer, const TensorF32& grad_out,
                           const TensorF32& input, double soft_lr) {
  if (layer.children.empty()) {
    return Status::Ok();
  }
  TensorF32 flat_in = flatten_seq_like(input);
  TensorF32 flat_gy = flatten_seq_like(grad_out);
  std::vector<TensorF32> ins(layer.children.size());
  std::vector<TensorF32> pres(layer.children.size());
  std::vector<TensorF32> posts(layer.children.size());
  TensorF32 current = flat_in;
  for (std::size_t i = 0; i < layer.children.size(); ++i) {
    auto& ch = layer.children[i];
    ch.backend = layer.backend;
    ch.multi_core = false;
    ins[i] = current;
    TensorF32 p, o;
    if (auto st = dense::forward(ch, current, &p, &o); !st) {
      return Status::Err(st.code, std::string("tween seq fwd: ") + st.message);
    }
    pres[i] = std::move(p);
    posts[i] = o;
    current = std::move(o);
  }
  for (std::size_t i = 0; i < layer.children.size(); ++i) {
    if (auto st = apply_dense_projected(layer.children[i], flat_gy, ins[i], pres[i], posts[i],
                                        soft_lr);
        !st) {
      return st;
    }
  }
  return Status::Ok();
}

// Go residual ChildOps leaf walk — same global gap on each F Dense (no chain rule).
Status apply_projected_res(residual::Layer& layer, const TensorF32& grad_out,
                           const TensorF32& input, double soft_lr) {
  if (layer.children.empty()) {
    return Status::Ok();
  }
  TensorF32 flat_in = flatten_seq_like(input);
  TensorF32 flat_gy = flatten_seq_like(grad_out);
  std::vector<TensorF32> ins(layer.children.size());
  std::vector<TensorF32> pres(layer.children.size());
  std::vector<TensorF32> posts(layer.children.size());
  TensorF32 current = flat_in;
  for (std::size_t i = 0; i < layer.children.size(); ++i) {
    auto& ch = layer.children[i];
    ch.backend = layer.backend;
    ch.multi_core = false;
    ins[i] = current;
    TensorF32 p, o;
    if (auto st = dense::forward(ch, current, &p, &o); !st) {
      return Status::Err(st.code, std::string("tween res fwd: ") + st.message);
    }
    pres[i] = std::move(p);
    posts[i] = o;
    current = std::move(o);
  }
  for (std::size_t i = 0; i < layer.children.size(); ++i) {
    if (auto st = apply_dense_projected(layer.children[i], flat_gy, ins[i], pres[i], posts[i],
                                        soft_lr);
        !st) {
      return st;
    }
  }
  return Status::Ok();
}

} // namespace

TensorF32 project_gap(const TensorF32& src, const std::vector<int64_t>& dst_shape) {
  TensorF32 dst;
  dst.shape = dst_shape;
  int dn = 1;
  for (int64_t d : dst_shape) {
    dn *= static_cast<int>(d);
  }
  if (dn <= 0) {
    return dst;
  }
  dst.data.resize(static_cast<std::size_t>(dn));
  const int sn = static_cast<int>(src.data.size());
  if (sn <= 0) {
    return dst;
  }
  if (sn == dn) {
    dst.data = src.data;
    return dst;
  }
  for (int i = 0; i < dn; ++i) {
    int lo = i * sn / dn;
    int hi = (i + 1) * sn / dn;
    if (hi <= lo) {
      hi = lo + 1;
    }
    if (hi > sn) {
      hi = sn;
    }
    double sum = 0.0;
    for (int j = lo; j < hi; ++j) {
      sum += static_cast<double>(src.data[static_cast<std::size_t>(j)]);
    }
    dst.data[static_cast<std::size_t>(i)] = static_cast<float>(sum / static_cast<double>(hi - lo));
  }
  return dst;
}

Status apply_projected_ops(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                           double soft_lr) {
  std::vector<stack::Child*> flat;
  stack::flatten_ops(ops, &flat);
  TensorF32 current = input;
  for (stack::Child* ch : flat) {
    if (!ch) continue;
    TensorF32 p, o;
    if (auto st = stack::child_forward(*ch, current, &p, &o); !st) {
      return st;
    }
    if (auto st = apply_projected(*ch, grad_out, current, soft_lr); !st) {
      return st;
    }
    current = std::move(o);
  }
  return Status::Ok();
}

Status apply_projected(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                       double soft_lr) {
  if (grad_out.data.empty() || input.data.empty()) {
    return Status::Ok();
  }
  switch (c.kind) {
  case stack::Kind::Sequential:
    return apply_projected_seq(c.sequential_op, grad_out, input, soft_lr);
  case stack::Kind::Residual:
    return apply_projected_res(c.residual_op, grad_out, input, soft_lr);
  case stack::Kind::NestedStack: {
    TensorF32 current = input;
    for (auto& ch : c.nested) {
      TensorF32 p, o;
      if (auto st = stack::child_forward(ch, current, &p, &o); !st) {
        return st;
      }
      if (auto st = apply_projected(ch, grad_out, current, soft_lr); !st) {
        return st;
      }
      current = std::move(o);
    }
    return Status::Ok();
  }
  default: {
    TensorF32 pre, post, g_in;
    if (auto st = stack::child_forward(c, input, &pre, &post); !st) {
      return st;
    }
    TensorF32 proj = project_gap(grad_out, post.shape);
    return stack::child_family_bp(c, proj, input, pre, soft_lr, &g_in);
  }
  }
}

Status apply_cell_projected(architecture::Cell& cell, const TensorF32& grad_out,
                            const TensorF32& input, const TensorF32& pre, const TensorF32& post,
                            double soft_lr) {
  if (cell.meta.is_disabled || grad_out.data.empty() || input.data.empty()) {
    return Status::Ok();
  }
  if (cell.op_kind == architecture::OpKind::Sequential) {
    return apply_projected_seq(cell.sequential_op, grad_out, input, soft_lr);
  }
  if (cell.op_kind == architecture::OpKind::Residual) {
    return apply_projected_res(cell.residual_op, grad_out, input, soft_lr);
  }
  TensorF32 proj = project_gap(grad_out, post.shape);
  if (proj.data.empty()) {
    return Status::Ok();
  }
  TensorF32 g_in, dW;
  if (auto st = runtime::backward::cell(cell, proj, input, pre, &g_in, &dW); !st) {
    return st;
  }
  return apply_one(&cell, dW, soft_lr);
}

Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "tween: nil loss");
  }
  if (fwd.steps.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "tween: empty forward tape");
  }
  if (auto st = runtime::training::mse(fwd.output, target, loss); !st) {
    return st;
  }
  TensorF32 gy;
  if (auto st = runtime::training::mse_grad(fwd.output, target, &gy); !st) {
    return st;
  }
  const double soft_lr = lr * 0.5;
  // Reverse hop — Go trainStackMixed + trainTweenLocal per cell.
  for (int i = static_cast<int>(fwd.steps.size()) - 1; i >= 0; --i) {
    const auto& step = fwd.steps[static_cast<std::size_t>(i)];
    architecture::Cell* cell =
        architecture::at(g, step.coord.z, step.coord.y, step.coord.x, step.coord.l);
    if (!cell || cell->meta.is_disabled) {
      continue;
    }
    if (auto st = apply_cell_projected(*cell, gy, step.input, step.pre, step.post, soft_lr); !st) {
      return st;
    }
    // Go trainTweenLocal: pass gap forward only when lengths match input.
    TensorF32 next;
    next.shape = step.input.shape;
    next.data.assign(step.input.data.size(), 0.0f);
    if (gy.data.size() == next.data.size()) {
      next.data = gy.data;
    }
    gy = std::move(next);
  }
  return Status::Ok();
}

} // namespace welvet::systems::tween
