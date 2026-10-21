#include "welvet/systems/tween_split.hpp"

#include "welvet/runtime/backward.hpp"
#include "welvet/runtime/training.hpp"
#include "welvet/systems/tween.hpp"

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
#include "welvet/stack.hpp"
#include "welvet/swiglu.hpp"

#include <string>
#include <utility>
#include <vector>

namespace welvet::systems::tween_split {
namespace {

TensorF32 clone_t(const TensorF32& t) {
  TensorF32 o;
  o.shape = t.shape;
  o.data = t.data;
  return o;
}

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

void scale_gap(TensorF32* g, double scale) {
  if (!g) return;
  for (float& v : g->data) {
    v = static_cast<float>(static_cast<double>(v) * scale);
  }
}

struct Leaf {
  stack::Child* child = nullptr;
  dense::Layer* dense = nullptr;
  architecture::Cell* cell = nullptr;
  TensorF32 input;
  TensorF32 pre;
  TensorF32 post;
};

Status apply_dense_leaf(dense::Layer& d, const TensorF32& gy, const TensorF32& input,
                        const TensorF32& pre, double lr) {
  if (gy.data.empty() || input.data.empty()) {
    return Status::Ok();
  }
  TensorF32 g_in, dW;
  if (auto st = dense::backward_f32(d, gy, input, pre, &g_in, &dW); !st) {
    return st;
  }
  if (lr != 0.0 && !dW.data.empty()) {
    return dense::apply_grad_sgd(&d, dW, lr);
  }
  return Status::Ok();
}

Status apply_child_leaf(stack::Child& c, const TensorF32& gy, const TensorF32& input,
                        const TensorF32& pre, double lr) {
  TensorF32 g_in;
  return stack::child_family_bp(c, gy, input, pre, lr, &g_in);
}

Status apply_cell_dw(architecture::Cell& cell, const TensorF32& dW, double lr) {
  if (dW.data.empty() || lr == 0.0) {
    return Status::Ok();
  }
  switch (cell.op_kind) {
  case architecture::OpKind::Dense:
    return dense::apply_grad_sgd(&cell.dense_op, dW, lr);
  case architecture::OpKind::RMSNorm:
    return rmsnorm::apply_grad_sgd(&cell.rmsnorm_op, dW, lr);
  case architecture::OpKind::LayerNorm:
    return layernorm::apply_grad_sgd(&cell.layernorm_op, dW, lr);
  case architecture::OpKind::SwiGLU:
    return swiglu::apply_grad_sgd(&cell.swiglu_op, dW, lr);
  case architecture::OpKind::Softmax:
    return Status::Ok();
  case architecture::OpKind::Embedding:
    return embedding::apply_grad_sgd(&cell.embedding_op, dW, lr);
  case architecture::OpKind::MHA:
    return mha::apply_grad_sgd(&cell.mha_op, dW, lr);
  case architecture::OpKind::CNN1:
    return cnn1::apply_grad_sgd(&cell.cnn1_op, dW, lr);
  case architecture::OpKind::CNN2:
    return cnn2::apply_grad_sgd(&cell.cnn2_op, dW, lr);
  case architecture::OpKind::CNN3:
    return cnn3::apply_grad_sgd(&cell.cnn3_op, dW, lr);
  case architecture::OpKind::ConvT1:
    return convt1::apply_grad_sgd(&cell.convt1_op, dW, lr);
  case architecture::OpKind::ConvT2:
    return convt2::apply_grad_sgd(&cell.convt2_op, dW, lr);
  case architecture::OpKind::ConvT3:
    return convt3::apply_grad_sgd(&cell.convt3_op, dW, lr);
  case architecture::OpKind::RNN:
    return rnn::apply_grad_sgd(&cell.rnn_op, dW, lr);
  case architecture::OpKind::LSTM:
    return lstm::apply_grad_sgd(&cell.lstm_op, dW, lr);
  case architecture::OpKind::Sequential:
    return sequential::apply_grad_sgd(&cell.sequential_op, dW, lr);
  case architecture::OpKind::Residual:
    return residual::apply_grad_sgd(&cell.residual_op, dW, lr);
  case architecture::OpKind::Mamba:
    return mamba::apply_grad_sgd(&cell.mamba_op, dW, lr);
  case architecture::OpKind::Stack:
    return stack::apply_grad_sgd(&cell.stack_op, dW, lr);
  default:
    return Status::Ok();
  }
}

Status apply_cell_leaf(architecture::Cell& cell, const TensorF32& gy, const TensorF32& input,
                       const TensorF32& pre, double lr) {
  TensorF32 g_in, dW;
  if (auto st = runtime::backward::cell(cell, gy, input, pre, &g_in, &dW); !st) {
    return st;
  }
  return apply_cell_dw(cell, dW, lr);
}

Status collect_seq_dense(sequential::Layer& layer, const TensorF32& input, std::vector<Leaf>* out) {
  TensorF32 flat_in = flatten_seq_like(input);
  TensorF32 current = flat_in;
  for (auto& ch : layer.children) {
    ch.backend = layer.backend;
    ch.multi_core = false;
    Leaf leaf;
    leaf.dense = &ch;
    leaf.input = current;
    TensorF32 p, o;
    if (auto st = dense::forward(ch, current, &p, &o); !st) {
      return st;
    }
    leaf.pre = std::move(p);
    leaf.post = o;
    out->push_back(std::move(leaf));
    current = std::move(o);
  }
  return Status::Ok();
}

Status collect_res_dense(residual::Layer& layer, const TensorF32& input, std::vector<Leaf>* out) {
  TensorF32 flat_in = flatten_seq_like(input);
  TensorF32 current = flat_in;
  for (auto& ch : layer.children) {
    ch.backend = layer.backend;
    ch.multi_core = false;
    Leaf leaf;
    leaf.dense = &ch;
    leaf.input = current;
    TensorF32 p, o;
    if (auto st = dense::forward(ch, current, &p, &o); !st) {
      return st;
    }
    leaf.pre = std::move(p);
    leaf.post = o;
    out->push_back(std::move(leaf));
    current = std::move(o);
  }
  return Status::Ok();
}

Status collect_child(stack::Child& c, const TensorF32& input, std::vector<Leaf>* out) {
  if (c.kind == stack::Kind::Sequential) {
    return collect_seq_dense(c.sequential_op, input, out);
  }
  if (c.kind == stack::Kind::Residual) {
    return collect_res_dense(c.residual_op, input, out);
  }
  if (c.kind == stack::Kind::NestedStack) {
    TensorF32 current = input;
    for (auto& ch : c.nested) {
      if (auto st = collect_child(ch, current, out); !st) {
        return st;
      }
      TensorF32 p, o;
      if (auto st = stack::child_forward(ch, current, &p, &o); !st) {
        return st;
      }
      current = std::move(o);
    }
    return Status::Ok();
  }
  if (stack::child_grad_w_size(c) <= 0) {
    return Status::Ok();
  }
  Leaf leaf;
  leaf.child = &c;
  leaf.input = input;
  TensorF32 p, o;
  if (auto st = stack::child_forward(c, input, &p, &o); !st) {
    return st;
  }
  leaf.pre = std::move(p);
  leaf.post = std::move(o);
  out->push_back(std::move(leaf));
  return Status::Ok();
}

Status collect_cell(architecture::Cell& cell, const TensorF32& input, const TensorF32& pre,
                    const TensorF32& post, std::vector<Leaf>* out) {
  if (cell.meta.is_disabled) {
    return Status::Ok();
  }
  if (cell.op_kind == architecture::OpKind::Sequential) {
    return collect_seq_dense(cell.sequential_op, input, out);
  }
  if (cell.op_kind == architecture::OpKind::Residual) {
    return collect_res_dense(cell.residual_op, input, out);
  }
  if (cell.op_kind == architecture::OpKind::Softmax || cell.op_kind == architecture::OpKind::None) {
    return Status::Ok();
  }
  Leaf leaf;
  leaf.cell = &cell;
  leaf.input = input;
  leaf.pre = pre;
  leaf.post = post;
  out->push_back(std::move(leaf));
  return Status::Ok();
}

Status apply_one_leaf(Leaf& leaf, const TensorF32& gy, double lr) {
  if (leaf.dense) {
    return apply_dense_leaf(*leaf.dense, gy, leaf.input, leaf.pre, lr);
  }
  if (leaf.child) {
    return apply_child_leaf(*leaf.child, gy, leaf.input, leaf.pre, lr);
  }
  if (leaf.cell) {
    return apply_cell_leaf(*leaf.cell, gy, leaf.input, leaf.pre, lr);
  }
  return Status::Ok();
}

Status apply_leaves(std::vector<Leaf>& leaves, const TensorF32& grad_out, double lr) {
  const int n = static_cast<int>(leaves.size());
  if (n == 0) {
    return Status::Ok();
  }
  const double scale = 1.0 / static_cast<double>(n);
  for (int i = 0; i < n; ++i) {
    Leaf& leaf = leaves[static_cast<std::size_t>(i)];
    if (leaf.post.data.empty()) {
      continue;
    }
    TensorF32 gy = systems::tween::project_gap(grad_out, leaf.post.shape);
    scale_gap(&gy, scale);
    if (auto st = apply_one_leaf(leaf, gy, lr); !st) {
      return Status::Err(st.code, std::string("tween_split leaf: ") + st.message);
    }
  }
  return Status::Ok();
}

Status apply_leaf_gx(Leaf& leaf, const TensorF32& gy, double lr, TensorF32* gx_out) {
  if (!gx_out) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split: nil gx");
  }
  *gx_out = TensorF32{};
  if (gy.data.empty() || leaf.input.data.empty()) {
    return Status::Ok();
  }
  if (leaf.dense) {
    TensorF32 dW;
    if (auto st = dense::backward_f32(*leaf.dense, gy, leaf.input, leaf.pre, gx_out, &dW); !st) {
      return st;
    }
    if (lr != 0.0 && !dW.data.empty()) {
      return dense::apply_grad_sgd(leaf.dense, dW, lr);
    }
    return Status::Ok();
  }
  if (leaf.child) {
    return stack::child_family_bp(*leaf.child, gy, leaf.input, leaf.pre, lr, gx_out);
  }
  if (leaf.cell) {
    TensorF32 dW;
    if (auto st = runtime::backward::cell(*leaf.cell, gy, leaf.input, leaf.pre, gx_out, &dW); !st) {
      return st;
    }
    return apply_cell_dw(*leaf.cell, dW, lr);
  }
  return Status::Ok();
}

Status apply_hidden_dw(std::vector<Leaf>& leaves, int begin, int end, const TensorF32& src,
                       double scale, double lr) {
  for (int i = begin; i < end; ++i) {
    Leaf& leaf = leaves[static_cast<std::size_t>(i)];
    if (leaf.post.data.empty()) {
      continue;
    }
    TensorF32 gi = systems::tween::project_gap(src, leaf.post.shape);
    scale_gap(&gi, scale);
    if (auto st = apply_one_leaf(leaf, gi, lr); !st) {
      return Status::Err(st.code, std::string("tween_split hidden: ") + st.message);
    }
  }
  return Status::Ok();
}

Status apply_head_proxy_leaves(std::vector<Leaf>& leaves, const TensorF32& grad_out, double lr) {
  const int n = static_cast<int>(leaves.size());
  if (n == 0) {
    return Status::Ok();
  }
  Leaf& head = leaves[static_cast<std::size_t>(n - 1)];
  if (head.post.data.empty()) {
    return apply_leaves(leaves, grad_out, lr);
  }
  TensorF32 g_head = systems::tween::project_gap(grad_out, head.post.shape);
  TensorF32 gx;
  if (auto st = apply_leaf_gx(head, g_head, lr, &gx); !st) {
    return Status::Err(st.code, std::string("tween_split head: ") + st.message);
  }
  if (n == 1) {
    return Status::Ok();
  }
  const TensorF32& src = gx.data.empty() ? grad_out : gx;
  const double scale = 1.0 / static_cast<double>(n - 1);
  return apply_hidden_dw(leaves, 0, n - 1, src, scale, lr);
}

Status collect_stack_leaves(stack::Layer& ops, const TensorF32& input, std::vector<Leaf>* leaves) {
  std::vector<stack::Child*> flat;
  stack::flatten_ops(ops, &flat);
  TensorF32 current = input;
  for (stack::Child* ch : flat) {
    if (!ch) continue;
    TensorF32 p, o;
    if (auto st = stack::child_forward(*ch, current, &p, &o); !st) {
      return st;
    }
    if (auto st = collect_child(*ch, current, leaves); !st) {
      return st;
    }
    current = std::move(o);
  }
  return Status::Ok();
}

Status collect_volume_leaves(architecture::Grid& g, const runtime::forward::Result& fwd,
                             std::vector<Leaf>* leaves) {
  for (const auto& step : fwd.steps) {
    architecture::Cell* cell =
        architecture::at(g, step.coord.z, step.coord.y, step.coord.x, step.coord.l);
    if (!cell) continue;
    if (auto st = collect_cell(*cell, step.input, step.pre, step.post, leaves); !st) {
      return st;
    }
  }
  return Status::Ok();
}

} // namespace

Status apply_split(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input, double lr) {
  std::vector<Leaf> leaves;
  if (auto st = collect_stack_leaves(ops, input, &leaves); !st) {
    return st;
  }
  return apply_leaves(leaves, grad_out, lr);
}

Status apply_split_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                         double lr) {
  std::vector<Leaf> leaves;
  if (auto st = collect_child(c, input, &leaves); !st) {
    return st;
  }
  return apply_leaves(leaves, grad_out, lr);
}

Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split: nil loss");
  }
  if (fwd.steps.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split: empty forward tape");
  }
  if (auto st = runtime::training::mse(fwd.output, target, loss); !st) {
    return st;
  }
  TensorF32 gy;
  if (auto st = runtime::training::mse_grad(fwd.output, target, &gy); !st) {
    return st;
  }
  std::vector<Leaf> leaves;
  if (auto st = collect_volume_leaves(g, fwd, &leaves); !st) {
    return st;
  }
  return apply_leaves(leaves, gy, lr);
}

Status apply_head_proxy(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                        double lr) {
  std::vector<Leaf> leaves;
  if (auto st = collect_stack_leaves(ops, input, &leaves); !st) {
    return st;
  }
  return apply_head_proxy_leaves(leaves, grad_out, lr);
}

Status apply_head_proxy_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                              double lr) {
  std::vector<Leaf> leaves;
  if (auto st = collect_child(c, input, &leaves); !st) {
    return st;
  }
  return apply_head_proxy_leaves(leaves, grad_out, lr);
}

Status train_volume_mse_head_proxy(architecture::Grid& g, const runtime::forward::Result& fwd,
                                   const TensorF32& target, double lr, double* loss) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split head_proxy: nil loss");
  }
  if (fwd.steps.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split head_proxy: empty forward tape");
  }
  if (auto st = runtime::training::mse(fwd.output, target, loss); !st) {
    return st;
  }
  TensorF32 gy;
  if (auto st = runtime::training::mse_grad(fwd.output, target, &gy); !st) {
    return st;
  }
  std::vector<Leaf> leaves;
  if (auto st = collect_volume_leaves(g, fwd, &leaves); !st) {
    return st;
  }
  return apply_head_proxy_leaves(leaves, gy, lr);
}

} // namespace welvet::systems::tween_split
