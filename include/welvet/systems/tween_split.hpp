#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::systems::tween_split {

// ModeTweenSplit: one global gap, split 1/N across every trainable leaf
// (Go trainTweenSplitOp / OpenSplitTape). Full lr (not soft_lr). No chain rule.
Status apply_split(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                   double lr);

// One Op: collect leaves under this child, then 1/N (Go trainTweenSplitFamily).
// Used by Line StepTweenSplit (per-hop on pop).
Status apply_split_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                         double lr);

// Volume: collect leaves from every enabled cell on the forward tape, then 1/N.
Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss);

// ModeTweenSplitHeadProxy: head full Jᵀg_y (+ gx); hidden 1/(N-1) P(gx_head).
Status apply_head_proxy(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                        double lr);
Status apply_head_proxy_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                              double lr);
Status train_volume_mse_head_proxy(architecture::Grid& g, const runtime::forward::Result& fwd,
                                   const TensorF32& target, double lr, double* loss);

} // namespace welvet::systems::tween_split
