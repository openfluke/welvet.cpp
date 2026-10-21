#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::systems::tween_split {

// ModeTweenSplit: one global gap, split 1/N across every trainable leaf
// (Go trainTweenSplitOp). Full lr (not soft_lr). No chain rule.
Status apply_split(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                   double lr);

// Volume: collect leaves from every enabled cell on the forward tape, then 1/N.
Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss);

} // namespace welvet::systems::tween_split
