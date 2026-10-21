#pragma once

#include "welvet/parallel.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/train_mode.hpp"

namespace welvet::runtime::parallel_train {

// Go parallel.TrainMSE: Family() dispatch (Tween/TweenChain/TweenSplit/TweenAlt/BP)
// then Forward → MSE → NoteLoss → Train (or OpenSplitTape / Alt recompute).
// Dense (+ poly) branches; splitCombineGrad covers all Combine modes (+ Filter gate SGD).
Status train_mse(parallel::Layer& layer, const TensorF32& input, const TensorF32& target,
                 TrainMode parent_mode, double lr, double* loss);

} // namespace welvet::runtime::parallel_train
