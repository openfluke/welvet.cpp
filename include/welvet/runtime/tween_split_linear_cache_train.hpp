#pragma once

#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::runtime::tween_split_linear_cache_train {

// Go TrainStackMSE(..., ModeTweenSplitLinearCache): Forward → MSE → cached linear walk.
Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target, double lr,
                       double* loss);

} // namespace welvet::runtime::tween_split_linear_cache_train
