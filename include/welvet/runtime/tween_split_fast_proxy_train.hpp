#pragma once

#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::runtime::tween_split_fast_proxy_train {

// Go TrainStackMSE(..., ModeTweenSplitFastProxy): Forward → MSE → fast proxy credit.
Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target, double lr,
                       double* loss);

} // namespace welvet::runtime::tween_split_fast_proxy_train
