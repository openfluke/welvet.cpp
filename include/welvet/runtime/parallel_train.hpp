#pragma once

#include "welvet/parallel.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/train_mode.hpp"

namespace welvet::runtime::parallel_train {

// Go parallel.TrainMSE: Forward → MSE → NoteLoss → Train (mixed BranchModes).
// Dense branches only; Combine Add/Avg supported for splitCombineGrad.
Status train_mse(parallel::Layer& layer, const TensorF32& input, const TensorF32& target,
                 TrainMode parent_mode, double lr, double* loss);

} // namespace welvet::runtime::parallel_train
