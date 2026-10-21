#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::runtime::freeze_train {

// Go TrainStackMSE(..., ModeFreeze): Forward → MSE → TrainStack returns nil (no ApplyGrad).
Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target, double lr,
                       double* loss);

// Volume: mse only — weights unchanged (ModeFreeze / IsFrozen).
Status train_volume_mse(architecture::Grid& g, const forward::Result& fwd, const TensorF32& target,
                        double lr, double* loss);

} // namespace welvet::runtime::freeze_train
