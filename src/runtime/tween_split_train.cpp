#include "welvet/runtime/tween_split_train.hpp"

#include "welvet/runtime/training.hpp"
#include "welvet/systems/tween_split.hpp"

#include <string>
#include <utility>
#include <vector>

namespace welvet::runtime::tween_split_train {

Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target, double lr,
                       double* loss) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split_train: nil loss");
  }
  std::vector<stack::Child*> flat;
  stack::flatten_ops(ops, &flat);
  if (flat.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split_train: empty stack");
  }

  // Forward once for loss (apply_split re-forwards for leaf collection — matches Go
  // OpenSplitTape collect + TrainGap).
  TensorF32 current = input;
  for (stack::Child* ch : flat) {
    TensorF32 p, o;
    if (auto st = stack::child_forward(*ch, current, &p, &o); !st) {
      return Status::Err(st.code, std::string("tween_split_train fwd: ") + st.message);
    }
    current = std::move(o);
  }
  if (auto st = training::mse(current, target, loss); !st) {
    return st;
  }
  TensorF32 gy;
  if (auto st = training::mse_grad(current, target, &gy); !st) {
    return st;
  }
  return systems::tween_split::apply_split(ops, gy, input, lr);
}

} // namespace welvet::runtime::tween_split_train
