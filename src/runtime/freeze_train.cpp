#include "welvet/runtime/freeze_train.hpp"

#include "welvet/runtime/training.hpp"

#include <string>
#include <utility>
#include <vector>

namespace welvet::runtime::freeze_train {

Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target, double lr,
                       double* loss) {
  (void)lr;
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "freeze_train: nil loss");
  }
  std::vector<stack::Child*> flat;
  stack::flatten_ops(ops, &flat);
  if (flat.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "freeze_train: empty stack");
  }

  TensorF32 current = input;
  for (stack::Child* ch : flat) {
    TensorF32 p, o;
    if (auto st = stack::child_forward(*ch, current, &p, &o); !st) {
      return Status::Err(st.code, std::string("freeze_train fwd: ") + st.message);
    }
    current = std::move(o);
  }
  return training::mse(current, target, loss);
}

Status train_volume_mse(architecture::Grid& g, const forward::Result& fwd, const TensorF32& target,
                        double lr, double* loss) {
  (void)g;
  (void)lr;
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "freeze_train: nil loss");
  }
  if (fwd.steps.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "freeze_train: empty forward tape");
  }
  return training::mse(fwd.output, target, loss);
}

} // namespace welvet::runtime::freeze_train
