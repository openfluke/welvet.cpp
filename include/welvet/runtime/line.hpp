#pragma once

#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <cstdint>
#include <vector>

namespace welvet::runtime::line {

// Go parallel.FlightRole — RoleTrain credits on pop; RoleAction emits post only.
enum class FlightRole : uint8_t {
  Train = 0,
  Action = 1,
};

// Credit applied when a RoleTrain sample pops the last child.
// FamilyBP = StepBP / StepTweenChain (chain rule @ full lr);
// FamilyTween = StepTween (projected @ lr*0.5);
// FamilyTweenSplit = StepTweenSplit (1/N projected gap @ full lr per hop);
// FamilyTweenAlt = StepTweenAlt (Split→Tween frozen gap per hop; AltTimes=1).
// FamilyTweenSplitHeadProxy = StepTweenSplitHeadProxy (head Jᵀgy; hidden 1/(N−1) P(gx)).
// FamilyTweenSplitLinear = StepTweenSplitLinear (affine Wᵀ walk, skip act′; 1/N GradWOnly).
// FamilyTweenSplitFastProxy = StepTweenSplitFastProxy (head GradWOnly; hidden 1/(N−1) P(Wᵀgy)).
// FamilyTweenSplitLinearCache = StepTweenSplitLinearCache (cached linear walk; Dense hops accel=nil).
// FamilyTweenSplitHeadProxyAsync = StepTweenSplitHeadProxyAsync (stale proxy; Dense hops accel=nil).
// FamilyTweenSplitSparse = StepTweenSplitSparse (head + one rotating hidden; Dense hops accel=nil).
enum class Credit : uint8_t {
  FamilyBP = 0,
  FamilyTween = 1,
  FamilyTweenSplit = 2,
  FamilyTweenAlt = 3,
  FamilyTweenSplitHeadProxy = 4,
  FamilyTweenSplitLinear = 5,
  FamilyTweenSplitFastProxy = 6,
  FamilyTweenSplitLinearCache = 7,
  FamilyTweenSplitHeadProxyAsync = 8,
  FamilyTweenSplitSparse = 9,
};

struct Flight {
  FlightRole role = FlightRole::Train;
  TensorF32 target;
  int stage = 0;
  std::vector<TensorF32> ins;
  std::vector<TensorF32> pres;
  std::vector<TensorF32> posts;
};

// Systolic 1D pipe over flattened Stack children (Go parallel.Line).
struct State {
  std::vector<Flight> flights;
};

struct TickResult {
  double loss = 0.0;
  bool popped_train = false;
  TensorF32 action_out;
};

// Inject one sample, advance every in-flight flight one child hop.
// RoleTrain requires target; on pop runs `credit` (MSE gap + reverse update).
Status tick(stack::Layer& ops, State* st, const TensorF32& input, const TensorF32* target,
            FlightRole role, double lr, TickResult* out, Credit credit = Credit::FamilyBP);

// RoleTrain convenience: returns loss from pop (0 if no train sample popped).
Status train_mse(stack::Layer& ops, State* st, const TensorF32& input, const TensorF32& target,
                 double lr, double* loss, Credit credit = Credit::FamilyBP);

int in_flight(const State& st);

} // namespace welvet::runtime::line
