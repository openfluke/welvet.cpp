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
// RoleTrain requires target; on pop runs FamilyBP (MSE gap + reverse BP+SGD).
Status tick(stack::Layer& ops, State* st, const TensorF32& input, const TensorF32* target,
            FlightRole role, double lr, TickResult* out);

// RoleTrain convenience: returns loss from pop (0 if no train sample popped).
Status train_mse(stack::Layer& ops, State* st, const TensorF32& input, const TensorF32& target,
                 double lr, double* loss);

int in_flight(const State& st);

} // namespace welvet::runtime::line
