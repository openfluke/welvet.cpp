#pragma once

#include "welvet/architecture.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <cstdint>
#include <vector>

namespace welvet::runtime::step {

// Discrete-time volumetric mesh clock (Go runtime/step).
// Each Forward tick updates every cell from double-buffered LayerData.
struct State {
  std::vector<TensorF32> layer_data;
  std::vector<TensorF32> next_buffer;
  std::vector<std::vector<TensorF32>> history_in;
  std::vector<std::vector<TensorF32>> history_pre;
  TensorF32 last_input;
  uint64_t step_count = 0;
};

void ensure_size(State* s, int n);
void set_input(State* s, const TensorF32& input);
void clear_history(State* s);

// One simultaneous grid tick. capture=true appends HistoryIn/HistoryPre for BPTT.
Status forward(architecture::Grid& g, State* s, bool capture);

// BPTT over captured history; layer_dws[i] is accumulated dW for cell i (may be empty).
Status backward(architecture::Grid& g, State* s, const TensorF32& grad_out, TensorF32* grad_in,
                std::vector<TensorF32>* layer_dws);

// Apply per-cell dW via FamilyBP ApplyGradSGD.
Status apply_sgd(architecture::Grid& g, const std::vector<TensorF32>& layer_dws, double lr);

// FamilyBP = MeshBP / MeshTweenChain (BPTT + SGD @ full lr).
// FamilyTween = MeshTween (last-tick projected local @ lr*0.5; no BPTT chain).
enum class Credit : int32_t {
  FamilyBP = 0,
  FamilyTween = 1,
};

// Mesh-clock train: Forward×ticks → MSE on last cell → credit update.
Status train_mse(architecture::Grid& g, const TensorF32& input, const TensorF32& target, int ticks,
                 double lr, double* loss, Credit credit = Credit::FamilyBP);

} // namespace welvet::runtime::step
