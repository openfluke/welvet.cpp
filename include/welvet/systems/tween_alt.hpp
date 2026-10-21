#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::systems::tween_alt {

// ModeTweenAlt volume: AltTimes × (Forward → Split @ full lr → Forward → Tween @ lr×0.5).
// Uses volume_dense for both forwards (same as TweenSplit/ModeTween volume wires).
Status train_volume_mse(architecture::Grid& g, const TensorF32& input, const TensorF32& target,
                        double lr, double* loss, int alt_times = 1);

} // namespace welvet::systems::tween_alt
