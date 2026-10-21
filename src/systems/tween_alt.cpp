#include "welvet/systems/tween_alt.hpp"

#include "welvet/systems/tween.hpp"
#include "welvet/systems/tween_split.hpp"

#include <string>

namespace welvet::systems::tween_alt {

Status train_volume_mse(architecture::Grid& g, const TensorF32& input, const TensorF32& target,
                        double lr, double* loss, int alt_times) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "tween_alt: nil loss");
  }
  const int times = alt_times < 1 ? 1 : alt_times;
  double last = 0.0;
  for (int i = 0; i < times; ++i) {
    runtime::forward::Result fwd;
    if (auto st = runtime::forward::volume_dense(g, input, &fwd); !st) {
      return Status::Err(st.code, std::string("tween_alt fwd split: ") + st.message);
    }
    if (auto st = tween_split::train_volume_mse(g, fwd, target, lr, &last); !st) {
      return Status::Err(st.code, std::string("tween_alt split: ") + st.message);
    }
    if (auto st = runtime::forward::volume_dense(g, input, &fwd); !st) {
      return Status::Err(st.code, std::string("tween_alt fwd tween: ") + st.message);
    }
    if (auto st = tween::train_volume_mse(g, fwd, target, lr, &last); !st) {
      return Status::Err(st.code, std::string("tween_alt tween: ") + st.message);
    }
  }
  *loss = last;
  return Status::Ok();
}

} // namespace welvet::systems::tween_alt
