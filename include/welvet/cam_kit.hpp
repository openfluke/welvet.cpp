#pragma once

namespace welvet {

// Go parallel.CamKit subset — Shadow KD + Memory surprise gate (DNA/Dream deferred).
struct CamKit {
  double shadow_coef = 0;      // 0 ⇒ 1.0 when Shadow cams exist
  double surprise_thresh = 0;  // ≤0 ⇒ Memory always trains
  double last_loss = 0;

  double effective_shadow_coef() const { return shadow_coef == 0.0 ? 1.0 : shadow_coef; }

  bool memory_allows_train() const {
    if (surprise_thresh <= 0.0) return true;
    return last_loss >= surprise_thresh;
  }

  void note_loss(double loss) { last_loss = loss; }
};

} // namespace welvet
