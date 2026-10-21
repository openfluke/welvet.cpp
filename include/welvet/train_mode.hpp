#pragma once

#include <cstdint>

namespace welvet {

// Go parallel.TrainMode iota — keep numeric parity for wire payloads.
enum class TrainMode : std::uint8_t {
  Inherit = 0,
  NormalBP = 1,
  StepBP = 2,
  Tween = 3,
  TweenChain = 4,
  StepTween = 5,
  StepTweenChain = 6,
  MeshBP = 7,
  MeshTween = 8,
  MeshTweenChain = 9,
  TweenSplit = 10,
  StepTweenSplit = 11,
  TweenAlt = 12,
  StepTweenAlt = 13,
  TweenSplitHeadProxy = 14,
  TweenSplitLinear = 15,
  TweenSplitFastProxy = 16,
  TweenSplitLinearCache = 17,
  TweenSplitHeadProxyAsync = 18,
  TweenSplitSparse = 19,
  MeshTweenSplit = 20,
  MeshTweenAlt = 21,
  MeshTweenSplitFastProxy = 22,
  MeshTweenSplitSparse = 23,
  StepTweenSplitHeadProxy = 24,
  StepTweenSplitLinear = 25,
  StepTweenSplitFastProxy = 26,
  StepTweenSplitLinearCache = 27,
  StepTweenSplitHeadProxyAsync = 28,
  StepTweenSplitSparse = 29,
  Freeze = 30,
  Shadow = 31,
  Adversarial = 32,
  Memory = 33,
};

inline TrainMode resolve_train_mode(TrainMode m, TrainMode parent) {
  if (m == TrainMode::Inherit) {
    return parent == TrainMode::Inherit ? TrainMode::NormalBP : parent;
  }
  return m;
}

inline bool train_mode_is_frozen(TrainMode m) {
  return m == TrainMode::Freeze || m == TrainMode::Shadow;
}

inline bool train_mode_is_shadow(TrainMode m) { return m == TrainMode::Shadow; }

// Go parallel.TrainMode.Family — collapses Step/Mesh scheduling to Op update class.
enum class TrainFamily : std::uint8_t {
  BP = 0,
  Tween = 1,
  TweenChain = 2,
  TweenSplit = 3,
  TweenAlt = 4,
  Freeze = 5,
};

inline TrainFamily train_mode_family(TrainMode m) {
  switch (m) {
  case TrainMode::Freeze:
  case TrainMode::Shadow:
    return TrainFamily::Freeze;
  case TrainMode::Adversarial:
  case TrainMode::Memory:
    return TrainFamily::BP; // signs/gates applied in mixed train
  case TrainMode::Tween:
  case TrainMode::StepTween:
  case TrainMode::MeshTween:
    return TrainFamily::Tween;
  case TrainMode::TweenChain:
  case TrainMode::StepTweenChain:
  case TrainMode::MeshTweenChain:
    return TrainFamily::TweenChain;
  case TrainMode::TweenSplit:
  case TrainMode::StepTweenSplit:
  case TrainMode::MeshTweenSplit:
  case TrainMode::TweenSplitHeadProxy:
  case TrainMode::TweenSplitLinear:
  case TrainMode::TweenSplitFastProxy:
  case TrainMode::TweenSplitLinearCache:
  case TrainMode::TweenSplitHeadProxyAsync:
  case TrainMode::TweenSplitSparse:
  case TrainMode::MeshTweenSplitFastProxy:
  case TrainMode::MeshTweenSplitSparse:
  case TrainMode::StepTweenSplitHeadProxy:
  case TrainMode::StepTweenSplitLinear:
  case TrainMode::StepTweenSplitFastProxy:
  case TrainMode::StepTweenSplitLinearCache:
  case TrainMode::StepTweenSplitHeadProxyAsync:
  case TrainMode::StepTweenSplitSparse:
    return TrainFamily::TweenSplit;
  case TrainMode::TweenAlt:
  case TrainMode::StepTweenAlt:
  case TrainMode::MeshTweenAlt:
    return TrainFamily::TweenAlt;
  default:
    return TrainFamily::BP;
  }
}

} // namespace welvet
