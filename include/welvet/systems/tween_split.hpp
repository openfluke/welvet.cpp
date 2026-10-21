#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::systems::tween_split {

// ModeTweenSplit: one global gap, split 1/N across every trainable leaf
// (Go trainTweenSplitOp / OpenSplitTape). Full lr (not soft_lr). No chain rule.
Status apply_split(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                   double lr);

// One Op: collect leaves under this child, then 1/N (Go trainTweenSplitFamily).
// Used by Line StepTweenSplit (per-hop on pop).
Status apply_split_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                         double lr);

// Volume: collect leaves from every enabled cell on the forward tape, then 1/N.
Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss);

// ModeTweenSplitHeadProxy: head full Jᵀg_y (+ gx); hidden 1/(N-1) P(gx_head).
Status apply_head_proxy(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                        double lr);
Status apply_head_proxy_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                              double lr);
Status train_volume_mse_head_proxy(architecture::Grid& g, const runtime::forward::Result& fwd,
                                   const TensorF32& target, double lr, double* loss);

// ModeTweenSplitLinear: reverse walk; each leaf 1/N P(gy); Dense gx = Wᵀgy (no act').
Status apply_linear(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                    double lr);
Status apply_linear_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                          double lr);
Status train_volume_mse_linear(architecture::Grid& g, const runtime::forward::Result& fwd,
                               const TensorF32& target, double lr, double* loss);

// ModeTweenSplitFastProxy: head GradWOnly on P(gy); proxy=Wᵀgy (no act'); hidden 1/(N-1) P(proxy).
Status apply_fast_proxy(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                        double lr);
Status apply_fast_proxy_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                              double lr);
Status train_volume_mse_fast_proxy(architecture::Grid& g, const runtime::forward::Result& fwd,
                                   const TensorF32& target, double lr, double* loss);

// ModeTweenSplitLinearCache: linear walk; refresh every CacheEvery (default 20);
// hit path rescale cached P(gy) by live/cacheNorm then 1/N GradWOnly.
Status apply_linear_cache(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                          double lr);
Status apply_linear_cache_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                                double lr, stack::SplitAccel* accel);
Status train_volume_mse_linear_cache(architecture::Grid& g, const runtime::forward::Result& fwd,
                                     const TensorF32& target, double lr, double* loss);

// ModeTweenSplitHeadProxyAsync: if stale proxy → hidden 1/(N-1) P(stale) first;
// head GradWOnly on P(gy); store proxy=Wᵀgy (nil pre); if no stale, hidden on fresh proxy.
Status apply_head_proxy_async(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                              double lr);
Status apply_head_proxy_async_child(stack::Child& c, const TensorF32& grad_out,
                                    const TensorF32& input, double lr, stack::SplitAccel* accel);
Status train_volume_mse_head_proxy_async(architecture::Grid& g, const runtime::forward::Result& fwd,
                                         const TensorF32& target, double lr, double* loss);

// ModeTweenSplitSparse: proxy=Wᵀgy (nil pre); head GradWOnly on P(gy);
// one hidden leaf at sparse%(N-1) GradWOnly on P(proxy) (no 1/(N-1) scale).
Status apply_sparse(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input, double lr);
Status apply_sparse_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                          double lr, stack::SplitAccel* accel);
Status train_volume_mse_sparse(architecture::Grid& g, const runtime::forward::Result& fwd,
                               const TensorF32& target, double lr, double* loss);

} // namespace welvet::systems::tween_split
