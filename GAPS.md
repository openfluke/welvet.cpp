# welvet.cpp gap census

**Version `0.1.47-m2-tweensplitfastproxy`** — see main [README](README.md) scoreboard (**70 / 100** engine + **13 / 100** extensions = **83 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

```bash
./scripts/linux_build_w2a.sh              # full go_v_cpp
./scripts/linux_build_w2a.sh --dense-train
./scripts/linux_build_w2a.sh --rmsnorm-train
./scripts/linux_build_w2a.sh --layernorm-train
./scripts/linux_build_w2a.sh --swiglu-train
./scripts/linux_build_w2a.sh --softmax-train
./scripts/linux_build_w2a.sh --embedding-train
./scripts/linux_build_w2a.sh --mha-train
./scripts/linux_build_w2a.sh --cnn-train
./scripts/linux_build_w2a.sh --convt-train
./scripts/linux_build_w2a.sh --rnn-train
./scripts/linux_build_w2a.sh --lstm-train
./scripts/linux_build_w2a.sh --sequential-train
./scripts/linux_build_w2a.sh --residual-train
./scripts/linux_build_w2a.sh --mamba-train
./scripts/linux_build_w2a.sh --stepbp-train
./scripts/linux_build_w2a.sh --tween-train
./scripts/linux_build_w2a.sh --tweenchain-train
./scripts/linux_build_w2a.sh --steptween-train
./scripts/linux_build_w2a.sh --steptweenchain-train
./scripts/linux_build_w2a.sh --meshtween-train
./scripts/linux_build_w2a.sh --meshtweenchain-train
./scripts/linux_build_w2a.sh --tweensplit-train
./scripts/linux_build_w2a.sh --steptweensplit-train
./scripts/linux_build_w2a.sh --tweenalt-train
./scripts/linux_build_w2a.sh --steptweenalt-train
./scripts/linux_build_w2a.sh --tweensplitheadproxy-train
./scripts/linux_build_w2a.sh --tweensplitlinear-train
./scripts/linux_build_w2a.sh --tweensplitfastproxy-train
./scripts/linux_build_w2a.sh --parallel
./scripts/linux_build_w2a.sh --seqmix
./scripts/linux_build_w2a.sh --all

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M2.39 Freeze + M2.38 StepTweenSplitSparse + StepTweenSplitHeadProxyAsync + StepTweenSplitLinearCache + StepTweenSplitFastProxy + StepTweenSplitLinear + StepTweenSplitHeadProxy + MeshTweenSplitSparse + MeshTweenSplit/Alt/SplitFastProxy + TweenSplitSparse + TweenSplitHeadProxyAsync + TweenSplitLinearCache + TweenSplitFastProxy + TweenSplitLinear + TweenSplitHeadProxy + StepTweenAlt + TweenAlt + StepTweenSplit + TweenSplit + MeshTween* + Step*/ModeTween* + MeshBP + StepBP Line + NormalBP credit)

| ID | Detail |
|---|---|
| Dense FormatNone×34 + SIMD | CPU + imported DotTile |
| **Dense NormalBP** | MatVecT + Backward + ApplySGD FormatNone×34; volume Step; w2a `WDB1`/`WVN1` |
| **RMSNorm NormalBP** | dγ/dx + ApplySGD FormatNone×34; volume Step; w2a `WRB1`/`WRV1` |
| **LayerNorm NormalBP** | dγ/dβ/dx + ApplySGD FormatNone×34; volume Step; w2a `WLB1`/`WLV1` |
| **SwiGLU NormalBP** | Gate/Up/Down + SiLU⊙ grads FormatNone×34; volume Step; w2a `WSB1`/`WSV1` |
| **Softmax NormalBP** | Jacobian dX all Kind (weightless); volume Step; w2a `WXB1`/`WXV1` |
| **Embedding NormalBP** | Scatter dW (dX=0); ApplySGD FormatNone×34; volume Step; w2a `WEB1`/`WEV1` |
| **MHA NormalBP** | ModeSelf attn+RoPEᵀ + Q/K/V/O grads FormatNone×34; volume Step; w2a `WHB1`/`WHV1` |
| **CNN1/2/3 NormalBP** | im2col+Dense bwd+col2im FormatNone×34; volume Step; w2a `C1/C2/C3` B/V |
| **ConvT1/2/3 NormalBP** | scatter adjoint + ApplySGD FormatNone×34; volume Step; w2a `T1/T2/T3` B/V |
| **RNN NormalBP** | BPTT Dense IH/HH + tanh′ + bias SGD FormatNone×34; volume Step; w2a `RNB1`/`RNV1` |
| **LSTM NormalBP** | BPTT 4-gate Dense + bias SGD FormatNone×34; volume Step; w2a `LSB1`/`LSV1` |
| **Sequential NormalBP** | Reverse Dense Dim→Dim chain + ApplySGD FormatNone×34; volume Step; w2a `SQB1`/`SQV1` |
| **Residual NormalBP** | F reverse Dense + skip grad + ApplySGD FormatNone×34; volume Step; w2a `REB1`/`REV1` |
| **Mamba NormalBP** | Scan BPTT + In/Out Dense + ALog/DSkip SGD FormatNone×34; volume Step; w2a `MBB1`/`MBV1` |
| **StepBP (Line)** | Thin Stack + `runtime/line` TickLine/TrainLineMSE FamilyBP on pop; STB1/STBO; w2a `--stepbp-train` ×34 full spectrum tables |
| **MeshBP** | `runtime/step` Forward+Backward+SGD; MBP1/MBPO Dense mesh-clock; ModeMeshBP volume 2×2×2×1 for NormalBP-signed kinds |
| **ModeTween** | `systems/tween` + `runtime/tween_train` projected local @ lr×0.5; TWB1 leaf ×34; TWV1 + TRV1…MBTV volume 2×2×2×1 all signed kinds; w2a `--tween-train` |
| **ModeTweenChain** | `systems/tween_chain` + `runtime/tween_chain_train` FamilyBP @ full lr; TCB1 leaf ×34; TCV1 + TRC1…MBTC volume Step 2×2×2×1; w2a `--tweenchain-train` |
| **StepTween** | Line FamilyTween on pop @ lr×0.5; STW1 leaf ×34; 3D = MeshTween volume (= ModeTween volume wires); w2a `--steptween-train` |
| **StepTweenChain** | Line FamilyBP on pop @ full lr; STC1 leaf ×34; 3D = MeshTweenChain volume (= ModeTweenChain wires); w2a `--steptweenchain-train` |
| **MeshTween** | `runtime/step` Credit::FamilyTween mesh-clock @ lr×0.5; MTW1 Dense 111/211/222; volume ModeTween; w2a `--meshtween-train` |
| **MeshTweenChain** | `runtime/step` Credit::FamilyBP mesh-clock (= MeshBP); MTC1 Dense 111/211/222; volume TweenChain; w2a `--meshtweenchain-train` |
| **MeshTweenSplit** | `runtime/step` Credit::FamilyTweenSplit mesh-clock; MTS1 Dense 111/211/222; volume TweenSplit; w2a `--meshtweensplit-train` |
| **MeshTweenAlt** | `runtime/step` Credit::FamilyTweenAlt mesh-clock; MTA1 Dense 111/211/222; volume TweenAlt; w2a `--meshtweenalt-train` |
| **MeshTweenSplitFastProxy** | `runtime/step` Credit::FamilyTweenSplitFastProxy mesh-clock; MTF1 Dense 111/211/222; volume FastProxy; w2a `--meshtweensplitfastproxy-train` |
| **MeshTweenSplitSparse** | `runtime/step` Credit::FamilyTweenSplitSparse mesh-clock; MSS1 Dense 111/211/222; volume Sparse; w2a `--meshtweensplitsparse-train` |
| **TweenSplit** | `systems/tween_split` 1/N projected gap @ full lr; TSB1 leaf ×34; TSP1/TRP1… volume; w2a `--tweensplit-train` |
| **StepTweenSplit** | Line FamilyTweenSplit 1/N @ full lr on pop; STS1 leaf ×34; 3D = MeshTweenSplit volume (= TweenSplit wires); w2a `--steptweensplit-train` |
| **StepTweenSplitHeadProxy** | Line FamilyTweenSplitHeadProxy on pop; STH1 leaf ×34; 3D = HeadProxy volume; w2a `--steptweensplitheadproxy-train` |
| **StepTweenSplitLinear** | Line FamilyTweenSplitLinear on pop; STL1 leaf ×34; 3D = Linear volume; w2a `--steptweensplitlinear-train` |
| **StepTweenSplitFastProxy** | Line FamilyTweenSplitFastProxy on pop; STF1 leaf ×34; 3D = FastProxy volume; w2a `--steptweensplitfastproxy-train` |
| **StepTweenSplitLinearCache** | Line FamilyTweenSplitLinearCache on pop; SLC1 leaf ×34; 3D = LinearCache volume; w2a `--steptweensplitlinearcache-train` |
| **StepTweenSplitHeadProxyAsync** | Line FamilyTweenSplitHeadProxyAsync on pop; SHA1 leaf ×34; 3D = HeadProxyAsync volume; w2a `--steptweensplitheadproxyasync-train` |
| **StepTweenSplitSparse** | Line FamilyTweenSplitSparse on pop; SSP1 leaf ×34; 3D = Sparse volume; w2a `--steptweensplitsparse-train` |
| **TweenAlt** | Split→Tween (AltTimes=1); TAB1 leaf ×34; TAV1/TRA1…MBTA volume; w2a `--tweenalt-train` |
| **StepTweenAlt** | Line FamilyTweenAlt Split→Tween on pop; STA1 leaf ×34; 3D = MeshTweenAlt volume (= TweenAlt wires); w2a `--steptweenalt-train` |
| **TweenSplitHeadProxy** | Head full Jᵀgy (+ gx); hidden 1/(N−1) P(gx_head); HPB1 leaf ×34; HPV1/HRP1… volume; w2a `--tweensplitheadproxy-train` |
| **TweenSplitLinear** | Reverse walk 1/N P(gy); Dense gx=Wᵀgy (no act'); LLB1 leaf ×34; LLV1/LRP1… volume; w2a `--tweensplitlinear-train` |
| **TweenSplitFastProxy** | Head GradWOnly on P(gy); proxy=Wᵀgy; hidden 1/(N−1) P(proxy); FPB1 leaf ×34; FPV1/FRP1… volume; w2a `--tweensplitfastproxy-train` |
| **TweenSplitLinearCache** | Cached linear walk; refresh every 20; hit rescale cache×live/norm; LCB1 leaf ×34; LCV1/CRP1… volume; w2a `--tweensplitlinearcache-train` |
| **TweenSplitHeadProxyAsync** | Stale-proxy hidden first; head GradWOnly; store Wᵀgy; HAB1 leaf ×34; HAV1/ARPA… volume; w2a `--tweensplitheadproxyasync-train` |
| **TweenSplitSparse** | Head GradWOnly; one hidden at sparse%(N−1) on P(Wᵀgy); SPB1 leaf ×34; SPV1/SRP1… volume; w2a `--tweensplitsparse-train` |
| **Freeze** | Fwd+mse; no ApplyGrad; FRB1 leaf ×34; FRV1/FRR1… volume; w2a `--freeze-train` |
| Transformer / composition | Sequential+Residual NormalBP, seqmix Kind, **Parallel Dense×7 combines** |
| Extended | ConvT, KMeans, Mamba NormalBP, Metacognition, GDN |
| CNN / RNN / LSTM | FormatNone×34 forwards |
| ABI wires | … / MSS1 / STH1 / STL1 / STF1 / SLC1 / SHA1 / SSP1 / FRB1 |

## gap? — remaining

- Parallel poly/nested/cameral/train; full cameral Stack
- Other layers’ Backward; Dense quants; remaining TrainModes; systems (DNA/evolution/TANHI/telemetry); model/IO; vendor accel

See `src/gaps/registry.cpp`.

Training runway checklist (leaf NormalBP → TrainModes): [`docs/TRAINING_CHECKLIST.md`](docs/TRAINING_CHECKLIST.md).

Note: Go Parallel also has a large **w2a/suites/parallel** suite (cameral/poly/train). C++ M1 is Dense combine host via **go_v_cpp** `WPL1` only.
