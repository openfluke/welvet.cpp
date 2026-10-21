# welvet.cpp gap census

**Version `0.1.46-m2-tweensplitlinear`** — see main [README](README.md) scoreboard (**70 / 100** engine + **13 / 100** extensions = **83 / 200**).

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
./scripts/linux_build_w2a.sh --parallel
./scripts/linux_build_w2a.sh --seqmix
./scripts/linux_build_w2a.sh --all

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M2.26 TweenSplitLinear + TweenSplitHeadProxy + StepTweenAlt + TweenAlt + StepTweenSplit + TweenSplit + MeshTween* + Step*/ModeTween* + MeshBP + StepBP Line + NormalBP credit)

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
| **TweenSplit** | `systems/tween_split` 1/N projected gap @ full lr; TSB1 leaf ×34; TSP1/TRP1… volume; w2a `--tweensplit-train` |
| **StepTweenSplit** | Line FamilyTweenSplit 1/N @ full lr on pop; STS1 leaf ×34; 3D = MeshTweenSplit volume (= TweenSplit wires); w2a `--steptweensplit-train` |
| **TweenAlt** | Split→Tween (AltTimes=1); TAB1 leaf ×34; TAV1/TRA1…MBTA volume; w2a `--tweenalt-train` |
| **StepTweenAlt** | Line FamilyTweenAlt Split→Tween on pop; STA1 leaf ×34; 3D = MeshTweenAlt volume (= TweenAlt wires); w2a `--steptweenalt-train` |
| **TweenSplitHeadProxy** | Head full Jᵀgy (+ gx); hidden 1/(N−1) P(gx_head); HPB1 leaf ×34; HPV1/HRP1… volume; w2a `--tweensplitheadproxy-train` |
| **TweenSplitLinear** | Reverse walk 1/N P(gy); Dense gx=Wᵀgy (no act'); LLB1 leaf ×34; LLV1/LRP1… volume; w2a `--tweensplitlinear-train` |
| Transformer / composition | Sequential+Residual NormalBP, seqmix Kind, **Parallel Dense×7 combines** |
| Extended | ConvT, KMeans, Mamba NormalBP, Metacognition, GDN |
| CNN / RNN / LSTM | FormatNone×34 forwards |
| ABI wires | … / STW1 / STC1 / STS1 / STA1 / MTW1 / MTC1 / TSB1 / TSP1 / TAB1 / TAV1 / HPB1 / HPV1 / LLB1 / LLV1 |

## gap? — remaining

- Parallel poly/nested/cameral/train; full cameral Stack; MeshTweenSplit / MeshTweenAlt / FastProxy/Sparse/LinearCache Split variants
- Other layers’ Backward; Dense quants; remaining TrainModes; systems (DNA/evolution/TANHI/telemetry); model/IO; vendor accel

See `src/gaps/registry.cpp`.

Training runway checklist (leaf NormalBP → TrainModes): [`docs/TRAINING_CHECKLIST.md`](docs/TRAINING_CHECKLIST.md).

Note: Go Parallel also has a large **w2a/suites/parallel** suite (cameral/poly/train). C++ M1 is Dense combine host via **go_v_cpp** `WPL1` only.
