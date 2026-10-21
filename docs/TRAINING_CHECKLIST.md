# Training checklist — welvet.cpp

One **TrainMode** at a time. Finish that mode’s table (or the rows you care about), bump version, move on.

Canonical enum: Go `layers/parallel/train_mode.go` (`TrainMode`).  
`inherit` (0) = resolve-only — not a ship target.

### Columns (every table)

| Column | Means |
|---|---|
| **Inside node** | Layer trains correctly in a `1×1×1` cell (single-node / ordinary NN case). go_v_cpp / exact_bit (or declared band). |
| **3D volumetric** | Same mode works when that layer sits in a multi-cell Grid (volume Forward/Backward + credit across cells). |
| **Fully signed off** | **All** supported FormatNone dtypes for that layer under this mode (not just f32). Softmax = dX only. GDN = f32-only in Go → sign off when f32 path is done. |

Tick `[x]` when done. Put the **release tag** in **Ver** when that row (or the whole mode table) is signed.

**Current engine:** `0.1.47-m2-tweensplitfastproxy`

---

## w2a Go↔C++ (Dense NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --dense-train`  
Wire: `WDB1`/`WDBO` (leaf), `WVN1`/`WVNO` (volume). Exact_bit vs Go.  
Timing: warm-up + `--bench=N` (in-process; no spawn) → **goµs / cppµs / spd%Δ** (`(cpp−go)/go×100`, neg ⇒ C++ faster).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 ReLU (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔C++ (RMSNorm NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --rmsnorm-train`  
Wire: `WRB1`/`WRBO` (leaf), `WRV1`/`WRVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention as Dense.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dγ/γ/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔C++ (LayerNorm NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --layernorm-train`  
Wire: `WLB1`/`WLBO` (leaf), `WLV1`/`WLVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention as Dense/RMSNorm.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dγβ/γβ/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔C++ (SwiGLU NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --swiglu-train`  
Wire: `WSB1`/`WSBO` (leaf), `WSV1`/`WSVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔C++ (Softmax NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --softmax-train`  
Wire: `WXB1`/`WXBO` (leaf), `WXV1`/`WXVO` (volume). Exact_bit vs Go (Gumbel = smoke).  
Timing: same `--bench=N` / spd%Δ convention. Softmax = dX only (weightless).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Standard Float32 (dX/loss) | [x] | exact_bit + speed |
| 2 | Leaf all Kind Float32 (+ gumbel smoke) | [x] | exact_bit; gumbel non-det |
| 3 | Leaf Standard × all 34 dtypes | [x] | exact_bit + speed table |
| 4 | Leaf all Kind × all 34 dtypes | [x] | exact_bit |
| 5 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 6 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 7 | Volume 1×1×1×1 × all 34 dtypes | [x] | + spectrum |
| 8 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 9 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 |

## w2a Go↔C++ (Embedding NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --embedding-train`  
Wire: `WEB1`/`WEBO` (leaf), `WEV1`/`WEVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. dX zeros (discrete IDs); dW scatter.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + spectrum |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 |

## w2a Go↔C++ (MHA NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --mha-train`  
Wire: `WHB1`/`WHBO` (leaf), `WHV1`/`WHVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. ModeSelf baseline (no dropout/QKNorm/cross).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf DecoderCausal Float32 (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf GQA + Encoder Float32 | [x] | exact_bit |
| 3 | Leaf DecoderCausal × all 34 dtypes | [x] | exact_bit + speed table |
| 4 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 5 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 6 | Volume 1×1×1×1 × all 34 dtypes | [x] | + spectrum |
| 7 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 8 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 |

## w2a Go↔C++ (CNN1/2/3 NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --cnn-train`  
Wire: `C1B1`/`C1BO`, `C2B1`/`C2BO`, `C3B1`/`C3BO` (leaf); `C*V1`/`C*VO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. im2col → Dense bwd → col2im.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Leaf Float32 CNN1/2/3 | [x] | exact_bit + speed |
| 4–6 | Leaf ×34 CNN1/2/3 | [x] | exact_bit spectrum |
| 7–9 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving pad |
| 10–12 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔C++ (ConvT1/2/3 NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --convt-train`  
Wire: `T1B1`/`T1BO`, `T2B1`/`T2BO`, `T3B1`/`T3BO` (leaf); `T*V1`/`T*VO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. Scatter forward + act Jacobian + scatter adjoint (no bias).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Leaf Float32 ConvT1/2/3 | [x] | exact_bit + speed |
| 4–6 | Leaf ×34 ConvT1/2/3 | [x] | exact_bit spectrum |
| 7–9 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving pad |
| 10–12 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔C++ (RNN NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --rnn-train`  
Wire: `RNB1`/`RNBO` (leaf); `RNV1`/`RNVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. BPTT via Dense IH/HH + tanh′; loom `[dIH|dHH|dBias]`.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | in=hid chain |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔C++ (LSTM NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --lstm-train`  
Wire: `LSB1`/`LSBO` (leaf); `LSV1`/`LSVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. BPTT 4-gate Dense; loom `[i|f|g|o]` packs.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | in=hid chain |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔C++ (Sequential NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --sequential-train`  
Wire: `SQB1`/`SQBO` (leaf); `SQV1`/`SQVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. Reverse Dense Dim→Dim chain; loom Depth×[Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving Dim |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔C++ (Residual NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --residual-train`  
Wire: `REB1`/`REBO` (leaf); `REV1`/`REVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. F reverse Dense + skip grad; loom Depth×[Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | y=F(x)+x |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔C++ (Mamba NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --mamba-train`  
Wire: `MBB1`/`MBBO` (leaf); `MBV1`/`MBVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. Scan BPTT + In/Out Dense; loom In|Out|ALog|DSkip.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving D |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

---

## w2a Go↔C++ (StepBP Line + MeshBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --stepbp-train`  
Wires: `STB1`/`STBO` (Line StepBP); `MBP1`/`MBPO` (mesh-clock MeshBP FamilyBP). Exact_bit vs Go.  
Each Line ×34 kind prints **dtype / band / exact%W / goµs / cppµs / spd%Δ**.  
3D: Dense mesh-clock 1×1×1×1 / 2×1×1×1 / 2×2×2×1 + ModeMeshBP volume 2×2×2×1 for all NormalBP-signed kinds (Go cameral ModeMeshBP = volume Forward+Step). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Dense multi-hop Float32 (W/loss) | [x] | warm-up ticks until first pop |
| 2 | Dense 1-hop ×34 | [x] | spectrum + speed table |
| 3–4 | RMSNorm f32 + ×34 | [x] | full spectrum + speed |
| 5–6 | LayerNorm f32 + ×34 | [x] | full spectrum + speed |
| 7–8 | Softmax f32 + ×34 | [x] | full spectrum + speed |
| 9–10 | SwiGLU f32 + ×34 | [x] | full spectrum + speed |
| 11–12 | Sequential f32 + ×34 | [x] | full spectrum + speed |
| 13–14 | Residual f32 + ×34 | [x] | full spectrum + speed |
| 15–16 | Mamba f32 + ×34 | [x] | full spectrum + speed |
| 17–18 | Embedding f32 + ×34 | [x] | full spectrum + speed |
| 19–20 | MHA f32 + ×34 | [x] | full spectrum + speed |
| 21–22 | CNN1 f32 + ×34 | [x] | full spectrum + speed (CNN2/3 share) |
| 23–24 | ConvT1 f32 + ×34 | [x] | full spectrum + speed (ConvT2/3 share) |
| 25–26 | RNN f32 + ×34 | [x] | full spectrum + speed |
| 27–28 | LSTM f32 + ×34 | [x] | full spectrum + speed |
| 29–31 | Dense MeshBP mesh-clock 111/211/222 | [x] | MBP1 exact_bit + speed |
| 32–49 | MeshBP volume 2×2×2×1 all signed kinds | [x] | ModeMeshBP = volume Step |

## w2a Go↔C++ (Tween ModeTween) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --tween-train`  
Wires: `TWB1`/`TWBO` (Stack ModeTween leaf); `TWV1`/`TWVO` + `TRV1`…`MBTV` (volume reverse-hop projected local @ lr×0.5). Exact_bit vs Go.  
Each leaf ×34 kind prints **dtype / band / exact%W / goµs / cppµs / spd%Δ**.  
3D: Dense 111/211/222 + ModeTween volume 2×2×2×1 for all NormalBP-signed kinds. CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Dense multi-hop Float32 (W/loss) | [x] | ModeTween soft_lr=lr×0.5 |
| 2 | Dense 1-hop ×34 | [x] | spectrum + speed table |
| 3–4 | RMSNorm f32 + ×34 | [x] | full spectrum + speed |
| 5–6 | LayerNorm f32 + ×34 | [x] | full spectrum + speed |
| 7–8 | Softmax f32 + ×34 | [x] | full spectrum + speed |
| 9–10 | SwiGLU f32 + ×34 | [x] | full spectrum + speed |
| 11–12 | Sequential f32 + ×34 | [x] | leaf-projected (no chain) |
| 13–14 | Residual f32 + ×34 | [x] | leaf-projected (no chain) |
| 15–16 | Mamba f32 + ×34 | [x] | full spectrum + speed |
| 17–18 | Embedding f32 + ×34 | [x] | full spectrum + speed |
| 19–20 | MHA f32 + ×34 | [x] | full spectrum + speed |
| 21–22 | CNN1 f32 + ×34 | [x] | full spectrum + speed (CNN2/3 share) |
| 23–24 | ConvT1 f32 + ×34 | [x] | full spectrum + speed (ConvT2/3 share) |
| 25–26 | RNN f32 + ×34 | [x] | full spectrum + speed |
| 27–28 | LSTM f32 + ×34 | [x] | full spectrum + speed |
| 29–31 | Dense Tween volume 111/211/222 | [x] | TWV1 exact_bit + speed |
| 32–48 | ModeTween volume 2×2×2×1 all signed kinds | [x] | TRV1…MBTV / C1TV…; reverse-hop @ lr×0.5 |

## w2a Go↔C++ (TweenChain ModeTweenChain) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --tweenchain-train`  
Wires: `TCB1`/`TCBO` (Stack ModeTweenChain leaf = FamilyBP @ full lr); `TCV1`/`TCVO` + `TRC1`…`MBTC` (volume Step = NormalBP credit). Exact_bit vs Go.  
Each leaf ×34 kind prints **dtype / band / exact%W / goµs / cppµs / spd%Δ**.  
3D: Dense 111/211/222 + ModeTweenChain volume 2×2×2×1 for all NormalBP-signed kinds. CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Dense multi-hop Float32 (W/loss) | [x] | ModeTweenChain = BP @ full lr |
| 2 | Dense 1-hop ×34 | [x] | spectrum + speed table |
| 3–4 | RMSNorm f32 + ×34 | [x] | full spectrum + speed |
| 5–6 | LayerNorm f32 + ×34 | [x] | full spectrum + speed |
| 7–8 | Softmax f32 + ×34 | [x] | full spectrum + speed |
| 9–10 | SwiGLU f32 + ×34 | [x] | full spectrum + speed |
| 11–12 | Sequential f32 + ×34 | [x] | full chain FamilyBP |
| 13–14 | Residual f32 + ×34 | [x] | full chain FamilyBP |
| 15–16 | Mamba f32 + ×34 | [x] | full spectrum + speed |
| 17–18 | Embedding f32 + ×34 | [x] | full spectrum + speed |
| 19–20 | MHA f32 + ×34 | [x] | full spectrum + speed |
| 21–22 | CNN1 f32 + ×34 | [x] | full spectrum + speed (CNN2/3 share) |
| 23–24 | ConvT1 f32 + ×34 | [x] | full spectrum + speed (ConvT2/3 share) |
| 25–26 | RNN f32 + ×34 | [x] | full spectrum + speed |
| 27–28 | LSTM f32 + ×34 | [x] | full spectrum + speed |
| 29–31 | Dense TweenChain volume 111/211/222 | [x] | TCV1 exact_bit + speed |
| 32–48 | ModeTweenChain volume 2×2×2×1 all signed kinds | [x] | TRC1…MBTC / C1TC…; volume Step |

## w2a Go↔C++ (TweenSplit ModeTweenSplit) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --tweensplit-train`  
Wires: `TSB1`/`TSBO` (Stack 1/N @ full lr); `TSP1`/`TSPO` + `TRP1`…`MBTP` (volume leaf collect 1/N). Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–28 | Leaf kinds ×34 | [x] | FamilyTweenSplit OpenSplitTape twin |
| 29–48 | Volume 2×2×2×1 all signed kinds | [x] | TSP1/TRP1… collect leaves 1/N |

## w2a Go↔C++ (StepTweenSplit ModeStepTweenSplit) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --steptweensplit-train`  
Wires: `STS1`/`STSO` (Line FamilyTweenSplit 1/N @ full lr on pop); 3D via TweenSplit volume wires `TSP1`/`TRP1`… (MeshTweenSplit volume credit). Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–28 | Line leaf kinds ×34 | [x] | FamilyTweenSplit per hop |
| 29–48 | MeshTweenSplit volume 2×2×2×1 | [x] | = ModeTweenSplit volume wires |

## w2a Go↔C++ (TweenAlt ModeTweenAlt) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --tweenalt-train`  
Wires: `TAB1`/`TABO` (Split→Tween AltTimes=1); `TAV1`/`TRA1`…`MBTA` volume. Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–28 | Leaf kinds ×34 | [x] | Split @ full lr then Tween @ lr×0.5 |
| 29–48 | Volume 2×2×2×1 all signed kinds | [x] | re-forward between phases |

## w2a Go↔C++ (StepTweenAlt ModeStepTweenAlt) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --steptweenalt-train`  
Wires: `STA1`/`STAO` (Line FamilyTweenAlt Split→Tween on pop); 3D via TweenAlt volume wires `TAV1`/`TRA1`… (MeshTweenAlt volume credit). Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–28 | Line leaf kinds ×34 | [x] | trainTweenAltOp frozen gap per hop |
| 29–48 | MeshTweenAlt volume 2×2×2×1 | [x] | = ModeTweenAlt volume wires |

## w2a Go↔C++ (MeshTween mesh-clock) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --meshtween-train`  
Wires: `MTW1`/`MTWO` (mesh-clock FamilyTween @ lr×0.5); 3D via ModeTween volume `TWV1`/`TRV1`…. Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Dense MeshTween mesh-clock 111/211/222 | [x] | MTW1 exact_bit + speed |
| 4–23 | MeshTween volume 2×2×2×1 all signed kinds | [x] | = ModeTween volume wires |

## w2a Go↔C++ (MeshTweenChain mesh-clock) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --meshtweenchain-train`  
Wires: `MTC1`/`MTCO` (mesh-clock FamilyBP ≡ MeshBP); 3D via ModeTweenChain volume `TCV1`/`TRC1`…. Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Dense MeshTweenChain mesh-clock 111/211/222 | [x] | MTC1 exact_bit + speed |
| 4–23 | MeshTweenChain volume 2×2×2×1 all signed kinds | [x] | = ModeTweenChain volume wires |

## w2a Go↔C++ (StepTween ModeStepTween) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --steptween-train`  
Wires: `STW1`/`STWO` (Line FamilyTween on pop @ lr×0.5); 3D via ModeTween volume wires `TWV1`/`TRV1`… (MeshTween volume credit). Exact_bit vs Go.  
Each leaf ×34 kind prints **dtype / band / exact%W / goµs / cppµs / spd%Δ**.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Dense multi-hop Float32 (W/loss) | [x] | Line StepTween soft_lr=lr×0.5 |
| 2 | Dense 1-hop ×34 | [x] | spectrum + speed table |
| 3–28 | Other signed kinds f32 + ×34 | [x] | Seq/Res leaf-projected |
| 29–48 | MeshTween volume 2×2×2×1 all signed kinds | [x] | = ModeTween volume wires |

## w2a Go↔C++ (StepTweenChain ModeStepTweenChain) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --steptweenchain-train`  
Wires: `STC1`/`STCO` (Line FamilyBP on pop @ full lr); 3D via ModeTweenChain volume wires `TCV1`/`TRC1`…. Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–28 | Line StepTweenChain leaf kinds ×34 | [x] | FamilyTweenChain ≡ FamilyBP |
| 29–48 | MeshTweenChain volume 2×2×2×1 | [x] | = ModeTweenChain volume wires |

## 1 — NormalBP

Chain-rule Backward + ApplyGradSGD. Start here. Dense first.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.21-m2-dense-nbp` |
| RMSNorm | [x] | [x] | [x] | `0.1.22-m2-rmsnorm-nbp` |
| LayerNorm | [x] | [x] | [x] | `0.1.23-m2-layernorm-nbp` |
| SwiGLU | [x] | [x] | [x] | `0.1.24-m2-swiglu-nbp` |
| Softmax | [x] | [x] | [x] | `0.1.25-m2-softmax-nbp` |
| Embedding | [x] | [x] | [x] | `0.1.26-m2-embedding-nbp` |
| MHA | [x] | [x] | [x] | `0.1.27-m2-mha-nbp` |
| CNN1 | [x] | [x] | [x] | `0.1.28-m2-cnn-nbp` |
| CNN2 | [x] | [x] | [x] | `0.1.28-m2-cnn-nbp` |
| CNN3 | [x] | [x] | [x] | `0.1.28-m2-cnn-nbp` |
| ConvT1 | [x] | [x] | [x] | `0.1.29-m2-convt-nbp` |
| ConvT2 | [x] | [x] | [x] | `0.1.29-m2-convt-nbp` |
| ConvT3 | [x] | [x] | [x] | `0.1.29-m2-convt-nbp` |
| RNN | [x] | [x] | [x] | `0.1.30-m2-rnn-lstm-nbp` |
| LSTM | [x] | [x] | [x] | `0.1.30-m2-rnn-lstm-nbp` |
| Sequential | [x] | [x] | [x] | `0.1.31-m2-sequential-nbp` |
| Residual | [x] | [x] | [x] | `0.1.32-m2-residual-nbp` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.33-m2-mamba-nbp` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** when NormalBP rows you need are signed → bump version (e.g. `0.1.21-…`).

---

## 2 — StepBP

Needs Step `Line` pipe + FamilyBP (NormalBP credit) on pop.  
Ship: thin Stack host + `runtime/line` TickLine/TrainLineMSE; wire `STB1`/`STBO`; w2a `--stepbp-train`.  
**3D volumetric** for StepBP = MeshBP (`runtime/step` mesh-clock FamilyBP + ModeMeshBP volume Step). CNN2/3 and ConvT2/3 share CNN1/ConvT1 FamilyBP dispatch.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| RMSNorm | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| LayerNorm | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| SwiGLU | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| Softmax | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| Embedding | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| MHA | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| CNN1 | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| CNN2 | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| CNN3 | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| ConvT1 | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| ConvT2 | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| ConvT3 | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| RNN | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| LSTM | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| Sequential | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| Residual | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.35-m2-meshbp` — Line Inside+×34 + MeshBP 3D for NormalBP-signed kinds; full cameral Stack still open.

---

## 3 — Tween

ModeTween: projected local credit @ `soft_lr = lr × 0.5` (Go `applyProjectedTween` — same gap on every leaf, no chain rule).  
Ship: `systems/tween` + `runtime/tween_train`; wires `TWB1`/`TWBO` + `TWV1`/`TWVO` + per-kind `TRV1`…`MBTV`; w2a `--tween-train`.  
CNN2/3 and ConvT2/3 share CNN1/ConvT1 dispatch. 3D = reverse-hop volume 2×2×2×1 for all NormalBP-signed kinds.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.36-m2-tween` |
| RMSNorm | [x] | [x] | [x] | `0.1.36-m2-tween` |
| LayerNorm | [x] | [x] | [x] | `0.1.36-m2-tween` |
| SwiGLU | [x] | [x] | [x] | `0.1.36-m2-tween` |
| Softmax | [x] | [x] | [x] | `0.1.36-m2-tween` |
| Embedding | [x] | [x] | [x] | `0.1.36-m2-tween` |
| MHA | [x] | [x] | [x] | `0.1.36-m2-tween` |
| CNN1 | [x] | [x] | [x] | `0.1.36-m2-tween` |
| CNN2 | [x] | [x] | [x] | `0.1.36-m2-tween` |
| CNN3 | [x] | [x] | [x] | `0.1.36-m2-tween` |
| ConvT1 | [x] | [x] | [x] | `0.1.36-m2-tween` |
| ConvT2 | [x] | [x] | [x] | `0.1.36-m2-tween` |
| ConvT3 | [x] | [x] | [x] | `0.1.36-m2-tween` |
| RNN | [x] | [x] | [x] | `0.1.36-m2-tween` |
| LSTM | [x] | [x] | [x] | `0.1.36-m2-tween` |
| Sequential | [x] | [x] | [x] | `0.1.36-m2-tween` |
| Residual | [x] | [x] | [x] | `0.1.36-m2-tween` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.36-m2-tween` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.36-m2-tween` — ModeTween Inside+×34 + volume 2×2×2×1 for NormalBP-signed kinds; TweenChain / MeshTween still open.

---

## 4 — TweenChain

ModeTweenChain: named Tween, credit = BP (Go `familyTweenChain` ≡ `familyBP` @ full lr).  
Ship: `systems/tween_chain` + `runtime/tween_chain_train`; wires `TCB1`/`TCBO` + `TCV1`/`TCVO` + per-kind `TRC1`…`MBTC`; w2a `--tweenchain-train`.  
CNN2/3 and ConvT2/3 share CNN1/ConvT1 dispatch. 3D = volume Step 2×2×2×1 for all NormalBP-signed kinds.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| RMSNorm | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| LayerNorm | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| SwiGLU | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| Softmax | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| Embedding | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| MHA | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| CNN1 | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| CNN2 | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| CNN3 | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| ConvT1 | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| ConvT2 | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| ConvT3 | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| RNN | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| LSTM | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| Sequential | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| Residual | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.37-m2-tweenchain` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.37-m2-tweenchain` — ModeTweenChain Inside+×34 + volume 2×2×2×1 for NormalBP-signed kinds.

---

## 5 — StepTween

Needs Line + Tween.  
Ship: `runtime/line` Credit::FamilyTween on pop (`apply_projected` @ `lr×0.5`); wire `STW1`/`STWO`; w2a `--steptween-train`.  
**3D volumetric** for StepTween = MeshTween volume (= ModeTween reverse-hop projected local; existing `TWV1`/`TRV1`… wires). CNN2/3 and ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| RMSNorm | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| LayerNorm | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| SwiGLU | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| Softmax | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| Embedding | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| MHA | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| CNN1 | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| CNN2 | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| CNN3 | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| ConvT1 | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| ConvT2 | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| ConvT3 | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| RNN | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| LSTM | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| Sequential | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| Residual | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.38-m2-steptween` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.38-m2-steptween` — Line StepTween Inside+×34 + MeshTween volume 2×2×2×1 (ModeTween credit).

---

## 6 — StepTweenChain

Needs Line + TweenChain.  
Ship: `runtime/line` Credit::FamilyBP on pop (same credit as StepBP / TweenChain); wire `STC1`/`STCO`; w2a `--steptweenchain-train`.  
**3D volumetric** = MeshTweenChain volume (= ModeTweenChain volume Step wires `TCV1`/`TRC1`…).

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| RMSNorm | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| LayerNorm | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| SwiGLU | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| Softmax | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| Embedding | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| MHA | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| CNN1 | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| CNN2 | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| CNN3 | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| ConvT1 | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| ConvT2 | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| ConvT3 | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| RNN | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| LSTM | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| Sequential | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| Residual | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.39-m2-steptweenchain` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.39-m2-steptweenchain` — Line StepTweenChain Inside+×34 + MeshTweenChain volume 2×2×2×1.

---

## 7 — MeshBP

Grid tick FamilyBP: `runtime/step` Forward×ticks + Backward BPTT + ApplyGradSGD (`MBP1`/`MBPO` for Dense mesh-clock).  
Go ModeMeshBP on grid also = volume Forward+Step (cameral_train) — signed 2×2×2×1 for all NormalBP-signed kinds via `--stepbp-train`.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| RMSNorm | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| LayerNorm | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| SwiGLU | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| Softmax | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| Embedding | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| MHA | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| CNN1 | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| CNN2 | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| CNN3 | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| ConvT1 | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| ConvT2 | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| ConvT3 | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| RNN | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| LSTM | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| Sequential | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| Residual | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.35-m2-meshbp` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |


## 8 — MeshTween

Needs Mesh + Tween.  
Ship: `runtime/step` `Credit::FamilyTween` — Forward×ticks → last-tick `apply_cell_projected` @ `lr×0.5` (`MTW1`/`MTWO`); w2a `--meshtween-train`.  
**3D volumetric** = ModeTween volume credit (`TWV1`/`TRV1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| RMSNorm | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| LayerNorm | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| SwiGLU | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| Softmax | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| Embedding | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| MHA | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| CNN1 | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| CNN2 | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| CNN3 | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| ConvT1 | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| ConvT2 | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| ConvT3 | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| RNN | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| LSTM | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| Sequential | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| Residual | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.40-m2-meshtween` — Dense mesh-clock MTW1 111/211/222 + ModeTween volume for NormalBP-signed kinds.

---

## 9 — MeshTweenChain

Needs Mesh + TweenChain.  
Ship: `runtime/step` `Credit::FamilyBP` under mesh clock (`MTC1`/`MTCO` ≡ MeshBP); w2a `--meshtweenchain-train`.  
**3D volumetric** = ModeTweenChain volume credit (`TCV1`/`TRC1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| RMSNorm | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| LayerNorm | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| SwiGLU | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| Softmax | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| Embedding | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| MHA | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| CNN1 | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| CNN2 | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| CNN3 | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| ConvT1 | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| ConvT2 | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| ConvT3 | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| RNN | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| LSTM | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| Sequential | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| Residual | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.40-m2-meshtween` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.40-m2-meshtween` — Dense mesh-clock MTC1 (= MeshBP) + TweenChain volume for NormalBP-signed kinds.

---

## 10 — TweenSplit

Ship: `systems::tween_split` + `runtime/tween_split_train` — collect trainable leaves, apply \(g_i=\frac1N P(g_y)\) @ full lr (`TSB1`/`TSP1`/`TRP1`…); w2a `--tweensplit-train`.  
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path. Softmax weightless (N may be 0).

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| RMSNorm | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| LayerNorm | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| SwiGLU | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| Softmax | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| Embedding | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| MHA | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| CNN1 | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| CNN2 | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| CNN3 | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| ConvT1 | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| ConvT2 | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| ConvT3 | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| RNN | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| LSTM | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| Sequential | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| Residual | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.41-m2-tweensplit` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.41-m2-tweensplit` — ModeTweenSplit Inside+×34 + volume 2×2×2×1 for NormalBP-signed kinds.

---

## 11 — StepTweenSplit

Needs Line + TweenSplit.  
Ship: `runtime/line` Credit::FamilyTweenSplit on pop (`apply_split_child` 1/N @ full lr); wire `STS1`/`STSO`; w2a `--steptweensplit-train`.  
**3D volumetric** = MeshTweenSplit volume (= ModeTweenSplit / TweenSplit volume wires `TSP1`/`TRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| RMSNorm | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| LayerNorm | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| SwiGLU | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| Softmax | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| Embedding | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| MHA | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| CNN1 | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| CNN2 | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| CNN3 | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| ConvT1 | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| ConvT2 | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| ConvT3 | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| RNN | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| LSTM | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| Sequential | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| Residual | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.42-m2-steptweensplit` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.42-m2-steptweensplit` — Line StepTweenSplit Inside+×34 + MeshTweenSplit volume 2×2×2×1 (TweenSplit credit).

---

## 12 — TweenAlt

Ship: `systems::tween_alt` + `runtime/tween_alt_train` — AltTimes × (fwd→Split @ full lr → fwd→Tween @ lr×0.5); wire `TAB1`/`TAV1`/`TRA1`…; w2a `--tweenalt-train`.  
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path. Default AltTimes=1.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| RMSNorm | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| LayerNorm | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| SwiGLU | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| Softmax | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| Embedding | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| MHA | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| CNN1 | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| CNN2 | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| CNN3 | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| ConvT1 | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| ConvT2 | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| ConvT3 | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| RNN | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| LSTM | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| Sequential | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| Residual | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.43-m2-tweenalt` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.43-m2-tweenalt` — ModeTweenAlt Inside+×34 + volume 2×2×2×1 for NormalBP-signed kinds.

---

## 13 — StepTweenAlt

Needs Line + TweenAlt.  
Ship: `runtime/line` Credit::FamilyTweenAlt on pop (`apply_alt_child` Split→Tween frozen gap); wire `STA1`/`STAO`; w2a `--steptweenalt-train`.  
**3D volumetric** = MeshTweenAlt volume (= ModeTweenAlt / TweenAlt volume wires `TAV1`/`TRA1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| RMSNorm | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| LayerNorm | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| SwiGLU | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| Softmax | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| Embedding | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| MHA | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| CNN1 | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| CNN2 | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| CNN3 | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| ConvT1 | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| ConvT2 | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| ConvT3 | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| RNN | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| LSTM | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| Sequential | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| Residual | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.44-m2-steptweenalt` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.44-m2-steptweenalt` — Line StepTweenAlt Inside+×34 + MeshTweenAlt volume 2×2×2×1 (TweenAlt credit).

---

## 14 — TweenSplitHeadProxy

Ship: `systems::tween_split::apply_head_proxy` + `runtime/tween_split_head_proxy_train` — head full Jᵀg_y (+ gx); hidden 1/(N−1) P(gx_head) @ full lr; wire `HPB1`/`HPV1`/`HRP1`…; w2a `--tweensplitheadproxy-train`.  
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| RMSNorm | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| LayerNorm | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| SwiGLU | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| Softmax | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| Embedding | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| MHA | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| CNN1 | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| CNN2 | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| CNN3 | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| ConvT1 | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| ConvT2 | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| ConvT3 | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| RNN | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| LSTM | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| Sequential | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| Residual | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.45-m2-tweensplitheadproxy` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.45-m2-tweensplitheadproxy` — ModeTweenSplitHeadProxy Inside+×34 + volume 2×2×2×1 for NormalBP-signed kinds.

---

## 15 — TweenSplitLinear

Ship: `systems::tween_split::apply_linear` + `runtime/tween_split_linear_train` — reverse walk; each leaf 1/N P(gy); Dense gx = Wᵀgy (no act'); wire `LLB1`/`LLV1`/`LRP1`…; w2a `--tweensplitlinear-train`.  
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| RMSNorm | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| LayerNorm | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| SwiGLU | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| Softmax | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| Embedding | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| MHA | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| CNN1 | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| CNN2 | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| CNN3 | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| ConvT1 | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| ConvT2 | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| ConvT3 | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| RNN | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| LSTM | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| Sequential | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| Residual | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.46-m2-tweensplitlinear` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.46-m2-tweensplitlinear` — ModeTweenSplitLinear Inside+×34 + volume 2×2×2×1 for NormalBP-signed kinds.

---

## 16 — TweenSplitFastProxy

Ship: `systems::tween_split::apply_fast_proxy` + `runtime/tween_split_fast_proxy_train` — head GradWOnly on P(gy); proxy=Wᵀgy (no act′); hidden 1/(N−1) P(proxy); wire `FPB1`/`FPV1`/`FRP1`…; w2a `--tweensplitfastproxy-train`.  
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| RMSNorm | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| LayerNorm | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| SwiGLU | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| Softmax | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| Embedding | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| MHA | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| CNN1 | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| CNN2 | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| CNN3 | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| ConvT1 | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| ConvT2 | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| ConvT3 | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| RNN | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| LSTM | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| Sequential | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| Residual | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.47-m2-tweensplitfastproxy` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.47-m2-tweensplitfastproxy` — ModeTweenSplitFastProxy Inside+×34 + volume 2×2×2×1 for NormalBP-signed kinds.

---

## 17 — TweenSplitLinearCache

Ship: `systems::tween_split::apply_linear_cache` + `runtime/tween_split_linear_cache_train` — linear walk with `SplitAccel` cache; refresh every 20 (or `CacheEvery`); hit rescales cached P(gy) by live/cacheNorm then 1/N GradWOnly; wire `LCB1`/`LCV1`/`CRP1`…; w2a `--tweensplitlinearcache-train`.  
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| RMSNorm | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| LayerNorm | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| SwiGLU | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| Softmax | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| Embedding | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| MHA | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| CNN1 | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| CNN2 | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| CNN3 | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| ConvT1 | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| ConvT2 | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| ConvT3 | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| RNN | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| LSTM | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| Sequential | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| Residual | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.48-m2-tweensplitlinearcache` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.48-m2-tweensplitlinearcache` — ModeTweenSplitLinearCache Inside+×34 + volume 2×2×2×1 for NormalBP-signed kinds.

---

## 18 — TweenSplitHeadProxyAsync

Ship: `systems::tween_split::apply_head_proxy_async` + `runtime/tween_split_head_proxy_async_train` — if stale proxy → hidden 1/(N−1) P(stale) first; head GradWOnly on P(gy); store proxy=Wᵀgy (nil pre); if no stale, hidden on fresh proxy; wire `HAB1`/`HAV1`/`ARPA`…; w2a `--tweensplitheadproxyasync-train`.  
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| RMSNorm | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| LayerNorm | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| SwiGLU | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| Softmax | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| Embedding | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| MHA | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| CNN1 | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| CNN2 | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| CNN3 | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| ConvT1 | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| ConvT2 | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| ConvT3 | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| RNN | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| LSTM | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| Sequential | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| Residual | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.49-m2-tweensplitheadproxyasync` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.49-m2-tweensplitheadproxyasync` — ModeTweenSplitHeadProxyAsync Inside+×34 + volume 2×2×2×1 for NormalBP-signed kinds.

---

## 19 — TweenSplitSparse

Ship: `systems::tween_split::apply_sparse` + `runtime/tween_split_sparse_train` — proxy=Wᵀgy (nil pre); head GradWOnly on P(gy); one hidden at sparse%(N−1) GradWOnly on P(proxy) (no 1/(N−1)); wire `SPB1`/`SPV1`/`SRP1`…; w2a `--tweensplitsparse-train`.  
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| RMSNorm | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| LayerNorm | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| SwiGLU | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| Softmax | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| Embedding | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| MHA | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| CNN1 | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| CNN2 | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| CNN3 | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| ConvT1 | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| ConvT2 | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| ConvT3 | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| RNN | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| LSTM | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| Sequential | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| Residual | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.50-m2-tweensplitsparse` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.50-m2-tweensplitsparse` — ModeTweenSplitSparse Inside+×34 + volume 2×2×2×1 for NormalBP-signed kinds.

---

## 20 — MeshTweenSplit

Needs Mesh + TweenSplit.  
Ship: `runtime/step` `Credit::FamilyTweenSplit` — Forward×ticks → last-tick per-cell 1/N split @ full lr (`MTS1`/`MTSO`); w2a `--meshtweensplit-train`.  
**3D volumetric** = ModeTweenSplit / TweenSplit volume credit (`TSP1`/`TRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| RMSNorm | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| LayerNorm | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| SwiGLU | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Softmax | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Embedding | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| MHA | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| CNN1 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| CNN2 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| CNN3 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| ConvT1 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| ConvT2 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| ConvT3 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| RNN | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| LSTM | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Sequential | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Residual | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.51-m2-meshtweensplit-alt-fastproxy` — Dense mesh-clock MTS1 111/211/222 + TweenSplit volume for NormalBP-signed kinds.

---

## 21 — MeshTweenAlt

Needs Mesh + TweenAlt.  
Ship: `runtime/step` `Credit::FamilyTweenAlt` — Forward×ticks → last-tick Split→Tween (`MTA1`/`MTAO`); w2a `--meshtweenalt-train`.  
**3D volumetric** = ModeTweenAlt / TweenAlt volume credit (`TAV1`/`TRA1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| RMSNorm | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| LayerNorm | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| SwiGLU | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Softmax | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Embedding | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| MHA | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| CNN1 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| CNN2 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| CNN3 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| ConvT1 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| ConvT2 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| ConvT3 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| RNN | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| LSTM | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Sequential | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Residual | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.51-m2-meshtweensplit-alt-fastproxy` — Dense mesh-clock MTA1 111/211/222 + TweenAlt volume for NormalBP-signed kinds.

---

## 22 — MeshTweenSplitFastProxy

Needs Mesh + TweenSplitFastProxy.  
Ship: `runtime/step` `Credit::FamilyTweenSplitFastProxy` — Forward×ticks → last-tick FastProxy (`MTF1`/`MTFO`); w2a `--meshtweensplitfastproxy-train`.  
**3D volumetric** = ModeTweenSplitFastProxy / TweenSplitFastProxy volume credit (`FPV1`/`FRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| RMSNorm | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| LayerNorm | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| SwiGLU | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Softmax | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Embedding | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| MHA | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| CNN1 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| CNN2 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| CNN3 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| ConvT1 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| ConvT2 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| ConvT3 | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| RNN | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| LSTM | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Sequential | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Residual | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.51-m2-meshtweensplit-alt-fastproxy` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.51-m2-meshtweensplit-alt-fastproxy` — Dense mesh-clock MTF1 111/211/222 + FastProxy volume for NormalBP-signed kinds.

---

## 23 — MeshTweenSplitSparse

Needs Mesh + TweenSplitSparse.  
Ship: `runtime/step` `Credit::FamilyTweenSplitSparse` — Forward×ticks → last-tick Sparse (`MSS1`/`MSSO`); w2a `--meshtweensplitsparse-train`.  
**3D volumetric** = ModeTweenSplitSparse / TweenSplitSparse volume credit (`SPV1`/`SRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| RMSNorm | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| LayerNorm | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| SwiGLU | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| Softmax | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| Embedding | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| MHA | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| CNN1 | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| CNN2 | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| CNN3 | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| ConvT1 | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| ConvT2 | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| ConvT3 | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| RNN | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| LSTM | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| Sequential | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| Residual | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.52-m2-meshtweensplitsparse` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.52-m2-meshtweensplitsparse` — Dense mesh-clock MSS1 111/211/222 + TweenSplitSparse volume for NormalBP-signed kinds.

---

## 24 — StepTweenSplitHeadProxy

Needs Line + TweenSplitHeadProxy.  
Ship: `runtime/line` `Credit::FamilyTweenSplitHeadProxy` on pop (`apply_head_proxy_child`); wire `STH1`/`STHO`; w2a `--steptweensplitheadproxy-train`.  
**3D volumetric** = ModeTweenSplitHeadProxy / TweenSplitHeadProxy volume credit (`HPV1`/`HRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| RMSNorm | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| LayerNorm | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| SwiGLU | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| Softmax | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| Embedding | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| MHA | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| CNN1 | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| CNN2 | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| CNN3 | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| ConvT1 | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| ConvT2 | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| ConvT3 | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| RNN | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| LSTM | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| Sequential | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| Residual | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.53-m2-steptweensplitheadproxy` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.53-m2-steptweensplitheadproxy` — Line StepTweenSplitHeadProxy Inside+×34 + HeadProxy volume 2×2×2×1.

---

## 25 — StepTweenSplitLinear

Needs Line + TweenSplitLinear.  
Ship: `runtime/line` `Credit::FamilyTweenSplitLinear` on pop (`apply_linear_child`); wire `STL1`/`STLO`; w2a `--steptweensplitlinear-train`.  
**3D volumetric** = ModeTweenSplitLinear / TweenSplitLinear volume credit (`LLV1`/`LRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| RMSNorm | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| LayerNorm | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| SwiGLU | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| Softmax | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| Embedding | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| MHA | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| CNN1 | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| CNN2 | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| CNN3 | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| ConvT1 | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| ConvT2 | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| ConvT3 | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| RNN | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| LSTM | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| Sequential | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| Residual | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.54-m2-steptweensplitlinear` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.54-m2-steptweensplitlinear` — Line StepTweenSplitLinear Inside+×34 + Linear volume 2×2×2×1.

---

## 26 — StepTweenSplitFastProxy

Needs Line + TweenSplitFastProxy.  
Ship: `runtime/line` `Credit::FamilyTweenSplitFastProxy` on pop (`apply_fast_proxy_child`); wire `STF1`/`STFO`; w2a `--steptweensplitfastproxy-train`.  
**3D volumetric** = ModeTweenSplitFastProxy / TweenSplitFastProxy volume credit (`FPV1`/`FRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| RMSNorm | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| LayerNorm | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| SwiGLU | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| Softmax | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| Embedding | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| MHA | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| CNN1 | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| CNN2 | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| CNN3 | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| ConvT1 | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| ConvT2 | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| ConvT3 | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| RNN | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| LSTM | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| Sequential | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| Residual | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.55-m2-steptweensplitfastproxy` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.55-m2-steptweensplitfastproxy` — Line StepTweenSplitFastProxy Inside+×34 + FastProxy volume 2×2×2×1.

---

## 27 — StepTweenSplitLinearCache

Needs Line + TweenSplitLinearCache.  
Ship: `runtime/line` `Credit::FamilyTweenSplitLinearCache` on pop (`apply_linear_cache_child`, Dense hops accel=nil); wire `SLC1`/`SLCO`; w2a `--steptweensplitlinearcache-train`.  
**3D volumetric** = ModeTweenSplitLinearCache / TweenSplitLinearCache volume credit (`LCV1`/`CRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| RMSNorm | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| LayerNorm | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| SwiGLU | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| Softmax | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| Embedding | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| MHA | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| CNN1 | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| CNN2 | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| CNN3 | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| ConvT1 | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| ConvT2 | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| ConvT3 | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| RNN | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| LSTM | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| Sequential | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| Residual | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | `0.1.56-m2-steptweensplitlinearcache` |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** `0.1.56-m2-steptweensplitlinearcache` — Line StepTweenSplitLinearCache Inside+×34 + LinearCache volume 2×2×2×1.

---

## 28 — StepTweenSplitHeadProxyAsync

Needs Line + TweenSplitHeadProxyAsync.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** bump version when this mode table advances.

---

## 29 — StepTweenSplitSparse

Needs Line + TweenSplitSparse.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** bump version when this mode table advances.

---

## 30 — Freeze

No weight update; forward + zero apply.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** bump version when this mode table advances.

---

## 31 — Shadow

Needs Freeze + CamKit (frozen teacher + KD).

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** bump version when this mode table advances.

---

## 32 — Adversarial

Negated LR.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** bump version when this mode table advances.

---

## 33 — Memory

Needs CamKit (surprise gate).

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** bump version when this mode table advances.

---

## Version log

Fill as you tick rows / finish mode tables.

| Ver | Mode / layers | Notes |
|---|---|---|
| `0.1.20-m1-par` | — | Forward Parallel; train not started |
| `0.1.21-m2-dense-nbp` | NormalBP Dense | Leaf+vol w2a exact_bit ×34; MatVecT + ApplySGD FormatNone |
| `0.1.22-m2-rmsnorm-nbp` | NormalBP RMSNorm | Leaf+vol w2a exact_bit ×34; dγ/dx + ApplySGD FormatNone |
| `0.1.23-m2-layernorm-nbp` | NormalBP LayerNorm | Leaf+vol w2a exact_bit ×34; dγ/dβ/dx + ApplySGD FormatNone |
| `0.1.24-m2-swiglu-nbp` | NormalBP SwiGLU | Leaf+vol w2a exact_bit ×34; Gate/Up/Down + SiLU⊙ grads |
| `0.1.25-m2-softmax-nbp` | NormalBP Softmax | Leaf+vol w2a exact_bit all Kind×34; Jacobian dX (weightless); gumbel smoke |
| `0.1.26-m2-embedding-nbp` | NormalBP Embedding | Leaf+vol w2a exact_bit ×34; scatter dW (dX=0) + ApplySGD |
| `0.1.27-m2-mha-nbp` | NormalBP MHA | Leaf+vol w2a exact_bit ×34; ModeSelf attn+RoPEᵀ + Q/K/V/O |
| `0.1.28-m2-cnn-nbp` | NormalBP CNN1/2/3 | Leaf+vol w2a exact_bit ×34; im2col+Dense bwd+col2im |
| `0.1.36-m2-tween` | ModeTween × signed kinds | Leaf+vol projected local @ lr×0.5; TWB1/TWV1/TRV1… |
| `0.1.37-m2-tweenchain` | ModeTweenChain × signed kinds | Leaf+vol FamilyBP @ full lr; TCB1/TCV1/TRC1… |
| `0.1.38-m2-steptween` | StepTween × signed kinds | Line FamilyTween @ lr×0.5; STW1; 3D MeshTween vol (= ModeTween) |
| `0.1.39-m2-steptweenchain` | StepTweenChain × signed kinds | Line FamilyBP @ full lr; STC1; 3D MeshTweenChain vol (= TweenChain) |
| `0.1.40-m2-meshtween` | MeshTween + MeshTweenChain × signed kinds | MTW1 FamilyTween mesh-clock; MTC1 FamilyBP (= MeshBP); volume ModeTween/TweenChain |
| `0.1.41-m2-tweensplit` | TweenSplit × signed kinds | 1/N projected gap @ full lr; TSB1/TSP1/TRP1…; leaf ×34 + volume |
| `0.1.42-m2-steptweensplit` | StepTweenSplit × signed kinds | Line FamilyTweenSplit 1/N @ full lr; STS1; 3D MeshTweenSplit vol (= TweenSplit) |
| `0.1.43-m2-tweenalt` | TweenAlt × signed kinds | Split→Tween AltTimes=1; TAB1/TAV1/TRA1…; leaf ×34 + volume |
| `0.1.44-m2-steptweenalt` | StepTweenAlt × signed kinds | Line FamilyTweenAlt Split→Tween; STA1; 3D MeshTweenAlt vol (= TweenAlt) |
| `0.1.45-m2-tweensplitheadproxy` | TweenSplitHeadProxy × signed kinds | Head Jᵀgy + hidden 1/(N−1) P(gx); HPB1/HPV1/HRP1…; leaf ×34 + volume |
| `0.1.46-m2-tweensplitlinear` | TweenSplitLinear × signed kinds | Reverse walk 1/N P(gy); Dense gx=Wᵀgy no act'; LLB1/LLV1/LRP1…; leaf ×34 + volume |
| `0.1.47-m2-tweensplitfastproxy` | TweenSplitFastProxy × signed kinds | Head GradWOnly + hidden 1/(N−1) P(Wᵀgy); FPB1/FPV1/FRP1…; leaf ×34 + volume |
| `0.1.48-m2-tweensplitlinearcache` | TweenSplitLinearCache × signed kinds | Cached linear walk; refresh every 20; LCB1/LCV1/CRP1…; leaf ×34 + volume |
| `0.1.49-m2-tweensplitheadproxyasync` | TweenSplitHeadProxyAsync × signed kinds | Stale-proxy hidden; head GradWOnly; store Wᵀgy; HAB1/HAV1/ARPA…; leaf ×34 + volume |
| `0.1.50-m2-tweensplitsparse` | TweenSplitSparse × signed kinds | Head GradWOnly + one hidden P(Wᵀgy); SPB1/SPV1/SRP1…; leaf ×34 + volume |
| `0.1.51-m2-meshtweensplit-alt-fastproxy` | MeshTweenSplit + MeshTweenAlt + MeshTweenSplitFastProxy | MTS1/MTA1/MTF1 mesh-clock; volume = TweenSplit/Alt/FastProxy |
| `0.1.52-m2-meshtweensplitsparse` | MeshTweenSplitSparse × signed kinds | MSS1 mesh-clock Sparse; volume = TweenSplitSparse |
| `0.1.53-m2-steptweensplitheadproxy` | StepTweenSplitHeadProxy × signed kinds | Line HeadProxy on pop; STH1; 3D HeadProxy vol |
| `0.1.54-m2-steptweensplitlinear` | StepTweenSplitLinear × signed kinds | Line Linear on pop; STL1; 3D Linear vol |
| `0.1.55-m2-steptweensplitfastproxy` | StepTweenSplitFastProxy × signed kinds | Line FastProxy on pop; STF1; 3D FastProxy vol |
| `0.1.56-m2-steptweensplitlinearcache` | StepTweenSplitLinearCache × signed kinds | Line LinearCache on pop; SLC1; 3D LinearCache vol |
