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

**Current engine:** `0.1.39-m2-steptweenchain`

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
Note: volume Tween credit already exact_bit via ModeTween / StepTween suites (`TWV1`…). Mesh-clock Inside (MBP1-style with Tween) still open.

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

## 9 — MeshTweenChain

Needs Mesh + TweenChain.  
Note: volume TweenChain credit already exact_bit via ModeTweenChain / StepTweenChain suites (`TCV1`…). Mesh-clock Inside still open.

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

## 10 — TweenSplit

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

## 11 — StepTweenSplit

Needs Line + TweenSplit.

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

## 12 — TweenAlt

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

## 13 — StepTweenAlt

Needs Line + TweenAlt.

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

## 14 — TweenSplitHeadProxy

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

## 15 — TweenSplitLinear

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

## 16 — TweenSplitFastProxy

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

## 17 — TweenSplitLinearCache

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

## 18 — TweenSplitHeadProxyAsync

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

## 19 — TweenSplitSparse

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

## 20 — MeshTweenSplit

Needs Mesh + TweenSplit.

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

## 21 — MeshTweenAlt

Needs Mesh + TweenAlt.

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

## 22 — MeshTweenSplitFastProxy

Needs Mesh + TweenSplitFastProxy.

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

## 23 — MeshTweenSplitSparse

Needs Mesh + TweenSplitSparse.

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

## 24 — StepTweenSplitHeadProxy

Needs Line + TweenSplitHeadProxy.

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

## 25 — StepTweenSplitLinear

Needs Line + TweenSplitLinear.

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

## 26 — StepTweenSplitFastProxy

Needs Line + TweenSplitFastProxy.

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

## 27 — StepTweenSplitLinearCache

Needs Line + TweenSplitLinearCache.

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
