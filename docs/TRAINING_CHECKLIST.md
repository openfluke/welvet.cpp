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

**Current engine:** `0.1.32-m2-residual-nbp`

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

---

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
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** when NormalBP rows you need are signed → bump version (e.g. `0.1.21-…`).

---

## 2 — StepBP

Needs Step `Line` pipe + NormalBP on the leaf.

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

## 3 — Tween

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

## 4 — TweenChain

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

## 5 — StepTween

Needs Line + Tween.

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

## 6 — StepTweenChain

Needs Line + TweenChain.

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

## 7 — MeshBP

Needs Mesh/Grid schedule + NormalBP.

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

## 8 — MeshTween

Needs Mesh + Tween.

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
| | | |
