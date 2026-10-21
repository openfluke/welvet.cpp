# welvet.cpp gap census

**Version `0.1.23-m2-layernorm-nbp`** — see main [README](README.md) scoreboard (**55 / 100** engine + **13 / 100** extensions = **68 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

```bash
./scripts/linux_build_w2a.sh              # full go_v_cpp
./scripts/linux_build_w2a.sh --dense-train
./scripts/linux_build_w2a.sh --rmsnorm-train
./scripts/linux_build_w2a.sh --layernorm-train
./scripts/linux_build_w2a.sh --parallel
./scripts/linux_build_w2a.sh --seqmix
./scripts/linux_build_w2a.sh --all

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M2.03 LayerNorm NormalBP)

| ID | Detail |
|---|---|
| Dense FormatNone×34 + SIMD | CPU + imported DotTile |
| **Dense NormalBP** | MatVecT + Backward + ApplySGD FormatNone×34; volume Step; w2a `WDB1`/`WVN1` |
| **RMSNorm NormalBP** | dγ/dx + ApplySGD FormatNone×34; volume Step; w2a `WRB1`/`WRV1` |
| **LayerNorm NormalBP** | dγ/dβ/dx + ApplySGD FormatNone×34; volume Step; w2a `WLB1`/`WLV1` |
| Transformer / composition | Sequential, Residual, seqmix Kind, **Parallel Dense×7 combines** |
| Extended | ConvT, KMeans, Mamba, Metacognition, GDN |
| CNN / RNN / LSTM | FormatNone×34 forwards |
| ABI wires | … / WDB1 / WVN1 / WRB1 / WRV1 / WLB1 / WLV1 |

## gap? — remaining

- Parallel poly/nested/cameral/train; Stack
- Other layers’ Backward; Dense quants; remaining TrainModes; systems; model/IO; vendor accel

See `src/gaps/registry.cpp`.

Training runway checklist (leaf NormalBP → TrainModes): [`docs/TRAINING_CHECKLIST.md`](docs/TRAINING_CHECKLIST.md).

Note: Go Parallel also has a large **w2a/suites/parallel** suite (cameral/poly/train). C++ M1 is Dense combine host via **go_v_cpp** `WPL1` only.
