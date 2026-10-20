# welvet.cpp gap census

**Version `0.1.20-m1-par`** — see main [README](README.md) scoreboard (**47 / 100** engine + **13 / 100** extensions = **60 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

```bash
./scripts/linux_build_w2a.sh              # full go_v_cpp
./scripts/linux_build_w2a.sh --parallel
./scripts/linux_build_w2a.sh --seqmix
./scripts/linux_build_w2a.sh --all

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M1.18 Parallel)

| ID | Detail |
|---|---|
| Dense FormatNone×34 + SIMD | CPU + imported DotTile |
| Transformer / composition | Sequential, Residual, seqmix Kind, **Parallel Dense×7 combines** |
| Extended | ConvT, KMeans, Mamba, Metacognition, GDN |
| CNN / RNN / LSTM | FormatNone×34 forwards |
| ABI wires | … / WSX1 / WPL1 |

## gap? — remaining

- Parallel poly/nested/cameral/train; Stack
- Backward; Dense quants; training; systems; model/IO; vendor accel

See `src/gaps/registry.cpp`.

Note: Go Parallel also has a large **w2a/suites/parallel** suite (cameral/poly/train). C++ M1 is Dense combine host via **go_v_cpp** `WPL1` only.
