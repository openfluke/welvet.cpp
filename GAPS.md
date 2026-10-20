# welvet.cpp gap census

**Version `0.1.17-m1-meta`** — see main [README](README.md) scoreboard (**46 / 100** engine + **13 / 100** extensions = **59 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

```bash
./scripts/linux_build_w2a.sh              # full go_v_cpp
./scripts/linux_build_w2a.sh --metacognition
./scripts/linux_build_w2a.sh --mamba
./scripts/linux_build_w2a.sh --all

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M1.15 Metacognition)

| ID | Detail |
|---|---|
| Dense FormatNone×34 + SIMD | CPU + imported DotTile |
| Transformer slice | MHA, SwiGLU, norms, Softmax, Embedding, Sequential, Residual |
| Extended | ConvT1–3, KMeans, Mamba, Metacognition |
| CNN / RNN / LSTM | FormatNone×34 forwards |
| ABI wires | … / WMB1 / WMC1 |

## gap? — remaining

- Metacognition custom rules wire / multi-step Identity morph; Mamba-N
- Stack, Parallel, GDN, remaining kinds
- Backward for existing layers; Dense quants; training; systems; model/IO; vendor accel

See `src/gaps/registry.cpp`.
