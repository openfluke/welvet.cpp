# welvet.cpp gap census

**Version `0.1.16-m1-mamba`** — see main [README](README.md) scoreboard (**45 / 100** engine + **13 / 100** extensions = **58 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

```bash
./scripts/linux_build_w2a.sh              # full go_v_cpp
./scripts/linux_build_w2a.sh --mamba
./scripts/linux_build_w2a.sh --residual
./scripts/linux_build_w2a.sh --all

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M1.14 Mamba)

| ID | Detail |
|---|---|
| Dense FormatNone×34 + SIMD | CPU + imported DotTile |
| Transformer slice | MHA, SwiGLU, norms, Softmax, Embedding, Sequential, Residual |
| Extended | ConvT1–3, KMeans, Mamba minimal SSM |
| CNN / RNN / LSTM | FormatNone×34 forwards |
| ABI wires | … / WKM1 / WSQ1 / WRE1 / WMB1 |

## gap? — remaining

- Mamba-N (DState/B/C/conv/SiLU); Sequential/Residual NewFromOps + backward
- Stack, Parallel, GDN, Metacognition, remaining kinds
- Backward for existing layers; Dense quants; training; systems; model/IO; vendor accel

See `src/gaps/registry.cpp`.
