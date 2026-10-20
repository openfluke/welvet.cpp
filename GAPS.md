# welvet.cpp gap census

**Version `0.1.14-m1-seq`** — see main [README](README.md) scoreboard (**44 / 100** engine + **13 / 100** extensions = **57 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

```bash
./scripts/linux_build_w2a.sh              # full go_v_cpp
./scripts/linux_build_w2a.sh --sequential
./scripts/linux_build_w2a.sh --kmeans
./scripts/linux_build_w2a.sh --all

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M1.12 Sequential)

| ID | Detail |
|---|---|
| Dense FormatNone×34 + SIMD | CPU + imported DotTile |
| Transformer slice | MHA, SwiGLU, norms, Softmax, Embedding, Sequential Dense chain |
| CNN / ConvT / RNN / LSTM / KMeans | FormatNone×34 forwards |
| ABI wires | … / WKM1 / WSQ1 |

## gap? — remaining

- Sequential NewFromOps (mixed SwiGLU/RMSNorm/LayerNorm); Sequential backward
- Residual, Stack, Parallel, GDN, Mamba, remaining kinds
- Backward for existing layers; Dense quants; training; systems; model/IO; vendor accel

See `src/gaps/registry.cpp`.
