# welvet.cpp gap census

**Version `0.1.19-m1-seqmix`** — see main [README](README.md) scoreboard (**47 / 100** engine + **13 / 100** extensions = **60 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

```bash
./scripts/linux_build_w2a.sh              # full go_v_cpp
./scripts/linux_build_w2a.sh --seqmix
./scripts/linux_build_w2a.sh --gdn
./scripts/linux_build_w2a.sh --all

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M1.17 seqmix)

| ID | Detail |
|---|---|
| Dense FormatNone×34 + SIMD | CPU + imported DotTile |
| Transformer slice | MHA, SwiGLU, norms, Softmax, Embedding, Sequential, Residual, **seqmix Kind** |
| Extended | ConvT1–3, KMeans, Mamba, Metacognition, GDN (Float32) |
| CNN / RNN / LSTM | FormatNone×34 forwards |
| ABI wires | … / WGD1 / WSX1 |

## gap? — remaining

- Stack, Parallel; KindConvMix; GDN×34 / BPTT; Mamba-N
- Backward for existing layers; Dense quants; training; systems; model/IO; vendor accel

See `src/gaps/registry.cpp`.
