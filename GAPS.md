# welvet.cpp gap census

**Version `0.1.13-m1-kmeans`** — see main [README](README.md) scoreboard (**43 / 100** engine + **13 / 100** extensions = **56 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

```bash
./scripts/linux_build_w2a.sh              # full go_v_cpp
./scripts/linux_build_w2a.sh --kmeans
./scripts/linux_build_w2a.sh --softmax
./scripts/linux_build_w2a.sh --all

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M1.11 KMeans)

| ID | Detail |
|---|---|
| Dense FormatNone×34 + SIMD | CPU + imported DotTile |
| Transformer slice | MHA, SwiGLU, RMSNorm, LayerNorm, Softmax (all Kind), Embedding |
| `layer.cnn1/2/3` / `layer.convt1/2/3` | im2col / scatter FormatNone×34 |
| `layer.rnn` / `layer.lstm` | FormatNone×34 |
| `layer.embedding` | gather FormatNone×34 |
| `layer.kmeans` | soft assignment probs\|features FormatNone×34 |
| ABI wires | … / WSM1 / WKM1 |

## gap? — remaining

- Backward for Softmax/KMeans/CNN/ConvT/RNN/LSTM/Embedding/MHA/…
- Compose layers, GDN, Mamba, remaining kinds
- Dense quants; training; systems; model/IO; vendor accel

See `src/gaps/registry.cpp`.
