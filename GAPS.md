# welvet.cpp gap census

**Version `0.1.11-m1-emb`** — see main [README](README.md) scoreboard (**41 / 100** engine + **13 / 100** extensions = **54 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

```bash
./scripts/linux_build_w2a.sh              # full go_v_cpp
./scripts/linux_build_w2a.sh --cnn
./scripts/linux_build_w2a.sh --convt
./scripts/linux_build_w2a.sh --rnn
./scripts/linux_build_w2a.sh --lstm
./scripts/linux_build_w2a.sh --embedding
./scripts/linux_build_w2a.sh --mha
./scripts/linux_build_w2a.sh --all

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M1.9 Embedding)

| ID | Detail |
|---|---|
| Dense FormatNone×34 + SIMD | CPU + imported DotTile |
| Transformer slice | MHA ModeSelf, SwiGLU, RMSNorm, LayerNorm, Embedding |
| `layer.cnn1/2/3` | im2col + Dense forward FormatNone×34 |
| `layer.convt1/2/3` | scatter forward FormatNone×34 |
| `layer.rnn` / `layer.lstm` | IH/HH + tanh / gates i,f,g,o FormatNone×34 |
| `layer.embedding` | vocab×emb gather FormatNone×34 |
| ABI wires | WVM1 / WVS1 / WVR1 / WVL1 / WVC1/2/3 / WVT1/2/3 / WRN1 / WLS1 / WEM1 |

## gap? — remaining

- Backward for CNN/ConvT/RNN/LSTM/Embedding/MHA/SwiGLU/norms
- Softmax, compose layers, remaining kinds
- Dense quants; training; systems; model/IO; vendor accel

See `src/gaps/registry.cpp`.
