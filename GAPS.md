# welvet.cpp gap census

**Version `0.1.8-m1-cnn`** — see main [README](README.md) scoreboard (**35 / 100** engine + **12 / 100** extensions = **47 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

```bash
./scripts/linux_build_w2a.sh              # full go_v_cpp
./scripts/linux_build_w2a.sh --cnn        # CNN1/2/3 only
./scripts/linux_build_w2a.sh --mha
./scripts/linux_build_w2a.sh --swiglu
./scripts/linux_build_w2a.sh --rmsnorm
./scripts/linux_build_w2a.sh --layernorm
./scripts/linux_build_w2a.sh --all

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M1.6 CNN1/2/3)

| ID | Detail |
|---|---|
| Dense FormatNone×34 + SIMD | CPU + imported DotTile |
| `arch.volume` + Dense volume Forward | Grid/Cell/HopOrder/remote + 3×3×3 |
| Transformer slice | MHA ModeSelf, SwiGLU, RMSNorm, LayerNorm |
| `layer.cnn1/2/3` | im2col + Dense forward FormatNone×34 |
| ABI wires | WVM1 / WVS1 / WVR1 / WVL1 / WVC1/2/3 |

## gap? — remaining

- CNN / MHA / SwiGLU / norm backward
- RNN, LSTM, remaining layer kinds
- Dense quants; training; systems; model/IO; vendor accel

See `src/gaps/registry.cpp`.
