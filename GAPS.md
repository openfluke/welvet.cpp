# welvet.cpp gap census

**Version `0.1.7-m1-layernorm`** — see main [README](README.md) scoreboard (**32 / 100** engine + **11 / 100** extensions = **43 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

```bash
# one-shot: Linux build + w2a Go↔C++ suite
./scripts/linux_build_w2a.sh              # full go_v_cpp
./scripts/linux_build_w2a.sh --mha
./scripts/linux_build_w2a.sh --swiglu
./scripts/linux_build_w2a.sh --rmsnorm
./scripts/linux_build_w2a.sh --layernorm
./scripts/linux_build_w2a.sh --all

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M1.5 LayerNorm)

| ID | Detail |
|---|---|
| Dense FormatNone×34 + SIMD | CPU + imported DotTile |
| `arch.volume` + Dense volume Forward | Grid/Cell/HopOrder/remote + 3×3×3 |
| `layer.mha` | ModeSelf forward FormatNone×34 |
| `layer.swiglu` | Gate/Up/Down + SiLU⊙ FormatNone×34 |
| `layer.rmsnorm` | x̂ + γ⊙ FormatNone×34 |
| `layer.layernorm` | x̂ + γ⊙+β FormatNone×34 |
| ABI wires | WVM1 / WVS1 / WVR1 / WVL1 |

## gap? — remaining

- MHA cross-attn, backward, dropout, QKNorm
- SwiGLU / RMSNorm / LayerNorm backward
- Other 18 layer kinds; Dense backward/quants; training; systems; model/IO; vendor accel

See `src/gaps/registry.cpp`.
