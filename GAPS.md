# welvet.cpp gap census

**Version `0.1.6-m1-rmsnorm`** — see main [README](README.md) scoreboard (**30 / 100** engine + **10 / 100** extensions = **40 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

```bash
# one-shot: Linux build + w2a Go↔C++ suite
./scripts/linux_build_w2a.sh           # dense + arch + mha + swiglu + rmsnorm
./scripts/linux_build_w2a.sh --mha     # MHA only
./scripts/linux_build_w2a.sh --swiglu  # SwiGLU only
./scripts/linux_build_w2a.sh --rmsnorm # RMSNorm only
./scripts/linux_build_w2a.sh --all     # entire w2a go test ./...

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (through M1.4 RMSNorm)

| ID | Detail |
|---|---|
| Dense FormatNone×34 + SIMD | CPU + imported DotTile |
| `arch.volume` + Dense volume Forward | Grid/Cell/HopOrder/remote + 3×3×3 |
| `layer.mha` | ModeSelf forward: causal/bi, RoPE/none, GQA, FormatNone×34 |
| `abi.mha_forward_f32` | WVM1 tool magic |
| `layer.swiglu` | Gate/Up/Down + SiLU⊙ forward, FormatNone×34 |
| `abi.swiglu_forward_f32` | WVS1 tool magic |
| `layer.rmsnorm` | x̂ + γ⊙ forward, FormatNone×34 |
| `abi.rmsnorm_forward_f32` | WVR1 tool magic |

## gap? — remaining

- MHA cross-attn, backward, dropout, QKNorm, sliding/prefix/ALiBi extras
- SwiGLU / RMSNorm backward
- Other 19 layer kinds; Dense backward/quants; training; systems; model/IO; vendor accel

See `src/gaps/registry.cpp`.
