# welvet.cpp gap census

**Version `0.1.1-m1-dense-forward-34dtypes`** — see main [README](README.md) scoreboard (**15 / 100** engine + **6 / 100** extensions = **21 / 200**).

Every unimplemented rebuild item is marked **`gap?`**. Implemented M1 slices flip to **done**.

Run the live census:

```bash
./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
# or
./build/example_dense_forward
```

## Done (M1)

| ID | Detail |
|---|---|
| `core.dtype.float32` | Float32 dtype gate |
| `core.activate.f64` | Activate / deriv in float64 |
| `core.fp8_fp4_codecs` | FP8/FP4 codecs |
| `weights.native.all_dtypes` | FormatNone pack/decode all 34 dtypes |
| `weights.matvec.f32_none` | FormatNone MatVec (f64 accumulate) × 34 |
| `dense.forward.f32_cpu` / `dense.all_dtypes` | Dense serial CPU forward × 34 |
| `abi.dense_forward_f32` | C ABI + `welvet_dense_fwd` (WVD2+dtype) |
| `backends.cpu` | Native CPU target |
| `platform.linux` | Host Linux build path |

## gap? — everything else for the ~100-point Welvet rebuild

See `src/gaps/registry.cpp` for the machine-readable list. Highlights:

- **33 remaining dtypes** + all Numeric activation host types
- **Dense** backward, multicore, SIMD, quant formats, all dtypes
- **22 other layer kinds** + seqmix
- **Architecture / runtime / training / Step-Mesh**
- **Systems** (DNA, evolution, Tween, TANHI, telemetry)
- **Model/IO** (tokenizer, ENTITY, HF, sampling, Lucy)
- **Vendor accel load layers:** NVIDIA, AMD, Intel GPU/NPU, Apple Metal/ANE, Qualcomm GPU/NPU, Google TPU
- **Platforms:** macOS, Windows, Android, iOS validated artifacts
- **NAC / Go bindings**

When you implement a row, set `implemented=true` in `registry.cpp` and update the README scorecard.
