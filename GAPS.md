# welvet.cpp gap census

**Version `0.1.3-m1-arch-baseline`** — see main [README](README.md) scoreboard (**22 / 100** engine + **8 / 100** extensions = **30 / 200**).

SIMD/GEMV code under `include/welvet/simd/` + `include/welvet/imported/` was **copied** from `welvet-to-rust_n_cpp/welvet_cpp` (kernels only — not that tree’s docs).

Every unimplemented rebuild item is marked **`gap?`**. Implemented slices flip to **done**.

Run the live census:

```bash
# one-shot: Linux build + w2a Go↔C++ suite
./scripts/linux_build_w2a.sh
# or: ./scripts/linux_build_w2a.sh --all   # entire w2a go test ./...

./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --gaps
```

## Done (M1 → M1.1)

| ID | Detail |
|---|---|
| `core.dtype.float32` | Float32 dtype gate |
| `core.activate.f64` | Activate / deriv in float64 |
| `core.fp8_fp4_codecs` | FP8/FP4 codecs |
| `core.layer_types` | LayerType 0–22 + LayerMeta |
| `weights.native.all_dtypes` | FormatNone pack/decode all 34 dtypes |
| `weights.matvec.f32_none` | FormatNone MatVec (f64 accumulate) × 34 |
| `dense.forward.f32_cpu` / `dense.all_dtypes` | Dense serial CPU forward × 34 |
| `dense.forward.simd` | Imported DotTile/DotI8 gemvSIMD |
| `arch.volume` | Grid / Cell / Coord / HopOrder / remote links |
| `runtime.forward.dense` | Volume Forward Dense-only hop walker |
| `abi.dense_forward_f32` | C ABI + `welvet_dense_fwd` (WVD3) |
| `abi.arch_topology` | C ABI + WVA1 / WVF1 tool magics |
| `backends.cpu` | Native CPU target |
| `platform.linux` | Host Linux build path |

## gap? — remaining Welvet rebuild

See `src/gaps/registry.cpp`. Highlights:

- Volume Forward for the other 22 layer kinds; Backward / training / Step-Mesh
- ParallelBranches / SequentialLayers combine modes
- Dense backward, multicore, quant formats
- Systems, model/IO, vendor accel, multi-OS artifacts, NAC

When you implement a row, set `implemented=true` in `registry.cpp` and update the README scorecard.
