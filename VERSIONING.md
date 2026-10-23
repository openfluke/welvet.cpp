# welvet.cpp versioning — same board as Go Welvet

**One scoreboard:** the **same ten categories / 100 points** as Go Welvet’s engine board.  
**Formula (identical to Go pre-1.0):** `version = 0.{round(earned)}` until **100 → 1.0.0**.

| Tag | Meaning |
|-----|---------|
| **0.67** | Current — ~67/100 on the Go-mirrored board (FormatNone layers + train deep; Model/IO + most Systems still open) |
| **0.x** | `round(earned)` from the engine table in [README](README.md) |
| **1.0.0** | Engine board **100/100** — full Go CPU parity (incl. Model/IO, Systems, CamSync, quants or explicit gate rewrite) |

## What counts toward the version

| In (moves `0.x`) | Out (does **not** inflate version) |
|------------------|-------------------------------------|
| Layers, FormatNone×34, train modes, Grid/Line/Mesh, Tween*, C ABI | Vendor GPU/NPU/TPU (sibling repos) |
| Same 10 Go categories | Packaging / multi-OS delivery stretch goals |

Vendor accel (NVIDIA / AMD / Intel GPU / Intel NPU / Qualcomm HTP) lives in **sibling repos** and does not change this `0.x` number. Qualcomm HTP = [welvet.qnn](https://github.com/openfluke/welvet.qnn) **1.0.0**.

## Honest gaps vs Go v1.2 (why not 1.0 yet)

| Gap | Go board pts blocked |
|-----|----------------------|
| Model/IO (tokenizer, ENTITY, transformer, sampling, HF) | **8** |
| Systems DNA / evolution / TANHI / telemetry | **~4** |
| CamSync / full cameral Stack | **~1–2** |
| 20 quant formats through Dense/Foundation | **~4–6** |
| Peak fused (Softmax SIMD, quant Dot*, decoder fuse) | **~10+** |

Layers + FormatNone dtypes + training modes are largely done — that is why we are **0.67**, not **0.2**.
