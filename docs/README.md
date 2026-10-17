# welvet 0.0.1 — dense forward SIMD: Go vs C++ vs Rust

Tiny baseline: same deep dense stack, **SIMD only**, exact welvet `ForwardSIMD` dtype dispatch in all three languages. Shared native weights (`WLVN`). Bit-identical outputs across Go / C++ / Rust for every FormatNone dtype below.

```bash
./scripts/run_all.sh deep          # all 34 dtypes (Go + C++ + Rust)
./scripts/run_all.sh deep float32  # DotTile head-to-head
./scripts/run_all.sh deep int8     # DotI8Tile head-to-head

# After changing welvet Go kernels only — keep saved C++/Rust JSON:
./scripts/rebench_go.sh deep
./scripts/rebench_go.sh deep float32
```

Profile **`deep`**: batch=8, 8× dense `256→256`, Linear, no bias, warm=30 / iters=300.  
JSON: `results/{go,cpp,rust,compare}_deep.json`.  
Times are **µs/op**. All rows **bit-identical** (same FNV hash) in both runs.

---

## Before (pre–Go kernel refresh)

First full `./scripts/run_all.sh deep`. **Wins:** Go **2** / C++ **31** / Rust **1**.

| dtype | Go | C++ | Rust | best |
|-------|---:|----:|-----:|:----:|
| float64 | 9565.4 | **4629.5** | 16855.4 | cpp |
| float32 | **1023.1** | 1156.3 | 1086.6 | go |
| float16 | **5647.9** | 11144.2 | 12259.4 | go |
| bfloat16 | 12538.9 | **2296.1** | 8546.9 | cpp |
| fp8e4m3 | 20325.9 | **10892.4** | 11274.6 | cpp |
| fp8e5m2 | 20123.5 | **10948.3** | 11407.9 | cpp |
| int64 | 22271.0 | **12245.0** | 31919.0 | cpp |
| int32 | 22495.4 | **11761.5** | 26900.1 | cpp |
| int16 | 22147.5 | **12099.6** | 21312.5 | cpp |
| int8 | 708.1 | **427.7** | 884.7 | cpp |
| uint64 | 21352.7 | **12983.1** | 36344.8 | cpp |
| uint32 | 20852.4 | **12336.7** | 31000.4 | cpp |
| uint16 | 20268.1 | **11870.1** | 25622.5 | cpp |
| uint8 | 4322.3 | **1458.4** | 1529.2 | cpp |
| int4 | 1287.2 | **992.5** | 1423.8 | cpp |
| uint4 | 22044.2 | **16037.2** | 26649.3 | cpp |
| fp4 | 11664.5 | 10374.3 | **10169.1** | rust |
| int2 | 1416.8 | **894.5** | 1397.5 | cpp |
| uint2 | 25167.3 | **17126.3** | 27720.7 | cpp |
| ternary | 1386.0 | **893.7** | 1386.9 | cpp |
| binary | 1326.6 | **926.1** | 1320.2 | cpp |
| int | 30230.4 | **12240.8** | 32445.5 | cpp |
| uint | 30061.9 | **13244.3** | 30005.6 | cpp |
| uintptr | 29653.2 | **13058.5** | 29902.4 | cpp |
| complex64 | 29202.5 | **11214.6** | 25555.5 | cpp |
| complex128 | 31087.5 | **12270.7** | 32619.8 | cpp |
| nf4 | 31278.2 | **13845.3** | 21963.8 | cpp |
| fp6 | 76871.4 | **30063.0** | 35180.3 | cpp |
| int6 | 76585.1 | **30339.3** | 35171.4 | cpp |
| uint6 | 75322.8 | **39423.8** | 44793.0 | cpp |
| int5 | 73682.1 | **26728.7** | 38414.1 | cpp |
| uint5 | 67208.1 | **33831.1** | 41722.9 | cpp |
| int3 | 53739.1 | **19606.6** | 32541.3 | cpp |
| uint3 | 52296.2 | **24871.4** | 36902.2 | cpp |

---

## After (Go Dense SIMD kernels refreshed)

Full `./scripts/run_all.sh deep` again after welvet Go upgrades (`WireF64`/`DotTileF64`, expand-once, BF16, alloc cleanup). C++/Rust ports unchanged in spirit; small timing jitter only. **Wins:** Go **24** / C++ **9** / Rust **1**.

| dtype | Go | C++ | Rust | best |
|-------|---:|----:|-----:|:----:|
| float64 | **1017.4** | 4662.6 | 17244.3 | go |
| float32 | **929.5** | 1162.2 | 1334.5 | go |
| float16 | **5336.5** | 11185.0 | 12304.1 | go |
| bfloat16 | 5045.3 | **2292.7** | 8521.8 | cpp |
| fp8e4m3 | 19822.1 | **10997.8** | 11322.1 | cpp |
| fp8e5m2 | 19448.1 | **10944.0** | 11207.7 | cpp |
| int64 | **1059.9** | 12242.2 | 32093.6 | go |
| int32 | **964.1** | 11776.2 | 26659.6 | go |
| int16 | **953.5** | 12091.2 | 21274.4 | go |
| int8 | 659.3 | **430.8** | 884.0 | cpp |
| uint64 | **3785.7** | 13101.0 | 36535.5 | go |
| uint32 | **3705.4** | 12335.3 | 30874.1 | go |
| uint16 | **3808.9** | 11796.3 | 25527.6 | go |
| uint8 | 7575.0 | **1457.9** | 1536.6 | cpp |
| int4 | 1350.7 | **999.3** | 1427.7 | cpp |
| uint4 | **3858.8** | 16108.1 | 26603.8 | go |
| fp4 | 11405.2 | 10338.9 | **10119.6** | rust |
| int2 | 1543.9 | **895.6** | 1390.3 | cpp |
| uint2 | **4931.0** | 17520.9 | 27565.0 | go |
| ternary | 1747.1 | **910.0** | 1393.3 | cpp |
| binary | 1413.0 | **930.5** | 1323.0 | cpp |
| int | **1043.4** | 12163.4 | 32339.9 | go |
| uint | **4715.0** | 13022.1 | 30953.6 | go |
| uintptr | **4775.3** | 13147.5 | 29958.3 | go |
| complex64 | **993.9** | 11289.5 | 25782.6 | go |
| complex128 | **1010.9** | 12062.0 | 32352.8 | go |
| nf4 | **4728.0** | 13913.0 | 21792.5 | go |
| fp6 | **10521.2** | 30152.0 | 35342.4 | go |
| int6 | **10144.6** | 29989.9 | 35542.0 | go |
| uint6 | **10489.1** | 39587.3 | 44942.7 | go |
| int5 | **9766.4** | 26946.1 | 38515.2 | go |
| uint5 | **10604.9** | 34166.1 | 41912.7 | go |
| int3 | **7409.0** | 20187.3 | 31982.9 | go |
| uint3 | **7853.8** | 25139.9 | 36061.4 | go |

---

## Before → after (all languages)

`Go Δ` = Go_before / Go_after (higher = Go got faster). C++/Rust columns are the same ports remeasured on the second full run (noise only).

| dtype | Go before | Go after | Go Δ | C++ before | C++ after | Rust before | Rust after |
|-------|----------:|---------:|-----:|-----------:|----------:|------------:|-----------:|
| float64 | 9565.4 | 1017.4 | **9.40×** | 4629.5 | 4662.6 | 16855.4 | 17244.3 |
| float32 | 1023.1 | 929.5 | 1.10× | 1156.3 | 1162.2 | 1086.6 | 1334.5 |
| float16 | 5647.9 | 5336.5 | 1.06× | 11144.2 | 11185.0 | 12259.4 | 12304.1 |
| bfloat16 | 12538.9 | 5045.3 | **2.49×** | 2296.1 | 2292.7 | 8546.9 | 8521.8 |
| fp8e4m3 | 20325.9 | 19822.1 | 1.03× | 10892.4 | 10997.8 | 11274.6 | 11322.1 |
| fp8e5m2 | 20123.5 | 19448.1 | 1.03× | 10948.3 | 10944.0 | 11407.9 | 11207.7 |
| int64 | 22271.0 | 1059.9 | **21.01×** | 12245.0 | 12242.2 | 31919.0 | 32093.6 |
| int32 | 22495.4 | 964.1 | **23.33×** | 11761.5 | 11776.2 | 26900.1 | 26659.6 |
| int16 | 22147.5 | 953.5 | **23.23×** | 12099.6 | 12091.2 | 21312.5 | 21274.4 |
| int8 | 708.1 | 659.3 | 1.07× | 427.7 | 430.8 | 884.7 | 884.0 |
| uint64 | 21352.7 | 3785.7 | **5.64×** | 12983.1 | 13101.0 | 36344.8 | 36535.5 |
| uint32 | 20852.4 | 3705.4 | **5.63×** | 12336.7 | 12335.3 | 31000.4 | 30874.1 |
| uint16 | 20268.1 | 3808.9 | **5.32×** | 11870.1 | 11796.3 | 25622.5 | 25527.6 |
| uint8 | 4322.3 | 7575.0 | 0.57× | 1458.4 | 1457.9 | 1529.2 | 1536.6 |
| int4 | 1287.2 | 1350.7 | 0.95× | 992.5 | 999.3 | 1423.8 | 1427.7 |
| uint4 | 22044.2 | 3858.8 | **5.71×** | 16037.2 | 16108.1 | 26649.3 | 26603.8 |
| fp4 | 11664.5 | 11405.2 | 1.02× | 10374.3 | 10338.9 | 10169.1 | 10119.6 |
| int2 | 1416.8 | 1543.9 | 0.92× | 894.5 | 895.6 | 1397.5 | 1390.3 |
| uint2 | 25167.3 | 4931.0 | **5.10×** | 17126.3 | 17520.9 | 27720.7 | 27565.0 |
| ternary | 1386.0 | 1747.1 | 0.79× | 893.7 | 910.0 | 1386.9 | 1393.3 |
| binary | 1326.6 | 1413.0 | 0.94× | 926.1 | 930.5 | 1320.2 | 1323.0 |
| int | 30230.4 | 1043.4 | **28.97×** | 12240.8 | 12163.4 | 32445.5 | 32339.9 |
| uint | 30061.9 | 4715.0 | **6.38×** | 13244.3 | 13022.1 | 30005.6 | 30953.6 |
| uintptr | 29653.2 | 4775.3 | **6.21×** | 13058.5 | 13147.5 | 29902.4 | 29958.3 |
| complex64 | 29202.5 | 993.9 | **29.38×** | 11214.6 | 11289.5 | 25555.5 | 25782.6 |
| complex128 | 31087.5 | 1010.9 | **30.75×** | 12270.7 | 12062.0 | 32619.8 | 32352.8 |
| nf4 | 31278.2 | 4728.0 | **6.62×** | 13845.3 | 13913.0 | 21963.8 | 21792.5 |
| fp6 | 76871.4 | 10521.2 | **7.31×** | 30063.0 | 30152.0 | 35180.3 | 35342.4 |
| int6 | 76585.1 | 10144.6 | **7.55×** | 30339.3 | 29989.9 | 35171.4 | 35542.0 |
| uint6 | 75322.8 | 10489.1 | **7.18×** | 39423.8 | 39587.3 | 44793.0 | 44942.7 |
| int5 | 73682.1 | 9766.4 | **7.54×** | 26728.7 | 26946.1 | 38414.1 | 38515.2 |
| uint5 | 67208.1 | 10604.9 | **6.34×** | 33831.1 | 34166.1 | 41722.9 | 41912.7 |
| int3 | 53739.1 | 7409.0 | **7.25×** | 19606.6 | 20187.3 | 32541.3 | 31982.9 |
| uint3 | 52296.2 | 7853.8 | **6.66×** | 24871.4 | 25139.9 | 36902.2 | 36061.4 |

---

## What this actually says

### 1. float32 DotTile (after)
| | µs/op |
|--|------:|
| Go | **929.5** |
| C++ | 1162.2 |
| Rust | 1334.5 |

**Go Plan 9 still wins float32.**

### 2. Where Go jumped
- Stream f64 family (float64, int16/32/64, complex*): **~9–30×** vs old Go; now beat C++ hard.
- Uint / nf4 / fp6 / int3–6: **~5–7×** vs old Go; beat C++ (~3×).
- bfloat16: **~2.5×** vs old Go; C++ packed path still faster.
- int8 / narrow: still C++ (~1.4–1.6× vs Go).
- uint8: C++/Rust still win; Go **regressed** (~0.57×) — follow-up.

### 3. Parity
Every dtype bit-identical across Go / C++ / Rust in both runs.

### 4. Bottom line
Wins flipped **2→24 for Go**. Re-measure Go alone with `./scripts/rebench_go.sh deep` if you only change welvet kernels.

---

## Kernel map (welvet Go after refresh)

| dtype family | kernel |
|--------------|--------|
| float32 | `DotTile` (AVX2 f32→f64) |
| int8 | act quant + `DotI8Tile` |
| int4 / int2 / ternary / binary | expand → `DotI8Tile` |
| float16 / bf16 / fp8 / fp4 | packed convert tiles + `DotTile` |
| uint8 | affine fused + `DotTile` |
| other uint* / nf4 / fp6 / int3/5/6 | expand once → `DotTile` |
| float64 / wide int / complex* | `WireF64` + `DotTileF64` (AVX2) |

Native weights: Go `TakeSnapshot` → `testdata/native/<profile>/<dtype>.bin` (`WLVN`).
