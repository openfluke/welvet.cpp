# welvet.cpp — Welvet rebuilt in C++

**welvet.cpp is a full C++ rebuild of Welvet, the 3D volumetric AI engine.** The goal is an independent native engine: numerical types, layers, volumetric architecture, forward/backward, training modes, systems, model loading, inference, and checkpoints — extended toward GPUs, NPUs, TPUs, and other supported compute devices.

It will run directly from C++ and through a stable C ABI. Go Welvet can also delegate work to it through **NAC — Native Acceleration Compute**. NAC is one integration path into the C++ engine; the C++ engine must also build, load models, execute volumes, and train without a Go process.

## Bundled with Go Welvet — automatic acceleration

**The delivery goal is to ship welvet.cpp as a native library alongside Go Welvet. Go Welvet's dispatcher can hand supported work to C++ when measurements predict better end-to-end performance.** Users should not need a separate C++ source checkout or manual per-layer FFI calls. Packaging provides the appropriate native artifact for the supported OS/architecture, either linked into the application or distributed alongside it.

The intended automatic path is:

1. Discover the bundled library and available CPU/GPU/NPU/TPU devices.
2. Query support for the exact layer, dtype, shape, numerical policy and forward/backward/update operation.
3. Compare eligible Go and C++ execution paths using measured profiles, including FFI, compilation, synchronization and transfer costs.
4. Select the faster eligible path under the caller's policy. Keep Go execution when native delegation would not help.
5. Delegate whole layers, stacks or connected partitions where possible, keeping weights, intermediate tensors and training state resident across calls.
6. Record the selected implementation and refresh the plan when shapes, devices, numerical policy or training requirements change.

Applications can choose **automatic placement** or **force an explicit backend**. Forced selection must fail clearly when unsupported. Automatic placement chooses among valid paths before execution; execution failures follow an explicit recovery policy. Missing optional native artifacts leave ordinary Go execution available under automatic placement, with the reason visible in diagnostics.

**C++ is not assumed faster than Go.** Existing Go CPU/SIMD paths remain candidates. Native acceleration earns selection through measured benefit and validated numerical/training behavior. This bundling and dispatcher are planned work, not implemented functionality today.

## Support scorecard — so far

**Version `0.1.21` — M2 Dense NormalBP + M1 forwards**. Engine board **52 / 100**, extensions **13 / 100**, combined **65 / 200**. Live inventory: `GAPS.md` / `welvet_dense_fwd --gaps` / `welvet_dense_fwd --version`.

- [ ] **Layers:** Dense + MHA + SwiGLU + norms + Softmax + Sequential + Residual + Parallel + CNN1/2/3 + ConvT1/2/3 + RNN + LSTM + Embedding + KMeans + Mamba + Metacognition + GDN forward — **22 / 23** kinds have any forward; **Dense NormalBP** signed; Stack remaining; seqmix Kind helper done.
- [x] **Dtypes:** FormatNone pack/decode + Dense MatVec for **all 34** storage types (quant formats still `gap?`).
- [x] **Forward compute:** Dense FormatNone×34 CPU/SIMD validated via `w2a` `Go_v_Cpp` **exact_bit** spectrum (other 22 kinds pending).
- [ ] **Backward compute:** **1 / 23** — Dense NormalBP (dX/dW/SGD + volume) w2a exact_bit ×34; others pending.
- [ ] **Compute targets:** CPU host path only; vendor accel loaders are `gap?` placeholders — see the hardware summary below.
- [ ] **Platforms:** Linux host build path present; macOS / Windows / Android / iOS toolchains are placeholders — **1 / 5** OS families buildable (not yet multi-OS CI validated).
- [x] **Runtime / architecture (partial):** volumetric Grid/Cell/HopOrder/remote links + Dense volume Forward/**Backward**/NormalBP Step; Step-Mesh / other layer kinds pending.
- [ ] **Systems:** DNA, evolution, Tween, TANHI and telemetry pending.
- [ ] **Model / IO:** tokenizer, ENTITY, transformer, sampling and HF support pending.
- [ ] **Post-v1 engine features:** Lucy/LPD, Lean, CamSync and train/action scheduling pending.
- [x] **Native use / integration (partial):** C++ API + C ABI for Dense forward + `welvet_dense_fwd` tool; Go delegation / NAC still `gap?`.

### Supported cards / compute

⬜ = planned, 🚧 = partial, ✅ = validated for a documented scope. No individual card models have been validated yet. Forward, backward, and update support are tracked independently.

| Compute target | Forward | Backward | Weight updates | Supported cards / devices so far |
|---|:---:|:---:|:---:|---|
| Native CPU — x86-64 / Arm64 | 🚧 | 🚧 | 🚧 | Dense FormatNone×34 fwd + NormalBP dX/dW/ApplySGD + volume Step (v0.1.21); other layers pending |
| NVIDIA GPU | ⬜ | ⬜ | ⬜ | None validated |
| AMD GPU | ⬜ | ⬜ | ⬜ | None validated |
| Intel GPU | ⬜ | ⬜ | ⬜ | None validated |
| Intel NPU | ⬜ | ⬜ | ⬜ | None validated |
| Apple GPU | ⬜ | ⬜ | ⬜ | None validated |
| Apple Neural Engine | ⬜ | ⬜ | ⬜ | None validated |
| Qualcomm GPU | ⬜ | ⬜ | ⬜ | None validated |
| Qualcomm NPU | ⬜ | ⬜ | ⬜ | None validated |
| Google TPU | ⬜ | ⬜ | ⬜ | None validated |

Counts refer to **the C++ engine**, independently of existing Go Welvet support and historical experiments. A target may eventually support only a documented subset; inference support never automatically checks its backward or update boxes. Detailed [layer](#layer-runway), [dtype](#dtype-runway--34-storage-types), and [hardware](#hardware-backend-runway) tables define the remaining runway.

Welvet is a **3D volumetric AI engine**. A `1 × 1 × 1` volume is the ordinary single-cell neural-network case, with layers inside that cell. Larger volumes connect cells containing their own layer stacks. The C++ rebuild must own and execute that same topology, including layers inside cells and connected groups of work across the volume, while preserving Welvet's state, numerical and training semantics.

The long-term destination is a standalone C++ Welvet engine that discovers available CPU/GPU/NPU/TPU devices, queries their capabilities, places compatible layers, keeps weights resident, and executes complete models and volumetric training. Applications can use its C++ API or stable C ABI; Go Welvet can use that same boundary for delegated acceleration.

| Project state | Readout |
|---|---|
| **Version** | **`0.1.21-m2-dense-nbp`** (`welvet_dense_fwd --version` / CMake `0.1.21`) |
| **Runway** | **52 / 100 engine + 13 / 100 extensions = 65 / 200** |
| **Milestone** | **M2.01** — Dense NormalBP leaf + volumetric (FormatNone×34, w2a `WDB1`/`WVN1`) |
| **Scope** | Full Welvet engine rebuild: architecture, runtime, layers, numerical types, training, systems, model/IO; extended hardware, integration and distribution |
| **Platforms** | Linux (host build) · macOS / Windows / Android / iOS (toolchain placeholders) |
| **Hardware families** | Native CPU · NVIDIA · AMD · Intel GPU/NPU · Apple GPU/Neural Engine · Qualcomm GPU/NPU · Google TPU |
| **Baseline inspected** | Welvet source identifying itself as v1.1.3; 18 October 2026 |
| **What exists here today** | Buildable `libwelvet` + C ABI + `welvet_dense_fwd` (…/WPL/WDB/WVN); Dense FormatNone×34 fwd+**NormalBP** + volume F/B/SGD + MHA/SwiGLU/norms/Softmax/Sequential/Residual/Parallel/seqmix + CNN + ConvT + RNN + LSTM + Embedding + KMeans + Mamba + Metacognition + GDN; `GAPS.md` census; research notes under [`docs/`](docs/README.md) |

This README is the **runway**: the implementation contract and scorecard for where welvet.cpp is going. Every unchecked cell is planned work. Welvet's Go engine score does not transfer to the C++ rebuild, and historical benchmark reports do not count as a shipped native backend.

## Contents

- [Relationship to Welvet](#relationship-to-welvet)
- [End-to-end delegation](#end-to-end-delegation)
- [Scoreboard and credit rules](#scoreboard-and-credit-rules)
- [Engine rebuild scoreboard — 100 points](#engine-rebuild-scoreboard--100-points)
- [Hardware and delivery extension scoreboard — 100 points](#hardware-and-delivery-extension-scoreboard--100-points)
- [Full engine subsystem runway](#full-engine-subsystem-runway)
- [Coverage axes](#coverage-axes)
- [Layer runway](#layer-runway)
- [Dtype runway — 34 storage types](#dtype-runway--34-storage-types)
- [Kernel and training runway](#kernel-and-training-runway)
- [Volumetric execution and placement](#volumetric-execution-and-placement)
- [Hardware backend runway](#hardware-backend-runway)
- [Platform and binary runway](#platform-and-binary-runway)
- [FFI and C ABI contract](#ffi-and-c-abi-contract)
- [Resident weights and memory](#resident-weights-and-memory)
- [Validation and performance](#validation-and-performance)
- [Implementation sequence](#implementation-sequence)

## Relationship to Welvet

| Repository | Responsibility |
|---|---|
| [welvet](https://github.com/openfluke/welvet) | Go engine; volumetric architecture, layer semantics, storage formats, training modes, model state, and existing CPU/SIMD/WebGPU paths |
| [welvet.cpp](https://github.com/openfluke/welvet.cpp) | Full standalone C++ engine, native kernels, runtime, systems, model/IO, device adapters, C++ API and stable C ABI |
| [w2a](https://github.com/openfluke/w2a) | Cross-engine validation, parity matrices, device checks, training checks, and benchmark evidence |

Both engines implement the Welvet model contract. The Go implementation is the parity reference; the C++ engine owns its own topology, model state, numerical storage, forward/backward runtime and training execution. A native run does not need Go to orchestrate each layer.

The main Go scoreboard provides the first ten categories and their weights. Here those categories start unchecked and earn credit only from C++ implementation evidence. The numerical-first focus remains: the detailed runway covers dtypes and layers, without a separate block-quantization format board. Exact compatibility with every Go serialization/import variant must be stated per feature before claiming complete interchange.

CPU tiled, SIMD, and WebGPU behavior from Go are porting references. C++ CPU/SIMD and device implementations need independent validation. WebGPU compatibility can be implemented as an adapter; native GPU/NPU/TPU adapters extend beyond the original Go backend set. Plan 9 assembly is translated or replaced by equivalent C++ intrinsics/native assembly rather than linked as a Go runtime dependency.

### Architecture sketch — intended flow

```text
C++ application / CLI                 Go Welvet or another FFI caller
          |                                       |
     C++ engine API                          stable C ABI
          |                                       |
          +----------- C++ Welvet engine ---------+
                              |
          model / ENTITY / tokenizer / transformer / sampling
                              |
             3D architecture: cells, stacks, branches, links
                              |
          forward / backward / training / Step / Mesh / systems
                              |
                capability queries + placement + execution
                              |
             resident weights + tensors + training state
                              |
          CPU | NVIDIA | AMD | Intel | Apple | Qualcomm | TPU
```

A device-specific binary is an execution resource. It does not get to change the layer's math, choose a different training rule, or silently narrow a tensor.

## End-to-end delegation

The intended lifecycle is:

1. **Start** the C++ engine directly, or load its library and negotiate ABI/features.
2. **Discover** devices and query exact operation, shape, dtype, layout, precision, and pass support.
3. **Create/load** a complete model and volume, or import delegated layers/subgraphs, including cells, children, state and numerical metadata.
4. **Plan** placement explicitly. Reject unsupported requests with a reason before execution.
5. **Upload/import** weights once, preserving their declared encoding and version.
6. **Execute** forward on resident resources; preserve the tape/state required for the requested training mode.
7. **Execute** backward: input gradients, parameter gradients, bias gradients, and recurrent state gradients where applicable.
8. **Update** the authoritative resident weights, or export gradients for an explicitly selected host update.
9. **Synchronize/export** only when required: host consumers, cross-device boundaries, CamSync, snapshots, migration, or shutdown.
10. **Release** handles after outstanding work completes.

For example, a future plan could assign a supported Dense block to a GPU, an inference-only compiled partition to an NPU, and other operations to the C++ CPU implementation. Training must use a separately validated placement plan; successful NPU inference does not imply backward support.

## Scoreboard and credit rules

| Mark | Meaning | Credit |
|---|---|---:|
| ⬜ | Planned; no validated implementation in this checkout | 0% |
| 🚧 | Implemented subset with reproducible evidence and listed gaps | 50% of its declared gate |
| ✅ | Declared acceptance gate passes, with artifacts and device/runtime identity | 100% of its declared gate |
| N/A | Mathematically irrelevant or outside a backend's explicitly declared hardware scope | No implementation credit; reason required |
| BLOCKED | Intended work cannot currently be validated; missing device, SDK, access, or unresolved semantics | 0% |

Each section below has a point budget. Before work earns points, split that budget into fixed, reviewable acceptance gates whose weights sum to the section budget. Score each gate independently, then add the results. Until that breakdown exists and evidence is linked, the section earns zero. A single passing dtype, OS, or layer cannot mark an entire family complete.

**No blanket hardware promises.** The goal is complete Welvet semantics in the portable native implementation, plus accelerated coverage on each target for the operations it can support. An unsupported device cell stays visible. An NPU that only supports inference can complete an inference gate but cannot earn backward or update credit. N/A must never hide a missing portable-native implementation.

**No silent substitution.** Explicit device selection either runs the requested supported path or returns an error. Automatic placement may choose among supported implementations before execution, but records that decision. Runtime recovery or migration requires an explicit policy and a visible event.

**Version policy:** engine rebuild and extensions have separate 100-point boards. Engine completion requires all declared core gates; hardware/delivery completion requires its own gates. Report `engine / 100`, `extensions / 100`, and optionally their sum `/ 200`. A high hardware score cannot cover a missing tokenizer, backward runtime, or system. Release versions are independent of Go Welvet and require an explicit release review.

## Engine rebuild scoreboard — 100 points

This mirrors the **ten categories and exact weights** of Go Welvet's main version scoreboard. The descriptions express the C++ numerical/layer scope; Go's completed cells are not copied as C++ achievements. The separate extension board adds the wider hardware and delivery goals.

| # | Go Welvet scoreboard category | Weight | C++ acceptance target | Status | Earned |
|---:|---|---:|---|:---:|---:|
| 1 | **Foundation** | 15 | Core tensors, 34 numerical dtypes, weights, activations, layout, tiling, CPU/SIMD/device abstraction and ownership | 🚧 | 9 |
| 2 | **Dense MatVec microkernel** | 15 | Shared Dense compute across numerical dtypes; forward, dX, dW/db, updates; SC/MC/SIMD and declared device paths | 🚧 | 12 |
| 3 | **Transformer stack** | 14 | MHA, SwiGLU, RMSNorm, LayerNorm, Softmax, Embedding, Sequential, Residual, Stack and seqmix | 🚧 | 13 |
| 4 | **CNN / RNN / LSTM** | 6 | CNN1–3, RNN, LSTM; numerical coverage, applicable forward/backward, state and train grids | 🚧 | 5 |
| 5 | **Extended layers** | 7 | GDN, ConvT1–3, Mamba, KMeans, Parallel, Metacognition; applicable gradients and state | 🚧 | 7 |
| 6 | **Runtime + architecture** | 8 | Standalone volumetric grid, dispatch, forward, backward, training, Step/Mesh, links and execution ordering | 🚧 | 5 |
| 7 | **Systems** | 5 | DNA, evolution, Tween, TANHI and telemetry implemented in C++ | ⬜ | 0 |
| 8 | **Model / IO** | 8 | Tokenizer, ENTITY, transformer execution, sampling, HF inspection/loading and model construction | ⬜ | 0 |
| 9 | **Training credit** | 8 | NormalBP/StepBP/MeshBP, Tween families, proxies, sparse/alternating credit, cameral BranchModes and CamSync | 🚧 | 1 |
| 10 | **Peak fused / no host ALU** | 14 | Validated native/SIMD/device kernels, attention/RoPE, norms, activations, conv/recurrence, embedding and decoder fusion | ⬜ | 0 |
| | **Engine rebuild total** | **100** | | **🚧** | **52 / 100** |

Partial credit notes (v0.1.21): Foundation **9/15**, Dense **12/15** (+ NormalBP FormatNone×34), Runtime+architecture **5/8** (+ Dense volume Backward/Step), Training credit **1/8** (NormalBP Dense only), Transformer **13/14**; CNN/RNN/LSTM **5/6**; Extended **7/7**.

Post-v1 Go features are included in the relevant categories: Lucy/LPD and Lean in model construction, CamSync in training credit, and train/action FlightRole in runtime scheduling. The original Go board excludes applications and unfinished stubs; those stay separately visible below rather than receiving inherited engine credit.

## Hardware and delivery extension scoreboard — 100 points

| # | Extension | Weight | Acceptance target | Status | Earned |
|---:|---|---:|---|:---:|---:|
| 1 | NVIDIA GPU | 10 | Declared real-device numerical/layer forward, backward and update profiles | ⬜ | 0 |
| 2 | AMD GPU | 10 | Independent adapter, exact supported device/runtime profiles and validation | ⬜ | 0 |
| 3 | Intel GPU + NPU | 10 | Independently reported GPU and NPU operation/pass support | ⬜ | 0 |
| 4 | Apple GPU + Neural Engine | 10 | Independently reported Metal/native GPU and supported Neural Engine partitions | ⬜ | 0 |
| 5 | Qualcomm GPU + NPU | 10 | Independently reported mobile GPU and NPU operation/pass support | ⬜ | 0 |
| 6 | Google TPU | 10 | Supported runtime bridge, compiled partitions and explicit local/remote execution | ⬜ | 0 |
| 7 | Heterogeneous placement + residency | 12 | Capability-driven partitions, resident weights, cross-device forward/backward and coherent updates | ⬜ | 0 |
| 8 | Public C++ API + C ABI / Go FFI | 10 | Direct native use, stable versioned C boundary and optional Go delegation | 🚧 | 9 |
| 9 | Platform binaries + distribution | 10 | Linux/macOS/Windows/Android/iOS, clean-environment load and standalone execution | 🚧 | 1 |
| 10 | Hardware validation + performance | 8 | Real-device matrices, provenance, transfer-inclusive timings and released-artifact checks | 🚧 | 3 |
| | **Extension total** | **100** | | **🚧** | **13 / 100** |

**Combined runway: 65 / 200.** Partial notes (v0.1.21): Dense NormalBP (`WDB1`/`WVN1`) leaf+volume exact_bit×34; Parallel (`WPL1`) still forward-only. Stack/poly Parallel / other TrainModes still gap.

“All GPUs/NPUs/TPUs” is the expansion ambition. Each supported device model, numerical type, layer, shape and pass must have an explicit tested capability profile. A working inference partition cannot earn training credit. Additional vendors and device generations can extend the coverage inventory without quietly changing these fixed weights.

## Full engine subsystem runway

### Foundation, weights and native compute

| Go reference | C++ rebuild target | Status |
|---|---|:---:|
| `core/` | Tensor representation, all 34 dtypes, numerical conversions, activations, execution configuration and layer identifiers | 🚧 |
| `weights/` | Authoritative dtype-native parameter storage, MatVec/transpose, gradient/update operations, snapshots and store traversal | ⬜ |
| `tiling/` | Single/multicore tiling, tails, workspace budgets and deterministic scheduling policy | ⬜ |
| `simd/` | Equivalent x86-64/Arm64 kernels and instruction detection; no Go runtime dependency | ⬜ |
| `webgpu/` | Reproduce applicable device semantics with validated C++ device paths; track WebGPU adapter separately from vendor-native paths | ⬜ |
| `fusedgpu/` | Decoder/operation fusion, resident intermediates and actual-device execution provenance | ⬜ |

### Architecture and runtime — engine §6

| Go reference | C++ rebuild target | Status |
|---|---|:---:|
| `architecture/` | Own the 3D volume, coordinates, cell layers, graph links and topology validation | 🚧 |
| `runtime/dispatch/` | Dispatch every supported operation and variant with explicit capability checks | ⬜ |
| `runtime/forward/` | Full model/volume execution without calling Go for layer orchestration | 🚧 |
| `runtime/backward/` | Reverse dependency traversal, tapes, gradients, joins and shared parameters | ⬜ |
| `runtime/training/` | Loss/credit/update orchestration, batches, state and training-mode selection | ⬜ |
| `runtime/step/` | Systolic pipes, Line/Stack ticks, fill/drain, weight generations and RoleTrain/RoleAction | ⬜ |
| Mesh/volumetric scheduling | Volume dependencies, branch joins and cross-cell credit propagation | ⬜ |

### Systems — engine §7, 5 points

| Go reference | Weight | C++ rebuild target | Status | Earned |
|---|---:|---|:---:|---:|
| `systems/dna/` | 1 | Encode/decode topology and parameters across all implemented operations; native dtype preserved | ⬜ | 0 |
| `systems/evolution/` | 1 | DNA splice, mutation, cloning and NEAT-compatible topology/state behavior | ⬜ | 0 |
| `systems/tween/` | 1 | Target-propagation/local-credit numerical operations and update budgets | ⬜ | 0 |
| `systems/tanhi/` | 1 | TANHI structural/activation reporting and compatible telemetry transport | ⬜ | 0 |
| `systems/telemetry/` | 1 | Structural blueprint, counts, estimates and execution/learning observability | ⬜ | 0 |
| **Subtotal** | **5** | | **⬜** | **0 / 5** |

### Model / IO — engine §8, 8 points

| Go reference | Weight | C++ rebuild target | Status | Earned |
|---|---:|---|:---:|---:|
| `model/tokenizer/` | 1.5 | BPE/HF tokenization, vocabulary/special tokens and encode/decode parity | ⬜ | 0 |
| `model/entity/` | 2 | ENTITY inspect/load/write, dtype-native blobs, cameral/Stack state and round trips for declared supported variants | ⬜ | 0 |
| `model/transformer/` | 2.5 | Decoder construction, generation, KV cache, LM head and runtime integration | ⬜ | 0 |
| `model/sampling/` | 1 | ArgMax, temperature/TopK, penalties, banned IDs and deterministic RNG policy | ⬜ | 0 |
| `model/hf/` | 1 | Snapshot inspection, architecture detection and declared tensor-loader conversions | ⬜ | 0 |
| **Subtotal** | **8** | | **⬜** | **0 / 8** |

### Additional current-engine and future surfaces

| Surface | C++ runway | Accounting | Status |
|---|---|---|:---:|
| `lucy/` | LPD model construction, TrainStackCE, Lean density/selection and scoring behavior | Relevant engine §8/§9 gates, fixed before credit | ⬜ |
| `model/wav2vec2/` | Inventory and port applicable speech-model construction/loading/runtime paths | Additional current-source parity gate; allocate within model scope before credit | ⬜ |
| CamSync / BlendStores | Group/cross-layer soft and hard blending, all trainable child stores | Engine §9; device coherence separately extension §7 | ⬜ |
| Seed / serialization / hardware / memory | Inventory Go stub/implemented behavior; define native counterparts and missing contracts explicitly | Future surfaces; no inherited points | ⬜ |
| Fountain / donate | Explicit coding/transport/resource-sharing contracts if implemented | Future surfaces; no inherited points | ⬜ |
| Remaining `stub/` packages | Clustering, ensemble, evaluation, grafting, grouping, introspection, observer, pipeline, templates, universal | Inventory separately; presence of a Go stub is not a completed C++ feature | ⬜ |
| `apps/` model shells | Future native CLI/application entry points built on the standalone engine | Application scope outside the fixed engine board | ⬜ |

**Standalone acceptance:** a native C++ application can construct or load a model, create its volume, execute forward, run the selected training rule, save/reload supported state, and continue execution with no Go process. FFI-only layer parity is insufficient for the full rebuild.

## Coverage axes

A support claim is a tuple, not a logo:

```text
(layer + variant + shape + numerical dtype + activation type
 + gradient type + accumulator type + pass + training mode
 + device/backend + OS/architecture + layout + residency + execution mode)
```

| Axis | Runway |
|---|---|
| Topology | `1×1×1`, `2×2×2`, `3×3×3`, non-cubic volumes, cell stacks, connected/remote boundaries |
| Layer inventory | 23 registered kinds in `core/layer.go`, plus `layers/seqmix` helper coverage |
| Storage dtype | All 34 `core.DType` entries |
| Activation host type | All 15 Go `core.Numeric` kinds; ABI uses explicit representations |
| Pass | Forward; backward-input; backward-weights/bias; backward-state when applicable |
| Training | Full backward and independently callable credit/update primitives |
| Execution | Single-core, multicore, SIMD, device kernels, compiled partitions, asynchronous queues |
| Device | CPU and each declared GPU/NPU/TPU profile |
| Platform | OS × architecture × ABI × vendor runtime version |
| Shape/layout | Batch, sequence, rank, stride, alignment, block tails, zero/invalid shapes |
| Precision | Stored, transferred, decoded, accumulated, output, gradient, and update precision |
| State | Stateless, recurrent, KV cache, saved activations, versioned weights, CamSync |

Numerical dtype, activation type, accumulator type, gradient type, and update type are separate axes. Weightless operations such as Softmax are scored on their applicable numerical computation and gradients.

## Layer runway

Dense FormatNone×34 forward is implemented (v0.1.1); remaining rows are unimplemented C++ engine targets. `dX` means the input-gradient contract; `dW/db` includes every trainable parameter, not just Dense children. State gradients and auxiliary losses follow Welvet's semantics.

| ID | Layer | Forward target | Backward target | Native F | Native B | Device coverage |
|---:|---|---|---|:---:|:---:|:---:|
| 0 | **Dense** | GEMV/GEMM, bias, activation | dX, dW, act'; ApplyGradSGD (NormalBP) | 🚧 | 🚧 | ⬜ |
| 1 | **MultiHeadAttention** | Q/K/V/O, attention, masks, RoPE, cache | Projection + attention + RoPE gradients | 🚧 | ⬜ | ⬜ |
| 2 | **SwiGLU** | Gate/up/down projections, SiLU product | All projections and product/SiLU derivatives | 🚧 | ⬜ | ⬜ |
| 3 | **RMSNorm** | RMS reduction and scale | dX and scale gradient | 🚧 | ⬜ | ⬜ |
| 4 | **CNN1** | 1D convolution, stride/padding/dilation | dX, kernel gradients, db | 🚧 | ⬜ | ⬜ |
| 5 | **CNN2** | 2D convolution, stride/padding/dilation | dX, kernel gradients, db | 🚧 | ⬜ | ⬜ |
| 6 | **CNN3** | 3D convolution, stride/padding/dilation | dX, kernel gradients, db | 🚧 | ⬜ | ⬜ |
| 7 | **RNN** | Recurrent projection + activation + state | Sequence/BPTT and recurrent state gradients | 🚧 | ⬜ | ⬜ |
| 8 | **LSTM** | Input/forget/output/candidate gates, h/c | Gate, parameter, h/c and sequence gradients | 🚧 | ⬜ | ⬜ |
| 9 | **LayerNorm** | Mean/variance reduction, scale and bias | dX, scale gradient, bias gradient | 🚧 | ⬜ | ⬜ |
| 10 | **ConvTransposed1D** | 1D transposed convolution | dX, kernel gradients, db | 🚧 | ⬜ | ⬜ |
| 11 | **ConvTransposed2D** | 2D transposed convolution | dX, kernel gradients, db | 🚧 | ⬜ | ⬜ |
| 12 | **ConvTransposed3D** | 3D transposed convolution | dX, kernel gradients, db | 🚧 | ⬜ | ⬜ |
| 13 | **Embedding** | Indexed gather from a dtype-native table | Repeated-index scatter/reduction; defined token dX | 🚧 | ⬜ | ⬜ |
| 14 | **KMeans** | Welvet soft assignment and centroid computation | Welvet-defined differentiable paths and updates | 🚧 | ⬜ | ⬜ |
| 15 | **Softmax** | Standard/grid/temperature and supported variants | Welvet-defined Jacobian-vector products | 🚧 | ⬜ | ⬜ |
| 16 | **Parallel** | Branches, routing, combine, cameral modes | Branch credit, combine/routing gradients | 🚧 | ⬜ | ⬜ |
| 17 | **Sequential** | Ordered child operation chain | Reverse chain over arbitrary supported children | 🚧 | ⬜ | ⬜ |
| 18 | **Residual** | F(x) + skip with shape checks | Branch derivative + skip gradient | 🚧 | ⬜ | ⬜ |
| 19 | **Metacognition** | Observed computation and rule application | Welvet-defined trainable paths and state | 🚧 | ⬜ | ⬜ |
| 20 | **Mamba** | Selective state-space scan and projections | Scan/state/projection gradients | 🚧 | ⬜ | ⬜ |
| 21 | **GDN** | Gated DeltaNet state update and output | Match Welvet truncated-BPTT contract | 🚧 | ⬜ | ⬜ |
| 22 | **Stack** | Heterogeneous nested operation chain | Nested child credit and parameter enumeration | ⬜ | ⬜ | ⬜ |
| helper | **seqmix** | Welvet sequence-mixing helpers | Gradients for delegated helper operations | 🚧 | ⬜ | ⬜ |

### Layer acceptance details

| Family | Required details before completion | Status |
|---|---|:---:|
| Dense | Transpose, batch, tails, strides, bias, activation, dX-only, dW-only, update-only, dtype boundaries | 🚧 |
| Attention | Causal/cross/GQA and variants supported by Welvet; masks, ALiBi, sliding windows, dropout/RNG, KV invalidation | ⬜ |
| Norms | Epsilon, reduction axes, small variance, large values, saved statistics and parameter gradients | ⬜ |
| Softmax | Stable reductions; masked rows; temperature; Sparsemax/Entmax/Gumbel contracts where present | 🚧 |
| Activations | Linear, ReLU, SiLU, GELU, Tanh, Sigmoid, LeakyReLU, ReLU2; matching derivatives | ⬜ |
| Convolution | Non-square/non-cubic inputs, dilation, groups where supported, overlap accumulation, transpose output shape | ⬜ |
| Recurrence/SSM | Initial/final state ownership, sequence boundaries, reset, tape size, truncation policy | ⬜ |
| Composition | Nested non-Dense children; shared stores; branch ordering; gradient accumulation; no duplicate updates | ⬜ |
| Discrete/index/rule ops | Explicit differentiability and zero/undefined gradient rules matching Welvet | ⬜ |

Shared Dense acceleration earns projection credit. It does not establish that attention reductions, convolution transforms, recurrence, or branch combination execute on the device. Host orchestration and host arithmetic must be reported separately.

## Dtype runway — 34 storage types

Weights remain in the selected numerical dtype. Storage precision is independent of activation, accumulator, gradient, and update precision. Every backend must publish these choices. No automatic FP32 conversion may masquerade as native FP64, integer, low-bit, or complex arithmetic.

FormatNone pack → MatVec → Dense forward validated **exact_bit** vs Go for all 34 (v0.1.1 / `w2a` Go_v_Cpp). Quant formats, backward, and storage-preserving updates remain `gap?`.

| ID | DType | Representation/round trip | Forward | Backward | Storage-preserving update | Device profile |
|---:|---|:---:|:---:|:---:|:---:|:---:|
| 0 | Float64 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 1 | Float32 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 2 | Float16 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 3 | BFloat16 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 4 | FP8E4M3 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 5 | FP8E5M2 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 6 | Int64 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 7 | Int32 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 8 | Int16 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 9 | Int8 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 10 | Uint64 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 11 | Uint32 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 12 | Uint16 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 13 | Uint8 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 14 | Int4 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 15 | Uint4 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 16 | FP4 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 17 | Int2 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 18 | Uint2 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 19 | Ternary | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 20 | Binary | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 21 | Int | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 22 | Uint | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 23 | Uintptr | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 24 | Complex64 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 25 | Complex128 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 26 | NF4 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 27 | FP6 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 28 | Int6 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 29 | Uint6 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 30 | Int5 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 31 | Uint5 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 32 | Int3 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| 33 | Uint3 | ✅ | ✅ | ⬜ | ⬜ | ⬜ |

### Numerical semantics gates

| Concern | Contract to establish | Status |
|---|---|:---:|
| Integer and low-bit learning | Match Welvet's declared gradient/update rules, overflow, saturation/wrap, rounding, and zero representation | ⬜ |
| Binary/ternary | Bit order, codes, padding, zero/update masks, gradient representation and explicit learning rule | ⬜ |
| Complex | Real/imaginary layout and actual Welvet operator semantics; document real-part-only paths where applicable | ⬜ |
| Native-width Go types | Resolve `int`, `uint`, `uintptr` width explicitly at the ABI boundary; never serialize a process pointer as tensor identity | ⬜ |
| FP8/FP6/FP4/NF4 | Exact code tables and special-value behavior; no relabelled integer encoding | ⬜ |
| Accumulation | Queryable accumulator dtype, reduction order, reproducibility policy and tolerance | ⬜ |
| Conversion | Explicit lossy conversion permission and documented scratch lifetime | ⬜ |
| Precision limits | Return unsupported for a device that cannot satisfy the requested arithmetic policy | ⬜ |

For discrete types, “backward” means the specified Welvet training operation. It is not a claim that an integer cast or a binary threshold has an ordinary smooth derivative. Match and name the learning rule before validating it.

## Kernel and training runway

### Shared compute primitives

| Primitive | Forward/use | Backward/update counterpart | Status |
|---|---|---|:---:|
| GEMV/GEMM | Dense and shared projections | GEMV-transpose, dX, outer product/dW, db reduction | ⬜ |
| Reduction | Sum, mean, variance, max, norm, log-sum-exp | Reduction gradients and broadcast | ⬜ |
| Elementwise | Bias, multiply, add, activations, gates | Derivatives, Jacobian products, accumulation | ⬜ |
| Layout/shape | View, transpose, reshape, concatenate, split | Inverse mapping and branch gradient routing | ⬜ |
| Gather/scatter | Embedding, indexing and routing | Duplicate-index accumulation and atomic/reduction policy | ⬜ |
| Convolution transforms | im2col/col2im or specialized native convolution | Overlap accumulation and filter gradients | ⬜ |
| Attention/position | QK, scaling, masks, normalization, AV, RoPE | Attention and positional gradients | ⬜ |
| Scan/recurrence | RNN/LSTM/Mamba/GDN state evolution | Declared BPTT/truncation and state gradients | ⬜ |
| Optimizer/update | Storage-preserving SGD as the first contract | Update masks, accumulation, learning-rate/rounding policy | ⬜ |

### Training-mode integration

The C++ engine must implement Welvet's credit assignment and scheduling, as well as expose the primitive subsets needed for delegated training calls. A backend must not run full backprop merely because a caller requested dW-only proxy credit.

| Family | Native integration target | Status |
|---|---|:---:|
| NormalBP / StepBP / MeshBP | True chain-rule gradients, correct tape lifetime and update order | ⬜ |
| Tween / StepTween / MeshTween | Broadcast/projected credit and local update primitives | ⬜ |
| TweenChain / StepTweenChain / MeshTweenChain | Chain-rule primitives under the selected schedule | ⬜ |
| TweenSplit / StepTweenSplit / MeshTweenSplit | Leaf enumeration, split credit, correct scaling | ⬜ |
| TweenAlt / StepTweenAlt / MeshTweenAlt | Alternating schedules without stale weight/state use | ⬜ |
| HeadProxy / StepHeadProxy variants | True head Jacobian, hidden dW-only work | ⬜ |
| Linear / StepLinear variants | Affine transpose walk under Welvet's derivative policy | ⬜ |
| FastProxy / StepFastProxy / MeshFastProxy variants | Head-transpose proxy plus hidden dW-only | ⬜ |
| LinearCache / StepLinearCache variants | Explicit cache generation and invalidation | ⬜ |
| HeadProxyAsync / StepHeadProxyAsync variants | Sample-delayed credit with tracked versions | ⬜ |
| Sparse / StepSparse / MeshSparse variants | Head plus selected hidden leaves; skipped work remains skipped | ⬜ |
| Cameral BranchModes | Per-branch mode/credit, independent stores and update ownership | ⬜ |
| CamSync / BlendStores | Soft/hard blending, groups, cross-layer pairs, resident coherence | ⬜ |

The exact 29 named mode definitions live in [Welvet's training-mode contract](https://github.com/openfluke/welvet/blob/main/training_modes.md). The families above are integration gates, not new mode names or completed implementations.

## Volumetric execution and placement

A tensor's three spatial dimensions and the engine's three-dimensional cell topology are different concepts. The C++ architecture and its descriptors must preserve both. A `1×1×1` topology can contain an attention stack, and a CNN3 layer can exist inside a single cell.

| Capability | Acceptance target | Status |
|---|---|:---:|
| Single cell | Normal layer stack executes directly in C++ with parity to Go Welvet | ⬜ |
| Multi-cell volume | Cell coordinates, layer ordering, links and boundary tensors remain intact | ⬜ |
| Non-cubic volumes | No hidden assumption that all three dimensions are equal | ⬜ |
| Per-layer placement | Cell/layer/child identity maps to a queryable device and execution path | ⬜ |
| Partition placement | Fuse supported consecutive operations without altering visible state/semantics | ⬜ |
| Heterogeneous forward | Explicit handoff across CPU/GPU/NPU/TPU partitions | ⬜ |
| Heterogeneous backward | Gradient edges, tape ownership and accumulation cross device boundaries correctly | ⬜ |
| Step systolic execution | Fill/drain, ticks, Line/Stack ordering, train/action roles and weight versions | ⬜ |
| Mesh execution | Dependency ordering, joins and credit routing over volumetric links | ⬜ |
| Remote boundaries | Transport remains an explicit integration boundary; no assumption that FFI is a network protocol | ⬜ |
| Placement cost | Account for upload/download, queue latency, compilation, memory pressure and reuse | ⬜ |
| Plan invalidation | Rebuild on shape, topology, weights/layout, device, precision or mode changes | ⬜ |

**Placement policy:** discover → query exact support → estimate total cost → select an allowed path → record plan → execute. “Accelerated” requires measured benefit for the stated workload. For a tiny `1×1×1` network, the C++ CPU implementation may be faster than a device transfer.

## Hardware backend runway

These are intended adapter families and investigation routes, **not verified SDK support claims**. Exact APIs, device availability, licensing, supported operating systems, and arithmetic/training restrictions must be researched and validated when each adapter is implemented. Every row starts unchecked.

| Target | Candidate integration route | Forward gate | Backward/update gate | Main contract risk | Status |
|---|---|---|---|---|:---:|
| Native CPU | Portable C++ kernels; x86-64/Arm64 SIMD and threading | All applicable Welvet operations | Full portable-native training baseline | Numerical parity, widths, instruction dispatch | ⬜ |
| NVIDIA GPU | CUDA and appropriate native libraries/custom kernels | Projections, attention, conv, reductions, dtype-native compute | Explicit dX/dW/db/state kernels and resident updates | Architecture/runtime constraints and transfer costs | ⬜ |
| AMD GPU | HIP/ROCm and/or Vulkan adapter investigation | Same operation inventory, independently validated | Explicit training kernels and update coverage | Device/OS/runtime combinations | ⬜ |
| Intel GPU | SYCL/Level Zero/OpenCL/native-library investigation | Queryable device-native operation coverage | Training coverage independently measured | Runtime split, numerical types and precision | ⬜ |
| Intel NPU | OpenVINO/NPU runtime or suitable vendor interface investigation | Compiled supported inference partitions | Only advertise if explicitly implemented and validated | Compiled graph limits; no inferred backward support | ⬜ |
| Apple GPU | Metal compute and suitable native-library investigation | Native kernels and resident tensors | Explicit backward kernels and updates | Device dtype/precision coverage and memory synchronization | ⬜ |
| Apple Neural Engine | Core ML/available supported runtime route investigation | Supported compiled inference partitions | Separate capability; never assume ANE training | Runtime placement visibility and operation restrictions | ⬜ |
| Qualcomm GPU | Vulkan/OpenCL/vendor route investigation | Supported native kernels | Separate training kernels and updates | Mobile runtime/device constraints | ⬜ |
| Qualcomm NPU | QNN/appropriate supported vendor runtime investigation | Compiled supported inference partitions | Only if supported and validated for the exact runtime | Dtype/shape restrictions and inference-only paths | ⬜ |
| Google TPU | PJRT/XLA or an appropriate supported TPU runtime bridge investigation | Compiled supported partitions | Explicit compiled backward/update partitions when available | Access model, compilation, transport and residency | ⬜ |

For TPU, distinguish a local C ABI client from the accelerator's actual location. If execution requires a service or remote runtime, report network/serialization costs, authentication ownership, session lifetime, and remote resident weights. Do not imply a cloud TPU is a local device pointer.

### Backend capability report

| Required field | Purpose | Status |
|---|---|:---:|
| Adapter/runtime/device identity | Exact implementation, hardware and driver/runtime versions | ⬜ |
| Layer/op + variant + pass | Forward, dX, dW, db, update, state gradients listed independently | ⬜ |
| Dtype/layout | Numerical representation and supported stride/alignment restrictions | ⬜ |
| Compute/accumulator precision | Native arithmetic versus permitted conversion | ⬜ |
| Shape bounds/dynamic shape | Accepted dimensions, batch/sequence limits and compilation requirements | ⬜ |
| State and tape | Required saved tensors, recompute policy and resource sizing | ⬜ |
| Residency/import/export | Copy, shared allocation, zero-copy eligibility and synchronization | ⬜ |
| Execution provenance | Actual device/path; host arithmetic; delegated/fused child coverage | ⬜ |
| Unsupported reason | Stable code plus useful detail; no generic success followed by substitution | ⬜ |

## Platform and binary runway

Each OS/architecture artifact is scored independently of its adapters. A library loading on Windows does not establish Intel NPU support; an Android GPU pass does not establish iOS support.

| Platform | Intended architecture | Intended artifact shape | ABI/load | CPU parity | Hardware validation | Status |
|---|---|---|:---:|:---:|:---:|:---:|
| Linux | x86-64 | Shared `.so`, optional static archive, headers and manifest | ⬜ | ⬜ | ⬜ | ⬜ |
| Linux | Arm64 | Shared `.so`, optional static archive, headers and manifest | ⬜ | ⬜ | ⬜ | ⬜ |
| macOS | Apple silicon Arm64 | `.dylib`/static library or suitable framework package | ⬜ | ⬜ | ⬜ | ⬜ |
| macOS | x86-64, where retained | Separate artifact and explicitly bounded support profile | ⬜ | ⬜ | ⬜ | ⬜ |
| Windows | x86-64 | `.dll`, import library, headers and runtime manifest | ⬜ | ⬜ | ⬜ | ⬜ |
| Windows | Arm64 | Separate `.dll` and toolchain/runtime profile | ⬜ | ⬜ | ⬜ | ⬜ |
| Android | Arm64 | NDK-built `.so` and integration package; JNI only if needed | ⬜ | ⬜ | ⬜ | ⬜ |
| iOS | Device Arm64 | Static/framework package integrated at application build time | ⬜ | ⬜ | ⬜ | ⬜ |
| iOS | Simulator profile | Separate simulator slice; device acceleration not implied | ⬜ | ⬜ | ⬜ | ⬜ |

### Distribution acceptance

| Deliverable | Required content | Status |
|---|---|:---:|
| Build configuration | Proposed CMake presets/options; portable core builds without all vendor SDKs | ⬜ |
| Native artifact | OS/arch, ABI version, build ID, CPU requirements, enabled adapters | ⬜ |
| Public SDK | C++ API and C header, ownership contract, descriptor schema, examples and compatibility policy | ⬜ |
| Runtime dependencies | Explicit vendor runtime requirements and compatible tested versions | ⬜ |
| Integrity and provenance | Checksums, compiler flags, dependency/license manifest, reproducible build metadata | ⬜ |
| Welvet loader | Locate configured artifact, validate ABI, report missing dependency/actionable error | ⬜ |
| Mobile packaging | Device/simulator distinction, signing/application integration and resource lifecycle | ⬜ |
| Installation proof | Clean-environment load + forward/backward smoke per supported artifact | ⬜ |

The delivery goal is consumable binaries so Welvet users do not need a local C++ engine source checkout or every vendor SDK. Static and platform-integrated delivery are part of that goal, especially on mobile. No install/build commands are claimed to work yet.

## FFI and C ABI contract

**Design target only; no public symbols have been implemented or frozen. A direct C++ API must support full engine use as well as the C ABI.** The boundary is C-compatible, with C++ details kept inside the library. This allows Go integration without depending on C++ name mangling, compiler-specific class layout, exceptions, or STL containers.

| Surface | Intended responsibility | Status |
|---|---|:---:|
| ABI negotiation | Major/minor compatibility, descriptor sizes, feature bits, enum translation | ⬜ |
| Context/device | Create/destroy context, enumerate devices, query capabilities | ⬜ |
| Tensor descriptors | Rank, fixed-width dimensions/strides, dtype, byte length, layout, alignment, memory domain | ⬜ |
| Weight descriptors | Numerical dtype, shape, layout, parameter identity and version | ⬜ |
| Layer descriptors | Welvet kind, operation version, attributes, child references and trainable stores | ⬜ |
| Engine/model handles | Create/load a volume or model, configure training, execute, sample, checkpoint and inspect systems state | ⬜ |
| Plans | Validate/compile a layer or partition; query resource and support requirements | ⬜ |
| Buffer handles | Allocate/import/upload/download/release with explicit ownership and lifetime | ⬜ |
| Execution | Submit forward, dX, dW/db, state gradient, update and synchronization operations | ⬜ |
| Tape/state handles | Saved forward state, recurrent state and cache identity tied to plan/weight generation | ⬜ |
| Queues/events | Wait/poll/cancel policy, dependency events, completion errors and teardown | ⬜ |
| Diagnostics | Structured status, unsupported reason, per-call error detail and execution provenance | ⬜ |
| Go binding | cgo/FFI wrapper, safe handle lifetime, host representation mapping and integration tests | ⬜ |

### Boundary rules

1. Use versioned, size-tagged descriptors and fixed-width scalar fields; validate ranges, overflow, byte counts and alignment.
2. Translate Welvet IDs explicitly. Matching enum numbers in today's source is not a durable wire contract.
3. Return opaque handles for native resources; never expose C++ objects or throw exceptions through C.
4. Document caller-owned versus library-owned memory and which allocator releases each allocation.
5. Go memory borrowed for a synchronous call cannot be retained for asynchronous device work. Async input uses owned/copied memory or a separately valid pin/import contract under Go's FFI rules.
6. Keep host pointers, device handles and remote buffer IDs distinct. Publish endianness and native-width conversion rules.
7. Tie backward tapes and derived device caches to exact shape/plan/weight generations.
8. Report failures at submission and completion. An accepted asynchronous submission is not a completed computation.
9. Release resources deterministically; cover partial initialization, in-flight work, device loss and repeated destruction.
10. Include the executed backend and precision policy in validation traces.

## Resident weights and memory

Offloading weights is central to NAC. Moving the entire weight matrix over FFI for every layer call would miss the main goal.

| Capability | Acceptance target | Status |
|---|---|:---:|
| Authoritative owner | Explicitly identify whether host or native storage owns the current parameter generation | ⬜ |
| Upload once / reuse | Repeated forward/backward calls reuse resident weights without redundant full uploads | ⬜ |
| Preserve storage truth | Declared numerical dtype remains authoritative; no undeclared retained FP32 master | ⬜ |
| Derived device layout | Cache device layout transformations with version, memory accounting and invalidation | ⬜ |
| Activations and tapes | Keep intermediate work resident across adjacent layers and backward where possible | ⬜ |
| Resident gradients | Accumulate and clear dW/db/dX under an explicit step/microbatch policy | ⬜ |
| Resident update | Update selected stores in declared storage semantics; invalidate dependent caches | ⬜ |
| Host update mode | Export gradients explicitly, update host truth, then refresh/invalidate device generation | ⬜ |
| CamSync | Blend the intended generation across resident stores; order with outstanding train/action work | ⬜ |
| Checkpoint/export | Export current logical bytes and recurrent/training state needed for Welvet `.entity` flows | ⬜ |
| Migration | Transfer current authoritative state before changing device or execution plan | ⬜ |
| Memory budget | Account for weights, replicas, gradients, activations, tape, scratch, compiled plans and caches | ⬜ |
| Shared/unified memory | Use only under a valid ownership and synchronization contract; avoid automatic zero-copy claims | ⬜ |
| Concurrency | Read/write hazards, shared weights, multiple contexts, overlapping inference and training | ⬜ |
| Device loss | Surface failure and state freshness; recovery never silently restores stale parameters | ⬜ |

Temporary conversion scratch is allowed only under the declared numeric policy and must be accounted for. Optimizer state or discrete-update residuals, when required by the chosen rule, must be named, typed, serialized as needed, and distinguished from hidden master weights.

## Validation and performance

The [w2a harness](https://github.com/openfluke/w2a) is the intended home for C++ engine acceptance suites and comparative evidence. Vendor-specific build checks may accompany native integration, but correctness credit comes from reproducible end-to-end validation against the Welvet contract, including standalone native runs.

| Validation gate | Required evidence | Status |
|---|---|:---:|
| Inventory synchronization | Layer IDs, dtype list, mode definitions and subsystem inventory pinned to a Welvet revision | ⬜ |
| Standalone engine | Construct/load → volume execution → training → save/reload with no Go process | ⬜ |
| Systems parity | DNA/evolution/Tween/TANHI/telemetry outputs and state transitions | ⬜ |
| Model/IO parity | Token IDs, model descriptors, ENTITY round trips, sampling and generation | ⬜ |
| Numerical representation parity | Exact dtype/metadata round trips, boundaries, special values, invalid encodings | ⬜ |
| Forward parity | Per-layer outputs with stated tolerance and executed path | ⬜ |
| Backward parity | dX/dW/db/state compared independently; tied/shared parameters included | ⬜ |
| Gradient checks | Finite differences for suitable smooth floating paths; rule-specific checks for discrete/nonsmooth paths | ⬜ |
| Training parity | Fresh forward → loss/credit → gradient → update over multiple steps; no replayed fixed gradients | ⬜ |
| Volume parity | `1³`, `2³`, `3³`, non-cubic, nested cells, mixed-device forward/backward | ⬜ |
| Modes/CamSync | Required primitive subset, schedule ordering, sparse skips, resident blend coherence | ⬜ |
| ABI robustness | Invalid handles/descriptors, lifetimes, concurrent calls, completion failures, memory leak checks | ⬜ |
| Capability honesty | Unsupported requests fail clearly; actual device/path and conversion recorded | ⬜ |
| Real hardware | Exact chip, OS, compiler, adapter, runtime/driver; missing hardware reported as unavailable | ⬜ |
| Distribution | Test the released artifact and loader, rather than only the source-tree build | ⬜ |

### Performance scoreboard template

| Workload / shape | Numerical dtype | Pass / mode | Welvet baseline | C++ kernel time | C++ end-to-end time | Transfer / compile cost | Resident bytes | Speedup | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| To be measured | — | — | — | — | — | — | — | — | ⬜ |

Measure cold load/compile, initial upload, warm resident execution, synchronization, and export separately. Report both kernel-only and transfer-inclusive timings, and amortization over repeated calls. Include CPU orchestration, peak memory, correctness tolerance, sample count, and hardware/software identity. Throughput, latency, memory reduction, and power efficiency are separate outcomes.

A C++ implementation is not automatically faster than Welvet's Go SIMD. Existing [Dense comparison notes](docs/README.md) are useful research, but their old commands refer to benchmark sources absent from this checkout. They do not establish a current NAC build, ABI, backward implementation, or hardware support. [Learning experiments](docs/comparison.md) document a particular synthetic task and learning rule; they are not general training-accuracy claims.

## Implementation sequence

| Milestone | Concrete outcome | Exit evidence | Status |
|---|---|---|:---:|
| M0 — Engine contract | Pin Go revision, enumerate all subsystems, freeze numerical semantics and both boards' sub-gates | Reviewed source inventory and cross-language fixtures | ⬜ |
| M1 — C++ foundation | Native build, tensors, dtype-native weights and Dense forward | Standalone C++ example + `w2a` Go_v_Cpp exact_bit×34 | 🚧 |
| M2 — Native training | Dense dX/dW/db/update, ownership and resident state | Multi-step parity and save/reload | 🚧 |
| M3 — Numerical breadth | All 34 numerical dtypes and declared arithmetic/update paths | Dtype/pass matrix across CPU/SIMD | ⬜ |
| M4 — Layer breadth | Transformer, CNN/ConvT, recurrence, extended and composed layers | All applicable layer forward/backward gates | ⬜ |
| M5 — Full runtime | Architecture, dispatch, volume forward/backward, training, Step/Mesh and credit families | Standalone multi-cell training and scheduling parity | ⬜ |
| M6 — Systems and model/IO | DNA/evolution/Tween/TANHI/telemetry, tokenizer, ENTITY, HF, transformer, sampling, Lucy | Native load → execute/train → checkpoint → continue | ⬜ |
| M7 — Device execution | First native GPU adapter, fused kernels and resident updates | Real-device numerical parity and measured timings | ⬜ |
| M8 — Hardware expansion | NVIDIA/AMD/Intel/Apple/Qualcomm/TPU profiles and heterogeneous placement | Independent device matrices and coherent gradient boundaries | ⬜ |
| M9 — APIs and distribution | Public C++ API, C ABI, Go wrapper and desktop/mobile artifacts | Clean-environment standalone and FFI smoke checks | ⬜ |
| M10 — Release review | Engine and extension scores reported independently | Verified gates, supported profiles and published artifacts | ⬜ |

Device development and API prototyping can proceed alongside engine work. Native standalone functionality remains a required outcome; successful Go-to-C++ delegation cannot substitute for missing C++ runtime, systems or model/IO features.

## Source layout (M1 scaffold)

```text
include/welvet/          public C++ engine API
include/welvet/simd/     DotTile / DotI8 / rows (copied from welvet-to-rust_n_cpp)
include/welvet/imported/ decode + packed expand + gemvSIMD + train stubs (copied)
include/welvet_c.h       stable C ABI
src/core/               tensors, dtypes, activations, codecs
src/weights/            FormatNone×34 encode/decode + serial MatVec
src/layers/dense/       Dense forward (CPU + SIMD bridge)
src/backends/           CPU + gap? vendor load placeholders
src/gaps/               rebuild census (gap? / done)
src/interop/            C ABI implementation
cmake/                  android/ios toolchain placeholders
tools/dense_fwd/        binary wire tool for w2a Go_v_Cpp
examples/dense_forward/ standalone smoke
GAPS.md                 human-readable gap census
```

Still `gap?`: Dense backward/updates, remaining layers, architecture, runtime, systems, model, lucy, fusedgpu, bindings/go, wire imported tuned_*/backward_native, full multi-OS CI.


## Source of truth and runway maintenance

The starting inventory was read from Welvet's `core/layer.go`, `core/dtype.go`, `core/tensor.go`, `layers/`, `training_modes.md`, and main README. Future capability manifests and fixtures should pin a commit rather than relying on a moving branch.

When completing a gate, replace its unchecked mark, link the implementation and validation artifact, state its tested scope, and recompute only its fixed point allocation. Keep unsupported cells and device restrictions visible. A new Welvet layer or dtype extends this runway explicitly; it does not become C++-supported by inheritance.

**Destination:** a complete standalone C++ Welvet engine, preserving the volumetric model and numerical/training semantics, with capability-driven CPU/GPU/NPU/TPU execution, resident state, a native API, and optional FFI integration with Go Welvet.
