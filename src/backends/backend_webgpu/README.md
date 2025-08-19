## Generated WebGPU code – quick guide and status

### What this folder contains
- `output.js`: Generated host (ES module) exporting `Compute_*` functions
- `kernel_*.wgsl`: Generated WGSL compute shader(s)
- `driver_triangle_count.js`: Deno driver to run triangle counting on a graph file

### How to run
```sh
cd starplat/graphcode/generated_webgpu
# Triangle Counting
deno run --allow-read --unstable-webgpu driver_triangle_count.js <path/to/graph.txt>
```

### Backend status (high-level)
- Host generation
  - Exports `Compute_*`; loads WGSL via `fetch('kernel_*.wgsl')`
  - Buffers: `adj_offsets`(b0), `adj_data`(b1), `params` uniform (b2), `result` atomic<u32> (b3)
  - Properties (Phase 0): per-property buffers emitted and bound at b4..N when present; otherwise a fallback `properties` buffer at b4
  - New API: `Compute_*(device, adj_dataBuffer, adj_offsetsBuffer, nodeCount, props = {})`
    - `props` supports passing pre-created GPU buffers per property; fallback buffers are created if missing
  - Explicit bind group layout creation; safe dispatch (>= 1 workgroup)
- Kernel generation
  - One kernel per `forall` (outer parallelism); nested control supported in WGSL
  - Helper `findEdge(u,w)`; built-ins like `g.count_outNbrs(v)` lowered
- Control/expressions
  - Decls, assigns, unary, if/else, while, do-while, for, break/continue, return
  - fixedPoint: host-side loop sequencing present (convergence detection to be refined)
  - Constants, arithmetic/relational/logical; property access lowered to per-property buffers (fallback supported)
- Reductions
  - Integer: `+=` via `atomicAdd`, `Min/Max` via `atomicMin/Max`
  - Float (experimental): CAS helpers `atomicAddF32/MinF32/MaxF32`

### Verified
- Triangle counting works end-to-end (smoke graph validated via driver)

### Recent changes (host + kernel)
- Added `Params` uniform buffer (b2) for `node_count`
- Introduced per-property buffer registry and WGSL declarations at b4..N
- Host accepts `props` map and binds provided property buffers when present
- Explicit bind group layout/pipeline layout creation

### Housekeeping (tests cleaned)
- Removed temporary test files:
  - `graphcode/generated_webgpu/triangle_smoke.txt`
  - `graphcode/generated_webgpu/driver_float_sum.js`
  - `graphcode/staticDSLCodes/float_sum_smoke`

### Build and generate
```sh
cd starplat/src
make -j8 | cat
./StarPlat -s -f ../graphcode/staticDSLCodes/triangle_counting_dsl -b webgpu | cat
# outputs here: starplat/graphcode/generated_webgpu/{output.js,kernel_0.wgsl}
```

### Notes
- Deno 2: use `--unstable-webgpu` (not the old `--unstable`)
- Drivers polyfill `fetch` to read local WGSL files

### Phase 0 TODO tracking
- DONE: Params uniform buffer (b2) and fixed core bindings (b0..b3)
- DONE: Type mapping scaffolding (DSL→WGSL/JS) and property registry
- DONE: Per-property WGSL declarations (b4..N) and property access lowering
- DONE: Explicit bind group layout; host API accepts `props`
- PENDING: Typed per-property buffers (i32/u32/f32/bool) instead of `atomic<u32>` placeholder
- PENDING: Host-side allocation/write/readback per property with usage flags (`in`/`out`/`inout`)
- PENDING: Refined convergence logic and result/copy-back protocol

### Roadmap: Phases and TODOs

- Phase 0 — Interface and types stabilization
  - DONE: Params uniform (b2), fixed bindings (`adj_offsets` b0, `adj_data` b1, `result` b3)
  - DONE: Type mapping scaffolding and property registry
  - DONE: Per-property WGSL declarations (b4..N) + access lowering
  - DONE: Explicit bind group layout; host accepts `props`
  - PENDING: Typed property arrays (i32/u32/f32/bool) instead of `atomic<u32>` placeholder
  - PENDING: Host property allocation/copy with `in|out|inout` semantics
  - PENDING: Convergence/result protocol refinements

- Phase 1 — DSL mapping completeness (host + WGSL)
  - Implement compound assignments (`+=,-=,*=,/=,|=,&=`) for ids, properties, index access
  - Complete relational/logical/unary typing/casts in WGSL
  - Graph methods: `neighbors_in`, `count_inNbrs`, `is_an_edge` fast path; weighted accessors
  - Reductions on properties and scalars with correct typed atomics/CAS for float
  - Fixed point: mark changes via compare-and-flag to drive convergence

- Phase 2 — Algorithm features (static)
  - PageRank: f32 rank arrays, damping constant, in-neighbor gather (float atomics via CAS)
  - SSSP (weighted): `dist` (i32/f32), relaxations with `atomicMin`, fixed point
  - Betweenness centrality: frontier arrays, `sigma` (counts) and `delta` (f32) with atomics
  - Triangle counting: maintain as verification case

- Phase 3 — Host/runtime ergonomics
  - Pipeline and shader module caching
  - Auto-generated bind groups per kernel (only used buffers)
  - Reusable drivers and CSR loaders (incl. reverse CSR, weights)
  - Optional copy-back of selected properties; clean API surface

- Phase 4 — Typing, portability, validation
  - Centralize DSL→WGSL type mapping; document lack of i64/f64
  - Safe casts; workgroup size tuning; robust bounds checks
  - Golden tests for codegen, e2e tests for TC/PR/SSSP/BC on small graphs

- Phase 5 — Optimizations
  - Sorted adjacency binary search for `is_an_edge`
  - Workgroup shared-memory tiling where applicable
  - Edge-parallel variants, degree-aware scheduling
  - Avoid unnecessary result readbacks; batch dispatch

