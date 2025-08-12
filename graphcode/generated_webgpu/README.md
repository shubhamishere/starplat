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
  - Buffers: `adj_data`, `adj_offsets`, `result` (atomic<u32>), `properties` (array<atomic<u32>>)
  - Writes node count to `properties[0]`; kernels read via `atomicLoad`
  - Safe dispatch (>= 1 workgroup)
- Kernel generation
  - One kernel per `forall` (outer parallelism); nested control supported in WGSL
  - Helper `findEdge(u,w)`; built-ins like `g.count_outNbrs(v)` lowered
- Control/expressions
  - Decls, assigns, unary, if/else, while, do-while, for, break/continue, return
  - fixedPoint: initial host loop sequencing present
  - Constants, arithmetic/relational/logical; property access; proc calls
- Reductions
  - Integer: `+=` via `atomicAdd`, `Min/Max` via `atomicMin/Max`
  - Float (experimental): CAS helpers `atomicAddF32/MinF32/MaxF32`

### Verified
- Triangle counting works end-to-end (smoke graph validated via driver)

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

