# WGSL Kernel Utilities

This directory contains modular WGSL utility functions organized by functionality.

## Files

- **`webgpu_atomics.wgsl`** - Atomic operations (atomicAddF32, atomicMinF32, etc.)
- **`webgpu_graph_methods.wgsl`** - Graph traversal and utility functions  
- **`webgpu_reductions.wgsl`** - Workgroup parallel reduction operations
- **`webgpu_memory.wgsl`** - Memory management and shared memory utilities

## Implementation Status

- [ ] **Phase 3.14**: Extract atomic operations from main utilities file
- [ ] **Phase 3.15**: Extract graph methods from main utilities file  
- [ ] **Phase 3.16**: Extract reduction operations from main utilities file
- [ ] **Phase 3.17**: Add memory management utilities

## Usage

These files will be included in the main `webgpu_utils.wgsl` file:

```wgsl
// In ../webgpu_utils.wgsl (future implementation)
// Include individual utility modules
{{include "wgsl_kernels/webgpu_atomics.wgsl"}}
{{include "wgsl_kernels/webgpu_graph_methods.wgsl"}} 
{{include "wgsl_kernels/webgpu_reductions.wgsl"}}
{{include "wgsl_kernels/webgpu_memory.wgsl"}}
```

## Design Goals

1. **Modularity** - Each file contains related functionality
2. **Testability** - Individual modules can be tested separately
3. **Maintainability** - Easy to update specific utility categories
4. **Performance** - Optimized implementations for each operation type
