# Constants and Type Definitions

This directory contains common constants, type definitions, and standard binding layouts for WebGPU utilities.

## Files

- **`webgpu_types.wgsl`** - Common type definitions and constants
- **`webgpu_bindings.wgsl`** - Standard binding group layouts

## Implementation Status

- [ ] **Phase 3.13**: Create directory structure ✓
- [ ] **Phase 3.14**: Add atomic operation constants
- [ ] **Phase 3.15**: Add graph method constants  
- [ ] **Phase 3.16**: Add reduction constants
- [ ] **Phase 3.18**: Standardize binding layouts

## Contents

### webgpu_types.wgsl
Common type definitions used across algorithms:
```wgsl
// Node and edge type aliases
alias NodeId = u32;
alias EdgeId = u32; 
alias Weight = f32;

// Algorithm-specific types
struct PageRankParams {
  damping_factor: f32;
  tolerance: f32;
  max_iterations: u32;
}

struct SSSPParams {
  source_node: NodeId;
  infinity_value: u32;
}
```

### webgpu_bindings.wgsl
Standard binding group layouts:
```wgsl
// Standard graph data bindings (0-6)
// @group(0) @binding(0) adj_offsets
// @group(0) @binding(1) adj_data  
// @group(0) @binding(2) rev_adj_offsets
// @group(0) @binding(3) rev_adj_data
// @group(0) @binding(4) params
// @group(0) @binding(5) result
// @group(0) @binding(6) properties

// Algorithm-specific bindings (7+)
// Dynamic based on algorithm requirements
```

## Design Goals

1. **Consistency** - Standardized types and constants across algorithms
2. **Clarity** - Clear naming conventions and documentation
3. **Extensibility** - Easy to add new types for future algorithms
4. **Compatibility** - Compatible with existing generated code
