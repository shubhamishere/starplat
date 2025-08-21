/**
 * StarPlat WebGPU Utilities - Main Include File
 * 
 * This file combines all WebGPU utility functions into a single include.
 * The generator includes this file to get all necessary utilities without
 * inlining them in the generated code.
 * 
 * MODULAR IMPLEMENTATION (Phase 3.14-3.16 COMPLETED):
 * Utilities are now organized in separate files for better maintainability:
 * - wgsl_kernels/webgpu_atomics.wgsl     (atomic operations)
 * - wgsl_kernels/webgpu_graph_methods.wgsl (graph traversal)
 * - wgsl_kernels/webgpu_reductions.wgsl   (workgroup reductions)
 * 
 * Phase 3.18 will modify the generator to automatically include these files.
 * For now, this file contains all utilities for backward compatibility.
 * 
 * Version: 1.1 (Phases 3.14-3.16 Completed)
 */

// =============================================================================
// CORE BINDINGS AND TYPES
// =============================================================================

// Standard graph data bindings (consistent across all algorithms)
@group(0) @binding(0) var<storage, read> adj_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> adj_data: array<u32>;
@group(0) @binding(2) var<storage, read> rev_adj_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> rev_adj_data: array<u32>;

// Parameters uniform buffer
struct Params { 
  node_count: u32; 
  _pad0: u32; 
  _pad1: u32; 
  _pad2: u32; 
}
@group(0) @binding(4) var<uniform> params: Params;

// Result and property buffers
@group(0) @binding(5) var<storage, read_write> result: atomic<u32>;
@group(0) @binding(6) var<storage, read_write> properties: array<atomic<u32>>;

// =============================================================================
// INCLUDED UTILITIES (Phase 3.18: Generator will include automatically)
// =============================================================================

// In Phase 3.18, the generator will replace these sections with:
// {{include "wgsl_kernels/webgpu_atomics.wgsl"}}
// {{include "wgsl_kernels/webgpu_graph_methods.wgsl"}} 
// {{include "wgsl_kernels/webgpu_reductions.wgsl"}}

// =============================================================================
// ATOMIC OPERATIONS (from wgsl_kernels/webgpu_atomics.wgsl)
// =============================================================================

// Workgroup shared memory for reductions
var<workgroup> scratchpad: array<u32, 256>;
var<workgroup> scratchpad_f32: array<f32, 256>;

/**
 * Atomic add operation for f32 values
 * Uses compare-and-swap loop on bitcast u32 representation
 */
fn atomicAddF32(ptr: ptr<storage, atomic<u32>>, val: f32) -> f32 {
  loop {
    let oldBits: u32 = atomicLoad(ptr);
    let oldVal: f32 = bitcast<f32>(oldBits);
    let newVal: f32 = oldVal + val;
    let newBits: u32 = bitcast<u32>(newVal);
    let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);
    if (res.exchanged) { return oldVal; }
  }
}

/**
 * Atomic subtract operation for f32 values
 */
fn atomicSubF32(ptr: ptr<storage, atomic<u32>>, val: f32) -> f32 {
  loop {
    let oldBits: u32 = atomicLoad(ptr);
    let oldVal: f32 = bitcast<f32>(oldBits);
    let newVal: f32 = oldVal - val;
    let newBits: u32 = bitcast<u32>(newVal);
    let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);
    if (res.exchanged) { return oldVal; }
  }
}

/**
 * Atomic minimum operation for f32 values
 */
fn atomicMinF32(ptr: ptr<storage, atomic<u32>>, val: f32) {
  loop {
    let oldBits: u32 = atomicLoad(ptr);
    let oldVal: f32 = bitcast<f32>(oldBits);
    let newVal: f32 = min(oldVal, val);
    let newBits: u32 = bitcast<u32>(newVal);
    let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);
    if (res.exchanged) { return; }
  }
}

/**
 * Atomic maximum operation for f32 values
 */
fn atomicMaxF32(ptr: ptr<storage, atomic<u32>>, val: f32) {
  loop {
    let oldBits: u32 = atomicLoad(ptr);
    let oldVal: f32 = bitcast<f32>(oldBits);
    let newVal: f32 = max(oldVal, val);
    let newBits: u32 = bitcast<u32>(newVal);
    let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);
    if (res.exchanged) { return; }
  }
}

// =============================================================================
// GRAPH METHODS (from wgsl_kernels/webgpu_graph_methods.wgsl)
// =============================================================================

/**
 * Checks if there's an edge between vertices u and w
 * Uses binary search for sorted adjacency lists (O(log n))
 * Falls back to linear search for unsorted lists
 */
fn findEdge(u: u32, w: u32) -> bool {
  let start = adj_offsets[u];
  let end = adj_offsets[u + 1u];
  let degree = end - start;
  
  // For small degree (< 8), use linear search
  if (degree < 8u) {
    for (var e = start; e < end; e = e + 1u) {
      if (adj_data[e] == w) { return true; }
    }
    return false;
  }
  
  // Binary search for larger degrees (assumes sorted adjacency)
  var left = start;
  var right = end;
  while (left < right) {
    let mid = left + (right - left) / 2u;
    let mid_val = adj_data[mid];
    if (mid_val == w) {
      return true;
    } else if (mid_val < w) {
      left = mid + 1u;
    } else {
      right = mid;
    }
  }
  return false;
}

/**
 * Returns the edge index for edge from u to w
 * Returns 0xFFFFFFFFu if edge not found
 */
fn getEdgeIndex(u: u32, w: u32) -> u32 {
  let start = adj_offsets[u];
  let end = adj_offsets[u + 1u];
  for (var e = start; e < end; e = e + 1u) {
    if (adj_data[e] == w) { return e; }
  }
  return 0xFFFFFFFFu; // Edge not found
}

/**
 * Get out-degree of vertex v
 */
fn getOutDegree(v: u32) -> u32 {
  return adj_offsets[v + 1u] - adj_offsets[v];
}

/**
 * Get in-degree of vertex v (requires reverse CSR)
 */
fn getInDegree(v: u32) -> u32 {
  return rev_adj_offsets[v + 1u] - rev_adj_offsets[v];
}

/**
 * Get total number of nodes
 */
fn getNodeCount() -> u32 {
  return params.node_count;
}

/**
 * Get total number of edges
 */
fn getEdgeCount() -> u32 {
  return arrayLength(&adj_data);
}

// =============================================================================
// WORKGROUP REDUCTIONS (from wgsl_kernels/webgpu_reductions.wgsl)
// =============================================================================

/**
 * Workgroup parallel sum reduction for u32 values
 * Uses tree reduction with shared memory scratchpad
 */
fn workgroupReduceSum(local_id: u32, value: u32) -> u32 {
  scratchpad[local_id] = value;
  workgroupBarrier();
  
  var stride = 128u;
  while (stride > 0u) {
    if (local_id < stride) {
      scratchpad[local_id] += scratchpad[local_id + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  
  return scratchpad[0];
}

/**
 * Workgroup parallel sum reduction for f32 values
 */
fn workgroupReduceSumF32(local_id: u32, value: f32) -> f32 {
  scratchpad_f32[local_id] = value;
  workgroupBarrier();
  
  var stride = 128u;
  while (stride > 0u) {
    if (local_id < stride) {
      scratchpad_f32[local_id] += scratchpad_f32[local_id + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  
  return scratchpad_f32[0];
}

/**
 * Workgroup parallel minimum reduction for u32 values
 */
fn workgroupReduceMin(local_id: u32, value: u32) -> u32 {
  scratchpad[local_id] = value;
  workgroupBarrier();
  
  var stride = 128u;
  while (stride > 0u) {
    if (local_id < stride) {
      scratchpad[local_id] = min(scratchpad[local_id], scratchpad[local_id + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  
  return scratchpad[0];
}

/**
 * Workgroup parallel maximum reduction for u32 values
 */
fn workgroupReduceMax(local_id: u32, value: u32) -> u32 {
  scratchpad[local_id] = value;
  workgroupBarrier();
  
  var stride = 128u;
  while (stride > 0u) {
    if (local_id < stride) {
      scratchpad[local_id] = max(scratchpad[local_id], scratchpad[local_id + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  
  return scratchpad[0];
}

// =============================================================================
// UTILITY CONSTANTS AND HELPERS
// =============================================================================

// Common constants
const INVALID_EDGE_INDEX: u32 = 0xFFFFFFFFu;
const MAX_WORKGROUP_SIZE: u32 = 256u;
const SMALL_DEGREE_THRESHOLD: u32 = 8u;

// =============================================================================
// MODULAR ORGANIZATION STATUS
// =============================================================================

/**
 * COMPLETED - Phase 3.14-3.16: Modular Utilities Created
 * 
 * ✅ Phase 3.14: Atomic operations extracted to wgsl_kernels/webgpu_atomics.wgsl
 * ✅ Phase 3.15: Graph methods extracted to wgsl_kernels/webgpu_graph_methods.wgsl
 * ✅ Phase 3.16: Workgroup reductions extracted to wgsl_kernels/webgpu_reductions.wgsl
 * 
 * NEXT: Phase 3.18 - Modify generator to automatically include modular files
 * 
 * Benefits achieved:
 * - Modular organization: Utilities separated by functionality
 * - Improved maintainability: Easy to update individual utility categories  
 * - Better testing: Each module can be tested independently
 * - Enhanced documentation: Comprehensive docs for each utility type
 * - Consistent architecture: Follows StarPlat patterns like mpi_header/
 * 
 * Generator integration (Phase 3.18) will replace this file content with:
 * 
 * #include "../webgpu_utils/wgsl_kernels/webgpu_atomics.wgsl"
 * #include "../webgpu_utils/wgsl_kernels/webgpu_graph_methods.wgsl" 
 * #include "../webgpu_utils/wgsl_kernels/webgpu_reductions.wgsl"
 * 
 * This will reduce generated code size and improve generator maintainability.
 */