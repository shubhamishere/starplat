/**
 * StarPlat WebGPU Workgroup Reduction Utilities
 * 
 * Parallel reduction operations using workgroup shared memory for efficient
 * aggregation within workgroups. Essential for algorithms requiring sum,
 * min, max operations across threads before final atomic updates.
 * 
 * All reductions use tree-based parallel algorithms with shared memory
 * scratchpads and proper workgroup barriers for synchronization.
 * 
 * Workgroup size: 256 threads (MAX_WORKGROUP_SIZE)
 * Reduction depth: 8 steps (log2(256))
 * 
 * Version: 1.0 (Phase 3.16)
 */

// =============================================================================
// WORKGROUP SHARED MEMORY DECLARATIONS
// =============================================================================

// Shared memory scratchpad for u32 reductions
var<workgroup> scratchpad: array<u32, 256>;

// Shared memory scratchpad for f32 reductions  
var<workgroup> scratchpad_f32: array<f32, 256>;

// Shared memory scratchpad for boolean reductions
var<workgroup> scratchpad_bool: array<u32, 256>; // Store bools as u32

// =============================================================================
// SUM REDUCTIONS
// =============================================================================

/**
 * Workgroup parallel sum reduction for u32 values
 * Uses tree reduction with shared memory scratchpad
 * 
 * @param local_id Thread's local index within workgroup (0-255)
 * @param value u32 value to contribute to sum
 * @return Sum of all values across workgroup (valid only for thread 0)
 */
fn workgroupReduceSum(local_id: u32, value: u32) -> u32 {
  // Store value in shared memory
  scratchpad[local_id] = value;
  workgroupBarrier();
  
  // Tree reduction: each step halves active threads
  var stride = 128u;
  while (stride > 0u) {
    if (local_id < stride) {
      scratchpad[local_id] += scratchpad[local_id + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  
  // Result valid only for thread 0
  return scratchpad[0];
}

/**
 * Workgroup parallel sum reduction for f32 values
 * Uses tree reduction with f32 shared memory scratchpad
 * 
 * @param local_id Thread's local index within workgroup (0-255)
 * @param value f32 value to contribute to sum
 * @return Sum of all f32 values across workgroup (valid only for thread 0)
 */
fn workgroupReduceSumF32(local_id: u32, value: f32) -> f32 {
  // Store value in f32 shared memory
  scratchpad_f32[local_id] = value;
  workgroupBarrier();
  
  // Tree reduction for f32 values
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

// =============================================================================
// MIN/MAX REDUCTIONS
// =============================================================================

/**
 * Workgroup parallel minimum reduction for u32 values
 * Finds the minimum value across all threads in workgroup
 * 
 * @param local_id Thread's local index within workgroup
 * @param value u32 value to contribute to minimum
 * @return Minimum value across workgroup (valid only for thread 0)
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
 * Finds the maximum value across all threads in workgroup
 * 
 * @param local_id Thread's local index within workgroup
 * @param value u32 value to contribute to maximum
 * @return Maximum value across workgroup (valid only for thread 0)
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

/**
 * Workgroup parallel minimum reduction for f32 values
 * Finds the minimum f32 value across all threads in workgroup
 * 
 * @param local_id Thread's local index within workgroup
 * @param value f32 value to contribute to minimum
 * @return Minimum f32 value across workgroup (valid only for thread 0)
 */
fn workgroupReduceMinF32(local_id: u32, value: f32) -> f32 {
  scratchpad_f32[local_id] = value;
  workgroupBarrier();
  
  var stride = 128u;
  while (stride > 0u) {
    if (local_id < stride) {
      scratchpad_f32[local_id] = min(scratchpad_f32[local_id], scratchpad_f32[local_id + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  
  return scratchpad_f32[0];
}

/**
 * Workgroup parallel maximum reduction for f32 values
 * Finds the maximum f32 value across all threads in workgroup
 * 
 * @param local_id Thread's local index within workgroup
 * @param value f32 value to contribute to maximum
 * @return Maximum f32 value across workgroup (valid only for thread 0)
 */
fn workgroupReduceMaxF32(local_id: u32, value: f32) -> f32 {
  scratchpad_f32[local_id] = value;
  workgroupBarrier();
  
  var stride = 128u;
  while (stride > 0u) {
    if (local_id < stride) {
      scratchpad_f32[local_id] = max(scratchpad_f32[local_id], scratchpad_f32[local_id + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  
  return scratchpad_f32[0];
}

// =============================================================================
// LOGICAL REDUCTIONS (AND/OR)
// =============================================================================

/**
 * Workgroup parallel logical AND reduction
 * Returns true only if ALL threads contribute true
 * 
 * @param local_id Thread's local index within workgroup
 * @param value Boolean value to contribute to AND
 * @return true if all values are true, false otherwise (valid only for thread 0)
 */
fn workgroupReduceAnd(local_id: u32, value: bool) -> bool {
  scratchpad_bool[local_id] = select(0u, 1u, value);
  workgroupBarrier();
  
  var stride = 128u;
  while (stride > 0u) {
    if (local_id < stride) {
      scratchpad_bool[local_id] = scratchpad_bool[local_id] & scratchpad_bool[local_id + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  
  return scratchpad_bool[0] != 0u;
}

/**
 * Workgroup parallel logical OR reduction
 * Returns true if ANY thread contributes true
 * 
 * @param local_id Thread's local index within workgroup
 * @param value Boolean value to contribute to OR
 * @return true if any value is true, false otherwise (valid only for thread 0)
 */
fn workgroupReduceOr(local_id: u32, value: bool) -> bool {
  scratchpad_bool[local_id] = select(0u, 1u, value);
  workgroupBarrier();
  
  var stride = 128u;
  while (stride > 0u) {
    if (local_id < stride) {
      scratchpad_bool[local_id] = scratchpad_bool[local_id] | scratchpad_bool[local_id + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  
  return scratchpad_bool[0] != 0u;
}

// =============================================================================
// COUNT REDUCTIONS  
// =============================================================================

/**
 * Workgroup parallel count reduction
 * Counts the number of threads for which condition is true
 * 
 * @param local_id Thread's local index within workgroup
 * @param condition Boolean condition to count
 * @return Number of threads where condition is true (valid only for thread 0)
 */
fn workgroupReduceCount(local_id: u32, condition: bool) -> u32 {
  scratchpad[local_id] = select(0u, 1u, condition);
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

// =============================================================================
// ADVANCED REDUCTIONS
// =============================================================================

/**
 * Workgroup parallel sum reduction with conditional inclusion
 * Sums values only from threads where condition is true
 * 
 * @param local_id Thread's local index within workgroup
 * @param value u32 value to potentially include in sum
 * @param condition Boolean condition for inclusion
 * @return Conditional sum across workgroup (valid only for thread 0)
 */
fn workgroupReduceSumIf(local_id: u32, value: u32, condition: bool) -> u32 {
  scratchpad[local_id] = select(0u, value, condition);
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
 * Workgroup parallel sum reduction for f32 values with conditional inclusion
 * 
 * @param local_id Thread's local index within workgroup
 * @param value f32 value to potentially include in sum
 * @param condition Boolean condition for inclusion
 * @return Conditional f32 sum across workgroup (valid only for thread 0)
 */
fn workgroupReduceSumIfF32(local_id: u32, value: f32, condition: bool) -> f32 {
  scratchpad_f32[local_id] = select(0.0, value, condition);
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

// =============================================================================
// REDUCTION UTILITY PATTERNS  
// =============================================================================

/**
 * Workgroup reduction with global atomic update
 * Performs workgroup reduction and atomically adds result to global storage
 * 
 * @param local_id Thread's local index within workgroup
 * @param value u32 value to contribute
 * @param global_ptr Pointer to global atomic accumulator
 */
fn workgroupReduceAndAtomicAdd(local_id: u32, value: u32, global_ptr: ptr<storage, atomic<u32>>) {
  let workgroup_sum = workgroupReduceSum(local_id, value);
  
  // Only thread 0 performs the global atomic update
  if (local_id == 0u) {
    atomicAdd(global_ptr, workgroup_sum);
  }
}

/**
 * Workgroup f32 reduction with global atomic update
 * 
 * @param local_id Thread's local index within workgroup
 * @param value f32 value to contribute
 * @param global_ptr Pointer to global atomic accumulator (as atomic<u32>)
 */
fn workgroupReduceAndAtomicAddF32(local_id: u32, value: f32, global_ptr: ptr<storage, atomic<u32>>) {
  let workgroup_sum = workgroupReduceSumF32(local_id, value);
  
  if (local_id == 0u) {
    atomicAddF32(global_ptr, workgroup_sum);
  }
}

// =============================================================================
// CONSTANTS FOR REDUCTION OPERATIONS
// =============================================================================

// Maximum workgroup size supported
const MAX_WORKGROUP_SIZE: u32 = 256u;

// Reduction tree depth (log2 of MAX_WORKGROUP_SIZE) 
const REDUCTION_STEPS: u32 = 8u;

// =============================================================================
// USAGE EXAMPLES AND PATTERNS
// =============================================================================

/**
 * Common Usage Patterns:
 * 
 * // Triangle counting with workgroup reduction
 * let local_triangles = // ... compute triangles for this thread
 * workgroupReduceAndAtomicAdd(local_id, local_triangles, &global_triangle_count);
 * 
 * // PageRank convergence checking
 * let converged_locally = abs(new_rank - old_rank) < tolerance;
 * let all_converged = workgroupReduceAnd(local_id, converged_locally);
 * if (local_id == 0u && all_converged) {
 *   atomicOr(&global_converged_flag, 1u);
 * }
 * 
 * // SSSP distance statistics
 * let finite_distance = dist[thread_id] != INFINITY;
 * let reachable_count = workgroupReduceCount(local_id, finite_distance);
 * if (local_id == 0u) {
 *   atomicAdd(&global_reachable_nodes, reachable_count);
 * }
 * 
 * // Betweenness centrality contribution aggregation
 * let contribution = // ... calculate contribution for this thread
 * let workgroup_total = workgroupReduceSumF32(local_id, contribution);
 * if (local_id == 0u) {
 *   atomicAddF32(&centrality[target_node], workgroup_total);
 * }
 */

/**
 * Performance Notes:
 * 
 * - Tree reductions are O(log n) in depth with O(n) total work
 * - Workgroup barriers synchronize shared memory access
 * - Only thread 0 should use the final reduction result
 * - Combining workgroup reductions with atomic updates reduces contention
 * - Shared memory usage: 256 * 4 bytes = 1KB per reduction type
 * - Multiple scratchpads allow concurrent reductions of different types
 */

/**
 * Memory Layout:
 * 
 * scratchpad[256]      -> u32 reductions (sum, min, max, count)
 * scratchpad_f32[256]  -> f32 reductions (sum, min, max) 
 * scratchpad_bool[256] -> boolean reductions (and, or)
 * 
 * Total shared memory: 3KB per workgroup
 */
