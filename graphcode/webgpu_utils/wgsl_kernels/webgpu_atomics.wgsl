/**
 * StarPlat WebGPU Atomic Operations Utilities
 * 
 * Custom atomic operations for f32 values using compare-and-swap loops
 * on bitcast u32 representations. Essential for PageRank, SSSP, and other
 * algorithms requiring float reductions with thread safety.
 * 
 * All operations use proper CAS loops with bitcast operations to ensure
 * correctness across WebGPU implementations.
 * 
 * Version: 1.0 (Phase 3.14)
 */

// =============================================================================
// ATOMIC OPERATIONS FOR F32 VALUES
// =============================================================================

/**
 * Atomic add operation for f32 values
 * Uses compare-and-swap loop on bitcast u32 representation
 * 
 * @param ptr Pointer to atomic<u32> storage containing f32 bits
 * @param val f32 value to add
 * @return Previous f32 value before addition
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
 * Uses compare-and-swap loop on bitcast u32 representation
 * 
 * @param ptr Pointer to atomic<u32> storage containing f32 bits
 * @param val f32 value to subtract
 * @return Previous f32 value before subtraction
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
 * Uses compare-and-swap loop on bitcast u32 representation
 * Essential for SSSP distance relaxation operations
 * 
 * @param ptr Pointer to atomic<u32> storage containing f32 bits
 * @param val f32 value to compare and potentially store
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
 * Uses compare-and-swap loop on bitcast u32 representation
 * 
 * @param ptr Pointer to atomic<u32> storage containing f32 bits
 * @param val f32 value to compare and potentially store
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
// ATOMIC UTILITY FUNCTIONS
// =============================================================================

/**
 * Atomic increment operation for u32 values (convenience wrapper)
 * @param ptr Pointer to atomic<u32> storage
 * @return Previous value before increment
 */
fn atomicIncU32(ptr: ptr<storage, atomic<u32>>) -> u32 {
  return atomicAdd(ptr, 1u);
}

/**
 * Atomic decrement operation for u32 values (convenience wrapper)
 * @param ptr Pointer to atomic<u32> storage  
 * @return Previous value before decrement
 */
fn atomicDecU32(ptr: ptr<storage, atomic<u32>>) -> u32 {
  return atomicSub(ptr, 1u);
}

/**
 * Atomic increment operation for f32 values (convenience wrapper)
 * @param ptr Pointer to atomic<u32> storage containing f32 bits
 * @return Previous f32 value before increment
 */
fn atomicIncF32(ptr: ptr<storage, atomic<u32>>) -> f32 {
  return atomicAddF32(ptr, 1.0);
}

/**
 * Atomic decrement operation for f32 values (convenience wrapper)
 * @param ptr Pointer to atomic<u32> storage containing f32 bits
 * @return Previous f32 value before decrement
 */
fn atomicDecF32(ptr: ptr<storage, atomic<u32>>) -> f32 {
  return atomicSubF32(ptr, 1.0);
}

// =============================================================================
// ATOMIC PATTERN HELPERS
// =============================================================================

/**
 * Atomic compare-and-flag operation for convergence detection
 * Atomically sets flag to 1 if condition is met
 * 
 * @param flag_ptr Pointer to convergence flag
 * @param condition Boolean condition to check
 */
fn atomicSetFlagIf(flag_ptr: ptr<storage, atomic<u32>>, condition: bool) {
  if (condition) {
    atomicOr(flag_ptr, 1u);
  }
}

/**
 * Atomic accumulation with conditional flagging
 * Adds value and sets flag if accumulation exceeds threshold
 * 
 * @param accum_ptr Pointer to accumulator
 * @param flag_ptr Pointer to threshold flag
 * @param val Value to add
 * @param threshold Threshold for flagging
 */
fn atomicAccumWithFlag(accum_ptr: ptr<storage, atomic<u32>>, flag_ptr: ptr<storage, atomic<u32>>, val: f32, threshold: f32) {
  let oldVal = atomicAddF32(accum_ptr, val);
  let newVal = oldVal + val;
  if (newVal >= threshold) {
    atomicOr(flag_ptr, 1u);
  }
}

// =============================================================================
// NOTES AND USAGE EXAMPLES
// =============================================================================

/**
 * Usage Examples:
 * 
 * // PageRank rank updates
 * atomicAddF32(&rank[v], delta_rank);
 * 
 * // SSSP distance relaxation  
 * atomicMinF32(&dist[v], new_distance);
 * 
 * // Triangle counting
 * atomicIncU32(&triangle_count);
 * 
 * // Betweenness centrality updates
 * atomicAddF32(&centrality[v], contribution);
 * 
 * // Convergence detection
 * let diff = abs(new_rank - old_rank);
 * atomicSetFlagIf(&converged, diff < tolerance);
 */

/**
 * Performance Notes:
 * 
 * - CAS loops may contend under high thread contention
 * - Consider workgroup-level reductions for heavy atomic usage
 * - f32 atomics are emulated and slower than native u32 atomics
 * - Use appropriate memory barriers for cross-workgroup synchronization
 */

/**
 * Testing Notes:
 * 
 * These functions will be tested in ../tests/atomic_tests.wgsl:
 * - Correctness under single-thread conditions
 * - Correctness under multi-thread contention  
 * - Performance comparison vs manual implementations
 * - Edge cases (NaN, infinity, overflow)
 */
