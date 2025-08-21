/**
 * StarPlat WebGPU Constants and Type Definitions
 * 
 * Centralized constants for WebGPU algorithms, binding layouts,
 * and common values used across all utilities and generated code.
 * 
 * Version: 1.0 (Phase 3.20)
 */

// =============================================================================
// STANDARD BINDING LAYOUT (Task 3.20)
// =============================================================================

// StarPlat WebGPU Standard Binding Layout:
// @group(0) @binding(0) - Forward adjacency offsets (CSR)
// @group(0) @binding(1) - Forward adjacency data (CSR)  
// @group(0) @binding(2) - Reverse adjacency offsets (CSR)
// @group(0) @binding(3) - Reverse adjacency data (CSR)
// @group(0) @binding(4) - Parameters uniform buffer
// @group(0) @binding(5) - Result atomic storage
// @group(0) @binding(6+) - Property buffers (dynamic)

const BINDING_ADJ_OFFSETS: u32 = 0u;
const BINDING_ADJ_DATA: u32 = 1u;
const BINDING_REV_ADJ_OFFSETS: u32 = 2u;
const BINDING_REV_ADJ_DATA: u32 = 3u;
const BINDING_PARAMS: u32 = 4u;
const BINDING_RESULT: u32 = 5u;
const BINDING_PROPERTIES_START: u32 = 6u;

// =============================================================================
// WORKGROUP AND DISPATCH CONSTANTS
// =============================================================================

// Standard workgroup configuration
const WORKGROUP_SIZE: u32 = 256u;
const WORKGROUP_SIZE_X: u32 = 256u;
const WORKGROUP_SIZE_Y: u32 = 1u;
const WORKGROUP_SIZE_Z: u32 = 1u;

// Workgroup memory limits
const MAX_SHARED_MEMORY_BYTES: u32 = 16384u; // 16KB typical limit
const SHARED_MEMORY_PER_THREAD: u32 = MAX_SHARED_MEMORY_BYTES / WORKGROUP_SIZE;

// Reduction tree depth (log2(WORKGROUP_SIZE))
const REDUCTION_STEPS: u32 = 8u;

// =============================================================================
// ALGORITHM-SPECIFIC CONSTANTS
// =============================================================================

// Graph algorithm constants
const INVALID_NODE: u32 = 0xFFFFFFFFu;
const INVALID_EDGE_INDEX: u32 = 0xFFFFFFFFu;
const INFINITY_U32: u32 = 0xFFFFFFFFu;
const INFINITY_F32: f32 = 3.4028235e+38; // f32::MAX

// Triangle counting
const TC_THRESHOLD: u32 = 1000u; // Threshold for algorithm switching

// PageRank
const PAGERANK_DAMPING: f32 = 0.85;
const PAGERANK_TOLERANCE: f32 = 1e-6;
const PAGERANK_MAX_ITERATIONS: u32 = 100u;

// SSSP
const SSSP_TOLERANCE: f32 = 1e-6;
const SSSP_MAX_ITERATIONS: u32 = 1000u;

// Betweenness Centrality  
const BC_TOLERANCE: f32 = 1e-6;
const BC_MAX_ITERATIONS: u32 = 1000u;

// =============================================================================
// OPTIMIZATION THRESHOLDS
// =============================================================================

// Graph method optimization thresholds
const SMALL_DEGREE_THRESHOLD: u32 = 8u;      // Linear vs binary search threshold
const LARGE_DEGREE_THRESHOLD: u32 = 64u;     // Algorithm switching threshold
const HUGE_DEGREE_THRESHOLD: u32 = 1024u;    // Special handling threshold

// Memory access patterns
const CACHE_LINE_SIZE: u32 = 64u;            // Bytes per cache line
const CACHE_LINE_ELEMENTS_U32: u32 = 16u;    // u32 elements per cache line
const PREFETCH_DISTANCE: u32 = 8u;           // Elements to prefetch ahead

// Atomic contention thresholds
const HIGH_CONTENTION_THRESHOLD: u32 = 32u;  // Threads per atomic target
const ATOMIC_BACKOFF_CYCLES: u32 = 100u;     // Cycles to wait on contention

// =============================================================================
// ERROR AND STATUS CODES
// =============================================================================

// Execution status codes
const STATUS_SUCCESS: u32 = 0u;
const STATUS_ERROR: u32 = 1u;
const STATUS_TIMEOUT: u32 = 2u;
const STATUS_MEMORY_ERROR: u32 = 3u;
const STATUS_CONVERGENCE_FAILED: u32 = 4u;

// Convergence flags
const CONVERGED_FLAG: u32 = 0u;     // 0 = converged
const NOT_CONVERGED_FLAG: u32 = 1u; // 1 = not converged
const ERROR_FLAG: u32 = 2u;         // 2 = error occurred

// =============================================================================
// TYPE SIZE CONSTANTS
// =============================================================================

const SIZEOF_U32: u32 = 4u;
const SIZEOF_I32: u32 = 4u;
const SIZEOF_F32: u32 = 4u;
const SIZEOF_U64: u32 = 8u;
const SIZEOF_I64: u32 = 8u;
const SIZEOF_F64: u32 = 8u;

// =============================================================================
// MATHEMATICAL CONSTANTS
// =============================================================================

const PI: f32 = 3.14159265359;
const E: f32 = 2.71828182846;
const SQRT_2: f32 = 1.41421356237;
const LOG_2: f32 = 0.69314718056;

// =============================================================================
// UTILITY MACROS (implemented as functions)
// =============================================================================

/**
 * Check if a value is power of 2
 */
fn isPowerOfTwo(x: u32) -> bool {
  return (x & (x - 1u)) == 0u && x != 0u;
}

/**
 * Next power of 2 greater than or equal to x
 */
fn nextPowerOfTwo(x: u32) -> u32 {
  var result = x;
  result -= 1u;
  result |= result >> 1u;
  result |= result >> 2u;
  result |= result >> 4u;
  result |= result >> 8u;
  result |= result >> 16u;
  return result + 1u;
}

/**
 * Fast integer division by power of 2
 */
fn divPowerOfTwo(x: u32, shift: u32) -> u32 {
  return x >> shift;
}

/**
 * Fast modulo by power of 2
 */
fn modPowerOfTwo(x: u32, mask: u32) -> u32 {
  return x & (mask - 1u);
}

/**
 * Calculate workgroups needed for node count
 */
fn calculateWorkgroups(nodeCount: u32) -> u32 {
  return (nodeCount + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
}

/**
 * Check if thread is within valid range
 */
fn isValidThread(threadId: u32, nodeCount: u32) -> bool {
  return threadId < nodeCount;
}

/**
 * Convert node index to workgroup and local index
 */
struct ThreadLocation {
  workgroup: u32,
  local: u32
}

fn getThreadLocation(globalId: u32) -> ThreadLocation {
  return ThreadLocation(
    globalId / WORKGROUP_SIZE,
    globalId % WORKGROUP_SIZE
  );
}

/**
 * Usage Examples:
 * 
 * // Binding usage
 * @group(0) @binding(BINDING_ADJ_OFFSETS) var<storage, read> adj_offsets: array<u32>;
 * 
 * // Constants usage
 * if (degree < SMALL_DEGREE_THRESHOLD) { /* linear search */ }
 * 
 * // Thread validation
 * if (!isValidThread(global_id.x, params.node_count)) { return; }
 * 
 * // Workgroup calculation
 * let workgroups = calculateWorkgroups(nodeCount);
 */
