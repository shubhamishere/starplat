/**
 * StarPlat WebGPU Error Handling and Validation Utilities
 * 
 * Comprehensive error detection, validation, and debugging utilities for
 * WebGPU kernels. Provides robust error checking, bounds validation,
 * and debug output capabilities for algorithm development and testing.
 * 
 * Version: 1.0 (Phase 3.10)
 */

// =============================================================================
// ERROR CODES AND STATUS CONSTANTS
// =============================================================================

// Execution status codes (matches constants/webgpu_constants.wgsl)
const STATUS_SUCCESS: u32 = 0u;
const STATUS_ERROR: u32 = 1u;
const STATUS_BOUNDS_ERROR: u32 = 2u;
const STATUS_MEMORY_ERROR: u32 = 3u;
const STATUS_CONVERGENCE_FAILED: u32 = 4u;
const STATUS_INVALID_INPUT: u32 = 5u;
const STATUS_OVERFLOW_ERROR: u32 = 6u;

// Error reporting buffer binding (optional, binding 7+ typically)
// @group(0) @binding(7) var<storage, read_write> error_buffer: array<atomic<u32>>;

// =============================================================================
// BOUNDS CHECKING AND VALIDATION
// =============================================================================

/**
 * Validate node index is within graph bounds
 * @param nodeId Node index to validate
 * @param nodeCount Total number of nodes in graph
 * @return true if valid, false if out of bounds
 */
fn isValidNode(nodeId: u32, nodeCount: u32) -> bool {
  return nodeId < nodeCount;
}

/**
 * Validate edge index is within adjacency bounds  
 * @param edgeIdx Edge index to validate
 * @param edgeCount Total number of edges
 * @return true if valid, false if out of bounds
 */
fn isValidEdge(edgeIdx: u32, edgeCount: u32) -> bool {
  return edgeIdx < edgeCount && edgeIdx != INVALID_EDGE_INDEX;
}

/**
 * Validate array access bounds
 * @param index Array index to validate
 * @param arraySize Size of the array
 * @return true if valid, false if out of bounds
 */
fn isValidArrayIndex(index: u32, arraySize: u32) -> bool {
  return index < arraySize;
}

/**
 * Validate workgroup and thread indices
 * @param globalId Global thread ID
 * @param localId Local thread ID within workgroup  
 * @param workgroupSize Size of workgroup
 * @return true if valid, false if invalid
 */
fn isValidThreadId(globalId: u32, localId: u32, workgroupSize: u32) -> bool {
  return localId < workgroupSize;
}

/**
 * Validate float value is finite and not NaN
 * @param value Float value to validate
 * @return true if finite, false if NaN or infinite
 */
fn isValidFloat(value: f32) -> bool {
  return !isnan(value) && !isinf(value);
}

/**
 * Validate CSR adjacency access
 * @param nodeId Source node
 * @param neighborIdx Neighbor index within adjacency
 * @param nodeCount Total nodes
 * @return true if valid CSR access, false otherwise
 */
fn isValidCSRAccess(nodeId: u32, neighborIdx: u32, nodeCount: u32) -> bool {
  if (nodeId >= nodeCount) { return false; }
  
  let start = adj_offsets[nodeId];
  let end = adj_offsets[nodeId + 1u];
  let edgeIdx = start + neighborIdx;
  
  return edgeIdx >= start && edgeIdx < end && edgeIdx < arrayLength(&adj_data);
}

// =============================================================================
// ERROR REPORTING AND DEBUGGING
// =============================================================================

/**
 * Report error to global error buffer (if available)
 * Atomically sets error status for host-side detection
 * @param errorCode Error code to report
 * @param threadId Thread ID reporting the error (for debugging)
 */
fn reportError(errorCode: u32, threadId: u32) {
  // If error buffer is available, report the error
  // atomicMax(&error_buffer[0], errorCode);
  // atomicAdd(&error_buffer[1], 1u); // Error count
  // atomicMax(&error_buffer[2], threadId); // Last error thread
  
  // For now, we can use the result buffer to signal errors
  atomicOr(&result, errorCode << 24u); // Use upper 8 bits for error codes
}

/**
 * Assert condition and report error if false
 * @param condition Condition to check
 * @param errorCode Error code to report if condition fails
 * @param threadId Thread ID for debugging
 * @return true if condition passed, false if failed
 */
fn assertCondition(condition: bool, errorCode: u32, threadId: u32) -> bool {
  if (!condition) {
    reportError(errorCode, threadId);
    return false;
  }
  return true;
}

/**
 * Bounds-checked node access with error reporting
 * @param nodeId Node index to access
 * @param nodeCount Total number of nodes
 * @param threadId Thread ID for error reporting
 * @return true if access is valid, false and reports error otherwise
 */
fn safeNodeAccess(nodeId: u32, nodeCount: u32, threadId: u32) -> bool {
  if (nodeId >= nodeCount) {
    reportError(STATUS_BOUNDS_ERROR, threadId);
    return false;
  }
  return true;
}

/**
 * Bounds-checked array access with error reporting
 * @param index Array index
 * @param arraySize Array size  
 * @param threadId Thread ID for error reporting
 * @return true if access is valid, false and reports error otherwise
 */
fn safeArrayAccess(index: u32, arraySize: u32, threadId: u32) -> bool {
  if (index >= arraySize) {
    reportError(STATUS_BOUNDS_ERROR, threadId);
    return false;
  }
  return true;
}

/**
 * Safe CSR neighbor iteration with bounds checking
 * @param nodeId Source node
 * @param nodeCount Total nodes
 * @param threadId Thread ID for error reporting
 * @return Neighbor range {start, end} or {0, 0} if invalid
 */
struct NeighborRange {
  start: u32,
  end: u32,
  valid: bool
}

fn getSafeNeighborRange(nodeId: u32, nodeCount: u32, threadId: u32) -> NeighborRange {
  var range: NeighborRange;
  
  if (!safeNodeAccess(nodeId, nodeCount, threadId)) {
    range.start = 0u;
    range.end = 0u;
    range.valid = false;
    return range;
  }
  
  range.start = adj_offsets[nodeId];
  range.end = adj_offsets[nodeId + 1u];
  range.valid = true;
  
  // Additional validation
  if (range.end < range.start || range.end > arrayLength(&adj_data)) {
    reportError(STATUS_MEMORY_ERROR, threadId);
    range.valid = false;
  }
  
  return range;
}

// =============================================================================
// NUMERICAL VALIDATION AND OVERFLOW DETECTION
// =============================================================================

/**
 * Safe addition with overflow detection for u32
 * @param a First operand
 * @param b Second operand
 * @param threadId Thread ID for error reporting
 * @return Sum if no overflow, reports error and returns max value if overflow
 */
fn safeAddU32(a: u32, b: u32, threadId: u32) -> u32 {
  let max_val = 0xFFFFFFFFu;
  if (a > max_val - b) {
    reportError(STATUS_OVERFLOW_ERROR, threadId);
    return max_val;
  }
  return a + b;
}

/**
 * Safe multiplication with overflow detection for u32
 * @param a First operand
 * @param b Second operand  
 * @param threadId Thread ID for error reporting
 * @return Product if no overflow, reports error and returns max value if overflow
 */
fn safeMulU32(a: u32, b: u32, threadId: u32) -> u32 {
  let max_val = 0xFFFFFFFFu;
  if (a != 0u && b > max_val / a) {
    reportError(STATUS_OVERFLOW_ERROR, threadId);
    return max_val;
  }
  return a * b;
}

/**
 * Safe float operation with NaN/infinity checking
 * @param value Float value to validate
 * @param threadId Thread ID for error reporting
 * @return Validated value or 0.0 if invalid
 */
fn safeFloat(value: f32, threadId: u32) -> f32 {
  if (!isValidFloat(value)) {
    reportError(STATUS_INVALID_INPUT, threadId);
    return 0.0;
  }
  return value;
}

/**
 * Safe division with zero-check
 * @param numerator Numerator
 * @param denominator Denominator
 * @param threadId Thread ID for error reporting
 * @return Division result or 0.0 if division by zero
 */
fn safeDivide(numerator: f32, denominator: f32, threadId: u32) -> f32 {
  if (abs(denominator) < 1e-10) {
    reportError(STATUS_INVALID_INPUT, threadId);
    return 0.0;
  }
  let result = numerator / denominator;
  return safeFloat(result, threadId);
}

// =============================================================================
// DEBUG OUTPUT AND LOGGING (Development Support)
// =============================================================================

/**
 * Debug counter for development (uses atomic increment)
 * Useful for counting iterations, operations, etc.
 * @param threadId Thread ID
 */
fn debugCount(threadId: u32) {
  atomicAdd(&result, 1u);
}

/**
 * Debug marker for specific conditions
 * Sets a bit pattern in result for debugging specific conditions
 * @param markerId Unique marker ID (0-7)
 * @param threadId Thread ID
 */
fn debugMark(markerId: u32, threadId: u32) {
  let marker_bit = 1u << (markerId + 16u); // Use bits 16-23 for debug markers
  atomicOr(&result, marker_bit);
}

/**
 * Thread validation entry point - call at start of kernel main
 * @param globalId Global thread ID
 * @param localId Local thread ID
 * @param nodeCount Number of nodes in graph
 * @return true if thread should proceed, false if should return early
 */
fn validateThreadEntry(globalId: u32, localId: u32, nodeCount: u32) -> bool {
  // Check basic thread validity
  if (!isValidThreadId(globalId, localId, 256u)) {
    reportError(STATUS_ERROR, globalId);
    return false;
  }
  
  // Check if thread has valid work (node to process)
  if (globalId >= nodeCount) {
    return false; // Not an error, just no work for this thread
  }
  
  return true;
}

// =============================================================================
// ERROR EXTRACTION UTILITIES (Host-side helpers via comments)
// =============================================================================

/**
 * Host-side Error Extraction:
 * 
 * // Extract error status from result buffer:
 * const errorCode = (resultValue >> 24) & 0xFF;
 * const actualResult = resultValue & 0x00FFFFFF;
 * 
 * if (errorCode !== 0) {
 *   console.error(`WebGPU kernel error: ${getErrorMessage(errorCode)}`);
 * }
 * 
 * function getErrorMessage(errorCode) {
 *   switch (errorCode) {
 *     case 1: return "General error";
 *     case 2: return "Bounds check failed";
 *     case 3: return "Memory access error";
 *     case 4: return "Convergence failed";
 *     case 5: return "Invalid input";
 *     case 6: return "Overflow detected";
 *     default: return `Unknown error: ${errorCode}`;
 *   }
 * }
 */

// =============================================================================
// USAGE EXAMPLES AND PATTERNS
// =============================================================================

/**
 * Example Usage in Algorithm Kernels:
 * 
 * @compute @workgroup_size(256)
 * fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
 *         @builtin(local_invocation_id) local_id: vec3<u32>) {
 *   let v = global_id.x;
 *   let local_idx = local_id.x;
 *   
 *   // Validate thread entry
 *   if (!validateThreadEntry(v, local_idx, params.node_count)) {
 *     return;
 *   }
 *   
 *   // Safe CSR access
 *   let range = getSafeNeighborRange(v, params.node_count, v);
 *   if (!range.valid) { return; }
 *   
 *   // Safe neighbor iteration
 *   for (var i = range.start; i < range.end; i++) {
 *     if (!safeArrayAccess(i, arrayLength(&adj_data), v)) { return; }
 *     
 *     let neighbor = adj_data[i];
 *     if (!safeNodeAccess(neighbor, params.node_count, v)) { continue; }
 *     
 *     // Safe algorithm operations...
 *     let newValue = safeAddU32(oldValue, increment, v);
 *     atomicAdd(&properties[neighbor], newValue);
 *   }
 * }
 */
