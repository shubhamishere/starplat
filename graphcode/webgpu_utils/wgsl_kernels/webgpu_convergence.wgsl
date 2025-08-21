/**
 * StarPlat WebGPU Enhanced Convergence Detection Utilities
 * 
 * Advanced convergence criteria and detection mechanisms for iterative algorithms
 * like PageRank, SSSP, and Betweenness Centrality. Provides multiple convergence
 * strategies beyond simple compare-and-flag.
 * 
 * Version: 1.0 (Phase 3.5)
 */

// =============================================================================
// CONVERGENCE CONSTANTS AND THRESHOLDS
// =============================================================================

// Default convergence tolerances
const DEFAULT_FLOAT_TOLERANCE: f32 = 1e-6;
const DEFAULT_RELATIVE_TOLERANCE: f32 = 1e-4;
const DEFAULT_ABSOLUTE_TOLERANCE: f32 = 1e-8;

// Convergence detection modes
const CONVERGENCE_ABSOLUTE: u32 = 0u;      // |new - old| < tolerance
const CONVERGENCE_RELATIVE: u32 = 1u;      // |new - old| / |old| < tolerance  
const CONVERGENCE_COMBINED: u32 = 2u;      // Both absolute and relative
const CONVERGENCE_L1_NORM: u32 = 3u;       // Sum of absolute differences
const CONVERGENCE_L2_NORM: u32 = 4u;       // Root mean square difference
const CONVERGENCE_MAX_DIFF: u32 = 5u;      // Maximum absolute difference

// Convergence state tracking
struct ConvergenceState {
  mode: u32,                    // Convergence detection mode
  absolute_tolerance: f32,      // Absolute tolerance threshold
  relative_tolerance: f32,      // Relative tolerance threshold
  max_iterations: u32,          // Maximum allowed iterations
  current_iteration: u32,       // Current iteration count
  converged: u32,              // 1 if converged, 0 if not
  total_difference: f32,        // Accumulated difference metric
  max_difference: f32,          // Maximum difference observed
  nodes_changed: u32           // Number of nodes that changed
}

// =============================================================================
// SINGLE-VALUE CONVERGENCE DETECTION
// =============================================================================

/**
 * Absolute convergence check for single f32 value
 * @param newValue New value in current iteration
 * @param oldValue Previous value from last iteration
 * @param tolerance Absolute tolerance threshold
 * @return true if converged (difference below tolerance)
 */
fn checkAbsoluteConvergence(newValue: f32, oldValue: f32, tolerance: f32) -> bool {
  return abs(newValue - oldValue) < tolerance;
}

/**
 * Relative convergence check for single f32 value
 * @param newValue New value in current iteration
 * @param oldValue Previous value from last iteration  
 * @param tolerance Relative tolerance threshold (percentage)
 * @return true if converged (relative change below tolerance)
 */
fn checkRelativeConvergence(newValue: f32, oldValue: f32, tolerance: f32) -> bool {
  if (abs(oldValue) < 1e-10) {
    // Avoid division by zero - use absolute convergence for small values
    return abs(newValue - oldValue) < tolerance;
  }
  return abs(newValue - oldValue) / abs(oldValue) < tolerance;
}

/**
 * Combined absolute and relative convergence check
 * @param newValue New value in current iteration
 * @param oldValue Previous value from last iteration
 * @param absTolerance Absolute tolerance threshold
 * @param relTolerance Relative tolerance threshold
 * @return true if both absolute and relative criteria are met
 */
fn checkCombinedConvergence(newValue: f32, oldValue: f32, absTolerance: f32, relTolerance: f32) -> bool {
  let absoluteOk = checkAbsoluteConvergence(newValue, oldValue, absTolerance);
  let relativeOk = checkRelativeConvergence(newValue, oldValue, relTolerance);
  return absoluteOk && relativeOk;
}

/**
 * Enhanced convergence check with NaN/infinity protection
 * @param newValue New value in current iteration
 * @param oldValue Previous value from last iteration
 * @param tolerance Tolerance threshold
 * @param mode Convergence mode (absolute/relative/combined)
 * @return true if converged, false if not converged or invalid values
 */
fn checkEnhancedConvergence(newValue: f32, oldValue: f32, tolerance: f32, mode: u32) -> bool {
  // Check for invalid values
  if (isnan(newValue) || isnan(oldValue) || isinf(newValue) || isinf(oldValue)) {
    return false; // Cannot converge with invalid values
  }
  
  switch (mode) {
    case CONVERGENCE_ABSOLUTE: {
      return checkAbsoluteConvergence(newValue, oldValue, tolerance);
    }
    case CONVERGENCE_RELATIVE: {
      return checkRelativeConvergence(newValue, oldValue, tolerance);
    }
    case CONVERGENCE_COMBINED: {
      return checkCombinedConvergence(newValue, oldValue, tolerance, tolerance * 0.1);
    }
    default: {
      return checkAbsoluteConvergence(newValue, oldValue, tolerance);
    }
  }
}

// =============================================================================
// WORKGROUP-LEVEL CONVERGENCE AGGREGATION
// =============================================================================

/**
 * Workgroup convergence detection using reduction
 * Aggregates convergence status across all threads in a workgroup
 * @param localId Thread's local ID within workgroup
 * @param hasConverged Individual thread's convergence status
 * @return true if ALL threads in workgroup have converged
 */
fn workgroupConvergenceAll(localId: u32, hasConverged: bool) -> bool {
  // Use boolean workgroup reduction (all threads must converge)
  return workgroupReduceAnd(localId, hasConverged);
}

/**
 * Workgroup convergence with tolerance for partial convergence
 * @param localId Thread's local ID within workgroup
 * @param hasConverged Individual thread's convergence status
 * @param threshold Minimum fraction of threads that must converge (0.0-1.0)
 * @return true if enough threads have converged
 */
fn workgroupConvergencePartial(localId: u32, hasConverged: bool, threshold: f32) -> bool {
  let convergedCount = workgroupReduceCount(localId, hasConverged);
  let workgroupSize = 256u; // Standard workgroup size
  let requiredCount = u32(f32(workgroupSize) * threshold);
  return convergedCount >= requiredCount;
}

/**
 * Workgroup difference aggregation for norm-based convergence
 * @param localId Thread's local ID within workgroup
 * @param difference Absolute difference for this thread
 * @param mode Aggregation mode (L1_NORM or L2_NORM)
 * @return Aggregated difference metric
 */
fn workgroupDifferenceAggregate(localId: u32, difference: f32, mode: u32) -> f32 {
  switch (mode) {
    case CONVERGENCE_L1_NORM: {
      return workgroupReduceSumF32(localId, abs(difference));
    }
    case CONVERGENCE_L2_NORM: {
      return sqrt(workgroupReduceSumF32(localId, difference * difference));
    }
    case CONVERGENCE_MAX_DIFF: {
      return workgroupReduceMaxF32(localId, abs(difference));
    }
    default: {
      return workgroupReduceSumF32(localId, abs(difference));
    }
  }
}

// =============================================================================
// ALGORITHM-SPECIFIC CONVERGENCE PATTERNS
// =============================================================================

/**
 * PageRank convergence detection
 * @param newRank New PageRank value
 * @param oldRank Previous PageRank value
 * @param tolerance Convergence tolerance
 * @return true if PageRank has converged for this node
 */
fn pageRankConvergence(newRank: f32, oldRank: f32, tolerance: f32) -> bool {
  // Use relative convergence for PageRank (values are normalized)
  return checkRelativeConvergence(newRank, oldRank, tolerance);
}

/**
 * SSSP convergence detection
 * @param newDistance New shortest path distance
 * @param oldDistance Previous distance
 * @param tolerance Convergence tolerance
 * @return true if SSSP has converged for this node
 */
fn ssspConvergence(newDistance: f32, oldDistance: f32, tolerance: f32) -> bool {
  // Use absolute convergence for SSSP distances
  return checkAbsoluteConvergence(newDistance, oldDistance, tolerance);
}

/**
 * Betweenness Centrality convergence detection
 * @param newCentrality New centrality value
 * @param oldCentrality Previous centrality value
 * @param tolerance Convergence tolerance
 * @return true if BC has converged for this node
 */
fn betweennessCentralityConvergence(newCentrality: f32, oldCentrality: f32, tolerance: f32) -> bool {
  // Use combined convergence for BC (handles different value ranges)
  return checkCombinedConvergence(newCentrality, oldCentrality, tolerance, tolerance * 0.01);
}

// =============================================================================
// CONVERGENCE STATE MANAGEMENT
// =============================================================================

/**
 * Initialize convergence state
 * @param mode Convergence detection mode
 * @param absTolerance Absolute tolerance
 * @param relTolerance Relative tolerance
 * @param maxIterations Maximum iterations allowed
 * @return Initialized convergence state
 */
fn initConvergenceState(mode: u32, absTolerance: f32, relTolerance: f32, maxIterations: u32) -> ConvergenceState {
  var state: ConvergenceState;
  state.mode = mode;
  state.absolute_tolerance = absTolerance;
  state.relative_tolerance = relTolerance;
  state.max_iterations = maxIterations;
  state.current_iteration = 0u;
  state.converged = 0u;
  state.total_difference = 0.0;
  state.max_difference = 0.0;
  state.nodes_changed = 0u;
  return state;
}

/**
 * Update convergence state with new iteration data
 * @param state Convergence state to update
 * @param totalDifference Total difference across all nodes
 * @param maxDifference Maximum difference observed
 * @param nodesChanged Number of nodes that changed
 * @param nodeCount Total number of nodes
 */
fn updateConvergenceState(state: ptr<function, ConvergenceState>, 
                         totalDifference: f32, 
                         maxDifference: f32, 
                         nodesChanged: u32, 
                         nodeCount: u32) {
  (*state).current_iteration += 1u;
  (*state).total_difference = totalDifference;
  (*state).max_difference = maxDifference;
  (*state).nodes_changed = nodesChanged;
  
  // Check convergence based on mode
  var hasConverged = false;
  switch ((*state).mode) {
    case CONVERGENCE_ABSOLUTE: {
      hasConverged = maxDifference < (*state).absolute_tolerance;
    }
    case CONVERGENCE_RELATIVE: {
      hasConverged = maxDifference < (*state).relative_tolerance;
    }
    case CONVERGENCE_L1_NORM: {
      let avgDifference = totalDifference / f32(nodeCount);
      hasConverged = avgDifference < (*state).absolute_tolerance;
    }
    case CONVERGENCE_L2_NORM: {
      let rmsDifference = sqrt(totalDifference / f32(nodeCount));
      hasConverged = rmsDifference < (*state).absolute_tolerance;
    }
    case CONVERGENCE_MAX_DIFF: {
      hasConverged = maxDifference < (*state).absolute_tolerance;
    }
    default: {
      hasConverged = maxDifference < (*state).absolute_tolerance;
    }
  }
  
  (*state).converged = select(0u, 1u, hasConverged);
}

/**
 * Check if algorithm should terminate
 * @param state Convergence state
 * @return true if should terminate (converged or max iterations reached)
 */
fn shouldTerminate(state: ConvergenceState) -> bool {
  return state.converged == 1u || state.current_iteration >= state.max_iterations;
}

// =============================================================================
// CONVERGENCE REPORTING AND STATISTICS
// =============================================================================

/**
 * Get convergence rate (difference reduction per iteration)
 * @param state Convergence state
 * @param initialDifference Initial difference at start
 * @return Convergence rate (0.0 = no improvement, 1.0 = converged)
 */
fn getConvergenceRate(state: ConvergenceState, initialDifference: f32) -> f32 {
  if (initialDifference < 1e-10 || state.current_iteration == 0u) {
    return 0.0;
  }
  let improvementRatio = 1.0 - (state.total_difference / initialDifference);
  return clamp(improvementRatio, 0.0, 1.0);
}

/**
 * Estimate iterations to convergence based on current rate
 * @param state Convergence state
 * @param targetDifference Target difference for convergence
 * @return Estimated remaining iterations (-1 if not converging)
 */
fn estimateIterationsToConvergence(state: ConvergenceState, targetDifference: f32) -> i32 {
  if (state.current_iteration < 2u || state.total_difference < targetDifference) {
    return 0; // Already converged or too early to estimate
  }
  
  // Simple geometric convergence rate estimation
  let currentRate = state.total_difference;
  let iterations = state.current_iteration;
  
  if (currentRate >= targetDifference * 0.99) {
    return -1; // Not converging
  }
  
  let reductionPerIteration = pow(currentRate / targetDifference, 1.0 / f32(iterations));
  if (reductionPerIteration >= 0.99) {
    return -1; // Converging too slowly
  }
  
  let remainingIterations = log(targetDifference / currentRate) / log(reductionPerIteration);
  return i32(remainingIterations);
}

// =============================================================================
// USAGE EXAMPLES AND PATTERNS
// =============================================================================

/**
 * Example Usage in PageRank Algorithm:
 * 
 * // Initialize convergence state
 * var convState = initConvergenceState(
 *   CONVERGENCE_RELATIVE,
 *   1e-8,  // absolute tolerance  
 *   1e-6,  // relative tolerance
 *   100u   // max iterations
 * );
 * 
 * // In main iteration loop:
 * let hasConverged = pageRankConvergence(newRank[v], oldRank[v], convState.relative_tolerance);
 * 
 * // Workgroup convergence detection:
 * let workgroupConverged = workgroupConvergenceAll(local_id.x, hasConverged);
 * if (local_id.x == 0u && workgroupConverged) {
 *   atomicAdd(&global_converged_count, 1u);
 * }
 * 
 * // Host-side convergence check:
 * // if (global_converged_count == total_workgroups) algorithm_converged = true;
 */

/**
 * Example Usage in SSSP Algorithm:
 * 
 * // Distance relaxation with convergence
 * let newDist = dist[u] + edgeWeight;
 * let hasChanged = newDist < dist[v];
 * 
 * if (hasChanged) {
 *   atomicMinF32(&dist[v], newDist);
 *   
 *   // Check convergence for this update
 *   let converged = ssspConvergence(newDist, dist[v], 1e-6);
 *   if (!converged) {
 *     atomicOr(&changed_flag, 1u);
 *   }
 * }
 */

/**
 * Performance Notes:
 * 
 * - Workgroup reductions add overhead but provide better convergence detection
 * - Use absolute convergence for distance-based algorithms (SSSP)
 * - Use relative convergence for normalized values (PageRank)
 * - Combined convergence provides robustness at higher computational cost
 * - L1/L2 norm convergence provides global convergence metrics
 * - Early termination saves significant computation for large graphs
 */
