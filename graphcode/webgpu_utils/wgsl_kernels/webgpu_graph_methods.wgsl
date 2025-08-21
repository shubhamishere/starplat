/**
 * StarPlat WebGPU Graph Methods and Utilities
 * 
 * Graph traversal, edge detection, and utility functions for WebGPU algorithms.
 * Optimized implementations using binary search for large degrees and linear
 * search for small degrees to maximize performance across different graph types.
 * 
 * These functions operate on CSR (Compressed Sparse Row) graph representation
 * with forward and reverse adjacency arrays for comprehensive graph algorithms.
 * 
 * Version: 1.0 (Phase 3.15)
 */

// =============================================================================
// GRAPH TRAVERSAL AND EDGE DETECTION
// =============================================================================

/**
 * Checks if there's an edge between vertices u and w
 * Uses hybrid approach: linear search for small degrees (< 8),
 * binary search for larger degrees (assumes sorted adjacency)
 * 
 * @param u Source vertex ID
 * @param w Target vertex ID
 * @return true if edge (u,w) exists, false otherwise
 */
fn findEdge(u: u32, w: u32) -> bool {
  let start = adj_offsets[u];
  let end = adj_offsets[u + 1u];
  let degree = end - start;
  
  // For small degree (< 8), use linear search (more cache-friendly)
  if (degree < SMALL_DEGREE_THRESHOLD) {
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
 * Useful for weighted graphs where edge index maps to weight array
 * 
 * @param u Source vertex ID
 * @param w Target vertex ID  
 * @return Edge index if found, INVALID_EDGE_INDEX (0xFFFFFFFF) if not found
 */
fn getEdgeIndex(u: u32, w: u32) -> u32 {
  let start = adj_offsets[u];
  let end = adj_offsets[u + 1u];
  
  // Linear search for edge index (preserves index information)
  for (var e = start; e < end; e = e + 1u) {
    if (adj_data[e] == w) { return e; }
  }
  return INVALID_EDGE_INDEX;
}

/**
 * Binary search variant of getEdgeIndex for sorted adjacency lists
 * More efficient for high-degree vertices in sorted graphs
 * 
 * @param u Source vertex ID
 * @param w Target vertex ID
 * @return Edge index if found, INVALID_EDGE_INDEX if not found
 */
fn getEdgeIndexBinary(u: u32, w: u32) -> u32 {
  let start = adj_offsets[u];
  let end = adj_offsets[u + 1u];
  
  var left = start;
  var right = end;
  while (left < right) {
    let mid = left + (right - left) / 2u;
    let mid_val = adj_data[mid];
    if (mid_val == w) {
      return mid;
    } else if (mid_val < w) {
      left = mid + 1u;
    } else {
      right = mid;
    }
  }
  return INVALID_EDGE_INDEX;
}

// =============================================================================
// REVERSE GRAPH TRAVERSAL (IN-NEIGHBORS)
// =============================================================================

/**
 * Checks if there's a reverse edge (incoming edge to u from w)
 * Uses reverse CSR representation for efficient in-neighbor access
 * 
 * @param u Target vertex ID (receiving edge)
 * @param w Source vertex ID (sending edge)
 * @return true if edge (w,u) exists in reverse CSR, false otherwise
 */
fn findReverseEdge(u: u32, w: u32) -> bool {
  let start = rev_adj_offsets[u];
  let end = rev_adj_offsets[u + 1u];
  let degree = end - start;
  
  // Use same hybrid approach as forward edges
  if (degree < SMALL_DEGREE_THRESHOLD) {
    for (var e = start; e < end; e = e + 1u) {
      if (rev_adj_data[e] == w) { return true; }
    }
    return false;
  }
  
  // Binary search for larger in-degrees
  var left = start;
  var right = end;
  while (left < right) {
    let mid = left + (right - left) / 2u;
    let mid_val = rev_adj_data[mid];
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
 * Returns the reverse edge index for incoming edge to u from w
 * 
 * @param u Target vertex ID
 * @param w Source vertex ID
 * @return Reverse edge index if found, INVALID_EDGE_INDEX if not found
 */
fn getReverseEdgeIndex(u: u32, w: u32) -> u32 {
  let start = rev_adj_offsets[u];
  let end = rev_adj_offsets[u + 1u];
  
  for (var e = start; e < end; e = e + 1u) {
    if (rev_adj_data[e] == w) { return e; }
  }
  return INVALID_EDGE_INDEX;
}

// =============================================================================
// DEGREE CALCULATIONS
// =============================================================================

/**
 * Get out-degree of vertex v (number of outgoing edges)
 * 
 * @param v Vertex ID
 * @return Number of outgoing edges from vertex v
 */
fn getOutDegree(v: u32) -> u32 {
  return adj_offsets[v + 1u] - adj_offsets[v];
}

/**
 * Get in-degree of vertex v (number of incoming edges)
 * Requires reverse CSR representation
 * 
 * @param v Vertex ID
 * @return Number of incoming edges to vertex v
 */
fn getInDegree(v: u32) -> u32 {
  return rev_adj_offsets[v + 1u] - rev_adj_offsets[v];
}

/**
 * Get total degree of vertex v (in-degree + out-degree)
 * For undirected graphs, this counts each edge twice
 * 
 * @param v Vertex ID
 * @return Total degree of vertex v
 */
fn getTotalDegree(v: u32) -> u32 {
  return getOutDegree(v) + getInDegree(v);
}

// =============================================================================
// GRAPH UTILITY FUNCTIONS
// =============================================================================

/**
 * Get total number of nodes in the graph
 * @return Total node count from parameters
 */
fn getNodeCount() -> u32 {
  return params.node_count;
}

/**
 * Get total number of edges in the graph
 * @return Total edge count from adjacency data length
 */
fn getEdgeCount() -> u32 {
  return arrayLength(&adj_data);
}

/**
 * Get total number of reverse edges (may differ from forward edges in directed graphs)
 * @return Total reverse edge count
 */
fn getReverseEdgeCount() -> u32 {
  return arrayLength(&rev_adj_data);
}

/**
 * Check if vertex ID is valid (within graph bounds)
 * 
 * @param v Vertex ID to check
 * @return true if vertex is valid, false otherwise
 */
fn isValidVertex(v: u32) -> bool {
  return v < getNodeCount();
}

/**
 * Check if edge index is valid (within edge bounds)
 * 
 * @param e Edge index to check
 * @return true if edge index is valid, false otherwise
 */
fn isValidEdgeIndex(e: u32) -> bool {
  return e < getEdgeCount() && e != INVALID_EDGE_INDEX;
}

// =============================================================================
// NEIGHBOR ITERATION HELPERS
// =============================================================================

/**
 * Get the start offset for iterating over outgoing neighbors of vertex v
 * 
 * @param v Vertex ID
 * @return Start index in adj_data for vertex v's neighbors
 */
fn getOutNeighborStart(v: u32) -> u32 {
  return adj_offsets[v];
}

/**
 * Get the end offset for iterating over outgoing neighbors of vertex v
 * 
 * @param v Vertex ID  
 * @return End index (exclusive) in adj_data for vertex v's neighbors
 */
fn getOutNeighborEnd(v: u32) -> u32 {
  return adj_offsets[v + 1u];
}

/**
 * Get the start offset for iterating over incoming neighbors of vertex v
 * 
 * @param v Vertex ID
 * @return Start index in rev_adj_data for vertex v's in-neighbors
 */
fn getInNeighborStart(v: u32) -> u32 {
  return rev_adj_offsets[v];
}

/**
 * Get the end offset for iterating over incoming neighbors of vertex v
 * 
 * @param v Vertex ID
 * @return End index (exclusive) in rev_adj_data for vertex v's in-neighbors
 */
fn getInNeighborEnd(v: u32) -> u32 {
  return rev_adj_offsets[v + 1u];
}

// =============================================================================
// CONSTANTS USED BY GRAPH METHODS
// =============================================================================

// Threshold for switching between linear and binary search
const SMALL_DEGREE_THRESHOLD: u32 = 8u;

// Invalid edge index sentinel value
const INVALID_EDGE_INDEX: u32 = 0xFFFFFFFFu;

// =============================================================================
// USAGE EXAMPLES AND PATTERNS
// =============================================================================

/**
 * Common Usage Patterns:
 * 
 * // Triangle counting: check if edge exists
 * if (u < w && findEdge(w, v)) {
 *   atomicIncU32(&triangle_count);
 * }
 * 
 * // PageRank: iterate over in-neighbors
 * let in_start = getInNeighborStart(v);
 * let in_end = getInNeighborEnd(v);
 * for (var i = in_start; i < in_end; i++) {
 *   let u = rev_adj_data[i];
 *   let contribution = rank[u] / f32(getOutDegree(u));
 *   atomicAddF32(&new_rank[v], contribution);
 * }
 * 
 * // SSSP: relax outgoing edges  
 * let out_start = getOutNeighborStart(u);
 * let out_end = getOutNeighborEnd(u);
 * for (var i = out_start; i < out_end; i++) {
 *   let v = adj_data[i];
 *   let edge_idx = i; // Edge index for weight lookup
 *   let new_dist = dist[u] + edge_weights[edge_idx];
 *   atomicMinF32(&dist[v], new_dist);
 * }
 * 
 * // Betweenness Centrality: bidirectional edge traversal
 * if (findEdge(u, v) && findReverseEdge(v, u)) {
 *   // Process bidirectional edge
 * }
 */

/**
 * Performance Notes:
 * 
 * - Binary search is O(log n) vs linear O(n), but has overhead
 * - Threshold of 8 neighbors balances cache effects vs logarithmic benefit
 * - Sorted adjacency lists required for binary search correctness
 * - In-neighbor access requires reverse CSR construction
 * - Edge index functions preserve weight mapping for weighted graphs
 */
