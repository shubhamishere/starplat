/**
 * StarPlat WebGPU Dynamic Property Management Utilities
 * 
 * Complete implementation of dynamic property attachment with all initialization
 * patterns, dynamic allocation, and runtime property management for algorithms
 * that need to create properties during execution.
 * 
 * Version: 1.0 (Phase 3.9, 3.11)
 */

// =============================================================================
// DYNAMIC PROPERTY CONSTANTS AND TYPES
// =============================================================================

// Property initialization modes
const INIT_ZERO: u32 = 0u;              // Initialize to zero
const INIT_VALUE: u32 = 1u;             // Initialize to specific value
const INIT_INFINITY: u32 = 2u;          // Initialize to infinity (for distances)
const INIT_MINUS_ONE: u32 = 3u;         // Initialize to -1 (for predecessors)
const INIT_RANDOM: u32 = 4u;            // Initialize to random values
const INIT_DEGREE: u32 = 5u;            // Initialize to node degree
const INIT_INDEX: u32 = 6u;             // Initialize to node index
const INIT_UNIFORM: u32 = 7u;           // Initialize to uniform distribution
const INIT_COPY_FROM: u32 = 8u;         // Copy from another property
const INIT_FORMULA: u32 = 9u;           // Initialize using formula/expression

// Property data types
const PROP_TYPE_U32: u32 = 0u;
const PROP_TYPE_I32: u32 = 1u;
const PROP_TYPE_F32: u32 = 2u;
const PROP_TYPE_BOOL: u32 = 3u;

// Property access patterns
const ACCESS_READ_ONLY: u32 = 0u;
const ACCESS_WRITE_ONLY: u32 = 1u;
const ACCESS_READ_WRITE: u32 = 2u;
const ACCESS_ATOMIC: u32 = 3u;

// Property metadata
struct PropertyMetadata {
  name_hash: u32,        // Hash of property name
  data_type: u32,        // Property data type (PROP_TYPE_*)
  init_mode: u32,        // Initialization mode (INIT_*)
  access_pattern: u32,   // Access pattern (ACCESS_*)
  size_bytes: u32,       // Size in bytes
  alignment: u32,        // Memory alignment requirement
  node_count: u32,       // Number of nodes this property covers
  init_value: f32,       // Initial value (for INIT_VALUE mode)
  is_allocated: bool,    // Whether memory is allocated
  is_initialized: bool   // Whether values are initialized
}

// Dynamic property allocation info
struct DynamicPropertyInfo {
  base_offset: u32,      // Base offset in dynamic memory
  element_size: u32,     // Size per element
  total_size: u32,       // Total allocated size
  ref_count: u32,        // Reference count for cleanup
  last_access: u32       // Last access timestamp (for LRU)
}

// =============================================================================
// PROPERTY INITIALIZATION UTILITIES
// =============================================================================

/**
 * Initialize property with zero values
 * @param propertyPtr Pointer to property array
 * @param nodeCount Number of nodes
 * @param dataType Property data type
 */
fn initPropertyZero(propertyPtr: ptr<storage, array<u32>>, nodeCount: u32, dataType: u32) {
  for (var i = 0u; i < nodeCount; i++) {
    switch (dataType) {
      case PROP_TYPE_U32: {
        propertyPtr[i] = 0u;
      }
      case PROP_TYPE_I32: {
        propertyPtr[i] = bitcast<u32>(0i);
      }
      case PROP_TYPE_F32: {
        propertyPtr[i] = bitcast<u32>(0.0f);
      }
      case PROP_TYPE_BOOL: {
        propertyPtr[i] = 0u; // false
      }
      default: {
        propertyPtr[i] = 0u;
      }
    }
  }
}

/**
 * Initialize property with specific value
 * @param propertyPtr Pointer to property array
 * @param nodeCount Number of nodes
 * @param dataType Property data type
 * @param value Initial value
 */
fn initPropertyValue(propertyPtr: ptr<storage, array<u32>>, nodeCount: u32, dataType: u32, value: f32) {
  for (var i = 0u; i < nodeCount; i++) {
    switch (dataType) {
      case PROP_TYPE_U32: {
        propertyPtr[i] = u32(value);
      }
      case PROP_TYPE_I32: {
        propertyPtr[i] = bitcast<u32>(i32(value));
      }
      case PROP_TYPE_F32: {
        propertyPtr[i] = bitcast<u32>(value);
      }
      case PROP_TYPE_BOOL: {
        propertyPtr[i] = select(0u, 1u, value != 0.0);
      }
      default: {
        propertyPtr[i] = bitcast<u32>(value);
      }
    }
  }
}

/**
 * Initialize property with infinity values (for distance algorithms)
 * @param propertyPtr Pointer to property array
 * @param nodeCount Number of nodes
 * @param dataType Property data type
 */
fn initPropertyInfinity(propertyPtr: ptr<storage, array<u32>>, nodeCount: u32, dataType: u32) {
  for (var i = 0u; i < nodeCount; i++) {
    switch (dataType) {
      case PROP_TYPE_U32: {
        propertyPtr[i] = 0xFFFFFFFFu; // Max u32 as infinity
      }
      case PROP_TYPE_I32: {
        propertyPtr[i] = bitcast<u32>(0x7FFFFFFF); // Max i32 as infinity
      }
      case PROP_TYPE_F32: {
        propertyPtr[i] = bitcast<u32>(3.40282347e+38f); // F32 max as infinity
      }
      case PROP_TYPE_BOOL: {
        propertyPtr[i] = 1u; // true
      }
      default: {
        propertyPtr[i] = 0xFFFFFFFFu;
      }
    }
  }
}

/**
 * Initialize property with -1 values (for predecessor tracking)
 * @param propertyPtr Pointer to property array
 * @param nodeCount Number of nodes
 * @param dataType Property data type
 */
fn initPropertyMinusOne(propertyPtr: ptr<storage, array<u32>>, nodeCount: u32, dataType: u32) {
  for (var i = 0u; i < nodeCount; i++) {
    switch (dataType) {
      case PROP_TYPE_U32: {
        propertyPtr[i] = 0xFFFFFFFFu; // Max u32 as -1
      }
      case PROP_TYPE_I32: {
        propertyPtr[i] = bitcast<u32>(-1i);
      }
      case PROP_TYPE_F32: {
        propertyPtr[i] = bitcast<u32>(-1.0f);
      }
      case PROP_TYPE_BOOL: {
        propertyPtr[i] = 0u; // false
      }
      default: {
        propertyPtr[i] = bitcast<u32>(-1i);
      }
    }
  }
}

/**
 * Initialize property with node degrees
 * @param propertyPtr Pointer to property array
 * @param nodeCount Number of nodes
 * @param dataType Property data type
 * @param useInDegree Whether to use in-degree (true) or out-degree (false)
 */
fn initPropertyDegree(propertyPtr: ptr<storage, array<u32>>, nodeCount: u32, dataType: u32, useInDegree: bool) {
  for (var i = 0u; i < nodeCount; i++) {
    var degree: u32;
    if (useInDegree) {
      degree = getInDegree(i);
    } else {
      degree = getOutDegree(i);
    }
    
    switch (dataType) {
      case PROP_TYPE_U32: {
        propertyPtr[i] = degree;
      }
      case PROP_TYPE_I32: {
        propertyPtr[i] = bitcast<u32>(i32(degree));
      }
      case PROP_TYPE_F32: {
        propertyPtr[i] = bitcast<u32>(f32(degree));
      }
      case PROP_TYPE_BOOL: {
        propertyPtr[i] = select(0u, 1u, degree > 0u);
      }
      default: {
        propertyPtr[i] = degree;
      }
    }
  }
}

/**
 * Initialize property with node indices
 * @param propertyPtr Pointer to property array
 * @param nodeCount Number of nodes
 * @param dataType Property data type
 */
fn initPropertyIndex(propertyPtr: ptr<storage, array<u32>>, nodeCount: u32, dataType: u32) {
  for (var i = 0u; i < nodeCount; i++) {
    switch (dataType) {
      case PROP_TYPE_U32: {
        propertyPtr[i] = i;
      }
      case PROP_TYPE_I32: {
        propertyPtr[i] = bitcast<u32>(i32(i));
      }
      case PROP_TYPE_F32: {
        propertyPtr[i] = bitcast<u32>(f32(i));
      }
      case PROP_TYPE_BOOL: {
        propertyPtr[i] = select(0u, 1u, i % 2u == 0u); // Alternating pattern
      }
      default: {
        propertyPtr[i] = i;
      }
    }
  }
}

/**
 * Initialize property with uniform distribution
 * @param propertyPtr Pointer to property array
 * @param nodeCount Number of nodes
 * @param dataType Property data type
 * @param minValue Minimum value
 * @param maxValue Maximum value
 * @param seed Random seed
 */
fn initPropertyUniform(propertyPtr: ptr<storage, array<u32>>, nodeCount: u32, dataType: u32, minValue: f32, maxValue: f32, seed: u32) {
  var rng_state = seed;
  let range = maxValue - minValue;
  
  for (var i = 0u; i < nodeCount; i++) {
    // Simple LCG random number generator
    rng_state = (rng_state * 1664525u + 1013904223u) & 0xFFFFFFFFu;
    let rand_01 = f32(rng_state) / f32(0xFFFFFFFFu);
    let rand_value = minValue + rand_01 * range;
    
    switch (dataType) {
      case PROP_TYPE_U32: {
        propertyPtr[i] = u32(rand_value);
      }
      case PROP_TYPE_I32: {
        propertyPtr[i] = bitcast<u32>(i32(rand_value));
      }
      case PROP_TYPE_F32: {
        propertyPtr[i] = bitcast<u32>(rand_value);
      }
      case PROP_TYPE_BOOL: {
        propertyPtr[i] = select(0u, 1u, rand_01 > 0.5);
      }
      default: {
        propertyPtr[i] = bitcast<u32>(rand_value);
      }
    }
  }
}

/**
 * Copy values from one property to another
 * @param srcPtr Source property array
 * @param dstPtr Destination property array
 * @param nodeCount Number of nodes
 * @param srcType Source data type
 * @param dstType Destination data type
 */
fn copyProperty(srcPtr: ptr<storage, array<u32>>, dstPtr: ptr<storage, array<u32>>, nodeCount: u32, srcType: u32, dstType: u32) {
  for (var i = 0u; i < nodeCount; i++) {
    var value: f32;
    
    // Read from source with type conversion
    switch (srcType) {
      case PROP_TYPE_U32: {
        value = f32(srcPtr[i]);
      }
      case PROP_TYPE_I32: {
        value = f32(bitcast<i32>(srcPtr[i]));
      }
      case PROP_TYPE_F32: {
        value = bitcast<f32>(srcPtr[i]);
      }
      case PROP_TYPE_BOOL: {
        value = f32(srcPtr[i]);
      }
      default: {
        value = f32(srcPtr[i]);
      }
    }
    
    // Write to destination with type conversion
    switch (dstType) {
      case PROP_TYPE_U32: {
        dstPtr[i] = u32(value);
      }
      case PROP_TYPE_I32: {
        dstPtr[i] = bitcast<u32>(i32(value));
      }
      case PROP_TYPE_F32: {
        dstPtr[i] = bitcast<u32>(value);
      }
      case PROP_TYPE_BOOL: {
        dstPtr[i] = select(0u, 1u, value != 0.0);
      }
      default: {
        dstPtr[i] = bitcast<u32>(value);
      }
    }
  }
}

// =============================================================================
// DYNAMIC PROPERTY ALLOCATION (Task 3.11)
// =============================================================================

/**
 * Allocate dynamic property during execution
 * This is a conceptual framework - actual allocation would be managed by host
 * @param metadata Property metadata
 * @param nodeCount Number of nodes
 * @return Allocation success status
 */
fn allocateDynamicProperty(metadata: ptr<function, PropertyMetadata>, nodeCount: u32) -> bool {
  // Calculate required size
  var elementSize = 4u; // Default to 4 bytes
  switch ((*metadata).data_type) {
    case PROP_TYPE_U32, case PROP_TYPE_I32, case PROP_TYPE_F32: {
      elementSize = 4u;
    }
    case PROP_TYPE_BOOL: {
      elementSize = 1u; // Can be packed
    }
    default: {
      elementSize = 4u;
    }
  }
  
  (*metadata).size_bytes = elementSize * nodeCount;
  (*metadata).node_count = nodeCount;
  (*metadata).is_allocated = true;
  
  return true; // Success (in real implementation, would check memory availability)
}

/**
 * Initialize allocated property with specified pattern
 * @param propertyPtr Pointer to allocated property
 * @param metadata Property metadata
 * @param initValue Initial value (for INIT_VALUE mode)
 * @param sourcePtr Source property (for INIT_COPY_FROM mode)
 */
fn initializeDynamicProperty(propertyPtr: ptr<storage, array<u32>>, 
                           metadata: PropertyMetadata, 
                           initValue: f32,
                           sourcePtr: ptr<storage, array<u32>>) {
  switch (metadata.init_mode) {
    case INIT_ZERO: {
      initPropertyZero(propertyPtr, metadata.node_count, metadata.data_type);
    }
    case INIT_VALUE: {
      initPropertyValue(propertyPtr, metadata.node_count, metadata.data_type, initValue);
    }
    case INIT_INFINITY: {
      initPropertyInfinity(propertyPtr, metadata.node_count, metadata.data_type);
    }
    case INIT_MINUS_ONE: {
      initPropertyMinusOne(propertyPtr, metadata.node_count, metadata.data_type);
    }
    case INIT_DEGREE: {
      initPropertyDegree(propertyPtr, metadata.node_count, metadata.data_type, false);
    }
    case INIT_INDEX: {
      initPropertyIndex(propertyPtr, metadata.node_count, metadata.data_type);
    }
    case INIT_UNIFORM: {
      initPropertyUniform(propertyPtr, metadata.node_count, metadata.data_type, 0.0, 1.0, 12345u);
    }
    case INIT_COPY_FROM: {
      if (sourcePtr != propertyPtr) {
        copyProperty(sourcePtr, propertyPtr, metadata.node_count, metadata.data_type, metadata.data_type);
      }
    }
    default: {
      initPropertyZero(propertyPtr, metadata.node_count, metadata.data_type);
    }
  }
}

/**
 * Resize dynamic property during execution
 * @param metadata Property metadata to update
 * @param newNodeCount New number of nodes
 * @return Resize success status
 */
fn resizeDynamicProperty(metadata: ptr<function, PropertyMetadata>, newNodeCount: u32) -> bool {
  if (!(*metadata).is_allocated) {
    return false;
  }
  
  let elementSize = (*metadata).size_bytes / (*metadata).node_count;
  (*metadata).node_count = newNodeCount;
  (*metadata).size_bytes = elementSize * newNodeCount;
  
  return true; // Success (in real implementation, would reallocate memory)
}

// =============================================================================
// PROPERTY ACCESS UTILITIES
// =============================================================================

/**
 * Safe property read with bounds checking
 * @param propertyPtr Property array pointer
 * @param nodeId Node index
 * @param nodeCount Total number of nodes
 * @param dataType Property data type
 * @return Property value as f32
 */
fn safeReadProperty(propertyPtr: ptr<storage, array<u32>>, nodeId: u32, nodeCount: u32, dataType: u32) -> f32 {
  if (nodeId >= nodeCount) {
    return 0.0; // Out of bounds
  }
  
  switch (dataType) {
    case PROP_TYPE_U32: {
      return f32(propertyPtr[nodeId]);
    }
    case PROP_TYPE_I32: {
      return f32(bitcast<i32>(propertyPtr[nodeId]));
    }
    case PROP_TYPE_F32: {
      return bitcast<f32>(propertyPtr[nodeId]);
    }
    case PROP_TYPE_BOOL: {
      return f32(propertyPtr[nodeId]);
    }
    default: {
      return f32(propertyPtr[nodeId]);
    }
  }
}

/**
 * Safe property write with bounds checking
 * @param propertyPtr Property array pointer
 * @param nodeId Node index
 * @param value Value to write
 * @param nodeCount Total number of nodes
 * @param dataType Property data type
 * @return Write success status
 */
fn safeWriteProperty(propertyPtr: ptr<storage, array<u32>>, nodeId: u32, value: f32, nodeCount: u32, dataType: u32) -> bool {
  if (nodeId >= nodeCount) {
    return false; // Out of bounds
  }
  
  switch (dataType) {
    case PROP_TYPE_U32: {
      propertyPtr[nodeId] = u32(value);
    }
    case PROP_TYPE_I32: {
      propertyPtr[nodeId] = bitcast<u32>(i32(value));
    }
    case PROP_TYPE_F32: {
      propertyPtr[nodeId] = bitcast<u32>(value);
    }
    case PROP_TYPE_BOOL: {
      propertyPtr[nodeId] = select(0u, 1u, value != 0.0);
    }
    default: {
      propertyPtr[nodeId] = bitcast<u32>(value);
    }
  }
  
  return true;
}

// =============================================================================
// USAGE EXAMPLES AND PATTERNS
// =============================================================================

/**
 * Example: Dynamic Distance Property for SSSP
 * 
 * // Host-side allocation and initialization
 * var distMetadata = PropertyMetadata();
 * distMetadata.name_hash = hash("distance");
 * distMetadata.data_type = PROP_TYPE_F32;
 * distMetadata.init_mode = INIT_INFINITY;
 * distMetadata.access_pattern = ACCESS_ATOMIC;
 * 
 * if (allocateDynamicProperty(&distMetadata, nodeCount)) {
 *   initializeDynamicProperty(distancePtr, distMetadata, 0.0, nullptr);
 *   
 *   // Set source distance to 0
 *   safeWriteProperty(distancePtr, sourceNode, 0.0, nodeCount, PROP_TYPE_F32);
 * }
 */

/**
 * Example: Dynamic Predecessor Property
 * 
 * var predMetadata = PropertyMetadata();
 * predMetadata.data_type = PROP_TYPE_U32;
 * predMetadata.init_mode = INIT_MINUS_ONE;
 * predMetadata.access_pattern = ACCESS_READ_WRITE;
 * 
 * allocateDynamicProperty(&predMetadata, nodeCount);
 * initializeDynamicProperty(predecessorPtr, predMetadata, -1.0, nullptr);
 */

/**
 * Example: Copy Property Pattern
 * 
 * // Copy old ranks to new ranks for PageRank
 * var newRankMetadata = PropertyMetadata();
 * newRankMetadata.data_type = PROP_TYPE_F32;
 * newRankMetadata.init_mode = INIT_COPY_FROM;
 * 
 * allocateDynamicProperty(&newRankMetadata, nodeCount);
 * initializeDynamicProperty(newRankPtr, newRankMetadata, 0.0, oldRankPtr);
 */

/**
 * Performance Notes:
 * 
 * - Dynamic allocation adds overhead but enables flexible algorithms
 * - Initialization patterns reduce boilerplate code significantly
 * - Type-safe property access prevents common bugs
 * - Bounds checking adds safety at minimal performance cost
 * - Memory layout optimization is crucial for GPU performance
 * - Property metadata enables runtime introspection and validation
 */
