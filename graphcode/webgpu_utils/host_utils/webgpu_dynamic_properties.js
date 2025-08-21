/**
 * StarPlat WebGPU Dynamic Property Management - Host Side
 * 
 * Host-side utilities for dynamic property allocation, initialization,
 * and management during algorithm execution. Complements the WGSL
 * dynamic property utilities.
 * 
 * Version: 1.0 (Phase 3.9, 3.11)
 */

// =============================================================================
// CONSTANTS AND TYPES
// =============================================================================

export const PropertyDataTypes = {
  U32: 0,
  I32: 1,
  F32: 2,
  BOOL: 3
};

export const InitializationModes = {
  ZERO: 0,
  VALUE: 1,
  INFINITY: 2,
  MINUS_ONE: 3,
  RANDOM: 4,
  DEGREE: 5,
  INDEX: 6,
  UNIFORM: 7,
  COPY_FROM: 8,
  FORMULA: 9
};

export const AccessPatterns = {
  READ_ONLY: 0,
  WRITE_ONLY: 1,
  READ_WRITE: 2,
  ATOMIC: 3
};

// =============================================================================
// DYNAMIC PROPERTY MANAGER
// =============================================================================

export class DynamicPropertyManager {
  constructor(device, nodeCount) {
    this.device = device;
    this.nodeCount = nodeCount;
    this.properties = new Map(); // name -> PropertyInfo
    this.allocatedBuffers = new Map(); // name -> GPUBuffer
    this.propertyMetadata = new Map(); // name -> metadata object
    this.totalAllocatedBytes = 0;
    this.maxAllowedBytes = 1024 * 1024 * 1024; // 1GB limit
  }

  /**
   * Create a new dynamic property
   * @param {string} name Property name
   * @param {Object} config Property configuration
   * @returns {Promise<boolean>} Success status
   */
  async createProperty(name, config = {}) {
    const {
      dataType = PropertyDataTypes.F32,
      initMode = InitializationModes.ZERO,
      accessPattern = AccessPatterns.READ_WRITE,
      initValue = 0.0,
      nodeCount = this.nodeCount,
      copyFromProperty = null,
      minValue = 0.0,
      maxValue = 1.0,
      randomSeed = Date.now()
    } = config;

    try {
      // Check if property already exists
      if (this.properties.has(name)) {
        console.warn(`Property '${name}' already exists, skipping creation`);
        return false;
      }

      // Calculate buffer size
      const elementSize = this.getElementSize(dataType);
      const bufferSize = elementSize * nodeCount;
      
      // Check memory limits
      if (this.totalAllocatedBytes + bufferSize > this.maxAllowedBytes) {
        throw new Error(`Memory limit exceeded: would allocate ${bufferSize} bytes, total would be ${this.totalAllocatedBytes + bufferSize}`);
      }

      // Create GPU buffer
      const buffer = this.device.createBuffer({
        size: Math.max(bufferSize, 4), // Minimum 4 bytes
        usage: this.getBufferUsage(accessPattern),
        label: `DynamicProperty_${name}`
      });

      // Initialize buffer data
      const initialData = await this.generateInitialData(
        dataType, initMode, nodeCount, initValue, 
        copyFromProperty, minValue, maxValue, randomSeed
      );

      if (initialData) {
        this.device.queue.writeBuffer(buffer, 0, initialData);
      }

      // Store property information
      const propertyInfo = {
        name,
        dataType,
        initMode,
        accessPattern,
        nodeCount,
        bufferSize,
        elementSize,
        buffer,
        createdAt: Date.now(),
        lastAccessed: Date.now(),
        accessCount: 0
      };

      this.properties.set(name, propertyInfo);
      this.allocatedBuffers.set(name, buffer);
      this.totalAllocatedBytes += bufferSize;

      console.log(`[DynamicProperty] Created '${name}': ${nodeCount} elements, ${bufferSize} bytes, type=${dataType}, init=${initMode}`);
      return true;

    } catch (error) {
      console.error(`[DynamicProperty] Failed to create property '${name}':`, error);
      return false;
    }
  }

  /**
   * Resize an existing property
   * @param {string} name Property name
   * @param {number} newNodeCount New number of nodes
   * @returns {Promise<boolean>} Success status
   */
  async resizeProperty(name, newNodeCount) {
    try {
      const propertyInfo = this.properties.get(name);
      if (!propertyInfo) {
        throw new Error(`Property '${name}' does not exist`);
      }

      const newBufferSize = propertyInfo.elementSize * newNodeCount;
      const sizeDifference = newBufferSize - propertyInfo.bufferSize;

      // Check memory limits for expansion
      if (sizeDifference > 0 && this.totalAllocatedBytes + sizeDifference > this.maxAllowedBytes) {
        throw new Error(`Memory limit exceeded during resize of '${name}'`);
      }

      // Create new buffer
      const newBuffer = this.device.createBuffer({
        size: Math.max(newBufferSize, 4),
        usage: this.getBufferUsage(propertyInfo.accessPattern),
        label: `DynamicProperty_${name}_resized`
      });

      // Copy existing data if shrinking or preserve data during expansion
      if (newNodeCount < propertyInfo.nodeCount) {
        // Shrinking: copy only the needed portion
        const copySize = newBufferSize;
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(propertyInfo.buffer, 0, newBuffer, 0, copySize);
        this.device.queue.submit([encoder.finish()]);
      } else if (newNodeCount > propertyInfo.nodeCount) {
        // Expanding: copy existing data and initialize new elements
        const existingSize = propertyInfo.bufferSize;
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(propertyInfo.buffer, 0, newBuffer, 0, existingSize);
        this.device.queue.submit([encoder.finish()]);

        // Initialize new elements
        const newElementCount = newNodeCount - propertyInfo.nodeCount;
        const newElementsData = await this.generateInitialData(
          propertyInfo.dataType, propertyInfo.initMode, newElementCount, 0.0
        );
        
        if (newElementsData) {
          this.device.queue.writeBuffer(newBuffer, existingSize, newElementsData);
        }
      } else {
        // Same size: just copy
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(propertyInfo.buffer, 0, newBuffer, 0, propertyInfo.bufferSize);
        this.device.queue.submit([encoder.finish()]);
      }

      // Update tracking
      this.totalAllocatedBytes += sizeDifference;
      
      // Destroy old buffer and update references
      propertyInfo.buffer.destroy();
      propertyInfo.buffer = newBuffer;
      propertyInfo.nodeCount = newNodeCount;
      propertyInfo.bufferSize = newBufferSize;
      propertyInfo.lastAccessed = Date.now();

      this.allocatedBuffers.set(name, newBuffer);

      console.log(`[DynamicProperty] Resized '${name}': ${newNodeCount} elements, ${newBufferSize} bytes`);
      return true;

    } catch (error) {
      console.error(`[DynamicProperty] Failed to resize property '${name}':`, error);
      return false;
    }
  }

  /**
   * Get property buffer
   * @param {string} name Property name
   * @returns {GPUBuffer|null} Property buffer or null if not found
   */
  getPropertyBuffer(name) {
    const buffer = this.allocatedBuffers.get(name);
    if (buffer) {
      const propertyInfo = this.properties.get(name);
      if (propertyInfo) {
        propertyInfo.lastAccessed = Date.now();
        propertyInfo.accessCount++;
      }
    }
    return buffer || null;
  }

  /**
   * Copy property values from one property to another
   * @param {string} sourceName Source property name
   * @param {string} targetName Target property name
   * @returns {boolean} Success status
   */
  copyProperty(sourceName, targetName) {
    try {
      const sourceInfo = this.properties.get(sourceName);
      const targetInfo = this.properties.get(targetName);

      if (!sourceInfo || !targetInfo) {
        throw new Error(`Source or target property not found: '${sourceName}' -> '${targetName}'`);
      }

      const minNodeCount = Math.min(sourceInfo.nodeCount, targetInfo.nodeCount);
      const copySize = minNodeCount * Math.min(sourceInfo.elementSize, targetInfo.elementSize);

      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(sourceInfo.buffer, 0, targetInfo.buffer, 0, copySize);
      this.device.queue.submit([encoder.finish()]);

      console.log(`[DynamicProperty] Copied '${sourceName}' to '${targetName}': ${copySize} bytes`);
      return true;

    } catch (error) {
      console.error(`[DynamicProperty] Failed to copy property:`, error);
      return false;
    }
  }

  /**
   * Read property values back to CPU
   * @param {string} name Property name
   * @returns {Promise<TypedArray|null>} Property values or null if failed
   */
  async readProperty(name) {
    try {
      const propertyInfo = this.properties.get(name);
      if (!propertyInfo) {
        throw new Error(`Property '${name}' does not exist`);
      }

      const readBuffer = this.device.createBuffer({
        size: propertyInfo.bufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });

      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(propertyInfo.buffer, 0, readBuffer, 0, propertyInfo.bufferSize);
      this.device.queue.submit([encoder.finish()]);

      await readBuffer.mapAsync(GPUMapMode.READ);
      const arrayBuffer = readBuffer.getMappedRange();
      
      // Create appropriate typed array based on data type
      let result;
      switch (propertyInfo.dataType) {
        case PropertyDataTypes.U32:
          result = new Uint32Array(arrayBuffer.slice());
          break;
        case PropertyDataTypes.I32:
          result = new Int32Array(arrayBuffer.slice());
          break;
        case PropertyDataTypes.F32:
          result = new Float32Array(arrayBuffer.slice());
          break;
        case PropertyDataTypes.BOOL:
          result = new Uint8Array(arrayBuffer.slice());
          break;
        default:
          result = new Uint32Array(arrayBuffer.slice());
      }

      readBuffer.unmap();
      readBuffer.destroy();

      propertyInfo.lastAccessed = Date.now();
      propertyInfo.accessCount++;

      return result;

    } catch (error) {
      console.error(`[DynamicProperty] Failed to read property '${name}':`, error);
      return null;
    }
  }

  /**
   * Write values to property
   * @param {string} name Property name
   * @param {TypedArray} data Data to write
   * @returns {boolean} Success status
   */
  writeProperty(name, data) {
    try {
      const propertyInfo = this.properties.get(name);
      if (!propertyInfo) {
        throw new Error(`Property '${name}' does not exist`);
      }

      const dataSize = data.byteLength;
      if (dataSize > propertyInfo.bufferSize) {
        throw new Error(`Data size (${dataSize}) exceeds property buffer size (${propertyInfo.bufferSize})`);
      }

      this.device.queue.writeBuffer(propertyInfo.buffer, 0, data);
      
      propertyInfo.lastAccessed = Date.now();
      propertyInfo.accessCount++;

      console.log(`[DynamicProperty] Wrote ${dataSize} bytes to '${name}'`);
      return true;

    } catch (error) {
      console.error(`[DynamicProperty] Failed to write property '${name}':`, error);
      return false;
    }
  }

  /**
   * Remove property and free memory
   * @param {string} name Property name
   * @returns {boolean} Success status
   */
  removeProperty(name) {
    try {
      const propertyInfo = this.properties.get(name);
      if (!propertyInfo) {
        console.warn(`Property '${name}' does not exist, skipping removal`);
        return false;
      }

      propertyInfo.buffer.destroy();
      this.totalAllocatedBytes -= propertyInfo.bufferSize;
      
      this.properties.delete(name);
      this.allocatedBuffers.delete(name);

      console.log(`[DynamicProperty] Removed '${name}': freed ${propertyInfo.bufferSize} bytes`);
      return true;

    } catch (error) {
      console.error(`[DynamicProperty] Failed to remove property '${name}':`, error);
      return false;
    }
  }

  /**
   * Get property statistics
   * @returns {Object} Statistics about all properties
   */
  getStatistics() {
    const properties = Array.from(this.properties.values());
    
    return {
      totalProperties: properties.length,
      totalAllocatedBytes: this.totalAllocatedBytes,
      memoryUtilization: this.totalAllocatedBytes / this.maxAllowedBytes,
      properties: properties.map(prop => ({
        name: prop.name,
        dataType: prop.dataType,
        nodeCount: prop.nodeCount,
        bufferSize: prop.bufferSize,
        accessCount: prop.accessCount,
        lastAccessed: prop.lastAccessed
      }))
    };
  }

  /**
   * Cleanup all properties
   */
  cleanup() {
    for (const [name, propertyInfo] of this.properties) {
      propertyInfo.buffer.destroy();
    }
    
    this.properties.clear();
    this.allocatedBuffers.clear();
    this.totalAllocatedBytes = 0;
    
    console.log(`[DynamicProperty] Cleaned up all properties`);
  }

  // =============================================================================
  // PRIVATE HELPER METHODS
  // =============================================================================

  getElementSize(dataType) {
    switch (dataType) {
      case PropertyDataTypes.U32:
      case PropertyDataTypes.I32:
      case PropertyDataTypes.F32:
        return 4;
      case PropertyDataTypes.BOOL:
        return 1;
      default:
        return 4;
    }
  }

  getBufferUsage(accessPattern) {
    let usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    
    // Add additional usage flags based on access pattern
    if (accessPattern === AccessPatterns.READ_ONLY) {
      // Read-only storage
    } else if (accessPattern === AccessPatterns.WRITE_ONLY) {
      // Write-only storage  
    } else if (accessPattern === AccessPatterns.ATOMIC) {
      // Atomic operations require read-write storage
    }
    
    return usage;
  }

  async generateInitialData(dataType, initMode, nodeCount, initValue = 0.0, copyFromProperty = null, minValue = 0.0, maxValue = 1.0, randomSeed = Date.now()) {
    const elementSize = this.getElementSize(dataType);
    const bufferSize = elementSize * nodeCount;
    
    switch (initMode) {
      case InitializationModes.ZERO:
        return new ArrayBuffer(bufferSize); // Already zeroed
        
      case InitializationModes.VALUE:
        return this.createValueArray(dataType, nodeCount, initValue);
        
      case InitializationModes.INFINITY:
        return this.createInfinityArray(dataType, nodeCount);
        
      case InitializationModes.MINUS_ONE:
        return this.createValueArray(dataType, nodeCount, -1);
        
      case InitializationModes.INDEX:
        return this.createIndexArray(dataType, nodeCount);
        
      case InitializationModes.UNIFORM:
        return this.createUniformArray(dataType, nodeCount, minValue, maxValue, randomSeed);
        
      case InitializationModes.COPY_FROM:
        if (copyFromProperty && this.properties.has(copyFromProperty)) {
          return await this.readProperty(copyFromProperty);
        }
        return new ArrayBuffer(bufferSize);
        
      default:
        return new ArrayBuffer(bufferSize);
    }
  }

  createValueArray(dataType, nodeCount, value) {
    switch (dataType) {
      case PropertyDataTypes.U32: {
        const arr = new Uint32Array(nodeCount);
        arr.fill(Math.max(0, Math.floor(value)));
        return arr;
      }
      case PropertyDataTypes.I32: {
        const arr = new Int32Array(nodeCount);
        arr.fill(Math.floor(value));
        return arr;
      }
      case PropertyDataTypes.F32: {
        const arr = new Float32Array(nodeCount);
        arr.fill(value);
        return arr;
      }
      case PropertyDataTypes.BOOL: {
        const arr = new Uint8Array(nodeCount);
        arr.fill(value !== 0 ? 1 : 0);
        return arr;
      }
      default:
        return new ArrayBuffer(nodeCount * 4);
    }
  }

  createInfinityArray(dataType, nodeCount) {
    switch (dataType) {
      case PropertyDataTypes.U32:
        return this.createValueArray(dataType, nodeCount, 0xFFFFFFFF);
      case PropertyDataTypes.I32:
        return this.createValueArray(dataType, nodeCount, 0x7FFFFFFF);
      case PropertyDataTypes.F32:
        return this.createValueArray(dataType, nodeCount, Number.MAX_VALUE);
      case PropertyDataTypes.BOOL:
        return this.createValueArray(dataType, nodeCount, 1);
      default:
        return this.createValueArray(dataType, nodeCount, 0xFFFFFFFF);
    }
  }

  createIndexArray(dataType, nodeCount) {
    switch (dataType) {
      case PropertyDataTypes.U32: {
        const arr = new Uint32Array(nodeCount);
        for (let i = 0; i < nodeCount; i++) arr[i] = i;
        return arr;
      }
      case PropertyDataTypes.I32: {
        const arr = new Int32Array(nodeCount);
        for (let i = 0; i < nodeCount; i++) arr[i] = i;
        return arr;
      }
      case PropertyDataTypes.F32: {
        const arr = new Float32Array(nodeCount);
        for (let i = 0; i < nodeCount; i++) arr[i] = i;
        return arr;
      }
      case PropertyDataTypes.BOOL: {
        const arr = new Uint8Array(nodeCount);
        for (let i = 0; i < nodeCount; i++) arr[i] = i % 2;
        return arr;
      }
      default:
        return this.createIndexArray(PropertyDataTypes.U32, nodeCount);
    }
  }

  createUniformArray(dataType, nodeCount, minValue, maxValue, seed) {
    const range = maxValue - minValue;
    let rngState = seed;
    
    // Simple LCG random number generator (same as WGSL version)
    function random() {
      rngState = ((rngState * 1664525 + 1013904223) >>> 0) & 0xFFFFFFFF;
      return rngState / 0xFFFFFFFF;
    }
    
    switch (dataType) {
      case PropertyDataTypes.U32: {
        const arr = new Uint32Array(nodeCount);
        for (let i = 0; i < nodeCount; i++) {
          arr[i] = Math.floor(minValue + random() * range);
        }
        return arr;
      }
      case PropertyDataTypes.I32: {
        const arr = new Int32Array(nodeCount);
        for (let i = 0; i < nodeCount; i++) {
          arr[i] = Math.floor(minValue + random() * range);
        }
        return arr;
      }
      case PropertyDataTypes.F32: {
        const arr = new Float32Array(nodeCount);
        for (let i = 0; i < nodeCount; i++) {
          arr[i] = minValue + random() * range;
        }
        return arr;
      }
      case PropertyDataTypes.BOOL: {
        const arr = new Uint8Array(nodeCount);
        for (let i = 0; i < nodeCount; i++) {
          arr[i] = random() > 0.5 ? 1 : 0;
        }
        return arr;
      }
      default:
        return this.createUniformArray(PropertyDataTypes.F32, nodeCount, minValue, maxValue, seed);
    }
  }
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Create a dynamic property manager
 * @param {GPUDevice} device WebGPU device
 * @param {number} nodeCount Number of nodes
 * @returns {DynamicPropertyManager} Property manager instance
 */
export function createDynamicPropertyManager(device, nodeCount) {
  return new DynamicPropertyManager(device, nodeCount);
}

/**
 * Usage Examples:
 * 
 * // Create property manager
 * const propManager = new DynamicPropertyManager(device, nodeCount);
 * 
 * // Create distance property for SSSP
 * await propManager.createProperty('distance', {
 *   dataType: PropertyDataTypes.F32,
 *   initMode: InitializationModes.INFINITY,
 *   accessPattern: AccessPatterns.ATOMIC
 * });
 * 
 * // Create predecessor property
 * await propManager.createProperty('predecessor', {
 *   dataType: PropertyDataTypes.U32,
 *   initMode: InitializationModes.MINUS_ONE,
 *   accessPattern: AccessPatterns.READ_WRITE
 * });
 * 
 * // Get buffer for use in compute shader
 * const distBuffer = propManager.getPropertyBuffer('distance');
 * 
 * // Read results back
 * const distances = await propManager.readProperty('distance');
 * console.log('Final distances:', distances);
 * 
 * // Cleanup
 * propManager.cleanup();
 */
