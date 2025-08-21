/**
 * StarPlat WebGPU Buffer Management Utilities
 * 
 * Centralized buffer creation, management, and operations extracted from 
 * generator patterns. Provides consistent buffer handling with automatic
 * cleanup, validation, and performance optimization.
 * 
 * Replaces repetitive generator code like:
 * - Lines 125-140: Buffer creation patterns
 * - Lines 147-157: Property readback logic
 * - Property buffer management throughout generator
 * 
 * Version: 1.0 (Phase 3.17)
 */

export class WebGPUBufferUtils {
  constructor(device) {
    this.device = device;
    this.createdBuffers = new Set(); // Track for cleanup
  }

  /**
   * Create storage buffer with data (extracted from generator pattern)
   * Replaces: device.createBuffer({ size: ..., usage: ..., mappedAtCreation: true })
   * 
   * @param {ArrayBuffer|TypedArray} data Data to store in buffer
   * @param {number} usage Additional usage flags (default: STORAGE | COPY_DST)
   * @returns {GPUBuffer} Created buffer with data
   */
  createStorageBuffer(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    
    // Copy data efficiently based on type
    const TypedArrayConstructor = data.constructor;
    new TypedArrayConstructor(buffer.getMappedRange()).set(data);
    buffer.unmap();
    
    this.createdBuffers.add(buffer);
    return buffer;
  }

  /**
   * Create empty storage buffer (extracted from generator property patterns)
   * Replaces generator lines 125, 134: resultBuffer/propertyBuffer creation
   * 
   * @param {number} sizeBytes Buffer size in bytes
   * @param {number} usage Usage flags
   * @returns {GPUBuffer} Empty buffer ready for GPU operations
   */
  createEmptyStorageBuffer(sizeBytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    const buffer = this.device.createBuffer({
      size: Math.max(sizeBytes, 4), // Ensure minimum size
      usage
    });
    
    this.createdBuffers.add(buffer);
    return buffer;
  }

  /**
   * Create uniform buffer (extracted from generator params pattern)
   * Replaces generator line 138: paramsBuffer creation
   * 
   * @param {ArrayBuffer|TypedArray} data Uniform data
   * @returns {GPUBuffer} Uniform buffer
   */
  createUniformBuffer(data) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    
    const TypedArrayConstructor = data.constructor;
    new TypedArrayConstructor(buffer.getMappedRange()).set(data);
    buffer.unmap();
    
    this.createdBuffers.add(buffer);
    return buffer;
  }

  /**
   * Write data to existing buffer (extracted from generator writeBuffer pattern)
   * Replaces generator line 136, 139: device.queue.writeBuffer calls
   * 
   * @param {GPUBuffer} buffer Target buffer
   * @param {ArrayBuffer|TypedArray} data Data to write
   * @param {number} offset Byte offset (default: 0)
   */
  writeBuffer(buffer, data, offset = 0) {
    this.device.queue.writeBuffer(buffer, offset, data);
  }

  /**
   * Create readback buffer for property copy-back (extracted from lines 147-157)
   * Replaces generator property readback pattern with cleaner interface
   * 
   * @param {number} sizeBytes Buffer size in bytes  
   * @returns {GPUBuffer} Buffer configured for CPU readback
   */
  createReadbackBuffer(sizeBytes) {
    const buffer = this.device.createBuffer({
      size: sizeBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    
    this.createdBuffers.add(buffer);
    return buffer;
  }

  /**
   * Copy buffer contents for readback (extracted from generator pattern)
   * Replaces generator lines 150: enc.copyBufferToBuffer pattern
   * 
   * @param {GPUBuffer} source Source buffer
   * @param {GPUBuffer} destination Readback buffer  
   * @param {number} size Bytes to copy (default: full source size)
   */
  copyBufferToReadback(source, destination, size = source.size) {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, destination, 0, size);
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Read data from GPU buffer (extracted from generator readback pattern)
   * Replaces generator lines 151-154: mapAsync + getMappedRange pattern
   * 
   * @param {GPUBuffer} readbackBuffer Buffer to read from
   * @param {Function} TypedArrayConstructor Constructor for result array
   * @returns {Promise<TypedArray>} Data read from GPU
   */
  async readBuffer(readbackBuffer, TypedArrayConstructor = Uint32Array) {
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const view = new TypedArrayConstructor(readbackBuffer.getMappedRange());
    const result = new TypedArrayConstructor(view); // Copy data
    readbackBuffer.unmap();
    return result;
  }

  /**
   * Comprehensive property buffer management (extracted from generator PropInfo pattern)
   * Replaces generator lines 127-135: property buffer creation and management
   */
  createPropertyBuffers(properties, nodeCount) {
    const buffers = {};
    const bindingEntries = [];
    
    for (const [name, propSpec] of Object.entries(properties)) {
      let buffer;
      
      if (propSpec.buffer) {
        // Use provided GPU buffer
        buffer = propSpec.buffer;
      } else if (propSpec.data) {
        // Create buffer from provided data
        buffer = this.createStorageBuffer(propSpec.data);
      } else {
        // Create empty buffer with specified or default size
        const elementSize = this._getElementSize(propSpec.type || 'u32');
        const sizeBytes = propSpec.sizeBytes || (nodeCount * elementSize);
        buffer = this.createEmptyStorageBuffer(sizeBytes);
        
        // Initialize with initial value if specified
        if (propSpec.initial !== undefined) {
          const TypedArray = this._getTypedArray(propSpec.type || 'u32');
          const initData = new TypedArray(sizeBytes / elementSize);
          initData.fill(propSpec.initial);
          this.writeBuffer(buffer, initData);
        }
      }
      
      buffers[name] = buffer;
      
      // Create binding entry for bind group
      bindingEntries.push({
        binding: propSpec.binding || (6 + Object.keys(buffers).length - 1),
        resource: { buffer }
      });
    }
    
    return { buffers, bindingEntries };
  }

  /**
   * Selective property readback (implements Task 3.4 - Selective copy-back)
   * Replaces generator lines 147-157 with more flexible interface
   * 
   * @param {Object} propertyBuffers Map of property name to buffer
   * @param {Object} readbackSpecs Specification of which properties to read back
   * @param {number} nodeCount Number of nodes for size calculation
   * @returns {Promise<Object>} Map of property name to read data
   */
  async readbackProperties(propertyBuffers, readbackSpecs, nodeCount) {
    const results = {};
    const readbackPromises = [];
    
    for (const [propertyName, spec] of Object.entries(readbackSpecs)) {
      if (!spec.readback && spec.usage !== 'out' && spec.usage !== 'inout') {
        continue; // Skip properties that don't need readback
      }
      
      const buffer = propertyBuffers[propertyName];
      if (!buffer) {
        console.warn(`[BufferUtils] Property buffer '${propertyName}' not found for readback`);
        continue;
      }
      
      // Determine data size and type
      const elementSize = this._getElementSize(spec.type || 'u32');
      const sizeBytes = spec.sizeBytes || (nodeCount * elementSize);
      const TypedArray = this._getTypedArray(spec.type || 'u32');
      
      // Create readback buffer and copy
      const readbackBuffer = this.createReadbackBuffer(sizeBytes);
      this.copyBufferToReadback(buffer, readbackBuffer, sizeBytes);
      
      // Queue readback operation
      const promise = this.readBuffer(readbackBuffer, TypedArray)
        .then(data => {
          results[propertyName] = data;
        });
      
      readbackPromises.push(promise);
    }
    
    // Wait for all readbacks to complete
    await Promise.all(readbackPromises);
    return results;
  }

  /**
   * Create standard graph buffers (CSR format)
   * Helper for common StarPlat graph buffer setup
   * 
   * @param {Object} csr Graph in CSR format
   * @returns {Object} Standard graph buffers
   */
  createGraphBuffers(csr) {
    const buffers = {
      adj_offsets: this.createStorageBuffer(csr.forward.offsets, GPUBufferUsage.STORAGE),
      adj_data: this.createStorageBuffer(csr.forward.data, GPUBufferUsage.STORAGE),
    };
    
    // Add reverse CSR if available
    if (csr.reverse) {
      buffers.rev_adj_offsets = this.createStorageBuffer(csr.reverse.offsets, GPUBufferUsage.STORAGE);
      buffers.rev_adj_data = this.createStorageBuffer(csr.reverse.data, GPUBufferUsage.STORAGE);
    }
    
    // Add weights if available  
    if (csr.isWeighted && csr.forward.weights) {
      buffers.weights = this.createStorageBuffer(csr.forward.weights, GPUBufferUsage.STORAGE);
    }
    
    // Create parameters uniform buffer
    const params = new Uint32Array([csr.nodeCount, 0, 0, 0]);
    buffers.params = this.createUniformBuffer(params);
    
    return buffers;
  }

  /**
   * Clean up all created buffers
   * Call this when algorithm execution is complete
   */
  cleanup() {
    for (const buffer of this.createdBuffers) {
      if (buffer.destroy) {
        buffer.destroy();
      }
    }
    this.createdBuffers.clear();
  }

  /**
   * Get buffer memory usage statistics
   * @returns {Object} Memory usage information
   */
  getMemoryUsage() {
    let totalSize = 0;
    let bufferCount = 0;
    
    for (const buffer of this.createdBuffers) {
      totalSize += buffer.size;
      bufferCount++;
    }
    
    return {
      bufferCount,
      totalSizeBytes: totalSize,
      totalSizeMB: (totalSize / 1024 / 1024).toFixed(2)
    };
  }

  // Private helper methods
  
  _getElementSize(type) {
    switch (type) {
      case 'u32':
      case 'i32': 
      case 'f32':
        return 4;
      case 'u64':
      case 'i64':
      case 'f64':
        return 8;
      case 'u16':
      case 'i16':
        return 2;
      case 'u8':
      case 'i8':
        return 1;
      default:
        return 4; // Default to 32-bit
    }
  }
  
  _getTypedArray(type) {
    switch (type) {
      case 'u32': return Uint32Array;
      case 'i32': return Int32Array;
      case 'f32': return Float32Array;
      case 'u16': return Uint16Array;
      case 'i16': return Int16Array;
      case 'u8': return Uint8Array;
      case 'i8': return Int8Array;
      default: return Uint32Array; // Default
    }
  }
}

/**
 * Usage Examples:
 * 
 * // Basic buffer operations
 * const bufferUtils = new WebGPUBufferUtils(device);
 * const data = new Uint32Array([1, 2, 3, 4]);
 * const buffer = bufferUtils.createStorageBuffer(data);
 * 
 * // Property buffer management (replaces generator patterns)
 * const properties = {
 *   rank: { type: 'f32', initial: 1.0, usage: 'inout' },
 *   visited: { type: 'u32', initial: 0, usage: 'out' }
 * };
 * const { buffers, bindingEntries } = bufferUtils.createPropertyBuffers(properties, nodeCount);
 * 
 * // Selective property readback (Task 3.4)
 * const readbackSpecs = {
 *   rank: { readback: true, type: 'f32' },
 *   visited: { usage: 'out', type: 'u32' }
 * };
 * const results = await bufferUtils.readbackProperties(buffers, readbackSpecs, nodeCount);
 * 
 * // CSR graph buffer setup
 * const graphBuffers = bufferUtils.createGraphBuffers(csr);
 * 
 * // Cleanup when done
 * bufferUtils.cleanup();
 */
