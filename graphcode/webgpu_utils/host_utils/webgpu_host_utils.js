/**
 * StarPlat WebGPU Host Utilities - Main Entry Point
 * 
 * Comprehensive host-side utilities that replace generator patterns with
 * modular, reusable, and maintainable code. Combines device management,
 * buffer operations, pipeline caching, and algorithm execution patterns.
 * 
 * This is the main API that generator will use instead of inlining
 * repetitive host-side patterns (Task 3.17 completion).
 * 
 * Version: 1.0 (Phase 3.17)
 */

import { WebGPUDeviceManager, getGlobalDeviceManager, initializeGlobalDevice, checkWebGPUSupport } from './webgpu_device_manager.js';
import { WebGPUBufferUtils } from './webgpu_buffer_utils.js';
import { WebGPUPipelineManager } from './webgpu_pipeline_manager.js';

/**
 * Main WebGPU execution context that orchestrates all utilities
 * Provides high-level API for StarPlat algorithm execution
 */
export class WebGPUExecutionContext {
  constructor(device, options = {}) {
    this.device = device;
    this.options = {
      enableCaching: true,
      verbose: false,
      autoCleanup: true,
      ...options
    };
    
    // Initialize utility managers
    this.bufferUtils = new WebGPUBufferUtils(device);
    this.pipelineManager = new WebGPUPipelineManager(device);
    
    // Execution tracking
    this.executionStats = {
      algorithmsExecuted: 0,
      totalExecutionTime: 0,
      lastExecutionTime: 0
    };
  }

  /**
   * Execute a complete StarPlat algorithm with all standard patterns
   * Replaces the entire generator function generation pattern
   * 
   * @param {Object} config Algorithm execution configuration
   * @returns {Promise<*>} Algorithm result
   */
  async executeAlgorithm(config) {
    const startTime = performance.now();
    
    try {
      if (this.options.verbose) {
        console.log(`[WebGPU Execution] Starting algorithm: ${config.name || 'unnamed'}`);
      }

      // 1. Setup graph buffers (replaces generator CSR buffer creation)
      const graphBuffers = this.bufferUtils.createGraphBuffers(config.graph);
      
      // 2. Setup property buffers (replaces generator property management)
      let propertyBuffers = {};
      let propertyBindings = [];
      if (config.properties) {
        const result = this.bufferUtils.createPropertyBuffers(config.properties, config.graph.nodeCount);
        propertyBuffers = result.buffers;
        propertyBindings = result.bindingEntries;
      }
      
      // 3. Create result buffer (replaces generator result buffer pattern)
      const resultBuffer = this.bufferUtils.createEmptyStorageBuffer(4);
      this.bufferUtils.writeBuffer(resultBuffer, new Uint32Array([0]));
      
      // 4. Combine all buffers for pipeline setup
      const allBuffers = {
        ...graphBuffers,
        result: resultBuffer,
        properties: Object.keys(propertyBuffers).length > 0 ? Object.values(propertyBuffers) : null
      };
      
      // 5. Execute kernels in sequence (replaces generator kernel launch pattern)
      let result = 0;
      for (let i = 0; i < config.kernels.length; i++) {
        const kernel = config.kernels[i];
        
        if (this.options.verbose) {
          console.log(`[WebGPU Execution] Launching kernel ${i}: ${kernel.name || `kernel_${i}`}`);
        }
        
        // Handle fixed-point iterations
        if (kernel.fixedPoint) {
          result = await this._executeFixedPointKernel(kernel, allBuffers, config);
        } else {
          // Single kernel execution
          await this.pipelineManager.executeCompute(
            kernel.wgslCode,
            allBuffers,
            { nodeCount: config.graph.nodeCount },
            { shaderLabel: kernel.name || `kernel_${i}` }
          );
        }
      }
      
      // 6. Read back result (replaces generator result readback)
      const resultReadbackBuffer = this.bufferUtils.createReadbackBuffer(4);
      this.bufferUtils.copyBufferToReadback(resultBuffer, resultReadbackBuffer);
      const resultArray = await this.bufferUtils.readBuffer(resultReadbackBuffer, Uint32Array);
      result = resultArray[0];
      
      // 7. Property readback (implements Task 3.4 - Selective copy-back)
      let propertyResults = {};
      if (config.properties && config.readbackProperties) {
        propertyResults = await this.bufferUtils.readbackProperties(
          propertyBuffers,
          config.readbackProperties,
          config.graph.nodeCount
        );
      }
      
      // 8. Update statistics
      const executionTime = performance.now() - startTime;
      this.executionStats.algorithmsExecuted++;
      this.executionStats.totalExecutionTime += executionTime;
      this.executionStats.lastExecutionTime = executionTime;
      
      if (this.options.verbose) {
        console.log(`[WebGPU Execution] Algorithm completed in ${executionTime.toFixed(2)}ms`);
      }
      
      return {
        result,
        properties: propertyResults,
        executionTime,
        stats: this.getExecutionStats()
      };
      
    } catch (error) {
      console.error('[WebGPU Execution] Algorithm failed:', error);
      throw error;
    } finally {
      // Auto-cleanup if enabled
      if (this.options.autoCleanup) {
        this.cleanup();
      }
    }
  }

  /**
   * Execute fixed-point iteration pattern (replaces generator fixed-point handling)
   * @param {Object} kernel Kernel configuration with fixed-point settings
   * @param {Object} buffers All algorithm buffers
   * @param {Object} config Algorithm configuration
   * @returns {Promise<number>} Final result
   */
  async _executeFixedPointKernel(kernel, buffers, config) {
    const maxIterations = kernel.fixedPoint.maxIterations || 100;
    const tolerance = kernel.fixedPoint.tolerance || 1e-6;
    let iteration = 0;
    let converged = false;
    
    // Create convergence flag buffer
    const convergenceBuffer = this.bufferUtils.createEmptyStorageBuffer(4);
    const buffersWithFlag = { ...buffers, convergenceFlag: convergenceBuffer };
    
    if (this.options.verbose) {
      console.log(`[WebGPU Execution] Starting fixed-point iteration (max: ${maxIterations})`);
    }
    
    while (iteration < maxIterations && !converged) {
      // Reset convergence flag
      this.bufferUtils.writeBuffer(convergenceBuffer, new Uint32Array([0]));
      
      // Execute kernel iteration  
      await this.pipelineManager.executeCompute(
        kernel.wgslCode,
        buffersWithFlag,
        { nodeCount: config.graph.nodeCount },
        { shaderLabel: `${kernel.name || 'fixedpoint'}_iter_${iteration}` }
      );
      
      // Check convergence
      const flagReadbackBuffer = this.bufferUtils.createReadbackBuffer(4);
      this.bufferUtils.copyBufferToReadback(convergenceBuffer, flagReadbackBuffer);
      const flagArray = await this.bufferUtils.readBuffer(flagReadbackBuffer, Uint32Array);
      converged = flagArray[0] === 0; // 0 means converged (no changes)
      
      iteration++;
      
      if (this.options.verbose && iteration % 10 === 0) {
        console.log(`[WebGPU Execution] Fixed-point iteration ${iteration}, converged: ${converged}`);
      }
    }
    
    if (this.options.verbose) {
      const status = converged ? 'converged' : 'max iterations reached';
      console.log(`[WebGPU Execution] Fixed-point completed after ${iteration} iterations (${status})`);
    }
    
    return iteration;
  }

  /**
   * Get execution statistics
   * @returns {Object} Performance and usage statistics
   */
  getExecutionStats() {
    return {
      ...this.executionStats,
      averageExecutionTime: this.executionStats.algorithmsExecuted > 0 
        ? this.executionStats.totalExecutionTime / this.executionStats.algorithmsExecuted 
        : 0,
      cacheStats: this.pipelineManager.getCacheStats(),
      memoryUsage: this.bufferUtils.getMemoryUsage()
    };
  }

  /**
   * Clean up all resources
   */
  cleanup() {
    this.bufferUtils.cleanup();
    // Note: Pipeline caches are preserved for reuse unless explicitly cleared
  }
}

/**
 * StarPlat WebGPU Algorithm Runner - High-level API
 * Provides complete replacement for generator-produced algorithm functions
 */
export class StarPlatWebGPURunner {
  constructor(options = {}) {
    this.options = {
      deviceOptions: {
        powerPreference: 'high-performance',
        verbose: false
      },
      executionOptions: {
        enableCaching: true,
        verbose: false,
        autoCleanup: true
      },
      ...options
    };
    
    this.deviceManager = null;
    this.executionContext = null;
  }

  /**
   * Initialize the runner with WebGPU device
   * @returns {Promise<void>}
   */
  async initialize() {
    checkWebGPUSupport();
    
    this.deviceManager = new WebGPUDeviceManager(this.options.deviceOptions);
    const device = await this.deviceManager.initialize();
    
    this.executionContext = new WebGPUExecutionContext(device, this.options.executionOptions);
    
    if (this.options.deviceOptions.verbose) {
      console.log('[StarPlat WebGPU] Runner initialized successfully');
    }
  }

  /**
   * Execute StarPlat algorithm with automatic resource management
   * Complete replacement for generator-produced algorithm functions
   * 
   * @param {Object} graph Graph in CSR format
   * @param {Object} algorithmConfig Algorithm configuration
   * @returns {Promise<*>} Algorithm result
   */
  async runAlgorithm(graph, algorithmConfig) {
    if (!this.executionContext) {
      await this.initialize();
    }
    
    // Validate graph size against device limits
    this.deviceManager.validateGraphSize(graph.nodeCount, graph.edgeCount || 0);
    
    const config = {
      graph,
      name: algorithmConfig.name,
      kernels: algorithmConfig.kernels,
      properties: algorithmConfig.properties,
      readbackProperties: algorithmConfig.readbackProperties
    };
    
    return await this.executionContext.executeAlgorithm(config);
  }

  /**
   * Get comprehensive performance statistics
   * @returns {Object} Complete performance report
   */
  getPerformanceReport() {
    const executionStats = this.executionContext ? this.executionContext.getExecutionStats() : {};
    const deviceCapabilities = this.deviceManager ? this.deviceManager.getCapabilities() : {};
    const adapterInfo = this.deviceManager ? this.deviceManager.getAdapterInfo() : {};
    
    return {
      device: adapterInfo,
      capabilities: deviceCapabilities,
      execution: executionStats,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Clean up all resources
   */
  cleanup() {
    if (this.executionContext) {
      this.executionContext.cleanup();
    }
    if (this.deviceManager) {
      this.deviceManager.destroy();
    }
  }
}

// Export individual utility classes for advanced usage
export { WebGPUDeviceManager, WebGPUBufferUtils, WebGPUPipelineManager };

// Export convenience functions
export { getGlobalDeviceManager, initializeGlobalDevice, checkWebGPUSupport };

/**
 * Convenience function to create and run algorithm in one call
 * Perfect replacement for generator-produced functions
 * 
 * @param {Object} graph Graph data in CSR format
 * @param {Object} algorithmConfig Algorithm configuration  
 * @param {Object} options Runner options
 * @returns {Promise<*>} Algorithm result
 */
export async function runStarPlatAlgorithm(graph, algorithmConfig, options = {}) {
  const runner = new StarPlatWebGPURunner(options);
  try {
    const result = await runner.runAlgorithm(graph, algorithmConfig);
    return result;
  } finally {
    if (options.autoCleanup !== false) {
      runner.cleanup();
    }
  }
}

/**
 * Usage Examples - Generator Replacement Patterns:
 * 
 * // OLD: Generated algorithm function (lines 121-160)
 * // export async function Compute_TC(device, adj_dataBuffer, adj_offsetsBuffer, nodeCount, ...)
 * 
 * // NEW: StarPlat runner usage
 * const runner = new StarPlatWebGPURunner({ verbose: true });
 * const result = await runner.runAlgorithm(csr, {
 *   name: 'TriangleCounting',
 *   kernels: [{ wgslCode, name: 'triangle_counting_kernel' }],
 *   properties: {
 *     triangles: { type: 'u32', initial: 0, usage: 'out' }
 *   },
 *   readbackProperties: {
 *     triangles: { readback: true, type: 'u32' }
 *   }
 * });
 * 
 * // OR: One-line convenience function
 * const result = await runStarPlatAlgorithm(csr, algorithmConfig);
 * 
 * // Access results
 * console.log('Triangle count:', result.result);
 * console.log('Execution time:', result.executionTime, 'ms');
 * console.log('Cache hit rate:', result.stats.cacheStats.pipelineCacheHitRate);
 */
