/**
 * StarPlat WebGPU Pipeline Management and Caching Utilities
 * 
 * Implements Task 3.1 (Pipeline caching) and Task 3.2 (Auto-bind groups)
 * extracted from generator patterns. Provides shader module caching,
 * compute pipeline caching, and automatic bind group generation.
 * 
 * Replaces repetitive generator code like:
 * - Lines 166-180: Shader module creation and pipeline setup
 * - Bind group layout creation patterns
 * - Pipeline recreation on every launch
 * 
 * Version: 1.0 (Phase 3.17)
 */

export class WebGPUPipelineManager {
  constructor(device) {
    this.device = device;
    
    // Caches for performance (Task 3.1)
    this.shaderModuleCache = new Map(); // WGSL code -> GPUShaderModule
    this.pipelineCache = new Map();     // Pipeline key -> GPUComputePipeline
    this.bindGroupLayoutCache = new Map(); // Layout key -> GPUBindGroupLayout
    
    // Statistics
    this.stats = {
      shaderModuleCacheHits: 0,
      shaderModuleCacheMisses: 0,
      pipelineCacheHits: 0,
      pipelineCacheMisses: 0,
      bindGroupLayoutCacheHits: 0,
      bindGroupLayoutCacheMisses: 0
    };
  }

  /**
   * Create or retrieve cached shader module (Task 3.1)
   * Replaces generator lines 166-168: shader module creation on every launch
   * 
   * @param {string} wgslCode WGSL shader code
   * @param {string} label Optional label for debugging
   * @returns {GPUShaderModule} Shader module (cached)
   */
  async getShaderModule(wgslCode, label = null) {
    // Use content hash as cache key for consistent caching
    const cacheKey = this._hashString(wgslCode);
    
    if (this.shaderModuleCache.has(cacheKey)) {
      this.stats.shaderModuleCacheHits++;
      return this.shaderModuleCache.get(cacheKey);
    }

    // Create new shader module
    const shaderModule = this.device.createShaderModule({ 
      code: wgslCode,
      label: label || `shader_${cacheKey.substring(0, 8)}`
    });

    // Check for compilation errors (extracted from generator error handling)
    if (shaderModule.getCompilationInfo) {
      const info = await shaderModule.getCompilationInfo();
      for (const message of info.messages || []) {
        const location = message.lineNum !== undefined ? `${message.lineNum}:${message.linePos}` : '';
        const logFn = message.type === 'error' ? console.error : console.warn;
        logFn(`[WGSL ${message.type}] ${location} ${message.message}`);
        
        if (message.type === 'error') {
          throw new Error(`WGSL compilation error: ${message.message}`);
        }
      }
    }

    this.shaderModuleCache.set(cacheKey, shaderModule);
    this.stats.shaderModuleCacheMisses++;
    return shaderModule;
  }

  /**
   * Auto-generate bind group layout from usage specification (Task 3.2)  
   * Replaces manual bind group layout creation in generator
   * 
   * @param {Object} bindings Binding specification
   * @returns {GPUBindGroupLayout} Generated bind group layout
   */
  createBindGroupLayout(bindings) {
    const layoutKey = this._getBindGroupLayoutKey(bindings);
    
    if (this.bindGroupLayoutCache.has(layoutKey)) {
      this.stats.bindGroupLayoutCacheHits++;
      return this.bindGroupLayoutCache.get(layoutKey);
    }

    const entries = [];
    
    // Standard StarPlat bindings (consistent across algorithms)
    if (bindings.adj_offsets) entries.push({ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
    if (bindings.adj_data) entries.push({ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
    if (bindings.rev_adj_offsets) entries.push({ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
    if (bindings.rev_adj_data) entries.push({ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
    if (bindings.params) entries.push({ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
    if (bindings.result) entries.push({ binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
    
    // Property bindings (dynamic, starting from binding 6)
    let nextBinding = 6;
    if (bindings.properties) {
      if (Array.isArray(bindings.properties)) {
        // Array of property specifications
        for (const prop of bindings.properties) {
          entries.push({
            binding: prop.binding || nextBinding++,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { 
              type: prop.readWrite === false ? 'read-only-storage' : 'storage'
            }
          });
        }
      } else {
        // Single property buffer
        entries.push({ binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
      }
    }
    
    // Weights buffer (if present)
    if (bindings.weights) entries.push({ binding: nextBinding++, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });

    const layout = this.device.createBindGroupLayout({ entries });
    this.bindGroupLayoutCache.set(layoutKey, layout);
    this.stats.bindGroupLayoutCacheMisses++;
    return layout;
  }

  /**
   * Create or retrieve cached compute pipeline (Task 3.1)
   * Replaces generator pipeline creation on every launch
   * 
   * @param {GPUShaderModule} shaderModule Compiled shader module
   * @param {GPUBindGroupLayout} bindGroupLayout Bind group layout
   * @param {string} entryPoint Entry point function name (default: 'main')
   * @param {Object} options Pipeline options
   * @returns {GPUComputePipeline} Compute pipeline (cached)
   */
  getComputePipeline(shaderModule, bindGroupLayout, entryPoint = 'main', options = {}) {
    const pipelineKey = this._getPipelineKey(shaderModule, bindGroupLayout, entryPoint, options);
    
    if (this.pipelineCache.has(pipelineKey)) {
      this.stats.pipelineCacheHits++;
      return this.pipelineCache.get(pipelineKey);
    }

    const pipelineLayout = this.device.createPipelineLayout({ 
      bindGroupLayouts: [bindGroupLayout] 
    });

    const pipeline = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint,
        constants: options.constants || {}
      },
      label: options.label || `pipeline_${pipelineKey.substring(0, 8)}`
    });

    this.pipelineCache.set(pipelineKey, pipeline);
    this.stats.pipelineCacheMisses++;
    return pipeline;
  }

  /**
   * Create bind group from buffers and layout
   * Auto-matches buffers to binding layout (Task 3.2)
   * 
   * @param {GPUBindGroupLayout} layout Bind group layout
   * @param {Object} buffers Buffer resources
   * @returns {GPUBindGroup} Configured bind group
   */
  createBindGroup(layout, buffers) {
    const entries = [];
    
    // Standard bindings (order matches layout creation)
    if (buffers.adj_offsets) entries.push({ binding: 0, resource: { buffer: buffers.adj_offsets } });
    if (buffers.adj_data) entries.push({ binding: 1, resource: { buffer: buffers.adj_data } });
    if (buffers.rev_adj_offsets) entries.push({ binding: 2, resource: { buffer: buffers.rev_adj_offsets } });
    if (buffers.rev_adj_data) entries.push({ binding: 3, resource: { buffer: buffers.rev_adj_data } });
    if (buffers.params) entries.push({ binding: 4, resource: { buffer: buffers.params } });
    if (buffers.result) entries.push({ binding: 5, resource: { buffer: buffers.result } });
    
    // Property bindings
    if (buffers.properties) {
      if (Array.isArray(buffers.properties)) {
        // Multiple property buffers
        for (let i = 0; i < buffers.properties.length; i++) {
          entries.push({ binding: 6 + i, resource: { buffer: buffers.properties[i] } });
        }
      } else {
        // Single property buffer
        entries.push({ binding: 6, resource: { buffer: buffers.properties } });
      }
    }
    
    // Weights buffer
    if (buffers.weights) {
      const weightBinding = 6 + (Array.isArray(buffers.properties) ? buffers.properties.length : (buffers.properties ? 1 : 0));
      entries.push({ binding: weightBinding, resource: { buffer: buffers.weights } });
    }

    return this.device.createBindGroup({ layout, entries });
  }

  /**
   * Complete pipeline setup with automatic caching and bind group generation
   * Replaces entire generator pipeline setup pattern (lines 164-180+)
   * 
   * @param {string} wgslCode WGSL shader code
   * @param {Object} buffers Buffer resources
   * @param {Object} options Pipeline options
   * @returns {Object} Complete pipeline setup
   */
  async setupPipeline(wgslCode, buffers, options = {}) {
    // Create/retrieve cached shader module
    const shaderModule = await this.getShaderModule(wgslCode, options.shaderLabel);
    
    // Auto-generate bind group layout from buffers
    const bindGroupLayout = this.createBindGroupLayout(buffers);
    
    // Create/retrieve cached pipeline
    const pipeline = this.getComputePipeline(
      shaderModule, 
      bindGroupLayout, 
      options.entryPoint || 'main',
      options
    );
    
    // Create bind group with auto-matching
    const bindGroup = this.createBindGroup(bindGroupLayout, buffers);
    
    return {
      pipeline,
      bindGroup,
      shaderModule,
      bindGroupLayout
    };
  }

  /**
   * Execute compute pipeline with automatic setup
   * Complete replacement for generator kernel launch pattern
   * 
   * @param {string} wgslCode WGSL shader code  
   * @param {Object} buffers Buffer resources
   * @param {Object} dispatchInfo Dispatch configuration
   * @param {Object} options Pipeline options
   * @returns {Promise<void>} Completion promise
   */
  async executeCompute(wgslCode, buffers, dispatchInfo, options = {}) {
    const { pipeline, bindGroup } = await this.setupPipeline(wgslCode, buffers, options);
    
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Calculate dispatch size
    const workgroupsX = dispatchInfo.workgroupsX || Math.ceil(dispatchInfo.nodeCount / 256);
    const workgroupsY = dispatchInfo.workgroupsY || 1;
    const workgroupsZ = dispatchInfo.workgroupsZ || 1;
    
    pass.dispatchWorkgroups(Math.max(workgroupsX, 1), workgroupsY, workgroupsZ);
    pass.end();
    
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Get pipeline cache statistics
   * @returns {Object} Cache performance statistics
   */
  getCacheStats() {
    const totalShaderRequests = this.stats.shaderModuleCacheHits + this.stats.shaderModuleCacheMisses;
    const totalPipelineRequests = this.stats.pipelineCacheHits + this.stats.pipelineCacheMisses;
    const totalLayoutRequests = this.stats.bindGroupLayoutCacheHits + this.stats.bindGroupLayoutCacheMisses;
    
    return {
      ...this.stats,
      shaderModuleCacheSize: this.shaderModuleCache.size,
      pipelineCacheSize: this.pipelineCache.size,
      bindGroupLayoutCacheSize: this.bindGroupLayoutCache.size,
      shaderModuleCacheHitRate: totalShaderRequests ? (this.stats.shaderModuleCacheHits / totalShaderRequests * 100).toFixed(1) + '%' : 'N/A',
      pipelineCacheHitRate: totalPipelineRequests ? (this.stats.pipelineCacheHits / totalPipelineRequests * 100).toFixed(1) + '%' : 'N/A',
      bindGroupLayoutCacheHitRate: totalLayoutRequests ? (this.stats.bindGroupLayoutCacheHits / totalLayoutRequests * 100).toFixed(1) + '%' : 'N/A'
    };
  }

  /**
   * Clear all caches (useful for development/testing)
   */
  clearCaches() {
    this.shaderModuleCache.clear();
    this.pipelineCache.clear();
    this.bindGroupLayoutCache.clear();
    
    // Reset statistics
    Object.keys(this.stats).forEach(key => this.stats[key] = 0);
  }

  // Private helper methods
  
  _hashString(str) {
    let hash = 0;
    if (str.length === 0) return hash.toString();
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString();
  }
  
  _getBindGroupLayoutKey(bindings) {
    return JSON.stringify(bindings, Object.keys(bindings).sort());
  }
  
  _getPipelineKey(shaderModule, bindGroupLayout, entryPoint, options) {
    // Use object properties that affect pipeline creation
    const keyData = {
      shaderModuleHash: shaderModule.label || 'anonymous',
      layoutHash: bindGroupLayout.label || 'anonymous', 
      entryPoint,
      constants: options.constants || {}
    };
    return JSON.stringify(keyData);
  }
}

/**
 * Usage Examples:
 * 
 * // Basic pipeline setup with caching
 * const pipelineManager = new WebGPUPipelineManager(device);
 * const { pipeline, bindGroup } = await pipelineManager.setupPipeline(wgslCode, buffers);
 * 
 * // Execute compute with automatic pipeline management
 * await pipelineManager.executeCompute(wgslCode, buffers, { nodeCount: 10000 });
 * 
 * // Check cache performance
 * console.log('Cache stats:', pipelineManager.getCacheStats());
 * 
 * // Auto-bind group generation (Task 3.2)
 * const layout = pipelineManager.createBindGroupLayout({
 *   adj_offsets: true,
 *   adj_data: true, 
 *   params: true,
 *   result: true,
 *   properties: [
 *     { binding: 6, readWrite: true },
 *     { binding: 7, readWrite: false }
 *   ]
 * });
 */
