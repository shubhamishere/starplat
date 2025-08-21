/**
 * StarPlat WebGPU Device Management Utilities
 * 
 * Centralized device initialization, capability checking, and error handling
 * for WebGPU operations. Extracts common patterns from the generator to
 * provide consistent device management across all algorithms.
 * 
 * Version: 1.0 (Phase 3.17)
 */

export class WebGPUDeviceManager {
  constructor(options = {}) {
    this.options = {
      // Device selection preferences
      powerPreference: 'high-performance',
      forceFallbackAdapter: false,
      
      // Feature requirements
      requiredFeatures: [],
      requiredLimits: {},
      
      // Error handling
      throwOnError: true,
      verbose: false,
      
      ...options
    };
    
    this.adapter = null;
    this.device = null;
    this.capabilities = null;
  }

  /**
   * Initialize WebGPU device with error handling and capability detection
   * @returns {Promise<GPUDevice>} Initialized WebGPU device
   */
  async initialize() {
    try {
      if (this.device) {
        if (this.options.verbose) {
          console.log('[WebGPU Device] Using existing device');
        }
        return this.device;
      }

      // Check WebGPU availability
      if (!navigator.gpu) {
        throw new Error('WebGPU not supported - please use a compatible browser');
      }

      // Request adapter with preferences
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: this.options.powerPreference,
        forceFallbackAdapter: this.options.forceFallbackAdapter
      });

      if (!this.adapter) {
        throw new Error('No WebGPU adapter found - WebGPU may not be supported on this system');
      }

      if (this.options.verbose) {
        console.log('[WebGPU Device] Adapter found:', {
          vendor: this.adapter.info?.vendor || 'unknown',
          architecture: this.adapter.info?.architecture || 'unknown',
          device: this.adapter.info?.device || 'unknown',
          description: this.adapter.info?.description || 'unknown'
        });
      }

      // Check required features
      const unsupportedFeatures = this.options.requiredFeatures.filter(
        feature => !this.adapter.features.has(feature)
      );
      
      if (unsupportedFeatures.length > 0) {
        throw new Error(`Required WebGPU features not supported: ${unsupportedFeatures.join(', ')}`);
      }

      // Request device with requirements
      this.device = await this.adapter.requestDevice({
        requiredFeatures: this.options.requiredFeatures,
        requiredLimits: this.options.requiredLimits
      });

      if (!this.device) {
        throw new Error('Failed to create WebGPU device');
      }

      // Set up error handling
      this.device.addEventListener('uncapturederror', (event) => {
        console.error('[WebGPU Device] Uncaptured error:', event.error);
        if (this.options.throwOnError) {
          throw new Error(`WebGPU device error: ${event.error.message}`);
        }
      });

      // Gather device capabilities
      await this._gatherCapabilities();

      if (this.options.verbose) {
        console.log('[WebGPU Device] Device initialized successfully');
        this.printCapabilities();
      }

      return this.device;

    } catch (error) {
      console.error('[WebGPU Device] Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Get current device, initializing if necessary
   * @returns {Promise<GPUDevice>} WebGPU device
   */
  async getDevice() {
    if (!this.device) {
      await this.initialize();
    }
    return this.device;
  }

  /**
   * Check if WebGPU is supported
   * @returns {boolean} True if WebGPU is available
   */
  static isSupported() {
    return typeof navigator !== 'undefined' && 
           typeof navigator.gpu !== 'undefined';
  }

  /**
   * Get device capabilities and limits
   * @returns {Object} Device capabilities
   */
  getCapabilities() {
    return this.capabilities;
  }

  /**
   * Check if specific feature is supported
   * @param {string} feature Feature name to check
   * @returns {boolean} True if feature is supported
   */
  supportsFeature(feature) {
    return this.adapter && this.adapter.features.has(feature);
  }

  /**
   * Get adapter info (if available)
   * @returns {Object|null} Adapter information
   */
  getAdapterInfo() {
    if (!this.adapter || !this.adapter.info) {
      return null;
    }
    
    return {
      vendor: this.adapter.info.vendor || 'unknown',
      architecture: this.adapter.info.architecture || 'unknown', 
      device: this.adapter.info.device || 'unknown',
      description: this.adapter.info.description || 'unknown'
    };
  }

  /**
   * Print device capabilities to console
   */
  printCapabilities() {
    if (!this.capabilities) return;

    console.log('[WebGPU Device] Capabilities:');
    console.log(`  Max compute workgroups per dimension: ${this.capabilities.maxComputeWorkgroupsPerDimension}`);
    console.log(`  Max compute workgroup size X: ${this.capabilities.maxComputeWorkgroupSizeX}`);
    console.log(`  Max compute workgroup size Y: ${this.capabilities.maxComputeWorkgroupSizeY}`);
    console.log(`  Max compute workgroup size Z: ${this.capabilities.maxComputeWorkgroupSizeZ}`);
    console.log(`  Max compute invocations per workgroup: ${this.capabilities.maxComputeInvocationsPerWorkgroup}`);
    console.log(`  Max storage buffer binding size: ${(this.capabilities.maxStorageBufferBindingSize / 1024 / 1024).toFixed(1)}MB`);
    console.log(`  Max buffer size: ${(this.capabilities.maxBufferSize / 1024 / 1024).toFixed(1)}MB`);
    
    if (this.adapter && this.adapter.features.size > 0) {
      console.log(`  Supported features: ${Array.from(this.adapter.features).join(', ')}`);
    }
  }

  /**
   * Validate graph can be processed with device limits
   * @param {number} nodeCount Number of nodes
   * @param {number} edgeCount Number of edges
   * @throws {Error} If graph exceeds device limits
   */
  validateGraphSize(nodeCount, edgeCount) {
    if (!this.capabilities) {
      console.warn('[WebGPU Device] Capabilities not available - skipping validation');
      return;
    }

    const maxWorkgroups = this.capabilities.maxComputeWorkgroupsPerDimension;
    const workgroupSize = 256; // Standard workgroup size
    const maxNodes = maxWorkgroups * workgroupSize;

    if (nodeCount > maxNodes) {
      throw new Error(
        `Graph too large: ${nodeCount} nodes exceeds maximum ${maxNodes} ` +
        `(${maxWorkgroups} workgroups × ${workgroupSize} threads)`
      );
    }

    // Estimate memory requirements (rough approximation)
    const nodeBytes = nodeCount * 4; // 4 bytes per node for properties
    const edgeBytes = edgeCount * 4; // 4 bytes per edge index
    const totalBytes = nodeBytes + edgeBytes * 2; // Forward + reverse CSR
    const maxBufferSize = this.capabilities.maxStorageBufferBindingSize;

    if (totalBytes > maxBufferSize) {
      throw new Error(
        `Graph memory requirements too large: ${(totalBytes / 1024 / 1024).toFixed(1)}MB ` +
        `exceeds maximum ${(maxBufferSize / 1024 / 1024).toFixed(1)}MB per buffer`
      );
    }
  }

  /**
   * Destroy device and clean up resources
   */
  destroy() {
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
    this.adapter = null;
    this.capabilities = null;
  }

  /**
   * Private method to gather device capabilities
   */
  async _gatherCapabilities() {
    if (!this.device) return;

    const limits = this.device.limits;
    this.capabilities = {
      maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
      maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY, 
      maxComputeWorkgroupSizeZ: limits.maxComputeWorkgroupSizeZ,
      maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxBufferSize: limits.maxBufferSize,
      maxBindGroups: limits.maxBindGroups,
      maxBindingsPerBindGroup: limits.maxBindingsPerBindGroup,
      
      // Computed values for StarPlat algorithms
      maxNodesPerDispatch: limits.maxComputeWorkgroupsPerDimension * 256,
      recommendedWorkgroupSize: Math.min(256, limits.maxComputeInvocationsPerWorkgroup)
    };
  }
}

/**
 * Singleton instance for global device management
 */
let globalDeviceManager = null;

/**
 * Get or create global device manager instance
 * @param {Object} options Device manager options
 * @returns {WebGPUDeviceManager} Global device manager
 */
export function getGlobalDeviceManager(options = {}) {
  if (!globalDeviceManager) {
    globalDeviceManager = new WebGPUDeviceManager(options);
  }
  return globalDeviceManager;
}

/**
 * Initialize global WebGPU device (convenience function)
 * @param {Object} options Device options
 * @returns {Promise<GPUDevice>} Initialized device
 */
export async function initializeGlobalDevice(options = {}) {
  const manager = getGlobalDeviceManager(options);
  return await manager.initialize();
}

/**
 * Check WebGPU support and provide helpful error messages
 * @throws {Error} With specific guidance if WebGPU not supported
 */
export function checkWebGPUSupport() {
  if (!WebGPUDeviceManager.isSupported()) {
    throw new Error(
      'WebGPU is not supported in this browser. ' +
      'Please use Chrome/Edge 113+, Firefox 113+, or Safari 16.4+ ' +
      'with WebGPU enabled in experimental features.'
    );
  }
}

/**
 * Usage Examples:
 * 
 * // Basic usage
 * const deviceManager = new WebGPUDeviceManager({ verbose: true });
 * const device = await deviceManager.initialize();
 * 
 * // Global singleton pattern
 * const device = await initializeGlobalDevice({ verbose: true });
 * 
 * // Check capabilities
 * const manager = getGlobalDeviceManager();
 * manager.validateGraphSize(10000, 50000);
 * manager.printCapabilities();
 * 
 * // Error handling
 * try {
 *   checkWebGPUSupport();
 *   const device = await initializeGlobalDevice();
 * } catch (error) {
 *   console.error('WebGPU initialization failed:', error.message);
 * }
 */
