/**
 * StarPlat WebGPU Error Handling and Validation Utilities - Host Side
 * 
 * Comprehensive error handling for WebGPU operations including device errors,
 * shader compilation errors, buffer validation, and kernel execution errors.
 * Complements the WGSL error handling utilities.
 * 
 * Version: 1.0 (Phase 3.10)
 */

// =============================================================================
// ERROR CODES AND CONSTANTS
// =============================================================================

export const ErrorCodes = {
  SUCCESS: 0,
  GENERAL_ERROR: 1,
  BOUNDS_ERROR: 2, 
  MEMORY_ERROR: 3,
  CONVERGENCE_FAILED: 4,
  INVALID_INPUT: 5,
  OVERFLOW_ERROR: 6,
  WEBGPU_ERROR: 7,
  SHADER_COMPILATION_ERROR: 8,
  BUFFER_ERROR: 9,
  PIPELINE_ERROR: 10
};

export const ErrorMessages = {
  [ErrorCodes.SUCCESS]: "Operation completed successfully",
  [ErrorCodes.GENERAL_ERROR]: "General kernel execution error",
  [ErrorCodes.BOUNDS_ERROR]: "Array bounds check failed",
  [ErrorCodes.MEMORY_ERROR]: "Memory access error",
  [ErrorCodes.CONVERGENCE_FAILED]: "Algorithm failed to converge",
  [ErrorCodes.INVALID_INPUT]: "Invalid input detected",
  [ErrorCodes.OVERFLOW_ERROR]: "Numerical overflow detected",
  [ErrorCodes.WEBGPU_ERROR]: "WebGPU device or API error",
  [ErrorCodes.SHADER_COMPILATION_ERROR]: "WGSL shader compilation error",
  [ErrorCodes.BUFFER_ERROR]: "Buffer creation or operation error",
  [ErrorCodes.PIPELINE_ERROR]: "Pipeline creation or execution error"
};

// =============================================================================
// ERROR HANDLING CLASS
// =============================================================================

export class WebGPUErrorHandler {
  constructor(options = {}) {
    this.options = {
      throwOnError: true,
      logErrors: true,
      collectMetrics: true,
      ...options
    };
    
    this.errorHistory = [];
    this.errorCounts = new Map();
    this.warningCounts = new Map();
  }

  /**
   * Extract error information from kernel result
   * @param {number} resultValue Raw result from kernel execution
   * @returns {Object} Parsed result with error information
   */
  parseKernelResult(resultValue) {
    const errorCode = (resultValue >> 24) & 0xFF;
    const actualResult = resultValue & 0x00FFFFFF;
    const debugMarkers = (resultValue >> 16) & 0xFF;
    
    return {
      errorCode,
      actualResult,
      debugMarkers,
      hasError: errorCode !== ErrorCodes.SUCCESS,
      errorMessage: this.getErrorMessage(errorCode)
    };
  }

  /**
   * Get human-readable error message
   * @param {number} errorCode Error code from kernel
   * @returns {string} Error message
   */
  getErrorMessage(errorCode) {
    return ErrorMessages[errorCode] || `Unknown error code: ${errorCode}`;
  }

  /**
   * Handle WebGPU device errors
   * @param {GPUDevice} device WebGPU device
   * @param {string} operation Operation being performed
   */
  setupDeviceErrorHandling(device, operation = "WebGPU operation") {
    device.addEventListener('uncapturederror', (event) => {
      this.handleError(new Error(`${operation}: ${event.error.message}`), {
        type: 'device_error',
        operation,
        errorType: event.error.constructor.name
      });
    });
  }

  /**
   * Validate shader compilation
   * @param {GPUShaderModule} shaderModule Compiled shader module
   * @param {string} shaderSource WGSL source code
   * @returns {Promise<boolean>} True if compilation successful
   */
  async validateShaderCompilation(shaderModule, shaderSource) {
    try {
      if (shaderModule.getCompilationInfo) {
        const info = await shaderModule.getCompilationInfo();
        
        let hasErrors = false;
        for (const message of info.messages || []) {
          const location = message.lineNum !== undefined 
            ? `Line ${message.lineNum}:${message.linePos || 0}` 
            : 'Unknown location';
          
          const logMessage = `[WGSL ${message.type}] ${location}: ${message.message}`;
          
          if (message.type === 'error') {
            this.handleError(new Error(`Shader compilation error: ${logMessage}`), {
              type: 'shader_compilation',
              location,
              wgslMessage: message.message,
              shaderSource: shaderSource.substring(0, 1000) // First 1000 chars for context
            });
            hasErrors = true;
          } else if (message.type === 'warning') {
            this.handleWarning(logMessage, {
              type: 'shader_warning',
              location,
              wgslMessage: message.message
            });
          }
        }
        
        return !hasErrors;
      }
      
      return true; // Assume success if getCompilationInfo not available
      
    } catch (error) {
      this.handleError(error, {
        type: 'shader_validation',
        operation: 'getCompilationInfo'
      });
      return false;
    }
  }

  /**
   * Validate buffer operations
   * @param {Object} bufferSpec Buffer specification
   * @param {number} nodeCount Number of nodes (for validation)
   * @returns {boolean} True if buffer spec is valid
   */
  validateBufferSpec(bufferSpec, nodeCount) {
    try {
      if (!bufferSpec) {
        throw new Error("Buffer specification is null or undefined");
      }

      if (bufferSpec.size && bufferSpec.size <= 0) {
        throw new Error(`Invalid buffer size: ${bufferSpec.size}`);
      }

      if (bufferSpec.usage && typeof bufferSpec.usage !== 'number') {
        throw new Error(`Invalid buffer usage flags: ${bufferSpec.usage}`);
      }

      // Validate buffer size against node count
      if (bufferSpec.size && nodeCount) {
        const maxExpectedSize = nodeCount * 16; // Allow up to 16 bytes per node
        if (bufferSpec.size > maxExpectedSize * 1000) { // 1000x safety factor
          this.handleWarning(`Large buffer size detected: ${bufferSpec.size} bytes for ${nodeCount} nodes`, {
            type: 'buffer_size_warning',
            bufferSize: bufferSpec.size,
            nodeCount
          });
        }
      }

      return true;
      
    } catch (error) {
      this.handleError(error, {
        type: 'buffer_validation',
        bufferSpec
      });
      return false;
    }
  }

  /**
   * Validate graph data for common issues
   * @param {Object} csr Graph in CSR format
   * @returns {boolean} True if graph data is valid
   */
  validateGraphData(csr) {
    try {
      if (!csr || !csr.forward) {
        throw new Error("Invalid CSR format: missing forward adjacency");
      }

      const { forward } = csr;
      if (!forward.offsets || !forward.data) {
        throw new Error("Invalid CSR format: missing offsets or data arrays");
      }

      if (forward.offsets.length === 0) {
        throw new Error("CSR offsets array is empty");
      }

      if (forward.offsets[0] !== 0) {
        throw new Error("CSR offsets must start with 0");
      }

      const nodeCount = forward.offsets.length - 1;
      const edgeCount = forward.data.length;

      if (forward.offsets[nodeCount] !== edgeCount) {
        throw new Error(`CSR format error: last offset (${forward.offsets[nodeCount]}) must equal edge count (${edgeCount})`);
      }

      // Check for out-of-bounds node references
      for (let i = 0; i < edgeCount; i++) {
        if (forward.data[i] >= nodeCount) {
          throw new Error(`Invalid node reference: ${forward.data[i]} >= ${nodeCount} at edge ${i}`);
        }
      }

      // Validate reverse CSR if present
      if (csr.reverse) {
        if (!csr.reverse.offsets || !csr.reverse.data) {
          throw new Error("Invalid reverse CSR format: missing offsets or data arrays");
        }
        
        if (csr.reverse.offsets.length !== forward.offsets.length) {
          throw new Error("Reverse CSR offsets length must match forward CSR");
        }
      }

      // Log statistics
      this.log(`Graph validation passed: ${nodeCount} nodes, ${edgeCount} edges`);
      
      return true;
      
    } catch (error) {
      this.handleError(error, {
        type: 'graph_validation',
        nodeCount: csr?.forward?.offsets?.length - 1,
        edgeCount: csr?.forward?.data?.length
      });
      return false;
    }
  }

  /**
   * Validate pipeline configuration
   * @param {Object} pipelineConfig Pipeline configuration
   * @returns {boolean} True if pipeline config is valid
   */
  validatePipelineConfig(pipelineConfig) {
    try {
      if (!pipelineConfig) {
        throw new Error("Pipeline configuration is null or undefined");
      }

      if (!pipelineConfig.compute || !pipelineConfig.compute.module) {
        throw new Error("Pipeline configuration missing compute shader module");
      }

      if (!pipelineConfig.layout) {
        throw new Error("Pipeline configuration missing layout");
      }

      const entryPoint = pipelineConfig.compute.entryPoint || 'main';
      if (typeof entryPoint !== 'string' || entryPoint.length === 0) {
        throw new Error(`Invalid entry point: ${entryPoint}`);
      }

      return true;
      
    } catch (error) {
      this.handleError(error, {
        type: 'pipeline_validation',
        pipelineConfig
      });
      return false;
    }
  }

  /**
   * Handle error with context and options
   * @param {Error} error Error object
   * @param {Object} context Additional context information
   */
  handleError(error, context = {}) {
    const errorInfo = {
      message: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString(),
      context
    };

    // Collect metrics
    if (this.options.collectMetrics) {
      this.errorHistory.push(errorInfo);
      const errorType = context.type || 'unknown';
      this.errorCounts.set(errorType, (this.errorCounts.get(errorType) || 0) + 1);
    }

    // Log error
    if (this.options.logErrors) {
      console.error('[WebGPU Error]', errorInfo);
    }

    // Throw if configured to do so
    if (this.options.throwOnError) {
      throw error;
    }
  }

  /**
   * Handle warning with context
   * @param {string} message Warning message
   * @param {Object} context Additional context information
   */
  handleWarning(message, context = {}) {
    const warningInfo = {
      message,
      timestamp: new Date().toISOString(),
      context
    };

    // Collect metrics
    if (this.options.collectMetrics) {
      const warningType = context.type || 'unknown';
      this.warningCounts.set(warningType, (this.warningCounts.get(warningType) || 0) + 1);
    }

    // Log warning
    if (this.options.logErrors) {
      console.warn('[WebGPU Warning]', warningInfo);
    }
  }

  /**
   * Get error statistics
   * @returns {Object} Error and warning statistics
   */
  getErrorStatistics() {
    return {
      totalErrors: this.errorHistory.length,
      errorCounts: Object.fromEntries(this.errorCounts),
      warningCounts: Object.fromEntries(this.warningCounts),
      recentErrors: this.errorHistory.slice(-10), // Last 10 errors
      errorTypes: Array.from(this.errorCounts.keys())
    };
  }

  /**
   * Clear error history and statistics
   */
  clearErrorHistory() {
    this.errorHistory = [];
    this.errorCounts.clear();
    this.warningCounts.clear();
  }

  /**
   * Log informational message
   * @param {string} message Message to log
   * @param {Object} context Optional context
   */
  log(message, context = {}) {
    if (this.options.logErrors) {
      console.log('[WebGPU Info]', message, context);
    }
  }
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Create error handler with default options
 * @param {Object} options Error handler options
 * @returns {WebGPUErrorHandler} Error handler instance
 */
export function createErrorHandler(options = {}) {
  return new WebGPUErrorHandler(options);
}

/**
 * Quick validation function for common WebGPU operations
 * @param {Object} params Validation parameters
 * @returns {boolean} True if all validations pass
 */
export function validateWebGPUOperation(params) {
  const errorHandler = new WebGPUErrorHandler({ throwOnError: false });
  
  let isValid = true;
  
  if (params.bufferSpec) {
    isValid = errorHandler.validateBufferSpec(params.bufferSpec, params.nodeCount) && isValid;
  }
  
  if (params.graph) {
    isValid = errorHandler.validateGraphData(params.graph) && isValid;
  }
  
  if (params.pipelineConfig) {
    isValid = errorHandler.validatePipelineConfig(params.pipelineConfig) && isValid;
  }
  
  return isValid;
}

/**
 * Usage Examples:
 * 
 * // Basic error handling setup
 * const errorHandler = new WebGPUErrorHandler({
 *   throwOnError: true,
 *   logErrors: true
 * });
 * 
 * // Setup device error handling
 * errorHandler.setupDeviceErrorHandling(device, "Algorithm execution");
 * 
 * // Validate shader compilation
 * const isValid = await errorHandler.validateShaderCompilation(shaderModule, wgslCode);
 * 
 * // Parse kernel results
 * const result = errorHandler.parseKernelResult(resultValue);
 * if (result.hasError) {
 *   console.error("Kernel error:", result.errorMessage);
 * }
 * 
 * // Validate graph data
 * if (!errorHandler.validateGraphData(csr)) {
 *   console.error("Invalid graph data");
 * }
 * 
 * // Get error statistics
 * const stats = errorHandler.getErrorStatistics();
 * console.log("Error statistics:", stats);
 */
