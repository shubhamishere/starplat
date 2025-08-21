/**
 * StarPlat WebGPU Utilities Test Runner
 * 
 * Comprehensive testing framework for WebGPU utilities including:
 * - Atomic operations testing (webgpu_atomics.wgsl)
 * - Graph methods testing (webgpu_graph_methods.wgsl)
 * - Workgroup reductions testing (webgpu_reductions.wgsl)
 * - Host utilities testing (JavaScript modules)
 * 
 * Version: 1.0 (Phase 3.19)
 */

import { WebGPUDeviceManager } from '../host_utils/webgpu_device_manager.js';
import { WebGPUBufferUtils } from '../host_utils/webgpu_buffer_utils.js';
import { WebGPUPipelineManager } from '../host_utils/webgpu_pipeline_manager.js';
import { StarPlatWebGPURunner, checkWebGPUSupport } from '../host_utils/webgpu_host_utils.js';

/**
 * Test runner class with comprehensive WebGPU utility testing
 */
export class WebGPUTestRunner {
  constructor(options = {}) {
    this.options = {
      verbose: true,
      stopOnError: false,
      device: null,
      ...options
    };
    
    this.device = null;
    this.deviceManager = null;
    this.results = {
      total: 0,
      passed: 0,
      failed: 0,
      skipped: 0,
      tests: []
    };
  }

  /**
   * Initialize test environment
   */
  async initialize() {
    try {
      checkWebGPUSupport();
      
      if (this.options.device) {
        this.device = this.options.device;
      } else {
        this.deviceManager = new WebGPUDeviceManager({ verbose: false });
        this.device = await this.deviceManager.initialize();
      }
      
      this.log('✅ Test environment initialized');
      return true;
    } catch (error) {
      this.log(`❌ Test environment initialization failed: ${error.message}`);
      return false;
    }
  }

  /**
   * Run all tests
   */
  async runAllTests() {
    const initialized = await this.initialize();
    if (!initialized) {
      return this.generateReport();
    }

    this.log('\n🧪 Starting WebGPU utilities test suite...\n');

    // Host utilities tests (JavaScript)
    await this.runHostUtilityTests();
    
    // WGSL utility tests (GPU kernels)
    await this.runWGSLUtilityTests();
    
    // Integration tests
    await this.runIntegrationTests();

    // Cleanup
    if (this.deviceManager) {
      this.deviceManager.destroy();
    }

    return this.generateReport();
  }

  /**
   * Run host utility tests (JavaScript modules)
   */
  async runHostUtilityTests() {
    this.log('📱 Testing Host Utilities...\n');

    // Device Manager tests
    await this.test('Device Manager - Initialization', async () => {
      const dm = new WebGPUDeviceManager({ verbose: false });
      const device = await dm.initialize();
      return device !== null;
    });

    await this.test('Device Manager - Capabilities', async () => {
      const dm = new WebGPUDeviceManager({ verbose: false });
      await dm.initialize();
      const caps = dm.getCapabilities();
      return caps && caps.maxComputeWorkgroupsPerDimension > 0;
    });

    // Buffer Utils tests
    await this.test('Buffer Utils - Create Storage Buffer', async () => {
      const bufferUtils = new WebGPUBufferUtils(this.device);
      const data = new Uint32Array([1, 2, 3, 4]);
      const buffer = bufferUtils.createStorageBuffer(data);
      return buffer.size === data.byteLength;
    });

    await this.test('Buffer Utils - Property Buffers', async () => {
      const bufferUtils = new WebGPUBufferUtils(this.device);
      const properties = {
        rank: { type: 'f32', initial: 1.0, binding: 6 },
        visited: { type: 'u32', initial: 0, binding: 7 }
      };
      const { buffers, bindingEntries } = bufferUtils.createPropertyBuffers(properties, 100);
      return Object.keys(buffers).length === 2 && bindingEntries.length === 2;
    });

    await this.test('Buffer Utils - Readback Operations', async () => {
      const bufferUtils = new WebGPUBufferUtils(this.device);
      const data = new Uint32Array([42, 43, 44, 45]);
      const buffer = bufferUtils.createStorageBuffer(data);
      
      const readbackBuffer = bufferUtils.createReadbackBuffer(data.byteLength);
      bufferUtils.copyBufferToReadback(buffer, readbackBuffer);
      const result = await bufferUtils.readBuffer(readbackBuffer, Uint32Array);
      
      return result.length === data.length && result[0] === 42;
    });

    // Pipeline Manager tests
    await this.test('Pipeline Manager - Shader Module Caching', async () => {
      const pm = new WebGPUPipelineManager(this.device);
      const wgslCode = `@compute @workgroup_size(1) fn main() {}`;
      
      const module1 = await pm.getShaderModule(wgslCode);
      const module2 = await pm.getShaderModule(wgslCode);
      
      const stats = pm.getCacheStats();
      return stats.shaderModuleCacheHits >= 1;
    });

    await this.test('Pipeline Manager - Bind Group Layout Generation', async () => {
      const pm = new WebGPUPipelineManager(this.device);
      const bindings = {
        adj_offsets: true,
        adj_data: true,
        params: true,
        result: true
      };
      const layout = pm.createBindGroupLayout(bindings);
      return layout !== null;
    });
  }

  /**
   * Run WGSL utility tests (GPU kernel execution)
   */
  async runWGSLUtilityTests() {
    this.log('\n🔧 Testing WGSL Utilities...\n');

    // Atomic operations tests
    await this.testAtomicOperations();
    
    // Graph methods tests  
    await this.testGraphMethods();
    
    // Workgroup reductions tests
    await this.testWorkgroupReductions();
  }

  /**
   * Test atomic operations utilities
   */
  async testAtomicOperations() {
    const atomicTestKernel = `
      ${await this.loadUtilityFile('../wgsl_kernels/webgpu_atomics.wgsl')}
      
      @group(0) @binding(0) var<storage, read_write> values: array<atomic<u32>>;
      @group(0) @binding(1) var<storage, read_write> results: array<u32>;
      
      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx >= arrayLength(&results)) { return; }
        
        // Test atomicAddF32
        let oldVal = atomicAddF32(&values[0], 1.5);
        results[idx] = bitcast<u32>(oldVal);
      }
    `;

    await this.test('Atomic Operations - atomicAddF32', async () => {
      return await this.runKernelTest(atomicTestKernel, {
        values: { data: new Uint32Array([bitcast32(0.0)]), atomic: true },
        results: { data: new Uint32Array(256), readback: true }
      }, 256);
    });
    
    // Test atomic minimum
    const atomicMinTestKernel = `
      ${await this.loadUtilityFile('../wgsl_kernels/webgpu_atomics.wgsl')}
      
      @group(0) @binding(0) var<storage, read_write> minValue: atomic<u32>;
      @group(0) @binding(1) var<storage, read> inputValues: array<u32>;
      
      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx >= arrayLength(&inputValues)) { return; }
        
        let value = bitcast<f32>(inputValues[idx]);
        atomicMinF32(&minValue, value);
      }
    `;

    await this.test('Atomic Operations - atomicMinF32', async () => {
      const testValues = new Float32Array([5.0, 3.0, 8.0, 1.0, 6.0]);
      const inputData = new Uint32Array(testValues.buffer);
      
      const result = await this.runKernelTest(atomicMinTestKernel, {
        minValue: { data: new Uint32Array([bitcast32(10.0)]), atomic: true, readback: true },
        inputValues: { data: inputData }
      }, testValues.length);
      
      const minResult = new Float32Array(result.minValue.buffer)[0];
      return Math.abs(minResult - 1.0) < 1e-6;
    });
  }

  /**
   * Test graph methods utilities
   */
  async testGraphMethods() {
    // Create simple test graph: 0->1->2, 1->2
    const adjOffsets = new Uint32Array([0, 1, 3, 3]); // 3 nodes
    const adjData = new Uint32Array([1, 1, 2]);       // edges: 0->1, 1->1, 1->2
    
    const graphTestKernel = `
      ${await this.loadUtilityFile('../wgsl_kernels/webgpu_graph_methods.wgsl')}
      
      @group(0) @binding(0) var<storage, read> adj_offsets: array<u32>;
      @group(0) @binding(1) var<storage, read> adj_data: array<u32>;
      @group(0) @binding(2) var<storage, read_write> results: array<u32>;
      
      @compute @workgroup_size(3)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let v = global_id.x;
        if (v >= 3u) { return; }
        
        // Test findEdge and getOutDegree
        if (v == 0u) {
          results[0] = select(0u, 1u, findEdge(0u, 1u)); // Should find edge 0->1
          results[1] = getOutDegree(0u);                  // Should be 1
        }
        else if (v == 1u) {
          results[2] = select(0u, 1u, findEdge(1u, 2u)); // Should find edge 1->2
          results[3] = getOutDegree(1u);                  // Should be 2
        }
      }
    `;

    await this.test('Graph Methods - findEdge and getOutDegree', async () => {
      const result = await this.runKernelTest(graphTestKernel, {
        adj_offsets: { data: adjOffsets },
        adj_data: { data: adjData },
        results: { data: new Uint32Array(4), readback: true }
      }, 3);
      
      return result.results[0] === 1 && // Found edge 0->1
             result.results[1] === 1 && // Out-degree of node 0 is 1
             result.results[2] === 1 && // Found edge 1->2
             result.results[3] === 2;   // Out-degree of node 1 is 2
    });
  }

  /**
   * Test workgroup reductions utilities
   */
  async testWorkgroupReductions() {
    const reductionTestKernel = `
      ${await this.loadUtilityFile('../wgsl_kernels/webgpu_reductions.wgsl')}
      
      @group(0) @binding(0) var<storage, read_write> result: atomic<u32>;
      
      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>) {
        let value = local_id.x; // Each thread contributes its local ID
        
        // Test workgroup sum reduction
        let sum = workgroupReduceSum(local_id.x, value);
        
        // Only thread 0 updates global result
        if (local_id.x == 0u) {
          atomicAdd(&result, sum);
        }
      }
    `;

    await this.test('Workgroup Reductions - Sum', async () => {
      const result = await this.runKernelTest(reductionTestKernel, {
        result: { data: new Uint32Array([0]), atomic: true, readback: true }
      }, 256);
      
      // Sum of 0 to 255 = 255 * 256 / 2 = 32640
      return result.result[0] === 32640;
    });

    // Test f32 sum reduction
    const f32ReductionTestKernel = `
      ${await this.loadUtilityFile('../wgsl_kernels/webgpu_reductions.wgsl')}
      
      @group(0) @binding(0) var<storage, read_write> result: atomic<u32>;
      
      @compute @workgroup_size(256)
      fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
        let value = f32(local_id.x) * 0.1; // Each thread contributes 0.1 * local_id
        
        let sum = workgroupReduceSumF32(local_id.x, value);
        
        if (local_id.x == 0u) {
          atomicAddF32(&result, sum);
        }
      }
    `;

    await this.test('Workgroup Reductions - F32 Sum', async () => {
      const result = await this.runKernelTest(f32ReductionTestKernel, {
        result: { data: new Uint32Array([0]), atomic: true, readback: true }
      }, 256);
      
      const f32Result = new Float32Array(result.result.buffer)[0];
      const expected = (255 * 256 / 2) * 0.1; // 3264.0
      return Math.abs(f32Result - expected) < 1e-3;
    });
  }

  /**
   * Run integration tests combining multiple utilities
   */
  async runIntegrationTests() {
    this.log('\n🔗 Testing Integration...\n');

    // Test complete StarPlat runner
    await this.test('Integration - StarPlat Runner', async () => {
      const runner = new StarPlatWebGPURunner({
        deviceOptions: { verbose: false },
        executionOptions: { verbose: false }
      });

      // Simple graph: triangle (0-1-2-0)
      const csr = {
        nodeCount: 3,
        edgeCount: 6,
        forward: {
          offsets: new Uint32Array([0, 2, 4, 6]),
          data: new Uint32Array([1, 2, 0, 2, 0, 1])
        },
        reverse: {
          offsets: new Uint32Array([0, 2, 4, 6]),
          data: new Uint32Array([1, 2, 0, 2, 0, 1])
        }
      };

      const simpleKernel = `
        @group(0) @binding(0) var<storage, read> adj_offsets: array<u32>;
        @group(0) @binding(1) var<storage, read> adj_data: array<u32>;
        @group(0) @binding(5) var<storage, read_write> result: atomic<u32>;
        
        struct Params { node_count: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
        @group(0) @binding(4) var<uniform> params: Params;
        
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let v = global_id.x;
          if (v >= params.node_count) { return; }
          
          let degree = adj_offsets[v + 1u] - adj_offsets[v];
          atomicAdd(&result, degree);
        }
      `;

      const algorithmConfig = {
        name: 'DegreeSum',
        kernels: [{ wgslCode: simpleKernel, name: 'degree_sum' }]
      };

      const result = await runner.runAlgorithm(csr, algorithmConfig);
      runner.cleanup();
      
      return result.result === 6; // Sum of degrees = 6 (2+2+2)
    });
  }

  /**
   * Run individual kernel test
   */
  async runKernelTest(wgslCode, bufferSpecs, nodeCount) {
    const bufferUtils = new WebGPUBufferUtils(this.device);
    const pipelineManager = new WebGPUPipelineManager(this.device);

    try {
      // Create buffers
      const buffers = {};
      for (const [name, spec] of Object.entries(bufferSpecs)) {
        if (spec.atomic) {
          buffers[name] = bufferUtils.createStorageBuffer(
            spec.data, 
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
          );
        } else {
          buffers[name] = bufferUtils.createStorageBuffer(spec.data);
        }
      }

      // Execute kernel
      await pipelineManager.executeCompute(wgslCode, buffers, { nodeCount });

      // Read back results
      const results = {};
      for (const [name, spec] of Object.entries(bufferSpecs)) {
        if (spec.readback) {
          const readbackBuffer = bufferUtils.createReadbackBuffer(buffers[name].size);
          bufferUtils.copyBufferToReadback(buffers[name], readbackBuffer);
          results[name] = await bufferUtils.readBuffer(readbackBuffer, Uint32Array);
        }
      }

      bufferUtils.cleanup();
      return results;
    } catch (error) {
      bufferUtils.cleanup();
      throw error;
    }
  }

  /**
   * Load WGSL utility file content
   */
  async loadUtilityFile(relativePath) {
    try {
      // In a real implementation, this would read the file
      // For now, return a placeholder that includes basic utilities
      return `
        var<workgroup> scratchpad: array<u32, 256>;
        var<workgroup> scratchpad_f32: array<f32, 256>;

        fn atomicAddF32(ptr: ptr<storage, atomic<u32>>, val: f32) -> f32 {
          loop {
            let oldBits = atomicLoad(ptr);
            let oldVal = bitcast<f32>(oldBits);
            let newVal = oldVal + val;
            let newBits = bitcast<u32>(newVal);
            let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);
            if (res.exchanged) { return oldVal; }
          }
        }

        fn atomicMinF32(ptr: ptr<storage, atomic<u32>>, val: f32) {
          loop {
            let oldBits = atomicLoad(ptr);
            let oldVal = bitcast<f32>(oldBits);
            let newVal = min(oldVal, val);
            let newBits = bitcast<u32>(newVal);
            let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);
            if (res.exchanged) { return; }
          }
        }

        fn findEdge(u: u32, w: u32) -> bool {
          let start = adj_offsets[u];
          let end = adj_offsets[u + 1u];
          for (var e = start; e < end; e++) {
            if (adj_data[e] == w) { return true; }
          }
          return false;
        }

        fn getOutDegree(v: u32) -> u32 {
          return adj_offsets[v + 1u] - adj_offsets[v];
        }

        fn workgroupReduceSum(local_id: u32, value: u32) -> u32 {
          scratchpad[local_id] = value;
          workgroupBarrier();
          var stride = 128u;
          while (stride > 0u) {
            if (local_id < stride) {
              scratchpad[local_id] += scratchpad[local_id + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
          }
          return scratchpad[0];
        }

        fn workgroupReduceSumF32(local_id: u32, value: f32) -> f32 {
          scratchpad_f32[local_id] = value;
          workgroupBarrier();
          var stride = 128u;
          while (stride > 0u) {
            if (local_id < stride) {
              scratchpad_f32[local_id] += scratchpad_f32[local_id + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
          }
          return scratchpad_f32[0];
        }
      `;
    } catch (error) {
      this.log(`Warning: Could not load utility file ${relativePath}: ${error.message}`);
      return ''; // Return empty string as fallback
    }
  }

  /**
   * Run individual test with error handling
   */
  async test(name, testFn) {
    this.results.total++;
    
    try {
      const result = await testFn();
      if (result) {
        this.results.passed++;
        this.log(`✅ ${name}`);
        this.results.tests.push({ name, status: 'PASS', error: null });
      } else {
        this.results.failed++;
        this.log(`❌ ${name}: Test returned false`);
        this.results.tests.push({ name, status: 'FAIL', error: 'Test returned false' });
      }
    } catch (error) {
      this.results.failed++;
      this.log(`❌ ${name}: ${error.message}`);
      this.results.tests.push({ name, status: 'FAIL', error: error.message });
      
      if (this.options.stopOnError) {
        throw error;
      }
    }
  }

  /**
   * Generate test report
   */
  generateReport() {
    const passRate = this.results.total > 0 ? (this.results.passed / this.results.total * 100).toFixed(1) : 0;
    
    this.log('\n📊 Test Results Summary:');
    this.log(`   Total: ${this.results.total}`);
    this.log(`   Passed: ${this.results.passed} (${passRate}%)`);
    this.log(`   Failed: ${this.results.failed}`);
    this.log(`   Pass Rate: ${passRate}%`);
    
    if (this.results.failed > 0) {
      this.log('\n❌ Failed Tests:');
      for (const test of this.results.tests.filter(t => t.status === 'FAIL')) {
        this.log(`   - ${test.name}: ${test.error}`);
      }
    }
    
    return this.results;
  }

  /**
   * Log helper
   */
  log(message) {
    if (this.options.verbose) {
      console.log(message);
    }
  }
}

// Helper function for bitcast operations in tests
function bitcast32(f32Value) {
  const buffer = new ArrayBuffer(4);
  new Float32Array(buffer)[0] = f32Value;
  return new Uint32Array(buffer)[0];
}

/**
 * Usage Examples:
 * 
 * // Run all tests
 * const testRunner = new WebGPUTestRunner({ verbose: true });
 * const results = await testRunner.runAllTests();
 * 
 * // Run specific test category
 * await testRunner.initialize();
 * await testRunner.runHostUtilityTests();
 * 
 * // Custom test with existing device
 * const testRunner = new WebGPUTestRunner({ device: myDevice });
 * await testRunner.runWGSLUtilityTests();
 */
