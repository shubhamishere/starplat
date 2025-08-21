# WebGPU Utilities Testing Infrastructure

This directory contains comprehensive tests for all WebGPU utilities, following test-driven development principles.

## Files

- **`test_runner.js`** - Test execution framework and runner
- **`atomic_tests.wgsl`** - Tests for atomic operations
- **`graph_tests.wgsl`** - Tests for graph traversal methods
- **`reduction_tests.wgsl`** - Tests for workgroup reduction operations  
- **`integration_tests.js`** - End-to-end integration tests
- **`performance_tests.js`** - Performance benchmarking tests

## Implementation Status

- [ ] **Phase 3.19**: Create comprehensive testing framework
- [ ] **Phase 3.14**: Add atomic operation tests
- [ ] **Phase 3.15**: Add graph method tests
- [ ] **Phase 3.16**: Add reduction operation tests
- [ ] **Phase 3.17**: Add host utility tests

## Testing Strategy

### 1. Unit Tests (WGSL)
Individual utility functions tested in isolation:
```bash
# Run atomic operation tests
deno run --allow-read --unstable-webgpu test_runner.js atomic_tests.wgsl

# Run graph method tests  
deno run --allow-read --unstable-webgpu test_runner.js graph_tests.wgsl
```

### 2. Integration Tests (JavaScript)
Complete workflows with real WebGPU devices:
```bash
# Run integration tests
deno run --allow-read --unstable-webgpu integration_tests.js
```

### 3. Performance Tests
Benchmarking against manual implementations:
```bash
# Run performance benchmarks
deno run --allow-read --unstable-webgpu performance_tests.js
```

## Test Categories

### Atomic Operations
- **Correctness**: atomicAddF32, atomicMinF32, atomicMaxF32
- **Concurrency**: Multiple threads accessing same location
- **Edge Cases**: NaN handling, overflow conditions

### Graph Methods  
- **findEdge**: Binary search vs linear search correctness
- **getEdgeIndex**: Edge existence and index accuracy
- **Degree functions**: Out-degree and in-degree calculations

### Workgroup Reductions
- **Sum reductions**: Integer and float variants
- **Min/Max reductions**: Correctness across workgroup sizes
- **Edge cases**: Single element, empty workgroups

### Host Utilities
- **Buffer management**: Creation, copying, cleanup
- **Device initialization**: Error handling, capability checks
- **Pipeline caching**: Cache hits, invalidation

## Golden Reference Tests

Tests compare WebGPU utility outputs against known-correct implementations:
- **Small graphs**: Hand-calculated expected results
- **Standard datasets**: Comparison with reference algorithms  
- **Edge cases**: Empty graphs, single nodes, disconnected components

## Continuous Integration

Tests are designed to run in CI environments:
- **Headless WebGPU**: Tests run without display requirements
- **Multiple platforms**: Cross-platform compatibility
- **Performance regression**: Detect performance degradation

## Usage

```bash
# Run all tests
cd graphcode/webgpu_utils/tests
deno run --allow-read --unstable-webgpu test_runner.js

# Run specific test category
deno run --allow-read --unstable-webgpu test_runner.js --category atomics

# Run with verbose output
deno run --allow-read --unstable-webgpu test_runner.js --verbose

# Generate test report
deno run --allow-read --unstable-webgpu test_runner.js --report results.json
```

## Integration with Phase 4

These tests provide the foundation for Phase 4 correctness testing:
- **Utility validation** ensures building blocks are correct
- **Integration tests** verify complete algorithm workflows
- **Performance tests** establish baseline measurements
- **Regression tests** prevent utility degradation

The testing infrastructure built here directly supports Phase 4's comprehensive algorithm validation goals.
