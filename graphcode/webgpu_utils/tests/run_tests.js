#!/usr/bin/env deno run --allow-read --allow-net

/**
 * StarPlat WebGPU Utilities Test Execution Script
 * 
 * Convenience script to run all WebGPU utility tests
 * Usage: deno run --allow-read --allow-net run_tests.js
 * 
 * Version: 1.0 (Phase 3.19)
 */

import { WebGPUTestRunner } from './test_runner.js';

/**
 * Main test execution function
 */
async function main() {
  console.log('🚀 StarPlat WebGPU Utilities Test Suite\n');
  console.log('Testing modular utilities from Phase 3.14-3.19:\n');
  console.log('  ✨ Task 3.14: webgpu_atomics.wgsl');
  console.log('  ✨ Task 3.15: webgpu_graph_methods.wgsl');
  console.log('  ✨ Task 3.16: webgpu_reductions.wgsl');
  console.log('  ✨ Task 3.17: Host utilities (JavaScript)');
  console.log('  ✨ Task 3.19: This testing infrastructure\n');

  // Initialize test runner with options
  const testRunner = new WebGPUTestRunner({
    verbose: true,
    stopOnError: false // Continue testing even if some tests fail
  });

  let results;
  
  try {
    // Run comprehensive test suite
    results = await testRunner.runAllTests();
    
  } catch (error) {
    console.error('\n💥 Test execution failed:', error.message);
    
    // Check if it's a WebGPU support issue
    if (error.message.includes('WebGPU')) {
      console.log('\n💡 WebGPU Support Information:');
      console.log('   This test requires WebGPU support in your browser.');
      console.log('   Supported browsers:');
      console.log('   • Chrome/Edge 113+ (stable)');
      console.log('   • Firefox 113+ (enabled in about:config)');
      console.log('   • Safari 16.4+ (enabled in Develop menu)');
      console.log('\n   Or run with Deno on systems with WebGPU support.');
    }
    
    process.exit(1);
  }

  // Display final results
  console.log('\n' + '='.repeat(60));
  
  if (results.failed === 0) {
    console.log(`🎉 All ${results.passed} tests passed! WebGPU utilities are working correctly.`);
    console.log('\n✅ Phase 3.14-3.19 Implementation: SUCCESS');
    console.log('\n📋 Ready for Phase 3.18 (Generator Integration)');
  } else {
    console.log(`⚠️  ${results.failed} out of ${results.total} tests failed.`);
    console.log('\n❌ Some WebGPU utilities need attention.');
    
    if (results.passed > 0) {
      console.log(`\n✅ ${results.passed} tests are working correctly.`);
    }
  }

  // Performance summary
  if (results.tests && results.tests.length > 0) {
    console.log('\n📊 Test Categories:');
    console.log(`   Host Utilities: ${countTestsByCategory(results.tests, 'Device Manager|Buffer Utils|Pipeline Manager')}`);
    console.log(`   WGSL Utilities: ${countTestsByCategory(results.tests, 'Atomic Operations|Graph Methods|Workgroup Reductions')}`);
    console.log(`   Integration: ${countTestsByCategory(results.tests, 'Integration')}`);
  }

  console.log('\n' + '='.repeat(60));
  
  // Exit with appropriate code
  process.exit(results.failed === 0 ? 0 : 1);
}

/**
 * Count tests by category pattern
 */
function countTestsByCategory(tests, categoryPattern) {
  const regex = new RegExp(categoryPattern);
  return tests.filter(test => regex.test(test.name)).length;
}

/**
 * Handle graceful shutdown
 */
process.on('SIGINT', () => {
  console.log('\n\n⏹️  Test execution interrupted by user');
  process.exit(130);
});

process.on('SIGTERM', () => {
  console.log('\n\n⏹️  Test execution terminated');
  process.exit(143);
});

// Run the tests
if (import.meta.main) {
  main().catch(error => {
    console.error('💥 Unexpected error:', error);
    process.exit(1);
  });
}
