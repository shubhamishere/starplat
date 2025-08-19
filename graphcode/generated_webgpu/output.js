export async function Compute_TC(device, adj_dataBuffer, adj_offsetsBuffer, nodeCount) {
  console.log('[WebGPU] Compute start: Compute_TC with nodeCount=', nodeCount);
  let result = 0;
  const resultBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const propertyBuffer = device.createBuffer({ size: Math.max(1, nodeCount) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));
  const paramsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([nodeCount, 0, 0, 0]));
  let triangle_count = 0;
  const kernel_res_0 = await launchkernel_0(device, adj_dataBuffer, adj_offsetsBuffer, paramsBuffer, resultBuffer, propertyBuffer, nodeCount);
  result = kernel_res_0;
  // [WebGPU] ignoring DSL return in host orchestration
  console.log('[WebGPU] Compute end: returning result', result);
  return result;
}

async function launchkernel_0(device, adj_dataBuffer, adj_offsetsBuffer, paramsBuffer, resultBuffer, propertyBuffer, nodeCount) {
  console.log('[WebGPU] launchkernel_0: begin');
  const shaderCode = await (await fetch('kernel_0.wgsl')).text();
  console.log('[WebGPU] launchkernel_0: WGSL fetched, size', shaderCode.length);
  const shaderModule = device.createShaderModule({ code: shaderCode });
  if (shaderModule.getCompilationInfo) {
    const info = await shaderModule.getCompilationInfo();
    for (const m of info.messages || []) {
      const s = m.lineNum !== undefined ? `${m.lineNum}:${m.linePos}` : '';
      console[(m.type === 'error') ? 'error' : 'warn']('[WGSL]', m.type, s, m.message);
    }
  }
  const bindGroupLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  const pipeline = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }), compute: { module: shaderModule, entryPoint: 'main' } });
  console.log('[WebGPU] launchkernel_0: pipeline created');
  
  // Using shared result/property buffers provided by caller
  const readBuffer = device.createBuffer({ 
    size: 4, 
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ 
  });
  
  const bindGroup = device.createBindGroup({ 
    layout: bindGroupLayout, 
    entries: [
      { binding: 0, resource: { buffer: adj_offsetsBuffer } },
      { binding: 1, resource: { buffer: adj_dataBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
      { binding: 3, resource: { buffer: resultBuffer } },
      { binding: 4, resource: { buffer: propertyBuffer } }
    ]
  });
  console.log('[WebGPU] launchkernel_0: bindGroup created');
  
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  
  // Dispatch one workgroup per 256 nodes (ensure at least 1 group)
  let __groups = Math.ceil(nodeCount / 256);
  if (__groups < 1) { __groups = 1; }
  pass.dispatchWorkgroups(__groups, 1, 1);
  pass.end();
  console.log('[WebGPU] launchkernel_0: dispatched groups', __groups);
  
  encoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, 4);
  device.queue.submit([encoder.finish()]);
  console.log('[WebGPU] launchkernel_0: submitted');
  
  try {
    await Promise.race([readBuffer.mapAsync(GPUMapMode.READ), new Promise((_,rej)=>setTimeout(()=>rej(new Error('mapAsync timeout')), 10000))]);
  } catch (e) { console.error('[WebGPU] mapAsync error:', e?.message || e); throw e; }
  const result = new Uint32Array(readBuffer.getMappedRange())[0];
  readBuffer.unmap();
  console.log('[WebGPU] launchkernel_0: result', result);
  
  return result;
}

