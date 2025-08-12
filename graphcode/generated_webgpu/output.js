export async function FloatSum(device, adj_dataBuffer, adj_offsetsBuffer, nodeCount) {
  let result = 0;
  let total;
  const kernel_res_0 = await launchkernel_0(device, adj_dataBuffer, adj_offsetsBuffer, nodeCount);
  result = kernel_res_0;
  return total;
  return result;
}

async function launchkernel_0(device, adj_dataBuffer, adj_offsetsBuffer, nodeCount) {
  const shaderCode = await (await fetch('kernel_0.wgsl')).text();
  const shaderModule = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: shaderModule, entryPoint: 'main' } });
  
  // Create result buffer for algorithm output
  const resultBuffer = device.createBuffer({ 
    size: 4, 
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST 
  });
  
  // Create property buffer for node properties
  const propertyBuffer = device.createBuffer({
    size: Math.max(1, nodeCount) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  // Write nodeCount to properties[0] for the kernel to read
  device.queue.writeBuffer(propertyBuffer, 0, new Uint32Array([nodeCount]));
  
  const readBuffer = device.createBuffer({ 
    size: 4, 
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ 
  });
  
  // Initialize result buffer to 0
  device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));
  
  const bindGroup = device.createBindGroup({ 
    layout: pipeline.getBindGroupLayout(0), 
    entries: [
      { binding: 0, resource: { buffer: adj_dataBuffer } },
      { binding: 1, resource: { buffer: adj_offsetsBuffer } },
      { binding: 2, resource: { buffer: resultBuffer } },
      { binding: 3, resource: { buffer: propertyBuffer } }
    ]
  });
  
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  
  // Dispatch one workgroup per 256 nodes (ensure at least 1 group)
  let __groups = Math.ceil(nodeCount / 256);
  if (__groups < 1) { __groups = 1; }
  pass.dispatchWorkgroups(__groups);
  pass.end();
  
  encoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, 4);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  
  await readBuffer.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(readBuffer.getMappedRange())[0];
  readBuffer.unmap();
  
  return result;
}

