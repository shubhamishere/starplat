export async function Compute_PR(device, adj_dataBuffer, adj_offsetsBuffer, nodeCount, props = {}, rev_adj_dataBuffer = null, rev_adj_offsetsBuffer = null) {
  console.log('[WebGPU] Compute start: Compute_PR with nodeCount=', nodeCount);
  let result = 0;
  const resultBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const propEntries = [];
  const pageRankBuffer = (props['pageRank'] && props['pageRank'].buffer) ? props['pageRank'].buffer : device.createBuffer({ size: (props['pageRank'] && props['pageRank'].data) ? props['pageRank'].data.byteLength : Math.max(1, nodeCount) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  if (props['pageRank'] && props['pageRank'].data && !(props['pageRank'].buffer)) { device.queue.writeBuffer(pageRankBuffer, 0, props['pageRank'].data); }
  propEntries.push({ binding: 6, resource: { buffer: pageRankBuffer } });
  const pageRank_nxtBuffer = (props['pageRank_nxt'] && props['pageRank_nxt'].buffer) ? props['pageRank_nxt'].buffer : device.createBuffer({ size: (props['pageRank_nxt'] && props['pageRank_nxt'].data) ? props['pageRank_nxt'].data.byteLength : Math.max(1, nodeCount) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  if (props['pageRank_nxt'] && props['pageRank_nxt'].data && !(props['pageRank_nxt'].buffer)) { device.queue.writeBuffer(pageRank_nxtBuffer, 0, props['pageRank_nxt'].data); }
  propEntries.push({ binding: 7, resource: { buffer: pageRank_nxtBuffer } });
  device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));
  const paramsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([nodeCount, 0, 0, 0]));
  let num_nodes = 0;
  // [skip] propNode/propEdge 'pageRank_nxt' is a GPU buffer
  // Property initialization: attachNodeProperty
  { const N = nodeCount; const initArr = new Float32Array(N);
    for (let i = 0; i < N; i++) { initArr[i] = (1 / num_nodes); }
    device.queue.writeBuffer(pageRankBuffer, 0, initArr); }
  { const N = nodeCount; const initArr = new Float32Array(N);
    for (let i = 0; i < N; i++) { initArr[i] = 0; }
    device.queue.writeBuffer(pageRank_nxtBuffer, 0, initArr); }
  let iterCount = 0;
  let diff;
  // Do-while loop (WebGPU host)
  let __dwIterations = 0; const __dwMax = 1000;
  do {
    device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));
  // Reset result before dispatch
  device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));
  const kernel_res_0 = await launchkernel_0(device, adj_dataBuffer, adj_offsetsBuffer, paramsBuffer, resultBuffer, propEntries, nodeCount, rev_adj_dataBuffer, rev_adj_offsetsBuffer);
  result = kernel_res_0;
  pageRank = pageRank_nxt;
  iterCount = iterCount + 1;
    __dwIterations++;
  } while (((diff > beta) && (iterCount < maxIter)) && __dwIterations < __dwMax);
  if (props['pageRank'] && (props['pageRank'].usage === 'out' || props['pageRank'].usage === 'inout' || props['pageRank'].readback === true)) {
    const sizeBytes_pageRank = (props['pageRank'].data) ? props['pageRank'].data.byteLength : Math.max(1, nodeCount) * 4;
    const rb_pageRank = device.createBuffer({ size: sizeBytes_pageRank, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    { const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(pageRankBuffer, 0, rb_pageRank, 0, sizeBytes_pageRank); device.queue.submit([enc.finish()]); }
    await rb_pageRank.mapAsync(GPUMapMode.READ);
    const view_pageRank = new Float32Array(rb_pageRank.getMappedRange());
    props['pageRank'].dataOut = new Float32Array(view_pageRank);
    rb_pageRank.unmap();
  }
  if (props['pageRank_nxt'] && (props['pageRank_nxt'].usage === 'out' || props['pageRank_nxt'].usage === 'inout' || props['pageRank_nxt'].readback === true)) {
    const sizeBytes_pageRank_nxt = (props['pageRank_nxt'].data) ? props['pageRank_nxt'].data.byteLength : Math.max(1, nodeCount) * 4;
    const rb_pageRank_nxt = device.createBuffer({ size: sizeBytes_pageRank_nxt, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    { const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(pageRank_nxtBuffer, 0, rb_pageRank_nxt, 0, sizeBytes_pageRank_nxt); device.queue.submit([enc.finish()]); }
    await rb_pageRank_nxt.mapAsync(GPUMapMode.READ);
    const view_pageRank_nxt = new Float32Array(rb_pageRank_nxt.getMappedRange());
    props['pageRank_nxt'].dataOut = new Float32Array(view_pageRank_nxt);
    rb_pageRank_nxt.unmap();
  }
  console.log('[WebGPU] Compute end: returning result', result);
  return result;
}

async function launchkernel_0(device, adj_dataBuffer, adj_offsetsBuffer, paramsBuffer, resultBuffer, propEntries, nodeCount, rev_adj_dataBuffer = null, rev_adj_offsetsBuffer = null) {
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
  const bindEntries = [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ];
  bindEntries.push({ binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
  bindEntries.push({ binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
  const bindGroupLayout = device.createBindGroupLayout({ entries: bindEntries });
  const pipeline = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }), compute: { module: shaderModule, entryPoint: 'main' } });
  console.log('[WebGPU] launchkernel_0: pipeline created');
  
  // Using shared result/property buffers provided by caller
  const readBuffer = device.createBuffer({ 
    size: 4, 
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ 
  });
  
  const entries = [
      { binding: 0, resource: { buffer: adj_offsetsBuffer } },
      { binding: 1, resource: { buffer: adj_dataBuffer } },
      { binding: 4, resource: { buffer: paramsBuffer } },
      { binding: 5, resource: { buffer: resultBuffer } }
  ];
  // Add reverse CSR buffers if provided
  if (rev_adj_offsetsBuffer) {
    entries.push({ binding: 2, resource: { buffer: rev_adj_offsetsBuffer } });
  }
  if (rev_adj_dataBuffer) {
    entries.push({ binding: 3, resource: { buffer: rev_adj_dataBuffer } });
  }
  entries.push(...propEntries);
  const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries });
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

async function launchkernel_1(device, adj_dataBuffer, adj_offsetsBuffer, paramsBuffer, resultBuffer, propEntries, nodeCount, rev_adj_dataBuffer = null, rev_adj_offsetsBuffer = null) {
  console.log('[WebGPU] launchkernel_1: begin');
  const shaderCode = await (await fetch('kernel_1.wgsl')).text();
  console.log('[WebGPU] launchkernel_1: WGSL fetched, size', shaderCode.length);
  const shaderModule = device.createShaderModule({ code: shaderCode });
  if (shaderModule.getCompilationInfo) {
    const info = await shaderModule.getCompilationInfo();
    for (const m of info.messages || []) {
      const s = m.lineNum !== undefined ? `${m.lineNum}:${m.linePos}` : '';
      console[(m.type === 'error') ? 'error' : 'warn']('[WGSL]', m.type, s, m.message);
    }
  }
  const bindEntries = [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ];
  bindEntries.push({ binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
  bindEntries.push({ binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
  const bindGroupLayout = device.createBindGroupLayout({ entries: bindEntries });
  const pipeline = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }), compute: { module: shaderModule, entryPoint: 'main' } });
  console.log('[WebGPU] launchkernel_1: pipeline created');
  
  // Using shared result/property buffers provided by caller
  const readBuffer = device.createBuffer({ 
    size: 4, 
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ 
  });
  
  const entries = [
      { binding: 0, resource: { buffer: adj_offsetsBuffer } },
      { binding: 1, resource: { buffer: adj_dataBuffer } },
      { binding: 4, resource: { buffer: paramsBuffer } },
      { binding: 5, resource: { buffer: resultBuffer } }
  ];
  // Add reverse CSR buffers if provided
  if (rev_adj_offsetsBuffer) {
    entries.push({ binding: 2, resource: { buffer: rev_adj_offsetsBuffer } });
  }
  if (rev_adj_dataBuffer) {
    entries.push({ binding: 3, resource: { buffer: rev_adj_dataBuffer } });
  }
  entries.push(...propEntries);
  const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries });
  console.log('[WebGPU] launchkernel_1: bindGroup created');
  
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  
  // Dispatch one workgroup per 256 nodes (ensure at least 1 group)
  let __groups = Math.ceil(nodeCount / 256);
  if (__groups < 1) { __groups = 1; }
  pass.dispatchWorkgroups(__groups, 1, 1);
  pass.end();
  console.log('[WebGPU] launchkernel_1: dispatched groups', __groups);
  
  encoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, 4);
  device.queue.submit([encoder.finish()]);
  console.log('[WebGPU] launchkernel_1: submitted');
  
  try {
    await Promise.race([readBuffer.mapAsync(GPUMapMode.READ), new Promise((_,rej)=>setTimeout(()=>rej(new Error('mapAsync timeout')), 10000))]);
  } catch (e) { console.error('[WebGPU] mapAsync error:', e?.message || e); throw e; }
  const result = new Uint32Array(readBuffer.getMappedRange())[0];
  readBuffer.unmap();
  console.log('[WebGPU] launchkernel_1: result', result);
  
  return result;
}

