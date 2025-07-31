async function Compute_TC(device, inputBuffer, outputBuffer, N) {
  // Main function body
  let triangle_count = 0;
  // PARALLEL FORALL (WebGPU kernel launch)
  // See kernel_0.wgsl for kernel code
  await launchkernel_0(device, inputBuffer, outputBuffer, N);
  // Unknown statement type
}

async function launchkernel_0(device, inputBuffer, outputBuffer, N) {
  // 1. Load WGSL
  const shaderCode = await (await fetch('kernel_0.wgsl')).text();
  const shaderModule = device.createShaderModule({ code: shaderCode });
  // 2. Create pipeline
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' }
  });
  // 3. Set up bind group
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } }
    ]
  });
  // 4. Encode and dispatch
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(N / 64));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
  // 5. Read back results (if needed)
  // TODO: Map outputBuffer and read results
}

