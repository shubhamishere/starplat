// Deno driver that uses the GENERATED output.js and kernel_0.wgsl
// Usage: deno run --allow-read --unstable-webgpu graphcode/generated_webgpu/driver_triangle_count.js <graph.txt>

import { DelimiterStream } from "jsr:@std/streams/delimiter-stream";
// Import generated functions (requires generator to export them)
import * as gen from "./output.js";

// Polyfill fetch for kernel_0.wgsl local file access used by generated code
if (typeof globalThis.fetch === "function") {
  const origFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    if (typeof input === "string" && input.endsWith(".wgsl")) {
      const path = new URL(input, import.meta.url);
      const text = await Deno.readTextFile(path);
      return new Response(text, { status: 200 });
    }
    return origFetch(input, init);
  };
}

async function main() {
  const filePath = Deno.args[0];
  if (!filePath) {
    console.error("Usage: deno run --allow-read --unstable driver_triangle_count.js <graph.txt>");
    Deno.exit(1);
  }

  const { adj_offsets, adj_data, nodeCountForAllocation } = await processGraph(filePath);

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter found");
  const device = await adapter.requestDevice();
  // Create GPU buffers for generated function signature
  const adjDataBuffer = createGPUBuffer(device, adj_data, GPUBufferUsage.STORAGE);
  const adjOffsetsBuffer = createGPUBuffer(device, adj_offsets, GPUBufferUsage.STORAGE);
  const result = await gen.Compute_TC(device, adjDataBuffer, adjOffsetsBuffer, nodeCountForAllocation);
  console.log(`Triangles: ${result}`);
}

async function processGraph(filePath) {
  const decoder = new TextDecoder();
  const nodes = new Set();
  let maxNode = 0;
  let edgeCount = 0;

  // Pass 1: count edges and nodes
  let file = await Deno.open(filePath, { read: true });
  for await (const chunk of file.readable.pipeThrough(new DelimiterStream(new TextEncoder().encode("\n")))) {
    if (chunk.length === 0) continue;
    const line = decoder.decode(chunk);
    if (line.startsWith("#")) continue;
    const [aStr, bStr] = line.trim().split(/\s+/);
    if (aStr === undefined || bStr === undefined) continue;
    const a = Number(aStr), b = Number(bStr);
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    if (a === b) continue;
    nodes.add(a); nodes.add(b);
    maxNode = Math.max(maxNode, a, b);
    edgeCount++;
  }

  // Build full (undirected) adjacency for the generated kernel's u < v / w > v filters
  const nodeCountForAllocation = maxNode + 1;
  const adjLists = Array.from({ length: nodeCountForAllocation }, () => []);

  file = await Deno.open(filePath, { read: true });
  for await (const chunk of file.readable.pipeThrough(new DelimiterStream(new TextEncoder().encode("\n")))) {
    if (chunk.length === 0) continue;
    const line = decoder.decode(chunk);
    if (line.startsWith("#")) continue;
    const parts = line.trim().split(/\s+/);
    if (parts.length < 2) continue;
    const u = Number(parts[0]);
    const v = Number(parts[1]);
    if (!Number.isFinite(u) || !Number.isFinite(v) || u === v) continue;
    adjLists[u].push(v);
    adjLists[v].push(u);
  }

  const adj_offsets = new Uint32Array(nodeCountForAllocation + 1);
  const flat = [];
  let offset = 0;
  for (let i = 0; i < nodeCountForAllocation; i++) {
    // sort and deduplicate neighbors
    adjLists[i].sort((a, b) => a - b);
    const unique = [];
    let prev = -1;
    for (const nbr of adjLists[i]) {
      if (nbr !== i && nbr !== prev) { // drop self-loops and dups
        unique.push(nbr);
        prev = nbr;
      }
    }
    adj_offsets[i] = offset;
    for (const nbr of unique) flat.push(nbr);
    offset += unique.length;
  }
  adj_offsets[nodeCountForAllocation] = offset;

  const adj_data = new Uint32Array(flat);
  return { adj_offsets, adj_data, nodeCountForAllocation };
}

function createGPUBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  const Ctor = data.constructor;
  new Ctor(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

main().catch((e) => { console.error(e); Deno.exit(1); });

