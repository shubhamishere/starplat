// Graph algorithm compute shader
@group(0) @binding(0) var<storage, read> adj_data: array<u32>;
@group(0) @binding(1) var<storage, read> adj_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> properties: array<atomic<u32>>;

fn atomicAddF32(ptr: ptr<storage, atomic<u32>>, val: f32) -> f32 {
  loop {
    let oldBits: u32 = atomicLoad(ptr);
    let oldVal: f32 = bitcast<f32>(oldBits);
    let newVal: f32 = oldVal + val;
    let newBits: u32 = bitcast<u32>(newVal);
    let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);
    if (res.exchanged) { return oldVal; }
  }
}
fn atomicMinF32(ptr: ptr<storage, atomic<u32>>, val: f32) {
  loop {
    let oldBits: u32 = atomicLoad(ptr);
    let oldVal: f32 = bitcast<f32>(oldBits);
    let newVal: f32 = min(oldVal, val);
    let newBits: u32 = bitcast<u32>(newVal);
    let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);
    if (res.exchanged) { return; }
  }
}
fn atomicMaxF32(ptr: ptr<storage, atomic<u32>>, val: f32) {
  loop {
    let oldBits: u32 = atomicLoad(ptr);
    let oldVal: f32 = bitcast<f32>(oldBits);
    let newVal: f32 = max(oldVal, val);
    let newBits: u32 = bitcast<u32>(newVal);
    let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);
    if (res.exchanged) { return; }
  }
}

/**
 * Checks if there's an edge between vertices u and w
 */
fn findEdge(u: u32, w: u32) -> bool {
  for (var edge = adj_offsets[u]; edge < adj_offsets[u + 1]; edge++) {
    if (adj_data[edge] == w) {
      return true;
    }
  }
  return false;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let v = global_id.x;
  let node_count = atomicLoad(&properties[0u]);
  
  if (v >= node_count) {
    return;
  }

  atomicAddF32(&result, f32(1));
}
