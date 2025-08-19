// Graph algorithm compute shader
@group(0) @binding(0) var<storage, read> adj_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> adj_data: array<u32>;
struct Params { node_count: u32; _pad0: u32; _pad1: u32; _pad2: u32; };
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> result: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> properties: array<atomic<u32>>;

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
  for (var e = adj_offsets[u]; e < adj_offsets[u + 1u]; e = e + 1u) {
    if (adj_data[e] == w) { return true; }
  }
  return false;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let v = global_id.x;
  let node_count = params.node_count;
  
  if (v >= node_count) {
    return;
  }

  for (var edge = adj_offsets[v]; edge < adj_offsets[v + 1u]; edge = edge + 1u) {
    let u = adj_data[edge];
    if ((u < v)) {
      for (var edge = adj_offsets[v]; edge < adj_offsets[v + 1u]; edge = edge + 1u) {
        let w = adj_data[edge];
        if ((w > v)) {
          if (findEdge(u, w)) {
            atomicAdd(&result, u32(1));
          }
        }
      }
    }
  }
}
