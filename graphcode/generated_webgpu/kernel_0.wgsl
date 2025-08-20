// Graph algorithm compute shader
@group(0) @binding(0) var<storage, read> adj_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> adj_data: array<u32>;
@group(0) @binding(2) var<storage, read> rev_adj_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> rev_adj_data: array<u32>;
struct Params { node_count: u32; _pad0: u32; _pad1: u32; _pad2: u32; };
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> result: atomic<u32>;
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
fn atomicSubF32(ptr: ptr<storage, atomic<u32>>, val: f32) -> f32 {
  loop {
    let oldBits: u32 = atomicLoad(ptr);
    let oldVal: f32 = bitcast<f32>(oldBits);
    let newVal: f32 = oldVal - val;
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
 * Uses binary search for sorted adjacency lists (O(log n))
 * Falls back to linear search for unsorted lists
 */
fn findEdge(u: u32, w: u32) -> bool {
  let start = adj_offsets[u];
  let end = adj_offsets[u + 1u];
  let degree = end - start;
  
  // For small degree (< 8), use linear search
  if (degree < 8u) {
    for (var e = start; e < end; e = e + 1u) {
      if (adj_data[e] == w) { return true; }
    }
    return false;
  }
  
  // Binary search for larger degrees (assumes sorted adjacency)
  var left = start;
  var right = end;
  while (left < right) {
    let mid = left + (right - left) / 2u;
    let mid_val = adj_data[mid];
    if (mid_val == w) {
      return true;
    } else if (mid_val < w) {
      left = mid + 1u;
    } else {
      right = mid;
    }
  }
  return false;
}

/**
 * Returns the edge index for edge from u to w
 * Returns 0xFFFFFFFFu if edge not found
 */
fn getEdgeIndex(u: u32, w: u32) -> u32 {
  let start = adj_offsets[u];
  let end = adj_offsets[u + 1u];
  for (var e = start; e < end; e = e + 1u) {
    if (adj_data[e] == w) { return e; }
  }
  return 0xFFFFFFFFu; // Edge not found
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let v = global_id.x;
  let node_count = params.node_count;
  
  if (v >= node_count) {
    return;
  }

  var in_degree = (rev_adj_offsets[v + 1] - rev_adj_offsets[v]);
  var out_degree = (adj_offsets[v + 1] - adj_offsets[v]);
  var nodes = params.node_count;
  var edges = arrayLength(&adj_data);
}
