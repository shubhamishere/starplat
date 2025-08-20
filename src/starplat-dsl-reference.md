# StarPlat DSL Reference Guide

## Overview
StarPlat is a domain-specific language (DSL) for graph analytics that generates high-performance parallel code for various backends (CUDA, OpenMP, WebGPU, etc.). This document provides a comprehensive reference for all StarPlat language constructs, graph-specific properties, and built-in functions.

## Table of Contents
1. [Basic Types](#basic-types)
2. [Graph Types](#graph-types)
3. [Property Types](#property-types)
4. [Function Declaration](#function-declaration)
5. [Control Flow](#control-flow)
6. [Graph Iteration](#graph-iteration)
7. [Graph Methods](#graph-methods)
8. [Property Operations](#property-operations)
9. [Reduction Operations](#reduction-operations)
10. [Advanced Constructs](#advanced-constructs)
11. [Constants and Literals](#constants-and-literals)
12. [Examples](#examples)

---

## Basic Types

StarPlat supports standard primitive types:

```cpp
int       // 32-bit signed integer
long      // 64-bit signed integer (backend dependent)
float     // 32-bit floating point
double    // 64-bit floating point (may map to float on some backends)
bool      // Boolean type (True/False)
```

---

## Graph Types

### Graph Declaration
```cpp
Graph g              // Undirected graph
Graph<int> g         // Graph with integer node/edge indices
```

### Specialized Graph Types
```cpp
DirectedGraph g      // Explicitly directed graph
Graph_List gl        // List of graphs
GeomCompleteGraph g  // Complete geometric graph
```

---

## Property Types

Properties are arrays associated with nodes or edges in the graph.

### Node Properties
```cpp
propNode<int> distances        // Integer property per node
propNode<float> weights        // Float property per node  
propNode<bool> visited         // Boolean property per node
propNode<double> values        // Double property per node
```

### Edge Properties
```cpp
propEdge<int> capacity         // Integer property per edge
propEdge<float> weights        // Float property per edge
propEdge<bool> used           // Boolean property per edge
```

### Property Access
```cpp
// Direct property access
node.property_name             // Access property of specific node
edge.property_name             // Access property of specific edge

// Array-style access
distances[v]                   // Access distance property of node v
weights[e]                     // Access weight property of edge e
```

---

## Function Declaration

### Basic Function Syntax
```cpp
function FunctionName(Graph g, propNode<int> prop, int param) {
    // Function body
    return result;
}
```

### Parameter Types
- `Graph g` - Graph parameter
- `propNode<T> prop` - Node property parameter
- `propEdge<T> prop` - Edge property parameter  
- `node src` - Single node parameter
- `edge e` - Single edge parameter
- Standard types (`int`, `float`, etc.)

---

## Control Flow

### Conditional Statements
```cpp
if (condition) {
    // statements
} else {
    // statements
}
```

### Loops
```cpp
// Standard for loop
for (int i = 0; i < count; i++) {
    // statements
}

// While loop
while (condition) {
    // statements
}

// Do-while loop
do {
    // statements
} while (condition);
```

### Fixed Point Iteration
```cpp
fixedPoint until (convergence_condition) {
    // Iterative computation
    // Continues until convergence condition is met
}
```

---

## Graph Iteration

### Node Iteration
```cpp
// Iterate over all nodes
forall(v in g.nodes()) {
    // Process node v
}

// Filtered node iteration
forall(v in g.nodes().filter(condition)) {
    // Process nodes matching condition
}
```

### Edge Iteration
```cpp
// Iterate over all edges
forall(e in g.edges()) {
    // Process edge e
}
```

### Neighbor Iteration
```cpp
// Iterate over outgoing neighbors
forall(nbr in g.neighbors(v)) {
    // Process neighbor nbr
}

// Iterate over incoming neighbors (reverse edges)
for (nbr in g.nodes_to(v)) {
    // Process incoming neighbor nbr
}

// Filtered neighbor iteration
forall(nbr in g.neighbors(v).filter(condition)) {
    // Process filtered neighbors
}
```

### Advanced Iteration
```cpp
// BFS iteration
iterateInBFS(v in g.nodes() from source) {
    // Process nodes in BFS order
}

// Reverse iteration (for algorithms like Betweenness Centrality)
iterateInReverse(v != source) {
    // Process nodes in reverse order
}
```

---

## Graph Methods

### Basic Graph Properties
```cpp
g.num_nodes()                  // Number of nodes in graph
g.num_edges()                  // Number of edges in graph
```

### Neighbor Operations
```cpp
g.neighbors(v)                 // Get outgoing neighbors of node v
g.nodes_to(v)                  // Get incoming neighbors of node v (reverse edges)
g.count_outNbrs(v)            // Count outgoing neighbors
g.count_inNbrs(v)             // Count incoming neighbors
```

### Edge Operations
```cpp
g.is_an_edge(u, v)            // Check if edge exists from u to v
g.get_edge(u, v)              // Get edge object between u and v
```

### Property Management
```cpp
g.attachNodeProperty(prop = initial_value)  // Initialize node property
g.attachEdgeProperty(prop = initial_value)  // Initialize edge property

// Multiple property initialization
g.attachNodeProperty(dist = INF, visited = False, parent = -1)
```

---

## Property Operations

### Property Assignment
```cpp
v.property = value             // Direct assignment
property[v] = value           // Array-style assignment

// Property copying
property1 = property2         // Copy entire property array
```

### Atomic Property Updates
StarPlat automatically handles thread-safety for property updates in parallel contexts:

```cpp
// These become atomic operations in parallel contexts
v.distance += 1               // Atomic addition
v.count -= 1                  // Atomic subtraction  
v.flags |= mask              // Atomic bitwise OR
v.flags &= mask              // Atomic bitwise AND
```

---

## Reduction Operations

### Basic Reductions
```cpp
// Sum reduction
sum += value                  // Add to sum
triangle_count += 1          // Increment counter

// Min/Max reductions  
<v.dist, v.updated> = <Min(v.dist, new_dist), True>  // Atomic min with flag
result = Max(a, b)           // Maximum of two values
```

### Reduction Operators
```cpp
Min(a, b)                    // Minimum of two values
Max(a, b)                    // Maximum of two values  
Sum(values)                  // Sum of values
Count(condition)             // Count elements matching condition
```

---

## Advanced Constructs

### Set Operations
```cpp
SetN<g> sourceSet           // Set of nodes from graph g
SetE<g> edgeSet            // Set of edges from graph g

// Set iteration
for (src in sourceSet) {
    // Process each source
}
```

### Container Types
```cpp
NodeMap(type)              // Map with node keys
Vector<type>              // Dynamic array
HashMap<key, value>       // Hash map
HashSet<type>            // Hash set
```

### Edge Access
```cpp
edge e = g.get_edge(u, v)  // Get edge between u and v
e.weight                   // Access edge property
e.source                   // Source node of edge
e.destination             // Destination node of edge
```

---

## Constants and Literals

### Special Constants
```cpp
INF                       // Infinity (large integer value)
True                      // Boolean true
False                     // Boolean false
```

### Numeric Literals
```cpp
42                        // Integer literal
3.14                      // Float literal
2.5f                      // Explicit float literal
1e6                       // Scientific notation
```

---

## Examples

### Triangle Counting
```cpp
function Compute_TC(Graph g) {
  long triangle_count = 0;
  forall(v in g.nodes()) {
    forall(u in g.neighbors(v).filter(u < v)) {
      forall(w in g.neighbors(v).filter(w > v)) {
        if (g.is_an_edge(u, w)) {
          triangle_count += 1;
        }
      }
    }
  }
  return triangle_count;
}
```

### PageRank
```cpp
function ComputePageRank(Graph g, float beta, float delta, int maxIter, propNode<float> pageRank) {
    float numNodes = g.num_nodes();
    propNode<float> pageRankNext;
    g.attachNodeProperty(pageRank = 1 / numNodes, pageRankNext = 0);
    int iterCount = 0;
    float diff;
    do {
        diff = 0.0;
        forall(v in g.nodes()) {
            float sum = 0.0;
            for (nbr in g.nodes_to(v)) {
                sum = sum + nbr.pageRank / g.count_outNbrs(nbr);
            }
            float newPageRank = (1 - delta) / numNodes + delta * sum;
            if(newPageRank - v.pageRank >= 0) {
                diff += newPageRank - v.pageRank;
            } else {
                diff += v.pageRank - newPageRank;
            }
            v.pageRankNext = newPageRank;
        }
        pageRank = pageRankNext;
        iterCount++;
    } while ((diff > beta) && (iterCount < maxIter));
}
```

### Single-Source Shortest Path (SSSP)
```cpp
function Compute_SSSP(Graph g, propNode<int> dist, propEdge<int> weight, node src) {
    propNode<bool> modified; 
    propNode<bool> modified_nxt;
    g.attachNodeProperty(dist = INF, modified = False, modified_nxt = False);
    src.modified = True; 
    src.dist = 0;
    
    bool finished = False;
    fixedPoint until (finished: !modified) {
        forall(v in g.nodes().filter(modified == True)) {
            forall(nbr in g.neighbors(v)) {          
                edge e = g.get_edge(v, nbr);
                <nbr.dist, nbr.modified_nxt> = <Min(nbr.dist, v.dist + e.weight), True>;
            }
        }
        modified = modified_nxt;
        g.attachNodeProperty(modified_nxt = False);
    }          
}
```

---

## Backend-Specific Notes

### WebGPU Backend
- Supports reverse edge traversal via `g.nodes_to(v)`
- Automatic atomic operations for property updates in parallel contexts
- Fixed-point iterations handled with host-side orchestration
- Property types mapped to appropriate WGSL buffer types

### CUDA Backend  
- Optimized memory access patterns for GPU execution
- Efficient atomic operations for reductions
- Support for dynamic parallelism where applicable

### OpenMP Backend
- Thread-safe property updates using OpenMP atomic directives
- Load balancing across CPU cores
- Vectorization optimizations where possible

---

## Language Design Philosophy

StarPlat is designed with the following principles:

1. **Graph-Native**: Built-in support for graph data structures and common graph operations
2. **Parallel-First**: Automatic parallelization with thread-safe property updates  
3. **Backend-Agnostic**: Same code generates optimized implementations for different architectures
4. **Algorithm-Focused**: High-level constructs that map naturally to graph algorithm patterns
5. **Performance-Oriented**: Generates high-performance code comparable to hand-optimized implementations

---

## Version Information

This reference covers StarPlat DSL constructs as implemented in the current version. For the latest updates and backend-specific features, refer to the individual backend documentation in `src/backends/`.
