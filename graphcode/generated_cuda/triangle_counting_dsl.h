// FOR BC: nvcc bc_dsl_v2.cu -arch=sm_60 -std=c++14 -rdc=true # HW must support CC 6.0+ Pascal or after
#ifndef GENCPP_TRIANGLE_COUNTING_DSL_H
#define GENCPP_TRIANGLE_COUNTING_DSL_H
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#include "../graph.hpp"
#include "../geomCompleteGraph.hpp"
#include "../dynamic_mst_delete_cuda/libcuda.cuh"
#include "../CUDA_GNN.cuh"
#include <cooperative_groups.h>

void Compute_TC(graph& g);



__device__ long triangle_count ; // DEVICE ASSTMENT in .h

__global__ void Compute_TC_kernel(int V, int E, int* d_meta, int* d_data){ // BEGIN KER FUN via ADDKERNEL
  float num_nodes  = V;
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { // FOR NBR ITR 
    int u = d_data[edge];
    if (u < v){ // if filter begin 
      for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { // FOR NBR ITR 
        int w = d_data[edge];
        if (w > v){ // if filter begin 
          if (findNeighborSorted(u, w, d_meta, d_data)){ // if filter begin 
            atomicAdd((unsigned long long*)& triangle_count, (unsigned long long)1);

          } // if filter end

        } // if filter end

      } //  end FOR NBR ITR. TMP FIX!

    } // if filter end

  } //  end FOR NBR ITR. TMP FIX!
} // end KER FUNC

#endif
