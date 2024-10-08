#include "convexHull.h"


void updateDistance(
  int*  label,float*  distance,int*  xCoord,int*  yCoord,
  int*  L,int*  R)
{




  int* d_L = (int*)memset(sizeof(int) * L.size());
  hipMemcpyToSymbol(d_L, &L, sizeof(int*), 0, hipMemcpyHostToDevice);
  float* d_distance = (float*)memset(sizeof(int) * distance.size());
  hipMemcpyToSymbol(d_distance, &distance, sizeof(float*), 0, hipMemcpyHostToDevice);
  int* d_R = (int*)memset(sizeof(int) * R.size());
  hipMemcpyToSymbol(d_R, &R, sizeof(int*), 0, hipMemcpyHostToDevice);
  int* d_label = (int*)memset(sizeof(int) * label.size());
  hipMemcpyToSymbol(d_label, &label, sizeof(int*), 0, hipMemcpyHostToDevice);

  const unsigned threadsPerBlock = 8;
  const unsigned numThreads = (V < threadsPerBlock) ? V : 8;
  const unsigned numBlocks = (V + threadsPerBlock - 1) / threadsPerBlock;

  updateDistance_kernel0<<<numBlocks, threadsPerBlock>>>(V, E, d_L, d_distance, d_R, d_label);
  hipDeviceSynchronize();
  hipMemcpyToSymbol(&L, d_L, sizeof(int*), 0, hipMemcpyDeviceToHost);
  hipMemcpyToSymbol(&distance, d_distance, sizeof(float*), 0, hipMemcpyDeviceToHost);
  hipMemcpyToSymbol(&R, d_R, sizeof(int*), 0, hipMemcpyDeviceToHost);
  hipMemcpyToSymbol(&label, d_label, sizeof(int*), 0, hipMemcpyDeviceToHost);


}
void convexHull(
  int  n,int*  xCoord,int*  yCoord)
{






}
