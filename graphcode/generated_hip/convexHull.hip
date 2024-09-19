#include "convexHull.h"


__global__ void convexHull(
  int  n,std::vector<int>  xCoord,std::vector<int>  yCoord)
{




  const unsigned threadsPerBlock = 8;
  const unsigned numThreads = (V < threadsPerBlock) ? V : 8;
  const unsigned numBlocks = (V + threadsPerBlock - 1) / threadsPerBlock;



  // Timing starts here
  hipEvent_t start;
  hipEventCreate(&start);
  hipEventRecord(start);

  std::vector<int> label;
  std::vector<float> distance;
  std::vector<int> L;
  std::vector<int> R;
  std::vector<int> mark;
  int A;  int B;  std::vector<int> extreme;
  extreme = extremePoints(xCoord,yCoord);
  A = extreme.front();
  B = extreme.back();
  int index = 0;  
  while (index < n ){

    label.push_back(0);

    if (index == A || index == B) {

      R.push_back(index);



    }  else {

      L.push_back(index);

      mark[index] = 1;


    }
    index++;



  }

  updateDistance(label,distance,xCoord,yCoord,L,R);

  int upperHull = 0;  float upperDist = 0;  int lowerHull = -1;  float lowerDist = -1;  

  R.push_back(upperHull);

  mark[upperHull] = 0;

  R.push_back(lowerHull);

  mark[lowerHull] = 0;
  int prevHullSize = -1;  int currHullSize = R.size();  
  while (currHullSize != prevHullSize ){

    updateDistance(label,distance,xCoord,yCoord,L,R);


    updateHull(distance,xCoord,yCoord,R,mark,label);


    updateLabel(label,R);

    prevHullSize = currHullSize;
    currHullSize = R.size();



  }



  hipEvent_t stop;
  hipEventCreate(&stop);
  hipEventRecord(stop);
  hipEventSynchronize(stop);
  float milliseconds = 0;
  hipEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Time taken: " << milliseconds << " ms" << std::endl;
  hipEventDestroy(start);
  hipEventDestroy(stop);
  // Timing ends here

}
