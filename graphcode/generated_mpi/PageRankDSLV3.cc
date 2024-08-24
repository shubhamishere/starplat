#include"PageRankDSLV3.h"

void ComputePageRank(Graph& g, float beta, float delta, int maxIter, 
  NodeProperty<float>& pageRank, boost::mpi::communicator world )
{
  float numNodes = (float)g.num_nodes( );
  NodeProperty<float> pageRankNext;
  pageRank.attachToGraph(&g, (float)1 / numNodes);
  pageRankNext.attachToGraph(&g, (float)0);
  int iterCount = 0;
  float diff = 0.0 ;
  do
  {
    diff = 0.000000;
    world.barrier();
    for (int v = g.start_node(); v <= g.end_node(); v ++) 
    {
      float sum = 0.000000;
      for (int nbr : g.getInNeighbors(v)) 
      {
        sum = sum + pageRank.getValue(nbr) / g.num_out_nbrs(nbr);
      }


      float newPageRank = (1 - delta) / numNodes + delta * sum;
      if (newPageRank - pageRank.getValue(v) >= 0 )
      {
        diff = ( diff + newPageRank - pageRank.getValue(v)) ;
      }
      else
      {
        diff = ( diff + pageRank.getValue(v) - newPageRank) ;
      }
      pageRankNext.setValue(v,newPageRank);
    }
    world.barrier();

    float diff_temp = diff;
    MPI_Allreduce(&diff_temp,&diff,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);


    pageRank = pageRankNext;
    iterCount++;
  }
  while((diff > beta) && (iterCount < maxIter));
}
