#ifndef GENCPP_PAGERANKDSLV3_H
#define GENCPP_PAGERANKDSLV3_H
#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<atomic>
#include<set>
#include<vector>
#include"../mpi_header/graph_mpi.h"

void ComputePageRank(Graph& g, float beta, float delta, int maxIter, 
  NodeProperty<float>& pageRank, boost::mpi::communicator world );

#endif
