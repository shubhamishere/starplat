#ifndef GENHIP_DELAUNAY_H
#define GENHIP_DELAUNAY_H

#include <iostream>
#include <climits>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "../graph.hpp"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

auto comparePoints(
  Point&  p1,Point&  p2);


void findPointBounds(
  std::vector<Point>&  points,std::vector<double>&  bounds);


// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
void createSuperTriangle(
  std::vector<Point>&  points,std::vector<double>&  bounds);


// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
void removeSuperTriangles(
  std::vector<Triangle>&  triangles,int  nv,std::vector<int>&  numTriangles);


// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
void removeDuplicateEdges(
  std::vector<Edge>&  edges,int  nedge);


// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
void createNewTriangles(
  std::vector<Edge>&  edges,int  nedge,std::vector<Triangle>&  triangles,std::vector<int>&  complete,
  std::vector<int>&  numTriangles,int  i);


// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
auto circumCircleCheck(
  double  xp,double  yp,double  x1,double  y1,
  double  x2,double  y2,double  x3,double  y3,
  std::vector<double>&  circumResults);


// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
auto Triangulate(
  std::vector<Point>&  points);


// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h

#endif
