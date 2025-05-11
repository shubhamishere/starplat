#include "delaunayGenerated.h"

auto comparePoints(
    Point &p1, Point &p2)
{

  return p1.x < p2.x;
}

void findPointBounds(
    std::vector<Point> &points, std::vector<double> &bounds)
{

  Point p0 = points[0];
  double xmin = p0.x;
  double ymin = p0.y;
  double xmax = xmin;
  double ymax = ymin;
  int i = 1;

  while (i < points.size())
  {
    Point pi = points[i];

    if (pi.x < xmin)
    {
      xmin = pi.x;
    }

    if (pi.x > xmax)
    {
      xmax = pi.x;
    }

    if (pi.y < ymin)
    {
      ymin = pi.y;
    }

    if (pi.y > ymax)
    {
      ymax = pi.y;
    }
    i++;
  }
  bounds[0] = xmin;
  bounds[1] = ymin;
  bounds[2] = xmax;
  bounds[3] = ymax;
}

void createSuperTriangle(
    std::vector<Point> &points, std::vector<double> &bounds)
{

  double xmin = bounds[0];
  double ymin = bounds[1];
  double xmax = bounds[2];
  double ymax = bounds[3];
  double dmax;
  double dx = xmax - xmin;
  double dy = ymax - ymin;

  if (dx > dy)
  {
    dmax = dx;
  }
  else
  {
    dmax = dy;
  }
  double xmid = (xmax + xmin) / 2;
  double ymid = (ymax + ymin) / 2;
  Point p1;
  p1.x = xmid - 20 * dmax;
  p1.y = ymid - dmax;
  points.push_back(p1);

  p1.x = xmid;
  p1.y = ymid + 20 * dmax;
  points.push_back(p1);

  p1.x = xmid + 20 * dmax;
  p1.y = ymid - dmax;
  points.push_back(p1);
}

void removeSuperTriangles(
    std::vector<Triangle> &triangles, int nv, std::vector<int> &numTriangles)
{

  int ntri = numTriangles[0];
  int i = 0;

  while (i < ntri)
  {
    Triangle t1 = triangles[i];
    int tp1 = t1.p1;
    int tp2 = t1.p2;
    int tp3 = t1.p3;

    if (tp1 >= nv || tp2 >= nv || tp3 >= nv)
    {
      triangles[i] = triangles[ntri - 1];
      ntri--;
      i--;
    }
    i++;
  }
  numTriangles[0] = ntri;
}

void removeDuplicateEdges(
    std::vector<Edge> &edges, int nedge)
{

  int j = 0;

  while (j < nedge - 1)
  {
    int k = j + 1;

    while (k < nedge)
    {
      Edge ej = edges[j];
      Edge ek = edges[k];
      int pj1 = ej.p1;
      int pj2 = ej.p2;
      int pk1 = ek.p1;
      int pk2 = ek.p2;

      if ((pj1 == pk2) && (pj2 == pk1))
      {
        Edge e1;
        e1.p1 = -1;
        e1.p2 = -1;
        edges[j] = e1;
        edges[k] = e1;
      }
      k++;
    }
    j++;
  }
}

void createNewTriangles(
    std::vector<Edge> &edges, int nedge, std::vector<Triangle> &triangles, std::vector<int> &complete,
    std::vector<int> &numTriangles, int i)
{

  int ntri = numTriangles[0];
  int j = 0;

  while (j < nedge)
  {
    Edge ej = edges[j];

    if ((ej.p1 >= 0 && ej.p2 >= 0))
    {
      Triangle t1;
      t1.p1 = ej.p1;
      t1.p2 = ej.p2;
      t1.p3 = i;
      triangles[ntri] = t1;
      complete[ntri] = false;
      ntri++;
    }
    j++;
  }
  numTriangles[0] = ntri;
}

auto circumCircleCheck(
    double xp, double yp, double x1, double y1,
    double x2, double y2, double x3, double y3,
    std::vector<double> &circumResults)
{

  double EPSILON = 1e-10;
  double D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));

  if (fabs(D) < EPSILON)
  {
    circumResults[0] = 0;
    circumResults[1] = 0;
    circumResults[2] = 0;
    return false;
  }
  double D1 = (x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2);
  double D2 = (x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1);
  double xc = D1 / D;
  double yc = D2 / D;
  double dx = x2 - xc;
  double dy = y2 - yc;
  double rsqr = dx * dx + dy * dy;
  double r = sqrt(rsqr);
  dx = xp - xc;
  dy = yp - yc;
  double drsqr = dx * dx + dy * dy;
  circumResults[0] = xc;
  circumResults[1] = yc;
  circumResults[2] = r;

  if (drsqr <= rsqr + EPSILON)
  {
    return true;
  }
  else
  {
    return false;
  }
}

auto Triangulate(
    std::vector<Point> &points)
{

  double EPSILON = 1e-06;
  int ntri = 0;
  int nedge = 0;
  int nv = points.size();
  int trimax = 4 * nv;
  int emax = 200;
  thrust::device_vector<Point> d_points = points;
  thrust::sort(d_points.begin(), d_points.end(), [] __host__ __device__(Point & p1, Point & p2)
               { return p1.x < p2.x; });
  thrust::copy(d_points.begin(), d_points.end(), points.begin());

  std::vector<int> complete(trimax, 0);
  std::vector<Edge> edges(emax);
  std::vector<Point> localPoints;
  localPoints = points;
  std::vector<double> bounds(4);
  findPointBounds(localPoints, bounds);

  createSuperTriangle(localPoints, bounds);

  std::vector<Triangle> triangles(3 * nv);
  Triangle tri;
  tri.p1 = nv;
  tri.p2 = nv + 1;
  tri.p3 = nv + 2;
  triangles[0] = tri;
  complete[0] = false;
  ntri = 1;
  int i = 0;
  double xp;
  double yp;
  double x1;
  double y1;
  double x2;
  double y2;
  double x3;
  double y3;
  double xc;
  double yc;
  double r;
  bool inside;
  int ind1;
  std::vector<int> numTriangles(1);

  while (i < nv)
  {
    Point pi = localPoints[i];
    xp = pi.x;
    yp = pi.y;
    nedge = 0;
    int j = 0;

    while (j < ntri)
    {

      if (complete[j] == false)
      {
        Triangle t1 = triangles[j];
        ind1 = t1.p1;
        Point p1 = localPoints[ind1];
        x1 = p1.x;
        y1 = p1.y;
        ind1 = t1.p2;
        p1 = localPoints[ind1];
        x2 = p1.x;
        y2 = p1.y;
        ind1 = t1.p3;
        p1 = localPoints[ind1];
        x3 = p1.x;
        y3 = p1.y;
        std::vector<double> circumResults(3);
        inside = circumCircleCheck(xp, yp, x1, y1, x2, y2, x3, y3, circumResults);
        xc = circumResults[0];
        yc = circumResults[1];
        r = circumResults[2];

        if (xc + r + EPSILON < xp)
        {
          complete[j] = true;
        }

        if (inside)
        {

          if (nedge + 3 >= emax)
          {
            emax = emax * 2;
            edges.resize(emax);
          }
          Edge e1;
          e1.p1 = t1.p1;
          e1.p2 = t1.p2;
          edges[nedge] = e1;
          e1.p1 = t1.p2;
          e1.p2 = t1.p3;
          edges[nedge + 1] = e1;
          e1.p1 = t1.p3;
          e1.p2 = t1.p1;
          edges[nedge + 2] = e1;
          nedge = nedge + 3;
          triangles[j] = triangles[ntri - 1];
          complete[j] = complete[ntri - 1];
          ntri--;
          j--;
        }
      }
      j++;
    }
    removeDuplicateEdges(edges, nedge);

    numTriangles[0] = ntri;
    createNewTriangles(edges, nedge, triangles, complete, numTriangles, i);

    ntri = numTriangles[0];
    i++;
  }
  numTriangles[0] = ntri;
  removeSuperTriangles(triangles, nv, numTriangles);

  ntri = numTriangles[0];
  return triangles;
}

int main(int argc, char *argv[])
{
  std::vector<Point> points;
  double x, y, z;
  std::string inputFileName = argv[1];
  auto startFileRead = std::chrono::high_resolution_clock::now();
  std::ifstream inputFile(inputFileName);
  while (inputFile >> x >> y >> z)
  {
    points.push_back(Point{x, y});
  }
  inputFile.close();
  auto endFileRead = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fileReadTime = endFileRead - startFileRead;
  std::cout << "File reading time: " << fileReadTime.count() << " seconds\n";
  auto startTriangulate = std::chrono::high_resolution_clock::now();

  // NOTE:
  // Due to uncertainity of copy/move semantics chosen by compiler,
  // the execution times may vary. In report, we do not consider time
  // taken by copy/move constructor of triangles when we return.
  // This is consistent with how we time CGAL's triangulation.
  std::vector<Triangle> triangles = Triangulate(points);
  auto endTriangulate = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> triangulateTime = endTriangulate - startTriangulate;
  std::cout << "Number of points: " << points.size() << std::endl;
  std::cout << "Triangulation time: " << triangulateTime.count() << " seconds\n";
  std::chrono::duration<double> totalTime = fileReadTime + triangulateTime;
  std::cout << "Total execution time: " << totalTime.count() << " seconds\n";
  return 0;
}