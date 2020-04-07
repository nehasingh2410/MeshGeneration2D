#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include <stdio.h>
#include <float.h>
#include <iostream>

#include "QuadTree.cuh"

#define MAX_POINTS_SIZE 600
#define MAX_VORONOI_EDGES 1000


extern __device__ double2 gpu_nncrust_edgesforeach_voronoithresholdpoint[MAX_VORONOI_EDGES][MAX_POINTS_SIZE*2];


extern __device__ double2 intersection_points[MAX_VORONOI_EDGES][MAX_POINTS_SIZE * 2];


extern __device__ double2 gpu_delaunay_edgesforeachvoronoi[MAX_VORONOI_EDGES][6 * MAX_POINTS_SIZE - 15];
extern __device__ int gpu_delaunay_edgesindexforeachvoronoi[MAX_VORONOI_EDGES][6 * MAX_POINTS_SIZE - 15];
extern __device__ int countof_gpu_delaunay_edgesforeachvoronoi[MAX_VORONOI_EDGES];

extern __device__ double2 gpu_nncrust_intersectionpoints_foreachvoronoi[MAX_VORONOI_EDGES][MAX_POINTS_SIZE * 2];

__device__ bool almost_equal(const double x, const double y, int ulp)
{
	// the machine epsilon has to be scaled to the magnitude of the values used
	// and multiplied by the desired precision in ULPs (units in the last place)
	return fabs(x - y) <= DBL_EPSILON * fabs(x + y) * static_cast<double>(ulp)
		// unless the result is subnormal
		|| fabs(x - y) < DBL_MIN;
	return true;
}

__device__ bool almost_equal(const double2 x, const double2 y, int ulp)
{
	// the machine epsilon has to be scaled to the magnitude of the values used
	// and multiplied by the desired precision in ULPs (units in the last place)
	bool x_bool = fabs(x.x - y.x) <= DBL_EPSILON * fabs(x.x + y.x) * static_cast<double>(ulp)
		// unless the result is subnormal
		|| fabs(x.x - y.x) < DBL_MIN;
	bool y_bool = fabs(x.y - y.y) <= DBL_EPSILON * fabs(x.y + y.y) * static_cast<double>(ulp)
		// unless the result is subnormal
		|| fabs(x.y - y.y) < DBL_MIN;
	return x_bool & y_bool;
}

__device__ double my_half(const double x)
{
	return 0.5 * x;
}

struct Delaunay_Vector2
{
	__device__ Delaunay_Vector2() = default;
	__device__ Delaunay_Vector2(const Delaunay_Vector2& v) = default;
	
	__device__ Delaunay_Vector2(const double vx, const double vy) :
		x(vx), y(vy)
	{}
	__device__ double dist2(const Delaunay_Vector2& v) const
	{
		const double dx = x - v.x;
		const double dy = y - v.y;
		return dx * dx + dy * dy;
	}

	__device__ double dist(const Delaunay_Vector2& v) const
	{
		return hypot(x - v.x, y - v.y);
	}

	__device__ double norm2() const
	{
		return x * x + y * y;
	}

	__device__ bool operator ==(const Delaunay_Vector2& v) const
	{
		return (this->x == v.x) && (this->y == v.y);
	}

	__device__  Delaunay_Vector2& operator=(const Delaunay_Vector2&) = default;
	
	double x;
	double y;
};

__device__ bool almost_equal(const Delaunay_Vector2& v1, const Delaunay_Vector2& v2, int ulp)
{
	return almost_equal(v1.x, v2.x, ulp) && almost_equal(v1.y, v2.y, ulp);
}

struct Simple_Edge
{
	using VertexType = Delaunay_Vector2;

	__device__ Simple_Edge() = default;

	__device__ Simple_Edge::Simple_Edge(double x1, double y1, double x2, double y2) :
		x1(x1), y1(y1), x2(x2), y2(y2)
	{}
	double x1;
	double y1;
	double x2;
	double y2;
};

struct Delaunay_Edge
{
	using VertexType = Delaunay_Vector2;

	__device__ Delaunay_Edge() = default;
	__device__ Delaunay_Edge(const Delaunay_Edge&) = default;
	
	__device__ Delaunay_Edge(const VertexType& v1, const VertexType& v2) :
		v(&v1), w(&v2)
	{}

	__device__ bool operator ==(const Delaunay_Edge& e) const
	{
		return (*(this->v) == *e.v && *(this->w) == *e.w) ||
			(*(this->v) == *e.w && *(this->w) == *e.v);
	}


	__device__ Delaunay_Edge& operator=(const Delaunay_Edge&) = default;
	
	
	const VertexType* v;
	const VertexType* w;
	bool isBad = false;
	bool isActive = false;
};

__device__ bool almost_equal(const Delaunay_Edge& e1, const Delaunay_Edge& e2, int ulp)
{
	return	(almost_equal(*e1.v, *e2.v, ulp) && almost_equal(*e1.w, *e2.w, ulp)) ||
		(almost_equal(*e1.v, *e2.w, ulp) && almost_equal(*e1.w, *e2.v, ulp));
}

struct Delaunay_Triangle
{
	using EdgeType = Delaunay_Edge;
	using VertexType = Delaunay_Vector2;

	__device__ Delaunay_Triangle() = default;
	__device__ Delaunay_Triangle(const Delaunay_Triangle&) = default;
	
	__device__ Delaunay_Triangle(const VertexType& v1, const VertexType& v2, const VertexType& v3) :
		a(&v1), b(&v2), c(&v3), isBad(false), isActive(false)
	{}

	__device__ bool containsVertex(const VertexType& v, int ulp) const
	{
		// return p1 == v || p2 == v || p3 == v;
		return almost_equal(*a, v, ulp) || almost_equal(*b, v, ulp) || almost_equal(*c, v, ulp);
	}
	__device__ bool circumCircleContains(const VertexType& v) const
	{
		const double ab = a->norm2();
		const double cd = b->norm2();
		const double ef = c->norm2();

		const double ax = a->x;
		const double ay = a->y;
		const double bx = b->x;
		const double by = b->y;
		const double cx = c->x;
		const double cy = c->y;

		const double circum_x = (ab * (cy - by) + cd * (ay - cy) + ef * (by - ay)) / (ax * (cy - by) + bx * (ay - cy) + cx * (by - ay));
		const double circum_y = (ab * (cx - bx) + cd * (ax - cx) + ef * (bx - ax)) / (ay * (cx - bx) + by * (ax - cx) + cy * (bx - ax));

		const VertexType circum(my_half(circum_x), my_half(circum_y));
		const double circum_radius = a->dist2(circum);
		const double dist = v.dist2(circum);
		return dist <= circum_radius;
	}

	__device__ Delaunay_Triangle& operator=(const Delaunay_Triangle&) = default;

	__device__ bool operator ==(const Delaunay_Triangle& t) const
	{
		return	(*this->a == *t.a || *this->a == *t.b || *this->a == *t.c) &&
			(*this->b == *t.a || *this->b == *t.b || *this->b == *t.c) &&
			(*this->c == *t.a || *this->c == *t.b || *this->c == *t.c);
	}
	const VertexType* a;
	const VertexType* b;
	const VertexType* c;
	bool isBad = false;
	bool isActive = false;
};

__device__ bool almost_equal(const Delaunay_Triangle& t1, const Delaunay_Triangle& t2, int ulp)
{
	return	(almost_equal(*t1.a, *t2.a, ulp) || almost_equal(*t1.a, *t2.b, ulp) || almost_equal(*t1.a, *t2.c, ulp)) &&
		(almost_equal(*t1.b, *t2.a, ulp) || almost_equal(*t1.b, *t2.b, ulp) || almost_equal(*t1.b, *t2.c, ulp)) &&
		(almost_equal(*t1.c, *t2.a, ulp) || almost_equal(*t1.c, *t2.b, ulp) || almost_equal(*t1.c, *t2.c, ulp));
}

__global__ void addKernel(double* c, const int* a, const int* b);

//__global__ void delaunayKernel(double2 d[4][MAX_POINTS_SIZE], int* e)
//{
//	int lane_idx = threadIdx.x;
//	using TriangleType = Delaunay_Triangle;
//	using EdgeType = Delaunay_Edge;
//	using VertexType = Delaunay_Vector2;
//
//	const int no_points = 50;
//
//	int no_of_points = e[lane_idx];
//
//	TriangleType _triangles[(no_points + 2) * 2];
//	EdgeType _edges[(no_points)* 3];
//	VertexType _vertices[no_points + 3];
//
//	for (int i = 0; i < no_of_points; i++)
//	{
//		_vertices[i].x = d[lane_idx][i].x;
//		_vertices[i].y = d[lane_idx][i].y;
//	}
//
//	// Determinate the super triangle
//	double minX = _vertices[0].x;
//	double minY = _vertices[0].y;
//	double maxX = minX;
//	double maxY = minY;
//
//	for (int i = 0; i < no_of_points; i++)
//	{
//		VertexType temp = _vertices[i];
//		if (_vertices[i].x < minX) minX = _vertices[i].x;
//		if (_vertices[i].y < minY) minY = _vertices[i].y;
//		if (_vertices[i].x > maxX) maxX = _vertices[i].x;
//		if (_vertices[i].y > maxY) maxY = _vertices[i].y;
//	}
//
//	const double dx = maxX - minX;
//	const double dy = maxY - minY;
//	const double deltaMax = fmax(dx, dy);
//	const double midx = my_half(minX + maxX);
//	const double midy = my_half(minY + maxY);
//
//	const VertexType p1(midx - 2000 * deltaMax, midy - deltaMax);
//	const VertexType p2(midx, midy + 2000 * deltaMax);
//	const VertexType p3(midx + 2000 * deltaMax, midy - deltaMax);
//
//	// Create a list of triangles, and add the supertriangle in it
//	_triangles[0] = TriangleType(p1, p2, p3);
//	_triangles[0].isActive = true;
//	int debug_count = 0;
//	for (int nop = 0; nop < no_of_points; nop++)
//	{
//		VertexType p = _vertices[nop];
//		EdgeType polygon[(no_points + 2) * 4];
//		int polygon_count = 0;
//
//		for (int i = 0; i < sizeof(_triangles) / sizeof(_triangles[0]); i++)
//		{
//			TriangleType t = _triangles[i];
//
//			if (t.isActive)
//			{
//
//				if (t.circumCircleContains(p))
//				{
//					_triangles[i].isBad = true;
//					polygon[polygon_count] = Delaunay_Edge{ *t.a, *t.b };
//					polygon[polygon_count].isActive = true;
//					polygon_count++;
//					polygon[polygon_count] = Delaunay_Edge{ *t.b, *t.c };
//					polygon[polygon_count].isActive = true;
//					polygon_count++;
//					polygon[polygon_count] = Delaunay_Edge{ *t.c, *t.a };
//					polygon[polygon_count].isActive = true;
//					polygon_count++;
//
//
//				}
//			}
//
//		}
//
//		for (int i = 0; i < sizeof(_triangles) / sizeof(_triangles[0]); i++)
//		{
//			// remove triangle if t.isbad = true
//			TriangleType t = _triangles[i];
//			if (t.isActive)
//			{
//				if (t.isBad)
//				{
//					_triangles[i].isActive = false;
//
//				}
//			}
//		}
//
//		for (int i = 0; i < (sizeof(polygon) / sizeof(polygon[0])); i++)
//		{
//			for (int j = i + 1; j < (sizeof(polygon) / sizeof(polygon[0])); j++)
//			{
//				if (polygon[i].isActive && polygon[j].isActive)
//				{
//					if (almost_equal(polygon[i], polygon[j], 2))
//					{
//						polygon[i].isBad = true;
//						polygon[j].isBad = true;
//
//
//					}
//				}
//
//			}
//		}
//
//		for (int i = 0; i < sizeof(polygon) / sizeof(polygon[0]); i++)
//		{
//			// remove edge if e.isbad = true
//			EdgeType e = polygon[i];
//			if (e.isActive)
//			{
//				if (e.isBad)
//				{
//					polygon[i].isActive = false;
//				}
//			}
//
//		}
//
//		for (int i = 0; i < sizeof(polygon) / sizeof(polygon[0]); i++)
//		{
//			EdgeType e = polygon[i];
//			if (e.isActive)
//			{
//
//				bool iterate_bool = true;
//				for (int i = 0; i < sizeof(_triangles) / sizeof(_triangles[0]); i++)
//				{
//
//					TriangleType t = _triangles[i];
//					if (iterate_bool && !(t.isActive))
//					{
//						_triangles[i] = TriangleType(*e.v, *e.w, _vertices[nop]);
//						_triangles[i].isActive = true;
//						iterate_bool = false;
//						debug_count++;
//
//					}
//				}
//			}
//		}
//
//	}
//
//	for (int i = 0; i < sizeof(_triangles) / sizeof(_triangles[0]); i++)
//	{
//		TriangleType t = _triangles[i];
//		if (t.isActive)
//		{
//
//			if (t.containsVertex(p1, 2) || t.containsVertex(p2, 2) || t.containsVertex(p3, 2))
//			{
//
//				_triangles[i].isActive = false;
//
//			}
//
//		}
//		// remove triangle if t.containsVertex(p1) || t.containsVertex(p2) || t.containsVertex(p3);
//	}
//
//
//	int edge_count = 0;
//	int triangle_count = 0;
//
//	for (int i = 0; i < sizeof(_triangles) / sizeof(_triangles[0]); i++)
//	{
//		TriangleType t = _triangles[i];
//
//		if (t.isActive)
//		{
//			_edges[edge_count] = Delaunay_Edge{ *t.a, *t.b };
//			_edges[edge_count].isActive = true;
//			//f[edge_count] = Simple_Edge{ t.a->x, t.a->y, t.b->x, t.b->y };
//			gpu_delaunay_edgesforeachvoronoi[lane_idx][edge_count] = make_double2(t.a->x, t.a->y);
//			gpu_delaunay_edgesforeachvoronoi[lane_idx][edge_count + 1] = make_double2(t.b->x, t.b->y);
//			
//			edge_count++;
//
//			_edges[edge_count] = Delaunay_Edge{ *t.b, *t.c };
//			_edges[edge_count].isActive = true;
//			//f[edge_count] = Simple_Edge{ t.b->x, t.b->y, t.c->x, t.c->y };
//			gpu_delaunay_edgesforeachvoronoi[lane_idx][edge_count + 1] = make_double2(t.b->x, t.b->y);
//			gpu_delaunay_edgesforeachvoronoi[lane_idx][edge_count + 2] = make_double2(t.c->x, t.b->y);
//			edge_count++;
//
//
//			_edges[edge_count] = Delaunay_Edge{ *t.c, *t.a };
//			_edges[edge_count].isActive = true;
//			//f[edge_count] = Simple_Edge{ t.c->x, t.c->y, t.a->x, t.a->y };
//			gpu_delaunay_edgesforeachvoronoi[lane_idx][edge_count + 2] = make_double2(t.b->x, t.b->y);
//			gpu_delaunay_edgesforeachvoronoi[lane_idx][edge_count + 3] = make_double2(t.c->x, t.b->y);
//			edge_count++;
//
//
//			triangle_count++;
//
//		}
//
//	}
//	countof_gpu_delaunay_edgesforeachvoronoi[lane_idx] = edge_count * 2;
//
//	int i = threadIdx.x;
//	Delaunay_Vector2 z1 = Delaunay_Vector2(1.12, 2.34);
//
//	printf("edge count: %d", edge_count);
//	//printf("debug count:%d", debug_count);
//	printf("triangle count: %d", triangle_count);
//	//printf(almost_equal(2.11111111111112, 2.11111111111111,10000) ? "true" : "false");
//
//}
__device__ double double2_distance(double2 a, double2 b)
{
	double dist_squared = (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);
	return sqrt(dist_squared);
}

__device__ double angle_in_triangle(double2 A, double2 B, double2 C)
{
	double c = double2_distance(A, B);
	double a = double2_distance(B, C);
	double b = double2_distance(C, A);

	double cosB = (a*a + c*c - b*b) / (2 * a*c);

	return acos(cosB);
}

__device__ bool onSegment(double2 p, double2 q, double2 r)
{

	if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
		q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
		return true;

	return false;
}

// To find orientation of ordered triplet (p, q, r). 
// The function returns following values 
// 0 --> p, q and r are colinear 
// 1 --> Clockwise 
// 2 --> Counterclockwise 
__device__ int orientation(double2 p, double2 q, double2 r)
{
	
	double val = (q.y - p.y) * (r.x - q.x) -
		(q.x - p.x) * (r.y - q.y);

	if (almost_equal(val, (double)0.0f, 2)) return 0;  // colinear 

	return (val > 0) ? 1 : 2; // clock or counterclock wise 
}

__device__ bool doIntersect(double2 p1, double2 q1, double2 p2, double2 q2)
{
	// Find the four orientations needed for general and 
	// special cases 
	int o1 = orientation(p1, q1, p2);
	int o2 = orientation(p1, q1, q2);
	int o3 = orientation(p2, q2, p1);
	int o4 = orientation(p2, q2, q1);

	// General case 
	if (o1 != o2 && o3 != o4)
		return true;

	// Special Cases 
	// p1, q1 and p2 are colinear and p2 lies on segment p1q1 
	if (o1 == 0 && onSegment(p1, p2, q1)) return true;

	// p1, q1 and q2 are colinear and q2 lies on segment p1q1 
	if (o2 == 0 && onSegment(p1, q2, q1)) return true;

	// p2, q2 and p1 are colinear and p1 lies on segment p2q2 
	if (o3 == 0 && onSegment(p2, p1, q2)) return true;

	// p2, q2 and q1 are colinear and q1 lies on segment p2q2 
	if (o4 == 0 && onSegment(p2, q1, q2)) return true;

	return false; // Doesn't fall in any of the above cases 
}

__device__ double2 lineLineIntersection(double2 A, double2 B, double2 C, double2 D)
{
	// Line AB represented as a1x + b1y = c1 
	double a1 = B.y - A.y;
	double b1 = A.x - B.x;
	double c1 = a1*(A.x) + b1*(A.y);

	// Line CD represented as a2x + b2y = c2 
	double a2 = D.y - C.y;
	double b2 = C.x - D.x;
	double c2 = a2*(C.x) + b2*(C.y);

	double determinant = a1*b2 - a2*b1;


	double x = (b2*c1 - b1*c2) / determinant;
	double y = (a1*c2 - a2*c1) / determinant;

	return make_double2(x, y);

}

__global__ void NNcrustKernel(int no_of_points, int voronoi_edge_index, double2 voronoiedge_endpoint_a, double2 voronoiedge_endpoint_b)
{
	int point_of_interest_idx = threadIdx.x;
	
	double2 point_of_interest = gpu_voronoi_thresholdpointsforeachedge[voronoi_edge_index][point_of_interest_idx];

	int neighbors_index[10];
	int no_of_neighbors = 0;

	for (int i = 0; i < 2 * countof_gpu_delaunay_edgesforeachvoronoi[voronoi_edge_index]; i++)
	{
		if (gpu_delaunay_edgesindexforeachvoronoi[voronoi_edge_index][i] == point_of_interest_idx)
		{
			neighbors_index[no_of_neighbors] = gpu_delaunay_edgesindexforeachvoronoi[voronoi_edge_index][i + 1 + (i % 2)*-2];
			no_of_neighbors++;
		}

	}

	int nearest_point_index = 0;

	double distance_between_poi_currentp;
	double distace_between_poi_nearest_point;
	double2 nearest_point;
	double2 current_point;

	distace_between_poi_nearest_point = double2_distance(point_of_interest, gpu_delaunay_edgesforeachvoronoi[voronoi_edge_index][nearest_point_index]);

	for (int i = 1; i < no_of_neighbors; i++)
	{
		current_point = gpu_delaunay_edgesforeachvoronoi[voronoi_edge_index][i];

		distance_between_poi_currentp = double2_distance(point_of_interest, current_point);

		if (distance_between_poi_currentp < distace_between_poi_nearest_point)
		{
			nearest_point_index = i;

			nearest_point = gpu_delaunay_edgesforeachvoronoi[voronoi_edge_index][nearest_point_index];

			distace_between_poi_nearest_point = double2_distance(point_of_interest, nearest_point);
		}


	}


	int halfneighbor_point_index = -1;
	double distance_between_poi_halfneighbor = 10e9;
	double2 halfneighbor_point = make_double2(0,0);
	bool half_exist = false;

	for (int i = 0; i < no_of_neighbors; i++)
	{
		current_point = gpu_delaunay_edgesforeachvoronoi[voronoi_edge_index][i];

		distance_between_poi_currentp = double2_distance(point_of_interest, gpu_delaunay_edgesforeachvoronoi[voronoi_edge_index][i]);

		if (i != nearest_point_index)
		{
			if (distance_between_poi_currentp < distance_between_poi_halfneighbor && angle_in_triangle(nearest_point, point_of_interest, current_point) > 1.57)
			{
				half_exist = true;

				halfneighbor_point_index = i;
				
				halfneighbor_point = gpu_delaunay_edgesforeachvoronoi[voronoi_edge_index][i];

				distance_between_poi_halfneighbor = double2_distance(point_of_interest, halfneighbor_point);
			}
		}
	}

	gpu_nncrust_edgesforeach_voronoithresholdpoint[voronoi_edge_index][2 * point_of_interest_idx] = nearest_point;
	gpu_nncrust_edgesforeach_voronoithresholdpoint[voronoi_edge_index][2 * point_of_interest_idx + 1] = halfneighbor_point;

	if (doIntersect(point_of_interest, nearest_point, voronoiedge_endpoint_a, voronoiedge_endpoint_b))
	{
		gpu_nncrust_intersectionpoints_foreachvoronoi[voronoi_edge_index][2 * point_of_interest_idx] = lineLineIntersection(point_of_interest, nearest_point, voronoiedge_endpoint_a, voronoiedge_endpoint_b);
	}
	else
	{
		gpu_nncrust_intersectionpoints_foreachvoronoi[voronoi_edge_index][2 * point_of_interest_idx] = make_double2(0, 0);
	}
	if (half_exist && doIntersect(point_of_interest, halfneighbor_point, voronoiedge_endpoint_a, voronoiedge_endpoint_b))
	{
		gpu_nncrust_intersectionpoints_foreachvoronoi[voronoi_edge_index][2 * point_of_interest_idx] = lineLineIntersection(point_of_interest, halfneighbor_point, voronoiedge_endpoint_a, voronoiedge_endpoint_b);
	}
	else
	{
		gpu_nncrust_intersectionpoints_foreachvoronoi[voronoi_edge_index][2 * point_of_interest_idx] = make_double2(0, 0);
	}
}

__global__ void print_NNcurst()
{
	int lane_idx = threadIdx.x;
	int n = countof_gpu_voronoi_thresholdpointsforeachedge[lane_idx];
	for (int i = 0; i < n; i++)
	{
		printf("nn curst %d : %f - (%f-- %f) \n", lane_idx, gpu_voronoi_thresholdpointsforeachedge[lane_idx][i],gpu_nncrust_edgesforeach_voronoithresholdpoint[lane_idx][i * 2], gpu_nncrust_edgesforeach_voronoithresholdpoint[lane_idx][i * 2 + 1]);
	}
}

__global__ void finalize(int no_of_points, int voronoi_edge_index, int* d_no_of_intersections, double2* d_intersections, double2* d_delaunayPoints)
{
	double2 intersectedpoints[30];
	int no_of_intersectionpoints = 0;
	printf("no of points : %d-%d\n", voronoi_edge_index, no_of_points);
	
	for (int i = 0; i < 2*no_of_points; i++)
	{
		double2 p = gpu_nncrust_intersectionpoints_foreachvoronoi[voronoi_edge_index][i];
		
		bool repeated_element = false;

		if (!almost_equal(p, make_double2(0, 0), 2))
		{
			for (int j = 0; j < no_of_intersectionpoints; j++)
			{
				if (almost_equal(p, intersectedpoints[j], 2))
				{
					repeated_element = true;
				}
			}
			if (!repeated_element)
			{
				intersectedpoints[no_of_intersectionpoints] = p;
				no_of_intersectionpoints++;
			}
		}
		
	}
	d_no_of_intersections[voronoi_edge_index] = no_of_intersectionpoints;
	
	double max_dist = 0;
	double2 ans = make_double2(0,0);
	int max_index = 0;

	for (int i = 0; i < no_of_intersectionpoints; i++)
	{
		printf("intersections: (%lf, %lf) \n", intersectedpoints[i].x, intersectedpoints[i].y);
		double temp_dist = double2_distance(d_delaunayPoints[voronoi_edge_index], intersectedpoints[i]);
		if (temp_dist > max_dist)
		{
			max_dist = temp_dist;
			max_index = i;
		}
	}
	ans = intersectedpoints[max_index];
	d_intersections[voronoi_edge_index] = ans;
}

__global__ void delaunayKernel(Line_Segment* d_lines, double2* d_delaunayPoints, int* d_no_of_intersections, double2* d_intersections)
{

	using TriangleType = Delaunay_Triangle;
	using EdgeType = Delaunay_Edge;
	using VertexType = Delaunay_Vector2;

	const int no_points = 50;
	int lane_idx = threadIdx.x;

	int no_of_points = countof_gpu_voronoi_thresholdpointsforeachedge[lane_idx];

	TriangleType _triangles[(no_points + 2) * 2];
	EdgeType _edges[(no_points)* 3];
	VertexType _vertices[no_points + 3];

	for (int i = 0; i < no_of_points; i++)
	{
		//_vertices[i] = d[i];
		_vertices[i].x = gpu_voronoi_thresholdpointsforeachedge[lane_idx][i].x;
		_vertices[i].y = gpu_voronoi_thresholdpointsforeachedge[lane_idx][i].y;
	}

	// Determinate the super triangle
	double minX = _vertices[0].x;
	double minY = _vertices[0].y;
	double maxX = minX;
	double maxY = minY;

	for (int i = 0; i < no_of_points; i++)
	{
		VertexType temp = _vertices[i];
		if (_vertices[i].x < minX) minX = _vertices[i].x;
		if (_vertices[i].y < minY) minY = _vertices[i].y;
		if (_vertices[i].x > maxX) maxX = _vertices[i].x;
		if (_vertices[i].y > maxY) maxY = _vertices[i].y;
	}

	const double dx = maxX - minX;
	const double dy = maxY - minY;
	const double deltaMax = fmax(dx, dy);
	const double midx = my_half(minX + maxX);
	const double midy = my_half(minY + maxY);

	const VertexType p1(midx - 2000 * deltaMax, midy - deltaMax);
	const VertexType p2(midx, midy + 2000 * deltaMax);
	const VertexType p3(midx + 2000 * deltaMax, midy - deltaMax);

	// Create a list of triangles, and add the supertriangle in it
	_triangles[0] = TriangleType(p1, p2, p3);
	_triangles[0].isActive = true;
	int debug_count = 0;
	for (int nop = 0; nop < no_of_points; nop++)
	{
		VertexType p = _vertices[nop];
		EdgeType polygon[(no_points + 2) * 4];
		int polygon_count = 0;

		for (int i = 0; i < sizeof(_triangles) / sizeof(_triangles[0]); i++)
		{
			TriangleType t = _triangles[i];

			if (t.isActive)
			{

				if (t.circumCircleContains(p))
				{
					_triangles[i].isBad = true;
					polygon[polygon_count] = Delaunay_Edge{ *t.a, *t.b };
					polygon[polygon_count].isActive = true;
					polygon_count++;
					polygon[polygon_count] = Delaunay_Edge{ *t.b, *t.c };
					polygon[polygon_count].isActive = true;
					polygon_count++;
					polygon[polygon_count] = Delaunay_Edge{ *t.c, *t.a };
					polygon[polygon_count].isActive = true;
					polygon_count++;


				}
			}

		}

		for (int i = 0; i < sizeof(_triangles) / sizeof(_triangles[0]); i++)
		{
			// remove triangle if t.isbad = true
			TriangleType t = _triangles[i];
			if (t.isActive)
			{
				if (t.isBad)
				{
					_triangles[i].isActive = false;

				}
			}
		}

		for (int i = 0; i < (sizeof(polygon) / sizeof(polygon[0])); i++)
		{
			for (int j = i + 1; j < (sizeof(polygon) / sizeof(polygon[0])); j++)
			{
				if (polygon[i].isActive && polygon[j].isActive)
				{
					if (almost_equal(polygon[i], polygon[j], 2))
					{
						polygon[i].isBad = true;
						polygon[j].isBad = true;


					}
				}

			}
		}

		for (int i = 0; i < sizeof(polygon) / sizeof(polygon[0]); i++)
		{
			// remove edge if e.isbad = true
			EdgeType e = polygon[i];
			if (e.isActive)
			{
				if (e.isBad)
				{
					polygon[i].isActive = false;
				}
			}

		}

		for (int i = 0; i < sizeof(polygon) / sizeof(polygon[0]); i++)
		{
			EdgeType e = polygon[i];
			if (e.isActive)
			{

				bool iterate_bool = true;
				for (int i = 0; i < sizeof(_triangles) / sizeof(_triangles[0]); i++)
				{

					TriangleType t = _triangles[i];
					if (iterate_bool && !(t.isActive))
					{
						_triangles[i] = TriangleType(*e.v, *e.w, _vertices[nop]);
						_triangles[i].isActive = true;
						iterate_bool = false;
						debug_count++;

					}
				}
			}
		}

	}

	for (int i = 0; i < sizeof(_triangles) / sizeof(_triangles[0]); i++)
	{
		TriangleType t = _triangles[i];
		if (t.isActive)
		{

			if (t.containsVertex(p1, 2) || t.containsVertex(p2, 2) || t.containsVertex(p3, 2))
			{

				_triangles[i].isActive = false;

			}

		}
		// remove triangle if t.containsVertex(p1) || t.containsVertex(p2) || t.containsVertex(p3);
	}


	int edge_count = 0;
	int triangle_count = 0;
	int diff = 0;
	int diff2 = 0;

	for (int i = 0; i < sizeof(_triangles) / sizeof(_triangles[0]); i++)
	{
		TriangleType t = _triangles[i];

		if (t.isActive)
		{
			_edges[edge_count] = Delaunay_Edge{ *t.a, *t.b };
			_edges[edge_count].isActive = true;
			
			gpu_delaunay_edgesforeachvoronoi[lane_idx][edge_count*2] = make_double2(t.a->x, t.a->y);
			gpu_delaunay_edgesforeachvoronoi[lane_idx][edge_count*2 + 1] = make_double2(t.b->x, t.b->y);

			diff = t.a - _vertices;
			diff2 = t.b - _vertices;

			gpu_delaunay_edgesindexforeachvoronoi[lane_idx][edge_count * 2] = diff;
			gpu_delaunay_edgesindexforeachvoronoi[lane_idx][edge_count * 2 + 1] = diff2;
			edge_count++;

			_edges[edge_count] = Delaunay_Edge{ *t.b, *t.c };
			_edges[edge_count].isActive = true;
			gpu_delaunay_edgesforeachvoronoi[lane_idx][edge_count*2] = make_double2(t.b->x, t.b->y);
			gpu_delaunay_edgesforeachvoronoi[lane_idx][edge_count*2 + 1] = make_double2(t.c->x, t.c->y);

			diff = t.b - _vertices;
			diff2 = t.c - _vertices;

			gpu_delaunay_edgesindexforeachvoronoi[lane_idx][edge_count * 2] = diff;
			gpu_delaunay_edgesindexforeachvoronoi[lane_idx][edge_count * 2 + 1] = diff2;
			edge_count++;


			_edges[edge_count] = Delaunay_Edge{ *t.c, *t.a };
			_edges[edge_count].isActive = true;
			gpu_delaunay_edgesforeachvoronoi[lane_idx][edge_count*2] = make_double2(t.c->x, t.c->y);
			gpu_delaunay_edgesforeachvoronoi[lane_idx][edge_count*2 + 1] = make_double2(t.a->x, t.a->y);

			diff = t.c - _vertices;
			diff2 = t.a - _vertices;

			gpu_delaunay_edgesindexforeachvoronoi[lane_idx][edge_count * 2] = diff;
			gpu_delaunay_edgesindexforeachvoronoi[lane_idx][edge_count * 2 + 1] = diff2;

			edge_count++;


			triangle_count++;

		}

	}
	countof_gpu_delaunay_edgesforeachvoronoi[lane_idx] = edge_count;

	NNcrustKernel << < 1, no_of_points >> > (no_of_points, lane_idx, d_lines[lane_idx].P1, d_lines[lane_idx].P2);
	finalize << < 1, 1 >> >(no_of_points, lane_idx, d_no_of_intersections, d_intersections, d_delaunayPoints);

	int i = threadIdx.x;
	Delaunay_Vector2 z1 = Delaunay_Vector2(1.12, 2.34);

	printf("edge count: %d", edge_count);
	//printf("debug count:%d", debug_count);
	printf("triangle count: %d", triangle_count);
	//printf(almost_equal(2.11111111111112, 2.11111111111111,10000) ? "true" : "false");

}

__global__ void print_delaunay()
{
	int lane_idx = threadIdx.x;
	int n = countof_gpu_delaunay_edgesforeachvoronoi[lane_idx];
	for (int i = 0; i < n; i++)
	{
		printf("%d : %f %f -- %f %f\n", lane_idx, gpu_delaunay_edgesforeachvoronoi[lane_idx][i * 2].x, gpu_delaunay_edgesforeachvoronoi[lane_idx][i * 2].y, gpu_delaunay_edgesforeachvoronoi[lane_idx][i * 2 + 1].x, gpu_delaunay_edgesforeachvoronoi[lane_idx][i * 2 + 1].y);
	}
}

__global__ void print_delaunayindex()
{
	int lane_idx = threadIdx.x;
	int n = countof_gpu_delaunay_edgesforeachvoronoi[lane_idx];
	for (int i = 0; i < n; i++)
	{
		printf("%d : %d -- %d\n", lane_idx, gpu_delaunay_edgesindexforeachvoronoi[lane_idx][i * 2], gpu_delaunay_edgesindexforeachvoronoi[lane_idx][i * 2 + 1]);
	}
}

