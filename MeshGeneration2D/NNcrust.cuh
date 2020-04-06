#pragma once
#include "delaunay.cuh"

#define MAX_POINTS_SIZE 600
#define MAX_VORONOI_EDGES 1000

extern __device__ double2 gpu_voronoi_thresholdpointsforeachedge[MAX_VORONOI_EDGES][MAX_POINTS_SIZE];
extern __device__ int countof_gpu_voronoi_thresholdpointsforeachedge[MAX_VORONOI_EDGES];

extern __device__ double2 gpu_delaunay_edgesforeachvoronoi[MAX_VORONOI_EDGES][6 * MAX_POINTS_SIZE - 15];
extern __device__ int gpu_delaunay_edgesindexforeachvoronoi[MAX_VORONOI_EDGES][6 * MAX_POINTS_SIZE - 15];
extern __device__ int countof_gpu_delaunay_edgesforeachvoronoi[MAX_VORONOI_EDGES];

extern __device__ double2 gpu_nncrust_edgesforeachvoronoi[MAX_VORONOI_EDGES][6 * MAX_POINTS_SIZE - 15];
extern __device__ int countof_gpu_nncrust_edgesforeachvoronoi[MAX_VORONOI_EDGES];

class NNcrust_edgematrix
{
	int lane_idx;
	int** adj_mat;
	int num_points;
	double obtuse_angle;

public:
	__host__ __device__ NNcrust_edgematrix(int lane_idx)
	{
		lane_idx = lane_idx;
		num_points = countof_gpu_voronoi_thresholdpointsforeachedge[lane_idx];
		adj_mat = (int**)malloc(num_points*sizeof(int*));

		for (int i = 0; i<num_points; i++){
			adj_mat[i] = (int*)malloc(num_points*sizeof(int));
		}

		for (int i = 0; i<num_points; i++){
			for (int j = 0; j<num_points; j++){
				adj_mat[i][j] = 0;
			}
		}

		for (int j = 0; j < 2 * countof_gpu_delaunay_edgesforeachvoronoi[lane_idx]; j++)
		{
			if (j % 2 == 0)
			{
				adj_mat[gpu_delaunay_edgesindexforeachvoronoi[lane_idx][j]][gpu_delaunay_edgesindexforeachvoronoi[lane_idx][j + 1]] = 1;
				adj_mat[gpu_delaunay_edgesindexforeachvoronoi[lane_idx][j+1]][gpu_delaunay_edgesindexforeachvoronoi[lane_idx][j]] = 1;
			}
		}
	}

	__host__ __device__ double NNcrust_distance(int lane_idx, int i, int j)
	{
		double2 ipt = gpu_voronoi_thresholdpointsforeachedge[lane_idx][i];
		double2 jpt = gpu_voronoi_thresholdpointsforeachedge[lane_idx][j];
		double dist = (ipt.x - jpt.x)*(ipt.x - jpt.x) + (ipt.y - jpt.y)*(ipt.y - jpt.y);
		return dist;
	}

	__host__ __device__ int getClosestPoint(int idx)
	{
		int closest = -1;
		for (int j = 0; j<num_points; j++){
			if (j != idx && closest == -1)
				closest = j;
			else if (j != idx && NNcrust_distance(idx, closest) > NNcrust_distance(idx, j)){
				closest = j;
			}
		}
		return closest;
	}


};




__host__ __device__ int NNcrust_getClosestPoint(int lane_idx, int idx)
{
	int closest = -1;

	double2 point_of_interest = gpu_voronoi_thresholdpointsforeachedge[lane_idx][idx];

	for (int j = 0; j < 2*countof_gpu_delaunay_edgesforeachvoronoi[lane_idx]; j++)
	{
		if (almost_equal(point_of_interest, gpu_delaunay_edgesforeachvoronoi[lane_idx][j], 2) && closest == -1)
		{
			if (j % 2 == 0)
			{
				closest = j + 1;
			}
			else
			{
				closest = j - 1;
			}
		}
		
		if (j != idx && closest == -1)
			closest = j;
		else if (j != idx && NNcrust_distance(lane_idx, idx, closest) > NNcrust_distance(lane_idx, idx, j))
		{
			closest = j;
		}
	}
	return closest;
}

__host__ __device__ int NNcrust_getClosestObtusePoint(int lane_idx, int idx){
	int closest = NNcrust_getClosestPoint(lane_idx, idx);
	double2 P1 = gpu_voronoi_thresholdpointsforeachedge[lane_idx][idx];
	double2 P2 = gpu_voronoi_thresholdpointsforeachedge[lane_idx][closest];
	int obtuse_closest = -1;
	for (int i = 0; i<num_points; i++){
		if (i != idx && i != closest){
			double2 P3 = points->getPoint(i);
			double angle = atan2(P3.y - P1.y, P3.x - P1.x) - atan2(P2.y - P1.y, P2.x - P1.x);
			if (angle > obtuse_angle || angle < -1 * obtuse_angle){
				if (obtuse_closest == -1){
					obtuse_closest = i;
				}
				else if (distance(idx, obtuse_closest) > distance(idx, i)){
					obtuse_closest = i;
				}
			}

		}

	}
	return obtuse_closest;
}


__global__ void NNcrustKernel()
{
	int lane_idx = threadIdx.x;
	int no_of_points = countof_gpu_voronoi_thresholdpointsforeachedge[lane_idx];

}