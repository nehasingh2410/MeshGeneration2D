#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/random.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <helper_cuda.h>
#include <list> 
#include <vector> 
#include <sstream>
#include <fstream>
#include <string> 
#include <stdio.h>
#include <time.h>
#include <iostream> 
#include <fstream> 
#include <math.h>
#include "global_datatype.h"

#define FULL_MASK 0xffffffff
#define NUM_THREADS_PER_BLOCK 128
#define STACK_SIZE 400
#define MAX_POINTS_SIZE 600 
#define MAX_VORONOI_EDGES 1000
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

extern __device__ double2 gpu_voronoi_thresholdpointsforeachedge[MAX_VORONOI_EDGES][MAX_POINTS_SIZE];
extern __device__ int countof_gpu_voronoi_thresholdpointsforeachedge[MAX_VORONOI_EDGES];

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

class Points
{
public:
	double *X;
	double *Y;
	int num_points;
	__host__ __device__ Points() : X(NULL), Y(NULL), num_points(0) {}

	__host__ __device__ Points(double *x, double *y) : X(x), Y(y) {}

	__host__ __device__ __forceinline__ double2 getPoint(int idx) const
	{
		return make_double2(X[idx], Y[idx]);
	}
	__host__ __device__ __forceinline__ double2 getLastPoint() const
	{
		return make_double2(X[num_points - 1], Y[num_points - 1]);
	}
	__host__ __device__ __forceinline__ void printPoint(int idx) const
	{
		printf("%f %f\n", X[idx], Y[idx]);
	}
	__host__ __device__ void copyPointsToHost(){
		double *newX = new double[MAX_POINTS_SIZE];
		double *newY = new double[MAX_POINTS_SIZE];
		checkCudaErrors(cudaMemcpy(newX, X, MAX_POINTS_SIZE*sizeof(double), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(newY, Y, MAX_POINTS_SIZE*sizeof(double), cudaMemcpyDeviceToHost));
		X = newX;
		Y = newY;
	}
	__host__ __device__ __forceinline__ void setPoint(int idx, const double2 &p)
	{
		X[idx] = p.x;
		Y[idx] = p.y;
	}
	__host__ __device__ void addPoint(double2 p){
		//For findInsidePoints we will use a Points class for each edge around which we
		//have to find the inside points.
		//Maximum possible result for findInsidePoints for an edge = MAX_POINTS_SIZE
		if (num_points == MAX_POINTS_SIZE){
			printf("Overflow\n\n");
			return;
		}
		X[num_points] = p.x;
		Y[num_points] = p.y;
		num_points += 1;
	}
	__host__ __device__ int getNumberOfPoints(){
		return num_points;
	}
	__host__ __device__ __forceinline__ void set(double *x, double *y)
	{
		X = x;
		Y = y;
	}
};

class Bounding_Box{
	double xMin, xMax, yMin, yMax;
public:
	__host__ __device__ Bounding_Box(){
		xMin = -700;
		yMin = -700;
		xMax = 700;
		yMax = 700;
	}
	__host__ __device__ double2 computeCenter(){
		double2 center;
		center.x = 0.5f * (xMin + xMax);
		center.y = 0.5f * (yMin + yMax);
		return center;
	}
	__host__ __device__ __forceinline__ double getxMax() const {
		return xMax;
	}
	__host__ __device__ __forceinline__ double getyMax() const {
		return yMax;
	}
	__host__ __device__ __forceinline__ double getyMin() const {
		return yMin;
	}
	__host__ __device__ __forceinline__ double getxMin() const {
		return xMin;
	}
	__host__ __device__ __forceinline__ void printBox() const {
		printf("%f %f %f %f --", xMin, yMax, xMax, yMax);
		printf("%f %f %f %f\n", xMax, yMin, xMin, yMin);
	}
	__host__ __device__ bool contains(const double2 &p) const {
		return (p.x >= xMin && p.y >= yMin && p.x < xMax && p.y < yMax);
	}
	__host__ __device__ void set(double x, double y, double X, double Y){
		xMin = x;
		yMin = y;
		xMax = X;
		yMax = Y;
	}
	__host__ __device__ int isInside(double x, double y){
		int res = (x <= xMax) && (x >= xMin) && (y <= yMax) && (y >= yMin);
		return res;
	}
};

class Quadtree_Node{
	//node index;
	int idx;
	Bounding_Box bb;
	//startIdx of points in the bb for the global data array
	int startIdx, endIdx;
	Quadtree_Node *NE, *NW, *SW, *SE;
public:
	__host__ __device__ Quadtree_Node() : idx(-1), startIdx(-1), endIdx(-1), NE(NULL), NW(NULL), SW(NULL), SE(NULL){

	}
	__host__ __device__ bool isNull(){
		return (idx == -1);
	}
	__host__ __device__ void setIdx(int idx){
		this->idx = idx;
	}
	__host__ __device__ int getIdx(){
		return idx;
	}
	__host__ __device__ void setBoundingBox(double x, double y, double X, double Y){
		bb.set(x, y, X, Y);
	}
	__host__ __device__ __forceinline__ Bounding_Box& getBoundingBox(){
		return bb;
	}
	__host__ __device__ void setRange(int s, int e){
		startIdx = s;
		endIdx = e;
	}
	__host__ __device__ __forceinline__ Quadtree_Node* getSW(){
		return SW;
	}
	__host__ __device__ __forceinline__ Quadtree_Node* getSE(){
		return SE;
	}
	__host__ __device__ __forceinline__ Quadtree_Node* getNW(){
		return NW;
	}
	__host__ __device__ __forceinline__ Quadtree_Node* getNE(){
		return NE;
	}
	__host__ __device__ __forceinline__ void setSW(Quadtree_Node* ptr){
		SW = ptr;
	}
	__host__ __device__ __forceinline__ void setNW(Quadtree_Node* ptr){
		NW = ptr;
	}
	__host__ __device__ __forceinline__ void setSE(Quadtree_Node* ptr){
		SE = ptr;
	}
	__host__ __device__ __forceinline__ void setNE(Quadtree_Node* ptr){
		NE = ptr;
	}
	__host__ __device__ __forceinline__ int isLeaf(){
		return (NE == NULL);
	}
	__host__ __device__ __forceinline__ int getStartIdx(){
		return startIdx;
	}
	__host__ __device__ __forceinline__ int getEndIdx(){
		return endIdx;
	}
	__host__ __device__ __forceinline__ int numberOfPoints(){
		return endIdx - startIdx;
	}
};

class Parameters
{
public:
	const int min_points_per_node;
	//Introduced to minimise shifting of points
	//can have values only 0 and 1 based on slot
	//points[points_slot] is input slot
	//points[(points_slot+1)%2] is output slot
	int points_slot;
	__host__ __device__ Parameters(int mppn) : min_points_per_node(mppn), points_slot(0) {}
	//copy constructor for the evaluation of children of current node
	__host__ __device__ Parameters(Parameters prm, bool) :
		min_points_per_node(prm.min_points_per_node),
		points_slot((prm.points_slot + 1) % 2)
	{}
};

class Quadtree_Stack
{
private:
	Quadtree_Node* arr[STACK_SIZE];
	int top;
public:
	__host__ __device__ Quadtree_Stack(){
		top = -1;
	}
	__host__ __device__ Quadtree_Node* push(Quadtree_Node* n){
		//check stack is full or not
		if (isFull()){
			return NULL;
		}
		++top;
		arr[top] = n;
		return n;
	}

	__host__ __device__ Quadtree_Node* pop(){
		//to store and print which number
		//is deleted
		Quadtree_Node* temp;
		//check for empty
		if (isEmpty())
			return NULL;
		temp = arr[top];
		--top;
		return temp;

	}
	__host__ __device__ int isEmpty(){
		if (top == -1)
			return 1;
		else
			return 0;
	}

	__host__ __device__ int isFull(){
		if (top == (STACK_SIZE - 1))
			return 1;
		else
			return 0;
	}
};

class Line_Segment{
public:
	double M, C;
	double2 P1, P2;
	bool isBad;
	__host__ __device__ Line_Segment() : M(0.0), C(0.0), P1(make_double2(0.0, 0.0)), P2(make_double2(0.0, 0.0)), isBad(false){};
	__host__ __device__ Line_Segment(double2 p1, double2 p2){
		M = (p1.y - p2.y) / (p1.x - p2.x);
		C = p1.y - M*(p1.x);
		P1 = p1;
		P2 = p2;
		isBad = false;
	}
	__host__ __device__ Line_Segment(double m, double c){
		M = m;
		C = c;
	}
	__host__ __device__ double getPerpendicularDistance(double2 p){
		double res = M*p.x - p.y + C;
		res = res / sqrt(1.0 + M*M);
		if (res < 0)
			res = (-1)*res;
		return res;
	}
	__host__ __device__ Line_Segment getLeftThreshold(double d){
		return Line_Segment(M, (C - d*sqrt(1.0 + M*M)));
	}
	__host__ __device__ Line_Segment getRightThreshold(double d){
		return Line_Segment(M, (C + d*sqrt(1.0 + M*M)));
	}
	__host__ __device__ int intersectsWithBox(Bounding_Box box){
		double minX = box.getxMin();
		double minX_Y = M*minX + C;
		double minY = box.getyMin();
		double minY_X = (minY - C) / M;
		double maxX = box.getxMax();
		double maxX_Y = M*maxX + C;
		double maxY = box.getyMax();
		double maxY_X = (maxY - C) / M;
		int res = box.isInside(minX, minX_Y) || box.isInside(minY_X, minY) || box.isInside(maxX, maxX_Y) || box.isInside(maxY_X, maxY);
		return res;
	}
	__host__ __device__ int insidePerpendicularBounds(Bounding_Box box){
		double perpM = -(1.00)*(1.0 / M);
		double C1 = P1.y - perpM*P1.x;
		double C2 = P2.y - perpM*P2.x;
		double2 boundaries[4];
		boundaries[0] = make_double2(box.getxMin(), box.getyMax());
		boundaries[1] = make_double2(box.getxMax(), box.getyMax());
		boundaries[2] = make_double2(box.getxMin(), box.getyMin());
		boundaries[3] = make_double2(box.getxMax(), box.getyMin());
		for (int i = 0; i<4; i++){
			int sign1 = ((boundaries[i].y - perpM*boundaries[i].x - C1) >= 0);
			int sign2 = ((boundaries[i].y - perpM*boundaries[i].x - C2) >= 0);
			if (sign1 != sign2)
				return 1;
		}
		return 0;
	}
	__host__ __device__ double2 getP1(){
		return P1;
	}
	__host__ __device__ double2 getP2(){
		return P2;
	}
	__host__ __device__ bool getIsBad(){
		return isBad;
	}
};

__global__ void print_gpu_voronoi_thresholdpointsforeachedge()
{
	int line_idx = threadIdx.x;
	for (int i = 0; i < countof_gpu_voronoi_thresholdpointsforeachedge[line_idx]; i++)
	{
		double2 p = gpu_voronoi_thresholdpointsforeachedge[line_idx][i];
		printf("thread_count: %d countof_gpu_voronoi_thresholdpoints: %d %f %f\n", line_idx, countof_gpu_voronoi_thresholdpointsforeachedge[line_idx], p.x, p.y);
	}

}

__device__ int nodeInsideThreshold(Line_Segment line, Bounding_Box box, Quadtree_Node* root, double threshold){
	double2 LT, RT, LB, RB;
	LT = make_double2(box.getxMin(), box.getyMax());
	RT = make_double2(box.getxMax(), box.getyMax());
	LB = make_double2(box.getxMin(), box.getyMin());
	RB = make_double2(box.getxMax(), box.getyMin());
	bool boxInside = ((line.getPerpendicularDistance(LT) <= threshold)
		|| (line.getPerpendicularDistance(RT) <= threshold)
		|| (line.getPerpendicularDistance(LB) <= threshold) ||
		(line.getPerpendicularDistance(RB) <= threshold));
	bool boxIntersects = ((line.intersectsWithBox(box)) || (line.getLeftThreshold(threshold).intersectsWithBox(box)) || (line.getRightThreshold(threshold).intersectsWithBox(box)));
	// printf("boxInside=%d,boxIntersects=%d \n",boxInside,boxIntersects);
	return line.insidePerpendicularBounds(box) && (boxInside || boxIntersects);

}

__global__ void findOuterThresholdPoints(Quadtree_Node *root, Points *points, Line_Segment *lines, Points *inside_points, double threshold){
	// printf("Threshold is %lf\n",threshold);
	int line_idx = threadIdx.x;
	printf("%d \n", line_idx);
	Quadtree_Stack qst;
	qst.push(root);
	int count_points_per_edge = 0;
	while (!qst.isEmpty()){
		Quadtree_Node* X = qst.pop();
		Bounding_Box box = X->getBoundingBox();
		if (X->isLeaf()){
			if (X->numberOfPoints() != 0){
				int no_of_pointsinleaf = X->getEndIdx() - X->getStartIdx();
				printf("line_idx: %d no_of_pointsinleaf: %d \n", line_idx, no_of_pointsinleaf);
				int count_in_for_looop = 0;
				for (int i = X->getStartIdx(); i<X->getEndIdx(); i++){
					double2 p = points[0].getPoint(i);
					gpu_voronoi_thresholdpointsforeachedge[line_idx][count_points_per_edge] = p;
					count_points_per_edge++;
					// printf("%f %f\n", line_idx, p.x, p.y);
					inside_points[line_idx].addPoint(p);

				}
			}
		}
		else{


			//nodeInsideThreshold returns 1 if the bounding box is inside the threshold or if the the box intersects any of the threshold lines or the line itself
			if (nodeInsideThreshold(lines[line_idx], box, X->getNE(), threshold)){
				qst.push(X->getNE());
			}
			if (nodeInsideThreshold(lines[line_idx], box, X->getNW(), threshold)){
				qst.push(X->getNW());
			}
			if (nodeInsideThreshold(lines[line_idx], box, X->getSE(), threshold)){
				qst.push(X->getSE());
			}
			if (nodeInsideThreshold(lines[line_idx], box, X->getSW(), threshold)){
				qst.push(X->getSW());
			}
		}

	}
	countof_gpu_voronoi_thresholdpointsforeachedge[line_idx] = count_points_per_edge;
}

__global__ void buildQuadtree(Quadtree_Node *root, Points *points, Parameters prmtrs){
	const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warpSize;

	//shared memory
	extern __shared__ int smem[];

	//warp_id and lane_id
	const int warp_id = threadIdx.x / warpSize;
	const int lane_id = threadIdx.x % warpSize;

	// Addresses of shared Memory
	volatile int *s_num_pts[4];
	for (int i = 0; i < 4; ++i)
		s_num_pts[i] = (volatile int *)&smem[i*NUM_WARPS_PER_BLOCK];


	int NUM_POINTS = root->numberOfPoints();
	Bounding_Box &box = root->getBoundingBox();
	//stop recursion if num_points <= minimum number of points required for recursion 
	if (NUM_POINTS <= prmtrs.min_points_per_node){
		//If in current iteration the points are in slot 1
		//shift them to slot 0
		//we want the output in the slot 0
		if (prmtrs.points_slot == 1)
		{
			int it = root->getStartIdx(), end = root->getEndIdx();
			for (it += threadIdx.x; it < end; it += NUM_THREADS_PER_BLOCK){
				if (it < end)
					points[0].setPoint(it, points[1].getPoint(it));
			}
		}

		return;
	}

	//get Center of the bounding box
	double2 center;
	center = box.computeCenter();
	//accomadate the excess points
	int NUM_POINTS_PER_WARP = max(warpSize, (NUM_POINTS + NUM_WARPS_PER_BLOCK - 1) / NUM_WARPS_PER_BLOCK);

	int warp_begin = root->getStartIdx() + warp_id*NUM_POINTS_PER_WARP;
	int warp_end = min(warp_begin + NUM_POINTS_PER_WARP, root->getEndIdx());

	//reset counts of warps
	if (lane_id == 0)
	{
		s_num_pts[0][warp_id] = 0;
		s_num_pts[1][warp_id] = 0;
		s_num_pts[2][warp_id] = 0;
		s_num_pts[3][warp_id] = 0;

	}

	//input points
	const Points &input = points[prmtrs.points_slot];

	//__any_sync(unsigned mask, predicate):
	//Evaluate predicate for all non-exited threads in mask and return non-zero if and only if predicate evaluates to non-zero for any of them.
	//count points in each warp that belong to which child
	for (int itr = warp_begin + lane_id; __any(itr < warp_end); itr += warpSize){
		bool is_active = itr < warp_end;
		//get the coordinates of the point
		double2 curP;
		if (is_active)
			curP = input.getPoint(itr);
		else
			curP = make_double2(0.0f, 0.0f);

		//consider standard anticlockwise quadrants for numbering 0 to 3

		//__ballot_sync(unsigned mask, predicate):
		//Evaluate predicate for all non-exited threads in mask and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active.
		//__popc
		//Count the number of bits that are set to 1 in a 32 bit integer.
		//top-right Quadrant (Quadrant - I)
		bool pred = is_active && curP.x >= center.x && curP.y >= center.y;
		int curMask = __ballot(pred);
		int cnt = __popc(curMask);
		if (cnt > 0 && lane_id == 0)
			s_num_pts[0][warp_id] += cnt;

		//top-left Quadrant (Quadrant - II)
		pred = is_active && curP.x < center.x && curP.y >= center.y;
		curMask = __ballot(pred);
		cnt = __popc(curMask);
		if (cnt > 0 && lane_id == 0)
			s_num_pts[1][warp_id] += cnt;

		//bottom-left Quadrant (Quadrant - III)
		pred = is_active && curP.x < center.x && curP.y < center.y;
		curMask = __ballot(pred);
		cnt = __popc(curMask);
		if (cnt > 0 && lane_id == 0)
			s_num_pts[2][warp_id] += cnt;

		//bottom-right Quadrant (Quadrant - IV)
		pred = is_active && curP.x >= center.x && curP.y < center.y;
		curMask = __ballot(pred);
		cnt = __popc(curMask);
		if (cnt > 0 && lane_id == 0)
			s_num_pts[3][warp_id] += cnt;
	}

	//sychronize warps
	//__syncthreads() acts as a barrier at which all threads in the block must wait before any is allowed to proceed
	__syncthreads();
	/*  	if(threadIdx.x == NUM_THREADS_PER_BLOCK - 1 && root->getIdx() == 1024){
	printf("Quadrant I : %d, %d \n", s_num_pts[0][0], s_num_pts[0][NUM_WARPS_PER_BLOCK-1]);
	for(int i = 0;i<NUM_WARPS_PER_BLOCK;i++){
	printf("%d ", s_num_pts[0][i]);
	}
	printf("\nQuadrant II : %d, %d \n", s_num_pts[1][0], s_num_pts[1][NUM_WARPS_PER_BLOCK-1]);
	for(int i = 0;i<NUM_WARPS_PER_BLOCK;i++){
	printf("%d ", s_num_pts[1][i]);
	}
	printf("\nQuadrant III : %d, %d \n", s_num_pts[2][0], s_num_pts[2][NUM_WARPS_PER_BLOCK-1]);
	for(int i = 0;i<NUM_WARPS_PER_BLOCK;i++){
	printf("%d ", s_num_pts[2][i]);
	}
	printf("\nQuadrant IV : %d, %d \n", s_num_pts[3][0], s_num_pts[3][NUM_WARPS_PER_BLOCK-1]);
	for(int i = 0;i<NUM_WARPS_PER_BLOCK;i++){
	printf("%d ", s_num_pts[3][i]);
	}
	printf("\n\n\n");
	}
	__syncthreads(); */



	// Scan the warps' results to know the "global" numbers.
	// First 4 warps scan the numbers of points per child (inclusive scan).
	// In the later code we have used warp id to select the quadrant and lane_id to select a warp.
	if (warp_id < 4)
	{
		int num_pts = lane_id < NUM_WARPS_PER_BLOCK ? s_num_pts[warp_id][lane_id] : 0;
#pragma unroll
		for (int offset = 1; offset < NUM_WARPS_PER_BLOCK; offset *= 2)
		{

			//T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);	
			int n = __shfl_up(num_pts, offset, NUM_WARPS_PER_BLOCK);

			if (lane_id >= offset)
				num_pts += n;
		}
		if (lane_id < NUM_WARPS_PER_BLOCK)
			s_num_pts[warp_id][lane_id] = num_pts;
	}
	//after this we will have the local offsets, i.e , if we have a warp with id X
	//then s_num_pts[0][x] will store the number of points having warp id <= x 
	//and belong to the 0th quadrant
	__syncthreads();
	// Compute global offsets.
	//here lane_id will index the warps
	if (warp_id == 0)
	{
		int sum = s_num_pts[0][NUM_WARPS_PER_BLOCK - 1];
		for (int row = 1; row < 4; ++row)
		{
			int tmp = s_num_pts[row][NUM_WARPS_PER_BLOCK - 1];
			if (lane_id < NUM_WARPS_PER_BLOCK)
				s_num_pts[row][lane_id] = s_num_pts[row][lane_id] + sum;
			sum += tmp;
		}
	}
	__syncthreads();
	//after this we have the global offsets, i.e, if warp id is X and quadrant q
	//then s_num_pts[q][x] will store the number of points having warp id <= x 
	//and belong to the quadrant <= q

	//make the Scan independent of the quadtree node you are currently in.
	// for this we just have to add the number of points that come before processing of the current node.
	if (threadIdx.x < 4 * NUM_WARPS_PER_BLOCK){
		int val = (threadIdx.x == 0) ? 0 : smem[threadIdx.x - 1];
		smem[threadIdx.x] = val + root->getStartIdx();
	}
	__syncthreads();
	//move points to the next slot
	Points &output = points[(prmtrs.points_slot + 1) % 2];

	//Mask for threads in a warp that are less than the current lane_id
	int lane_mask_lt = (1 << lane_id) - 1;
	// Move Points to the appropriate slot 
	// Quadtree sort implementation

	for (int itr = warp_begin + lane_id; __any(itr < warp_end); itr += warpSize){
		bool is_active = itr < warp_end;

		double2 curP;
		if (is_active){
			curP = input.getPoint(itr);
		}
		else{
			curP = make_double2(0.0f, 0.0f);
		}

		//counting QUADRANT I points
		bool pred = is_active && curP.x >= center.x && curP.y >= center.y;
		int curMask = __ballot(pred);
		int cnt = __popc(curMask & lane_mask_lt);
		int dest = s_num_pts[0][warp_id] + cnt;
		if (pred)
			output.setPoint(dest, curP);
		if (lane_id == 0)
			s_num_pts[0][warp_id] += __popc(curMask);

		//countin QUADRANT II points
		pred = is_active && curP.x < center.x && curP.y >= center.y;
		curMask = __ballot(pred);
		cnt = __popc(curMask & lane_mask_lt);
		dest = s_num_pts[1][warp_id] + cnt;
		if (pred)
			output.setPoint(dest, curP);
		if (lane_id == 0)
			s_num_pts[1][warp_id] += __popc(curMask);

		//countin QUADRANT III points
		pred = is_active && curP.x < center.x && curP.y < center.y;
		curMask = __ballot(pred);
		cnt = __popc(curMask & lane_mask_lt);
		dest = s_num_pts[2][warp_id] + cnt;
		if (pred)
			output.setPoint(dest, curP);
		if (lane_id == 0)
			s_num_pts[2][warp_id] += __popc(curMask);

		//countin QUADRANT IV points
		pred = is_active && curP.x >= center.x && curP.y < center.y;
		curMask = __ballot(pred);
		cnt = __popc(curMask & lane_mask_lt);
		dest = s_num_pts[3][warp_id] + cnt;
		if (pred)
			output.setPoint(dest, curP);
		if (lane_id == 0)
			s_num_pts[3][warp_id] += __popc(curMask);

	}
	__syncthreads();
	//last thread will launch new block 
	if (threadIdx.x == NUM_THREADS_PER_BLOCK - 1){
		//create children for next level
		// set index, bb, startIdx, endIdx and NE, NW, SE, SW children.
		//Index is used just for sake of future extension if some changes are required then
		//children nodes
		// std::cout << "( " << box.getxMin() << "," << box.getyMin() << ") , (" << box.getxMax() << "," << box.getyMax() << ") " << std::endl;	
		//print top left and top right points
		Quadtree_Node* NEC = (Quadtree_Node*)malloc(sizeof(Quadtree_Node));
		Quadtree_Node* NWC = (Quadtree_Node*)malloc(sizeof(Quadtree_Node));
		Quadtree_Node* SWC = (Quadtree_Node*)malloc(sizeof(Quadtree_Node));
		Quadtree_Node* SEC = (Quadtree_Node*)malloc(sizeof(Quadtree_Node));
		//set Bounding Box
		//printf("Center: %f %f\n", center.x, center.y);
		NEC->setBoundingBox(center.x, center.y, box.getxMax(), box.getyMax());
		NWC->setBoundingBox(box.getxMin(), center.y, center.x, box.getyMax());
		SWC->setBoundingBox(box.getxMin(), box.getyMin(), center.x, center.y);
		SEC->setBoundingBox(center.x, box.getyMin(), box.getxMax(), center.y);

		//set the start and end ranges
		//print the range of indices for children
		/* 		printf("(%d, %d), ", root->getStartIdx(), s_num_pts[0][warp_id]);
		printf("(%d, %d), ", s_num_pts[0][warp_id], s_num_pts[1][warp_id]);
		printf("(%d, %d), ", s_num_pts[1][warp_id], s_num_pts[2][warp_id]);
		printf("(%d, %d)\n", s_num_pts[2][warp_id], s_num_pts[3][warp_id]);
		*/

		NEC->setRange(root->getStartIdx(), s_num_pts[0][warp_id]);
		NWC->setRange(s_num_pts[0][warp_id], s_num_pts[1][warp_id]);
		SWC->setRange(s_num_pts[1][warp_id], s_num_pts[2][warp_id]);
		SEC->setRange(s_num_pts[2][warp_id], s_num_pts[3][warp_id]);

		//set the root children 
		root->setNE(NEC);
		root->setNW(NWC);
		root->setSW(SWC);
		root->setSE(SEC);

		//launch children
		buildQuadtree << <1, NUM_THREADS_PER_BLOCK, 4 * NUM_WARPS_PER_BLOCK*sizeof(int) >> >(NEC, points, Parameters(prmtrs, true));
		buildQuadtree << <1, NUM_THREADS_PER_BLOCK, 4 * NUM_WARPS_PER_BLOCK*sizeof(int) >> >(NWC, points, Parameters(prmtrs, true));
		buildQuadtree << <1, NUM_THREADS_PER_BLOCK, 4 * NUM_WARPS_PER_BLOCK*sizeof(int) >> >(SWC, points, Parameters(prmtrs, true));
		buildQuadtree << <1, NUM_THREADS_PER_BLOCK, 4 * NUM_WARPS_PER_BLOCK*sizeof(int) >> >(SEC, points, Parameters(prmtrs, true));
	}
}

__global__ void printQuadtree(Quadtree_Node *root){
	Bounding_Box box = root->getBoundingBox();
	box.printBox();
	if (root == NULL) printf("it is null");
	if (root->getNE() != NULL){
		printQuadtree << <1, 1 >> >(root->getNE());
		printQuadtree << <1, 1 >> >(root->getNW());
		printQuadtree << <1, 1 >> >(root->getSE());
		printQuadtree << <1, 1 >> >(root->getSW());

	}

}

__global__ void printPoints(Points *inside_points, int num_of_lines){
	for (int i = 0; i<num_of_lines; i++){
		int num_points = inside_points[i].getNumberOfPoints();
		for (int j = 0; j<num_points; j++){
			double2 p = inside_points[i].getPoint(j);
			printf("%d %f %f\n", i, p.x, p.y);
		}
	}
}

Points* initializeInsidePoints(int num_of_edges){
	Points *h_points = new Points[num_of_edges]; //= (Points*)malloc(num_of_edges*sizeof(Points));
	vector< thrust::device_vector<double> > X(num_of_edges);
	vector< thrust::device_vector<double> > Y(num_of_edges);


	for (int i = 0; i<num_of_edges; i++){
		X[i].resize(MAX_POINTS_SIZE);
		Y[i].resize(MAX_POINTS_SIZE);
		h_points[i].set(thrust::raw_pointer_cast(&X[i][0]), thrust::raw_pointer_cast(&Y[i][0]));
	}

	//device_points
	Points *d_points;
	checkCudaErrors(cudaMalloc((void**)&d_points, num_of_edges*sizeof(Points)));
	checkCudaErrors(cudaMemcpy(d_points, h_points, num_of_edges*sizeof(Points), cudaMemcpyHostToDevice));
	return d_points;
}