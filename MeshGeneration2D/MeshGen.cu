
#include "QuadTree.cuh"
#include "delaunay.cuh"

//new header
#include "global_datatype.h"
#include "Topology.h"
#include "Geometry.h"
#include "MeshRefinement.h"
#include "CreateDelaunay.h"

__device__ double2 gpu_voronoi_thresholdpointsforeachedge[MAX_VORONOI_EDGES][MAX_POINTS_SIZE];
__device__ int countof_gpu_voronoi_thresholdpointsforeachedge[MAX_VORONOI_EDGES];
__device__ double2 gpu_nncrust_edgesforeach_voronoithresholdpoint[MAX_VORONOI_EDGES][MAX_POINTS_SIZE * 2];
__device__ double2 gpu_nncrust_intersectionpoints_foreachvoronoi[MAX_VORONOI_EDGES][MAX_POINTS_SIZE * 2];

__device__ double2 gpu_delaunay_edgesforeachvoronoi[MAX_VORONOI_EDGES][6*MAX_POINTS_SIZE - 15];
__device__ int gpu_delaunay_edgesindexforeachvoronoi[MAX_VORONOI_EDGES][6 * MAX_POINTS_SIZE - 15];
__device__ int countof_gpu_delaunay_edgesforeachvoronoi[MAX_VORONOI_EDGES];



int main()
{

	std::string inputFile = "2.5width_4patels.txt";
	// std::string outputFile = "InnerPoints(2.5width_4patels.txt).txt";
	// freopen(outputFile.c_str() , "w", stdout);
	const int max_depth = 10;
	const int min_points_per_node = 5; // Min points per node
	int num_points = -1;

	//Read Points from file and put it into x0(X points) and y0(Y Points)
	std::vector<Point_2> OriginalSample, RandomSample;
	clock_t start = clock();
	std::list<double> stlX, stlY;
	std::ifstream source(inputFile);
	if (source.is_open()){
		int i = 0;
		for (std::string line; std::getline(source, line); i += 1)   //read stream line by line
		{
			std::istringstream in(line);
			double x, y;
			in >> x >> y;
			Point_2 original(x, y);
			OriginalSample.push_back(original);
			stlX.push_back(x);
			stlY.push_back(y);
		}
	}
	else{
		printf("No");
		exit(1);
	}
	/*
	std::ifstream input("neha1.txt");
	int num_of_points = 0;
	std::string data;
	while (getline(input, data))
	{
	Point_2 original;
	std::istringstream stream(data);
	while (stream >> original)
	{
	OriginalSample.push_back(original);
	++num_of_points;
	}
	}
	*/
	clock_t end = clock();
	double run_time = ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout << "File Reading Time: " << run_time << std::endl;
	num_points = stlX.size();
	std::cout << "Number of Points: " << num_points << std::endl;
	
	//Set Cuda Device
	int device_count = 0, device = -1, warp_size = 0;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	std::cout << device_count << endl;
	for (int i = 0; i < device_count; ++i)
	{
		cudaDeviceProp properties;
		checkCudaErrors(cudaGetDeviceProperties(&properties, i));
		if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
		{
			device = i;
			warp_size = properties.warpSize;
			std::cout << "Running on GPU: " << i << " (" << properties.name << ")" << std::endl;
			std::cout << "Warp Size: " << warp_size << std::endl;
			std::cout << "Threads Per Block: " << properties.maxThreadsPerBlock << std::endl;
			break;
		}
		std::cout << "GPU " << i << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
	}
	if (device == -1)
	{
		//cdpQuadTree requires SM 3.5 or higher to use CUDA Dynamic Parallelism.  Exiting...
		exit(EXIT_SUCCESS);
	}
	cudaSetDevice(device);

	start = clock();
	cudaFree(0);
	cudaThreadSynchronize();
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	end = clock();
	run_time = ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout << "cudaFree Time: " << run_time << std::endl;

	start = clock();
	thrust::device_vector<double> x0(stlX.begin(), stlX.end());
	thrust::device_vector<double> y0(stlY.begin(), stlY.end());
	thrust::device_vector<double> x1(num_points);
	thrust::device_vector<double> y1(num_points);
	end = clock();
	run_time = ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout << "Data Conversion Time: " << run_time << std::endl;

	//copy pointers to the points into the device because kernels don't support device_vector as input they accept raw_pointers
	//Thrust data types are not understood by a CUDA kernel and need to be converted back to its underlying pointer. 
	//host_points(h for host, d for device)

	Points h_points[2];
	h_points[0].set(thrust::raw_pointer_cast(&x0[0]), thrust::raw_pointer_cast(&y0[0]));
	h_points[1].set(thrust::raw_pointer_cast(&x1[0]), thrust::raw_pointer_cast(&y1[0]));


	//device_points
	Points *d_points;
	checkCudaErrors(cudaMalloc((void**)&d_points, 2 * sizeof(Points)));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(d_points, h_points, 2 * sizeof(Points), cudaMemcpyHostToDevice));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	end = clock();
	run_time = ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout << "GPU Data Transfer Time: " << run_time << std::endl;

	//Setting Cuda Heap size for dynamic memory allocation	
	size_t size = 1024 * 1024 * 1024;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);

	//Copy root node from host to device
	Quadtree_Node h_root;
	h_root.setRange(0, num_points);
	h_root.setIdx(1024);
	Quadtree_Node* d_root;
	checkCudaErrors(cudaMalloc((void**)&d_root, sizeof(Quadtree_Node)));
	checkCudaErrors(cudaMemcpy(d_root, &h_root, sizeof(Quadtree_Node), cudaMemcpyHostToDevice));

	//set the recursion limit based on max_depth
	//maximum possible depth is 24 levels
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);
	Parameters prmtrs(min_points_per_node);
	const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warp_size;
	const int SHARED_MEM_SIZE = 4 * NUM_WARPS_PER_BLOCK*sizeof(int);
	start = clock();
	buildQuadtree << <1, NUM_THREADS_PER_BLOCK, SHARED_MEM_SIZE >> >(d_root, d_points, prmtrs);
	cudaDeviceSynchronize();
	end = clock();
	run_time = ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout << "Kernel Execution Time: " << run_time << std::endl;

	// Need CPU part to create Delaunay of seed points
	for(int i=0;i<7;i++)
	{
		int n=std::rand()%(num_of_points-1);
		RandomSample.push_back(OriginalSample.at(n));
		//if(outputRandomSample.is_open()){outputRandomSample<<OriginalSample.at(n)<<std::endl;}
	}
	Delaunay dt;
	create_Delaunay(dt, RandomSample);
	
	Edge_iterator eit = dt.finite_edges_begin();
		for (; eit != dt.finite_edges_end(); ++eit)
		{//2
			if (eit->first->correct_segments[eit->second] == false)
			{
				//std::cout << ".....................................Inside............................." << std::endl;
				//std::cout << eit->first->vertex((eit->second + 1) % 3)->point() << " " << eit->first->vertex((eit->second + 2) % 3)->point() << std::endl;
				iterate = false;
				CGAL::Object o = dt.dual(eit);

				const Segment_2* s = CGAL::object_cast<Segment_2>(&o);
				const Ray_2* r = CGAL::object_cast<Ray_2>(&o);

				int num_of_intersections = 0;
				Segment_2* temp = new Segment_2;
				ThreshPoints.clear(); NewThreshPoints.clear(); Neighbors.clear(); Neighbor_Segments.clear();

				if (r)
				{  
					if (tree.rootNode->rectangle.has_on_bounded_side((*r).source())){
						*temp = convToSeg(tree.rootNode->rectangle, *r);
					}
				}
				if (s)
				{	
					*temp = *s;
				}
			//................here we need the GPU function call................
			//the return will give us the points to insert into seed (multiple intersection/farthest point) and edges to mark as restricted (1 intersection)
								
				/*ThreshPoints = insidePoints(tree, *temp, OT);
				check_duplicate(ThreshPoints);

				Delaunay dt_thresh;
				create_Delaunay(dt_thresh, ThreshPoints);

				NewThreshPoints = insidePoints(tree, *temp, IT);
				check_duplicate(NewThreshPoints);
				//std::cout<<"Thresh Points "<<ThreshPoints.size()<<"		NewThreshPoints		"<<NewThreshPoints.size()<<std::endl;

				if (ThreshPoints.size() > 2 && NewThreshPoints.size() > 1)
				{
					NNCrust(dt_thresh, NewThreshPoints, Neighbors, Neighbor_Segments, *temp, IT);
					num_of_intersections = check_Intersection(*temp, Neighbor_Segments);
					
					if (num_of_intersections > 1)
					{
						Point_2 Far_Point = get_Farthest_Point(*temp, Neighbor_Segments, eit->first->vertex((eit->second + 1) % 3)->point());
						listing.insert(std::pair<Edge, Point_2>(*eit, Far_Point));
						//std::cout << "Insertion Point	" << Far_Point << std::endl;
					}  //5

					else if (num_of_intersections == 1)   //this is for marking the restricted delaunay edge
					{
						//std::cout<<"Correct Edge	"<<eit->first->vertex((eit->second+1)%3)->point()<<" "<<eit->first->vertex((eit->second+2)%3)->point()<<std::endl;
						eit->first->correct_segments[eit->second] = true;
						fh opp_face = eit->first->neighbor(eit->second);
						int opp_index = opp_face->index(eit->first);
						opp_face->correct_segments[opp_index] = true;
					}*/
				}
				delete temp;
			}
		}//2
	
		if (!listing.empty()) // listing contains points to be inserted and their corresponding delaunay edge
		{
			//std::cin >> check;
			int n = 0;
			iterate = true;
			//std::cout << ".....................................True............................." << std::endl;
			for (std::multimap<Edge, Point_2>::iterator m_it = listing.begin(); m_it != listing.end(); ++m_it)
			{
				vh vh1, vh2;
				vh1 = ((*m_it).first).first->vertex((((*m_it).first).second + 1) % 3);
				vh2 = ((*m_it).first).first->vertex((((*m_it).first).second + 2) % 3);
				if ((dt_sample.is_edge(vh1, vh2)) && (((*m_it).first).first->correct_segments[((*m_it).first).second] == false))
				{
					vh tempVhandle = dt_sample.insert((*m_it).second);	
					//std::cout << "Point Inserted	" << tempVhandle->point() << std::endl;
					deFace(dt_sample, tempVhandle);
					markEdge(dt_sample, tree, tempVhandle);
				}
			}
			//std::cout<<"total_inEdgeMarking	"<<n<<std::endl;
		}

	
	// GPU part ...  ALL THIS WILL COME INSIDE .... if (ThreshPoints.size() > 2 && NewThreshPoints.size() > 1)...
	checkCudaErrors(cudaGetLastError());
	printQuadtree << <1, 1 >> >(d_root);
	int num_of_lines = 4; 		// These will be dynamic or set a high number as it will keep on increasing 
	printf("Before Inside Initialization\n");
	Points* d_inside_points = initializeInsidePoints(num_of_lines);
	//printf("After Inside points\n");
	cudaDeviceSynchronize();
	Line_Segment *h_lines = new Line_Segment[num_of_lines];

	// can use a for loop for voronoi edges
	h_lines[0] = Line_Segment(make_double2(100.0, -200.0), make_double2(0.0, 300.0));
	h_lines[1] = Line_Segment(make_double2(0.0, 300.0), make_double2(600.0, 650.0));
	h_lines[2] = Line_Segment(make_double2(0.0, 300.0), make_double2(-550.0, 680.0));
	h_lines[3] = Line_Segment(make_double2(100.0, -200.0), make_double2(-600.0, -650.0));


	Line_Segment *d_lines;
	checkCudaErrors(cudaMalloc((void**)&d_lines, num_of_lines*sizeof(Line_Segment)));
	checkCudaErrors(cudaMemcpy(d_lines, h_lines, num_of_lines*sizeof(Line_Segment), cudaMemcpyHostToDevice));
	double threshold = 10;

	double2 *h_delaunayPoints = new double2[num_of_lines];
	h_delaunayPoints[0] = make_double2(10.0,4.0);
	h_delaunayPoints[1] = make_double2(11.0,5.0);
	h_delaunayPoints[2] = make_double2(12.5, 6.0);
	h_delaunayPoints[3] = make_double2(13.2, 7.0);

	double2* d_delaunayPoints;

	checkCudaErrors(cudaMalloc((void**)&d_delaunayPoints, num_of_lines*sizeof(double2)));
	checkCudaErrors(cudaMemcpy(d_delaunayPoints, h_delaunayPoints, num_of_lines*sizeof(double2), cudaMemcpyHostToDevice));

	int *h_no_of_intersections = new int[num_of_lines];
	int* d_no_of_intersections;
	checkCudaErrors(cudaMalloc((void**)&d_no_of_intersections, num_of_lines*sizeof(int)));

	double2 *h_intersections = new double2[num_of_lines];
	double2* d_intersections;
	checkCudaErrors(cudaMalloc((void**)&d_intersections, num_of_lines*sizeof(double2)));
	


	std::cout << "Outer threshold Points: " << std::endl;
	start = clock();
	findOuterThresholdPoints << <1, 4 >> >(d_root, d_points, d_lines, d_inside_points, threshold);
	cudaDeviceSynchronize();
	end = clock();
	run_time = ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout << "Outer threshold Execution Time: " << run_time << std::endl;
	printPoints << <1, 1 >> >(d_inside_points, num_of_lines); // no. of line, points


	printf("____________________________");
	print_gpu_voronoi_thresholdpointsforeachedge << <1, 4 >> >();

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	// Launch a kernel on the GPU with one thread for each element.
	delaunayKernel << <1, 4 >> > (d_lines, d_delaunayPoints, d_no_of_intersections, d_intersections);
	print_delaunay << <1, 4 >> > ();
	print_delaunayindex << <1, 4 >> > ();
	print_NNcurst << <1, 4 >> > ();

	

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
	cudaStatus = cudaMemcpy(h_no_of_intersections, d_no_of_intersections, num_of_lines*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	};

	cudaStatus = cudaMemcpy(h_intersections, d_intersections, num_of_lines*sizeof(double2), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	};


	for (int i = 0; i < num_of_lines; i++)
	{
		cout << h_no_of_intersections[i] << " " << h_intersections[i].x <<" "<< h_intersections[i].y << endl;
	}

	int i;
	std::cin >> i;
}
