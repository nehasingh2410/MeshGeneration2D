
#include "QuadTree.cuh"
#include "delaunay.cuh"

__device__ double2 gpu_voronoi_thresholdpointsforeachedge[MAX_VORONOI_EDGES][MAX_POINTS_SIZE];
__device__ int countof_gpu_voronoi_thresholdpointsforeachedge[MAX_VORONOI_EDGES];
__device__ double2 gpu_nncrust_edgesforeach_voronoithresholdpoint[MAX_VORONOI_EDGES][MAX_POINTS_SIZE * 2];

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


	checkCudaErrors(cudaGetLastError());
	printQuadtree << <1, 1 >> >(d_root);
	int num_of_lines = 4;
	printf("Before Inside Initialization\n");
	Points* d_inside_points = initializeInsidePoints(num_of_lines);
	//printf("After Inside points\n");
	cudaDeviceSynchronize();
	Line_Segment *h_lines = new Line_Segment[num_of_lines];
	h_lines[0] = Line_Segment(make_double2(100.0, -200.0), make_double2(0.0, 300.0));
	h_lines[1] = Line_Segment(make_double2(0.0, 300.0), make_double2(600.0, 650.0));
	h_lines[2] = Line_Segment(make_double2(0.0, 300.0), make_double2(-550.0, 680.0));
	h_lines[3] = Line_Segment(make_double2(100.0, -200.0), make_double2(-600.0, -650.0));


	Line_Segment* d_lines;
	checkCudaErrors(cudaMalloc((void**)&d_lines, num_of_lines*sizeof(Line_Segment)));
	checkCudaErrors(cudaMemcpy(d_lines, h_lines, num_of_lines*sizeof(Line_Segment), cudaMemcpyHostToDevice));
	double threshold = 10;

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
	delaunayKernel << <1, 4 >> > ();
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
	int i;
	std::cin >> i;
}