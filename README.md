# MeshGeneration2D

Mesh generation is the process of generating discrete geometric cells from
continuous geometric space and it has a lot applications in rendering,
physical simulations, finite element analysis etc.

The goal of this project is to parallelize the process of mesh generation 
given a initial sample points on a GPU. The process of mesh generation include
series of steps like quad tree construction, delaunay triangulation, nn crust
curve reconstruction etc. Our goal is to find and implement the best way to
parallelize each step of the process on a GPU so that the overall time for
mesh generation is significantly lower than the CPU based approach. We
used CUDA parallel computing platform to parallelize the entire process.

From the original set of sample points we take some seed points and
compute the voronoi edges. The voronoi edges and the threshold values
computed on CPU at each step are passed to the GPU where the threshold
points are computed using the quadtree and then delaunay triangulation is
done on set of threshold points for each voronoi edge parallelly. Using
delaunay triangulation the nn crust algorithm gives us a curve which we can
use to find the intersection with our voronoi edge to compute the sample
point which can be added to the initial set of seed points. Again this entire
process is repeated until the desired mesh is obtained.
