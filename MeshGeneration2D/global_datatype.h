#ifndef GLOBAL_DATATYPES_H
#define GLOBAL_DATATYPES_H

#include <cstdio>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <cassert>
#include <stdlib.h>
#include <stdio.h>
#include <stack>
#include <queue>
#include <sstream>
#include <time.h>
#include <cmath>
#include <map>
#include <unordered_map>
// #include <boost/system/config.hpp>

#include <CGAL/Quotient.h>
#include <CGAL/Simple_cartesian.h>
//#include <CGAL/Direction_2.h>

//#ifdef WIN32
#include <CGAL/Filtered_kernel.h>
//#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#else
	//#include <CGAL/Arithmetic_filter.h>
//#endif
#include <CGAL\Triangulation_2.h>
#include <CGAL/Triangulation_face_base_2.h>
#include <CGAL/Triangulation_vertex_base_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

#include <CGAL/Triangulation_euclidean_traits_2.h>
#include <CGAL/Delaunay_triangulation_2.h>

#include <CGAL/intersections.h>
#include <CGAL/squared_distance_2.h>
#include <CGAL/Point_2.h>

using namespace std;

//#undef CGAL_CARTESIAN_H



//new class for 2d triangle

//Define a new vertex base class for storing the vertices in 2D triangulation of each facet of the input

template <class Gt, class Vb = CGAL::Triangulation_vertex_base_with_info_2<unsigned int,Gt> >
class Delaunay_vertex_2D : public Vb
{
	typedef Vb											Base;

public:
	typedef typename Vb::Vertex_handle					Vertex_handle;
	typedef typename Vb::Face_handle					Face_handle;
	typedef typename Vb::Point							Point;


	//rebind mechanism
	template < typename TDS2>
	struct	Rebind_TDS {
		typedef	typename Vb::template Rebind_TDS<TDS2>::Other	Vb2;
		typedef Delaunay_vertex_2D<Gt, Vb2>						Other;
	};

public:
	Delaunay_vertex_2D():Base() {init();}
	Delaunay_vertex_2D( const Point& p) : Base(p) {init();}
	Delaunay_vertex_2D( const Point& p, Face_handle f) : Base(f,p) {init();}
	Delaunay_vertex_2D( Face_handle f): Base(f) {init();}
	~Delaunay_vertex_2D() {}
	int id;

private:
	inline void init()
	{
		id=0;
	}
};

//Define a new face base class for storing the facets of the 2D triangulation of each facet of the input

template <class Gt, class Fb = CGAL::Triangulation_face_base_2<Gt> >
class Delaunay_face_2D : public Fb
{

	typedef Fb											Base;
	typedef typename Fb::Triangulation_data_structure   TDS;

public:
	typedef Gt											Geom_traits;
	typedef TDS											Triangulation_data_structure;
	typedef typename TDS::Vertex_handle					Vertex_handle;
	typedef typename TDS::Face_handle					Face_handle;

	//rebind mechanism
	template < typename TDS2>
	struct	Rebind_TDS {
		typedef	typename Fb::template Rebind_TDS<TDS2>::Other	Fb2;
		typedef Delaunay_face_2D<Gt, Fb2>						Other;
	};

	bool face_is_inside;
	bool correct_segments[3];
	int component;

	Delaunay_face_2D():Base() {init();}
	Delaunay_face_2D( Vertex_handle v0, Vertex_handle v1, Vertex_handle v2) : Base(v0, v1, v2) {init();}
	Delaunay_face_2D( Vertex_handle v0, Vertex_handle v1, Vertex_handle v2, Face_handle n0, Face_handle n1, Face_handle n2) : Base(v0, v1, v2, n0, n1, n2) {init();}
	~Delaunay_face_2D() {}
	inline void init() {
		correct_segments[0] = false;
		correct_segments[1] = false;
		correct_segments[2] = false;

		face_is_inside = false;
		component = 0;
	}
};



//Globals for 2D
typedef CGAL::Simple_cartesian<double> Rep;
//typedef CGAL::Exact_predicates_exact_constructions_kernel Rep;
//typedef CGAL::Filtered_kernel<Rep> my_K;
typedef CGAL::Exact_predicates_inexact_constructions_kernel my_K;
struct K : public my_K {} ;

//typedef CGAL::Direction_2<my_K>									Direction_2;

typedef CGAL::Triangulation_euclidean_traits_2<K>				Traits_2d;
typedef Delaunay_vertex_2D<K>									vb1;
typedef Delaunay_face_2D<K>										fb1;
typedef CGAL::Triangulation_data_structure_2<vb1,fb1>			Tds_2d;
typedef CGAL::Delaunay_triangulation_2<Traits_2d, Tds_2d>		Delaunay;
//typedef CGAL::Segment_2<Rep>									segment_2d;
typedef CGAL::Segment_2<Rep>									line_2d;
//typedef CGAL::Segment_2											line_2d;
//typedef my_K::Segment_2											line_2d;
//typedef CGAL::Direction_2<Rep>									Direction_2;

typedef Delaunay::Face_circulator								Face_circulator;
typedef Delaunay::Finite_faces_iterator							Finite_faces_iterator_2d;
typedef Delaunay::Finite_vertices_iterator						Finite_vertices_iterator_2d;
typedef Delaunay::Finite_edges_iterator							e_fit;
typedef Delaunay::Vertex_circulator								vc;
typedef Delaunay::Face_iterator									fit;
typedef Delaunay::All_faces_iterator							all_fit;
typedef Delaunay::Vertex_circulator								Vertex_circulator;
typedef Delaunay::Vertex_handle									vh;
typedef Delaunay::Face_circulator								Face_circulator;
typedef Delaunay::Face_handle									fh;
typedef Delaunay::Point											Point;

#define OT 3500
#define IT 3000

//End of Header Files
typedef my_K::Intersect_2									Intersect_2;

typedef CGAL::Point_2<Rep>									Point_1;
typedef CGAL::Line_2<Rep>									Line_2;
typedef Delaunay::Edge_iterator                     		Edge_iterator;
typedef Delaunay::Edge_circulator                     		Edge_circulator;
typedef Delaunay::Vertex_iterator							Vertex_iterator;
typedef my_K::Point_2		                     		    Point_2;
typedef my_K::Vector_2		                     		    Vector_2;
typedef my_K::Segment_2		                     		    Segment_2;
typedef my_K::Ray_2										    Ray_2;
//typedef my_K::Line_2										Line_2;
typedef my_K::Direction_2									Direction_2;
typedef my_K::Iso_rectangle_2								Iso_rectangle_2;
typedef Delaunay::Face										Face;
typedef Delaunay::Edge										Edge;

extern int MAX_NUM;
//double total_inTopology = 0;
//double total_inGeometry = 0;
//double total_inManifold = 0;

//double total_circumcenter_position = 0;

extern std::vector<Point_2> ThreshPoints, NewThreshPoints;
extern std::vector<vector <Point_2> > Neighbors;
extern std::vector<Segment_2> Neighbor_Segments;
//typedef pair<vh, vh>										edge_vh;

//typedef Delaunay::Iso_rectangle_2											Iso_rectangle_2;

/*typedef CGAL::Point_2<Rep>									Point_2d;
typedef CGAL::Ray_2<Rep>									Ray_2d;
typedef CGAL::Iso_rectangle_2<Rep>								Iso_rectangle_2d;
typedef CGAL::Segment_2<Rep>									Segment_2d;
typedef CGAL::Triangle_2<Rep>									Triangle_2d;
*/

extern std::multimap<Edge, Point_2> listing;


#endif //GLOBAL_DATATYPE_H
