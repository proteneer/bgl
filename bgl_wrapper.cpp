//=======================================================================
// Copyright 2009 Trustees of Indiana University.
// Authors: Michael Hansen
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#include <fstream>
#include <iostream>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/mcgregor_common_subgraphs.hpp>
#include <boost/property_map/shared_array_property_map.hpp>

template <typename Graph>
struct example_callback {

  typedef typename boost::graph_traits<Graph>::vertices_size_type VertexSizeFirst;

  example_callback(const Graph& graph1) :
    m_graph1(graph1) { }

  template <typename CorrespondenceMapFirstToSecond,
            typename CorrespondenceMapSecondToFirst>
  bool operator()(CorrespondenceMapFirstToSecond correspondence_map_1_to_2,
                  CorrespondenceMapSecondToFirst correspondence_map_2_to_1,
                  VertexSizeFirst subgraph_size) {

    // Fill membership map for first graph
    typedef typename boost::property_map<Graph, boost::vertex_index_t>::type VertexIndexMap;
    typedef boost::shared_array_property_map<bool, VertexIndexMap> MembershipMap;
      
    MembershipMap membership_map1(num_vertices(m_graph1),
                                  get(boost::vertex_index, m_graph1));

    boost::fill_membership_map<Graph>(m_graph1, correspondence_map_1_to_2, membership_map1);

    // Generate filtered graphs using membership map
    typedef typename boost::membership_filtered_graph_traits<Graph, MembershipMap>::graph_type
      MembershipFilteredGraph;

    MembershipFilteredGraph subgraph1 =
      make_membership_filtered_graph(m_graph1, membership_map1);

    // Print the graph out to the console
    std::cout << "Found common subgraph (size " << subgraph_size << ")" << std::endl;
    // print_graph(subgraph1);
    // std::cout << std::endl;

    // Explore the entire space
    return (true);
  }

private:
  const Graph& m_graph1;
  VertexSizeFirst m_max_subgraph_size;
};


/*
int main (int argc, char *argv[]) {

  // Using a vecS graph here so that we don't have to mess around with
  // a vertex index map; it will be implicit.
  typedef adjacency_list<listS, vecS, directedS,
    property<vertex_name_t, unsigned int,
    property<vertex_index_t, unsigned int> >,
    property<edge_name_t, unsigned int> > Graph;

  typedef graph_traits<Graph>::vertex_descriptor Vertex;
  typedef graph_traits<Graph>::edge_descriptor Edge;

  typedef property_map<Graph, vertex_name_t>::type VertexNameMap;
  typedef property_map<Graph, edge_name_t>::type EdgeNameMap;

  // Test maximum and unique variants on known graphs
  Graph graph_simple1, graph_simple2;
  example_callback<Graph> user_callback(graph_simple1);

  VertexNameMap vname_map_simple1 = get(vertex_name, graph_simple1);
  VertexNameMap vname_map_simple2 = get(vertex_name, graph_simple2);

  // Graph that looks like a triangle
  put(vname_map_simple1, add_vertex(graph_simple1), 1);
  put(vname_map_simple1, add_vertex(graph_simple1), 2);
  put(vname_map_simple1, add_vertex(graph_simple1), 3);

  add_edge(0, 1, graph_simple1);
  add_edge(0, 2, graph_simple1);
  add_edge(1, 2, graph_simple1);

  std::cout << "First graph:" << std::endl;
  print_graph(graph_simple1);
  std::cout << std::endl;

  // Triangle with an extra vertex
  put(vname_map_simple2, add_vertex(graph_simple2), 1);
  put(vname_map_simple2, add_vertex(graph_simple2), 2);
  put(vname_map_simple2, add_vertex(graph_simple2), 3);
  put(vname_map_simple2, add_vertex(graph_simple2), 4);

  add_edge(0, 1, graph_simple2);
  add_edge(0, 2, graph_simple2);
  add_edge(1, 2, graph_simple2);
  add_edge(1, 3, graph_simple2);

  std::cout << "Second graph:" << std::endl;
  print_graph(graph_simple2);
  std::cout << std::endl;

  // All subgraphs
  std::cout << "mcgregor_common_subgraphs:" << std::endl;
  mcgregor_common_subgraphs
    (graph_simple1, graph_simple2, true, user_callback,
     vertices_equivalent(make_property_map_equivalent(vname_map_simple1, vname_map_simple2))); 
  std::cout << std::endl;

  // Unique subgraphs
  std::cout << "mcgregor_common_subgraphs_unique:" << std::endl;
  mcgregor_common_subgraphs_unique
    (graph_simple1, graph_simple2, true, user_callback,
     vertices_equivalent(make_property_map_equivalent(vname_map_simple1, vname_map_simple2))); 
  std::cout << std::endl;

  // Maximum subgraphs
  std::cout << "mcgregor_common_subgraphs_maximum:" << std::endl;
  mcgregor_common_subgraphs_maximum
    (graph_simple1, graph_simple2, true, user_callback,
     vertices_equivalent(make_property_map_equivalent(vname_map_simple1, vname_map_simple2))); 
  std::cout << std::endl;

  // Maximum, unique subgraphs
  std::cout << "mcgregor_common_subgraphs_maximum_unique:" << std::endl;
  mcgregor_common_subgraphs_maximum_unique
    (graph_simple1, graph_simple2, true, user_callback,
     vertices_equivalent(make_property_map_equivalent(vname_map_simple1, vname_map_simple2))); 

  return 0;
}
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>


namespace py = pybind11;


// ----------------
// Regular C++ code
// ----------------


boost::adjacency_list<> make_graph(const py::array_t<int, py::array::c_style> &bonds, size_t num_atoms) {
  boost::adjacency_list<> g;
  std::vector<boost::adjacency_list<>::vertex_descriptor> vertices;
  for(auto i=0; i < num_atoms; i++) {
    boost::adjacency_list<>::vertex_descriptor v = boost::add_vertex(g);
    vertices.push_back(v);
  }

  auto ptr = bonds.data();
  auto num_bonds = bonds.size()/2;

  std::vector<boost::adjacency_list<>::edge_descriptor> edges;

    

  for(int i=0; i < num_bonds; i++) {
    int src = ptr[i*2+0];
    int dst = ptr[i*2+1];
    auto edge_result = boost::add_edge(vertices[src], vertices[dst], g);
    edges.push_back(edge_result.first);
    // tbd: assert edge_result.second returns True
    // g.add_edge(src, dst)
  }

  std::cout << "Made a graph with " << boost::num_vertices(g) << " vertices and " << boost::num_edges(g) << " edges" << std::endl;

  return g;
}

// multiply all entries by 2.0
// input:  std::vector ([...]) (read only)
// output: std::vector ([...]) (new copy)
const py::array_t<int, py::array::c_style> mcs(
  const py::array_t<double, py::array::c_style> &coords_a,
  const py::array_t<int, py::array::c_style> &bonds_a,
  const py::array_t<double, py::array::c_style> &coords_b,
  const py::array_t<int, py::array::c_style> &bonds_b,
  float threshold) {


    size_t num_atoms_a = coords_a.size() / 3;
    size_t num_atoms_b = coords_b.size() / 3;

    boost::adjacency_list<> g_a = make_graph(bonds_a, num_atoms_a);
    boost::adjacency_list<> g_b = make_graph(bonds_b, num_atoms_b);

    example_callback<boost::adjacency_list<> > user_callback(g_a);

    // boost::mcgregor_common_subgraphs_maximum(
    boost::mcgregor_common_subgraphs_maximum_unique(
      g_a,
      g_b,
      true,
      user_callback
    ); 
  // std::vector<double> output;

  // std::transform(
  //   input.begin(),
  //   input.end(),
  //   std::back_inserter(output),
  //   [](double x) -> double { return 2.*x; }
  // );

  // N.B. this is equivalent to (but there are also other ways to do the same)
  //
  // std::vector<double> output(input.size());
  //
  // for ( size_t i = 0 ; i < input.size() ; ++i )
  //   output[i] = 2. * input[i];

  // return output;
  py::array_t<int, py::array::c_style> foo;
  return foo;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(bgl_wrapper,m)
{
  m.doc() = "pybind11 example plugin";

  m.def("mcs", &mcs, "Multiply all entries of a list by 2.0");
}