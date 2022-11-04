//=======================================================================
// Copyright 2009 Trustees of Indiana University.
// Authors: Michael Hansen
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#include <stdexcept>
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


/// define the boost-graph
typedef boost::adjacency_list<
  boost::vecS,
  boost::vecS,
  boost::undirectedS> Graph;

typedef typename boost::graph_traits<Graph>::vertices_size_type VertexSize;


#include <chrono>


// tbd: thread-safety?
struct MCSResult {


public:
  MCSResult() : largest(0), last_update(std::chrono::steady_clock::now()) {};

  VertexSize largest;
  std::chrono::steady_clock::time_point last_update;
  std::vector<int> core;

};

struct callback {

public:

  callback(MCSResult *result, int timeout, Graph g_a, Graph g_b) : g_a_(g_a), g_b_(g_b), result_(result), timeout_(timeout) {}

  template <typename CorrespondenceMapFirstToSecond,
            typename CorrespondenceMapSecondToFirst>
  bool operator()(CorrespondenceMapFirstToSecond correspondence_map_1_to_2,
                  CorrespondenceMapSecondToFirst correspondence_map_2_to_1,
                  VertexSize subgraph_size) {

    auto now = std::chrono::steady_clock::now();

    // Print the graph out to the console
    if(subgraph_size > result_->largest) {
      // std::cout << this << " | found larger common subgraph of size " << subgraph_size << std::endl;
      result_->largest = subgraph_size;
      std::vector<int> core;
      BGL_FORALL_VERTICES_T(vertex_a, g_a_, Graph) {
        // Skip unmapped vertices
        if (get(correspondence_map_1_to_2, vertex_a) != boost::graph_traits<Graph>::null_vertex()) {
          core.push_back(vertex_a);
          core.push_back(get(correspondence_map_1_to_2, vertex_a))  ;
          // std::cout << vertex_a << " <-> " << get(correspondence_map_1_to_2, vertex_a) << std::endl;
        }
      }

      result_->core = core;

      // tbd: only accept if we satisfy chiral restraints
      // tbd: add support for additional cores
      // reset time
      result_->last_update = std::chrono::steady_clock::now();
      return true;
    }
    
    return std::chrono::duration_cast<std::chrono::seconds>(now - result_->last_update).count() < timeout_;
        
  }

private:

  Graph g_a_;
  Graph g_b_;
  MCSResult *result_;
  int timeout_;

};


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

Graph make_graph(
  const py::array_t<int, py::array::c_style> &bonds,
  size_t num_atoms) {
  Graph g;
  std::vector<Graph::vertex_descriptor> vertices;

  for(size_t i=0; i < num_atoms; i++) {
    Graph::vertex_descriptor v = boost::add_vertex(g);
    vertices.push_back(v);
  }

  auto bond_ptr = bonds.data();
  size_t num_bonds = bonds.size()/2;

  std::vector<Graph::edge_descriptor> edges;

  for(size_t i=0; i < num_bonds; i++) {
    int src = bond_ptr[i*2+0];
    int dst = bond_ptr[i*2+1];
    auto edge_result = boost::add_edge(vertices[src], vertices[dst], g);
    edges.push_back(edge_result.first);

    if(edge_result.second != true) {
      throw std::runtime_error("bad edge");
    }

  }

  return g;
}

struct edge_always_equivalent {

  template <typename ItemFirst,
            typename ItemSecond>
  bool operator()(const ItemFirst&, const ItemSecond&) {
    return true;
  }
};

struct atom_predicate {

private:

  const py::array_t<int, py::array::c_style> &predicates_;

public:

  atom_predicate(
    const py::array_t<int, py::array::c_style> &predicates
    ) : 
    predicates_(predicates) {};

  bool operator() (
    const boost::graph_traits<Graph>::vertex_descriptor &first,
    const boost::graph_traits<Graph>::vertex_descriptor &second) {

      // size_t num_atoms_a = predicates_.shape()[0];
      size_t num_atoms_b = predicates_.shape()[1];

      size_t idx = first*num_atoms_b + second;
      auto pred_ptr = predicates_.data();

      if(idx >= predicates_.size()) {
        throw std::runtime_error("OOB");
      }
      bool result = pred_ptr[idx];

      return result;

  }
};

// re-use some code from boost::detail internals to setup an initial atom-map
template < typename GraphFirst, typename GraphSecond,
  typename VertexIndexMapFirst, typename VertexIndexMapSecond,
  typename EdgeEquivalencePredicate, typename VertexEquivalencePredicate,
  typename SubGraphInternalCallback >
void mcgregor_common_subgraphs_v2(
  const GraphFirst& graph1, const GraphSecond& graph2,
  const VertexIndexMapFirst vindex_map1,
  const VertexIndexMapSecond vindex_map2,
  EdgeEquivalencePredicate edges_equivalent,
  VertexEquivalencePredicate vertices_equivalent,
  bool only_connected_subgraphs,
  SubGraphInternalCallback subgraph_callback,
  const py::array_t<int, py::array::c_style> initial_core) {
  typedef boost::detail::mcgregor_common_subgraph_traits< GraphFirst, GraphSecond,
    VertexIndexMapFirst, VertexIndexMapSecond >
    SubGraphTraits;

  std::map<int, int> initial_core_12;
  std::map<int, int> initial_core_21;

  for(int i=0; i < initial_core.size()/2; i++) {
    int a = initial_core.data()[i*2+0];
    int b = initial_core.data()[i*2+1];
    initial_core_12.insert({a, b});
    initial_core_21.insert({b, a});
  }

  typename SubGraphTraits::correspondence_map_first_to_second_type
    correspondence_map_1_to_2(num_vertices(graph1), vindex_map1);


  typedef typename boost::graph_traits< GraphFirst >::vertex_descriptor VertexFirst;
  std::stack< VertexFirst > vertex_stack1;

  BGL_FORALL_VERTICES_T(vertex1, graph1, GraphFirst)
  {
    if(initial_core_12.find(vertex1) == initial_core_12.end()) {
      put(correspondence_map_1_to_2, vertex1, boost::graph_traits< GraphSecond >::null_vertex());
    } else {
      put(correspondence_map_1_to_2, vertex1, initial_core_12.at(vertex1)); 
      vertex_stack1.push(vertex1);
    }
  }

  typename SubGraphTraits::correspondence_map_second_to_first_type
    correspondence_map_2_to_1(num_vertices(graph2), vindex_map2);

  BGL_FORALL_VERTICES_T(vertex2, graph2, GraphSecond)
  {
    if(initial_core_21.find(vertex2) == initial_core_21.end()) {
      put(correspondence_map_2_to_1, vertex2, boost::graph_traits< GraphFirst >::null_vertex());
    } else {
      put(correspondence_map_2_to_1, vertex2, initial_core_21.at(vertex2));
    }
  }

  boost::detail::mcgregor_common_subgraphs_internal(graph1, graph2, vindex_map1,
    vindex_map2, correspondence_map_1_to_2, correspondence_map_2_to_1,
    vertex_stack1, edges_equivalent, vertices_equivalent,
    only_connected_subgraphs, subgraph_callback);
}


const py::array_t<int, py::array::c_style> mcs(
  const py::array_t<int, py::array::c_style> &predicates,
  const py::array_t<int, py::array::c_style> &bonds_a,
  const py::array_t<int, py::array::c_style> &bonds_b,
  int timeout,
  const py::array_t<int, py::array::c_style> &initial_core) {

    size_t num_atoms_a = predicates.shape()[0];
    size_t num_atoms_b = predicates.shape()[1];

    Graph g_a = make_graph(bonds_a, num_atoms_a);
    Graph g_b = make_graph(bonds_b, num_atoms_b);

    MCSResult result;
    callback user_callback(&result, timeout, g_a, g_b);

    // check that initial_core satisifies given predicates
    for(int i=0; i < initial_core.size()/2; i++) {
      int a = initial_core.data()[i*2+0];
      int b = initial_core.data()[i*2+1];
      int pred = predicates[a*num_atoms_b + b];
      if(!pred) {
        throw std::runtime_error("Initial core predicate fails to satisfy given predicate.")
      }
    }

    result = mcgregor_common_subgraphs_v2(
      g_a,
      g_b,
      boost::get(boost::vertex_index, g_a),
      boost::get(boost::vertex_index, g_b),
      edge_always_equivalent(),
      atom_predicate(predicates),
      true,
      user_callback,
      initial_core
    );

    int num_core_atoms = result.core.size()/2;

    py::array_t<int, py::array::c_style> core({num_core_atoms, 2});

    for(int i=0; i < core.size(); i++) {
      core.mutable_data()[i] = result.core[i];
    }

    return core;
}

namespace py = pybind11;

PYBIND11_MODULE(bgl_wrapper,m)
{
  m.doc() = "boost::mcgregor_common_subgraphs wrapper";

  m.def("mcs", &mcs, "maximum common substructure (respecting pairwise atom predicates)");
}