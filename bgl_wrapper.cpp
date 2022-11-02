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
  MCSResult() : largest(0) {};

  VertexSize largest;
  std::vector<int> core;

};

struct callback {

public:

  callback(MCSResult *result, int timeout, Graph g_a, Graph g_b) : g_a_(g_a), g_b_(g_b), result_(result), timeout_(timeout), start_(std::chrono::steady_clock::now()) {}

  template <typename CorrespondenceMapFirstToSecond,
            typename CorrespondenceMapSecondToFirst>
  bool operator()(CorrespondenceMapFirstToSecond correspondence_map_1_to_2,
                  CorrespondenceMapSecondToFirst correspondence_map_2_to_1,
                  VertexSize subgraph_size) {

    // Print the graph out to the console
    if(subgraph_size > result_->largest) {
      std::cout << this << " | found larger common subgraph of size " << subgraph_size << std::endl;
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
      
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>(end - start_).count() < timeout_;
  }

private:

  Graph g_a_;
  Graph g_b_;
  MCSResult *result_;
  int timeout_;
  std::chrono::steady_clock::time_point start_;

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

      size_t num_atoms_a = predicates_.shape()[0];
      size_t num_atoms_b = predicates_.shape()[1];

      auto idx = first*num_atoms_b + second;
      auto pred_ptr = predicates_.data();

      if(idx >= predicates_.size()) {
        throw std::runtime_error("OOB");
      }
      bool result = pred_ptr[idx];

      return result;

  }
};


const py::array_t<int, py::array::c_style> mcs(
  const py::array_t<int, py::array::c_style> &predicates,
  const py::array_t<int, py::array::c_style> &bonds_a,
  const py::array_t<int, py::array::c_style> &bonds_b,
  int timeout) {

    size_t num_atoms_a = predicates.shape()[0];
    size_t num_atoms_b = predicates.shape()[1];

    Graph g_a = make_graph(bonds_a, num_atoms_a);
    Graph g_b = make_graph(bonds_b, num_atoms_b);

    MCSResult result;
    callback user_callback(&result, timeout, g_a, g_b);

    boost::mcgregor_common_subgraphs_unique(
      g_a,
      g_b,
      boost::get(boost::vertex_index, g_a),
      boost::get(boost::vertex_index, g_b),
      edge_always_equivalent(),
      atom_predicate(predicates),
      true,
      user_callback
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
  m.doc() = "pybind11 example plugin";

  m.def("mcs", &mcs, "Multiply all entries of a list by 2.0");
}