//=======================================================================
// Copyright 2009 Trustees of Indiana University.
// Authors: Michael Hansen
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// boost
#include <boost/lexical_cast.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/mcgregor_common_subgraphs.hpp>
#include <boost/property_map/shared_array_property_map.hpp>

// pybind
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
using Vertex = Graph::vertex_descriptor;
using VertexSize = boost::graph_traits<Graph>::vertices_size_type;
using Edge = Graph::edge_descriptor;

using Clock = std::chrono::steady_clock;

namespace py = pybind11;
template<class T>
using PyArray = py::array_t<T, py::array::c_style>;

// tbd: thread-safety?
struct MCSResult {

public:
  MCSResult() : largest(0), last_update(Clock::now()) {};

  VertexSize largest;
  Clock::time_point last_update;
  std::vector<int> core;
};

struct callback {

public:

  callback(MCSResult *result, int timeout, Graph g_a, Graph g_b) : g_a_(g_a), g_b_(g_b), result_(result), timeout_seconds_(timeout), num_graphs_searched_(0) {}

  template <typename CorrespondenceMapFirstToSecond, typename Unused>
  bool operator()(CorrespondenceMapFirstToSecond correspondence_map_1_to_2,
                  Unused unused,
                  VertexSize subgraph_size) {

    auto now = Clock::now();
    num_graphs_searched_ += 1;
    
    if(subgraph_size > result_->largest) {
      //std::cout << this << " | found larger common subgraph of size " << subgraph_size << std::endl;
      result_->largest = subgraph_size;
      std::vector<int> core;
      BGL_FORALL_VERTICES_T(vertex_a, g_a_, Graph) {
        // Skip unmapped vertices
        if (get(correspondence_map_1_to_2, vertex_a) != boost::graph_traits<Graph>::null_vertex()) {
          core.push_back(vertex_a);
          core.push_back(get(correspondence_map_1_to_2, vertex_a))  ;
          
          // Print the graph out to the console
          //std::cout << vertex_a << " <-> " << get(correspondence_map_1_to_2, vertex_a) << std::endl;
        }
      }

      result_->core = core;

      // tbd: only accept if we satisfy chiral restraints
      // tbd: add support for additional cores
      // reset time
      result_->last_update = Clock::now();
      return true;
    }

    auto seconds_since_last_update = std::chrono::duration_cast<std::chrono::seconds>(now - result_->last_update).count();
    bool timed_out = seconds_since_last_update > timeout_seconds_;
    if(timed_out) {
      std::cout << this << " | timed out! (after " << num_graphs_searched_ << " unique subgraphs considered)" << std::endl;
    }

    return not timed_out;
  }

private:

  Graph g_a_;
  Graph g_b_;
  MCSResult *result_;
  int timeout_seconds_;
  int num_graphs_searched_;

};


Graph make_graph(
  const PyArray<int> &bonds,
  size_t num_atoms) {
  Graph g;

  std::vector<Vertex> vertices;

  for(size_t i=0; i < num_atoms; i++) {
    auto v = boost::add_vertex(g);
    vertices.push_back(v);
  }

  auto bond_ptr = bonds.data();
  size_t num_bonds = bonds.size()/2;

  std::vector<Edge> edges;

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

  template <typename ItemFirst, typename ItemSecond>
  bool operator()(const ItemFirst&, const ItemSecond&) {
    return true;
  }

};

// TODO: use copy of python predicates array?
struct atom_predicate {

private:

  const PyArray<int> &predicates_;

public:

  atom_predicate(
    const PyArray<int> &predicates
    ) : 
    predicates_(predicates) {};

  bool operator() (
    const Vertex &first,
    const Vertex &second) {

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

// TODO: factor mcs from python wrapper
const PyArray<int> mcs(
  const PyArray<int> &predicates,
  const PyArray<int> &bonds_a,
  const PyArray<int> &bonds_b,
  int timeout) {

    size_t num_atoms_a = predicates.shape()[0];
    size_t num_atoms_b = predicates.shape()[1];

    Graph g_a = make_graph(bonds_a, num_atoms_a);
    Graph g_b = make_graph(bonds_b, num_atoms_b);

    MCSResult result;
    callback user_callback(&result, timeout, g_a, g_b);

    // boost::mcgregor_common_subgraphs_unique(
    boost::mcgregor_common_subgraphs(
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
  PyArray<int> core({num_core_atoms, 2});
  for(int i=0; i < core.size(); i++) {
    core.mutable_data()[i] = result.core[i];
  }

  return core;
}

// pybind wrapper
PYBIND11_MODULE(bgl_wrapper,m)
{
  m.doc() = "boost::mcgregor_common_subgraphs wrapper";
  m.def("mcs", &mcs, "maximum common substructure (respecting pairwise atom predicates)");
}
