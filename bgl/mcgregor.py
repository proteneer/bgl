# maximum common subgraph routines based off of the mcgregor paper
import numpy as np
import copy
import time

# when computed on a leaf node this is equal to the number of edges mapped.
def arcs_left(marcs):
    # courtesy of jfass
    return np.sum(np.any(marcs, 1))

# this is mainly used for self-consistent
def compute_marcs_given_maps(g1, g2, map_1_to_2):
    num_a_edges = g1.n_edges
    num_b_edges = g2.n_edges
    marcs = np.ones((num_a_edges, num_b_edges), dtype=np.int32)
    for v1, v2 in map_1_to_2.items():
        for e1 in g1.get_edges(v1):
            if v2 is not None:
                e2 = g2.get_edges(v2)
                # set any non-adjacent edges to zero
                for ej in range(num_b_edges):
                    if ej not in e2:
                        marcs[e1][ej] = 0
            else:
                # v1 is explicitly mapped to None, so we zero out all edges
                for ej in range(num_b_edges):
                    marcs[e1][ej] = 0

    return marcs


# this is the bottleneck
def refine_marcs(g1, g2, new_v1, new_v2, marcs):
    new_marcs = marcs.copy()
    num_b_edges = g2.n_edges
    for e1 in g1.get_edges(new_v1):
        # don't if new_v2 here since new_v2 may be zero!
        if new_v2 is not None:
            e2 = g2.get_edges(new_v2)
            # set any non-adjacent edges to zero
            for ej in range(num_b_edges):
                if ej not in e2:
                    new_marcs[e1][ej] = 0
        else:
            # v1 is explicitly mapped to None, so we zero out all edges
            for ej in range(num_b_edges):
                new_marcs[e1][ej] = 0

    return new_marcs


class MCSResult:
    def __init__(self, maps_1_to_2):
        self.all_maps = [maps_1_to_2]
        self.num_edges = 0


class Graph:
    def __init__(self, n_vertices, edges):
        self.n_vertices = n_vertices
        self.n_edges = len(edges)
        self.edges = edges

        cmat = np.zeros((n_vertices, n_vertices))
        for i, j in edges:
            cmat[i][j] = 1
            cmat[j][i] = 1

        # list of lists, n_vertices x n_vertices
        self.lol_vertices = []
        for idx in range(n_vertices):
            nbs = []
            for jdx in range(n_vertices):
                if cmat[idx][jdx]:
                    nbs.append(jdx)
            self.lol_vertices.append(nbs)

        # list of lists, n_vertices x n_edges
        self.lol_edges = [[] for _ in range(n_vertices)]

        # note: lol_edges are not sorted.
        for edge_idx, (src, dst) in enumerate(edges):
            self.lol_edges[src].append(edge_idx)
            self.lol_edges[dst].append(edge_idx)

    def get_edges(self, vertex):
        return self.lol_edges[vertex]


def mcs(predicate, bonds_a, bonds_b, timeout):

    n_a = predicate.shape[0]
    n_b = predicate.shape[1]

    assert n_a <= n_b

    g_a = Graph(n_a, bonds_a)
    g_b = Graph(n_b, bonds_b)

    map_a_to_b = {}
    mcs_result = MCSResult(map_a_to_b)
    marcs = compute_marcs_given_maps(g_a, g_b, map_a_to_b)

    start_time = time.time()
    timeout = 60 # tbd make dynamic

    recursion(g_a, g_b, map_a_to_b, 0, marcs, mcs_result, predicate, start_time, timeout)

    all_cores = []

    for atom_map in mcs_result.all_maps:
        core = []
        for a, b in atom_map.items():
            if b is not None:
                core.append((a,b))
        core = np.array(sorted(core))
        all_cores.append(core)

    return all_cores


def recursion(g1, g2, map_1_to_2, layer, marcs, mcs_result, predicate, start_time, timeout):

    if time.time() - start_time > timeout:
        print("timed out")
        return

    n_a = g1.n_vertices
    n_b = g2.n_vertices

    assert n_a <= n_b

    num_edges = arcs_left(marcs)

    # every atom has been mapped
    if layer == n_a:
        if mcs_result.num_edges < num_edges:
            mcs_result.all_maps = [map_1_to_2]
            mcs_result.num_edges = num_edges
        elif mcs_result.num_edges == num_edges:
            # print("Found an equal or better complete map with", num_edges, "edges")
            mcs_result.all_maps.append(map_1_to_2)
        return

    if num_edges < mcs_result.num_edges:
        return

    mapped_2_set = set(map_1_to_2.values())

    # check possible subtrees
    found = False
    for jdx in range(n_b):
        if jdx not in mapped_2_set and predicate[layer][jdx]:
            new_map = copy.deepcopy(map_1_to_2)
            new_map[layer] = jdx
            new_marcs = refine_marcs(g1, g2, layer, jdx, marcs)
            recursion(g1, g2, new_map, layer + 1, new_marcs, mcs_result, predicate, start_time, timeout)
            found = True

    # handle the case where we have no valid matches (due to the predicate conditions)
    # (ytz): do we always want to consider this to be a valid possibility?   
    if not found:
        new_map = copy.deepcopy(map_1_to_2)
        new_map[layer] = None
        new_marcs = refine_marcs(g1, g2, layer, None, marcs) # we can make this probably affect only a subslice!
        recursion(g1, g2, new_map, layer + 1, new_marcs, mcs_result, predicate, start_time, timeout)


def test_compute_marcs():

    #             0       1       2       3       4       5       6       7       8       9
    g1_edges = [[0, 1], [0, 2], [1, 2], [1, 3], [1, 4], [3, 6], [4, 6], [7, 4], [2, 7], [2, 5]]
    g1_n_vertices = 8

    g1 = Graph(g1_n_vertices, g1_edges)

    assert g1.lol_vertices == [
        [1, 2],
        [0, 2, 3, 4],
        [0, 1, 5, 7],
        [1, 6],
        [1, 6, 7],
        [2],
        [3, 4],
        [2, 4],
    ]

    #             0       1       2       3       4       5       6       7
    g2_edges = [[0, 1], [0, 2], [1, 4], [1, 3], [3, 6], [2, 5], [4, 5], [4, 7]]
    g2_n_vertices = 8

    g2 = Graph(g2_n_vertices, g2_edges)

    print("No seed")

    map_1_to_2 = {}
    num_edges = 0
    mcs_result = MCSResult(map_1_to_2, num_edges)
    marcs = compute_marcs_given_maps(g1, g2, map_1_to_2)

    predicates = np.ones((g1.n_vertices, g2.n_vertices))

    recursion(g1, g2, map_1_to_2, 0, marcs, mcs_result, predicates)


    print("Pre initialized")

    map_1_to_2 = {0: 0}
    num_edges = 0
    mcs_result = MCSResult(map_1_to_2, num_edges)
    marcs = compute_marcs_given_maps(g1, g2, map_1_to_2)
    recursion(g1, g2, map_1_to_2, 0 + 1, marcs, mcs_result, predicates)  # note layer + 1


if __name__ == "__main__":

    # recursion({}, 3, 3, 0)
    test_compute_marcs()
