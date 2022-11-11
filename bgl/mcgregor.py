# maximum common subgraph routines based off of the mcgregor paper
import numpy as np
import copy
import time

# when computed on a leaf node this is equal to the number of edges mapped.
def arcs_left(marcs):
    # courtesy of jfass
    return np.sum(np.any(marcs, 1))


UNMAPPED = -1


def initialize_marcs_given_predicate(g1, g2, predicate):
    num_a_edges = g1.n_edges
    num_b_edges = g2.n_edges
    marcs = np.ones((num_a_edges, num_b_edges), dtype=np.int32)
    for e_a in range(num_a_edges):
        src_a, dst_a = g1.edges[e_a]
        for e_b in range(num_b_edges):
            src_b, dst_b = g2.edges[e_b]
            # an edge is allowed in two cases:
            # 1) src_a can map to src_b, and dst_a can map dst_b
            # 2) src_a can map to dst_b, and dst_a can map src_b
            # if either 1 or 2 is satisfied, we skip, otherwise
            # we can confidently reject the mapping
            if predicate[src_a][src_b] and predicate[dst_a][dst_b]:
                continue
            elif predicate[src_a][dst_b] and predicate[dst_a][src_b]:
                continue
            else:
                marcs[e_a][e_b] = 0

    return marcs


# this is the bottleneck
def refine_marcs(g1, g2, new_v1, new_v2, marcs):
    """
    return vertices that have changed
    """
    new_marcs = marcs.copy()
    for e1 in g1.get_edges(new_v1):
        # don't if new_v2 here since new_v2 may be zero!
        if new_v2 != UNMAPPED:
            new_marcs[e1] &= g2.get_edges_as_vector(new_v2)
        else:
            # v1 is explicitly mapped to None, so we zero out all edges
            new_marcs[e1, :] = 0

    return new_marcs


class MCSResult:
    def __init__(self, maps_1_to_2):
        self.all_maps = [maps_1_to_2]
        self.num_edges = 0
        self.timed_out = False
        self.nodes_visited = 0


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

        self.ve_matrix = np.zeros((self.n_vertices, self.n_edges), dtype=np.int32)
        for vertex_idx, edges in enumerate(self.lol_edges):
            for edge_idx in edges:
                self.ve_matrix[vertex_idx][edge_idx] = 1

    def get_edges(self, vertex):
        return self.lol_edges[vertex]

    def get_edges_as_vector(self, vertex):
        # return edges as a boolean vetor
        return self.ve_matrix[vertex]


def build_predicate_matrix(n_a, n_b, priority_idxs):
    assert len(priority_idxs) == n_a
    pmat = np.zeros((n_a, n_b), dtype=np.int32)
    for idx, jdxs in enumerate(priority_idxs):
        for jdx in jdxs:
            pmat[idx][jdx] = 1
    return pmat


def mcs(n_a, n_b, priority_idxs, bonds_a, bonds_b, timeout):

    assert n_a <= n_b

    g_a = Graph(n_a, bonds_a)
    g_b = Graph(n_b, bonds_b)

    map_a_to_b = AtomMap(n_a, n_b)
    mcs_result = MCSResult(map_a_to_b)

    predicate = build_predicate_matrix(n_a, n_b, priority_idxs)
    marcs = initialize_marcs_given_predicate(g_a, g_b, predicate)

    start_time = time.time()

    priority_idxs = tuple(tuple(x) for x in priority_idxs)

    recursion(g_a, g_b, map_a_to_b, 0, marcs, mcs_result, priority_idxs, start_time, timeout)

    print("=====NODES VISITED", mcs_result.nodes_visited)

    all_cores = []

    for atom_map in mcs_result.all_maps:
        core = []
        for a, b in enumerate(atom_map.map_1_to_2):
            if b != UNMAPPED:
                core.append((a, b))
        core = np.array(sorted(core))
        all_cores.append(core)

    return all_cores, mcs_result.timed_out


class AtomMap:
    def __init__(self, n_a, n_b):
        self.map_1_to_2 = [UNMAPPED] * n_a
        self.map_2_to_1 = [UNMAPPED] * n_b

    def add(self, idx, jdx):
        self.map_1_to_2[idx] = jdx
        self.map_2_to_1[jdx] = idx

    def pop(self, idx, jdx):
        self.map_1_to_2[idx] = UNMAPPED
        self.map_2_to_1[jdx] = UNMAPPED


def recursion(g1, g2, atom_map, layer, marcs, mcs_result, priority_idxs, start_time, timeout):

    mcs_result.nodes_visited += 1

    if time.time() - start_time > timeout:
        mcs_result.timed_out = True
        return

    n_a = g1.n_vertices

    num_edges = arcs_left(marcs)

    # every atom has been mapped
    if layer == n_a:
        if mcs_result.num_edges < num_edges:
            mcs_result.all_maps = [copy.deepcopy(atom_map)]
            mcs_result.num_edges = num_edges
        elif mcs_result.num_edges == num_edges:
            # print("Found an equal or better complete map with", num_edges, "edges")
            mcs_result.all_maps.append(copy.deepcopy(atom_map))
        return

    if num_edges < mcs_result.num_edges:
        return

    # check possible subtrees
    found = False
    # priority_idxs has shape n_a x n_b, typically this is spatially sorted based on distance
    for jdx in priority_idxs[layer]:
        if atom_map.map_2_to_1[jdx] == UNMAPPED:
            atom_map.add(layer, jdx)
            new_marcs = refine_marcs(g1, g2, layer, jdx, marcs)
            recursion(g1, g2, atom_map, layer + 1, new_marcs, mcs_result, priority_idxs, start_time, timeout)
            atom_map.pop(layer, jdx)
            found = True

    # handle the case where we have no valid matches (due to the predicate conditions)
    # (ytz): do we always want to consider this to be a valid possibility?
    if not found:
        # atom_map[layer] = None # don't need, since default is a no-map
        new_marcs = refine_marcs(g1, g2, layer, UNMAPPED, marcs)  # we can make this probably affect only a subslice!
        recursion(g1, g2, atom_map, layer + 1, new_marcs, mcs_result, priority_idxs, start_time, timeout)
        # atom_map.pop(layer) # don't need to pop, never added anything


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

    test_compute_marcs()
