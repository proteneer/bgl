# maximum common subgraph routines based off of the mcgregor paper
import numpy as np
import copy

# when computed on a leaf node this is equal to the number of edges mapped.
def arcs_left(marcs):
    arcsleft = 0
    for row in marcs:
        if np.sum(row) > 0:
            arcsleft += 1
    return arcsleft


def compute_marcs_given_maps(g1, g2, map_1_to_2):

    num_a_edges = g1.n_edges
    num_b_edges = g2.n_edges

    marcs = np.ones((num_a_edges, num_b_edges), dtype=np.int32)

    for v1, v2 in map_1_to_2.items():
        for e1 in g1.get_edges(v1):
            e2 = g2.get_edges(v2)
            # set any non-adjacent edges to zero
            for ej in range(num_b_edges):
                if ej not in e2:
                    marcs[e1][ej] = 0

    return marcs


class MCSResult:
    def __init__(self, map_1_to_2, map_2_to_1, num_edges):
        self.map_1_to_2 = map_1_to_2
        self.map_2_to_1 = map_2_to_1
        self.num_edges = num_edges


def mcgregor(g1, g2, map_1_to_2, map_2_to_1, mcs_result):

    marcs = compute_marcs_given_maps(g1, g2, map_1_to_2)
    num_edges = arcs_left(marcs)

    # leaf node, only update best num_edges found at leaf node
    if len(map_1_to_2) == g1.n_vertices:
        if mcs_result.num_edges <= num_edges:
            print("Found an equal or better complete map", num_edges, "mapping", map_1_to_2)
            mcs_result.map_1_to_2 = map_1_to_2
            mcs_result.map_2_to_1 = map_2_to_1
            mcs_result.num_edges = num_edges

        return

    # we are not a leaf node
    if num_edges <= mcs_result.num_edges:
        print("Skipping subtree")
        return

    for v1 in range(g1.n_vertices):

        if v1 in map_1_to_2:
            continue

        for v2 in range(g2.n_vertices):

            if v2 in map_2_to_1:
                continue

            # tbd avoid deepcopy and just pop v1,v2
            map_1_to_2_copy = copy.deepcopy(map_1_to_2)
            map_2_to_1_copy = copy.deepcopy(map_2_to_1)

            map_1_to_2_copy[v1] = v2
            map_2_to_1_copy[v2] = v1

            mcgregor(g1, g2, map_1_to_2_copy, map_2_to_1_copy, mcs_result)


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

    # map_1_to_2 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    # map_2_to_1 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    # marcs = compute_marcs_given_maps(g1, g2, map_1_to_2)
    # print(marcs)

    # assert 0

    # map_1_to_2 = {0: 0, 1: 1}
    # map_2_to_1 = {0: 0, 1: 1}
    map_1_to_2 = {}
    map_2_to_1 = {}
    num_edges = 0
    mcs_result = MCSResult(map_1_to_2, map_2_to_1, num_edges)

    mcgregor(g1, g2, map_1_to_2, map_2_to_1, mcs_result)
    # res = compute_marcs_given_maps(g1, g2, map_1_to_2)


if __name__ == "__main__":
    test_compute_marcs()
