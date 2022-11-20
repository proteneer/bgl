# maximum common subgraph routines based off of the mcgregor paper
import numpy as np
import copy
import time

# when computed on a leaf node this is equal to the number of edges mapped.
# def arcs_left(marcs):
    # return np.count_nonzero(marcs)  # slow

def arcs_left(marcs): 
    return np.sum(np.any(marcs, 1))

# def arcs_left(marcs):
    # return np.count_nonzero(marcs)  # slow

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


# def refine_marcs(g1, g2, new_v1, new_v2, marcs, num_edges):
#     """
#     return vertices that have changed
#     """
#     new_marcs = copy.copy(marcs)  # [m for m in marcs]
#     removed_edges = 0

#     # process rows
#     for e1 in g1.get_edges(new_v1):
#         orig = new_marcs[e1]
#         if new_v2 != UNMAPPED:
#             new_val = orig & g2.get_edges_as_vector(new_v2)
#         else:
#             new_val = 0

#         new_marcs[e1] = new_val

#         if orig > 0 and new_val == 0:
#             removed_edges += 1

#     return new_marcs, num_edges - removed_edges

def refine_marcs(g1, g2, new_v1, new_v2, marcs):
    """
    return vertices that have changed
    """
    new_marcs = copy.copy(marcs)  # [m for m in marcs]
    # process rows
    for e1 in g1.get_edges(new_v1):
        if new_v2 != UNMAPPED:
            new_marcs[e1] &= g2.get_edges_as_vector(new_v2)
        else:
            new_marcs[e1] = np.zeros(g2.n_edges)

    # process columns
    if new_v2 != UNMAPPED:
        for e2 in g2.get_edges(new_v2):
            new_marcs[:, e2] &= g1.get_edges_as_vector(new_v1)

    return new_marcs


class MCSResult:
    def __init__(self):
        self.all_maps = []
        self.num_edges = 0
        self.timed_out = False
        self.nodes_visited = 0

def convert_matrix_to_bits(arr):
    res = []
    for row in arr:
        seq = "".join([str(x) for x in row.tolist()])
        res.append(int(seq, 2))
    return res

class Graph:
    def __init__(self, n_vertices, edges):
        self.n_vertices = n_vertices
        self.n_edges = len(edges)
        self.edges = edges

        cmat = np.zeros((n_vertices, n_vertices), dtype=np.int32)
        for i, j in edges:
            cmat[i][j] = 1
            cmat[j][i] = 1

        self.cmat = cmat

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

        # boolean version of ve_matrix
        self.ve_bits = convert_matrix_to_bits(self.ve_matrix)

    def get_neighbors(self, vertex):
        return self.lol_vertices[vertex]

    def get_edges(self, vertex):
        return self.lol_edges[vertex]

    def get_edges_as_vector(self, vertex):
        return self.ve_matrix[vertex]

    def get_edges_as_int(self, vertex):
        return self.ve_bits[vertex]


def max_tree_size(priority_list):
    cur_layer_size = 1
    layer_sizes = [cur_layer_size]
    for neighbors in priority_list:
        cur_layer_size *= len(neighbors)
        layer_sizes.append(cur_layer_size)
    return sum(layer_sizes)


def build_predicate_matrix(n_a, n_b, priority_idxs):
    assert len(priority_idxs) == n_a
    pmat = np.zeros((n_a, n_b), dtype=np.int32)
    for idx, jdxs in enumerate(priority_idxs):
        for jdx in jdxs:
            pmat[idx][jdx] = 1
    return pmat

def mcs(n_a, n_b, priority_idxs, bonds_a, bonds_b, timeout, max_cores):

    assert n_a <= n_b

    g_a = Graph(n_a, bonds_a)
    g_b = Graph(n_b, bonds_b)

    predicate = build_predicate_matrix(n_a, n_b, priority_idxs)
    marcs = initialize_marcs_given_predicate(g_a, g_b, predicate)

    priority_idxs = tuple(tuple(x) for x in priority_idxs)
    # marcs = convert_matrix_to_bits(marcs)
    start_time = time.time()

    map_a_to_b = [UNMAPPED] * n_a
    map_b_to_a = [UNMAPPED] * n_b
    mcs_result = MCSResult()

    recursion(
        g_a, g_b, map_a_to_b, map_b_to_a, 0, marcs, mcs_result, priority_idxs, start_time, timeout, max_cores
    )
    all_cores = []

    print(f"====[NODES VISITED {mcs_result.nodes_visited} | CORE_SIZE {len([x != UNMAPPED for x in mcs_result.all_maps[0]])} | NUM_EDGES {mcs_result.num_edges} | time taken: {time.time()-start_time} | time out? {mcs_result.timed_out}]=====")

    for atom_map_1_to_2 in mcs_result.all_maps:
        core = []
        for a, b in enumerate(atom_map_1_to_2):
            if b != UNMAPPED:
                core.append((a, b))
        core = np.array(sorted(core))
        all_cores.append(core)

    return all_cores, mcs_result.timed_out


def atom_map_add(map_1_to_2, map_2_to_1, idx, jdx):
    map_1_to_2[idx] = jdx
    map_2_to_1[jdx] = idx


def atom_map_pop(map_1_to_2, map_2_to_1, idx, jdx):
    map_1_to_2[idx] = UNMAPPED
    map_2_to_1[jdx] = UNMAPPED


def recursion(
    g1,
    g2,
    atom_map_1_to_2,
    atom_map_2_to_1,
    layer,
    marcs,
    mcs_result,
    priority_idxs,
    start_time,
    timeout,
    max_cores,
):
    num_edges = arcs_left(marcs)
    # print("layer", layer, "num_edges", num_edges)

    mcs_result.nodes_visited += 1

    if time.time() - start_time > timeout:
        mcs_result.timed_out = True
        return

    n_a = g1.n_vertices

    # every atom has been mapped
    if layer == n_a:
        # assert 0
        if mcs_result.num_edges < num_edges:
            mcs_result.all_maps = [copy.copy(atom_map_1_to_2)]
            mcs_result.num_edges = num_edges
        elif mcs_result.num_edges == num_edges and len(mcs_result.all_maps) < max_cores:
            mcs_result.all_maps.append(copy.copy(atom_map_1_to_2))
            pass
        return

    # (ytz): note equality, since we want redundant edges, if we don't, then there is
    # another ~3x speed-up we can get if we *only* care about getting a single largest mcs

    if max_cores == 1:
        if num_edges <= mcs_result.num_edges:
            return
    else:
        if num_edges < mcs_result.num_edges:
            return

    # prioritize mappings that maximize edge count
    for jdx in priority_idxs[layer]:
        if atom_map_2_to_1[jdx] == UNMAPPED:  # optimize later
            atom_map_add(atom_map_1_to_2, atom_map_2_to_1, layer, jdx)
            new_marcs = refine_marcs(g1, g2, layer, jdx, marcs)
            recursion(
                g1,
                g2,
                atom_map_1_to_2,
                atom_map_2_to_1,
                layer + 1,
                new_marcs,
                mcs_result,
                priority_idxs,
                start_time,
                timeout,
                max_cores,
            )
            atom_map_pop(atom_map_1_to_2, atom_map_2_to_1, layer, jdx)

    # always allow for explicitly not mapping layer atom
    new_marcs = refine_marcs(g1, g2, layer, UNMAPPED, marcs)
    recursion(
        g1,
        g2,
        atom_map_1_to_2,
        atom_map_2_to_1,
        layer + 1,
        new_marcs,
        mcs_result,
        priority_idxs,
        start_time,
        timeout,
        max_cores,
    )


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
