import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import networkx as nx
from collections import defaultdict

from scipy.spatial.distance import cdist
from scipy.stats import special_ortho_group

from bgl import mcgregor


def score_2d(conf, norm=2):
    # get the goodness of a 2D depiction
    # low_score = good, high_score = bad

    score = 0
    for idx, (x0, y0, _) in enumerate(conf):
        for x1, y1, _ in conf[idx + 1 :]:
            score += 1 / ((x0 - x1) ** norm + (y0 - y1) ** norm)

    return score / len(conf)


def generate_good_rotations(mol_a, mol_b, num_rotations=3, max_rotations=100):

    assert num_rotations < max_rotations

    # generate some good rotations so that the viewing angle is pleasant, (so clashes are minimized):
    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)

    scores = []
    rotations = []
    for _ in range(max_rotations):
        r = special_ortho_group.rvs(3)
        score_a = score_2d(conf_a @ r.T)
        score_b = score_2d(conf_b @ r.T)
        # take the bigger of the two scores
        scores.append(max(score_a, score_b))
        rotations.append(r)

    perm = np.argsort(scores)
    return np.array(rotations)[perm][:num_rotations]


def get_romol_conf(mol):
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf / 10  # from angstroms to nm


def get_jordan_center(mol):
    g = nx.Graph()
    for bond in mol.GetBonds():
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        g.add_edge(src, dst)
    return nx.center(g)[0]


def get_romol_bonds(mol):
    bonds = []

    for bond in mol.GetBonds():
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append((src, dst))

    return np.array(bonds, dtype=np.int32)


def get_cores(mol_a, mol_b, ring_cutoff, chain_cutoff, timeout, connected_core, max_cores):

    assert max_cores > 0

    if mol_a.GetNumAtoms() > mol_b.GetNumAtoms():
        all_cores, timed_out = _get_cores_impl(
            mol_b, mol_a, ring_cutoff, chain_cutoff, timeout, connected_core, max_cores
        )
        new_cores = []
        for core in all_cores:
            core = np.array([(x[1], x[0]) for x in core], dtype=core.dtype)
            new_cores.append(core)
        return new_cores, timed_out
    else:
        all_cores, timed_out = _get_cores_impl(
            mol_a, mol_b, ring_cutoff, chain_cutoff, timeout, connected_core, max_cores
        )
        return all_cores, timed_out


def bfs(g, atom):
    depth = 0
    cur_layer = [atom]
    levels = {}
    while len(levels) != g.GetNumAtoms():
        next_layer = []
        for layer_atom in cur_layer:
            levels[layer_atom.GetIdx()] = depth
            for nb_atom in layer_atom.GetNeighbors():
                if nb_atom.GetIdx() not in levels:
                    next_layer.append(nb_atom)
        cur_layer = next_layer
        depth += 1
    levels_array = [-1] * g.GetNumAtoms()
    for i, l in levels.items():
        levels_array[i] = l
    return levels_array


def reorder_atoms(mol):
    center_idx = get_jordan_center(mol)
    center_atom = mol.GetAtomWithIdx(center_idx)
    levels = bfs(mol, center_atom)
    perm = np.argsort(levels)
    new_mol = Chem.RenumberAtoms(mol, perm.tolist())
    return new_mol, perm


def _get_cores_impl(mol_a, mol_b, ring_cutoff, chain_cutoff, timeout, connected_core, max_cores):
    """
    Find a reasonable core between two molecules. This function takes in two cutoff parameters:

    If either atom i or atom j then the dist(i,j) < ring_cutoff, otherwise dist(i,j) < chain_cutoff

    Additional notes
    ----------------
    1) The returned cores are sorted in increasing order based on the rmsd of the alignment.
    2) The number of cores atoms may vary slightly, but the number of mapped edges are the same.

    TBD: disallow SP3->SP2 hybridization changes.
    TBD: check for chiral restraints/parity

    Parameters
    ----------
    mol_a: Chem.Mol
        Input molecule a. Must have a conformation.

    mol_b: Chem.Mol
        Input molecule b. Must have a conformation.

    ring_cutoff: float
        The distance cutoff that ring atoms must satisfy.

    chain_cutoff: float
        The distance cutoff that non-ring atoms must satisfy.

    timeout: int
        Maximum number of seconds before returning.

    connected_core: bool
        Set to True to only keep the largest connected
        subgraph in the mapping. The definition of connected
        here is different from McGregor. Here it means there
        is a way to reach the mapped atom without traversing
        over a non-mapped atom.

    max_cores: int or float
        maximum number of maximal cores to store, this can be an +np.inf if you want
        every core - when set to 1 this enables a faster predicate that allows for more pruning.

    Returns
    -------
    np.array of shape (C,2)


    """
    mol_a, perm = reorder_atoms(mol_a)  # UNINVERT

    bonds_a = get_romol_bonds(mol_a)
    bonds_b = get_romol_bonds(mol_b)
    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)

    priority_idxs = []  # ordered list of atoms to consider

    for idx, a_xyz in enumerate(conf_a):
        atom_i = mol_a.GetAtomWithIdx(idx)
        dijs = []

        allowed_idxs = set()
        for jdx, b_xyz in enumerate(conf_b):
            atom_j = mol_b.GetAtomWithIdx(jdx)
            dij = np.linalg.norm(a_xyz - b_xyz)
            dijs.append(dij)
            if atom_i.IsInRing() or atom_j.IsInRing():
                if dij < ring_cutoff:
                    allowed_idxs.add(jdx)
            else:
                if dij < chain_cutoff:
                    allowed_idxs.add(jdx)

        final_idxs = []
        for idx in np.argsort(dijs):
            if idx in allowed_idxs:
                final_idxs.append(idx)

        priority_idxs.append(final_idxs)

    # for idx, r in enumerate(priority_idxs):
    # print("atom", idx, "in mol_a has", len(r), "allowed matches")

    n_a = len(conf_a)
    n_b = len(conf_b)

    all_cores, timed_out = mcgregor.mcs(n_a, n_b, priority_idxs, bonds_a, bonds_b, timeout, max_cores)

    # undo the sort
    for core in all_cores:
        inv_core = []
        for atom in core[:, 0]:
            inv_core.append(perm[atom])
        core[:, 0] = inv_core

    return all_cores, timed_out

    if connected_core:
        all_cores = remove_disconnected_components(mol_a, mol_b, all_cores)

    dists = []
    # rmsd, note that len(core) is not the same, only the number of edges is
    for core in all_cores:
        r_i = conf_a[core[:, 0]]
        r_j = conf_b[core[:, 1]]
        r2_ij = np.sum(np.power(r_i - r_j, 2))
        rmsd = np.sqrt(r2_ij / len(core))
        dists.append(rmsd)

    sorted_cores = []
    for p in np.argsort(dists):
        sorted_cores.append(all_cores[p])

    return sorted_cores, timed_out


def remove_disconnected_components(mol_a, mol_b, cores):
    """
    Iterate over the cores and filter out the disconnected
    mappings from each core. Return a list of cores that
    have the largest remaining mapping.
    """
    filtered_cores = []
    for core in cores:
        core_a = list(core[:, 0])
        core_b = list(core[:, 1])
        g_mol_a = mol_to_mapped_bonds_graph(mol_a, core_a)
        g_mol_b = mol_to_mapped_bonds_graph(mol_b, core_b)

        largest_cc_a = max(nx.connected_components(g_mol_a), key=len)
        largest_cc_b = max(nx.connected_components(g_mol_b), key=len)

        # pick the smaller connected mapping
        new_core_idxs = []
        if len(largest_cc_a) < len(largest_cc_b):
            # mol_a has the smaller cc
            for atom_idx in largest_cc_a:
                new_core_idxs.append(core_a.index(atom_idx))
        else:
            # mol_b has the smaller cc
            for atom_idx in largest_cc_b:
                new_core_idxs.append(core_b.index(atom_idx))
        new_core = core[new_core_idxs]
        filtered_cores.append(new_core)

    filtered_cores_by_size = defaultdict(list)
    for core in filtered_cores:
        filtered_cores_by_size[len(core)].append(core)

    # Return the largest core(s)
    return filtered_cores_by_size[max(filtered_cores_by_size.keys())]


def mol_to_mapped_bonds_graph(mol, mapped_idxs) -> nx.Graph:
    """
    Convert a given mol to networkx graph, keeping only
    the bonds that are mapped in the mapped_idxs.

    Parameters
    ----------
    mol:
        Molecule to convert.

    mapped_idxs: List of int
        The core atom idxs for the given molecule only.
    """
    g_mol = nx.Graph()
    mapped_set = set(mapped_idxs)

    for bond in mol.GetBonds():
        atom_i = bond.GetBeginAtomIdx()
        atom_j = bond.GetEndAtomIdx()
        if atom_i in mapped_set and atom_j in mapped_set:
            g_mol.add_edge(atom_i, atom_j)

    return g_mol


def recenter_mol(mol):

    conf = mol.GetConformer(0).GetPositions()
    center_conf = conf - np.mean(conf, axis=0)
    new_conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, pos in enumerate(np.asarray(center_conf)):
        new_conf.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))

    mol_copy = Chem.Mol(mol)
    mol_copy.RemoveAllConformers()
    mol_copy.AddConformer(new_conf)
    return mol_copy


def randomly_rotate_mol(mol, rotation_matrix):
    mol = recenter_mol(mol)
    conf = mol.GetConformer(0).GetPositions()

    new_conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, pos in enumerate(np.asarray(conf)):
        rot_pos = rotation_matrix @ pos
        new_conf.SetAtomPosition(idx, (float(rot_pos[0]), float(rot_pos[1]), float(rot_pos[2])))

    mol_copy = Chem.Mol(mol)
    mol_copy.RemoveAllConformers()
    mol_copy.AddConformer(new_conf)
    return mol_copy


def get_mol_name(mol) -> str:
    """Return the title for the given mol"""
    return mol.GetProp("_Name")


def plot_atom_mapping_grid(mol_a, mol_b, core, num_rotations=5):

    mol_a_3d = recenter_mol(mol_a)
    mol_b_3d = recenter_mol(mol_b)

    extra_rotations = generate_good_rotations(mol_a, mol_b, num_rotations)

    extra_mols = []

    atom_colors_a = {}
    atom_colors_b = {}
    for (a_idx, b_idx), rgb in zip(core, np.random.random((len(core), 3))):
        atom_colors_a[int(a_idx)] = tuple(rgb.tolist())
        atom_colors_b[int(b_idx)] = tuple(rgb.tolist())

    hals = [core[:, 0].tolist(), core[:, 1].tolist()]
    hacs = [atom_colors_a, atom_colors_b]

    for rot in extra_rotations:
        extra_mols.append(randomly_rotate_mol(mol_a_3d, rot))
        extra_mols.append(randomly_rotate_mol(mol_b_3d, rot))
        hals.append(core[:, 0].tolist())
        hals.append(core[:, 1].tolist())
        hacs.append(atom_colors_a)
        hacs.append(atom_colors_b)

    num_mols = len(extra_mols) + 2

    legends = []
    for _ in range(num_mols - 2):
        legends.append("")
    legends.append(get_mol_name(mol_a) + " (3D)")
    legends.append(get_mol_name(mol_b) + " (3D)")

    return Draw.MolsToGridImage(
        [mol_a_3d, mol_b_3d, *extra_mols],
        molsPerRow=2,
        highlightAtomLists=hals,
        highlightAtomColors=hacs,
        subImgSize=(400, 300),
        legends=legends,
        useSVG=True,
    )
