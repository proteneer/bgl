import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

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


def generate_good_rotations(mol_a, mol_b, num_rotations=3, max_rotations=1000):

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


def get_romol_bonds(mol):
    bonds = []

    for bond in mol.GetBonds():
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append((src, dst))

    return np.array(bonds, dtype=np.int32)


def get_core(mol_a, mol_b, ring_cutoff, chain_cutoff, timeout=10):
    """
    Find a reasonable core between two molecules. This function takes in two cutoff parameters:

    If either atom i or atom j then the dist(i,j) < ring_cutoff, otherwise dist(i,j) < chain_cutoff

    TBD: disallow SP3->SP2 hybridization changes.
    TBD: allow multiple maps to be returned.
    TBD: check for chiral restraints/parity in the C++ code directly.

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

    Returns
    -------
    np.array of shape (C,2)

    """

    bonds_a = get_romol_bonds(mol_a)
    bonds_b = get_romol_bonds(mol_b)
    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)

    predicate = np.zeros((mol_a.GetNumAtoms(), mol_b.GetNumAtoms()), dtype=np.int32)

    for idx, a_xyz in enumerate(conf_a):
        atom_i = mol_a.GetAtomWithIdx(idx)
        for jdx, b_xyz in enumerate(conf_b):
            atom_j = mol_b.GetAtomWithIdx(jdx)
            dij = np.linalg.norm(a_xyz - b_xyz)
            if atom_i.IsInRing() or atom_j.IsInRing():
                if dij < ring_cutoff:
                    predicate[idx][jdx] = 1
            else:
                if dij < chain_cutoff:
                    predicate[idx][jdx] = 1


    print("???")
    for r in predicate:
        print(r)
    print("???")

    core = mcgregor.mcs(predicate, bonds_a, bonds_b, timeout)

    return core


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
