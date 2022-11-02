import numpy as np
import simtk.unit
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D


def recenter_mol(mol):
    mol_copy = Chem.Mol(mol)
    conf = mol.GetConformer(0).GetPositions()
    center_conf = conf - np.mean(conf, axis=0)
    new_conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, pos in enumerate(np.asarray(center_conf)):
        new_conf.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol_copy.RemoveAllConformers()
    mol_copy.AddConformer(new_conf)
    return mol_copy



def get_mol_name(mol) -> str:
    """Return the title for the given mol"""
    return mol.GetProp("_Name")


def plot_atom_mapping_grid(mol_a, mol_b, core, show_idxs=False):

    mol_a_3d = recenter_mol(mol_a)
    mol_b_3d = recenter_mol(mol_b)

    atom_colors_a = {}
    atom_colors_b = {}
    for c_idx, ((a_idx, b_idx), rgb) in enumerate(zip(core, np.random.random((len(core), 3)))):
        atom_colors_a[int(a_idx)] = tuple(rgb.tolist())
        atom_colors_b[int(b_idx)] = tuple(rgb.tolist())

    if show_idxs:
        for atom in mol_a_3d.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
        for atom in mol_b_3d.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

    return Draw.MolsToGridImage(
        [mol_a_3d, mol_b_3d],
        molsPerRow=2,
        highlightAtomLists=[
            core[:, 0].tolist(),
            core[:, 1].tolist(),
        ],
        highlightAtomColors=[atom_colors_a, atom_colors_b],
        subImgSize=(600, 400),
        legends=[
            get_mol_name(mol_a) + " (3D)",
            get_mol_name(mol_b) + " (3D)",
        ],
        useSVG=True,
    )

