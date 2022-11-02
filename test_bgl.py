from build import bgl_wrapper

from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

import atom_mapping

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

mols = Chem.SDMolSupplier(
    "/Users/yzhao/Code/timemachine/timemachine/testsystems/data/ligands_40.sdf",
    removeHs=False
)

mol_a = mols[0]
mol_b = mols[1]

bonds_a = get_romol_bonds(mol_a)
bonds_b = get_romol_bonds(mol_b)
conf_a = get_romol_conf(mol_a)
conf_b = get_romol_conf(mol_b)

dij = cdist(conf_a, conf_b)
res = dij < 0.2
res = res.astype(np.int32)

print("num_atoms_a", mol_a.GetNumAtoms(), "num_bonds_a", len(bonds_a))
print("num_atoms_b", mol_b.GetNumAtoms(), "num_bonds_b", len(bonds_b))
print("dij shape", res.shape)

for line in res:
    print(line)

timeout = 10

core = bgl_wrapper.mcs(res, bonds_a, bonds_b, timeout)

res = atom_mapping.plot_atom_mapping_grid(mol_a, mol_b, core)


def get_mol_name(mol) -> str:
    """Return the title for the given mol"""
    return mol.GetProp("_Name")

with open(f"atom_mapping_{get_mol_name(mol_a)}_to_{get_mol_name(mol_b)}.svg", "w") as fh:
    fh.write(res)

print(core)