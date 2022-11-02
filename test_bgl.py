from build import bgl_wrapper

from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import networkx as nx

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
    # removeHs=True
)

# mols = [
#     Chem.MolFromSmiles("C1CCC1CCCC"),
#     Chem.MolFromSmiles("CC1CCCC1CCC"),
# ]
# for m in mols:
    # AllChem.EmbedMolecule(m)

bonds_a = get_romol_bonds(mols[0])
bonds_b = get_romol_bonds(mols[1])
conf_a = get_romol_conf(mols[0])
conf_b = get_romol_conf(mols[1])

bgl_wrapper.mcs(conf_a, bonds_a, conf_b, bonds_b, 0.2)