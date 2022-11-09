from rdkit import Chem
from bgl import atom_mapping

mols = Chem.SDMolSupplier("datasets/hif2a.sdf", removeHs=False)
mols = [m for m in mols]

for idx, mol_a in enumerate(mols):
    for mol_b in mols[idx + 1 :]:

        core = atom_mapping.get_core(mol_b, mol_a, ring_cutoff=0.1, chain_cutoff=0.0)
        # core = atom_mapping.get_core(mol_b, mol_a, ring_cutoff=0.1, chain_cutoff=0.2)
        res = atom_mapping.plot_atom_mapping_grid(mol_b, mol_a, core, num_rotations=5)

        def get_mol_name(mol) -> str:
            """Return the title for the given mol"""
            return mol.GetProp("_Name")

        with open(f"atom_mapping_{get_mol_name(mol_a)}_to_{get_mol_name(mol_b)}.svg", "w") as fh:
            fh.write(res)

        print("wrote core", core)
