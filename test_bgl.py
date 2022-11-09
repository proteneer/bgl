from rdkit import Chem
import atom_mapping

mols = Chem.SDMolSupplier("datasets/hif2a.sdf", removeHs=False)
mols = [m for m in mols]

for idx, mol_a in enumerate(mols):
    for mol_b in mols[idx + 1 :]:

        core = atom_mapping.get_core(mol_a, mol_b, ring_cutoff=0.1, chain_cutoff=2.0)
        res = atom_mapping.plot_atom_mapping_grid(mol_a, mol_b, core, num_rotations=5)

        def get_mol_name(mol) -> str:
            """Return the title for the given mol"""
            return mol.GetProp("_Name")

        with open(f"atom_mapping_{get_mol_name(mol_a)}_to_{get_mol_name(mol_b)}.svg", "w") as fh:
            fh.write(res)

        print(core)
