from rdkit import Chem
from bgl import atom_mapping

mols = Chem.SDMolSupplier("datasets/hif2a.sdf", removeHs=False)
mols = [m for m in mols]

for idx, mol_a in enumerate(mols):
    for mol_b in mols[idx + 1 :]:

        print(f"{mol_b.GetProp('_Name')} -> {mol_a.GetProp('_Name')}")
        # all_cores = atom_mapping.get_core(mol_b, mol_a, ring_cutoff=0.1, chain_cutoff=0.0)
        all_cores = atom_mapping.get_core(mol_b, mol_a, ring_cutoff=0.1, chain_cutoff=0.2)
        for core_idx, core in enumerate(all_cores):
            res = atom_mapping.plot_atom_mapping_grid(mol_b, mol_a, core, num_rotations=5)

            def get_mol_name(mol) -> str:
                """Return the title for the given mol"""
                return mol.GetProp("_Name")

            # check ordering of mol_a and mol_b
            with open(f"atom_mapping_{get_mol_name(mol_b)}_to_{get_mol_name(mol_a)}_core_{core_idx}.svg", "w") as fh:
                fh.write(res)

            print(sorted(core.tolist()))

        assert 0

        # print("wrote core", core)

        # assert 0
