from rdkit import Chem
from bgl import atom_mapping
from line_profiler import LineProfiler

mols = Chem.SDMolSupplier("datasets/hif2a.sdf", removeHs=False)
mols = [m for m in mols]


def get_mol_name(mol) -> str:
    """Return the title for the given mol"""
    return mol.GetProp("_Name")


def run():

    for idx, mol_a in enumerate(mols[:5]):
        for mol_b in mols[idx + 1 : 5]:

            all_cores, _ = atom_mapping.get_cores(
                mol_a,
                mol_b,
                ring_cutoff=0.1,
                chain_cutoff=0.2,
                max_visits=1e7,  # 10 million max nodes to visit
                connected_core=False,
                max_cores=1000000,
            )
            # note that this is probably the bottleneck for hif2a
            # for core_idx, core in enumerate(all_cores):
            #     res = atom_mapping.plot_atom_mapping_grid(mol_a, mol_b, core, num_rotations=5)

            #     with open(
            #         f"atom_mapping_{get_mol_name(mol_a)}_to_{get_mol_name(mol_b)}_core_{core_idx}.svg", "w"
            #     ) as fh:
            #         fh.write(res)

            print(
                f"{mol_a.GetProp('_Name')} -> {mol_b.GetProp('_Name')} has {len(all_cores)} cores of size {len(all_cores[0])}"
            )


import time

if __name__ == "__main__":

    start_time = time.time()
    run()
    print(time.time() - start_time)

    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    # run()
    # pr.disable()
    # from pstats import Stats
    # stats = Stats(pr)
    # stats.sort_stats("tottime").print_stats(10)

    # lp = LineProfiler()
    # lp.add_function(mcgregor.recursion)
    # lp_wrapper = lp(run)
    # lp_wrapper()
    # lp.print_stats()
