import pymol
import os

def gen_pock(data_dir, cutoff):
    complex_dir = [name for name in os.listdir(data_dir) if len(name) == 4]
    for pid in complex_dir:
        complex_path = os.path.join(data_dir, pid)
        ligand_path = os.path.join(complex_path, f'{pid}_ligand.mol2')
        protein_path= os.path.join(complex_path, f'{pid}_protein.pdb')

        if os.path.exists(os.path.join(complex_path, f'pocket_{cutoff}A.pdb')):
            continue

        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(ligand_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {pid}_ligand around {cutoff}')
        pymol.cmd.save(os.path.join(complex_path, f'{pid}_pocket_{cutoff}.pdb'), 'Pocket')
        pymol.cmd.delete('all')


def transligmol2_to_pdb(data_dir):
    complex_dir = [name for name in os.listdir(data_dir) if len(name) == 4]
    for pid in complex_dir:
        complex_path = os.path.join(data_dir, pid)
        ligand_mol2path = os.path.join(complex_path, f'{pid}_ligand.mol2')
        ligand_pdbpath = os.path.join(complex_path, f'{pid}_ligand.pdb')
        if(os.path.exists(ligand_pdbpath)):
            continue
        else:
            print('covert {}_ligand to pdbfile'.format(pid))

            os.system(f'obabel {ligand_mol2path} -O {ligand_pdbpath} -d')

#transligmol2_to_pdb('./raw_data/casf2013/')
gen_pock('./raw_data/PDBbind_v2020_refined/refined-set/', 4)
