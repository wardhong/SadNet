import Bio
import os
import numpy as np
print("Biopython v" + Bio.__version__)

from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
'''
with open('./processed/exclude',mode='r') as file:
    exclude = file.read().strip(',').split(',')
filepath = ["./raw_data/PDBbind_v2020_refined/refined-set"]
p_list = []
for f in filepath:
    Dir=os.listdir(f)
    for n in Dir:
        if n in exclude:
            continue
        if n!="index" and n!="readme" :
            p = f + "/" + n + "/" + n + "_protein.pdb"
            p_list.append(p)
'''
prot_name2016 = list()
with open('./processed/data_list/input_list_general.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split()
        prot_name2016.append(line[6])
p_list2016 = prot_name2016

prot_name2 = list()
affi ={}
with open('./processed/data_list/input_list_2020_new.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split()
        prot_name2.append(line[6])
        affi[line[6][-16:-12]]=line[5]
p_list = prot_name2

def get_seq():
    parser=PDBParser()
    file_path = './processed/seq_2020_fast.fasta'
    with open(file_path, mode='a', encoding='utf-8') as file_obj:
        for p in p_list:
            protein = parser.get_structure(p[-16:-12], p)
            ppb = Bio.PDB.PPBuilder()
            title = '>' + p[-16:-12] + '\n'
            file_obj.write(title)
            for pp in ppb.build_peptides(protein):
                seq = str(pp.get_sequence()) + '\n'
                file_obj.write(seq)

def get_emb():
    my_npz = np.load('C:/Users/hqs/Desktop/output_emb.npz',allow_pickle=True)
    dic = my_npz['10gs'].reshape(-1, 1)[0][0]['seq'][1:-1, :]
    print(dic)
    print(dic.shape)



#get_seq()
#get_emb()

#查重2020refined与2016general
new_pdb=[]
repeat_pdb=[]
seq_2016_general = []

parser=PDBParser()
for p in p_list2016:
    seq = ''
    protein = parser.get_structure(p[-16:-12], p)
    ppb = Bio.PDB.PPBuilder()
    for pp in ppb.build_peptides(protein):
        seq = seq + str(pp.get_sequence())
    seq_2016_general.append(seq)

for p in p_list:
    seq = ''
    protein = parser.get_structure(p[-16:-12], p)
    ppb = Bio.PDB.PPBuilder()
    for pp in ppb.build_peptides(protein):
        seq = seq + str(pp.get_sequence())
    if seq in seq_2016_general:
        repeat_pdb.append(p[-16:-12])
    else:
        new_pdb.append(p[-16:-12])
        f = "./raw_data/PDBbind_v2020_refined/refined-set/"
        n = p[-16:-12]
        l = f + n + "/" + n + "_ligand.mol2" + "\t" + f + "/" + n + "/" + n + "_ligand.pdb"
        pro = f + n + "/" + n + "_pocket.pdb" + "\t" + f + "/" + n + "/" + n + "_pocket_8.pdb"
        p2 = f + n + "/" + n + "_protein.pdb"
        with open('./processed/data_list/input_list_2020_1146', mode='a') as file:
            file.write(n + "\t" + pro +"\t" + l + "\t" +affi[n]+"\t"+p2+"\n")
'''
with open('./new_seq.txt', mode='w', encoding='utf-8') as file_obj:
    file_obj.write(str(new_pdb)+str(len(new_pdb))+'\n')
    file_obj.write(str(repeat_pdb)+str(len(repeat_pdb)))
    '''





