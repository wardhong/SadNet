import Bio
import os
print("Biopython v" + Bio.__version__)

from Bio.PDB import PDBParser
from Bio.PDB import PPBuilder
from Bio.PDB import PDBIO

#custom select
class Select():
    def accept_model(self, model):
        return True
    def accept_chain(self, chain):
        return True
    def accept_residue(self, residue):
        return True
    def accept_atom(self, atom):
        #print("atom id:" + atom.get_id())
        #print("atom name:" + atom.get_name())
        if atom.get_name() == 'CA':
            #print("True")
            return True
        else:
            return False

# Parse and get basic information
#filepath = "./processed/data_list/input_list.txt"
#filepath = './raw_data/PDBbind_v2020_refined/refined-set'
p_list = []
with open('./processed/data_list/input_list.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split()
        p_list.append(line[6])


parser=PDBParser()
'''
for p in ['./raw_data/refined-set/1a4k/1a4k_protein.pdb']:
    protein = parser.get_structure(p[-16:-12], p)
    print("Protein name: " + p[-16:-12])
    print("Model: " + str(protein[0]))

    #initialize IO
    io=PDBIO()

    #write to output file
    io.set_structure(protein)
    io.save('./processed/protein_αC/{}_protein_c.pdb'.format(p[-16:-12]), Select())
'''

    #get seq
seqfile = open('./processed/seq.fasta','w')
for p in p_list:
    protein = parser.get_structure(p[-16:-12], p)
    ppb = PPBuilder()
    seqfile.write('>'+p[-16:-12]+'\n')
    for pp in ppb.build_peptides(protein):
        seqfile.write(str(pp.get_sequence())+'\n')
