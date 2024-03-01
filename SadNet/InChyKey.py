from rdkit import Chem
from rdkit.Chem import rdmolops
import os
import h5py
import numpy as np
import pubchempy as pcp

def get_inchikey():
    index = 0
    code_name = list()
    li_name = list()
    exclude500 = list()
    inchikey = list()
    if os.path.isfile('./raw_data/InChiKey.txt'):
        with open('./raw_data/InChiKey.txt','r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                code_name.append(line[0])
                inchikey.append(line[1])
    else:
        file=open('./raw_data/InChiKey.txt','w')
        with open('./input_list.txt') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                code_name.append(line[0])
                li_name.append(line[3])
        with open('./processed/exclude500.txt','r') as ex:
            line = ex.read()
            exclude500 = line.split(',')

        for i in range(len(li_name)):
            if code_name[i] not in exclude500:
                index = index + 1
                print(str(index)+'/'+ str(len(li_name)-len(exclude500)+1))
                mol_li = Chem.MolFromMol2File(li_name[i])
                #mol_li = Chem.MolToSmiles(mol_li)
                try:
                    key = Chem.inchi.MolToInchiKey(mol_li).encode()
                except:
                    print(mol_li)
                    key = code_name[i] + '无法转化为inchikey'

                inchikey.append(key)
                file.write(code_name[i] + '\t' + str(key) + '\n')
    return inchikey
def get_cid():
    index = 0
    code_name = list()
    li_name = list()
    exclude500 = list()
    cid = list()
    file = open('./raw_data/cid.txt', 'w')
    with open('./input_list.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            code_name.append(line[0])
            li_name.append(line[4])
    with open('./processed/exclude500.txt', 'r') as ex:
        line = ex.read()
        exclude500 = line.split(',')

    for i in range(len(li_name)):
        if code_name[i] not in exclude500:
            index = index + 1
            print(str(index) + '/' + str(len(li_name) - len(exclude500) + 1))
            mol_li = Chem.MolFromMolFile(li_name[i])

            try:
                key = Chem.inchi.MolToInchiKey(mol_li).encode()
            except:
                print(mol_li)
                key = code_name[i] + '无法转化为inchikey'

            cid.append(key)
            file.write(code_name[i] + '\t' + str(key) + '\n')
    return cid

def read_h5():
    f = h5py.File('D:/安装包/A5.h5', 'r')
    for group in f.keys():
        print(group)
        # 根据一级组名获得其下面的组
        group_read = f[group]
        # 遍历该一级组下面的子组
        data = np.array(group_read)
        print(data.shape)
    print(f.keys())
    keys = np.array(f['keys'])
    value = np.array(f['V'])
    return keys,value
#get_cid()
li_chi = get_inchikey()
keys, value = read_h5()
keys = keys.tolist()
k1 = keys[0]
index = []
no_index_key = []
for i in li_chi:
    try:
        index.append(keys.index(i))
    except:
        no_index_key.append(i)
print(len(index))
print(index)
print(len(no_index_key))
print(no_index_key)
