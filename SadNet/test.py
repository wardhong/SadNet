import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from GraphPairDataset import *
from training import *
from torch_geometric.data import Data, Batch
import h5py
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
from model.mul_intra import *
list_2013 = []
with open('./raw_data/casf2013/index/2013_core_data.lst') as file:
    for line in file.readlines():
        if line[0] != '#':
            line = line.split()
            list_2013.append(line[0])
test_2013_id = []

test_pdb, test_id = [], []
val_pdb, val_id = [], []
train_pdb, train_id = [], []
test_2020, test_id_2020 = [], []
external_test_data = False

if external_test_data == True:
    dataframe = pd.read_pickle('./5A_2020_test.pkl')       #选择提取哪些数据
    file = open('./raw_data/2020_test.txt')
    for line in file.readlines():
        pdbid = line.strip()
        test_2020.append(pdbid)
    for i in range(len(dataframe)):
        if dataframe.iloc[i]['code'] in test_2020:
            test_id_2020.append(i)
    print(len(test_id_2020))
    with open('split_idx_fast_2020', 'w') as f:
        f.write('{}'.format(test_id_2020))
else:
    dataframe = pd.read_pickle('./6A_refined.pkl')

    # 取出核心集pdb号
    file = open('./raw_data/test_refined.txt')
    for line in file.readlines():
        pdbid = line.strip()
        test_pdb.append(pdbid)

    file = open('./raw_data/val_refined.txt')
    for line in file.readlines():
        pdbid = line.strip()
        val_pdb.append(pdbid)

    file = open('./raw_data/train_refined.txt')
    for line in file.readlines():
        pdbid = line.strip()
        train_pdb.append(pdbid)
    for i in range(len(dataframe)):
        if dataframe.iloc[i]['code'] in test_pdb:
            test_id.append(i)
        elif dataframe.iloc[i]['code'] in list_2013:
            test_2013_id.append(i)
        elif dataframe.iloc[i]['code'] in train_pdb:
            train_id.append(i)
        elif dataframe.iloc[i]['code'] in val_pdb:
            val_id.append(i)

    print(len(train_id), len(val_id), len(test_id), len(test_2013_id))
    with open('split_idx_refined_fast_del2013_6A', 'w') as f:
        f.write('[{},{},{}]'.format(train_id, val_id, test_id))

"""
    def forward(self, data1, data2, data3):
        x1, drug_intra, drug_edge_attr, batch1 = data1.x, data1.edge_index, data1.edge_attr, data1.batch
        x2, prot_intra, prot_edge_attr, batch2 = data2.x, data2.edge_index, data2.edge_attr, data2.batch

        x1 = self.node_embed(x1)
        x2 = self.node_embed(x2)

        x1 = self.conv1(x1, drug_intra, drug_edge_attr)
        x1 = self.relu(x1)
        x1, drug_intra, drug_edge_attr, batch1, _, _ = self.SAGPooling1(x1, drug_intra,drug_edge_attr,batch1)
        x11 = torch.cat([global_mean_pool(x1, batch1), global_max_pool(x1, batch1)], dim=1)
        x1 = self.conv2(x1, drug_intra, drug_edge_attr)
        x1 = self.relu(x1)
        x1, drug_intra, drug_edge_attr, batch1, _, _ = self.SAGPooling2(x1, drug_intra, drug_edge_attr, batch1)
        x12 = torch.cat([global_mean_pool(x1, batch1), global_max_pool(x1, batch1)], dim=1)
        x1 = self.conv3(x1, drug_intra, drug_edge_attr)
        x1 = self.relu(x1)
        x1, drug_intra, drug_edge_attr, batch1, _, _ = self.SAGPooling3(x1, drug_intra, drug_edge_attr, batch1)
        x13 = torch.cat([global_mean_pool(x1, batch1), global_max_pool(x1, batch1)], dim=1)

        x_drug = x11 + x12 + x13
        x_l, mask_l = utils.to_dense_batch(x1, batch1)
        #x_l = torch.unsqueeze(x1, 1)

        x2 = self.conv1(x2, prot_intra, prot_edge_attr)
        x2 = self.relu(x2)
        x2, prot_intra, prot_edge_attr, batch2, _, _ = self.SAGPooling1(x2, drug_intra, drug_edge_attr,batch2)
        x2 = self.conv2(x2, prot_intra, prot_edge_attr)
        x2 = self.relu(x2)
        x2, prot_intra, prot_edge_attr, batch2, _, _ = self.SAGPooling2(x2, drug_intra, drug_edge_attr, batch2)
        x2 = self.conv3(x2, prot_intra, prot_edge_attr)
        x2 = self.relu(x2)
        #x2 = global_mean_pool(x2, batch2)
        x_p, mask = utils.to_dense_batch(x2, batch2)
        mask = torch.unsqueeze(torch.unsqueeze(mask, 1), 1)
        #x22 = global_mean_pool(x2, batch2)
        x2 = self.ln(self.dropout(self.ca(x_l, x_p, x_p, None)))
        x2 = torch.mean(x2, dim=1)
        x = self.relu(torch.cat((x_drug,x2),1))


        h1 = self.fc1(x)
        x = self.relu(h1)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out, h1
"""