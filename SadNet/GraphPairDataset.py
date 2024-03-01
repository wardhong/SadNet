import os
import numpy as np
import pandas as pd
import pickle
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
#from torch_geometric.loader import DataLoader
from sklearn.linear_model import LinearRegression
from torch_geometric import data as DATA
from torch_geometric.data import Batch
from torch.utils.data import Dataset
from torch_geometric.utils.convert import to_networkx
import torch

class GraphPairDataset(Dataset):
    def __init__(self, root='./processed/多batch版本/',dataset=None,data=None,loc='./processed/多batch版本/'):
        self.root = root+dataset+'.pt'
        self.dataset = dataset
        self.loc = loc+dataset+'.pt'
        if os.path.isfile(self.loc):
            print('Pre-processed data found: {}, loading ...'.format(self.loc))
            with open(self.loc, "rb",) as f:
                self.graph = pickle.load(f)
        else:
            print('Pre-processed data {} not found, doing processing...'.format(self.loc))
            if self.dataset == 'train':
                self.drugs_x = data.train_drugs_x
                self.prot_x = data.train_prot_x
                self.drugs_intra = data.train_drugs_intra
                self.prot_intra = data.train_prot_intra
                self.inter = data.train_inter
                self.y = data.train_y
                self.ami_index = data.train_ami_index
                self.ami_embedding = data.train_ami_embedding
                self.ami_dis = data.train_ami_dis
                self.ami_dis_li = data.train_ami_dis_li
                #self.drug_featrue = data.train_drug_feature
                self.edge_feat_l = data.train_edge_feat_l
                self.edge_feat_p = data.train_edge_feat_p
            elif self.dataset == 'valid':
                self.drugs_x = data.valid_drugs_x
                self.prot_x = data.valid_prot_x
                self.drugs_intra = data.valid_drugs_intra
                self.prot_intra = data.valid_prot_intra
                self.inter = data.valid_inter
                self.y = data.valid_y
                self.ami_index = data.valid_ami_index
                self.ami_embedding = data.valid_ami_embedding
                self.ami_dis = data.valid_ami_dis
                self.ami_dis_li = data.valid_ami_dis_li
                #self.drug_featrue = data.valid_drug_feature
                self.edge_feat_l = data.valid_edge_feat_l
                self.edge_feat_p = data.valid_edge_feat_p
            elif self.dataset == 'test':
                self.drugs_x = data.test_drugs_x
                self.prot_x = data.test_prot_x
                self.drugs_intra = data.test_drugs_intra
                self.prot_intra = data.test_prot_intra
                self.inter = data.test_inter
                self.y = data.test_y
                self.ami_index = data.test_ami_index
                self.ami_embedding = data.test_ami_embedding
                self.ami_dis = data.test_ami_dis
                self.ami_dis_li = data.test_ami_dis_li
                #self.drug_featrue = data.test_drug_feature
                self.edge_feat_l = data.test_edge_feat_l
                self.edge_feat_p = data.test_edge_feat_p

            else:#if self.dataset == 'test_2020_fast' or self.dataset == 'test_2013':
                self.drugs_x = data.drugs_x
                self.prot_x = data.prot_x
                self.drugs_intra = data.drugs_intra
                self.prot_intra = data.prot_intra
                self.inter = data.inter
                self.y = data.y
                self.ami_index = data.ami_index
                self.ami_embedding = data.ami_embedding
                self.ami_dis = data.ami_dis
                self.ami_dis_li = data.ami_dis_li
                #self.drug_featrue = data.drug_feature
                self.edge_feat_l = data.edge_feat_l
                self.edge_feat_p = data.edge_feat_p

            self.graph={}
            '''
            for i in range(len(self.y)):
                #delta = 1000 - len(self.ami_embedding[i])
                #emb = np.pad(self.ami_embedding[i].astype(np.float32),((0,delta),(0,0)),'constant',constant_values=0)
                GCNData = DATA.Data( x=torch.FloatTensor(np.zeros((991,52),dtype= np.int)),
                                     edge_index=torch.LongTensor(self.a_inter[i]).transpose(1, 0),
                                     y=torch.FloatTensor([self.y[i]]))
                GCNData1 = DATA.Data(x=torch.FloatTensor(self.drugs_x[i].astype(np.int64)),    #配体图
                                     edge_index=torch.LongTensor(self.drugs_intra[i]),
                                     edge_attr=torch.FloatTensor(self.edge_feat_l[i]))
                GCNData2 = DATA.Data(x=torch.FloatTensor(self.prot_x[i].astype(np.int64)),     #口袋图
                                     edge_index=torch.LongTensor(self.prot_intra[i]),
                                     edge_attr=torch.FloatTensor(self.edge_feat_p[i]))
                GCNData3 = DATA.Data(x=torch.FloatTensor(self.ami_embedding[i].astype(np.float32)),   #氨基酸图
                                     edge_index=torch.LongTensor(self.ami_index[i]).transpose(1, 0),
                                     edge_attr=torch.FloatTensor(self.ami_dis[i]).unsqueeze(1),
                                     ami_dis_li=torch.FloatTensor(self.ami_dis_li[i]),
                                     drug_feature=torch.FloatTensor(self.drug_featrue[i]).unsqueeze(0),
                                     emb=torch.FloatTensor(self.ami_embedding[i]).unsqueeze(0).unsqueeze(0))
                self.graph[i] = [GCNData,GCNData1,GCNData2,GCNData3]
            '''
            for i in range(len(self.y)):
                #GCNData = DATA.Data(x=torch.FloatTensor(self.inter[i]).transpose(1, 0))
                GCNData1 = DATA.Data(x=torch.FloatTensor(self.drugs_x[i]),  # 配体图
                                     edge_index=torch.LongTensor(self.drugs_intra[i]),
                                     edge_attr=torch.FloatTensor(self.edge_feat_l[i]),
                                     y=torch.FloatTensor([self.y[i]]),
                                     inter=torch.FloatTensor(self.inter[i]).unsqueeze(0))   #(3,n)->(n,3)保证dim=1维度相同
                GCNData2 = DATA.Data(x=torch.FloatTensor(self.prot_x[i]),  # 口袋图
                                     edge_index=torch.LongTensor(self.prot_intra[i]),
                                     edge_attr=torch.FloatTensor(self.edge_feat_p[i]))
                GCNData3 = DATA.Data(x=torch.FloatTensor(self.ami_embedding[i]),  # 氨基酸图
                                     edge_index=torch.LongTensor(self.ami_index[i]).transpose(1, 0),
                                     edge_attr=torch.FloatTensor(self.ami_dis[i]).unsqueeze(1),
                                     ami_dis_li=torch.FloatTensor(self.ami_dis_li[i]))
                                     #drug_feature=torch.FloatTensor(self.drug_featrue[i]).unsqueeze(0))
                                     #emb=torch.FloatTensor(self.ami_embedding[i]).unsqueeze(0).unsqueeze(0))

                self.graph[i] = [ GCNData1,GCNData2,GCNData3]
            with open(self.root, "wb") as f:
                pickle.dump(self.graph,f)

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        GCNData1, GCNData2, GCNData3 = self.graph[idx]
        return GCNData1, GCNData2, GCNData3

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def mae(y,f):
    mae = (np.abs(y-f)).mean()
    return mae

def sd(y,f):
    f,y = f.reshape(-1,1),y.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(f,y)
    y_ = lr.predict(f)
    sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
    return sd