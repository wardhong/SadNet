import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphSAGE,GATConv, global_max_pool , global_mean_pool, GINEConv
import math
from torch_geometric.data import Batch
import networkx as nx
import matplotlib.pyplot as plt

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim*n_heads)
        self.w_k = nn.Linear(hid_dim, hid_dim*n_heads)
        self.w_v = nn.Linear(hid_dim, hid_dim*n_heads)
        self.fc = nn.Linear(hid_dim*n_heads, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = math.sqrt(hid_dim)

    def forward(self, query, key, value, mask=None):
        # query = key = value [batch size, sent_len, hid_dim]
        bsz = query.shape[0]
        n_p = len(value[0])
        n_l = len(query[0])
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch_size, sent_len, hid_dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # K, V = [batch_size, n_heads, sent_len_K, hid_dim / n_heads]
        # Q = [batch_size, n_heads, sent_len_Q, hid_dim / n_heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch_size, n_heads, sent_len_Q, sent_len_K]

        if mask != None:
            mask = mask.unsqueeze(2).repeat(1,1,n_p).view(bsz, -1, n_l, n_p)
            #energy = energy.masked_fill(mask == 0, float('-inf'))
            attention = self.do(F.softmax(energy, dim=-1))
            attention = attention.masked_fill(mask == 0, 0)
        else:
            attention = self.do(F.softmax(energy, dim=-1))
        #energy = energy.masked_fill(mask == 0, float('-inf'))
        #attention = self.do(F.softmax(energy, dim=-1))
        # attention = attention.masked_fill(mask == 0, 0)
        #attention = attention / torch.sum(attention, dim=-1, keepdim=True)

        # attention = [batch_size, n_heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)
        # x = [batch_size, n_heads, sent_len_Q, hid_dim / n_heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch_size, sent_len_Q, n_heads, hid_dim * n_heads]
        x = x.view(bsz, -1, self.n_heads * self.hid_dim)
        # x = [batch_size, sent_len_Q, hid_dim]
        x = self.fc(x).view(bsz, -1, self.hid_dim)
        # x = [batch_size, sent_len_Q, hid_dim]

        return x

class pocket_only_net(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=52, output_dim1=128, output_dim2=64, dropout=0.4):
        self.num_features_xd = num_features_xd
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        super(pocket_only_net, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(output_dim2, output_dim2)
        self.conv2 = GCNConv(output_dim2, output_dim2 * 2)
        #self.conv3 = GCNConv(output_dim2, output_dim2)
        #self.conv4 = GATConv(output_dim2, output_dim2 * 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, self.n_output)

        self.w1, self.b1 = nn.Parameter(
            torch.randn(self.num_features_xd, self.output_dim1, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim1, requires_grad=True))
        self.w2, self.b2 = nn.Parameter(
            torch.randn(self.output_dim1, self.output_dim2, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim2, requires_grad=True))

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data1, data2, data3):
        x1, drug_intra, drug_edge_attr, batch1 = data1.x, data1.edge_index, data1.edge_attr, data1.batch
        x2, prot_intra, prot_edge_attr, batch2 = data2.x, data2.edge_index, data2.edge_attr, data2.batch

        x1 = self.node_embed(x1)
        x2 = self.node_embed(x2)

        x1 = self.conv1(x1, drug_intra, edge_attr=drug_edge_attr)
        x1 = self.relu(x1)
        x1 = self.conv2(x1, drug_intra)#, edge_attr=drug_edge_attr)
        x1 = self.relu(x1)

        x1 = global_mean_pool(x1, batch1)

        x2 = self.conv1(x2, prot_intra, edge_attr=prot_edge_attr)
        x2 = self.relu(x2)
        x2 = self.conv2(x2, prot_intra)#, edge_attr=prot_edge_attr)
        x2 = self.relu(x2)

        x2 = global_mean_pool(x2, batch2)

        x = torch.cat((x1,x2),1)
        #x = torch.cat((x1,x2),0)

        #x = self.conv3(x, edge_inter)
        #x = self.relu(x)

        #x = gmp(x, batch)       # global max pooling

        # add some dense layers
        h1 = self.fc1(x)
        #x = self.relu(x)
        x = self.relu(h1)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out, h1


class ami_only_net(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=52, output_dim1=128, output_dim2=64,output_dim3=1900,output_dim4=1024,output_dim5=256, dropout=0.3):
        self.num_features_xd = num_features_xd
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        super(ami_only_net, self).__init__()

        self.n_output = n_output
        self.conv1 = GCNConv(output_dim2, output_dim2 * 2)
        self.conv2 = GATConv(output_dim2 * 2, output_dim2 * 4)

        self.conv4 = GCNConv(output_dim3,output_dim4-1)
        self.conv5 = GCNConv(output_dim4,output_dim5)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(8 * output_dim2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, self.n_output)

        self.w1, self.b1 = nn.Parameter(
            torch.randn(self.num_features_xd, self.output_dim1, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim1,
                        requires_grad=True))
        self.w2, self.b2 = nn.Parameter(
            torch.randn(self.output_dim1, self.output_dim2, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim2,
                        requires_grad=True))
        torch.nn.init.kaiming_normal_(self.w1)
        torch.nn.init.kaiming_normal_(self.w2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data1, data2, data3):
        x1, drug_intra, edge_attr, batch1 = data1.x, data1.edge_index,data1.edge_attr, data1.batch
        #x2, prot_intra = data2.x, data2.edge_index
        #edge_inter, y, batch = data0.edge_index, data0.y, data0.batch
        x3, ami_intra, ami_dis, ami_batch, ami_dis_li = data3.x, data3.edge_index, data3.edge_attr, data3.batch, data3.ami_dis_li
        #ami_dis_li = self.dis_attention(ami_dis_li)

        ami = self.conv4(x3, ami_intra, ami_dis)
        ami = self.relu(ami)
        ami = torch.cat((ami,ami_dis_li.unsqueeze(1)),dim=1)
        ami = self.conv5(ami, ami_intra, ami_dis)
        ami = self.relu(ami)
        ami = global_mean_pool(ami, ami_batch)


        x1 = self.node_embed(x1)
        x1 = self.conv1(x1, drug_intra)#, edge_attr)
        x1 = self.relu(x1)
        x1 = self.conv2(x1, drug_intra, edge_attr)
        x1 = self.relu(x1)
        x1 = global_mean_pool(x1, batch1)       # global max pooling
        '''
        x1 = self.fc00(drug_feat)
        x1 = self.fc01(x1)
        '''
        x = torch.cat((ami, x1), 1)

        # add some dense layers
        h2 = self.fc1(x)
        x = self.relu(h2)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out, h2


class pocket_and_ami(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=52, output_dim1=128, output_dim2=64,output_dim3=1900,output_dim4=1024,output_dim5=128, dropout=0.2):
        self.num_features_xd = num_features_xd
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        super(pocket_and_ami, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(output_dim2, output_dim2)
        self.conv2 = GCNConv(output_dim2, output_dim2 * 2)

        self.conv3 = GCNConv(output_dim3,output_dim4-1)
        self.conv4 = GCNConv(output_dim4,output_dim5)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(384, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, self.n_output)

        self.w1, self.b1 = nn.Parameter(
            torch.randn(self.num_features_xd, self.output_dim1, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim1,
                        requires_grad=True))
        self.w2, self.b2 = nn.Parameter(
            torch.randn(self.output_dim1, self.output_dim2, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim2,
                        requires_grad=True))
        self.w3, self.b3 = nn.Parameter(
            torch.randn(208, 256, requires_grad=True)), nn.Parameter(
            torch.zeros(256,requires_grad=True))
        self.w4, self.b4 = nn.Parameter(
            torch.randn(256, 128, requires_grad=True)), nn.Parameter(
            torch.zeros(128, requires_grad=True))

        #torch.nn.init.kaiming_normal_(self.w1)
        #torch.nn.init.kaiming_normal_(self.w2)

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data0, data1, data2, data3):
        x1, drug_intra, drug_edge_attr, batch1 = data1.x, data1.edge_index,data1.edge_attr, data1.batch
        x2, prot_intra, prot_edge_attr, batch2 = data2.x, data2.edge_index,data2.edge_attr, data2.batch
        #edge_inter, y, batch = data0.edge_index, data0.y, data0.batch
        x3, ami_intra, ami_dis, ami_dis_li, drug_feature, ami_batch = data3.x, data3.edge_index, data3.edge_attr, data3.ami_dis_li, data3.drug_feature, data3.batch

        x1 = self.node_embed(x1)
        x2 = self.node_embed(x2)

        x1 = self.conv1(x1, drug_intra, drug_edge_attr)
        x1 = self.relu(x1)
        x1 = self.conv2(x1, drug_intra)
        x1 = self.relu(x1)

        x1 = global_mean_pool(x1, batch1)

        x2 = self.conv1(x2, prot_intra, prot_edge_attr)
        x2 = self.relu(x2)
        x2 = self.conv2(x2, prot_intra)
        x2 = self.relu(x2)

        x2 = global_mean_pool(x2, batch2)

        ami = self.conv3(x3, ami_intra, ami_dis)
        ami = self.relu(ami)
        ami = torch.cat((ami, ami_dis_li.unsqueeze(1)), dim=1)
        ami = self.conv4(ami, ami_intra, ami_dis)
        ami = self.relu(ami)
        ami = global_mean_pool(ami, ami_batch)

        x = torch.cat((x1,x2,ami),1)

        # add some dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out, None

class ami_only_net2(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=52, output_dim1=128, output_dim2=64,output_dim3=1900,output_dim4=1024,output_dim5=128, dropout=0.2):
        self.num_features_xd = num_features_xd
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        super(ami_only_net2, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(output_dim2, output_dim2)
        self.conv2 = GCNConv(output_dim2, output_dim2 * 2)

        self.conv3 = nn.Conv1d(in_channels=1,out_channels=10,kernel_size=5)
        self.conv4 = nn.Conv1d(in_channels=10,out_channels=20,kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=20,out_channels=30,kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc00 = nn.Linear(9460, 1024)
        self.fc01 = nn.Linear(1024, 128)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(4 * output_dim2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, self.n_output)

        self.w1, self.b1 = nn.Parameter(
            torch.randn(self.num_features_xd, self.output_dim1, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim1,
                        requires_grad=True))
        self.w2, self.b2 = nn.Parameter(
            torch.randn(self.output_dim1, self.output_dim2, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim2,
                        requires_grad=True))
        torch.nn.init.kaiming_normal_(self.w1)
        torch.nn.init.kaiming_normal_(self.w2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data0, data1, data2, data3):
        x1, drug_intra, edge_attr, batch1 = data1.x, data1.edge_index,data1.edge_attr, data1.batch
        #x2, prot_intra = data2.x, data2.edge_index
        #edge_inter, y, batch = data0.edge_index, data0.y, data0.batch
        x3, ami_intra, ami_dis, ami_batch, ami_dis_li, drug_feat, embedding = data3.x, data3.edge_index, data3.edge_attr, data3.batch, data3.ami_dis_li, data3.drug_feature, data3.emb

        #ami_dis_li = self.dis_attention(ami_dis_li)

        embedding = nn.functional.adaptive_max_pool2d(embedding,(1,1900)).squeeze(1)
        ami = self.conv3(embedding)
        ami = self.relu(ami)
        ami = self.pool(ami)
        ami = self.conv4(ami)
        ami = self.relu(ami)
        ami = self.pool(ami)
        #ami = self.conv5(ami)
        #ami = self.relu(ami)
        #ami = self.pool(ami)
        ami = ami.view(ami.size(0), -1)
        ami = self.fc00(ami)
        ami = self.fc01(ami)


        x1 = self.node_embed(x1)
        x1 = self.conv1(x1, drug_intra, edge_attr)
        x1 = self.relu(x1)
        x1 = self.conv2(x1, drug_intra)#, edge_attr)
        x1 = self.relu(x1)
        x1 = global_mean_pool(x1, batch1)       # global max pooling

        x = torch.cat((ami, x1), 1)

        # add some dense layers
        h2 = self.fc1(x)
        x = self.relu(h2)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out, h2


class fusion(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super(fusion, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)


    def forward(self, h1, h2):
        h1 = self.relu(h1)
        h2 = self.relu(h2)
        h = torch.cat((h1, h2),dim=1)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc3(h)
        h = self.relu(h)
        h = self.dropout(h)
        out = self.out(h)
        return out

class fusion2(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(fusion2, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)


    def forward(self, h1, h2):
        h1 = self.relu(h1)
        h2 = self.relu(h2)
        h = torch.cat((h1, h2),dim=1)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc3(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc4(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc5(h)
        h = self.relu(h)
        h = self.dropout(h)
        out = self.out(h)
        return out

class fusionNTN(torch.nn.Module):
    def __init__(self, hid_dim=256,  n_heads=1, dropout=0.5):
        super(fusionNTN, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hid_dim)
        #self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.bil = nn.Bilinear(in1_features=256, in2_features=256, out_features=256)
        self.fc = nn.Linear(512, 256)
        self.fc00 = nn.Linear(512, 256)
        self.fc01 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.out = nn.Linear(32, 1)


    def forward(self, h1, h2):
        h1 = self.fc00(self.relu(h1))
        h2 = self.fc01(self.relu(h2))
        h = self.bil(h1, h2) + self.fc(torch.cat((h1, h2), dim=1))
        #h = self.ln(self.dropout(self.sa(h, h, h, None))).squeeze(1)
        h = self.relu(h)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc3(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc4(h)
        h = self.relu(h)
        h = self.dropout(h)
        out = self.out(h)
        return out




