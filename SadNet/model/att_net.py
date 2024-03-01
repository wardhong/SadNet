import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphSAGE,GATConv, global_max_pool, global_mean_pool, global_add_pool, GINEConv, SAGPooling, MessagePassing,NNConv
from torch_geometric import utils
import math
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch_geometric.data import Batch
import networkx as nx
import matplotlib.pyplot as plt
from gtrick.pyg import VirtualNode
device = 'cuda'
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

class Interaction_Att(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.w_l1 = nn.Linear(hid_dim, hid_dim * n_heads)
        self.w_l2 = nn.Linear(hid_dim, hid_dim * n_heads)
        self.w_p1 = nn.Linear(hid_dim, hid_dim * n_heads)
        self.w_p2 = nn.Linear(hid_dim, hid_dim * n_heads)
        self.fc11 = nn.Linear(hid_dim * n_heads, hid_dim)
        self.fc12 = nn.Linear(2 * hid_dim, hid_dim)
        self.fc21 = nn.Linear(hid_dim * n_heads, hid_dim)
        self.fc22 = nn.Linear(2 * hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn_i = nn.BatchNorm2d(n_heads)
        self.scale = math.sqrt(hid_dim)

    def forward(self, ligand, prot, inter):
        # query = key = value [batch size, sent_len, hid_dim]
        bsz = ligand.shape[0]
        ligand1 = self.relu(self.w_l1(ligand))
        ligand2 = self.relu(self.w_l2(ligand))
        prot1 = self.relu(self.w_p1(prot))
        prot2 = self.relu(self.w_p2(prot))
        # 1 = [batch_size, m(n), hid_dim*n_head]
        # 2 = [batch_size, m(n), hid_dim]
        ligand1 = ligand1.view(bsz, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)
        ligand2 = ligand2.view(bsz, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)
        prot1 = prot1.view(bsz, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)
        prot2 = prot2.view(bsz, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)
        # ligand = [batch_size, n_heads, m, hid_dim]
        # prot = [batch_size, n_heads,n, hid_dim]

        #adj = 1/inter.unsqueeze(1)
        #energy = self.bn_i(torch.matmul(ligand1, prot1.permute(0, 1, 3, 2)) )#/ self.scale)
        energy = torch.matmul(ligand1, prot1.permute(0, 1, 3, 2)) / self.scale
        #att = energy.masked_fill(inter == 0, 0)
        att = torch.mul(energy, inter)
        att = self.do(F.softmax(att, dim=-1))
        #print(att[0][1][0])
        # energy = [batch_size, n_heads, m, n]
        # inter. = [batch_size,       1, m, n]
        # att    = [batch_size, n_heads, m, n]

        ligand3 = torch.matmul(att, prot2)
        # l3 = [batch_size, n_heads, m, hid_dim]
        ligand3 = ligand3.permute(0, 2, 1, 3).contiguous()
        # l3 = [batch_size, m, n_heads, hid_dim]
        ligand3 = ligand3.view(bsz, -1, self.n_heads * self.hid_dim)
        # l3 = [batch_size, m, hid_dim*n_head]
        ligand3 = torch.cat((self.fc11(ligand3).view(bsz, -1, self.hid_dim), ligand), dim=2)
        # l3 = [batch_size, sent_len_Q, hid_dim]
        ligand3 = self.do(self.relu(self.fc12(ligand3)))

        prot3 = torch.matmul(att.permute(0, 1, 3, 2), ligand2)
        # p3 = [batch_size, n_heads, n, hid_dim]
        prot3 = prot3.permute(0, 2, 1, 3).contiguous()
        # p3 = [batch_size, n, n_heads, hid_dim]
        prot3 = prot3.view(bsz, -1, self.n_heads * self.hid_dim)
        # p3 = [batch_size, m, hid_dim*n_head]
        prot3 = torch.cat((self.fc21(prot3).view(bsz, -1, self.hid_dim), prot), dim=2)
        # p3 = [batch_size, sent_len_Q, hid_dim]
        prot3 = self.do(self.relu(self.fc22(prot3)))
        return ligand3, prot3

class get_mask(nn.Module):
    def __init__(self):
        super().__init__()
    #(0,2,3,3.5,4,4.5,5,8)
    def forward(self, inter):
        matrix1 = inter.clone()
        matrix1[(matrix1 > 0) & (matrix1 <= 3)] = 1
        matrix1[(matrix1 > 3)] = 0
        value1 = 1/inter.clone()
        value1 = value1.masked_fill(matrix1==0, 0)
        value1 = value1.unsqueeze(1)
        matrix1 = matrix1.unsqueeze(1)

        matrix2 = inter.clone()
        matrix2[(matrix2 <= 1)] = 0
        matrix2[(matrix2 > 1) & (matrix2 <= 4)] = 1
        matrix2[(matrix2 > 4)] = 0
        value2 = 1/inter.clone()
        value2 = value2.masked_fill(matrix2 == 0, 0)
        value2 = value2.unsqueeze(1)
        matrix2 = matrix2.unsqueeze(1)

        matrix3 = inter.clone()
        matrix3[(matrix3 <= 2)] = 0
        matrix3[(matrix3 > 2) & (matrix3 <= 5)] = 1
        matrix3[(matrix3 > 5)] = 0
        value3 = 1/inter.clone()
        value3 = value3.masked_fill(matrix3 == 0, 0)
        value3 = value3.unsqueeze(1)
        matrix3 = matrix3.unsqueeze(1)

        matrix4 = inter.clone()
        matrix4[(matrix4 <= 3)] = 0
        matrix4[(matrix4 > 3) & (matrix4 <= 6)] = 1
        matrix4[(matrix4 > 6)] = 0
        value4 = 1/inter.clone()
        value4 = value4.masked_fill(matrix4 == 0, 0)
        value4 = value4.unsqueeze(1)
        matrix4 = matrix4.unsqueeze(1)

        matrix5 = inter.clone()
        matrix5[(matrix5 <= 4)] = 0
        matrix5[(matrix5 > 4) & (matrix5 <= 7)] = 1
        matrix5[(matrix5 > 7)] = 0
        value5 = 1/inter.clone()
        value5 = value5.masked_fill(matrix5 == 0, 0)
        value5 = value5.unsqueeze(1)
        matrix5 = matrix5.unsqueeze(1)

        matrix6 = inter.clone()
        matrix6[(matrix6 <= 5)] = 0
        matrix6[(matrix6 > 5) & (matrix6 <= 8)] = 1
        matrix6[(matrix6 > 8)] = 0
        value6 = 1/inter.clone()
        value6 = value6.masked_fill(matrix6 == 0, 0)
        value6 = value6.unsqueeze(1)
        matrix6 = matrix6.unsqueeze(1)

        matrix7 = inter.clone()
        matrix7[(matrix7 <= 6)] = 0
        matrix7[(matrix7 > 6) & (matrix7 <= 9)] = 1
        matrix7[(matrix7 > 9)] = 0
        value7 = 1/inter.clone()
        value7 = value7.masked_fill(matrix7 == 0, 0)
        value7 = value7.unsqueeze(1)
        matrix7 = matrix7.unsqueeze(1)

        matrix8 = inter.clone()
        matrix8[matrix8 <= 7] = 0
        matrix8[(matrix8 > 7)] = 1
        value8 = 1/inter.clone()
        value8 = value8.masked_fill(matrix8 == 0, 0)
        value8 = value8.masked_fill(value8 == float('inf'), 0)
        value8 = value8.unsqueeze(1)
        matrix8 = matrix8.unsqueeze(1)

        #matrix = torch.cat((matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7, matrix8), dim=1)
        matrix = torch.cat((value1, value2, value3, value4, value5, value6, value7, value8), dim=1)
        return matrix

class get_mask2(nn.Module):
    def __init__(self):
        super().__init__()
    #(0,2,3,3.5,4,4.5,5,8)
    def forward(self, inter):
        matrix1 = inter.clone()
        matrix1[(matrix1 > 0) & (matrix1 <= 1)] = 1
        matrix1[(matrix1 > 1)] = 0
        value1 = 1/inter.clone()
        value1 = value1.masked_fill(matrix1==0, 0)
        value1 = value1.unsqueeze(1)
        matrix1 = matrix1.unsqueeze(1)

        matrix2 = inter.clone()
        matrix2[(matrix2 <= 1)] = 0
        matrix2[(matrix2 > 1) & (matrix2 <= 2)] = 1
        matrix2[(matrix2 > 2)] = 0
        value2 = 1/inter.clone()
        value2 = value2.masked_fill(matrix2 == 0, 0)
        value2 = value2.unsqueeze(1)
        matrix2 = matrix2.unsqueeze(1)

        matrix3 = inter.clone()
        matrix3[(matrix3 <= 2)] = 0
        matrix3[(matrix3 > 2) & (matrix3 <= 3)] = 1
        matrix3[(matrix3 > 3)] = 0
        value3 = 1/inter.clone()
        value3 = value3.masked_fill(matrix3 == 0, 0)
        value3 = value3.unsqueeze(1)
        matrix3 = matrix3.unsqueeze(1)

        matrix4 = inter.clone()
        matrix4[(matrix4 <= 3)] = 0
        matrix4[(matrix4 > 3) & (matrix4 <= 4)] = 1
        matrix4[(matrix4 > 4)] = 0
        value4 = 1/inter.clone()
        value4 = value4.masked_fill(matrix4 == 0, 0)
        value4 = value4.unsqueeze(1)
        matrix4 = matrix4.unsqueeze(1)

        matrix5 = inter.clone()
        matrix5[(matrix5 <= 4)] = 0
        matrix5[(matrix5 > 4) & (matrix5 <= 5)] = 1
        matrix5[(matrix5 > 5)] = 0
        value5 = 1/inter.clone()
        value5 = value5.masked_fill(matrix5 == 0, 0)
        value5 = value5.unsqueeze(1)
        matrix5 = matrix5.unsqueeze(1)

        matrix6 = inter.clone()
        matrix6[(matrix6 <= 5)] = 0
        matrix6[(matrix6 > 5) & (matrix6 <= 6)] = 1
        matrix6[(matrix6 > 6)] = 0
        value6 = 1/inter.clone()
        value6 = value6.masked_fill(matrix6 == 0, 0)
        value6 = value6.unsqueeze(1)
        matrix6 = matrix6.unsqueeze(1)

        matrix7 = inter.clone()
        matrix7[(matrix7 <= 6)] = 0
        matrix7[(matrix7 > 6) & (matrix7 <= 7)] = 1
        matrix7[(matrix7 > 7)] = 0
        value7 = 1/inter.clone()
        value7 = value7.masked_fill(matrix7 == 0, 0)
        value7 = value7.unsqueeze(1)
        matrix7 = matrix7.unsqueeze(1)

        matrix8 = inter.clone()
        matrix8[matrix8 <= 7] = 0
        matrix8[(matrix8 > 7)] = 1
        value8 = 1/inter.clone()
        value8 = value8.masked_fill(matrix8 == 0, 0)
        value8 = value8.masked_fill(value8 == float('inf'), 0)
        value8 = value8.unsqueeze(1)
        matrix8 = matrix8.unsqueeze(1)

        #matrix = torch.cat((matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7, matrix8), dim=1)
        matrix = torch.cat((value1, value2, value3, value4, value5, value6, value7, value8), dim=1)
        return matrix

class virtual_node(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0.5, residual=False):
        super(virtual_node, self).__init__()
        self.dropout = dropout
        # Add residual connection or not
        self.residual = residual

        # Set the initial virtual node embedding to 0.
        self.vn_emb = nn.Embedding(1, in_feats)
        # nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        if in_feats == out_feats:
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(in_feats, out_feats)

        # MLP to transform virtual node at every layer
        self.mlp_vn = nn.Sequential(
            nn.Linear(out_feats, 2 * out_feats),
            nn.BatchNorm1d(2 * out_feats),
            nn.ReLU(),
            nn.Linear(2 * out_feats, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU())

        self.reset_parameters()

    def reset_parameters(self):
        if not isinstance(self.linear, nn.Identity):
            self.linear.reset_parameters()

        for c in self.mlp_vn.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.vn_emb.weight.data, 0)

    def update_node_emb(self, x, edge_index, batch, vx=None, a=0.5):
        # Virtual node embeddings for graphs
        if vx is None:
            vx = self.vn_emb(torch.zeros(
                batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        # Add message from virtual nodes to graph nodes
        h = x + a * vx[batch]
        return h, vx

    def update_vn_emb(self, x, batch, vx):
        # Add message from graph nodes to virtual nodes
        vx = self.linear(vx)
        vx_temp = global_add_pool(x, batch) + vx

        # transform virtual nodes using MLP
        vx_temp = self.mlp_vn(vx_temp)

        if self.residual:
            vx = vx + F.dropout(
                vx_temp, self.dropout, training=self.training)
        else:
            vx = F.dropout(
                vx_temp, self.dropout, training=self.training)

        return vx

class pocket_att_net2(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=52, hid_dim=128, output_dim1=128, output_dim2=64, dropout=0.1):
        self.num_features_xd = num_features_xd
        self.hid_dim = hid_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.n_output = n_output

        super(pocket_att_net2, self).__init__()

        #self.ca1 = SelfAttention(hid_dim=self.hid_dim//2, n_heads=2, dropout=dropout)
        self.ica1 = Interaction_Att(hid_dim=self.hid_dim//2, n_heads=8, dropout=dropout)
        self.ica2 = Interaction_Att(hid_dim=self.hid_dim // 2, n_heads=8, dropout=dropout)
        self.ln1 = nn.LayerNorm(self.hid_dim//2)
        #self.ca2 = SelfAttention(hid_dim=self.hid_dim, n_heads=1, dropout=dropout)
        self.ln2 = nn.LayerNorm(self.hid_dim//2)
        self.fc00 = nn.Linear(1900, 1024)
        self.fc01 = nn.Linear(1024, 512)
        self.fc02 = nn.Linear(512, 63)

        self.conv1 = GATConv(output_dim2, output_dim2)
        self.conv2 = GATConv(output_dim2, output_dim2)
        self.conv3 = GATConv(output_dim2, output_dim2)
        self.conv4 = GATConv(output_dim2, output_dim2)
        self.conv5 = GATConv(output_dim2, output_dim2)
        self.conv6 = GATConv(output_dim2, output_dim2)
        self.conv7 = GCNConv(output_dim2, output_dim2)
        self.conv8 = GCNConv(output_dim2, output_dim2)
        self.conv9 = GCNConv(output_dim2, output_dim2)

        self.vns1 = virtual_node(output_dim2, output_dim2, dropout)
        self.vns2 = virtual_node(output_dim2, output_dim2, dropout)
        self.vns3 = virtual_node(output_dim2, output_dim2, dropout)

        self.get_mask = get_mask()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_att1 = nn.Linear(64, 1)
        self.fc_att2 = nn.Linear(64, 1)

        self.tanh = nn.Tanh()
        self.fc0 = nn.Linear(128, 128)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, self.n_output)

        self.w1, self.b1 = nn.Parameter(
            torch.randn(self.num_features_xd, self.output_dim1, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim1,
                        requires_grad=True))
        self.w2, self.b2 = nn.Parameter(
            torch.randn(self.output_dim1, self.output_dim2, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim2,
                        requires_grad=True))

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data1, data2, data3):
        x1, drug_intra, drug_edge_attr, batch1, inter = data1.x, data1.edge_index, data1.edge_attr, data1.batch, data1.inter
        x2, prot_intra, prot_edge_attr, batch2 = data2.x, data2.edge_index, data2.edge_attr, data2.batch
        x3, ami_intra, ami_dis, ami_batch, ami_dis_li = data3.x, data3.edge_index, data3.edge_attr, data3.batch, data3.ami_dis_li

        ami = self.relu(self.fc00(x3))
        ami = self.relu(self.fc01(ami))
        ami = self.relu(self.fc02(ami))
        ami0 = torch.cat((ami, ami_dis_li.unsqueeze(1)), dim=1)
        ami = self.conv7(ami0, ami_intra, ami_dis)
        ami = self.relu(ami)
        ami1 = global_mean_pool(ami, ami_batch)
        ami = self.conv8(ami, ami_intra, ami_dis)
        ami = self.relu(ami)
        ami2 = global_mean_pool(ami, ami_batch)
        ami = ami1 + ami2

        x1 = self.node_embed(x1)
        x2 = self.node_embed(x2)

        x11 = self.relu(self.conv1(x1, drug_intra, drug_edge_attr))
        x1, ami = self.vns1.update_node_emb(x=x11, edge_index=drug_intra, batch=batch1, vx=ami, a=0.3)
        x12 = self.relu(self.conv2(x1, drug_intra, drug_edge_attr))
        ami = self.vns1.update_vn_emb(x12, batch1, ami)

        #x1, vx = self.vns2.update_node_emb(x=x12, edge_index=drug_intra, batch=batch1, vx=ami, a=0.3)
        #x13 = self.relu(self.conv3(x1, drug_intra, drug_edge_attr))
        ###x13 = self.virtual_update_node(x=x1, vx=ami, batch=batch1)
        x1 = x11+x12#+x13

        x_l, mask_l = utils.to_dense_batch(x1, batch1)
        #x_l = torch.unsqueeze(x1, 1)

        x2 = self.conv4(x2, prot_intra, prot_edge_attr)
        x21 = self.relu(x2)
        x2 = self.conv5(x21, prot_intra, prot_edge_attr)
        x22 = self.relu(x2)
        #x2 = self.conv6(x22, prot_intra, prot_edge_attr)
        #x23 = self.relu(x2)
        x2 = x21+x22#+x23

        x_p, mask_p = utils.to_dense_batch(x2, batch2)
        inter = self.get_mask(inter)
        x_l, x_p = self.ica1(x_l, x_p, inter)

        #x_l, x_p = self.ica2(x_l, x_p, inter)

        #x_l = self.ln1(x_l)
        #x_p = self.ln2(x_p)

        a_l = self.tanh(self.fc_att1(x_l))
        x_l = torch.mul(x_l, a_l)
        x_l = torch.cat([torch.mean(x_l, dim=1), torch.max(x_l, dim=1).values], dim=1)
        a_p = self.tanh(self.fc_att2(x_p))
        x_p = torch.mul(x_p, a_p)
        x_p = torch.cat([torch.mean(x_p, dim=1), torch.max(x_p, dim=1).values], dim=1)

        x = torch.cat((x_l, x_p), dim=1)
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

class pocket_att_net3(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=52, hid_dim=128, output_dim1=128, output_dim2=64, dropout=0.3):
        self.num_features_xd = num_features_xd
        self.hid_dim = hid_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.n_output = n_output

        super(pocket_att_net3, self).__init__()

        #self.ca1 = SelfAttention(hid_dim=self.hid_dim//2, n_heads=2, dropout=dropout)
        self.ica1 = Interaction_Att(hid_dim=self.hid_dim//2, n_heads=8, dropout=dropout)
        self.ica2 = Interaction_Att(hid_dim=self.hid_dim // 2, n_heads=8, dropout=dropout)
        self.ln1 = nn.LayerNorm(self.hid_dim//2)
        #self.ca2 = SelfAttention(hid_dim=self.hid_dim, n_heads=1, dropout=dropout)
        self.ln2 = nn.LayerNorm(self.hid_dim//2)
        self.fc00 = nn.Linear(1900, 1024)
        self.fc01 = nn.Linear(1024, 512)
        self.fc02 = nn.Linear(512, 63)

        self.conv1 = GATConv(output_dim2, output_dim2)
        self.conv2 = GATConv(output_dim2, output_dim2)
        self.conv3 = GATConv(output_dim2, output_dim2)
        self.conv4 = GATConv(output_dim2, output_dim2)
        self.conv5 = GATConv(output_dim2, output_dim2)
        self.conv6 = GATConv(output_dim2, output_dim2)
        self.conv7 = GCNConv(output_dim2, output_dim2)
        self.conv8 = GCNConv(output_dim2, output_dim2)
        self.conv9 = GCNConv(output_dim2, output_dim2)
        ###
        self.conv10 = GATConv(64, 64)
        self.conv11 = GATConv(64, 64)
        ###
        self.vns1 = virtual_node(output_dim2, output_dim2, dropout)
        self.vns2 = virtual_node(output_dim2, output_dim2, dropout)
        self.vns3 = virtual_node(output_dim2, output_dim2, dropout)

        self.get_mask = get_mask()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_att1 = nn.Linear(64, 1)
        self.fc_att2 = nn.Linear(64, 1)

        self.tanh = nn.Tanh()
        self.fc0 = nn.Linear(128, 128)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, self.n_output)

        self.w1, self.b1 = nn.Parameter(
            torch.randn(self.num_features_xd, self.output_dim1, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim1,
                        requires_grad=True))
        self.w2, self.b2 = nn.Parameter(
            torch.randn(self.output_dim1, self.output_dim2, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim2,
                        requires_grad=True))

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data1, data2, data3):
        x1, drug_intra, drug_edge_attr, batch1, inter = data1.x, data1.edge_index, data1.edge_attr, data1.batch, data1.inter
        x2, prot_intra, prot_edge_attr, batch2 = data2.x, data2.edge_index, data2.edge_attr, data2.batch
        x3, ami_intra, ami_dis, ami_batch, ami_dis_li = data3.x, data3.edge_index, data3.edge_attr, data3.batch, data3.ami_dis_li

        ami = self.relu(self.fc00(x3))
        ami = self.relu(self.fc01(ami))
        ami = self.relu(self.fc02(ami))
        ami0 = torch.cat((ami, ami_dis_li.unsqueeze(1)), dim=1)
        ami = self.conv7(ami0, ami_intra, ami_dis)
        ami = self.relu(ami)
        ami1 = global_mean_pool(ami, ami_batch)
        ami = self.conv8(ami, ami_intra, ami_dis)
        ami = self.relu(ami)
        ami2 = global_mean_pool(ami, ami_batch)
        ami = ami1 + ami2

        x1 = self.node_embed(x1)
        x2 = self.node_embed(x2)

        x11 = self.relu(self.conv1(x1, drug_intra, drug_edge_attr))
        x1, ami = self.vns1.update_node_emb(x=x11, edge_index=drug_intra, batch=batch1, vx=ami, a=0.3)
        x12 = self.relu(self.conv2(x1, drug_intra, drug_edge_attr))
        ami = self.vns1.update_vn_emb(x12, batch1, ami)
        x12 = x11 + x12

        '''x1, vx = self.vns2.update_node_emb(x=x12, edge_index=drug_intra, batch=batch1, vx=ami, a=0.3)
        x13 = self.relu(self.conv3(x1, drug_intra, drug_edge_attr))'''

        #x1 = x11+x12+x13

        x_l, mask_l = utils.to_dense_batch(x12, batch1)
        #x_l = torch.unsqueeze(x1, 1)

        x2 = self.conv4(x2, prot_intra, prot_edge_attr)
        x21 = self.relu(x2)
        x2 = self.conv5(x21, prot_intra, prot_edge_attr)
        x22 = self.relu(x2)
        x2 = self.conv6(x22, prot_intra, prot_edge_attr)
        x23 = self.relu(x2)
        x2 = x21+x22+x23

        x_p, mask_p = utils.to_dense_batch(x2, batch2)
        inter = self.get_mask(inter)
        x_l, x_p = self.ica1(x_l, x_p, inter)
        ###
        x_l2 = x_l.view(-1, 64)
        x_p2 = x_p.view(-1, 64)
        x_l2 = self.relu(self.conv10(x_l2, drug_intra, drug_edge_attr))
        x1, vx = self.vns2.update_node_emb(x=x_l2, edge_index=drug_intra, batch=batch1, vx=ami, a=0.3)
        x13 = self.relu(self.conv3(x1, drug_intra, drug_edge_attr))

        x_p2 = self.relu(self.conv11(x_p2, prot_intra, prot_edge_attr))
        x_l, mask_l = utils.to_dense_batch(x13, batch1)
        x_p, mask_p = utils.to_dense_batch(x_p2, batch2)
        x_l, x_p = self.ica2(x_l, x_p, inter)
        #x_l, x_p = self.ica2(x_l, x_p, inter)

        #x_l = self.ln1(x_l)
        #x_p = self.ln2(x_p)

        a_l = self.tanh(self.fc_att1(x_l))
        x_l = torch.mul(x_l, a_l)
        x_l = torch.cat([torch.mean(x_l, dim=1), torch.max(x_l, dim=1).values], dim=1)
        a_p = self.tanh(self.fc_att2(x_p))
        x_p = torch.mul(x_p, a_p)
        x_p = torch.cat([torch.mean(x_p, dim=1), torch.max(x_p, dim=1).values], dim=1)

        x = torch.cat((x_l, x_p), dim=1)
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

class pocket_att_net4(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=52, hid_dim=128, output_dim1=128, output_dim2=128, dropout=0.3):
        self.num_features_xd = num_features_xd
        self.hid_dim = hid_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.n_output = n_output

        super(pocket_att_net4, self).__init__()

        self.ica1 = Interaction_Att(hid_dim=self.hid_dim, n_heads=8, dropout=dropout)
        #self.ln1 = nn.LayerNorm(self.hid_dim)
        #self.ln2 = nn.LayerNorm(self.hid_dim)

        self.conv1 = GATConv(output_dim2, output_dim2)
        self.conv2 = GATConv(output_dim2, output_dim2)
        self.conv3 = GATConv(output_dim2, output_dim2)
        self.conv4 = GATConv(output_dim2, output_dim2)
        self.conv5 = GATConv(output_dim2, output_dim2)
        self.conv6 = GATConv(output_dim2, output_dim2)

        self.get_mask = get_mask()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_att1 = nn.Linear(128, 1)
        self.fc_att2 = nn.Linear(128, 1)

        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, self.n_output)

        self.w1, self.b1 = nn.Parameter(
            torch.randn(self.num_features_xd, self.output_dim1, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim1,
                        requires_grad=True))
        self.w2, self.b2 = nn.Parameter(
            torch.randn(self.output_dim1, self.output_dim2, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim2,
                        requires_grad=True))

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data1, data2, data3):
        x1, drug_intra, drug_edge_attr, batch1, inter = data1.x, data1.edge_index, data1.edge_attr, data1.batch, data1.inter
        x2, prot_intra, prot_edge_attr, batch2 = data2.x, data2.edge_index, data2.edge_attr, data2.batch

        x1 = self.node_embed(x1)
        x2 = self.node_embed(x2)

        x11 = self.relu(self.conv1(x1, drug_intra, drug_edge_attr))
        x12 = self.relu(self.conv2(x11, drug_intra, drug_edge_attr))
        x13 = self.relu(self.conv3(x12, drug_intra, drug_edge_attr))
        x1 = x11+x12+x13

        x_l, mask_l = utils.to_dense_batch(x1, batch1)

        x2 = self.conv4(x2, prot_intra, prot_edge_attr)
        x21 = self.relu(x2)
        x2 = self.conv5(x21, prot_intra, prot_edge_attr)
        x22 = self.relu(x2)
        x2 = self.conv6(x22, prot_intra, prot_edge_attr)
        x23 = self.relu(x2)
        x2 = x21+x22+x23

        x_p, mask_p = utils.to_dense_batch(x2, batch2)
        inter = self.get_mask(inter)
        x_l, x_p = self.ica1(x_l, x_p, inter)

        #a_l = self.tanh(self.fc_att1(x_l))
        #x_l = torch.mul(x_l, a_l)
        x_l = torch.mean(x_l, dim=1)
        #a_p = self.tanh(self.fc_att2(x_p))
        #x_p = torch.mul(x_p, a_p)
        x_p = torch.mean(x_p, dim=1)

        x = torch.cat((x_l, x_p), dim=1)
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

class ami_att_net2(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=52, output_dim1=128, output_dim2=64,output_dim3=1900,output_dim4=1024,output_dim5=256, dropout=0.1):
        self.num_features_xd = num_features_xd
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        super(ami_att_net2, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(output_dim1, output_dim1)
        self.conv2 = GATConv(output_dim1, output_dim1)
        self.conv3 = GATConv(output_dim1, output_dim1)
        self.vns1 = virtual_node(output_dim1, output_dim1, dropout)
        self.vns2 = virtual_node(output_dim1, output_dim1, dropout)
        self.fc00 = nn.Linear(1900, 1024)
        self.fc01 = nn.Linear(1024, 512)
        self.fc02 = nn.Linear(512, 127)
        self.fc03 = nn.Linear(64, 128)
        self.conv4 = GCNConv(output_dim1, output_dim1)
        self.conv5 = GCNConv(output_dim1, output_dim1)
        self.conv6 = GCNConv(output_dim1, output_dim1)
        self.SAGPooling1 = SAGPooling(in_channels=128, ratio=0.75)
        self.SAGPooling2 = SAGPooling(in_channels=128, ratio=0.75)
        self.SAGPooling3 = SAGPooling(in_channels=128, ratio=0.75)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc0 = nn.Linear(output_dim5, output_dim1)
        self.fc_att = nn.Linear(output_dim1, 1)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, self.n_output)

        self.w1, self.b1 = nn.Parameter(
            torch.randn(self.num_features_xd, self.output_dim1, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim1,
                        requires_grad=True))
        self.w2, self.b2 = nn.Parameter(
            torch.randn(self.output_dim1, self.output_dim2, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim2,
                        requires_grad=True))

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data1, data2, data3):
        x1, drug_intra, drug_edge_attr, batch1 = data1.x, data1.edge_index,data1.edge_attr, data1.batch
        x3, ami_intra, ami_dis, ami_batch, ami_dis_li = data3.x, data3.edge_index, data3.edge_attr, data3.batch, data3.ami_dis_li

        ami = self.relu(self.fc00(x3))
        ami = self.relu(self.fc01(ami))
        ami = self.relu(self.fc02(ami))
        ami0 = torch.cat((ami, ami_dis_li.unsqueeze(1)), dim=1)

        ami = self.conv4(ami0, ami_intra, ami_dis)
        ami = self.relu(ami)
        #ami, ami_intra, ami_dis, ami_batch, _, _ = self.SAGPooling1(ami, ami_intra, ami_dis, ami_batch)
        ami1 =global_mean_pool(ami, ami_batch)
        ami = self.conv5(ami, ami_intra, ami_dis)
        ami = self.relu(ami)
        #ami, ami_intra, ami_dis, ami_batch, _, _ = self.SAGPooling2(ami, ami_intra, ami_dis, ami_batch)
        ami2 = global_mean_pool(ami, ami_batch)
        ami = self.conv6(ami, ami_intra, ami_dis)
        ami = self.relu(ami)
        #ami, ami_intra, ami_dis, ami_batch, _, _ = self.SAGPooling3(ami, ami_intra, ami_dis, ami_batch)
        ami3 = global_mean_pool(ami, ami_batch)
        ami = ami1+ami2+ami3
        #ami = torch.zeros(ami_batch[-1]+1, 128, dtype=torch.int8).cuda()

        x1 = self.node_embed(x1)
        x1 = self.fc03(x1)
        x1, drug = self.vns1.update_node_emb(x=x1, edge_index=drug_intra, batch=batch1, a=0.2)
        x11 = self.relu(self.conv1(x1, drug_intra, drug_edge_attr))
        drug = self.vns1.update_vn_emb(x11, batch1, drug)

        x1, drug = self.vns2.update_node_emb(x=x11, edge_index=drug_intra, batch=batch1,vx=drug, a=0.2)
        x12 = self.relu(self.conv2(x1, drug_intra, drug_edge_attr))

        x13 = self.relu(self.conv3(x1, drug_intra, drug_edge_attr))
        '''
        #x1, drug_intra, drug_edge_attr, batch1, _, _ = self.SAGPooling1(x1, drug_intra, drug_edge_attr, batch1)
        x11 = global_mean_pool(x1, batch1)
        x1 = self.relu(self.conv2(x1, drug_intra, drug_edge_attr))
        #x1, drug_intra, drug_edge_attr, batch1, _, _ = self.SAGPooling2(x1, drug_intra, drug_edge_attr, batch1)
        x12 = global_mean_pool(x1, batch1)
        x1 = self.relu(self.conv3(x1, drug_intra, drug_edge_attr))
        #x1, drug_intra, drug_edge_attr, batch1, _, _ = self.SAGPooling3(x1, drug_intra, drug_edge_attr, batch1)
        x13 = global_mean_pool(x1, batch1)
        x1 = x11+x12+x13
        '''
        x1 = global_mean_pool(x11+x12+x13, batch1)
        x = torch.cat((x1, ami), dim=1)

        # add some dense layers
        h2 = self.fc1(x)
        x = self.relu(h2)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out, h2

class ami_att_net(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=52, output_dim1=128, output_dim2=64,output_dim3=1900,output_dim4=1024,output_dim5=256, dropout=0.2):
        self.num_features_xd = num_features_xd
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        super(ami_att_net, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(output_dim1, output_dim1)
        self.conv2 = GATConv(output_dim1, output_dim1)
        self.conv3 = GATConv(output_dim1, output_dim1)

        self.fc00 = nn.Linear(1900, 1024)
        self.fc01 = nn.Linear(1024, 512)
        self.fc02 = nn.Linear(512, 127)
        self.fc03 = nn.Linear(64, 128)
        self.conv4 = GATConv(output_dim1, output_dim1)
        self.conv5 = GATConv(output_dim1, output_dim1)
        self.conv6 = GATConv(output_dim1, output_dim1)

        self.ln = nn.LayerNorm(output_dim1)
        self.ca = SelfAttention(hid_dim=output_dim1, n_heads=1, dropout=dropout)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc0 = nn.Linear(output_dim5, output_dim1)
        self.fc_att = nn.Linear(output_dim1, 1)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(output_dim5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, self.n_output)

        self.w1, self.b1 = nn.Parameter(
            torch.randn(self.num_features_xd, self.output_dim1, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim1,
                        requires_grad=True))
        self.w2, self.b2 = nn.Parameter(
            torch.randn(self.output_dim1, self.output_dim2, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim2,
                        requires_grad=True))

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data1, data2, data3):
        x1, drug_intra, drug_edge_attr, batch1 = data1.x, data1.edge_index,data1.edge_attr, data1.batch
        x3, ami_intra, ami_dis, ami_batch, ami_dis_li = data3.x, data3.edge_index, data3.edge_attr, data3.batch, data3.ami_dis_li

        ami = self.relu(self.fc00(x3))
        ami = self.relu(self.fc01(ami))
        ami = self.relu(self.fc02(ami))
        ami0 = torch.cat((ami, ami_dis_li.unsqueeze(1)), dim=1)
        ami = self.conv4(ami0, ami_intra, ami_dis)
        ami1 = self.relu(ami)
        ami = self.conv5(ami1, ami_intra, ami_dis)
        ami2 = self.relu(ami)
        ami = self.conv6(ami2, ami_intra, ami_dis)
        ami3 = self.relu(ami)
        ami = ami0+ami1+ami2+ami3
        #ami, ami_intra, ami_dis, ami_batch, _, _ = self.SAGPooling3(ami, ami_intra, ami_dis, ami_batch)
        #torch.set_printoptions(profile="full")
        #print(ami_intra)


        prot, mask_p = utils.to_dense_batch(ami, ami_batch)

        x1 = self.node_embed(x1)
        x10 = self.fc03(x1)
        x1 = self.conv1(x10, drug_intra, drug_edge_attr)
        x11 = self.relu(x1)
        x1 = self.conv2(x11, drug_intra, drug_edge_attr)
        x12 = self.relu(x1)
        x1 = self.conv3(x12, drug_intra, drug_edge_attr)
        x13 = self.relu(x1)
        #print(ami)
        x1 = x10+x11+x12+x13
        drug, mask_l = utils.to_dense_batch(x1, batch1)

        ami_att = self.ln(self.dropout(self.ca(drug, prot, prot, mask_l)))
        x = torch.cat((drug, ami_att), dim=2)
        x = self.fc0(x)
        x = self.relu(x)

        a = self.fc_att(x).squeeze(2)
        a = self.tanh(a).unsqueeze(2)
        x = torch.mul(x, a)
        x = torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1).values], dim=1)

        # add some dense layers
        h2 = self.fc1(x)
        x = self.relu(h2)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out, h2

class att_fusion(torch.nn.Module):
    def __init__(self, hid_dim=512,  n_heads=4, dropout=0.3):
        super(att_fusion, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)

        self.fc00 = nn.Linear(512, 256)
        self.fc01 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.out = nn.Linear(32, 1)


    def forward(self, h1, h2):
        h1 = self.fc00(self.relu(h1))
        h2 = self.fc01(self.relu(h2))
        h = self.relu(torch.cat((h1, h2), dim=1))
        #h = self.fc1(h)
        #h = self.relu(h)
        h = self.ln(self.dropout(self.sa(h, h, h, None))).squeeze()

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

class att_fusion2(torch.nn.Module):
    def __init__(self, hid_dim=512,  n_heads=4, dropout=0.1):
        super(att_fusion2, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)

        self.fc00 = nn.Linear(512, 256)
        self.fc01 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, h1, h2):
        h1 = self.fc00(self.relu(h1))
        h2 = self.fc01(self.relu(h2))
        m = torch.max(h1, h2)
        n = h1 - h2
        h = self.relu(torch.cat((m, n), dim=1))
        #h = self.fc1(h)
        #h = self.relu(h)
        h = self.ln(self.dropout(self.sa(h, h, h, None))).squeeze()

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

#inter
'''
inter, mask_inter = utils.to_dense_batch(inter, batch0)
inter_dis = inter[:, :, 2]
inter_dis = inter_dis.masked_fill(mask_inter == 0, float('inf'))
#torch.set_printoptions(profile="full")
#print(mask_inter)
inter_drug = inter[:, :, 0].long()
ind = torch.LongTensor(np.array(list(range(batch0[-1]+1)))).unsqueeze(1)
ligand_f = x_l[ind, inter_drug, :]
inter_prot =inter[:, :, 1].long()
prot_f = x_p[ind, inter_prot, :]
edge_f = torch.cat((ligand_f, prot_f), dim=2)
mask_inter = mask_inter.unsqueeze(2).repeat(1, 1, self.output_dim1)
edge_f = edge_f.masked_fill(mask_inter==0, 0)
x = self.fc0(edge_f)
x = self.relu(x)
x = self.dropout(x)
#a = self.fc_att(x).squeeze(2)
mm = 1/inter_dis
a = self.tanh(self.fc_att(torch.cat((x, mm.unsqueeze(2)), dim=2)))
x = torch.mul(x, a)
x = torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1).values], dim=1)

#x2_att, _ = self.ca01(x_l, x_p, x_p)
#x2_att = self.ln1(self.dropout(x2_att))
x2_att = self.ln1(self.dropout(self.ca1(x_l, x_p, x_p)))
x = torch.cat((x_l, x2_att), dim=2)
x = self.fc0(x)
x = self.relu(x)

a = self.fc_att(x).squeeze(2)
a = self.tanh(a).unsqueeze(2)
x = torch.mul(x, a)
x = torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1).values], dim=1)
'''

class pocket_structure_net(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=52, hid_dim=128, output_dim1=128, output_dim2=64, dropout=0.2):
        self.num_features_xd = num_features_xd
        self.hid_dim = hid_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.n_output = n_output

        super(pocket_structure_net, self).__init__()

        self.fc00 = nn.Linear(1900, 1024)
        self.fc01 = nn.Linear(1024, 512)
        self.fc02 = nn.Linear(512, 63)

        self.conv1 = GATConv(hid_dim, hid_dim)
        self.conv2 = GATConv(hid_dim, hid_dim)
        self.conv3 = GATConv(hid_dim, hid_dim)
        self.conv4 = GATConv(hid_dim, hid_dim)
        self.conv5 = GATConv(hid_dim, hid_dim)
        self.conv6 = GATConv(hid_dim, hid_dim)


        '''self.conv7 = GCNConv(hid_dim, hid_dim)
        self.conv8 = GCNConv(hid_dim, hid_dim)
        self.conv9 = GCNConv(hid_dim, hid_dim)
        self.vns1 = virtual_node(hid_dim, hid_dim, dropout)
        self.vns2 = virtual_node(hid_dim, hid_dim, dropout)'''


        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.SAGPooling1 = SAGPooling(in_channels=128, ratio=0.9)
        #self.SAGPooling2 = SAGPooling(in_channels=128, ratio=0.9)
        self.se1 = nn.Linear(15168, 2000)#49737
        self.se2 = nn.Linear(2000, 15168)

        self.fc1 = nn.Linear(256, 512)#256,512
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)

        self.out = nn.Linear(64, self.n_output)

        self.w1, self.b1 = nn.Parameter(
            torch.randn(self.num_features_xd, self.hid_dim, requires_grad=True)), nn.Parameter(
            torch.zeros(self.hid_dim,
                        requires_grad=True))
        self.w2, self.b2 = nn.Parameter(
            torch.randn(self.hid_dim, self.hid_dim, requires_grad=True)), nn.Parameter(
            torch.zeros(self.hid_dim,
                        requires_grad=True))

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data1, data2, data3):
        x1, drug_intra, drug_edge_attr, batch1, inter = data1.x, data1.edge_index, data1.edge_attr, data1.batch, data1.inter
        x2, prot_intra, prot_edge_attr, batch2 = data2.x, data2.edge_index, data2.edge_attr, data2.batch

        bs = batch1[-1] + 1
        drug_atoms = int(len(x1)/bs)
        prot_atoms = int(len(x2)/bs)

        x1 = self.node_embed(x1)
        x2 = self.node_embed(x2)

        x11 = self.relu(self.conv1(x1, drug_intra, drug_edge_attr))
        #x1, ami = self.vns1.update_node_emb(x=x11, edge_index=drug_intra, batch=batch1, vx=ami, a=0.1)
        x12 = self.relu(self.conv2(x11, drug_intra, drug_edge_attr))
        #ami = self.vns1.update_vn_emb(x12, batch1, ami)

        #x1, ami = self.vns2.update_node_emb(x=x12, edge_index=drug_intra, batch=batch1, vx=ami, a=0.1)
        x13 = self.relu(self.conv3(x12, drug_intra, drug_edge_attr))
        x1 = x11+x12+x13


        x_l, mask_l = utils.to_dense_batch(x1, batch1)
        #x_l = torch.unsqueeze(x1, 1)

        x2 = self.conv4(x2, prot_intra, prot_edge_attr)
        x21 = self.relu(x2)
        x2 = self.conv5(x21, prot_intra, prot_edge_attr)
        x22 = self.relu(x2)
        x2 = self.conv6(x22, prot_intra, prot_edge_attr)
        x23 = self.relu(x2)
        x2 = x21+x22+x23
        #x2, prot_intra, prot_edge_attr, batch2, prot_numb, _ = self.SAGPooling2(x2, prot_intra, prot_edge_attr, batch2)

        x_p, mask_p = utils.to_dense_batch(x2, batch2)

        '''drug_numb = torch.sort(drug_numb)[0]
        prot_numb = torch.sort(prot_numb)[0]
        drug_numb1 = drug_numb%drug_atoms
        prot_numb1 = prot_numb%prot_atoms

        inter = inter[batch1,drug_numb1]
        inter = torch.chunk(inter,bs,dim=0)
        inter = torch.stack(inter,dim=0)
        inter = inter.transpose(1,2)
        inter = inter[batch2,prot_numb1]
        inter = torch.chunk(inter, bs, dim=0)
        inter = torch.stack(inter, dim=0)
        inter = inter.transpose(1, 2)'''
        #mask_inter = inter <= 20
        #replacement = torch.tensor(float("inf"), dtype=torch.float).to("cuda")
        #inter = torch.where(mask_inter, inter, replacement)

        inter_att = 1 / inter
        inter_att1 = torch.flatten(inter_att, start_dim=1, end_dim=2).unsqueeze(2)
        #inter_att1 = self.sigmoid(inter_att1)
        length1 = len(x_l[1])
        length2 = len(x_p[1])
        x_l1 = x_l.repeat_interleave(length2,dim=1)
        x_p1 = x_p.repeat(1,length1,1)
        xx1 = torch.cat((x_l1, x_p1), dim=2)

        c = torch.mean(xx1, dim=2)
        c = self.relu(self.se1(c))
        c = self.sigmoid(self.se2(c)).unsqueeze(2)
        xx1 = xx1 * c

        xx1 = xx1 * inter_att1

        xx1 = torch.mean(xx1, dim=1)

        '''
        xx1 = list(torch.split(xx1,split_size_or_sections=192,dim=1))
        for i in range(len(xx1)):
            xx1[i] = torch.mean(xx1[i],dim=1)
        xx1 = torch.cat(xx1,dim=0)
        xx1, drug_intra, drug_edge_attr, batch1, _, _ = self.SAGPooling1(xx1, drug_intra, drug_edge_attr, batch1)
        xx1 = global_mean_pool(xx1, batch1)
        #xx1 = torch.mean(xx1, dim=1)
        '''

        h1 = self.fc1(xx1)
        x = self.relu(h1)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        #print(out)

        return out, h1


class pocket_structure_net2(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=52, hid_dim=128, output_dim1=128, output_dim2=64, dropout=0.1):
        self.num_features_xd = num_features_xd
        self.hid_dim = hid_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.n_output = n_output

        super(pocket_structure_net2, self).__init__()

        self.fc00 = nn.Linear(1900, 1024)
        self.fc01 = nn.Linear(1024, 512)
        self.fc02 = nn.Linear(512, 63)

        self.conv1 = GATConv(hid_dim, hid_dim)
        self.conv2 = GATConv(hid_dim, hid_dim)
        self.conv3 = GATConv(hid_dim, hid_dim)
        self.conv4 = GATConv(hid_dim, hid_dim)
        self.conv5 = GATConv(hid_dim, hid_dim)
        self.conv6 = GATConv(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.SAGPooling1 = SAGPooling(in_channels=128, ratio=0.9)

        self.se1 = nn.Linear(15168, 2000)#49737
        self.se2 = nn.Linear(2000, 15168)

        self.fc1 = nn.Linear(256, 512)#256,512
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)

        self.out = nn.Linear(64, self.n_output)

        self.w1, self.b1 = nn.Parameter(
            torch.randn(self.num_features_xd, self.hid_dim, requires_grad=True)), nn.Parameter(
            torch.zeros(self.hid_dim,
                        requires_grad=True))
        self.w2, self.b2 = nn.Parameter(
            torch.randn(self.hid_dim, self.hid_dim, requires_grad=True)), nn.Parameter(
            torch.zeros(self.hid_dim,
                        requires_grad=True))
        self.ica1 = Interaction_Att(hid_dim=self.hid_dim, n_heads=8, dropout=dropout)
        self.get_mask = get_mask()

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data1, data2, data3):
        x1, drug_intra, drug_edge_attr, batch1, inter = data1.x, data1.edge_index, data1.edge_attr, data1.batch, data1.inter
        x2, prot_intra, prot_edge_attr, batch2 = data2.x, data2.edge_index, data2.edge_attr, data2.batch

        x1 = self.node_embed(x1)
        x2 = self.node_embed(x2)

        x11 = self.relu(self.conv1(x1, drug_intra, drug_edge_attr))
        x12 = self.relu(self.conv2(x11, drug_intra, drug_edge_attr))
        x13 = self.relu(self.conv3(x12, drug_intra, drug_edge_attr))
        x1 = x11+x12+x13

        x_l, mask_l = utils.to_dense_batch(x1, batch1)

        x2 = self.conv4(x2, prot_intra, prot_edge_attr)
        x21 = self.relu(x2)
        x2 = self.conv5(x21, prot_intra, prot_edge_attr)
        x22 = self.relu(x2)
        x2 = self.conv6(x22, prot_intra, prot_edge_attr)
        x23 = self.relu(x2)
        x2 = x21+x22+x23

        x_p, mask_p = utils.to_dense_batch(x2, batch2)

        inter1 = self.get_mask(inter)
        x_l, x_p = self.ica1(x_l, x_p, inter1)

        #mask_inter = inter <= 20
        #replacement = torch.tensor(float("inf"), dtype=torch.float).to("cuda")
        #inter = torch.where(mask_inter, inter, replacement)

        inter_att = 1/(inter+1)
        inter_att1 = torch.flatten(inter_att,start_dim=1,end_dim=2).unsqueeze(2)
        #inter_att1 = self.sigmoid(inter_att1)
        length1 = len(x_l[1])
        length2 = len(x_p[1])
        x_l1 = x_l.repeat_interleave(length2,dim=1)
        x_p1 = x_p.repeat(1,length1,1)
        xx1 = torch.cat((x_l1, x_p1), dim=2)

        #c = torch.mean(xx1, dim=2)
        #c = self.relu(self.se1(c))
        #c = self.sigmoid(self.se2(c)).unsqueeze(2)
        #xx1 = xx1 * c

        xx1 = xx1 * inter_att1

        #xx1 = torch.cat([torch.mean(xx1, dim=1), torch.max(xx1, dim=1).values], dim=1)
        xx1 = torch.mean(xx1, dim=1)

        h1 = self.fc1(xx1)
        x = self.relu(h1)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        #print(out)

        return out, h1

class pocket_eazy(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=52, hid_dim=128, output_dim1=128, output_dim2=64, dropout=0.1):
        self.num_features_xd = num_features_xd
        self.hid_dim = hid_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.n_output = n_output

        super(pocket_eazy, self).__init__()

        #self.ca1 = SelfAttention(hid_dim=self.hid_dim//2, n_heads=2, dropout=dropout)
        self.ica1 = Interaction_Att(hid_dim=self.hid_dim//2, n_heads=8, dropout=dropout)
        self.ica2 = Interaction_Att(hid_dim=self.hid_dim // 2, n_heads=8, dropout=dropout)
        self.ln1 = nn.LayerNorm(self.hid_dim//2)

        self.conv1 = GATConv(output_dim2, output_dim2)
        self.conv2 = GATConv(output_dim2, output_dim2)
        self.conv3 = GATConv(output_dim2, output_dim2)
        self.conv4 = GATConv(output_dim2, output_dim2)
        self.conv5 = GATConv(output_dim2, output_dim2)
        self.conv6 = GATConv(output_dim2, output_dim2)

        self.get_mask = get_mask()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.tanh = nn.Tanh()
        self.fc0 = nn.Linear(128, 128)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, self.n_output)

        self.w1, self.b1 = nn.Parameter(
            torch.randn(self.num_features_xd, self.output_dim1, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim1,
                        requires_grad=True))
        self.w2, self.b2 = nn.Parameter(
            torch.randn(self.output_dim1, self.output_dim2, requires_grad=True)), nn.Parameter(
            torch.zeros(self.output_dim2,
                        requires_grad=True))

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data1, data2, data3):
        x1, drug_intra, drug_edge_attr, batch1, inter = data1.x, data1.edge_index, data1.edge_attr, data1.batch, data1.inter
        x2, prot_intra, prot_edge_attr, batch2 = data2.x, data2.edge_index, data2.edge_attr, data2.batch

        x1 = self.node_embed(x1)
        x2 = self.node_embed(x2)

        x11 = self.relu(self.conv1(x1, drug_intra, drug_edge_attr))
        #x1, ami = self.vns1.update_node_emb(x=x11, edge_index=drug_intra, batch=batch1, vx=ami, a=0)
        x12 = self.relu(self.conv2(x11, drug_intra, drug_edge_attr))
        #x1, vx = self.vns2.update_node_emb(x=x12, edge_index=drug_intra, batch=batch1, vx=ami, a=0)
        x13 = self.relu(self.conv3(x12, drug_intra, drug_edge_attr))
        ###x13 = self.virtual_update_node(x=x1, vx=ami, batch=batch1)
        x1 = x11+x12+x13

        x_l, mask_l = utils.to_dense_batch(x1, batch1)
        #x_l = torch.unsqueeze(x1, 1)

        x2 = self.conv4(x2, prot_intra, prot_edge_attr)
        x21 = self.relu(x2)
        x2 = self.conv5(x21, prot_intra, prot_edge_attr)
        x22 = self.relu(x2)
        x2 = self.conv6(x22, prot_intra, prot_edge_attr)
        x23 = self.relu(x2)
        x2 = x21+x22+x23

        x_p, mask_p = utils.to_dense_batch(x2, batch2)

        x_l = torch.cat([torch.mean(x_l, dim=1), torch.max(x_l, dim=1).values], dim=1)
        x_p = torch.cat([torch.mean(x_p, dim=1), torch.max(x_p, dim=1).values], dim=1)

        x = torch.cat((x_l, x_p), dim=1)
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