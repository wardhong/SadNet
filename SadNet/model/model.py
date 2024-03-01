import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,GraphSAGE, global_max_pool as gmp

# 考虑非共价键力
class GCNNetNC(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=53, output_dim1=128, output_dim2=64, dropout=0.2):
        self.num_features_xd = num_features_xd
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        super(GCNNetNC, self).__init__()

        self.n_output = n_output
        self.conv1 = GCNConv(output_dim2, output_dim2)
        self.conv2 = GCNConv(output_dim2, output_dim2*2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(2*output_dim2, 512)
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
        torch.nn.init.kaiming_normal_(self.w1)
        torch.nn.init.kaiming_normal_(self.w2)

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data):
        x, edge_inter, edge_intra, batch = data.x, data.edge_index1,data.edge_index2, data.batch

        x = self.node_embed(x)

        x = self.conv1(x, edge_inter)
        x = self.relu(x)

        x = self.conv2(x, edge_inter)
        x = self.relu(x)

        x = gmp(x, batch)       # global max pooling

        # add some dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out

#考虑共价键力
class GCNNetC(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=53, output_dim1=128, output_dim2=64, dropout=0.2):
        self.num_features_xd = num_features_xd
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        super(GCNNetC, self).__init__()

        self.n_output = n_output
        self.conv1 = GCNConv(output_dim2, output_dim2)
        self.conv2 = GCNConv(output_dim2, output_dim2 * 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(2 * output_dim2, 512)
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
        torch.nn.init.kaiming_normal_(self.w1)
        torch.nn.init.kaiming_normal_(self.w2)

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data):
        x, edge_inter, edge_intra, batch = data.x, data.edge_index1,data.edge_index2, data.batch

        x = self.node_embed(x)

        x = self.conv1(x, edge_intra)
        x = self.relu(x)

        x = self.conv2(x, edge_intra)
        x = self.relu(x)

        x = gmp(x, batch)       # global max pooling

        # add some dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out

#两者都考虑
class GCNNetCNC(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=53, output_dim1=128, output_dim2=64, dropout=0.2):
        self.num_features_xd = num_features_xd
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        super(GCNNetCNC, self).__init__()

        self.n_output = n_output
        self.conv1 = GCNConv(output_dim2, output_dim2*2)
        self.conv2 = GCNConv(output_dim2*2, output_dim2)
        self.conv3 = GCNConv(output_dim2, output_dim2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(output_dim2, 512)
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
        torch.nn.init.kaiming_normal_(self.w1)
        torch.nn.init.kaiming_normal_(self.w2)

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data):
        x, edge_inter, edge_intra, batch = data.x, data.edge_index1,data.edge_index2, data.batch

        x = self.node_embed(x)

        h1 = self.conv1(x, edge_intra)
        h1 = self.relu(h1)
        h1 = self.conv2(h1,edge_intra)
        h1 = x + self.relu(h1)

        h2 = self.conv3(h1, edge_inter)
        h2 = self.relu(h2)
        #h2 = self.conv2(h2, edge_inter)
        h = h1 + self.relu(h2)

        h = gmp(h, batch)

        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        out = self.out(h)
        return out

class GCNNetCNC2(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=53, output_dim1=128, output_dim2=64, dropout=0.2):
        self.num_features_xd = num_features_xd
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        super(GCNNetCNC2, self).__init__()

        self.n_output = n_output
        self.conv1 = GCNConv(output_dim2, output_dim2*2)
        self.conv2 = GCNConv(output_dim2*2, output_dim2)
        self.conv3 = GCNConv(output_dim2, output_dim2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(output_dim2*2, 512)
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
        torch.nn.init.kaiming_normal_(self.w1)
        torch.nn.init.kaiming_normal_(self.w2)

    def node_embed(self,x):
        x = x@self.w1+self.b1
        x = self.relu(x)
        x = x@self.w2+self.b2
        x = self.relu(x)
        return x

    def forward(self, data):
        x, edge_inter, edge_intra, batch = data.x, data.edge_index1,data.edge_index2, data.batch

        x = self.node_embed(x)

        h1 = self.conv1(x, edge_intra)
        h1 = self.relu(h1)
        h1 = self.conv2(h1,edge_intra)
        h1 = self.relu(h1)
        h1 = gmp(h1,batch)

        h2 = self.conv1(x, edge_inter)
        h2 = self.relu(h2)
        h2 = self.conv2(h2, edge_inter)
        h2 = self.relu(h2)
        h2 = gmp(h2, batch)

        h = torch.cat((h1,h2),1)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        out = self.out(h)
        return out