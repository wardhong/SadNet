import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.nn import GCNConv,GraphSAGE,GATConv, global_max_pool as gmp
import torch.nn.functional as F
from torch import matmul,Tensor
from typing import Union,Tuple

class SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')

        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = nn.Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = nn.Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()



    def forward(self, x: Union[Tensor], edge_index,size = None) -> Tensor:
        #if isinstance(x, Tensor):
            #x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j


    def message_and_aggregate(self, adj_t,x) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)


# 考虑非共价键力
class SAGENC(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=53, output_dim1=128, output_dim2=64, dropout=0.2):
        self.num_features_xd = num_features_xd
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        super(SAGENC, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(in_channels=output_dim2, out_channels=output_dim2)
        self.conv2 = GCNConv(in_channels=output_dim2, out_channels=output_dim2*2)

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