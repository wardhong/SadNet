from torch_geometric.nn import MessagePassing
import torch.nn as nn
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
