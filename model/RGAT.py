import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from .Basemodel import BaseModel

class RGAT(BaseModel):
    def __init__(self, in_dim,
                 out_dim,
                 h_dim,
                 etypes,
                 num_heads,
                 num_hidden_layers=1,
                 dropout=0):
        super(RGAT, self).__init__()
        self.rel_names = etypes
        self.layers = nn.ModuleList()

        self.fc_list = nn.ModuleDict({
            key: nn.Linear(in_d, h_dim, bias=True) for key, in_d in in_dim.items()
        })

        for key, fc in self.fc_list.items():
            nn.init.xavier_normal_(fc.weight, gain=1.414)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)

        # # input 2 hidden
        self.layers.append(RGATLayer(
            h_dim, h_dim, num_heads, self.rel_names, activation=F.relu, dropout=dropout, last_layer_flag=False))
        for i in range(num_hidden_layers):
            self.layers.append(RGATLayer(
                h_dim*num_heads, h_dim, num_heads, self.rel_names, activation=F.relu, dropout=dropout, last_layer_flag=False
            ))
        self.layers.append(RGATLayer(
            h_dim*num_heads, out_dim, num_heads, self.rel_names, activation=None, last_layer_flag=True))
        return

    def forward(self, hg, processed_features=None):

        h_dict = {}
        for key, feature in processed_features.items():
            if key in self.fc_list:
                h_dict[key] = self.fc_list[key](feature)
            else:
                raise KeyError(f"Feature '{key}' does not have a corresponding FC layer in fc_list.")

        if hasattr(hg, 'ntypes'):
            # full graph training,
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
            # minibatch training, block
            for layer, block in zip(self.layers, hg):
                h_dict = layer(block, h_dict)
        return h_dict


class RGATLayer(nn.Module):

    def __init__(self,
                 in_feat,
                 out_feat,
                 num_heads,
                 rel_names,
                 activation=None,
                 dropout=0.0,
                 last_layer_flag=False,
                 bias=True):
        super(RGATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.last_layer_flag=last_layer_flag
        self.conv = dglnn.HeteroGraphConv({
            rel: dgl.nn.pytorch.GATConv(in_feat, out_feat, num_heads=num_heads, bias=bias, allow_zero_in_degree=True)
            for rel in rel_names
        })

    def forward(self, g, h_dict):
        h_dict = self.conv(g, h_dict)
        out_put = {}
        for n_type, h in h_dict.items():
            if self.last_layer_flag:
                h = h.mean(1)
            else:
                h = h.flatten(1)
            out_put[n_type] = h.squeeze()
        return out_put