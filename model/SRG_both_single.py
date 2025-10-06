import sys

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from .Basemodel import BaseModel
from torch.nn import Identity


class SRG_both_single(BaseModel):
    def __init__(self, in_dim,
                 hidden_dim,
                 out_dim,
                 etypes,
                 num_bases=-1,
                 num_hidden_layers=1,
                 use_self_loop=False,
                 use_residual=False,
                 feat_drop=0.0,
                 edge_drop=0.0,
                 use_self_gating=False,
                 batchnorm=False):
        super(SRG_both_single, self).__init__()
        self.in_dim = in_dim
        self.h_dim = hidden_dim
        self.out_dim = out_dim
        self.rel_names = list(set(etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.use_self_loop = use_self_loop
        self.use_self_gating = use_self_gating


        # self.fc_list = nn.ModuleDict({
        #     key: nn.Linear(in_d, hidden_dim, bias=True) for key, in_d in in_dim.items()
        # })
        #
        # for key, fc in self.fc_list.items():
        #     nn.init.xavier_normal_(fc.weight, gain=1.414)
        #     if fc.bias is not None:
        #         nn.init.zeros_(fc.bias)

        # Feature mapping layers (fc_list with optional gating)
        self.fc_list = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(in_d, hidden_dim, bias=True),  # 特征映射
                nn.Sigmoid() if use_self_gating else nn.Identity()  # 判断是否加 Self-Gating
            ) for key, in_d in in_dim.items()
        })

        for key, module in self.fc_list.items():
            fc = module[0]  # 第一个层是线性变换
            nn.init.xavier_normal_(fc.weight, gain=1.414)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)

        self.layers = nn.ModuleList()
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.rel_names,
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,residual=use_residual,
                feat_drop=feat_drop, edge_drop=edge_drop,batchnorm=batchnorm
                ))
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=None,
            self_loop=self.use_self_loop, residual=use_residual,
        feat_drop=feat_drop, edge_drop=edge_drop))

    def forward(self, hg, processed_features):
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
class RelGraphConvLayer(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 residual=False,
                 feat_drop=0.0,
                 edge_drop=0.,
                 batchnorm=False):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.batchnorm = batchnorm
        self.residual = residual
        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='both', weight=False, bias=False)
            for rel in rel_names
        })
        self.edge_drop_rate = edge_drop
        self.feat_drop = nn.Dropout(feat_drop)
        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                # 使用 Basis 分解
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                # 使用全局共享权重矩阵
                self.shared_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
                nn.init.xavier_uniform_(self.shared_weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))
        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_feat)

        if self.residual:
            if in_feat != out_feat:
                self.res_fc = nn.Linear(in_feat, out_feat, bias=False)  # 调整维度
            else:
                self.res_fc = Identity()  # 直接保留输入特征

    def forward(self, g, inputs):
        g = g.local_var()

        if self.use_weight:
            if self.use_basis:
                # Basis 分解生成共享权重
                shared_weight = self.basis()
            else:
                # 直接使用全局共享权重
                shared_weight = self.shared_weight

            # 为所有关系分配同一个权重
            wdict = {rel: {'weight': shared_weight} for rel in self.rel_names}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        # Drop feature
        inputs_src = {k: self.feat_drop(v) for k, v in inputs_src.items()}

        if self.edge_drop_rate >0:
            all_nodes = {ntype: th.arange(g.num_nodes(ntype), device=g.device) for ntype in g.ntypes}
            edges_to_keep = {
                rel: (th.rand(g.number_of_edges(etype=rel), device=g.device) >= self.edge_drop_rate)
                for rel in g.etypes
            }
            # 生成新的子图
            g = dgl.edge_subgraph(
                g,
                edges={rel: edges_to_keep[rel] for rel in g.etypes}
            )
            # 确保节点未丢失
            try:
                for ntype in g.ntypes:
                    if g.num_nodes(ntype) < len(all_nodes[ntype]):
                        missing_nodes = all_nodes[ntype][~g.has_nodes(all_nodes[ntype])]
                        g.add_nodes(len(missing_nodes), ntype=ntype)
            except Exception as e:
                print("We do not recommend using drop edge for HGNN datasets except for the DMGI and AMHGNN dataset.")
                print(f"Error encountered: {str(e)}")
                exit(1)  # 停止代码运行
        hs = self.conv(g, inputs_src, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            if self.residual:
                h_res = inputs_dst[ntype]
                if self.res_fc is not None:
                    h_res = self.res_fc(h_res)  # 调整维度
                h = h + h_res  # 加残差
            if self.batchnorm:
                h = self.bn(h)
            return h

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}



