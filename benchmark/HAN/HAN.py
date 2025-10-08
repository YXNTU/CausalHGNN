import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dgl.nn.pytorch import HANConv  # DGL >= 0.9 usually has this
    _HAS_HANCONV = True
except Exception:
    _HAS_HANCONV = False

import dgl
from dgl.nn.pytorch import GATConv


class _MetaPathGAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_heads):
        super().__init__()
        self.gat1 = GATConv(in_dim, hid_dim, num_heads=num_heads, allow_zero_in_degree=True)
        self.gat2 = GATConv(hid_dim * num_heads, out_dim, num_heads=1, allow_zero_in_degree=True)

    def forward(self, g, x):
        h = self.gat1(g, x)
        h = F.elu(h.flatten(1))
        h = self.gat2(g, h).squeeze(1)
        return h


class _HANFallback(nn.Module):
    def __init__(self, in_dims: dict, hidden_dim: int, out_dim: int,
                 category: str, metapaths, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.category = category
        self.metapaths = metapaths
        self.proj = nn.ModuleDict({nt: nn.Linear(in_dims[nt], hidden_dim) for nt in in_dims})
        self.mp_gats = nn.ModuleList(
            [_MetaPathGAT(hidden_dim, hidden_dim // num_heads if hidden_dim >= num_heads else hidden_dim,
                          hidden_dim, num_heads=num_heads) for _ in metapaths]
        )
        self.attn = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_dim, out_dim)

    def forward(self, hg: dgl.DGLHeteroGraph, feats: dict):
        h_proj = {nt: F.elu(self.proj[nt](feats[nt])) for nt in feats}
        embs = []
        for mp, block in zip(self.metapaths, self.mp_gats):
            g_mp = dgl.metapath_reachable_graph(hg, mp)
            x = h_proj[self.category]
            embs.append(block(g_mp, x))
        H = torch.stack(embs, dim=1)
        alpha = torch.softmax(self.attn(self.dropout(H)).squeeze(-1), dim=1)
        h = (alpha.unsqueeze(-1) * H).sum(dim=1)
        out = self.cls(self.dropout(h))
        return {self.category: out}


class HAN(nn.Module):
    def __init__(self, in_dim: dict, hidden_dim: int, out_dim: int,
                 category: str, metapaths, num_heads: int = 8, num_hidden_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.category = category
        self.metapaths = metapaths
        self.dropout = nn.Dropout(dropout)
        if _HAS_HANCONV:
            self.proj = nn.ModuleDict({nt: nn.Linear(in_dim[nt], hidden_dim) for nt in in_dim})
            layers = []
            input_dim = hidden_dim
            for _ in range(max(1, num_hidden_layers)):
                layers.append(HANConv(in_feats=input_dim,
                                      out_feats=hidden_dim,
                                      num_heads=num_heads,
                                      metapaths=metapaths,
                                      bias=True))
                input_dim = hidden_dim * num_heads
            self.han_layers = nn.ModuleList(layers)
            self.cls = nn.Linear(input_dim, out_dim)
            self.use_fallback = False
        else:
            self.fallback = _HANFallback(in_dim, hidden_dim, out_dim, category, metapaths,
                                         num_heads=num_heads, dropout=dropout)
            self.use_fallback = True

    def forward(self, hg: dgl.DGLHeteroGraph, feats: dict):
        if self.use_fallback:
            return self.fallback(hg, feats)
        h = {nt: F.elu(self.proj[nt](feats[nt])) for nt in feats}
        x = h
        for layer in self.han_layers:
            x = layer(hg, x)
            x = {nt: x_nt.mean(1) for nt, x_nt in x.items()}
            x = {nt: F.elu(self.dropout(x_nt)) for nt, x_nt in x.items()}
        logits = self.cls(x[self.category])
        return {self.category: logits}
