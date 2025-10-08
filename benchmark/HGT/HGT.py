import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class HGTLayer(nn.Module):
    def __init__(self, in_dims_per_type, hidden_dim, num_heads, dropout=0.0, norm=True):
        super().__init__()
        self.ntypes = list(in_dims_per_type.keys())
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_Q = nn.ModuleDict()
        self.W_K = nn.ModuleDict()
        self.W_V = nn.ModuleDict()
        for nt in self.ntypes:
            self.W_Q[nt] = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
            self.W_K[nt] = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
            self.W_V[nt] = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)

        self.rel_Wk = nn.ParameterDict()
        self.rel_Wv = nn.ParameterDict()
        self.rel_mu = nn.ParameterDict()
        self._rel_inited = set()

        self.use_norm = norm
        if norm:
            self.norms = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.ntypes})

        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.ModuleDict({nt: nn.Identity() for nt in self.ntypes})

    def _ensure_rel_params(self, rel_key: str):
        if rel_key in self._rel_inited:
            return
        H, d = self.num_heads, self.head_dim
        self.rel_Wk[rel_key] = nn.Parameter(torch.eye(d).repeat(H, 1, 1))
        self.rel_Wv[rel_key] = nn.Parameter(torch.eye(d).repeat(H, 1, 1))
        self.rel_mu[rel_key] = nn.Parameter(torch.ones(H))
        self._rel_inited.add(rel_key)

    def forward(self, hg: dgl.DGLHeteroGraph, h_dict: dict):
        device = next(self.parameters()).device
        H, d = self.num_heads, self.head_dim

        Q_dict, K_dict, V_dict = {}, {}, {}
        for nt in h_dict:
            x = h_dict[nt]
            Q = self.W_Q[nt](x).view(-1, H, d)
            K = self.W_K[nt](x).view(-1, H, d)
            V = self.W_V[nt](x).view(-1, H, d)
            Q_dict[nt], K_dict[nt], V_dict[nt] = Q, K, V

        agg = {nt: torch.zeros(h_dict[nt].shape[0], H, d, device=device) for nt in h_dict}

        for (src, etype, dst) in hg.canonical_etypes:
            rel_key = f"{src}__{etype}__{dst}"
            self._ensure_rel_params(rel_key)
            g = hg[(src, etype, dst)]

            Q = Q_dict[dst]
            K = K_dict[src]
            V = V_dict[src]
            Wk = self.rel_Wk[rel_key].to(device)
            Wv = self.rel_Wv[rel_key].to(device)
            mu = self.rel_mu[rel_key].to(device)

            Kp = torch.einsum("nhd,hde->nhe", K, Wk)
            Vp = torch.einsum("nhd,hde->nhe", V, Wv)
            Qp = Q

            g.srcdata["K"] = Kp
            g.srcdata["V"] = Vp
            g.dstdata["Q"] = Qp

            def _edge_score(edges):
                score = (edges.src["K"] * edges.dst["Q"]).sum(-1) / math.sqrt(d)
                score = score * mu
                return {"score": score}

            g.apply_edges(_edge_score)
            a = edge_softmax(g, g.edata["score"])
            g.edata["a"] = a
            g.update_all(fn.u_mul_e("V", "a", "m"), fn.sum("m", "h_dst"))

            if "h_dst" in g.dstdata:
                agg[dst] = agg[dst] + g.dstdata.pop("h_dst")

            for k in ["K", "V"]:
                g.srcdata.pop(k, None)
            g.dstdata.pop("Q", None)
            g.edata.pop("score", None)
            g.edata.pop("a", None)

        out = {}
        for nt in h_dict:
            m = agg[nt].reshape(agg[nt].shape[0], -1)
            res = self.res_proj[nt](h_dict[nt])
            y = m + res
            if self.use_norm:
                y = self.norms[nt](y)
            y = F.elu(self.dropout(y))
            out[nt] = y
        return out


class HGT(nn.Module):
    def __init__(self, in_dims, hidden_dim, out_dim, category, num_heads=8, num_layers=2, dropout=0.0, norm=True, **kwargs):
        super().__init__()
        self.category = category
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.in_proj = nn.ModuleDict({nt: nn.Linear(in_dims[nt], hidden_dim) for nt in in_dims})
        self.layers = nn.ModuleList([HGTLayer(in_dims, hidden_dim, num_heads, dropout=dropout, norm=norm) for _ in range(num_layers)])

        self.ffn = nn.ModuleDict()
        self.ffn_norm = nn.ModuleDict() if norm else None
        ffn_hidden = hidden_dim * 2
        for nt in in_dims:
            self.ffn[nt] = nn.Sequential(
                nn.Linear(hidden_dim, ffn_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden, hidden_dim),
                nn.Dropout(dropout),
            )
        if norm:
            self.ffn_norm = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in in_dims})

        self.cls = nn.Linear(hidden_dim, out_dim)

    def forward(self, hg: dgl.DGLHeteroGraph, feats: dict):
        h = {nt: F.elu(self.in_proj[nt](feats[nt])) for nt in feats}
        for layer in self.layers:
            h = layer(hg, h)
            new_h = {}
            for nt, x in h.items():
                y = self.ffn[nt](x)
                y = x + y
                if self.ffn_norm is not None:
                    y = self.ffn_norm[nt](y)
                new_h[nt] = y
            h = new_h
        logits = self.cls(h[self.category])
        return {self.category: logits}
