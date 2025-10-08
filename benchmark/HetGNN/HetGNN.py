# HetGNN.py
# Minimal, DGL-based HetGNN-style model:
# - Per-node-type content encoders (MLP)
# - Per-relation linear message + mean aggregation
# - Type-level attention to fuse relation messages per dst type
# - Classifier on the target 'category' node type

from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class ContentEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HetGNN(nn.Module):
    """
    Lightweight HetGNN-style encoder.
    Inputs (expected):
      - hg: DGLHeteroGraph
      - node_feats: Dict[str, Tensor] mapping node type -> raw features
    Returns:
      - logits dict with key = target category type (others = None)
    """

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        out_dim: int,
        category: str,
        etypes: List[Tuple[str, str, str]],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.category = category
        self.dropout = nn.Dropout(dropout)

        # per-node-type content encoders
        self.encoders = nn.ModuleDict({
            ntype: ContentEncoder(in_dim, hidden_dim, dropout)
            for ntype, in_dim in in_dims.items()
        })

        # per-relation linear projection on source features
        self.rel_linears = nn.ModuleDict()
        self.rel_dst_groups: Dict[str, List[str]] = {}
        for (src, etype, dst) in etypes:
            key = f"{src}:{etype}:{dst}"
            self.rel_linears[key] = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.rel_dst_groups.setdefault(dst, []).append(key)

        # type-level attention per dst type
        self.type_att = nn.ModuleDict()
        for dst, _rel_keys in self.rel_dst_groups.items():
            self.type_att[dst] = nn.Linear(hidden_dim, 1, bias=False)

        # classifier for target category
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, hg: dgl.DGLHeteroGraph, node_feats: Dict[str, torch.Tensor]):
        # 1) content encoders
        h = {ntype: self.encoders[ntype](feat) for ntype, feat in node_feats.items()}

        # 2) relation-wise aggregation -> type-level attention
        h_new = {ntype: torch.zeros_like(h[ntype]) for ntype in h}

        for dst, rel_keys in self.rel_dst_groups.items():
            rel_msgs = []
            for key in rel_keys:
                src, etype, dsttype = key.split(":")
                g_rel = hg[(src, etype, dsttype)]
                src_h = self.rel_linears[key](h[src])

                with g_rel.local_scope():
                    g_rel.srcdata["h"] = src_h
                    g_rel.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_nei"))
                    msg = g_rel.dstdata["h_nei"]  # (N_dst, d)
                rel_msgs.append(msg)

            if len(rel_msgs) == 0:
                h_comb = h[dst]
            elif len(rel_msgs) == 1:
                h_comb = rel_msgs[0]
            else:
                # (N_dst, R, d)
                stacked = torch.stack(rel_msgs, dim=1)
                # attention over relation axis
                att = self.type_att[dst](stacked)            # (N_dst, R, 1)
                att_w = torch.softmax(att.squeeze(-1), 1)    # (N_dst, R)
                h_comb = (stacked * att_w.unsqueeze(-1)).sum(dim=1)

            # residual + dropout
            h_new[dst] = self.dropout(h_comb + h[dst])

        logits = {ntype: None for ntype in h_new}
        logits[self.category] = self.cls(h_new[self.category])
        return logits


# For compatibility if someone does `from HetGNN import Model`
Model = HetGNN
