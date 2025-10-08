import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import datetime
import time
import statistics
from typing import List

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
import dgl

from model.SRG_both_rel import SRG_both_rel
from model.SRG_both_single import SRG_both_single
from model.SRG_right_single import SRG_right_single
from model.RGCN import RGCN
from model.GCN import GCN
from model.GCN_both_rel import GCN_both_rel
from model.GCN_right_rel import GCN_right_rel
from model.GCN_both_single import GCN_both_single
from model.GCN_right_single import GCN_right_single
from utils.pytorchtools import EarlyStopping


def load_pth(input_path: str, use_homo: bool, simple: int):
    input_path = input_path.strip()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder, file_name = input_path.split("/", 1)
    target_dir = os.path.join(base_dir, "data", folder)
    if use_homo and simple >= 0:
        prefix = "di_homo_processed_"
    elif use_homo and simple == -1:
        prefix = "homo_processed_"
    else:
        prefix = "processed_"
    pth_file = os.path.join(target_dir, f"{prefix}{file_name}.pth")
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Directory not found: {target_dir}")
    if not os.path.exists(pth_file):
        raise FileNotFoundError(f"File not found: {pth_file}")
    print(f"Loading file: {pth_file}")
    return torch.load(pth_file)


def process_features(node_features, target_key, feats_type, device):
    processed_features = {}
    in_dims = {}
    for key, features in node_features.items():
        if key == target_key:
            if feats_type in (0, 1, 2, 5):
                in_dims[key] = features.shape[1]
                processed_features[key] = features.to(device)
            elif feats_type == 3:
                dim = features.shape[0]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                in_dims[key] = dim
                processed_features[key] = torch.sparse_coo_tensor(
                    indices, values, torch.Size([dim, dim]), device=device, dtype=torch.float32
                )
            elif feats_type == 4:
                in_dims[key] = 64
                processed_features[key] = torch.empty(features.shape[0], 64).uniform_(-1, 1).to(device)
        else:
            if feats_type == 0:
                in_dims[key] = features.shape[1]
                processed_features[key] = features.to(device)
            elif feats_type in (1, 5):
                in_dims[key] = 10
                processed_features[key] = torch.zeros((features.shape[0], 10), device=device)
            elif feats_type in (2, 3, 4):
                dim = features.shape[0]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                in_dims[key] = dim
                processed_features[key] = torch.sparse_coo_tensor(
                    indices, values, torch.Size([dim, dim]), device=device, dtype=torch.float32
                )
    return processed_features, in_dims


def find_k_hop_metapaths(hg: dgl.DGLHeteroGraph, category: str, K: int) -> List[List[str]]:
    paths = []

    def dfs(current_type, path, depth):
        if depth == K:
            if current_type == category:
                paths.append(path)
            return
        for (src, etype, dst) in hg.canonical_etypes:
            if src == current_type:
                dfs(dst, path + [etype], depth + 1)

    dfs(category, [], 0)
    return paths


def compute_khop_heterophily_dgl(hg: dgl.DGLHeteroGraph, category: str, labels: torch.Tensor, K: int) -> float:
    metapaths = find_k_hop_metapaths(hg, category, K)
    if not metapaths:
        print(f"[Warning] No valid {K}-hop metapaths for {category}")
        return 0.0
    print(f"[Info] Found {len(metapaths)} {K}-hop metapaths for '{category}':")
    for mp in metapaths:
        print("  -", mp)
    all_src, all_dst = [], []
    for path in metapaths:
        try:
            subgraph = dgl.metapath_reachable_graph(hg, path)
            subgraph = subgraph.remove_self_loop()
            src, dst = subgraph.edges()
            all_src.append(src)
            all_dst.append(dst)
        except Exception as e:
            print(f"[Warning] Skipping metapath {path} due to error: {e}")
    if len(all_src) == 0:
        return 0.0
    merged_src = torch.cat(all_src)
    merged_dst = torch.cat(all_dst)
    label_src = labels[merged_src]
    label_dst = labels[merged_dst]
    mismatches = (label_src != label_dst).sum().item()
    return mismatches / len(merged_src)


def run(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    feats_type = args.feats_type

    data = load_pth(args.dataset, args.homo, args.simple)
    hg = data['hg'].to(device)
    labels = data['labels'].to(device)
    num_classes = data['num_classes']
    train_idx = data['train_idx'].to(device)
    val_idx = data['val_idx'].to(device)
    test_idx = data['test_idx'].to(device)
    node_features = data['node_features']
    category = data['category']

    if args.homo:
        if feats_type == 4:
            processed_features = [torch.empty(f.shape[0], 64).uniform_(-1, 1).to(device) for f in node_features]
            in_dims = [64 for _ in node_features]
        else:
            processed_features = [f.to(device) for f in node_features]
            in_dims = [f.shape[1] for f in node_features]
    else:
        processed_features, in_dims = process_features(node_features, category, feats_type, device)

    criterion = nn.CrossEntropyLoss()
    test_ma_F1 = []
    test_mi_F1 = []
    etypes = [etype[1] for etype in hg.canonical_etypes]

    n_category = labels.shape[0]
    labeled_idx = torch.unique(torch.cat([train_idx, val_idx, test_idx])).to(device)
    correct_counts = torch.zeros(n_category, dtype=torch.long)

    os.makedirs("checkpoint", exist_ok=True)
    safe_ds = args.dataset.replace("/", "_")

    for fre in range(args.repeat):
        if args.homo:
            net = GCN(
                in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                feat_drop=args.feat_drop, edge_drop=args.edge_drop
            ).to(device)
            if args.simple == 0:
                net = GCN_both_rel(
                    in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                    num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                    feat_drop=args.feat_drop, edge_drop=args.edge_drop
                ).to(device)
            elif args.simple == 1:
                net = GCN_both_single(
                    in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                    num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                    feat_drop=args.feat_drop, edge_drop=args.edge_drop
                ).to(device)
            elif args.simple == 2:
                net = GCN_right_rel(
                    in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                    num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                    feat_drop=args.feat_drop, edge_drop=args.edge_drop
                ).to(device)
            elif args.simple == 3:
                net = GCN_right_single(
                    in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                    num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                    feat_drop=args.feat_drop, edge_drop=args.edge_drop
                ).to(device)
        elif args.simple == 0:
            net = SRG_both_rel(
                in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                feat_drop=args.feat_drop, edge_drop=args.edge_drop
            ).to(device)
        elif args.simple == 1:
            net = SRG_both_single(
                in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                feat_drop=args.feat_drop, edge_drop=args.edge_drop
            ).to(device)
        elif args.simple == 2:
            net = SRG_right_single(
                in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                feat_drop=args.feat_drop, edge_drop=args.edge_drop
            ).to(device)
        else:
            net = RGCN(
                in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                feat_drop=args.feat_drop, edge_drop=args.edge_drop
            ).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ckpt_path = os.path.join("checkpoint", f"checkpoint_{safe_ds}_{fre}.pt")
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=ckpt_path)

        for epoch in range(args.epoch):
            t_start = time.time()
            net.train()
            logits = net(hg, processed_features)[category]
            train_loss = criterion(logits[train_idx], labels[train_idx])
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            t_end = time.time()
            print(f"Epoch {epoch:05d} | Train_Loss: {train_loss.item():.4f} | Time: {t_end - t_start:.4f}")

            t_start = time.time()
            net.eval()
            with torch.no_grad():
                logits = net(hg, processed_features)[category]
                val_loss = criterion(logits[val_idx], labels[val_idx])
            t_end = time.time()
            print(f"Epoch {epoch:05d} | Val_Loss {val_loss.item():.4f} | Time(s) {t_end - t_start:.4f}")
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print("Early stopping!")
                break

        net.load_state_dict(torch.load(ckpt_path))
        net.eval()
        with torch.no_grad():
            logits = net(hg, processed_features)[category]
            y_test_true = labels[test_idx].cpu().numpy()
            y_test_pred = logits[test_idx].argmax(dim=1).cpu().numpy()
            macro_f1_test = f1_score(y_test_true, y_test_pred, average='macro')
            micro_f1_test = f1_score(y_test_true, y_test_pred, average='micro')
            print("\nFinal Results:")
            print(f"macro_f1_test: {macro_f1_test:.4f}")
            print(f"micro_f1_test: {micro_f1_test:.4f}")
            test_ma_F1.append(macro_f1_test)
            test_mi_F1.append(micro_f1_test)

        net.load_state_dict(torch.load(ckpt_path))
        net.eval()
        with torch.no_grad():
            y_pred = net(hg, processed_features)[category].argmax(dim=1)
        li = labeled_idx.to(y_pred.device).long()
        if li.numel() > 0 and li.max().item() < y_pred.shape[0] and li.max().item() < labels.shape[0]:
            cm_li = (y_pred[li] == labels[li]).to(torch.long).cpu()
            per_repeat_correct = torch.zeros_like(correct_counts)
            per_repeat_correct[li.cpu()] = cm_li
            correct_counts += per_repeat_correct
        else:
            print("[Warn] labeled_idx out of range, skip counting this repeat.")

    avg_ma_F1 = statistics.mean(test_ma_F1)
    avg_mi_F1 = statistics.mean(test_mi_F1)
    print("Average Macro F1:", avg_ma_F1)
    print("Average Micro F1:", avg_mi_F1)

    base_result_dir = "results"
    dataset_dir = os.path.join(base_result_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(dataset_dir, f"run_{timestamp}.txt")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("=== Experiment Results ===\n\n")
        f.write(">> Hyperparameters:\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\n>> Test Results:\n")
        for i, (ma, mi) in enumerate(zip(test_ma_F1, test_mi_F1)):
            f.write(f"Run {i + 1}: Macro F1 = {ma:.4f}, Micro F1 = {mi:.4f}\n")
        avg_ma_F1 = statistics.mean(test_ma_F1)
        avg_mi_F1 = statistics.mean(test_mi_F1)
        f.write(f"\nAverage Macro F1: {avg_ma_F1:.4f}\n")
        f.write(f"Average Micro F1: {avg_mi_F1:.4f}\n")

    print(f"Results saved to {result_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats_type', type=int, default=0)
    ap.add_argument('--hidden_dim', type=int, default=64)
    ap.add_argument('--epoch', type=int, default=300)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--repeat', type=int, default=5)
    ap.add_argument('--num_layers', type=int, default=0)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--feat_drop', type=float, default=0.0)
    ap.add_argument('--edge_drop', type=float, default=0.0)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--dataset', type=str, default='DMGI/ACM')
    ap.add_argument('--num_heads', type=int, default=8)
    ap.add_argument('--use_residual', action='store_true')
    ap.add_argument('--use_self_gating', action='store_true')
    ap.add_argument('--batchnorm', action='store_true')
    ap.add_argument('--vc', type=int, default=0)
    ap.add_argument('--homo', action='store_true')
    ap.add_argument('--simple', type=int, default=-1)
    args = ap.parse_args()
    run(args)
