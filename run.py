import sys

from torch import nn

from model.SRG_both_rel import SRG_both_rel
from model.SRG_both_single import SRG_both_single
from model.SRG_right_single import SRG_right_single
from model.RGCN import RGCN
from model.GCN import GCN
from model.GCN_both_rel import GCN_both_rel
from model.GCN_right_rel import GCN_right_rel
from model.GCN_both_single import GCN_both_single
from model.GCN_right_single import GCN_right_single
sys.path.append('../../')
import time
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from utils.pytorchtools import EarlyStopping
import dgl
from sklearn.metrics import f1_score




def load_pth(input_path: str, use_homo: bool, simple: int):
    """
    根据输入路径动态解析文件夹，并自动加载 `processed_*.pth` 或 `homo_processed_*.pth` 文件。
    """
    # 去除输入路径中的多余空格或换行符
    input_path = input_path.strip()

    # 基础目录，假设与当前脚本同级的 'data' 目录
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 分离文件夹和文件名
    folder, file_name = input_path.split("/")
    target_dir = os.path.join(base_dir, "data", folder)

    # 根据 `use_homo` 决定前缀
    if use_homo and simple >= 0:
        prefix = "di_homo_processed_"
    elif use_homo and simple == -1:
        prefix = "homo_processed_"
    else:
        prefix = "processed_"
    # prefix = "homo_processed_" if use_homo else "processed_"

    # 构建完整的文件路径
    pth_file = os.path.join(target_dir, f"{prefix}{file_name}.pth")

    # 检查文件夹是否存在
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Directory not found: {target_dir}")

    # 检查目标文件是否存在
    if not os.path.exists(pth_file):
        raise FileNotFoundError(f"File not found: {pth_file}")

    print(f"Loading file: {pth_file}")

    # 加载 .pth 文件
    data = torch.load(pth_file)
    return data


def process_features(node_features, target_key, feats_type, device):
    """
    根据指定的逻辑处理字典形式的特征。

    参数：
        node_features (dict): 字典，包含所有特征。键是特征名称，值是特征矩阵。
        target_key (str): 目标节点对应的键，例如 'category'。
        feats_type (int): 特征处理类型。
        device (torch.device): 目标设备。

    返回：
        processed_features (dict): 处理后的特征字典。
        in_dims (dict): 每个特征的输入维度字典。
    """
    processed_features = {}
    in_dims = {}

    for key, features in node_features.items():
        if key == target_key:
            # 处理目标节点特征
            if feats_type == 0 or feats_type == 1 or feats_type == 2 or feats_type == 5:
                in_dims[key] = features.shape[1]  # 保留特征维度
                processed_features[key] = features.to(device)  # 保留原始特征
            elif feats_type == 3:
                # 对于 feats_type == 3 的目标节点特征，处理为单位向量
                dim = features.shape[0]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                in_dims[key] = dim  # 单位向量的维度为节点数
                processed_features[key] = torch.sparse_coo_tensor(indices, values, torch.Size([dim, dim]),
                                                                  device=device, dtype=torch.float32)
            elif feats_type == 4:
                in_dims[key] = 64
                processed_features[key] = torch.empty(features.shape[0], 64).uniform_(-1, 1).to(device)
        else:
            # 处理非目标节点特征
            if feats_type == 0:
                # 原始特征
                in_dims[key] = features.shape[1]
                processed_features[key] = features.to(device)
            elif feats_type == 1 or feats_type == 5:
                # 零向量 (feats_type == 1 或 5)
                in_dims[key] = 10  # 固定特征维度为10
                processed_features[key] = torch.zeros((features.shape[0], 10), device=device)
            elif feats_type == 2 or feats_type == 4:
                # 单位向量 (feats_type == 2 或 4)
                dim = features.shape[0]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                in_dims[key] = dim  # 单位向量的维度为节点数
                processed_features[key] = torch.sparse_coo_tensor(indices, values, torch.Size([dim, dim]),
                                                                  device=device, dtype=torch.float32)
            elif feats_type == 3:
                # 所有特征为单位向量
                dim = features.shape[0]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                in_dims[key] = dim  # 单位向量的维度为节点数
                processed_features[key] = torch.sparse_coo_tensor(indices, values, torch.Size([dim, dim]),
                                                                  device=device, dtype=torch.float32)
            elif feats_type == 4:
                in_dims[key] = 64
                processed_features[key] = torch.empty(features.shape[0], 64).uniform_(-1, 1).to(device)
    return processed_features, in_dims



import dgl
import torch
from typing import List

def find_k_hop_metapaths(hg: dgl.DGLHeteroGraph, category: str, K: int) -> List[List[str]]:
    """
    Recursively find all metapaths of length K starting and ending at `category`.
    """
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
    """
    Compute the K-hop heterophily rate for a given node type in a heterogeneous DGL graph.
    Compatible with older DGL versions (only accept one metapath per call).
    """
    metapaths = find_k_hop_metapaths(hg, category, K)

    if not metapaths:
        print(f"[Warning] No valid {K}-hop metapaths for {category}")
        return 0.0

    print(f"[Info] Found {len(metapaths)} {K}-hop metapaths for '{category}':")
    for mp in metapaths:
        print("  -", mp)

    # Collect edges
    all_src = []
    all_dst = []

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
    # ratio = compute_khop_heterophily_dgl(hg,category,labels,6)
    if args.homo:
        if feats_type == 4:
            processed_features = [torch.empty(f.shape[0], 64).uniform_(-1, 1).to(device) for f in node_features]
            in_dims = [64 for f in node_features]
        else:
            processed_features = [f.to(device) for f in node_features]
            in_dims = [f.shape[1] for f in node_features]
    else:
        processed_features, in_dims = process_features(node_features, category, feats_type, device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    test_ma_F1 = []
    test_mi_F1 = []
    etypes = [etype[1] for etype in hg.canonical_etypes]

    # 仅统计 category 节点（有标签的这一类），而不是整图的所有节点
    n_category = labels.shape[0]  # e.g., 3025
    labeled_idx = torch.unique(torch.cat([  # 训练/验证/测试三者并集
        train_idx, val_idx, test_idx
    ])).to(device)

    # 每个 category 节点在 N 次重复中“预测正确的轮次”
    correct_counts = torch.zeros(n_category, dtype=torch.long)

    for fre in range(args.repeat):
        if args.homo:
            net = GCN(in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                      num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                      feat_drop=args.feat_drop, edge_drop=args.edge_drop).to(device)
            if args.simple == 0:
                net = GCN_both_rel(in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc,
                                   etypes=etypes,
                                   num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                                   feat_drop=args.feat_drop, edge_drop=args.edge_drop).to(device)
            elif args.simple == 1:
                net = GCN_both_single(in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc,
                                   etypes=etypes,
                                   num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                                   feat_drop=args.feat_drop, edge_drop=args.edge_drop).to(device)
            elif args.simple == 2:
                net = GCN_right_rel(in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc,
                                   etypes=etypes,
                                   num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                                   feat_drop=args.feat_drop, edge_drop=args.edge_drop).to(device)
            elif args.simple == 3:
                net = GCN_right_single(in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc,
                                   etypes=etypes,
                                   num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                                   feat_drop=args.feat_drop, edge_drop=args.edge_drop).to(device)
        elif args.simple == 0:
            net = SRG_both_rel(in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                       num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                       feat_drop=args.feat_drop, edge_drop=args.edge_drop).to(device)
        elif args.simple == 1:
            net = SRG_both_single(in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                       num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                       feat_drop=args.feat_drop, edge_drop=args.edge_drop).to(device)
        elif args.simple == 2:
            net = SRG_right_single(in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                       num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                       feat_drop=args.feat_drop, edge_drop=args.edge_drop).to(device)
        else:
            net = RGCN(in_dim=in_dims, hidden_dim=args.hidden_dim, out_dim=num_classes + args.vc, etypes=etypes,
                       num_hidden_layers=args.num_layers, use_residual=args.use_residual,
                       feat_drop=args.feat_drop, edge_drop=args.edge_drop).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                       save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, fre))


        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()
            logits = net(hg, processed_features)
            logits = logits[category]
            train_loss = criterion(logits[train_idx], labels[train_idx])

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, train_loss.item(), t_end - t_start))

            t_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                logits = net(hg, processed_features)
                logits = logits[category]
                val_loss = criterion(logits[val_idx], labels[val_idx])
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, fre)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits = net(hg, processed_features)
            logits = logits[category]

            y_test_true = labels[test_idx].cpu().numpy()
            y_test_pred = logits[test_idx].argmax(dim=1).cpu().numpy()

            macro_f1_test = f1_score(y_test_true, y_test_pred, average='macro')
            micro_f1_test = f1_score(y_test_true, y_test_pred, average='micro')

            final_metrics = {
                'macro_f1_test': macro_f1_test,
                'micro_f1_test': micro_f1_test
            }
            print("\nFinal Results:")
            for key, value in final_metrics.items():
                print(f"{key}: {value:.4f}")
            test_ma_F1.append(macro_f1_test)
            test_mi_F1.append(micro_f1_test)

        net.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, fre)))
        net.eval()
        with torch.no_grad():
            logits = net(hg, processed_features)[category]  # 形状: [n_category, num_classes]
            y_pred = logits.argmax(dim=1)  # 形状: [n_category]

        # 只在有标签的节点上比较
        li = labeled_idx.to(y_pred.device).long()  # train/val/test 的并集
        # 一些健壮性检查（如果索引越界就跳过累计并提示）
        if li.numel() > 0 and li.max().item() < y_pred.shape[0] and li.max().item() < labels.shape[0]:
            cm_li = (y_pred[li] == labels[li]).to(torch.long).cpu()  # 逐点正确与否（0/1）
            # 将本轮的正确结果累计回到全长度的计数器上
            per_repeat_correct = torch.zeros_like(correct_counts)
            per_repeat_correct[li.cpu()] = cm_li
            correct_counts += per_repeat_correct
        else:
            print(
                f"[Warn] labeled_idx contains indices ≥ y_pred({y_pred.shape[0]}) or labels({labels.shape[0]}), skip counting this repeat.")

    import statistics
    avg_ma_F1 = statistics.mean(test_ma_F1)
    avg_mi_F1 = statistics.mean(test_mi_F1)
    print("Average Macro F1:", avg_ma_F1)
    print("Average Micro F1:", avg_mi_F1)

    # 计算“被预测正确的概率”，并保留 1 位小数
    probs = (correct_counts.float() / args.repeat).numpy()
    li = labeled_idx.detach().cpu().numpy()
    probs_li = np.round(probs[li], 1)

    # 目录：results\homo 或 results\hete
    out_dir = os.path.join('results', 'homo' if args.homo else 'hete')
    os.makedirs(out_dir, exist_ok=True)

    # 文件名：把 dataset 里的斜杠替换为下划线
    dataset_name = args.dataset.replace('/', '_')
    out_path = os.path.join(out_dir, f'{dataset_name}.csv')

    # 写 CSV，两列：node, prob（只导出有标签的节点）
    import pandas as pd
    df_out = pd.DataFrame({'node': li, 'prob': probs_li})
    df_out.to_csv(out_path, index=False, float_format='%.1f')
    print(f"[Saved] {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats_type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                         '4 - only term features (id vec for others);' +
                         '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num_layers', type=int, default=0)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--feat_drop', type=float, default=0.)
    ap.add_argument('--edge_drop', type=float, default=0.)
    ap.add_argument('--weight_decay', type=float, default=0.)
    ap.add_argument('--dataset', type=str, default='DMGI/ACM')
    ap.add_argument('--num_heads', type=int, default=8)
    ap.add_argument('--use_residual', action='store_true', help='Use residual connections in the model')
    ap.add_argument('--use_self_gating', action='store_true', help='Use Self Gating in the model')
    ap.add_argument('--batchnorm', action='store_true', help='Use batch norm')
    ap.add_argument('--vc', type=int, default=0, help='Virtual Class')
    ap.add_argument('--homo', action='store_true', help='Use Self Gating in the model')
    ap.add_argument('--simple', type=int, default=-1)
    args = ap.parse_args()
    run(args)
