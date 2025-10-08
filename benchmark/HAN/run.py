import os
import time
import datetime
import statistics

import torch
import torch.nn as nn
import dgl
from sklearn.metrics import f1_score

from HAN import HAN
from pytorchtools import EarlyStopping


# ===== Fixed dataset path (no CLI args needed) =====
# Assumes processed dataset is stored at <repo_root>/data/DMGI/processed_ACM.pth
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PTH_PATH = os.path.join(REPO_ROOT, "data", "DMGI", "processed_ACM.pth")

# ===== Hyperparameters consistent with the official HAN (ACM) implementation =====
HIDDEN_DIM = 64
NUM_HEADS = 8
NUM_LAYERS = 1
DROPOUT = 0.6
LR = 5e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 30
REPEAT = 5


def load_fixed_pth():
    """Load the fixed ACM dataset."""
    if not os.path.exists(PTH_PATH):
        raise FileNotFoundError(f"Processed dataset not found: {PTH_PATH}")
    return torch.load(PTH_PATH)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = load_fixed_pth()

    hg: dgl.DGLHeteroGraph = data["hg"].to(device)
    labels = data["labels"].to(device)
    num_classes = data["num_classes"]
    train_idx = data["train_idx"].to(device)
    val_idx = data["val_idx"].to(device)
    test_idx = data["test_idx"].to(device)
    node_features = {k: v.to(device) for k, v in data["node_features"].items()}
    in_dims = {k: v.shape[1] for k, v in node_features.items()}
    category = data["category"]

    # ===== Use projected 1-hop meta-paths (PAP / PLP) =====
    # '0' corresponds to PAP, '1' corresponds to PLP
    metapaths = [['0'], ['1']]
    print(f"[INFO] Using projected meta-paths: {metapaths}  # '0'->PAP, '1'->PLP")

    criterion = nn.CrossEntropyLoss()
    macro_list, micro_list = [], []

    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)

    for r in range(REPEAT):
        model = HAN(
            in_dim=in_dims,
            hidden_dim=HIDDEN_DIM,
            out_dim=num_classes,
            category=category,
            metapaths=metapaths,
            num_heads=NUM_HEADS,
            num_hidden_layers=max(1, NUM_LAYERS),
            dropout=DROPOUT,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        ckpt_path = os.path.join(ckpt_dir, f"rep{r}.pt")
        stopper = EarlyStopping(patience=PATIENCE, verbose=True, save_path=ckpt_path)

        for epoch in range(EPOCHS):
            t0 = time.time()
            model.train()
            logits = model(hg, node_features)[category]
            loss = criterion(logits[train_idx], labels[train_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t1 = time.time()

            print(f"Epoch {epoch:04d} | Train loss {loss.item():.4f} | {(t1 - t0):.3f}s")

            model.eval()
            with torch.no_grad():
                val_logits = model(hg, node_features)[category]
                val_loss = criterion(val_logits[val_idx], labels[val_idx])
            print(f"Epoch {epoch:04d} | Val loss {val_loss.item():.4f}")

            stopper(val_loss, model)
            if stopper.early_stop:
                print("Early stopping.")
                break

        # ===== Evaluation =====
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        with torch.no_grad():
            test_logits = model(hg, node_features)[category]
            y_true = labels[test_idx].cpu().numpy()
            y_pred = test_logits[test_idx].argmax(1).cpu().numpy()
        macro = f1_score(y_true, y_pred, average="macro")
        micro = f1_score(y_true, y_pred, average="micro")
        print(f"[Run {r}] Macro-F1={macro:.4f} | Micro-F1={micro:.4f}")
        macro_list.append(macro)
        micro_list.append(micro)

    print(f"\nAvg Macro-F1: {statistics.mean(macro_list):.4f}")
    print(f"Avg Micro-F1: {statistics.mean(micro_list):.4f}")

    # ===== Save results directly under results/ =====
    res_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(res_dir, exist_ok=True)
    result_path = os.path.join(res_dir, "HAN_reproduction_results.txt")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("=== HAN Reproduction Results ===\n")
        f.write("Dataset: ACM\n")
        f.write(f"Meta-paths: {metapaths}\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  hidden_dim={HIDDEN_DIM}, heads={NUM_HEADS}, layers={NUM_LAYERS}\n")
        f.write(f"  dropout={DROPOUT}, lr={LR}, weight_decay={WEIGHT_DECAY}\n")
        f.write(f"  epochs={EPOCHS}, patience={PATIENCE}, repeat={REPEAT}\n\n")
        for i, (ma, mi) in enumerate(zip(macro_list, micro_list)):
            f.write(f"Run {i}: Macro-F1={ma:.4f}, Micro-F1={mi:.4f}\n")
        f.write(f"\nAverage Macro-F1: {statistics.mean(macro_list):.4f}\n")
        f.write(f"Average Micro-F1: {statistics.mean(micro_list):.4f}\n")
        f.write(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"[INFO] Results saved to {result_path}")


if __name__ == "__main__":
    main()
