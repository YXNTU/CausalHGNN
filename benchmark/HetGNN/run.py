# run.py
# Fixed to run HetGNN on dataset "HetGNN/Academic2" with given hyperparams.

import os
import time
import datetime
import statistics
from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import dgl

from HetGNN import HetGNN  # import model


# ===== Fixed hyperparameters (as requested) =====
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 0.00001

DIM          = 128        # hidden dim
MAX_EPOCH    = 500
BATCH_SIZE   = 64
WINDOW_SIZE  = 5
NUM_WORKERS  = 4
BATCHES_PER_EPOCH = 50

RW_LENGTH = 50
RW_WALKS  = 10
RWR_PROB  = 0.5

PATIENCE        = 20
MINI_BATCH_FLAG = True

DROPOUT = 0.2
REPEAT  = 5
SEED    = 0


def set_seed(seed: int = 0):
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=40, verbose=True, save_path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


def default_data_path(repo_root: str, dataset_tag: str):
    family, name = dataset_tag.split("/")
    fname = f"processed_{name}.pth"
    return os.path.join(repo_root, "data", family, fname)


def load_processed_pth(pth_path: str) -> Dict:
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Processed dataset not found: {pth_path}")
    data = torch.load(pth_path)
    required = ["hg", "labels", "num_classes", "train_idx", "val_idx",
                "test_idx", "node_features", "category"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {pth_path}: {missing}")
    return data


def iterate_minibatches(indices: torch.Tensor, batch_size: int):
    n = indices.shape[0]
    perm = torch.randperm(n, device=indices.device)
    for i in range(0, n, batch_size):
        yield indices[perm[i:i + batch_size]]


def train_eval_one(
    data: Dict,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    dropout=DROPOUT,
    hidden_dim=DIM,
    patience=PATIENCE,
    epochs=MAX_EPOCH,
    repeat=REPEAT,
    device=None,
    ckpt_dir=None,
):
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hg: dgl.DGLHeteroGraph = data["hg"].to(device)
    labels = data["labels"].to(device)
    num_classes = data["num_classes"]
    train_idx = data["train_idx"].to(device)
    val_idx   = data["val_idx"].to(device)
    test_idx  = data["test_idx"].to(device)
    node_features = {k: v.to(device) for k, v in data["node_features"].items()}
    in_dims = {k: v.shape[1] for k, v in node_features.items()}
    category = data["category"]

    criterion = nn.CrossEntropyLoss()
    macro_list, micro_list = [], []
    os.makedirs(ckpt_dir, exist_ok=True)

    for r in range(repeat):
        model = HetGNN(
            in_dims=in_dims,
            hidden_dim=hidden_dim,
            out_dim=num_classes,
            category=category,
            etypes=hg.canonical_etypes,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        ckpt_path = os.path.join(ckpt_dir, f"hetgnn_rep{r}.pt")
        stopper = EarlyStopping(patience=patience, verbose=True, save_path=ckpt_path)
        t_start = time.time()

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            # full forward; batch loss (index on logits)
            for batch in iterate_minibatches(train_idx, BATCH_SIZE):
                logits_dict = model(hg, node_features)
                logits = logits_dict[category]
                loss = criterion(logits[batch], labels[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # validation
            model.eval()
            with torch.no_grad():
                val_logits = model(hg, node_features)[category]
                val_loss = criterion(val_logits[val_idx], labels[val_idx])

            if epoch % 1 == 0:
                print(f"Epoch {epoch:04d} | Train {total_loss:.4f} | Val {val_loss.item():.4f}")

            stopper(val_loss, model)
            if stopper.early_stop:
                print("Early stopping.")
                break

        elapse = time.time() - t_start

        # test
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        with torch.no_grad():
            test_logits = model(hg, node_features)[category]
            y_true = labels[test_idx].cpu().numpy()
            y_pred = test_logits[test_idx].argmax(1).cpu().numpy()

        macro = f1_score(y_true, y_pred, average="macro")
        micro = f1_score(y_true, y_pred, average="micro")
        print(f"[Rep {r}] Macro-F1={macro:.4f} | Micro-F1={micro:.4f} | Time={elapse:.1f}s")
        macro_list.append(macro)
        micro_list.append(micro)

    return statistics.mean(macro_list), statistics.mean(micro_list), macro_list, micro_list


def main():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    dataset = "HetGNN/Academic2"

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pth_path = default_data_path(repo_root, dataset)
    print(f"[INFO] Loading: {pth_path}")
    data = load_processed_pth(pth_path)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoint", dataset.replace("/", "_"))

    avg_ma, avg_mi, mas, mis = train_eval_one(
        data,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        hidden_dim=DIM,
        patience=PATIENCE,
        epochs=MAX_EPOCH,
        repeat=REPEAT,
        device=device,
        ckpt_dir=ckpt_dir,
    )

    log_path = os.path.join(results_dir, f"HetGNN_{dataset.replace('/', '_')}_results.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== HetGNN Results ===\n")
        f.write(f"Dataset: {dataset}\n")
        f.write("Hyperparameters (fixed):\n")
        f.write(f"  dim={DIM}, dropout={DROPOUT}\n")
        f.write(f"  learning_rate={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}\n")
        f.write(f"  max_epoch={MAX_EPOCH}, patience={PATIENCE}, batch_size={BATCH_SIZE}\n")
        f.write(f"  window_size={WINDOW_SIZE}, rw_length={RW_LENGTH}, rw_walks={RW_WALKS}, rwr_prob={RWR_PROB}\n")
        f.write(f"  mini_batch_flag={MINI_BATCH_FLAG}, num_workers={NUM_WORKERS}, batches_per_epoch={BATCHES_PER_EPOCH}\n\n")
        for i, (ma, mi) in enumerate(zip(mas, mis)):
            f.write(f"Run {i}: Macro-F1={ma:.4f}, Micro-F1={mi:.4f}\n")
        f.write(f"\nAverage Macro-F1: {avg_ma:.4f}\n")
        f.write(f"Average Micro-F1: {avg_mi:.4f}\n")
        f.write(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"[INFO] Results saved to {log_path}")
    print("\n========== Summary ==========")
    print(f"{dataset}: Avg Macro-F1={avg_ma:.4f} | Avg Micro-F1={avg_mi:.4f} | Log={log_path}")


if __name__ == "__main__":
    main()
