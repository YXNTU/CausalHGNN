import os
import time
import datetime
import statistics
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import dgl
from HGT import HGT

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT = 0.4
BATCH_SIZE = 5120
PATIENCE = 40
HIDDEN_DIM = 64
OUT_DIM = 3
NUM_LAYERS = 2
NUM_HEADS = 8
NORM = True
EPOCHS = 1000
REPEAT = 1
SEED = 0


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


def default_data_path(repo_root, dataset_tag: str):
    family, name = dataset_tag.split("/")
    fname = f"processed_{name}.pth"
    return os.path.join(repo_root, "data", family, fname)


def load_processed_pth(pth_path: str):
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Processed dataset not found: {pth_path}")
    data = torch.load(pth_path)
    required = ["hg", "labels", "num_classes", "train_idx", "val_idx", "test_idx", "node_features", "category"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {pth_path}: {missing}")
    return data


def train_eval_one(
    data,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    dropout=DROPOUT,
    hidden_dim=HIDDEN_DIM,
    out_dim=OUT_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    patience=PATIENCE,
    norm=NORM,
    epochs=EPOCHS,
    repeat=REPEAT,
    device=None,
    ckpt_dir=None,
):
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hg: dgl.DGLHeteroGraph = data["hg"].to(device)
    labels = data["labels"].to(device)
    num_classes = data["num_classes"]
    train_idx = data["train_idx"].to(device)
    val_idx = data["val_idx"].to(device)
    test_idx = data["test_idx"].to(device)
    node_features = {k: v.to(device) for k, v in data["node_features"].items()}
    in_dims = {k: v.shape[1] for k, v in node_features.items()}
    category = data["category"]
    if out_dim != num_classes:
        print(f"[WARN] out_dim ({out_dim}) != num_classes ({num_classes}), override to num_classes.")
        out_dim = num_classes

    criterion = nn.CrossEntropyLoss()
    macro_list, micro_list = [], []
    os.makedirs(ckpt_dir, exist_ok=True)

    for r in range(repeat):
        model = HGT(
            in_dims=in_dims,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            category=category,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            etypes=hg.canonical_etypes,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        ckpt_path = os.path.join(ckpt_dir, f"hgt_rep{r}.pt")
        stopper = EarlyStopping(patience=patience, verbose=True, save_path=ckpt_path)
        t_start = time.time()

        for epoch in range(epochs):
            model.train()
            logits = model(hg, node_features)[category]
            loss = criterion(logits[train_idx], labels[train_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_logits = model(hg, node_features)[category]
                val_loss = criterion(val_logits[val_idx], labels[val_idx])
            if epoch % 1 == 0:
                print(f"Epoch {epoch:04d} | Train {loss.item():.4f} | Val {val_loss.item():.4f}")
            stopper(val_loss, model)
            if stopper.early_stop:
                print("Early stopping.")
                break

        elapse = time.time() - t_start
        model.load_state_dict(torch.load(ckpt_path))
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
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    datasets = ["MAGNN/IMDB", "MAGNN/DBLP"]
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    all_logs = []

    for ds in datasets:
        print(f"\n========== Running dataset: {ds} ==========")
        pth_path = default_data_path(repo_root, ds)
        data = load_processed_pth(pth_path)
        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoint", ds.replace("/", "_"))
        avg_ma, avg_mi, mas, mis = train_eval_one(
            data,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            dropout=DROPOUT,
            hidden_dim=HIDDEN_DIM,
            out_dim=OUT_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            patience=PATIENCE,
            norm=NORM,
            epochs=EPOCHS,
            repeat=REPEAT,
            device=device,
            ckpt_dir=ckpt_dir,
        )
        log_path = os.path.join(results_dir, f"HGT_{ds.replace('/', '_')}_results.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== HGT Results ===\n")
            f.write(f"Dataset: {ds}\n")
            f.write("Hyperparameters (fixed):\n")
            f.write(f"  hidden_dim={HIDDEN_DIM}, heads={NUM_HEADS}, layers={NUM_LAYERS}, norm={NORM}\n")
            f.write(f"  dropout={DROPOUT}, lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}\n")
            f.write(f"  epochs={EPOCHS}, patience={PATIENCE}, repeat={REPEAT}\n\n")
            for i, (ma, mi) in enumerate(zip(mas, mis)):
                f.write(f"Run {i}: Macro-F1={ma:.4f}, Micro-F1={mi:.4f}\n")
            f.write(f"\nAverage Macro-F1: {avg_ma:.4f}\n")
            f.write(f"Average Micro-F1: {avg_mi:.4f}\n")
            f.write(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"[INFO] Results saved to {log_path}")
        all_logs.append((ds, log_path, avg_ma, avg_mi))

    print("\n========== Summary ==========")
    for ds, log, ma, mi in all_logs:
        print(f"{ds}: Avg Macro-F1={ma:.4f} | Avg Micro-F1={mi:.4f} | Log={log}")


if __name__ == "__main__":
    main()
