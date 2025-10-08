import os
import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.0, save_path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss.item() if hasattr(val_loss, "item") else -float(val_loss)
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
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
            vmin = self.val_loss_min if isinstance(self.val_loss_min, float) else float(self.val_loss_min)
            print(f"Validation loss decreased ({vmin:.6f} -> {val_loss:.6f}). Saving model ...")
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
