import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience; self.verbose = verbose; self.counter = 0
        self.best_score = None; self.early_stop = False
        self.val_loss_min = np.inf; self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score; self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score; self.save_checkpoint(val_loss, model, path); self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), path + "/checkpoint.pth")
        self.val_loss_min = val_loss

class dotdict(dict):
    __getattr__ = dict.get; __setattr__ = dict.__setitem__; __delattr__ = dict.__delitem__

class StandardScaler:
    def __init__(self): self.mean = 0.; self.std = 1.
    def fit(self, data):
        self.mean = data.mean(0); self.std = data.std(0)
        self.std = np.where(self.std == 0, 1.0, self.std)
    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std
    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

def adjust_learning_rate(optimizer, epoch, args):
    lradj = getattr(args, "lradj", "cosine")
    if lradj == "cosine":
        warmup = getattr(args, "warmup_epochs", 2); T_max = getattr(args, "train_epochs", 50)
        lr_min = getattr(args, "lr_min", 1e-6); lr_max = args.learning_rate
        if epoch < warmup:
            lr = lr_max * (epoch + 1) / max(warmup, 1)
        else:
            progress = (epoch - warmup) / max(T_max - warmup, 1)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups: pg["lr"] = lr
    else:
        lr = args.learning_rate
        for pg in optimizer.param_groups: pg["lr"] = lr
    return lr
