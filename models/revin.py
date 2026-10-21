import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al., 2022).
    Normalizes each sample over the time dimension, then de-normalizes
    the prediction — removes train/test distribution shift without
    leaking future statistics.

    Usage:
        revin = RevIN(num_features)
        x_norm = revin(x, mode="norm")    # before model
        y_denorm = revin(y, mode="denorm") # after model output
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.mean = None   # populated by norm()
        self.std  = None   # populated by norm()
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            if self.mean is None or self.std is None:
                raise RuntimeError(
                    "RevIN.denorm called before norm. "
                    "Ensure revin_dec.mean and .std are set.")
            return self._denormalize(x)
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")

    def _get_statistics(self, x):
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std  = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * self.std + self.mean
