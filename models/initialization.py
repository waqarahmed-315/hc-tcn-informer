"""Reference implementation of sequence-derived initialization.

The core algorithms are implemented here, but paper-specific numerical
configurations are deliberately not bundled. This module alone is not a
complete reproduction package for any particular experiment.
"""

import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

_SKIP = ("norm", "layernorm", "embed", "embedding", "affine", "revin")
_REQUIRED = (
    "target_std", "dt", "burn_in", "hc_r", "hc_state", "chen_state",
    "hc_coefficients", "chen_coefficients",
)


def should_skip(name):
    return any(fragment in name.lower() for fragment in _SKIP)


def eligible_weight_items(model):
    return [(name, parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad and not should_skip(name)
            and "bias" not in name.lower()]


def eligible_bias_items(model):
    return [(name, parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad and not should_skip(name)
            and "bias" in name.lower()]


def load_initialization_config(path):
    """Load explicitly supplied settings; never silently replace missing values."""
    if not path:
        raise ValueError("A configuration file is required for chaotic initialization.")
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    missing = [key for key in _REQUIRED if key not in config or config[key] is None]
    if missing:
        raise ValueError(f"Incomplete configuration: {', '.join(missing)}")
    for name, expected in (("hc_coefficients", ("a", "b", "c", "d", "e", "f")),
                           ("chen_coefficients", ("a", "b", "c"))):
        if not isinstance(config[name], dict) or any(
            key not in config[name] or config[name][key] is None for key in expected
        ):
            raise ValueError(f"{name} must provide coefficients: {', '.join(expected)}")
    if len(config["hc_state"]) != 4 or len(config["chen_state"]) != 3:
        raise ValueError("hc_state must have four values; chen_state must have three")
    if config["dt"] <= 0 or config["burn_in"] < 0 or config["target_std"] <= 0:
        raise ValueError("dt and target_std must be positive; burn_in must be nonnegative")
    return config


class HyperChaotic4D:
    """Four-dimensional source defined using explicitly supplied coefficients."""

    def __init__(self, coefficients, r, dt, state):
        self.a = float(coefficients["a"])
        self.b = float(coefficients["b"])
        self.c = float(coefficients["c"])
        self.d = float(coefficients["d"])
        self.e = float(coefficients["e"])
        self.f = float(coefficients["f"])
        self.r = float(r)
        self.dt = float(dt)
        self.state = tuple(float(value) for value in state)

    def generate(self, kept_steps, burn_in):
        x, y, z, w = self.state
        out = np.empty((kept_steps, 4), dtype=np.float64)
        for i in range(burn_in + kept_steps):
            dx = self.a * (y - x - w) + self.b * y * z
            dy = self.c * (4 * x + y) - x * z
            dz = self.d * x - self.e * z + x * y
            dw = self.r * x + self.f * (3 * y * z + y * y)
            x += dx * self.dt
            y += dy * self.dt
            z += dz * self.dt
            w += dw * self.dt
            if i >= burn_in:
                out[i - burn_in] = (x, y, z, w)
        return out.reshape(-1)


class Chen3D:
    """Three-dimensional Chen source with explicit parameters."""

    def __init__(self, coefficients, dt, state):
        self.a = float(coefficients["a"])
        self.b = float(coefficients["b"])
        self.c = float(coefficients["c"])
        self.dt = float(dt)
        self.state = tuple(float(value) for value in state)

    def generate(self, kept_steps, burn_in):
        x, y, z = self.state
        out = np.empty((kept_steps, 3), dtype=np.float64)
        for i in range(burn_in + kept_steps):
            dx = self.a * (y - x)
            dy = (self.c - self.a) * x - x * z + self.c * y
            dz = x * y - self.b * z
            x += dx * self.dt
            y += dy * self.dt
            z += dz * self.dt
            if i >= burn_in:
                out[i - burn_in] = (x, y, z)
        return out.reshape(-1)


def _cache_key(source, n_values, config):
    details = {"source": source, "n_values": int(n_values), "config": config}
    payload = json.dumps(details, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_source_sequence(source, n_values, config, cache_dir="./init_cache"):
    if source not in ("chaotic", "hyperchaotic"):
        raise ValueError("source must be 'chaotic' or 'hyperchaotic'")
    if n_values <= 0:
        return np.empty(0, dtype=np.float32)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{source}_{_cache_key(source, n_values, config)}.npy")
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r")
        if arr.size >= n_values:
            return np.asarray(arr[:n_values], dtype=np.float32)

    dims = 4 if source == "hyperchaotic" else 3
    kept_steps = math.ceil(n_values / dims) + 5
    if source == "hyperchaotic":
        kept_steps = max(kept_steps, 2_000_000)
        generator = HyperChaotic4D(
            config["hc_coefficients"], config["hc_r"],
            config["dt"], config["hc_state"],
        )
    else:
        generator = Chen3D(
            config["chen_coefficients"], config["dt"], config["chen_state"]
        )
    sequence = generator.generate(kept_steps, burn_in=int(config["burn_in"]))
    sequence = ((sequence - sequence.mean()) / (sequence.std() + 1e-8)).astype(np.float32)
    np.save(path, sequence)
    return sequence[:n_values]


def weight_statistics(model, bins=200, hist_range=(-1.0, 1.0)):
    weights = eligible_weight_items(model)
    if not weights:
        return {}
    values = np.concatenate([
        parameter.detach().float().cpu().numpy().ravel()
        for _, parameter in weights
    ]).astype(np.float64)
    std = float(values.std())
    ac1 = (float(np.corrcoef(values[:-1], values[1:])[0, 1])
           if values.size > 1 and std > 0 else float("nan"))
    histogram, _ = np.histogram(values, bins=bins, range=hist_range)
    prob = histogram[histogram > 0].astype(np.float64)
    prob = prob / prob.sum() if prob.size else prob
    entropy = float(-(prob * np.log(prob)).sum()) if prob.size else float("nan")
    return {
        "weight_mean": float(values.mean()),
        "weight_variance": float(values.var()),
        "weight_std": std,
        "weight_entropy_hist": entropy,
        "weight_lag1_autocorr": ac1,
        "entropy_bins": bins,
        "entropy_range": list(hist_range),
        "fraction_outside_entropy_range": float(np.mean(
            (values < hist_range[0]) | (values > hist_range[1])
        )),
        "eligible_weight_count": int(values.size),
    }


def apply_initialization(model, mode="standard", config_path=None,
                         cache_dir="./init_cache"):
    """Use standard/Xavier/He baselines or a configured dynamical source.

    Paper-specific chaotic/hyperchaotic settings are never inferred silently.
    """
    mode = str(mode).lower()
    if mode not in ("standard", "xavier", "he", "chaotic", "hyperchaotic"):
        raise ValueError(f"Unknown initialization mode: {mode}")
    start = time.perf_counter()
    weights = eligible_weight_items(model)
    biases = eligible_bias_items(model)
    settings = None

    if mode == "standard":
        pass  # Existing PyTorch defaults remain unchanged.
    elif mode in ("xavier", "he"):
        for _, parameter in weights:
            if parameter.ndim >= 2:
                if mode == "xavier":
                    nn.init.xavier_uniform_(parameter)
                else:
                    nn.init.kaiming_normal_(parameter, mode="fan_in", nonlinearity="relu")
            else:
                nn.init.normal_(parameter, mean=0.0, std=0.02)
        for _, parameter in biases:
            nn.init.zeros_(parameter)
    else:
        settings = load_initialization_config(config_path)
        n_values = sum(parameter.numel() for _, parameter in weights)
        sequence = get_source_sequence(mode, n_values, settings, cache_dir=cache_dir)
        sequence = sequence * float(settings["target_std"])
        offset = 0
        with torch.no_grad():
            for _, parameter in weights:
                count = parameter.numel()
                chunk = np.array(sequence[offset:offset + count], copy=True)
                offset += count
                if parameter.ndim >= 2:
                    chunk /= np.sqrt(max(float(parameter.shape[1]), 1.0))
                parameter.copy_(torch.from_numpy(chunk).reshape(parameter.shape)
                                .to(dtype=parameter.dtype, device=parameter.device))
            for _, parameter in biases:
                parameter.zero_()

    stats = weight_statistics(model)
    stats.update({
        "init_mode": mode,
        "custom_initialization_time_s": float(time.perf_counter() - start),
        "eligible_bias_count": int(sum(p.numel() for _, p in biases)),
        "skipped_parameter_count": int(sum(
            p.numel() for name, p in model.named_parameters() if should_skip(name)
        )),
    })
    if settings is not None:
        stats.update({
            "hc_r": float(settings["hc_r"]),
            "dt": float(settings["dt"]),
            "burn_in": int(settings["burn_in"]),
            "hc_state": list(settings["hc_state"]),
            "chen_state": list(settings["chen_state"]),
        })
    return stats
