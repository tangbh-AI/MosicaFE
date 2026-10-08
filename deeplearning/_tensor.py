"""Explicit NumPy point-function and differentiable Torch batch-function contracts."""
from __future__ import annotations

import numpy as np
import torch


def tensor(value, *, device, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)


def columns(value, n, *, device, dtype):
    value = tensor(value, device=device, dtype=dtype)
    if value.ndim == 0:
        return value.expand(n, 1)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[0] != n:
        raise ValueError(f"Expected scalar or (N, C) values with N={n}, got {tuple(value.shape)}")
    return value


def field(value, x, backend="numpy"):
    """numpy: f(point), torch: f(batch); never guess vectorization or detach torch."""
    if not callable(value):
        return columns(value, len(x), device=x.device, dtype=x.dtype)
    if backend == "torch":
        return columns(value(x), len(x), device=x.device, dtype=x.dtype)
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'torch'")
    points = x.detach().cpu().numpy()
    return columns(np.asarray([value(p) for p in points]), len(x), device=x.device, dtype=x.dtype)


def weighted_mse(error, weights=None):
    if error.ndim == 1:
        error = error[:, None]
    if weights is None:
        return error.square().mean()
    w = tensor(weights, device=error.device, dtype=error.dtype).reshape(-1)
    if len(w) != len(error) or not torch.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative and have positive sum, one per point")
    return (w * error.square().mean(dim=1)).sum() / w.sum()
