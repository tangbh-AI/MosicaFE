"""Pointwise derivatives for batch-independent networks (no BatchNorm/batch coupling)."""
from __future__ import annotations

import torch


def gradient(u, x, component=0):
    """Return (N, input_dim), retaining the graph for residual backpropagation."""
    if not x.requires_grad:
        raise ValueError("Coordinates must have requires_grad=True")
    v = u if u.ndim == 1 else u[:, component]
    if not v.requires_grad:
        return x * 0
    g = torch.autograd.grad(v.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    return x * 0 if g is None else g + x * 0


def derivative(u, x, axis, order=1, component=0):
    if not 0 <= axis < x.shape[1] or order < 1:
        raise ValueError("axis out of range or order < 1")
    v = u if u.ndim == 1 else u[:, component:component + 1]
    for _ in range(order):
        v = gradient(v, x)[:, axis:axis + 1]
    return v


def laplacian(u, x, axes=None, component=0):
    axes = tuple(range(x.shape[1])) if axes is None else tuple(axes)
    if not axes or len(set(axes)) != len(axes):
        raise ValueError("axes must be nonempty and unique")
    return sum(derivative(u, x, axis, order=2, component=component) for axis in axes)


def divergence(v, x, axes=None):
    axes = tuple(range(x.shape[1])) if axes is None else tuple(axes)
    if v.ndim != 2 or v.shape[1] != len(axes):
        raise ValueError("Vector components must match divergence axes")
    return sum(derivative(v, x, axis, component=i) for i, axis in enumerate(axes))
