"""Point conditions, including Dirichlet, initial, Neumann and custom operators."""
from __future__ import annotations

import numpy as np

from ._tensor import field, tensor, weighted_mse
from .autodiff import gradient


class PointCondition:
    """operator(x,u)->(N,C); omit operator for Dirichlet/data/initial values."""

    def __init__(self, points, values=0., *, name="boundary", weight=1.,
                 weights=None, backend="numpy", component=None, operator=None):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or len(self.points) == 0 or not np.isfinite(self.points).all():
            raise ValueError("Condition points must be a nonempty finite (N, dim) array")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Condition weight must be positive")
        if backend not in ("numpy", "torch"):
            raise ValueError("backend must be 'numpy' or 'torch'")
        self.values, self.name, self.weight = values, str(name), float(weight)
        self.weights, self.backend = weights, backend
        self.component, self.operator = component, operator

    def loss(self, model, *, device, dtype):
        x = tensor(self.points, device=device, dtype=dtype).detach().requires_grad_(True)
        u = model(x)
        pred = self.operator(x, u) if self.operator else u
        if pred.ndim == 1:
            pred = pred[:, None]
        if self.component is not None:
            if not 0 <= self.component < pred.shape[1]:
                raise ValueError(f"Condition {self.name}: component index out of range")
            pred = pred[:, self.component:self.component + 1]
        target = field(self.values, x, self.backend)
        if target.shape != pred.shape:
            raise ValueError(f"Condition {self.name}: target {tuple(target.shape)} != output {tuple(pred.shape)}")
        return self.weight * weighted_mse(pred - target, self.weights)


class DirichletBC(PointCondition):
    pass


class InitialCondition(PointCondition):
    def __init__(self, points, values=0., **kwargs):
        kwargs.setdefault("name", "initial")
        super().__init__(points, values, **kwargs)


class NeumannBC(PointCondition):
    """Prescribe du/dn (not -kappa*du/dn); normals must be explicit unit vectors."""

    def __init__(self, points, normals, values=0., *, component=0, **kwargs):
        normals = np.asarray(normals, dtype=float)
        if normals.shape != np.asarray(points).shape or not np.allclose(np.linalg.norm(normals, axis=1), 1.):
            raise ValueError("normals must have the points' shape and unit row norms")
        kwargs.setdefault("name", "neumann")
        self.normals = normals
        self.flux_component = component
        super().__init__(points, values, operator=self._normal_derivative, **kwargs)

    def _normal_derivative(self, x, u):
        normals = tensor(self.normals, device=x.device, dtype=x.dtype)
        return (gradient(u, x, self.flux_component) * normals).sum(dim=1, keepdim=True)
