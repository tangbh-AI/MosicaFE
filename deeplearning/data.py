"""Supervised PDE data, including quadrature samples of FEM/VEM/RT0 solutions."""
from __future__ import annotations

import numpy as np

from ..core.quadrature import cell_quadrature
from ._tensor import tensor, weighted_mse


class PDEData:
    def __init__(self, points, values, weights=None):
        self.points = np.asarray(points, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if self.values.ndim == 1:
            self.values = self.values[:, None]
        if (self.points.ndim != 2 or not len(self.points) or self.values.ndim != 2
                or len(self.points) != len(self.values) or not np.isfinite(self.points).all()
                or not np.isfinite(self.values).all()):
            raise ValueError("Data must be finite (N, dim) points and (N, channels) values")
        self.weights = np.ones(len(self.points)) if weights is None else np.asarray(weights, dtype=float).ravel()
        if (len(self.weights) != len(self.points) or not np.isfinite(self.weights).all()
                or np.any(self.weights < 0) or self.weights.sum() <= 0):
            raise ValueError("Data weights must be finite, nonnegative and have positive sum")

    @classmethod
    def from_fe(cls, space, solution, *, field="scalar", order=2):
        """Sample reconstructed fields, weighted by actual cell quadrature.

        RT0 requires field='pressure' or 'flux'; its solution is (Q,P).
        Negative-weight quadrature is invalid for MSE and is rejected.
        """
        mixed = getattr(space, "is_mixed", False)
        if mixed:
            if field not in ("pressure", "flux") or not isinstance(solution, (tuple, list)) or len(solution) != 2:
                raise ValueError("RT0 data requires (Q,P) and field='pressure' or 'flux'")
            q, p = solution
        elif field != "scalar":
            raise ValueError("FEM/VEM data uses field='scalar'")
        points, values, weights = [], [], []
        for c in range(space.mesh.n_cells):
            x, w = cell_quadrature(space.mesh, c, order)
            if np.any(w < 0):
                raise ValueError("MSE requires positive quadrature; use order=2")
            if mixed:
                v = space.evaluate_flux(c, q, x) if field == "flux" else space.evaluate(c, p, x)
            else:
                v = space.evaluate(c, solution, x)
            points.append(x)
            values.append(np.asarray(v).reshape(len(x), -1))
            weights.append(w)
        return cls(np.vstack(points), np.vstack(values), np.concatenate(weights))

    def loss(self, model, *, device, dtype):
        x = tensor(self.points, device=device, dtype=dtype)
        y = tensor(self.values, device=device, dtype=dtype)
        pred = model(x)
        if pred.shape != y.shape:
            raise ValueError(f"Data output {tuple(y.shape)} != model output {tuple(pred.shape)}")
        return weighted_mse(pred - y, self.weights)
