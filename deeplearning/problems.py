"""Compose residual, boundary/initial conditions and optional traditional-method data."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ._tensor import field, tensor, weighted_mse
from .conditions import DirichletBC, NeumannBC
from .data import PDEData
from .domain import as_domain
from .equations import EllipticResidual


class SupervisedProblem:
    """Ordinary MLP approximation from PDE solution data; data supplies the physics."""
    def __init__(self, network, data, domain=None):
        if not isinstance(network, nn.Module) or not isinstance(data, PDEData):
            raise TypeError("Expected a torch.nn.Module and PDEData")
        self.network, self.data = network, data
        self.domain = None if domain is None else as_domain(domain)
        self.solution = None

    def loss_terms(self, *, device, dtype):
        return {"data": self.data.loss(self.network, device=device, dtype=dtype)}

    def solve(self, config=None, **kwargs):
        from .trainer import Trainer
        return Trainer(config, **kwargs).fit(self)

    def metadata(self):
        return {"kind": type(self).__name__, "input_dim": self.data.points.shape[1],
                "output_dim": self.data.values.shape[1],
                "bounds": None if self.domain is None else self.domain.bounds.tolist()}

    def state_dict(self):
        return {"data": _data_state(self.data)}

    def load_state_dict(self, state):
        self.data = _load_data(state["data"])


class PINNProblem:
    """Strong-form PINN, default -Delta u=f with soft Dirichlet conditions.

    Generic residual(x, u) must use torch operations; output is (N, residual_channels).
    Pass explicit conditions for time-dependent PDEs, systems or mixed boundaries.
    """

    def __init__(self, domain, network, *, residual=None, f=0., g=0., kappa=1.,
                 reaction=0., backend="numpy", conditions=None, n_interior=512,
                 n_boundary=128, seed=0, interior_points=None, interior_weights=None,
                 residual_weight=1., boundary_weight=10., data=None, data_weight=1.):
        if not isinstance(network, nn.Module):
            raise TypeError("network must be a torch.nn.Module")
        if any(isinstance(m, (nn.ReLU, nn.LeakyReLU)) for m in network.modules()):
            raise ValueError("Second-order PINNs need smooth activations; use tanh/silu/softplus")
        if any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in network.modules()):
            raise ValueError("PINN autodiff requires batch-independent networks; BatchNorm is unsupported")
        for w in (residual_weight, boundary_weight, data_weight):
            if not np.isfinite(w) or w <= 0:
                raise ValueError("Loss weights must be positive finite numbers")
        self.domain, self.network = as_domain(domain), network
        self.residual = residual if residual is not None else EllipticResidual(f, kappa, reaction, backend)
        if interior_points is None:
            self.interior_points, self.interior_weights = self.domain.interior(n_interior, seed)
            if interior_weights is not None:
                raise ValueError("interior_weights requires explicit interior_points")
        else:
            self.interior_points = np.asarray(interior_points, dtype=float)
            self.interior_weights = np.ones(len(self.interior_points)) if interior_weights is None else np.asarray(interior_weights)
        x = self.interior_points
        if x.ndim != 2 or x.shape[1] != self.domain.dim or not len(x) or not np.isfinite(x).all():
            raise ValueError("interior_points must be finite (N, domain.dim)")
        w = np.asarray(self.interior_weights).ravel()
        if len(w) != len(x) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("Invalid interior_weights")
        if conditions is None:
            bx, bw = self.domain.boundary(n_boundary, seed + 1)
            conditions = [DirichletBC(bx, g, weights=bw, backend=backend, weight=boundary_weight)]
        self.conditions = list(conditions)
        names = [c.name for c in self.conditions]
        if len(set(names)) != len(names) or any(n in ("residual", "data", "total") for n in names):
            raise ValueError("Condition names must be unique and not residual/data/total")
        if any(c.points.shape[1] != self.domain.dim for c in self.conditions):
            raise ValueError("Condition coordinates must match domain.dim")
        if data is not None and not isinstance(data, PDEData):
            raise TypeError("data must be PDEData")
        self.residual_weight, self.data_weight = float(residual_weight), float(data_weight)
        self.data, self.solution = data, None

    def loss_terms(self, *, device, dtype):
        x = tensor(self.interior_points, device=device, dtype=dtype).detach().requires_grad_(True)
        u = self.network(x)
        r = self.residual(x, u)
        if not isinstance(r, torch.Tensor) or r.ndim not in (1, 2) or len(r) != len(x):
            raise ValueError("residual(x,u) must return a torch Tensor shaped (N,) or (N,C)")
        out = {"residual": self.residual_weight * weighted_mse(r, self.interior_weights)}
        out.update({c.name: c.loss(self.network, device=device, dtype=dtype) for c in self.conditions})
        if self.data is not None:
            out["data"] = self.data_weight * self.data.loss(self.network, device=device, dtype=dtype)
        return out

    def solve(self, config=None, **kwargs):
        from .trainer import Trainer
        return Trainer(config, **kwargs).fit(self)

    def metadata(self):
        residual = self.residual
        constants = {k: v for k, v in vars(residual).items() if isinstance(v, (str, int, float))} if hasattr(residual, "__dict__") else {}
        return {"kind": type(self).__name__, "bounds": self.domain.bounds.tolist(),
                "residual": getattr(residual, "__qualname__", type(residual).__qualname__),
                "constants": constants, "residual_weight": self.residual_weight,
                "data_weight": self.data_weight, "has_data": self.data is not None,
                "conditions": [{"type": type(c).__name__, "name": c.name, "weight": c.weight,
                                "component": c.component, "has_operator": c.operator is not None,
                                **({"flux_component": c.flux_component} if isinstance(c, NeumannBC) else {})}
                               for c in self.conditions]}

    def state_dict(self):
        conditions = []
        parameter = next(self.network.parameters())
        for c in self.conditions:
            x = tensor(c.points, device=parameter.device, dtype=parameter.dtype)
            state = {"points": c.points.copy(), "values": field(c.values, x, c.backend).detach().cpu().numpy(),
                     "weights": c.weights}
            if isinstance(c, NeumannBC):
                state["normals"] = c.normals.copy()
            conditions.append(state)
        return {"interior_points": self.interior_points.copy(), "interior_weights": self.interior_weights,
                "conditions": conditions, "data": _data_state(self.data)}

    def load_state_dict(self, state):
        self.interior_points, self.interior_weights = state["interior_points"], state["interior_weights"]
        for c, s in zip(self.conditions, state["conditions"]):
            c.points, c.values, c.weights = s["points"], s["values"], s["weights"]
            if isinstance(c, NeumannBC):
                c.normals = s["normals"]
        self.data = _load_data(state["data"])


def _data_state(data):
    return None if data is None else {"points": data.points, "values": data.values, "weights": data.weights}


def _load_data(state):
    return None if state is None else PDEData(**state)
