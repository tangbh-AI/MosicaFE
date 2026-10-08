"""Neural fields with differentiable evaluation and traditional-method adapters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ._tensor import tensor
from .autodiff import gradient


class NeuralSolution:
    def __init__(self, model, domain=None, *, history=None, run_dir=None):
        self.model, self.domain = model, domain
        self.history = [] if history is None else history
        self.run_dir = None if run_dir is None else Path(run_dir)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def _points(self, points):
        x = tensor(points, device=self.device, dtype=self.dtype)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError("points must be (dim,) or (N,dim); use [:,None] for 1D batches")
        return x

    def __call__(self, points):
        """NumPy scalar (single scalar field) or (N,) / (N,C) array."""
        single = np.ndim(points) == 1
        with torch.no_grad():
            out = self.model(self._points(points)).detach().cpu().numpy()
        if out.shape[1] == 1:
            out = out[:, 0]
        return out[0] if single else out

    predict = __call__

    def tensor(self, points):
        """Differentiable (N,C) output; preserves an input tensor's coordinate graph."""
        return self.model(self._points(points))

    def gradient(self, points, component=0):
        x = self._points(points).detach().requires_grad_(True)
        with torch.enable_grad():
            out = gradient(self.model(x), x, component).detach().cpu().numpy()
        return out

    def to_fe(self, space, component=0):
        """Nodal interpolation into Lagrange/VEM, not L2 projection."""
        if getattr(space, "is_mixed", False):
            raise ValueError("RT0 requires to_pressure/to_flux; face dofs are integrated fluxes")
        if self.domain is not None and self.domain.dim != space.mesh.dim:
            raise ValueError("Coordinate dimensions differ; evaluate a space-time slice explicitly before FE transfer")
        return space.interpolate(lambda p: self._component(p, component))

    def _component(self, p, component):
        out = np.asarray(self(p))
        if out.ndim == 0:
            if component != 0:
                raise ValueError("Scalar field only has component=0")
            return float(out)
        return float(out[component])

    def to_pressure(self, space, component=0):
        """RT0 P0 cell averages using the existing pressure interpolation operator."""
        if not getattr(space, "is_mixed", False):
            raise ValueError("to_pressure requires RT0 space")
        return space.interpolate_pressure(lambda p: self._component(p, component))

    def to_flux(self, space, *, kappa=1., components=None):
        """Global RT0 face integrals: q=-kappa*grad(u), or explicit vector channels."""
        if not getattr(space, "is_mixed", False):
            raise ValueError("to_flux requires RT0 space")
        if components is not None:
            if len(components) != space.mesh.dim:
                raise ValueError("Vector channels must match spatial dimension")
            q = lambda p: np.asarray(self(p))[list(components)]
        else:
            if self.model(self._points(space.mesh.nodes[:1])).shape[1] != 1:
                raise ValueError("Gradient-derived flux requires scalar u; choose vector components")
            q = lambda p: -(kappa(p) if callable(kappa) else kappa) * self.gradient(p)[0]
        return space.interpolate_flux(q)

    def compute_errors(self, exact, exact_grad=None, *, mesh=None, quad_order=4):
        """Reuse MosicaFE quadrature for the true neural field, not its nodal interpolant."""
        from types import SimpleNamespace
        from ..postprocess.error import compute_errors
        mesh = mesh if mesh is not None else getattr(self.domain, "mesh", None)
        if mesh is None:
            raise ValueError("Provide mesh=... for integral error norms")
        if self.model(self._points(mesh.nodes[:1])).shape[1] != 1:
            raise ValueError("Integral error adapter currently supports scalar fields")
        space = _NeuralFieldView(self, mesh)
        return compute_errors(SimpleNamespace(space=space, mesh=mesh, is_mixed=False),
                              self(mesh.nodes), exact, exact_grad, quad_order=quad_order)

    @classmethod
    def load(cls, checkpoint, *, device="cpu"):
        """Load an MLP for inference. Rebuild custom models and use Trainer for resume."""
        from .domain import Domain
        from .networks import MLP
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("format_version") != 1 or state.get("model_spec") is None:
            raise ValueError("Checkpoint must be format v1 with an MLP model_spec")
        model = MLP(**state["model_spec"])
        model.to(device=device, dtype=getattr(torch, state["config"]["dtype"]))
        model.load_state_dict(state["model"])
        model.eval()
        bounds = state["problem_metadata"].get("bounds")
        return cls(model, None if bounds is None else Domain(bounds), history=state["history"],
                   run_dir=Path(checkpoint).resolve().parent)


class _NeuralFieldView:
    """Read-only postprocess view of a trained model; never a matrix assembly space."""
    degree = 1
    is_mixed = False

    def __init__(self, solution, mesh):
        self.solution, self.mesh, self.dof_coords = solution, mesh, mesh.nodes

    def evaluate(self, cell, values, points):
        return self.solution(points)

    def evaluate_gradient(self, cell, values, points):
        return self.solution.gradient(points)

    def _evaluate_function(self, f, points):
        return np.asarray([f(p) for p in points], dtype=float)
