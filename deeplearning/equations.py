"""Strong residuals: -div(kappa grad u) + c*u = f and u_t-kappa*Delta(u)=f."""
from __future__ import annotations

from ._tensor import field
from .autodiff import divergence, gradient, laplacian, derivative


class EllipticResidual:
    def __init__(self, f=0., kappa=1., reaction=0., backend="numpy"):
        if callable(kappa) and backend != "torch":
            raise ValueError("Variable kappa must use a differentiable torch batch function")
        self.f, self.kappa, self.reaction, self.backend = f, kappa, reaction, backend

    def __call__(self, x, u):
        if u.shape[1] != 1:
            raise ValueError("Built-in elliptic residuals require scalar network output")
        flux = field(self.kappa, x, self.backend) * gradient(u, x)
        return -divergence(flux, x) + field(self.reaction, x, self.backend) * u - field(self.f, x, self.backend)


class HeatResidual:
    def __init__(self, f=0., kappa=1., time_axis=-1, backend="numpy"):
        if callable(kappa):
            raise ValueError("HeatResidual currently accepts constant scalar kappa")
        self.f, self.kappa, self.time_axis, self.backend = f, kappa, time_axis, backend

    def __call__(self, x, u):
        axis = self.time_axis % x.shape[1]
        if u.shape[1] != 1 or x.shape[1] < 2:
            raise ValueError("HeatResidual requires scalar u and space-time coordinates")
        spatial = [i for i in range(x.shape[1]) if i != axis]
        return derivative(u, x, axis) - self.kappa * laplacian(u, x, axes=spatial) - field(self.f, x, self.backend)
