"""反应扩散方程（演示"如何扩展一门新 PDE"）。

.. math::

   -\\nabla\\cdot(\\kappa\\nabla u)+c\\,u=f\\ \\text{in}\\ \\Omega,
   \\qquad u=g\\ \\text{on}\\ \\partial\\Omega

弱形式：:math:`\\kappa(\\nabla u,\\nabla v)+c(u,v)=(f,v)`，于是

.. code-block:: python

   A = kappa * space.stiffness_matrix() + c * space.mass_matrix()
   b = space.load_vector(f)

整个实现不到 20 行——这正是 `spaces` 暴露矩阵级接口的价值。
"""

from __future__ import annotations

import numpy as np

from .base import BasePhysics
from .registry import register_problem

__all__ = ["ReactionDiffusionProblem"]


@register_problem("reaction_diffusion")
class ReactionDiffusionProblem(BasePhysics):
    """``-kappa Δu + c u = f``。"""

    def __init__(
        self,
        space,
        f=None,
        g=None,
        reaction_coeff: float = 1.0,
        diffusion_coeff: float = 1.0,
        **kwargs,
    ):
        super().__init__(space, f=f, g=g, **kwargs)
        self.c = reaction_coeff
        self.kappa = diffusion_coeff

    def assemble(self):
        K = self.space.stiffness_matrix()
        M = self.space.mass_matrix()
        A = self.kappa * K + self.c * M
        b = self.space.load_vector(self.f)
        A, b = self.apply_boundary_conditions(A, b)
        return A, b

    def __repr__(self):  # pragma: no cover - 仅展示
        return (
            f"<ReactionDiffusionProblem space={self.space.name} "
            f"kappa={self.kappa} c={self.c}>"
        )
