"""Poisson 方程（三套原有程序共同求解的模型问题）。

.. math::

   -\\nabla\\cdot(\\kappa\\nabla u)=f\\ \\text{in}\\ \\Omega,\\qquad
   u=g\\ \\text{on}\\ \\partial\\Omega

弱形式：找 :math:`u\\in V` 使

.. math::

   \\int_\\Omega \\kappa\\nabla u\\cdot\\nabla v\\,dx
   =\\int_\\Omega f v\\,dx\\qquad\\forall v\\in V_0

本类对 **所有 H1 协调的空间**（Lagrange 有限元、虚拟元）都适用——
这正是 MosicaFE 把"元"与"方程"解耦的收益：

.. code-block:: python

   u_fem = PoissonProblem(FESpace.lagrange(mesh, 2), f=src, g=zero).solve()
   u_vem = PoissonProblem(FESpace.vem(mesh),       f=src, g=zero).solve()
"""

from __future__ import annotations

from .base import BasePhysics
from .registry import register_problem

__all__ = ["PoissonProblem"]


@register_problem("poisson")
class PoissonProblem(BasePhysics):
    """标量扩散（Poisson）问题。"""

    def __init__(self, space, f=None, g=None, kappa=1.0, source=None, **kwargs):
        # source 是 f 的别名（三套老脚本里都叫 f，guidebook 也叫 f）
        if f is None and source is not None:
            f = source
        super().__init__(space, f=f, g=g, **kwargs)
        self.kappa = kappa

    # ------------------------------------------------------------------
    def assemble(self):
        """组装 ``kappa * K`` 与载荷向量，并施加 Dirichlet 条件。"""
        A = self.space.stiffness_matrix()
        if not isinstance(self.kappa, (int, float)):
            raise TypeError(
                "kappa 目前只支持常数；变系数扩散需要在 space 里实现"
                "带系数的单元刚度矩阵（扩展点见 spaces/base.py 的 docstring）。"
            )
        if float(self.kappa) != 1.0:
            A = float(self.kappa) * A
        b = self.space.load_vector(self.f)
        A, b = self.apply_boundary_conditions(A, b)
        return A, b

    # ------------------------------------------------------------------
    def energy_norm(self, u, A=None):
        """能量范数 :math:`\\sqrt{u^T A u}`（用于收敛性与条件数分析）。"""
        if A is None:
            A, _ = self.assemble()
        u = self._as_dof_vector(u)
        return float(np.sqrt(max(u @ (A @ u), 0.0)))

    def _as_dof_vector(self, u):
        import numpy as np

        u = np.asarray(u, dtype=float).ravel()
        if u.size != self.n_dofs:
            raise ValueError(f"解向量长度 {u.size} 与自由度数 {self.n_dofs} 不符")
        return u

    # ------------------------------------------------------------------
    def __repr__(self):  # pragma: no cover - 仅展示
        return f"<PoissonProblem space={self.space.name} kappa={self.kappa}>"
