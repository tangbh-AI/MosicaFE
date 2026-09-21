"""物理问题的抽象基类与注册表。

扩展新 PDE 的标准流程（与 guidebook 第 12.2 节一致）：

.. code-block:: python

   from MosicaFE.physics.base import BasePhysics
   from MosicaFE.physics.registry import register_problem

   @register_problem("helmholtz")
   class HelmholtzProblem(BasePhysics):
       def __init__(self, space, f=None, g=None, k=4.0):
           super().__init__(space, f=f, g=g)
           self.k = k

       def assemble(self):
           A = self.space.stiffness_matrix() - self.k**2 * self.space.mass_matrix()
           b = self.space.load_vector(self.f)
           return self.apply_boundary_conditions(A, b)

关键点是：**空间已经把"单元矩阵"算好了**，物理层只做线性组合与边界处理，
所以一个新的二阶椭圆问题通常只需要十几行。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..solvers.linear import Solver

__all__ = ["BasePhysics"]


class BasePhysics(ABC):
    """所有 PDE 问题对象的基类。"""

    name = "base"
    is_mixed = False

    def __init__(self, space, f=None, g=None, **kwargs):
        self.space = space
        self.mesh = space.mesh
        self.n_dofs = space.n_dofs
        self.f = f if f is not None else (lambda x: 0.0)
        self.g = g if g is not None else (lambda x: 0.0)
        self.options = dict(kwargs)
        self.solution = None
        self._system = None
        self.boundary_dofs = space.boundary_dofs()

    # ------------------------------------------------------------------
    @abstractmethod
    def assemble(self):
        """组装并返回 ``(A, b)``（已施加边界条件）。"""

    # ------------------------------------------------------------------
    def apply_boundary_conditions(self, A, b, values=None):
        """强加 Dirichlet 条件 ``u = g``（默认齐次）。"""
        dofs = self.boundary_dofs
        if values is None:
            values = self.space._evaluate_function(
                self.g, self.space.dof_coords[dofs]
            ) if len(dofs) else np.zeros(0)
        return Solver.apply_dirichlet(A, b, dofs, values)

    # ------------------------------------------------------------------
    def solve(self, method: str = "auto", **kwargs):
        """组装并求解，返回自由度向量。"""
        A, b = self.assemble()
        self._system = (A, b)
        symmetric = self.options.get("symmetric", True)
        u = Solver.solve(A, b, method=method, symmetric=symmetric, **kwargs)
        self.solution = u
        return u

    # ------------------------------------------------------------------
    def compute_error(self, u, exact, norm_type: str = "L2", exact_grad=None):
        """计算误差范数（见 :mod:`MosicaFE.postprocess.error`）。"""
        from ..postprocess.error import compute_error

        return compute_error(self, u, exact, norm_type=norm_type, exact_grad=exact_grad)

    def compute_errors(self, u, exact, exact_grad=None):
        """一次性返回 ``{"L2":..., "H1":..., "Linf":...}``。"""
        from ..postprocess.error import compute_errors

        return compute_errors(self, u, exact, exact_grad=exact_grad)

    # ------------------------------------------------------------------
    def exact_source(self, exact, laplacian):
        """由精确解构造源项：``f = -Δu``（制造解法的便利函数）。"""
        return lambda x: -float(laplacian(x))

    def __repr__(self):  # pragma: no cover - 仅展示
        return (
            f"<{type(self).__name__} space={self.space.name} "
            f"n_dofs={self.n_dofs} dim={self.mesh.dim}>"
        )
