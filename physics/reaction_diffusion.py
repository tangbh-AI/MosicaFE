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

# 模块职责：反应扩散方程的离散
# 在扩散刚度矩阵上加反应系数乘质量矩阵，以表示方程中的零阶反应项。
# 源项和边界处理沿用公共物理层接口，区别集中在组装时矩阵块的线性组合。


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
        # 沿用公共源项和边界设置，另存扩散与反应系数，供刚度和质量项组合使用。
        super().__init__(space, f=f, g=g, **kwargs)
        self.c = reaction_coeff
        self.kappa = diffusion_coeff

    # 工作块：加入反应项的矩阵贡献
    # 由空间分别提供刚度和质量，再按反应系数组合为方程矩阵；源项载荷和 Dirichlet 条件仍按统一流程处理。
    def assemble(self):
        # 将扩散系数乘刚度矩阵、反应系数乘质量矩阵相加，再计算源项载荷并施加边界约束。
        K = self.space.stiffness_matrix()
        M = self.space.mass_matrix()
        A = self.kappa * K + self.c * M
        b = self.space.load_vector(self.f)
        A, b = self.apply_boundary_conditions(A, b)
        return A, b

    def __repr__(self):  # pragma: no cover - 仅展示
        # 返回对象的可读说明，方便交互查看；此方法只组织展示信息。
        return (
            f"<ReactionDiffusionProblem space={self.space.name} "
            f"kappa={self.kappa} c={self.c}>"
        )
