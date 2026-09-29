"""亥姆霍兹方程（低频时谐波的模型问题）。

.. math::

   -\\nabla\\cdot(\\kappa\\nabla u)-k^2u=f\\ \\text{in}\\ \\Omega,
   \\qquad u=g\\ \\text{on}\\ \\partial\\Omega

弱形式：找 :math:`u\\in V_g` 使

.. math::

   \\int_\\Omega \\kappa\\nabla u\\cdot\\nabla v\\,dx
   -k^2\\int_\\Omega u v\\,dx
   =\\int_\\Omega f v\\,dx\\qquad\\forall v\\in V_0

实现的全部内容就是两个已算好的矩阵做一次线性组合——与
:class:`~MosicaFE.physics.poisson.PoissonProblem` 完全平行：

.. code-block:: python

   A = kappa * space.stiffness_matrix() - k ** 2 * space.mass_matrix()
   b = space.load_vector(f)

因此它对**所有 H1 协调的空间**（P1/P2/Q1/Q2 与虚拟元）都适用：

.. code-block:: python

   u_fem = HelmholtzProblem(FESpace.lagrange(mesh, 2), f=src, g=zero, k=1.0).solve()
   u_vem = HelmholtzProblem(FESpace.vem(mesh),        f=src, g=zero, k=1.0).solve()

适定性（"低频"到底指什么）
---------------------------

当 :math:`k^2` 小于 :math:`-\\Delta` 的 Dirichlet 第一特征值
:math:`\\lambda_1` 时，双线性型正定，离散矩阵
:math:`A=\\kappa K-k^2M` 对称正定，解存在唯一；单位正方形上
:math:`\\lambda_1=2\\pi^2\\approx19.74`，即要求
:math:`k<\\sqrt{2}\\pi\\approx4.443`。:math:`k^2` 越过 :math:`\\lambda_1`
（或与其它特征值重合）就进入**共振**区，矩阵变为对称不定甚至奇异，
必须改用 ``direct`` / ``minres`` 之类的求解器，且误差会被放大。

另外，数值色散（污染）误差要求网格分辨波：:math:`kh\\lesssim1`；
否则即使矩阵正定，加密网格也看不到最优收敛阶。

关于符号名：波数在代码里写作 ``k``（小写），以免与 guidebook 附录 B 中
表示网格单元的 :math:`K` 混淆；:math:`-\\Delta u+k^2u=f`（屏蔽 Poisson
方程）请用 :class:`~MosicaFE.physics.reaction_diffusion.ReactionDiffusionProblem`
取 ``reaction_coeff=k**2``。
"""

from __future__ import annotations

from .base import BasePhysics
from .registry import register_problem

__all__ = ["HelmholtzProblem"]


@register_problem("helmholtz")
class HelmholtzProblem(BasePhysics):
    """``-div(kappa grad u) - k**2 u = f``（``kappa``、``k`` 均为常数）。

    Parameters
    ----------
    space : BaseFESpace
        任意 H1 协调的空间（``FESpace.lagrange`` / ``FESpace.vem``）。
    f : callable, optional
        源项 ``f(x)``（默认 0）。
    g : callable, optional
        Dirichlet 数据 ``g(x)``（默认 0）。
    k : float
        波数（实常数）。``k**2 < lambda_1`` 时系统对称正定。
    kappa : float
        扩散系数（常数，默认 1）。
    source : callable, optional
        ``f`` 的别名（与三套老脚本、guidebook 的叫法保持一致）。
    """

    name = "helmholtz"

    def __init__(self, space, f=None, g=None, k=1.0, kappa=1.0,
                 source=None, **kwargs):
        if f is None and source is not None:
            f = source
        super().__init__(space, f=f, g=g, **kwargs)
        self.k = self._as_real(k, "k")
        self.kappa = self._as_real(kappa, "kappa")

    # ------------------------------------------------------------------
    @staticmethod
    def _as_real(value, name):
        """只接受实常数系数（复数波数需要复数求解器，这里明确拒绝）。"""
        import numbers

        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(
                f"HelmholtzProblem 的 {name} 目前只支持实常数，收到 {value!r}"
            )
        return float(value)

    # ------------------------------------------------------------------
    def assemble(self):
        """组装 ``kappa*K - k**2*M`` 与载荷向量，并施加 Dirichlet 条件。"""
        A = self.space.stiffness_matrix() - self.k ** 2 * self.space.mass_matrix()
        if self.kappa != 1.0:
            A = self.kappa * A
        b = self.space.load_vector(self.f)
        A, b = self.apply_boundary_conditions(A, b)
        return A, b

    # ------------------------------------------------------------------
    def __repr__(self):  # pragma: no cover - 仅展示
        return (
            f"<HelmholtzProblem space={self.space.name} k={self.k:g} "
            f"kappa={self.kappa:g} dofs={self.n_dofs}>"
        )
