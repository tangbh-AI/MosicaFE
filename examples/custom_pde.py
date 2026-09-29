"""扩展示例：如何在本库中加入一门新的 PDE。

只要能用"空间提供的矩阵"写出弱形式，新问题就是十几行。

注意：亥姆霍兹方程 ``-Laplacian(u) - k**2 u = f`` 已经是**内置方程**
（``MosicaFE.HelmholtzProblem`` / ``solve_helmholtz``），这里不再重复实现，
而是把它当作"用起来有多短"的对照，再给出一个真正需要自己动手的例子：
带点源（Dirac 右端项）的 Poisson 问题。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MosicaFE import (  # noqa: E402
    FESpace,
    HelmholtzProblem,
    Mesh,
    PoissonProblem,
    ReactionDiffusionProblem,
)
from MosicaFE.physics.base import BasePhysics  # noqa: E402
from MosicaFE.physics.registry import available_problems, register_problem  # noqa: E402


# ---------------------------------------------------------------------------
# 1) 直接复用内置的反应扩散问题
# ---------------------------------------------------------------------------
def demo_reaction_diffusion():
    c = 10.0
    exact = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])          # noqa: E731
    source = lambda x: (2.0 * np.pi ** 2 + c) * exact(x)                   # noqa: E731
    grad = lambda x: np.array([                                            # noqa: E731
        np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]),
        np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
    ])
    mesh = Mesh.rectangle(nx=32, ny=32)
    V = FESpace.lagrange(mesh, degree=2)
    problem = ReactionDiffusionProblem(V, f=source, g=lambda x: 0.0, reaction_coeff=c)
    u = problem.solve()
    print("reaction_diffusion:", problem.compute_errors(u, exact, grad))


# ---------------------------------------------------------------------------
# 2) 内置方程有多短：亥姆霍兹 -Δu - k²u = f（无需自己实现）
# ---------------------------------------------------------------------------
def demo_helmholtz():
    """内置 HelmholtzProblem：与 Poisson 完全同一套用法。"""
    k = 1.0
    exact = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])          # noqa: E731
    source = lambda x: (2.0 * np.pi ** 2 - k ** 2) * exact(x)              # noqa: E731
    mesh = Mesh.rectangle(nx=32, ny=32)
    V = FESpace.vem(mesh)          # 虚拟元也能直接用，和 Poisson 一样
    problem = HelmholtzProblem(V, f=source, g=lambda x: 0.0, k=k)
    u = problem.solve(method="direct")
    err = np.abs(u - V.interpolate(exact)).max()
    print("helmholtz（内置，VEM 空间）max error = %.3e" % err)


# ---------------------------------------------------------------------------
# 3) 真正需要自己写的例子：带点源的 Poisson 问题
#    -Δu = f + δ(x - x0)
#
#    点源不是"能用坐标求值的函数"，因此没法塞进 load_vector(f)；
#    必须在 assemble 里自己往右端项上加一项。对 P1/Q1/VEM 这类顶点型空间，
#    节点基满足 φ_i(x0) = δ_ij（x0 取节点时），所以把 1 加到该自由度上
#    就是精确的 ∫δ(x-x0)φ_i dx。
# ---------------------------------------------------------------------------
@register_problem("point_source_poisson")
class PointSourcePoissonProblem(BasePhysics):
    """``-Δu = f + δ(x - x0)``，点源落在最近的自由度上。"""

    def __init__(self, space, x0, f=None, g=None, weight=1.0, **kwargs):
        super().__init__(space, f=f, g=g, **kwargs)
        self.x0 = np.asarray(x0, dtype=float)
        self.weight = float(weight)

    def assemble(self):
        A = self.space.stiffness_matrix()
        b = self.space.load_vector(self.f)
        dist = np.linalg.norm(self.space.dof_coords - self.x0, axis=1)
        b[int(np.argmin(dist))] += self.weight
        A, b = self.apply_boundary_conditions(A, b)
        return A, b


def demo_point_source():
    mesh = Mesh.rectangle(nx=24, ny=24)
    V = FESpace.lagrange(mesh, degree=1)
    problem = PointSourcePoissonProblem(V, x0=(0.5, 0.5), f=lambda x: 0.0,
                                        g=lambda x: 0.0)
    u = problem.solve(method="direct")
    print("point source: max u = %.3e, at %s"
          % (u.max(), np.round(V.dof_coords[int(np.argmax(u))], 3)))


def main():
    print("已注册的 PDE：", available_problems())
    demo_reaction_diffusion()
    demo_helmholtz()
    demo_point_source()
    # 顺便确认内置 Poisson 仍然可用
    mesh = Mesh.rectangle(nx=16, ny=16)
    V = FESpace.lagrange(mesh, 1)
    u = PoissonProblem(V, f=lambda x: 1.0, g=lambda x: 0.0).solve()
    print("poisson 自由度 =", u.size)


if __name__ == "__main__":
    main()
