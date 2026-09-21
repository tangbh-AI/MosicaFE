"""扩展示例：如何在本库中加入一门新的 PDE。

只要能用"空间提供的矩阵"写出弱形式，新问题就是十几行。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MosicaFE import FESpace, Mesh, PoissonProblem, ReactionDiffusionProblem  # noqa: E402
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
# 2) 自己写一门新方程：亥姆霍兹 -Δu - k²u = f
# ---------------------------------------------------------------------------
@register_problem("helmholtz")
class HelmholtzProblem(BasePhysics):
    """``-Δu - k² u = f``（弱形式：``(∇u,∇v) - k²(u,v) = (f,v)``）。"""

    def __init__(self, space, f=None, g=None, k: float = 1.0, **kwargs):
        super().__init__(space, f=f, g=g, **kwargs)
        self.k = k

    def assemble(self):
        A = self.space.stiffness_matrix() - self.k ** 2 * self.space.mass_matrix()
        b = self.space.load_vector(self.f)
        return self.apply_boundary_conditions(A, b)


def demo_helmholtz():
    k = 1.0
    exact = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])          # noqa: E731
    source = lambda x: (2.0 * np.pi ** 2 - k ** 2) * exact(x)              # noqa: E731
    mesh = Mesh.rectangle(nx=32, ny=32)
    V = FESpace.vem(mesh)          # 虚拟元也能直接用
    problem = HelmholtzProblem(V, f=source, g=lambda x: 0.0, k=k)
    u = problem.solve(method="direct")
    err = np.abs(u - V.interpolate(exact)).max()
    print("helmholtz（VEM 空间）max error = %.3e" % err)


def main():
    print("已注册的 PDE：", available_problems())
    demo_reaction_diffusion()
    demo_helmholtz()
    # 顺便确认内置 Poisson 仍然可用
    mesh = Mesh.rectangle(nx=16, ny=16)
    V = FESpace.lagrange(mesh, 1)
    u = PoissonProblem(V, f=lambda x: 1.0, g=lambda x: 0.0).solve()
    print("poisson 自由度 =", u.size)


if __name__ == "__main__":
    main()
