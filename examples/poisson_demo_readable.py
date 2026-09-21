"""同一问题的"分步"写法：显式体现"选元 / 选求解器 / 选可视化"，

所有空间族都换成 3 行即可，符合 guidebook 的调用习惯。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MosicaFE import (  # noqa: E402
    FESpace,
    Mesh,
    MixedPoissonProblem,
    PoissonProblem,
    plot_solution,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


def exact(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def source(x):
    return 2.0 * np.pi ** 2 * exact(x)


def exact_grad(x):
    return np.array(
        [
            np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]),
            np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
        ]
    )


def zero(x):
    return 0.0


def demo_lagrange(mesh):
    """经典有限元：P2 三角形。"""
    V = FESpace.lagrange(mesh, degree=2)
    problem = PoissonProblem(V, f=source, g=zero)
    u = problem.solve()
    print("FEM-P2 :", problem.compute_errors(u, exact, exact_grad))
    return u, V


def demo_vem(mesh):
    """虚拟元：同一段代码，只换一个空间。"""
    V = FESpace.vem(mesh, degree=1)
    problem = PoissonProblem(V, f=source, g=zero)
    u = problem.solve()
    print("VEM-P1 :", problem.compute_errors(u, exact, exact_grad))
    return u, V


def demo_rt0(mesh):
    """RT0×P0 混合元：换空间 + 换问题类，解是 (通量, 压力)。"""
    V = FESpace.rt0(mesh)
    problem = MixedPoissonProblem(V, f=source)
    Q, P = problem.solve()
    print("RT0-P0 :", problem.compute_errors((Q, P), exact, lambda x: -exact_grad(x)))
    return P, None


def main():
    mesh = Mesh.rectangle(nx=32, ny=32)
    u_fem, V = demo_lagrange(mesh)
    u_vem, V_vem = demo_vem(mesh)
    u_rt0, _ = demo_rt0(mesh)

    plot_solution(u_fem, mesh, V, title="FEM P2",
                  filename=os.path.join(OUT, "demo_fem_p2.png"))
    plot_solution(u_vem, mesh, V_vem, title="VEM P1",
                  filename=os.path.join(OUT, "demo_vem_p1.png"))
    print("RT0 pressure range:", float(np.min(u_rt0)), float(np.max(u_rt0)))


if __name__ == "__main__":
    main()
