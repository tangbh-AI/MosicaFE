"""横向对比：同一个 Poisson 问题，用不同的"元"求解并比较精度与收敛阶。

这正是把三套程序整合成一个库之后才方便做的事。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MosicaFE import FESpace, Mesh, MixedPoissonProblem, PoissonProblem  # noqa: E402
from MosicaFE.postprocess import plot_convergence  # noqa: E402
from MosicaFE.postprocess.error import estimate_convergence_rate  # noqa: E402

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


def study_h1(label, space_factory, levels=(4, 8, 16, 32), mesh_factory=None):
    mesh_factory = mesh_factory or (lambda n: Mesh.rectangle(nx=n, ny=n))
    hs, l2, h1 = [], [], []
    for n in levels:
        mesh = mesh_factory(n)
        V = space_factory(mesh)
        problem = PoissonProblem(V, f=source, g=zero)
        u = problem.solve()
        e = problem.compute_errors(u, exact, exact_grad)
        hs.append(mesh.get_mesh_size())
        l2.append(e["L2"])
        h1.append(e["H1"])
    print(
        "%-22s L2=%.2e (rate %.2f)   H1=%.2e (rate %.2f)"
        % (
            label,
            l2[-1],
            estimate_convergence_rate(hs, l2, use_last=3),
            h1[-1],
            estimate_convergence_rate(hs, h1, use_last=3),
        )
    )
    return hs, {"L2": np.asarray(l2), "H1": np.asarray(h1)}


def study_mixed(levels=(4, 8, 16, 32)):
    hs, l2 = [], []
    for n in levels:
        mesh = Mesh.rectangle(nx=n, ny=n)
        V = FESpace.rt0(mesh)
        problem = MixedPoissonProblem(V, f=source)
        Q, P = problem.solve()
        e = problem.compute_errors((Q, P), exact, lambda x: -exact_grad(x))
        hs.append(mesh.get_mesh_size())
        l2.append(e["L2"])
    print(
        "%-22s L2=%.2e (rate %.2f)   ——   RT0 的压力是分片常数，一阶收敛"
        % ("RT0-P0 (mixed)", l2[-1], estimate_convergence_rate(hs, l2, use_last=3))
    )
    return hs, {"L2": np.asarray(l2)}


def main():
    print("=" * 78)
    print("同一 Poisson 问题、不同离散方法的对比")
    print("=" * 78)
    tri = lambda n: Mesh.rectangle(nx=n, ny=n)  # noqa: E731
    quad = lambda n: Mesh.rectangle(nx=n, ny=n, element_type="quad")  # noqa: E731
    penta = lambda n: Mesh.pentagon(nx=n, ny=n)  # noqa: E731

    study_h1("FEM P1 (triangle)", lambda m: FESpace.lagrange(m, 1), mesh_factory=tri)
    study_h1("FEM P2 (triangle)", lambda m: FESpace.lagrange(m, 2), mesh_factory=tri)
    study_h1("FEM Q1 (quad)", lambda m: FESpace.lagrange(m, 1), mesh_factory=quad)
    study_h1("FEM Q2 (quad)", lambda m: FESpace.lagrange(m, 2), mesh_factory=quad)
    study_h1("VEM k=1 (triangle)", lambda m: FESpace.vem(m), mesh_factory=tri)
    study_h1("VEM k=1 (quad)", lambda m: FESpace.vem(m), mesh_factory=quad)
    study_h1("VEM k=1 (pentagon)", lambda m: FESpace.vem(m), mesh_factory=penta)
    study_mixed()

    hs, errs = study_h1("FEM P2 (绘图用)", lambda m: FESpace.lagrange(m, 2),
                        mesh_factory=tri)
    plot_convergence(hs, errs, filename=os.path.join(OUT, "convergence_p2.png"))


if __name__ == "__main__":
    main()
