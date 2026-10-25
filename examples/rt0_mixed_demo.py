"""RT0×P0 混合元示例：把 RT0 当作 MosicaFE 的一个"元"来用。

与 FEM / VEM 的示例对照着看最清楚——RT0 现在是库自身的一个方法，
不再需要任何外部后端：

* 空间：``FESpace.rt0(mesh)``（三角形 / 四面体网格）；
* 问题：``MixedPoissonProblem``（鞍点问题，解是 ``(Q, P)``）；
* 空间自己提供两个块：``flux_mass_matrix()``（(Φ,Φ)）与
  ``divergence_matrix()``（(P,Φ)），物理层只做拼装；
* 后处理：``compute_errors`` / ``mixed_error_norms`` / ``plot_pressure_2d`` /
  ``plot_flux_2d`` 都是原生的。

运行::

    & "D:\\anaconda\\envs\\fealpy\\python.exe" MosicaFE\\examples\\rt0_mixed_demo.py
"""

# 模块职责：RT0 混合元的完整使用示例
# 分别演示二维、三维计算与网格加密实验，检查压力、通量误差和局部质量守恒。
# 精确通量由精确解的负梯度给出；图像分别展示分片常数压力和重构的向量通量。


import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MosicaFE import FESpace, Mesh, MixedPoissonProblem  # noqa: E402
from MosicaFE.postprocess import (  # noqa: E402
    estimate_convergence_rate,
    mixed_error_norms,
    plot_flux_2d,
    plot_pressure_2d,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 制造解：p = sin(πx)sin(πy)（单位正方形，齐次 Dirichlet）
# ---------------------------------------------------------------------------
def exact(x):
    # 提供已知精确解，用于制造源项并评估数值误差；求解过程中无需把精确解作为未知量输入。
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def source(x):
    # 给出制造解对应的源项 f=-Δu，供单元载荷积分调用。
    return 2.0 * np.pi ** 2 * exact(x)


def exact_flux(x):
    """q = -∇p。"""
    return -np.pi * np.array(
        [
            np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]),
            np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
        ]
    )


def exact_3d(x):
    # 定义三维正弦制造解，作为三维误差和收敛性检查的基准。
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])


def source_3d(x):
    # 给出三维正弦解对应的源项 3π²u。
    return 3.0 * np.pi ** 2 * exact_3d(x)


def exact_flux_3d(x):
    """q = -∇p（3D 版本）。"""
    s, c = np.sin, np.cos
    return -np.pi * np.array(
        [
            c(np.pi * x[0]) * s(np.pi * x[1]) * s(np.pi * x[2]),
            s(np.pi * x[0]) * c(np.pi * x[1]) * s(np.pi * x[2]),
            s(np.pi * x[0]) * s(np.pi * x[1]) * c(np.pi * x[2]),
        ]
    )


def demo_2d(n: int = 24):
    # 执行二维混合 Poisson 算例，检查通量、压力误差并输出可视化结果。
    print("=" * 72)
    print("① 二维三角形网格：空间、自由度和两个块")
    print("=" * 72)
    mesh = Mesh.rectangle(nx=n, ny=n)               # 三角形网格
    V = FESpace.rt0(mesh)                           # 选一个"元"：RT0 x P0

    print(f"   网格：{mesh.summary()}")
    print(f"   空间：{V.summary()}")
    print(f"   通量自由度（每个面一个）：{V.n_flux}")
    print(f"   压力自由度（每个单元一个）：{V.n_pressure}")
    print(f"   边界面个数（Dirichlet 是自然条件）：{V.n_boundary_facets}")

    M = V.flux_mass_matrix()
    B = V.divergence_matrix()
    shape = lambda A: tuple(A.shape)  # noqa: E731
    print(f"   (Φ,Φ) 块 M 形状 {shape(M)}；(P,Φ) 块 B 形状 {shape(B)}")

    # ---- 求解（两种等价的方式）----
    problem = MixedPoissonProblem(V, f=source)
    Q, P = problem.solve()                          # 默认 "auto" -> Schur 凝聚
    Q_d, P_d = MixedPoissonProblem(V, f=source).solve(method="direct")
    print(f"   solve('auto') 与 solve('direct') 差："
          f"max|ΔP| = {np.abs(P - P_d).max():.2e}")

    # ---- 后处理 ----
    errs = problem.compute_errors((Q, P), exact, exact_flux)
    print("   误差：L2(压力) = %.3e，L2(通量) = %.3e，L∞(压力) = %.3e"
          % (errs["L2"], errs["flux_L2"], errs["Linf"]))
    print("   质量守恒最大偏差：%.2e" % problem.check_conservation())

    pts = np.array([[0.5, 0.5], [0.25, 0.75]])      # 任意点上的通量重构
    print("   q_h(0.5,0.5) =", np.round(problem.flux_at_points(pts)[0], 6),
          " 精确值 =", np.round(exact_flux(pts[0]), 6))

    plot_pressure_2d(mesh, P, title="RT0-P0 pressure (P0)",
                     filename=os.path.join(OUT, "rt0_pressure.png"))
    plot_flux_2d(space=V, Q=Q, title="RT0 flux (q_h at cell centroids)",
                 filename=os.path.join(OUT, "rt0_flux.png"))
    return mesh


def demo_convergence(levels=(4, 8, 16, 32)):
    # 在一组逐渐加密的网格上重复计算，观察离散误差下降的速度。
    print()
    print("=" * 72)
    print("② 收敛阶：压力与通量都是一阶（RT0×P0 的理论阶）")
    print("=" * 72)
    hs, pq, qq = [], [], []
    for n in levels:
        mesh = Mesh.rectangle(nx=n, ny=n)
        V = FESpace.rt0(mesh)
        problem = MixedPoissonProblem(V, f=source)
        Q, P = problem.solve()
        # 也可以直接用 postprocess 里的原生误差函数
        norms = mixed_error_norms(V, P, Q, exact, exact_flux)
        hs.append(mesh.get_mesh_size())
        pq.append(norms["p_l2"])
        qq.append(norms["q_l2"])
        print(f"   n={n:3d}  h={hs[-1]:.5f}  L2(p)={pq[-1]:.3e}  L2(q)={qq[-1]:.3e}")
    print(f"   压力 L2 阶 = {estimate_convergence_rate(hs, pq, use_last=3):.3f}"
          f"（理论 1）")
    print(f"   通量 L2 阶 = {estimate_convergence_rate(hs, qq, use_last=3):.3f}"
          f"（理论 1）")


def demo_3d(levels=(2, 4, 8)):
    # 执行三维混合 Poisson 算例，核对压力、通量以及单元质量守恒。
    print()
    print("=" * 72)
    print("③ 三维四面体网格：同一段代码，只换网格")
    print("=" * 72)
    hs, pq, qq = [], [], []
    for n in levels:
        mesh = Mesh.box(nx=n, ny=n, nz=n, element_type="tet")
        V = FESpace.rt0(mesh)
        problem = MixedPoissonProblem(V, f=source_3d)
        Q, P = problem.solve()
        e = problem.compute_errors((Q, P), exact_3d, exact_flux_3d)
        hs.append(mesh.get_mesh_size())
        pq.append(e["L2"])
        qq.append(e["flux_L2"])
        print(f"   n={n:2d}  cells={mesh.n_cells:5d}  facets={V.n_flux:5d}  "
              f"L2(p)={e['L2']:.3e}  L2(q)={e['flux_L2']:.3e}")
    print(f"   压力 L2 阶 = {estimate_convergence_rate(hs, pq, use_last=2):.3f}"
          f"；通量 L2 阶 = {estimate_convergence_rate(hs, qq, use_last=2):.3f}")


def main():
    # 依次执行二维混合求解、收敛实验与三维算例，集中展示 RT0 通量和压力的使用。
    demo_2d()
    demo_convergence()
    demo_3d()


if __name__ == "__main__":
    main()
