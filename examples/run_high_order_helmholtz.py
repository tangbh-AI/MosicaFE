"""任意阶 Lagrange 元求解二维 Helmholtz 方程。"""

import os
import numpy as np

from _high_order_common import OUT_DIR, exact, exact_grad, make_mesh
from MosicaFE import (
    FESpace, HelmholtzProblem, compute_errors,
    plot_mesh, plot_solution, plot_error, plot_convergence,
)

# 修改这里即可切换有限元阶数、网格数量和 Helmholtz 波数
DEGREE = 3
SIZES = [8, 12, 16, 20]
WAVE_NUMBER = 1.0


def source(x):
    """-Δu-k²u=f，精确解 u=sin(πx)sin(πy)。"""
    return (2 * np.pi**2 - WAVE_NUMBER**2) * exact(x)


def main():
    h_values = []
    errors = {"L2": [], "H1": []}

    for n in SIZES:
        # 创建三角形网格，构造 P_k 有限元空间，求解 Helmholtz 方程
        mesh = make_mesh(n)
        space = FESpace.lagrange(mesh, degree=DEGREE)
        problem = HelmholtzProblem(
            space, f=source, g=lambda x: 0.0, k=WAVE_NUMBER
        )
        u = problem.solve(method="direct")

        # 计算误差，并保存不同网格尺寸对应的数据
        err = compute_errors(problem, u, exact, exact_grad=exact_grad)
        h_values.append(mesh.get_mesh_size())
        for name in errors:
            errors[name].append(err[name])
        print(f"P{DEGREE}, n={n}: L2={err['L2']:.3e}, H1={err['H1']:.3e}")

        # 最细网格绘制三张独立图片，使用 MosicaFE 自带函数
        if n == SIZES[-1]:
            prefix = os.path.join(OUT_DIR, f"helmholtz_P{DEGREE}")
            plot_mesh(mesh, filename=prefix + "_mesh.png")
            plot_solution(u, mesh, space, filename=prefix + "_solution.png")
            plot_error(u, mesh, space, exact=exact,
                       filename=prefix + "_error.png")

    # 至少两个网格尺寸才能绘制收敛曲线
    if len(SIZES) >= 2:
        plot_convergence(
            h_values, errors,
            reference_rates=(DEGREE, DEGREE + 1),
            filename=os.path.join(OUT_DIR, f"helmholtz_P{DEGREE}_convergence.png"),
        )


if __name__ == "__main__":
    main()
