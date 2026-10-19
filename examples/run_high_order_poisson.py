"""三阶 Lagrange 元求解二维 Poisson 方程"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MosicaFE import (
    Mesh, solve_poisson, compute_errors,
    plot_mesh, plot_solution, plot_error, plot_convergence
)

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


# 精确解
def exact(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


# 右端项
def source(x):
    return 2 * np.pi**2 * exact(x)


# 精确解梯度
def exact_grad(x):
    return np.pi * np.array([
        np.cos(np.pi*x[0]) * np.sin(np.pi*x[1]),
        np.sin(np.pi*x[0]) * np.cos(np.pi*x[1])
    ])


def main():
    h_values = []
    errors = {"L2": [], "H1": []}

    for n in [8, 12, 16, 20]:
        # 生成网格
        mesh = Mesh.rectangle(nx=n, ny=n)

        # P3 Lagrange 有限元求解
        out = solve_poisson(
            mesh, element="P3", f=source,
            g=lambda x: 0.0, solver="direct",
            return_all=True
        )

        u, V = out["u"], out["space"]

        # 计算误差
        err = compute_errors(out["problem"], u, exact, exact_grad)
        h_values.append(1 / n)

        for key in errors:
            errors[key].append(err[key])

        print(f"n={n}, L2={err['L2']:.3e}, H1={err['H1']:.3e}")

        # 只对最细网格分别绘制网格、数值解和逐点绝对误差
        # 直接调用 MosicaFE 的可视化函数，每张图独立保存
        if n == 20:
            plot_mesh(
                mesh, title="P3 Mesh (n=20)",
                filename=os.path.join(OUT, "mesh.png")
            )

            plot_solution(
                u, mesh, V, title="P3 Solution",
                filename=os.path.join(OUT, "solution.png")
            )

            plot_error(
                u, mesh, V, exact=exact,
                title="P3 Pointwise Absolute Error",
                filename=os.path.join(OUT, "error.png")
            )

    # 直接调用 MosicaFE 绘制独立的 L2、H1 收敛曲线图
    plot_convergence(
        h_values, errors,
        reference_rates=(3, 4),
        filename=os.path.join(OUT, "convergence.png")
    )


if __name__ == "__main__":
    main()