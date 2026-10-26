"""任意 k 阶三角形 Lagrange 元求解二维 Poisson 方程。

只需修改 main() 中的网格划分数和 element="P3" 的阶数。
运行：python MosicaFE/examples/run_pk_poisson_simple.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from MosicaFE import Mesh, plot_solution, solve_poisson  # noqa: E402


# 精确解 u=sin(pi*x)sin(pi*y)，源项 f=-Delta u
def exact(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def source(x):
    return 2.0 * np.pi**2 * exact(x)


def exact_grad(x):
    return np.pi * np.array([np.cos(np.pi*x[0])*np.sin(np.pi*x[1]), np.sin(np.pi*x[0])*np.cos(np.pi*x[1])])


def main():
    # ---- 选网格 ---------------------------------------------------------
    mesh = Mesh.rectangle(nx=8, ny=8, element_type="triangle")

    # ---- 选元 + 选求解器 + 求解 ------------------------------------------
    out = solve_poisson(mesh, element="P3", f=source, g=lambda x: 0.0, solver="direct", return_all=True)

    # ---- 选可视化 -------------------------------------------------------
    plot_solution(out["u"], mesh, out["space"], title="P3 solution", filename=os.path.join(os.path.dirname(__file__), "poisson_p3.png"))

    # ---- 误差分析 -------------------------------------------------------
    print("L2、H1误差：", out["problem"].compute_errors(out["u"], exact, exact_grad))


if __name__ == "__main__":
    main()
