"""最简用法示例：主文件只有几行代码。

运行::

    & "D:\\anaconda\\envs\\fealpy\\python.exe" MosicaFE\\examples\\poisson_demo.py

想换一种"元"或"求解器"，只改 main() 里的几个字符串即可。
"""

# 模块职责：最简 Poisson 调用示例
# 用正弦函数构造可验证的源项，以公共 API 完成网格生成、方程求解和结果展示。
# 先运行 main 观察完整流程，再修改 element 或网格细分数，比较离散方法与分辨率对结果的影响。


import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MosicaFE import Mesh, plot_solution, solve_poisson  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 问题数据：-Δu = f in (0,1)^2, u = 0 on ∂Ω, 精确解 u = sin(πx)sin(πy)
# ---------------------------------------------------------------------------
def exact(x):
    # 提供已知精确解，用于制造源项并评估数值误差；求解过程中无需把精确解作为未知量输入。
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def source(x):
    # 给出制造解对应的源项 f=-Δu，供单元载荷积分调用。
    return 2.0 * np.pi ** 2 * exact(x)


def main():
    # ---- 选网格 ---------------------------------------------------------
    mesh = Mesh.rectangle(nx=32, ny=32)

    # ---- 选元 + 选求解器 + 求解 -----------------------------------------
    out = solve_poisson(mesh, element="P2", f=source, g=lambda x: 0.0,
                        solver="direct", return_all=True)
    u, V = out["u"], out["space"]

    # ---- 选可视化 -------------------------------------------------------
    plot_solution(u, mesh, V, title="P2 solution",
                  filename=os.path.join(OUT, "poisson_p2.png"))
    print("自由度个数：", u.size, "；解的范围：", float(u.min()), float(u.max()))


if __name__ == "__main__":
    main()
