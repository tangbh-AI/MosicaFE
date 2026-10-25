"""三维 VEM 主程序，结构与二维及 FEM 主程序一致。

运行：python examples/vem_poisson_3d.py
main() 的 n_points 控制网格数量，degree 控制 VEM 阶数。
"""

# 模块职责：三维独立主程序，与二维分开运行和保存；截面图使用真实平面上的 VEM 投影值。


import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MosicaFE import Mesh, plot_mesh, plot_solution, solve_poisson  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "vem_3d")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 问题数据：-Δu = f in (0,1)^3, u = 0 on ∂Ω
# 精确解 u = sin(πx)sin(πy)sin(πz)，因此 f = 3π²u。
# ---------------------------------------------------------------------------
def exact(x):
    # 提供已知精确解，用于制造源项并评估数值误差；求解过程中无需把精确解作为未知量输入。
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])


def source(x):
    # 给出制造解对应的源项 f=-Δu，供单元载荷积分调用。
    return 3.0 * np.pi ** 2 * exact(x)


def exact_grad(x):
    # 返回精确解各坐标方向的偏导数，供 H1 半范数误差评估使用。
    sx, sy, sz = np.sin(np.pi*x)
    cx, cy, cz = np.cos(np.pi*x)
    return np.pi * np.array([cx*sy*sz, sx*cy*sz, sx*sy*cz])


def main(n_points=192, degree=2):
    # 执行三维完整流程：生成逐单元面数可变的多面体网格，求解后计算真实中截面，并保存误差与耗时。
    started = time.perf_counter()
    # ---- 选网格 ---------------------------------------------------------
    mesh = Mesh.voronoi_polyhedron(n_points=n_points, seed=42)

    # ---- 选元 + 选求解器 + 求解 -----------------------------------------
    k = degree  # VEM 阶数，由 main() 的 degree 参数选择。
    out = solve_poisson(mesh, element="VEM", degree=k, f=source, g=lambda x: 0.0,
                        solver="direct", return_all=True)
    u, V = out["u"], out["space"]
    errors = out["problem"].compute_errors(u, exact, exact_grad)
    elapsed = time.perf_counter() - started  # 建网格、求解和误差计算，不含绘图。

    # ---- 选可视化：绘制 z=0.5 的中截面 -----------------------------------
    plot_mesh(mesh, filename=os.path.join(OUT, f"mesh_n{n_points}.png"))
    plot_solution(u, mesh, V, title=f"VEM{k} solution (3D, {n_points} cells)", z_slice=0.5,
                  filename=os.path.join(OUT, f"solution_k{k}_n{n_points}.png"))
    # 高阶 VEM 的 u 包含积分矩，节点值才适合用来报告解的范围。
    values = V.nodal_values(u)
    print("自由度个数：", u.size, "；节点解的范围：", float(values.min()), float(values.max()))
    print(f"L2 误差：{errors['L2']:.6e}；H1 误差：{errors['H1']:.6e}；计算耗时：{elapsed:.2f} 秒", flush=True)
    result = dict(dim=3, k=k, cells=mesh.n_cells, seed=42, nodes=mesh.n_nodes, dofs=u.size,
                  h=mesh.get_mesh_size(), seconds=elapsed, **errors)
    with open(os.path.join(OUT, f"result_k{k}_n{n_points}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    main()
