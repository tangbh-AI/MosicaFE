"""二维 VEM 主程序：问题数据 → 选网格 → 选元并求解 → 可视化。

运行：python examples/vem_poisson.py
只需修改 main() 参数：degree=3 表示三阶，n_points 控制网格数量。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# HERE 是本程序所在的 examples 文件夹；上两级是 MosicaFE 的父目录。
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from MosicaFE import Mesh, plot_mesh, plot_solution, solve_poisson  # noqa: E402

# 图片和结果保存到 examples/output/vem_2d，与阶数和求解算法无关。
OUT = HERE / "output" / "vem_2d"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 问题数据：-Δu = f in (0,1)^2, u = 0 on ∂Ω, 精确解 u = sin(πx)sin(πy)
# ---------------------------------------------------------------------------
def exact(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def source(x):
    return 2.0 * np.pi ** 2 * exact(x)


def exact_grad(x):
    return np.pi * np.array([np.cos(np.pi*x[0])*np.sin(np.pi*x[1]),
                             np.sin(np.pi*x[0])*np.cos(np.pi*x[1])])


def main(n_points=1024, degree=3):
    # degree 直接传给求解器：3 就是三阶 VEM，无需再定义 k。
    started = time.perf_counter()
    # ---- 选网格 ---------------------------------------------------------
    mesh = Mesh.voronoi_polygon(n_points=n_points, seed=42)

    # ---- 选元 + 选求解器 + 求解 -----------------------------------------
    out = solve_poisson(mesh, element="VEM", degree=degree, f=source, g=lambda x: 0.0,
                        solver="direct", return_all=True)
    u, V = out["u"], out["space"]
    errors = out["problem"].compute_errors(u, exact, exact_grad)
    elapsed = time.perf_counter() - started  # 建网格、求解和误差计算，不含绘图。

    # ---- 选可视化 -------------------------------------------------------
    plot_mesh(mesh, filename=OUT / f"mesh_n{n_points}.png")
    plot_solution(u, mesh, V, title=f"VEM{degree} solution (2D, {n_points} cells)",
                  filename=OUT / f"solution_k{degree}_n{n_points}.png")
    # 高阶 VEM 的 u 包含积分矩；解的范围应读取节点值，不能直接取 u.min/max。
    values = V.nodal_values(u)
    print(f"VEM 阶数：{degree}；自由度个数：", u.size, "；节点解的范围：", float(values.min()), float(values.max()))
    print(f"L2 误差：{errors['L2']:.6e}；H1 误差：{errors['H1']:.6e}；计算耗时：{elapsed:.2f} 秒", flush=True)
    # 保存误差，便于与其他阶数和网格比较。
    result = dict(dim=2, k=degree, cells=mesh.n_cells, seed=42, nodes=mesh.n_nodes, dofs=u.size,
                  h=mesh.get_mesh_size(), seconds=elapsed, **errors)
    with open(OUT / f"result_k{degree}_n{n_points}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    main()
