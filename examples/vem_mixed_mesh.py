"""一行生成混合形状网格并求解 Poisson：python examples/vem_mixed_mesh.py。

main(dim=2) 为二维多边形；main(dim=3) 为三维混合多面体。
网格密度、边数、扰动和分锥比例直接在 main 中的 Mesh 调用处调整。
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from MosicaFE import Mesh, plot_mesh, plot_solution, solve_poisson  # noqa: E402


def main(dim=2, degree=2):
    if dim == 2:
        mesh = Mesh.mixed_polygon(nx=12, ny=12, max_sides=8, seed=42)
    elif dim == 3:
        mesh = Mesh.mixed_polyhedron(nx=4, ny=4, nz=2, max_sides=8, split=.35, seed=42)
    else:
        raise ValueError('dim 请选择 2 或 3')

    # -Δu=f，单位正方形/立方体上 u=0。
    exact = lambda x: np.prod(np.sin(np.pi*x))
    source = lambda x: dim*np.pi**2*exact(x)
    gradient = lambda x: np.array([
        np.pi*np.cos(np.pi*x[j])*np.prod(np.sin(np.pi*np.delete(x, j)))
        for j in range(dim)
    ])
    out = solve_poisson(mesh, element='VEM', degree=degree, f=source,
                        solver='direct', return_all=True)
    errors = out['problem'].compute_errors(out['u'], exact, gradient)

    folder = HERE / 'output' / 'vem_mixed'
    folder.mkdir(parents=True, exist_ok=True)
    plot_mesh(mesh, filename=folder / f'mesh_{dim}d.png')
    plot_solution(out['u'], mesh, out['space'],
                  filename=folder / f'solution_{dim}d_k{degree}.png')
    print(f'{dim}D: {mesh.n_cells} 个单元，{mesh.n_nodes} 个顶点，{out["space"].n_dofs} 个自由度')
    print('顶点数 → 单元数：', dict(sorted(Counter(mesh.vertices_per_cell).items())))
    print('投影误差：', errors)
    return out


if __name__ == '__main__':
    main()
