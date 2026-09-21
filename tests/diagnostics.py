# -*- coding: utf-8 -*-
"""诊断脚本：VEM 稳定化方案的数值对比、多项式再生测试等。

这些结果记录在 progress/VERIFICATION.md 中，也用于说明 guidebook 里
VEM 稳定化项的设计选择。

用法::

    & "D:\\anaconda\\envs\\fealpy\\python.exe" MosicaFE\\tests\\diagnostics.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import MosicaFE as mf  # noqa: E402
from MosicaFE.core.quadrature import cell_quadrature  # noqa: E402
from MosicaFE.postprocess.error import estimate_convergence_rate  # noqa: E402
from MosicaFE.tests.run_checks import (  # noqa: E402
    ZERO,
    exact_2d,
    exact_3d,
    flux_2d,
    grad_2d,
    grad_3d,
    source_2d,
    source_3d,
)


def report(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def vem_convergence(mesh_factory, stabilization, levels=(4, 8, 16, 32)):
    hs, l2, h1 = [], [], []
    for n in levels:
        mesh = mesh_factory(n)
        V = mf.FESpace.vem(mesh, stabilization=stabilization)
        p = mf.PoissonProblem(V, f=source_2d, g=ZERO)
        u = p.solve()
        e = p.compute_errors(u, exact_2d, grad_2d)
        hs.append(mesh.get_mesh_size())
        l2.append(e["L2"])
        h1.append(e["H1"])
    return (
        estimate_convergence_rate(hs, l2, use_last=3),
        estimate_convergence_rate(hs, h1, use_last=3),
        l2,
    )


def study_stabilization():
    report("VEM 稳定化方案对比（2D，制造解 u=sin(pi x)sin(pi y)）")
    factories = {
        "triangle": lambda n: mf.Mesh.rectangle(nx=n, ny=n),
        "quad": lambda n: mf.Mesh.rectangle(nx=n, ny=n, element_type="quad"),
        "pentagon": lambda n: mf.Mesh.pentagon(nx=n, ny=n),
    }
    for name, factory in factories.items():
        print(f"\n-- {name} 网格 --")
        for stab in ("trace", "legacy", "none"):
            rl2, rh1, l2 = vem_convergence(factory, stab)
            print(
                f"   stabilization={stab:7s} L2 rate={rl2:6.3f}  H1 rate={rh1:6.3f}"
                f"   L2 errors={['%.2e' % v for v in l2]}"
            )


def study_voronoi():
    report("VEM 在 Voronoi 多边形网格上的收敛（随机多边形、无退化角）")
    hs, l2, h1 = [], [], []
    for npts in (50, 100, 200, 400, 800, 1600, 3200):
        mesh = mf.Mesh.voronoi_polygon(n_points=npts, seed=7)
        V = mf.FESpace.vem(mesh)
        p = mf.PoissonProblem(V, f=source_2d, g=ZERO)
        u = p.solve()
        e = p.compute_errors(u, exact_2d, grad_2d)
        # 随机网格用"平均单元直径"作为 h 更稳定
        hs.append(float(np.mean([mesh.get_cell_diameter(c) for c in range(mesh.n_cells)])))
        l2.append(e["L2"])
        h1.append(e["H1"])
        print(
            f"   seeds={npts:5d} cells={mesh.n_cells:5d} h={hs[-1]:.4f} "
            f"L2={e['L2']:.3e} H1={e['H1']:.3e}"
        )
    print(f"   L2 rate = {estimate_convergence_rate(hs, l2, use_last=3):.3f}")
    print(f"   H1 rate = {estimate_convergence_rate(hs, h1, use_last=3):.3f}")


def study_reconstruction():
    report("VEM 重构误差（用精确解的顶点值做线性最小二乘重构）")
    for name, factory in {
        "triangle": lambda n: mf.Mesh.rectangle(nx=n, ny=n),
        "quad": lambda n: mf.Mesh.rectangle(nx=n, ny=n, element_type="quad"),
        "pentagon": lambda n: mf.Mesh.pentagon(nx=n, ny=n),
    }.items():
        hs, es = [], []
        for n in (4, 8, 16, 32):
            mesh = factory(n)
            V = mf.FESpace.vem(mesh)
            nodal_exact = np.array([exact_2d(p) for p in mesh.nodes])
            tot = 0.0
            for c in range(mesh.n_cells):
                pts, w = cell_quadrature(mesh, c, 4)
                uh = V.evaluate(c, nodal_exact, pts)
                ex = np.array([exact_2d(p) for p in pts])
                tot += float(np.sum(w * (uh - ex) ** 2))
            hs.append(mesh.get_mesh_size())
            es.append(np.sqrt(tot))
        print(f"   {name:9s} rate = {estimate_convergence_rate(hs, es, use_last=3):.3f}")


def study_polynomial_reproduction():
    report("单元矩阵的基本性质：多项式再生、对称性、正定性")
    for name, mesh, element in (
        ("tri-P1", mf.Mesh.rectangle(nx=4, ny=4), "P1"),
        ("tri-P2", mf.Mesh.rectangle(nx=4, ny=4), "P2"),
        ("quad-Q2", mf.Mesh.rectangle(nx=4, ny=4, element_type="quad"), "Q2"),
        ("hex-Q1", mf.Mesh.box(nx=2, ny=2, nz=2, element_type="hex"), "Q1"),
        ("tet-P1", mf.Mesh.box(nx=2, ny=2, nz=2), "P1"),
        ("vem-poly", mf.Mesh.pentagon(nx=4, ny=4), "VEM"),
        ("vem-voronoi", mf.Mesh.voronoi_polygon(30, seed=1), "VEM"),
    ):
        V = mf.make_space(mesh, element)
        K = V.stiffness_matrix()
        Ay = K.toarray() if hasattr(K, "toarray") else np.asarray(K)
        asym = np.abs(Ay - Ay.T).max()
        ev = np.linalg.eigvalsh(0.5 * (Ay + Ay.T))
        lin = lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1]
        u = V.interpolate(lin)
        energy = float(u @ (Ay @ u))
        exact_energy = 13.0 * (1.0 if mesh.dim == 2 else 0.0)
        print(
            f"   {name:13s} ndof={V.n_dofs:5d} |A-A^T|={asym:.2e} "
            f"lam_min={ev.min():.3e} energy={energy:.4f}"
        )
        del exact_energy


def main():
    study_stabilization()
    study_voronoi()
    study_reconstruction()
    study_polynomial_reproduction()
    study_rt0_native()
    study_tensor_geometry()


# ---------------------------------------------------------------------------
# 四边形 / 六面体的几何变换（第 7 号 bug 的回归诊断）
# ---------------------------------------------------------------------------
def study_tensor_geometry():
    """雅可比约定 + 旋转/畸变网格上的收敛阶。

    第 7 号 bug：``tensor_jacobian`` 返回的是转置雅可比，对轴对齐网格不可见，
    但旋转/畸变网格上会给出错误梯度。这里把判据固定下来。
    """
    from MosicaFE.core.utils import apply_affine_map, tensor_jacobian

    report("四边形 / 六面体：雅可比约定与旋转不变性（第 7 号 bug 的回归）")

    def fd(v, xi, eps=1e-6):
        xi = np.asarray(xi, float)
        J = np.zeros((xi.size, xi.size))
        for k in range(xi.size):
            e = np.zeros(xi.size)
            e[k] = eps
            J[:, k] = (apply_affine_map(xi + e, v)[0]
                       - apply_affine_map(xi - e, v)[0]) / (2 * eps)
        return J

    cases = (
        ("轴对齐正方形", [[0, 0], [1, 0], [1, 1], [0, 1]], [0.3, 0.7]),
        ("平行四边形", [[0, 0], [1, 0.2], [0.8, 1.1], [-0.2, 0.9]], [0.3, 0.7]),
        ("畸变四边形", [[0, 0], [1, 0.1], [1.2, 1.0], [0.1, 0.9]], [0.35, 0.6]),
        ("畸变六面体", [[0, 0, 0], [1, .1, 0], [1.1, 1, .05], [.1, 1, 0],
                        [0, 0, 1], [1, .1, 1.05], [1.05, 1, 1], [.1, 1, .95]],
         [0.3, 0.6, 0.4]),
    )
    for tag, verts, xi in cases:
        v = np.asarray(verts, float)
        J, _ = tensor_jacobian(v, np.asarray(xi, float))
        print(f"   |J - J_有限差分|  {tag:12s} = {np.abs(J - fd(v, xi)).max():.2e}")

    def rot_mesh(n, deg, et="quad"):
        base = (mf.Mesh.rectangle(nx=n, ny=n, element_type=et) if et == "quad"
                else mf.Mesh.box(nx=n, ny=n, nz=n, element_type=et))
        th = np.deg2rad(deg)
        R = (np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
             if et == "quad" else
             np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0],
                       [0, 0, 1]]))
        mesh = mf.Mesh.from_arrays(np.asarray(base.nodes, float) @ R.T,
                                   base.elements, et, name="rot")
        return mesh, R

    def l2(mesh, R, dim, degree):
        def exact(x):
            xi = R.T @ np.asarray(x, float)
            if dim == 2:
                return np.sin(np.pi * xi[0]) * np.sin(np.pi * xi[1])
            return (np.sin(np.pi * xi[0]) * np.sin(np.pi * xi[1])
                    * np.sin(np.pi * xi[2]))
        V = mf.FESpace.lagrange(mesh, degree)
        problem = mf.PoissonProblem(V, f=lambda x: dim * np.pi ** 2 * exact(x),
                                    g=lambda x: 0.0)
        u = problem.solve()
        return problem.compute_error(u, exact, "L2")

    print("   —— Q1 四边形网格：旋转不变性（四种角度的误差应逐位相同）——")
    ref = None
    for deg in (0, 15, 30, 45):
        hs, es = [], []
        for n in (4, 8, 16):
            mesh, R = rot_mesh(n, deg, "quad")
            hs.append(mesh.get_mesh_size())
            es.append(l2(mesh, R, 2, 1))
        if ref is None:
            ref = es
        print(f"   旋转 {deg:2d}°: L2 = {['%.4e' % v for v in es]}  "
              f"阶 = {estimate_convergence_rate(hs, es, use_last=2):.3f}  "
              f"与 0° 之差 = {np.abs(np.array(es) - np.array(ref)).max():.1e}")

    print("   —— Q2 四边形与 Q1 六面体（旋转 30°）——")
    hs, es = [], []
    for n in (4, 8):
        mesh, R = rot_mesh(n, 30, "quad")
        hs.append(mesh.get_mesh_size())
        es.append(l2(mesh, R, 2, 2))
    print(f"   Q2 旋转 30°: 阶 = {estimate_convergence_rate(hs, es, use_last=2):.3f}"
          f"（理论 3）")
    hs, es = [], []
    for n in (2, 4, 8):
        mesh, R = rot_mesh(n, 30, "hex")
        hs.append(mesh.get_mesh_size())
        es.append(l2(mesh, R, 3, 1))
    print(f"   Q1 旋转 30°（六面体）: 阶 = "
          f"{estimate_convergence_rate(hs, es, use_last=2):.3f}（理论 2）")

    print("   —— patch test：畸变四边形（线性精确解）——")
    base = mf.Mesh.rectangle(nx=4, ny=4, element_type="quad")
    nodes = np.asarray(base.nodes, float)
    inner = ((nodes[:, 0] > 1e-12) & (nodes[:, 0] < 1 - 1e-12)
             & (nodes[:, 1] > 1e-12) & (nodes[:, 1] < 1 - 1e-12))
    nodes[inner] += np.random.default_rng(11).normal(0, 0.25 / 4, (inner.sum(), 2))
    mesh = mf.Mesh.from_arrays(nodes, base.elements, "quad", name="distorted")
    lin = lambda x: 1.0 + 2.0 * x[0] - 0.5 * x[1]
    for degree in (1, 2):
        V = mf.FESpace.lagrange(mesh, degree)
        u = mf.PoissonProblem(V, f=lambda x: 0.0, g=lin).solve()
        print(f"   Q{degree}: max|u_h - u| = "
              f"{float(np.abs(u - V.interpolate(lin)).max()):.3e}")


# ---------------------------------------------------------------------------
# RT0：从"内嵌后端"改为库原生实现之后的独立自检
# ---------------------------------------------------------------------------
def study_rt0_native():
    """RT0 的三条独立性质 + 直接法/静力凝聚的一致性 + 收敛阶。

    这些量都是"实现是否正确"的硬证据：现在 RT0 的每一个数字都由 MosicaFE
    自己的代码（facets / rt0 / mixed_poisson / error）算出来，不经过任何外部后端。
    """
    from MosicaFE.core.facets import build_facet_topology
    from MosicaFE.core.quadrature import simplex_quadrature
    from MosicaFE.spaces.rt0 import rt0_flux_basis, rt0_mass_matrices

    report("RT0：单元基函数 / 精确质量矩阵 / 面拓扑（原生实现自检）")

    rng = np.random.default_rng(2026)
    worst_mass = 0.0
    worst_flux = 0.0
    for dim in (2, 3):
        for _ in range(20):
            v = rng.random((dim + 1, dim))
            v[1:] = v[0] + 1.5 * (v[1:] - v[0]) + 0.2
            J = np.array([v[i + 1] - v[0] for i in range(dim)])
            if np.linalg.det(J) < 0:
                v[[0, 1]] = v[[1, 0]]
            # (1) 精确闭式质量矩阵 vs 高阶求积
            M = rt0_mass_matrices(v[None, ...])[0]
            pts, w = simplex_quadrature(v, order=3 if dim == 2 else 2)
            phi = rt0_flux_basis(v, pts)
            Mq = np.einsum("q,qid,qjd->ij", w, phi, phi)
            worst_mass = max(worst_mass, float(np.abs(M - Mq).max() / np.abs(M).max()))
            # (2) 面通量归一化 ∫_{f_i} φ_i·n_i ds = 1
            for i in range(dim + 1):
                face = v[[j for j in range(dim + 1) if j != i]]
                fp, fw = simplex_quadrature(face, order=3)
                if dim == 2:
                    t = face[1] - face[0]
                    n = np.array([t[1], -t[0]])
                else:
                    n = np.cross(face[1] - face[0], face[2] - face[0])
                if np.dot(n, face.mean(axis=0) - v[i]) < 0:
                    n = -n
                n = n / np.linalg.norm(n)
                val = float(np.sum(fw * (rt0_flux_basis(v, fp)[:, i, :] @ n)))
                worst_flux = max(worst_flux, abs(val - 1.0))

    print(f"   (1) 精确质量矩阵 vs 求积  最大相对误差 = {worst_mass:.2e}")
    print(f"   (2) 面通量归一化误差      最大偏差     = {worst_flux:.2e}")

    # (3) 面拓扑：共享面与定向
    for tag, mesh in (
        ("2D 三角 8x8", mf.Mesh.rectangle(nx=8, ny=8)),
        ("3D 四面体 3x3x3", mf.Mesh.box(nx=3, ny=3, nz=3, element_type="tet")),
    ):
        topo = build_facet_topology(mesh)
        opp = all(
            len(cells) != 2
            or (
                topo.signs[cells[0]][
                    list(topo.cell_facets[cells[0]]).index(f)
                ]
                * topo.signs[cells[1]][
                    list(topo.cell_facets[cells[1]]).index(f)
                ]
                == -1.0
            )
            for f, cells in enumerate(topo.facet_cells)
        )
        print(
            f"   (3) {tag:16s} 面={topo.n_facets:5d} 内部={topo.n_interior:5d} "
            f"边界={topo.n_boundary:4d} 相邻面符号相反={opp}"
        )

    # (4) 直接法 vs 静力凝聚；离散守恒
    report("RT0：求解方式一致性、守恒性与收敛阶")
    mesh = mf.Mesh.rectangle(nx=16, ny=16)
    V = mf.FESpace.rt0(mesh)
    p_direct = mf.MixedPoissonProblem(V, f=source_2d).solve(method="direct")
    p_schur = mf.MixedPoissonProblem(V, f=source_2d).solve(method="schur")
    dP = float(np.abs(p_direct[1] - p_schur[1]).max())
    dQ = float(np.abs(p_direct[0] - p_schur[0]).max())
    print(f"   direct vs schur：|ΔP| = {dP:.2e}，|ΔQ| = {dQ:.2e}")
    for method in ("direct", "schur"):
        problem = mf.MixedPoissonProblem(V, f=source_2d)
        problem.solve(method=method)
        print(f"   {method:6s} 质量守恒最大偏差 = {problem.check_conservation():.2e}")

    cases = (
        ("2D 三角形", 2, (4, 8, 16, 32), lambda n: mf.Mesh.rectangle(nx=n, ny=n),
         exact_2d, flux_2d, source_2d),
        ("3D 四面体", 3, (2, 4, 8),
         lambda n: mf.Mesh.box(nx=n, ny=n, nz=n, element_type="tet"),
         exact_3d, lambda x: -grad_3d(x), source_3d),
    )
    for tag, _dim, levels, mesh_factory, exact, flux, src in cases:
        hs, pq, qq = [], [], []
        for n in levels:
            mesh = mesh_factory(n)
            V = mf.FESpace.rt0(mesh)
            problem = mf.MixedPoissonProblem(V, f=src)
            Q, P = problem.solve()
            e = problem.compute_errors((Q, P), exact, flux)
            hs.append(mesh.get_mesh_size())
            pq.append(e["L2"])
            qq.append(e["flux_L2"])
        print(
            f"   {tag}：压力 L2 阶 = "
            f"{estimate_convergence_rate(hs, pq, use_last=2):.3f}，"
            f"通量 L2 阶 = {estimate_convergence_rate(hs, qq, use_last=2):.3f}"
        )


if __name__ == "__main__":
    main()
