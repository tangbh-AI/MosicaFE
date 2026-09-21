# -*- coding: utf-8 -*-
"""MosicaFE 自检脚本：把三种元在各个网格上跑一遍并打印收敛阶。

用法（在 D:\\mosica 目录下）::

    & "D:\\anaconda\\envs\\fealpy\\python.exe" MosicaFE\\tests\\run_checks.py

不依赖 pytest，纯粹为了"一条命令看结果"。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import MosicaFE as mf  # noqa: E402
from MosicaFE.postprocess.error import estimate_convergence_rate, rate_table  # noqa: E402


# ---------------------------------------------------------------------------
# 制造解
# ---------------------------------------------------------------------------
def exact_2d(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def grad_2d(x):
    return np.array(
        [
            np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]),
            np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
        ]
    )


def source_2d(x):
    return 2.0 * np.pi ** 2 * exact_2d(x)


def flux_2d(x):
    return -grad_2d(x)


def exact_3d(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])


def grad_3d(x):
    s = np.sin
    c = np.cos
    return np.pi * np.array(
        [
            c(np.pi * x[0]) * s(np.pi * x[1]) * s(np.pi * x[2]),
            s(np.pi * x[0]) * c(np.pi * x[1]) * s(np.pi * x[2]),
            s(np.pi * x[0]) * s(np.pi * x[1]) * c(np.pi * x[2]),
        ]
    )


def source_3d(x):
    return 3.0 * np.pi ** 2 * exact_3d(x)


ZERO = lambda x: 0.0  # noqa: E731


def check_2d(element, levels=(4, 8, 16, 32), mesh_factory=None, label=None):
    """2D 收敛率检查，返回 (h, errors)。"""
    if mesh_factory is None:
        mesh_factory = lambda n: mf.Mesh.rectangle(nx=n, ny=n)  # noqa: E731
    hs, errs = [], {"L2": [], "H1": [], "Linf": []}
    for n in levels:
        mesh = mesh_factory(n)
        V = mf.make_space(mesh, element)
        problem = mf.PoissonProblem(V, f=source_2d, g=ZERO)
        u = problem.solve()
        e = problem.compute_errors(u, exact_2d, grad_2d)
        hs.append(mesh.get_mesh_size())
        for k in errs:
            errs[k].append(e[k])
    hs = np.asarray(hs)
    errs = {k: np.asarray(v) for k, v in errs.items()}
    print(f"\n=== {label or element} ===")
    print(rate_table(hs, errs))
    for k in ("L2", "H1"):
        print(f"  rate[{k}] (last 3) = {estimate_convergence_rate(hs, errs[k], use_last=3):.3f}")
    return hs, errs


def check_3d(element, levels=(2, 4, 8), label=None):
    hs, errs = [], {"L2": [], "H1": [], "Linf": []}
    for n in levels:
        mesh = mf.Mesh.box(nx=n, ny=n, nz=n, element_type="hex" if element in ("Q1", "Q2") else "tet")
        V = mf.make_space(mesh, element)
        problem = mf.PoissonProblem(V, f=source_3d, g=ZERO)
        u = problem.solve()
        e = problem.compute_errors(u, exact_3d, grad_3d)
        hs.append(mesh.get_mesh_size())
        for k in errs:
            errs[k].append(e[k])
    hs = np.asarray(hs)
    errs = {k: np.asarray(v) for k, v in errs.items()}
    print(f"\n=== {label or element} ===")
    print(rate_table(hs, errs))
    for k in ("L2", "H1"):
        print(f"  rate[{k}] (last 2) = {estimate_convergence_rate(hs, errs[k], use_last=2):.3f}")
    return hs, errs


def check_vem_3d(levels=(2, 4, 8)):
    """3D 虚拟元（四面体 / 六面体网格）。"""
    for et in ("tet", "hex"):
        hs, errs = [], {"L2": [], "H1": [], "Linf": []}
        for n in levels:
            mesh = mf.Mesh.box(nx=n, ny=n, nz=n, element_type=et)
            V = mf.FESpace.vem(mesh)
            problem = mf.PoissonProblem(V, f=source_3d, g=ZERO)
            u = problem.solve()
            e = problem.compute_errors(u, exact_3d, grad_3d)
            hs.append(mesh.get_mesh_size())
            for k in errs:
                errs[k].append(e[k])
        hs = np.asarray(hs)
        errs = {k: np.asarray(v) for k, v in errs.items()}
        print(f"\n=== VEM 3D on {et} mesh ===")
        print(rate_table(hs, errs))
        for k in ("L2", "H1"):
            print(f"  rate[{k}] (last 2) = {estimate_convergence_rate(hs, errs[k], use_last=2):.3f}")


def check_patch_test():
    """线性精确解的 patch test：离散解应当精确等于精确解。"""
    print("\n=== patch test（非齐次 Dirichlet，线性精确解）===")
    exact = lambda x: 1.0 + 2.0 * x[0] + 0.5 * x[1]  # noqa: E731
    zero = lambda x: 0.0  # noqa: E731
    meshes = {
        "triangle": mf.Mesh.rectangle(nx=4, ny=4),
        "quad": mf.Mesh.rectangle(nx=4, ny=4, element_type="quad"),
        "pentagon": mf.Mesh.pentagon(nx=4, ny=4),
        "voronoi": mf.Mesh.voronoi_polygon(30, seed=3),
    }
    rows = []
    for name, mesh in meshes.items():
        V = mf.FESpace.vem(mesh)
        problem = mf.PoissonProblem(V, f=zero, g=exact)
        u = problem.solve()
        err = float(np.abs(u - V.interpolate(exact)).max())
        rows.append((f"VEM/{name}", err))
    for elem, mesh in (("P1", mf.Mesh.rectangle(nx=4, ny=4)),
                       ("Q1", mf.Mesh.rectangle(nx=4, ny=4, element_type="quad"))):
        V = mf.make_space(mesh, elem)
        problem = mf.PoissonProblem(V, f=lambda x: 0.0, g=exact)
        u = problem.solve()
        rows.append((f"FEM/{elem}", float(np.abs(u - V.interpolate(exact)).max())))
    for label, err in rows:
        flag = "OK " if err < 1e-10 else "FAIL"
        print(f"   [{flag}] {label:14s} max|u_h - u| = {err:.3e}")
    return all(err < 1e-10 for _, err in rows)


def check_rt0(levels=(4, 8, 16, 32)):
    hs, errs = [], {"L2": [], "H1": [], "Linf": []}
    for n in levels:
        mesh = mf.Mesh.rectangle(nx=n, ny=n)
        V = mf.FESpace.rt0(mesh)
        problem = mf.MixedPoissonProblem(V, f=source_2d)
        Q, P = problem.solve()
        e = problem.compute_errors((Q, P), exact_2d, flux_2d)
        hs.append(mesh.get_mesh_size())
        errs["L2"].append(e["L2"])
        errs["Linf"].append(e["Linf"])
        errs["H1"].append(e["H1"])
    hs = np.asarray(hs)
    errs = {k: np.asarray(v) for k, v in errs.items()}
    print("\n=== RT0-P0 (mixed) ===")
    print(rate_table(hs, errs))
    print(f"  rate[L2 pressure] (last 3) = {estimate_convergence_rate(hs, errs['L2'], use_last=3):.3f}")
    return hs, errs


def main():
    print("MosicaFE 自检")
    print("=" * 70)
    ok = check_patch_test()
    print("2D 三角网格：")
    check_2d("P1")
    check_2d("P2")
    print("\n2D 四边形（Q）与多边形（VEM，含五边形）网格：")
    quad = lambda n: mf.Mesh.rectangle(nx=n, ny=n, element_type="quad")  # noqa: E731
    tri = lambda n: mf.Mesh.rectangle(nx=n, ny=n)  # noqa: E731
    penta = lambda n: mf.Mesh.pentagon(nx=n, ny=n)  # noqa: E731
    check_2d("Q1", mesh_factory=quad)
    check_2d("Q2", mesh_factory=quad)
    check_2d("VEM", mesh_factory=tri)
    check_2d("VEM", mesh_factory=quad, label="VEM on quad mesh")
    check_2d("VEM", mesh_factory=penta, label="VEM on pentagon mesh")
    print("\n3D 六面体/四面体网格：")
    check_3d("P1")
    check_3d("Q1")
    check_vem_3d()
    check_rt0()
    check_rt0_3d()
    print("\npatch test 通过：", ok)
    print("\n全部检查结束。")


def check_rt0_3d(levels=(2, 4, 8)):
    hs, l2, q2 = [], [], []
    for n in levels:
        mesh = mf.Mesh.box(nx=n, ny=n, nz=n, element_type="tet")
        V = mf.FESpace.rt0(mesh)
        problem = mf.MixedPoissonProblem(V, f=source_3d)
        Q, P = problem.solve()
        e = problem.compute_errors((Q, P), exact_3d, lambda x: -grad_3d(x))
        hs.append(mesh.get_mesh_size())
        l2.append(e["L2"])
        q2.append(e["H1"])
    print("\n=== RT0-P0 (mixed) 3D ===")
    print(rate_table(np.asarray(hs), {
        "L2": np.asarray(l2), "H1": np.asarray(q2), "Linf": np.full(len(hs), np.nan)
    }))
    print(f"  rate[pressure L2] = {estimate_convergence_rate(hs, l2, use_last=2):.3f}")
    print(f"  rate[flux L2]     = {estimate_convergence_rate(hs, q2, use_last=2):.3f}")


if __name__ == "__main__":
    main()
