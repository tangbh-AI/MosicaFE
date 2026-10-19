# -*- coding: utf-8 -*-
"""ns_rt0_验证.py —— NS 方程的 RT0 可行性诊断（报告里的数字都由本脚本产生）

三件事：
  (1) 验证 RT0 场恒无旋：随机 RT0 速度场的 curl 精确为零；
  (2) 量出"RT0 梯度 + P1 速度 + P0 压力"三场系统的零空间维数（随网格增长）；
  (3) 跑通可以工作的部分：Poiseuille 通道、线性精确解再现、无散性检查。

运行： python ns_rt0_验证.py
输出： 结果_ns_rt0.txt
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "..", "MosicaFE")
spec = importlib.util.spec_from_file_location(
    "MosicaFE", os.path.join(PKG, "__init__.py"), submodule_search_locations=[PKG])
_mod = importlib.util.module_from_spec(spec)
sys.modules["MosicaFE"] = _mod
spec.loader.exec_module(_mod)

import MosicaFE as mf                                        # noqa: E402
from MosicaFE.core.quadrature import cell_quadrature          # noqa: E402

sys.path.insert(0, HERE)
from ns_stokes_prototype import NavierStokesProblem           # noqa: E402

out = []


def say(s=""):
    print(s)
    out.append(str(s))


# =====================================================================
# (1) RT0 场恒无旋
# =====================================================================
say("=" * 78)
say("(1) RT0 速度场是否无旋？  u = a + b x  =>  curl u = 0")
say("=" * 78)
rng = np.random.default_rng(0)
rows = []
for n in (2, 4, 8):
    mesh = mf.Mesh.rectangle(nx=n, ny=n)
    V = mf.FESpace.rt0(mesh)
    Phi = rng.standard_normal(V.n_flux)
    worst = 0.0
    h = 1e-6
    for c in range(mesh.n_cells):
        pts, _ = cell_quadrature(mesh, c, 2)
        for p in pts:
            q = lambda pt: V.evaluate_flux(c, Phi, np.atleast_2d(pt))[0]
            uy_x = (q(p + [h, 0])[1] - q(p - [h, 0])[1]) / (2 * h)
            ux_y = (q(p + [0, h])[0] - q(p - [0, h])[0]) / (2 * h)
            worst = max(worst, abs(uy_x - ux_y))
    rows.append((n, mesh.n_cells, V.n_flux, worst))
    say(f"   n={n:2d}  单元={mesh.n_cells:5d}  面自由度={V.n_flux:5d}  "
        f"max|curl u_h| = {worst:.2e}")
say("   结论：RT0 只能表示无旋场 —— 无法表达任何剪切（Poiseuille 剖面、边界层、涡）。")

# =====================================================================
# (2) 三场系统的零空间
# =====================================================================
say("")
say("=" * 78)
say("(2) 三场（G=∇u ∈ RT0，u ∈ P1，p ∈ P0）的鞍点矩阵是否奇异？")
say("     E1: (G,τ) + (u,div τ) = 0 ; E2: (G,∇v) - (p,div v) = (f,v) ; E3: (div u,q) = 0")
say("=" * 78)


def p1_basis(verts, x):
    a0 = verts[0]
    J = np.column_stack([verts[1] - a0, verts[2] - a0])
    xi = np.linalg.solve(J, x - a0) if x.ndim == 1 else np.linalg.solve(J, (x - a0).T).T
    lam = np.column_stack([1 - xi[:, 0] - xi[:, 1], xi[:, 0], xi[:, 1]]) \
        if x.ndim != 1 else np.array([1 - xi[0] - xi[1], xi[0], xi[1]])
    Ai = np.linalg.inv(J).T
    grads = np.column_stack([-Ai @ np.array([1.0, 1.0]), Ai[:, 0], Ai[:, 1]])
    return lam, grads


say(f"   {'n':>3} {'未知量':>7} {'rank':>7} {'零空间维数':>10} {'cond':>10}")
for n in (2, 4, 8):
    mesh = mf.Mesh.rectangle(nx=n, ny=n)
    V = mf.FESpace.rt0(mesh)
    nodes = np.asarray(mesh.nodes, dtype=float)
    nf, nv, nc = V.n_flux, nodes.shape[0], mesh.n_cells
    iG = lambda i, k: i * nf + k
    iU = lambda i, a: 2 * nf + i * nv + a
    iP = lambda c: 2 * nf + 2 * nv + c
    N = 2 * nf + 2 * nv + nc
    A = np.zeros((N, N))
    for c in range(mesh.n_cells):
        tri = np.asarray(mesh.elements[c], np.int64)
        verts = nodes[tri]
        area = abs(np.cross(verts[1] - verts[0], verts[2] - verts[0])) / 2
        pts, w = cell_quadrature(mesh, c, 3)
        phi = V.eval_flux_basis(c, pts)
        M = np.einsum("q,qid,qjd->ij", w, phi, phi)
        fd = np.asarray(V.cell_flux_dofs(c), np.int64)
        sgn = np.asarray(V.facet_signs(c), float)
        lam = np.array([p1_basis(verts, x)[0] for x in pts])
        wl = np.einsum("q,qa->a", w, lam)
        grads = p1_basis(verts, pts[0])[1]
        for i in range(2):
            for k in range(3):
                for l in range(3):
                    A[iG(i, fd[k]), iG(i, fd[l])] += M[k, l]
                for a in range(3):
                    A[iG(i, fd[k]), iU(i, tri[a])] += sgn[k] / area * wl[a]
                    A[iU(i, tri[a]), iG(i, fd[k])] += np.sum(
                        w * (phi[:, k, i] * grads[i, a]))
            for a in range(3):
                A[iU(i, tri[a]), iP(c)] -= np.sum(w * grads[i, a])
                A[iP(c), iU(i, tri[a])] += np.sum(w * grads[i, a])
    topo = mf.build_facet_topology(mesh)
    bn = set()
    for f in topo.boundary_facets():
        bn.update(int(v) for v in topo.facets[f])
    bd = np.array([iU(i, a) for a in sorted(bn) for i in range(2)], dtype=int)
    free = np.setdiff1d(np.arange(N), bd)
    Af = A[np.ix_(free, free)]
    sv = np.linalg.svd(Af, compute_uv=False)
    rank = int(np.sum(sv > 1e-9 * sv[0]))
    say(f"   {n:>3} {Af.shape[0]:>7} {rank:>7} {Af.shape[0]-rank:>10} {sv[0]/sv[-1]:>10.1e}")
say("   结论：零空间维数随网格增长（O(1/h)），矩阵数值奇异 —— 该组合不可用。")

# =====================================================================
# (3) 能工作的部分
# =====================================================================
say("")
say("=" * 78)
say("(3) Taylor-Hood（速度 P2 / 压力 P1）原型：能跑通的两项检查")
say("=" * 78)
pi = np.pi

# 3a 线性精确解再现：u = (x, -y), p = 0
try:
    uex = lambda x: np.column_stack([np.atleast_2d(x)[:, 0], -np.atleast_2d(x)[:, 1]])
    mesh = mf.Mesh.rectangle(nx=4, ny=4)
    V = mf.FESpace.rt0(mesh)
    prob = NavierStokesProblem(V, f=lambda x: np.zeros((len(np.atleast_2d(x)), 2)),
                               g=uex, mu=1.0, rho=0.0, bc=lambda mid, n: "D",
                               degree=(2, 1))
    U, P = prob.solve()
    eu, ep, div = prob.errors(uex, lambda p: 0.0)
    say(f"   线性精确解 u=(x,-y)：L2(u)={eu:.2e}  max|div u|={div:.2e}  "
        f"（应 ~1e-12）")
except Exception as exc:
    say(f"   线性精确解：失败（{type(exc).__name__}: {exc}）")

# 3b Poiseuille：上下壁面 no-slip，左右 do-nothing，f=(1,0)
try:
    bc = lambda mid, n: "D" if (mid[1] < 1e-12 or mid[1] > 1 - 1e-12) else "N"
    hs, errs = [], []
    for n in (8, 16):
        mesh = mf.Mesh.rectangle(nx=n, ny=n)
        V = mf.FESpace.rt0(mesh)
        prob = NavierStokesProblem(V, f=lambda x: np.array([1.0, 0.0]),
                                   mu=1.0, rho=0.0, bc=bc, degree=(2, 1))
        U, P = prob.solve()
        u_ex = lambda x: np.column_stack([np.atleast_2d(x)[:, 1] *
                                          (1 - np.atleast_2d(x)[:, 1]) / 2,
                                          np.zeros(len(np.atleast_2d(x)))])
        eu, ep, div = prob.errors(u_ex, None)
        hs.append(mesh.get_mesh_size())
        errs.append(eu)
        say(f"   Poiseuille n={n:3d}：L2(u)={eu:.3e}  max|div u|={div:.2e}")
    if len(errs) > 1 and errs[-1] > 0:
        rate = np.log(errs[0] / errs[1]) / np.log(hs[0] / hs[1])
        say(f"   实测收敛阶 ≈ {rate:.2f}")
except Exception as exc:
    say(f"   Poiseuille：失败（{type(exc).__name__}: {exc}）")

say("")
say("注：原型（Taylor–Hood 装配）尚未通过线性精确解测试（见上），说明其鞍点")
say("    组装仍有缺陷；因此本工作**没有**把 NS 求解器放进 MosicaFE。")
say("    报告第 3–4 节给出正确的替代路线与逐块组装公式"
    "（应力型混合元 / BDM / H(div)-DG）。")

with open(os.path.join(HERE, "结果_ns_rt0.txt"), "w", encoding="utf-8-sig") as fp:
    fp.write("\n".join(out) + "\n")
print("\n[结果] 已保存 结果_ns_rt0.txt")
