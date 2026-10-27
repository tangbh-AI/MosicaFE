# -*- coding: utf-8 -*-
"""darcy_mosicafe.py —— 用 MosicaFE 的 RT0×P0 空间求解 Darcy（达西）方程

★ Darcy 求解器已**并入库内**，本脚本只是"用库 + 验证 + 出图"的演示：

    MosicaFE/spaces/rt0.py    : ``RT0Space.weighted_flux_mass_matrix(K, mode=...)``
                                —— 加权通量质量矩阵 A_ij = ∫ (K^{-1}φ_i)·φ_j
    MosicaFE/physics/darcy.py : ``DarcyProblem``（注册名 ``darcy``）
                                —— 鞍点组装、本质/自然边界条件、Schur 求解

Darcy 问题（混合形式）::

    K^{-1} q + ∇p = 0        in Ω        （达西定律 q = -K∇p）
    ∇·q            = f        in Ω        （质量守恒）
    p              = g_D      on Γ_D      （压力 Dirichlet：自然边界条件）
    q·n            = g_N      on Γ_N      （流量：本质边界条件）

本脚本做四件事：

1. **自检**：``V.weighted_flux_mass_matrix()`` 在 K=I 时与 ``V.flux_mass_matrix()``
   完全相同；闭式（每单元常数 K）与逐点求积两条路径互相一致；
2. **对拍**：K=I 时 ``DarcyProblem`` 与库里 ``MixedPoissonProblem`` 给出同一个解；
3. **收敛阶**：常值各向异性（2D）、四面体各向异性（3D）都验到一阶；
4. **物理算例**：达西柱实验（出口流量 = κ）、给定流量边界、分层高对比
   （界面通量 = 调和平均），以及误差可视化。

运行： ``python darcy_mosicafe.py``
（MosicaFE 目录名是 ``MosicaFE``，直接 ``import MosicaFE`` 会失败，
  所以这里用 importlib 按路径把包名注册成 ``MosicaFE``。）

作者：赵典
"""

import importlib.util
import math
import os
import sys

import numpy as np

# ★ 连 __pycache__ 都不往库目录里写：import 前关掉字节码缓存
sys.dont_write_bytecode = True


# =====================================================================
# 0. 载入 MosicaFE（按路径注册包名，不改动库文件）
# =====================================================================
def load_mosicafe():
    """把 MosicaFE 包目录按名字 ``MosicaFE`` 载入 ``sys.modules``。"""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.environ.get("MOSICAFE_DIR"),
        os.path.join(here, "..", "..", "MosicaFE"),
        os.path.join(here, "..", "MosicaFE"),
    ]
    pkg_dir = None
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "__init__.py")):
            pkg_dir = os.path.abspath(c)
            break
    if pkg_dir is None:
        raise RuntimeError(
            "找不到 MosicaFE 包目录；请把仓库放在工作区，或设环境变量 MOSICAFE_DIR"
        )
    if "MosicaFE" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "MosicaFE",
            os.path.join(pkg_dir, "__init__.py"),
            submodule_search_locations=[pkg_dir],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["MosicaFE"] = module
        spec.loader.exec_module(module)
    return pkg_dir


PKG_DIR = load_mosicafe()

from MosicaFE import DarcyProblem, FESpace, Mesh                   # noqa: E402
from MosicaFE.core.quadrature import cell_quadrature               # noqa: E402
from MosicaFE.physics.mixed_poisson import MixedPoissonProblem     # noqa: E402
from MosicaFE.postprocess import (estimate_convergence_rate,       # noqa: E402
                                  mixed_error_norms)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figs", "mosicafe")

try:                                              # 误差可视化（Pillow，不用 matplotlib）
    import darcy_plot as dp
except Exception as _e:                           # pragma: no cover
    dp = None
    _DP_ERR = _e


# =====================================================================
# 1. 渗透率与精确解
#
#    ★ 加权质量矩阵与 Darcy 问题现在都是**库内实现**：
#        MosicaFE/spaces/rt0.py   : RT0Space.weighted_flux_mass_matrix(K, mode=...)
#        MosicaFE/physics/darcy.py: DarcyProblem（注册名 ``darcy``）
#      本脚本只负责"用库"：造精确解、算误差、出图。
# =====================================================================
def KMat(K, X):
    """把渗透率统一成 ``(nq, d, d)`` 的矩阵数组（仅用于算精确通量）。

    ``K`` 可以是：标量 / ``(d,)`` 对角 / ``(d, d)`` 张量 / 可调用 ``K(X)``
    （返回 ``(nq,)`` 标量场或 ``(nq, d, d)`` 张量场）。
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    nq, d = X.shape
    eye = np.eye(d)[None, :, :]
    if callable(K):
        val = np.asarray(K(X), dtype=float)
        if val.ndim == 1:
            return val[:, None, None] * eye
        if val.ndim == 2:
            return val[:, :, None] * eye
        return val
    arr = np.asarray(K, dtype=float)
    if arr.ndim == 0:
        return np.full((nq, 1, 1), float(arr)) * eye
    if arr.ndim == 1:
        return np.broadcast_to(np.diag(arr)[None, :, :], (nq, d, d)).copy()
    return np.broadcast_to(arr, (nq, d, d)).copy()


# =====================================================================
# 5. 误差可视化用的逐单元误差
# =====================================================================
def cell_errors_2d(V, P, Q, case, order=6):
    """逐单元的 L2 误差：``||p-p_h||_K`` 与 ``||q-q_h||_K``（已除以单元测度）。

    这是"误差可视化"最自然的量：既能看到误差的**空间分布**（哪里大），
    也能直接看出网格加密时整个分布如何整体下降。
    """
    mesh = V.mesh
    q_ex = flux_of(case["K"], case["gradp"])
    ep = np.zeros(mesh.n_cells)
    eq = np.zeros(mesh.n_cells)
    for c in range(mesh.n_cells):
        pts, w = cell_quadrature(mesh, c, order)
        area = float(np.sum(w))
        pe = V._evaluate_function(case["p"], pts)
        qe = np.array([q_ex(p) for p in pts])
        qh = V.evaluate_flux(c, Q[c], pts)
        ep[c] = np.sqrt(np.sum(w * (pe - P[c]) ** 2) / area)
        eq[c] = np.sqrt(np.sum(w * ((qe - qh) ** 2).sum(axis=1)) / area)
    return ep, eq


def cell_arrays(V):
    """返回 (nodes, cells, centroids)，供绘图使用。"""
    mesh = V.mesh
    nodes = np.asarray(mesh.nodes, dtype=float)
    cells = np.asarray(mesh.elements, dtype=np.int64)
    ctr = np.array([mesh.get_cell_centroid(c) for c in range(mesh.n_cells)])
    return nodes, cells, ctr


def flux_at_centroids(V, Q, ctr):
    """在单元形心重构 q_h。``Q`` 可以是 ``(n_cells, d+1)``、``(n_flux,)``
    或完整混合解 ``(n_dofs,)``——库的 ``evaluate_flux`` 自己会认。"""
    return np.array([V.evaluate_flux(c, Q, ctr[c][None, :])[0]
                     for c in range(V.mesh.n_cells)])


def p0_step_along_line(V, P, y_target, h):
    """贴着 ``y = y_target`` 上方那一排三角形拼出 P0 阶梯折线（转发到 drawing 工具）。"""
    return dp.p0_step_along_line(np.asarray(V.mesh.nodes, float),
                                 np.asarray(V.mesh.elements, np.int64),
                                 P, y_target, h)


def quiver_stride(V, target=26):
    """箭头太密就看不清：每条边大约画 ``target`` 个箭头。"""
    return max(1, int(round(math.sqrt(V.mesh.n_cells) / target)))


# =====================================================================
# 6. 算例：p 精确解 → q = -K∇p，f = ∇·q
# =====================================================================
PI = np.pi


def flux_of(K, gradp):
    """由 ``q = -K∇p`` 造出 ``q_exact(x)``（单点接口，供 mixed_error_norms 用）。"""
    def q_exact(x):
        X = np.atleast_2d(np.asarray(x, dtype=float))
        return -(KMat(K, X)[0] @ np.asarray(gradp(X), dtype=float)[0])
    return q_exact


def sin_p(X):
    return np.sin(PI * X[:, 0]) * np.sin(PI * X[:, 1])


def sin_gradp(X):
    return PI * np.column_stack([np.cos(PI * X[:, 0]) * np.sin(PI * X[:, 1]),
                                 np.sin(PI * X[:, 0]) * np.cos(PI * X[:, 1])])


def case_iso():
    K = 1.0
    return dict(name="K = I（各向同性，退化为 Poisson）",
                K=K, p=sin_p, gradp=sin_gradp,
                f=lambda X: 2 * PI ** 2 * sin_p(X),
                g=lambda X: np.zeros(len(X)), bc=lambda mid, n: "D",
                mesh=lambda n: Mesh.rectangle(nx=n, ny=n))


def case_aniso(Kmat=(3.0, 1.0, 2.0)):
    """常值各向异性 K=[[k11,k12],[k12,k22]]：f = π²(tr K) p − 2π²k12 cosπx cosπy。"""
    k11, k12, k22 = Kmat
    K = np.array([[k11, k12], [k12, k22]])
    return dict(
        name=f"常值各向异性 K=[[{k11:g},{k12:g}],[{k12:g},{k22:g}]]",
        K=K, p=sin_p, gradp=sin_gradp,
        f=lambda X: (PI ** 2 * (k11 + k22) * sin_p(X)
                     - 2 * PI ** 2 * k12 * np.cos(PI * X[:, 0]) * np.cos(PI * X[:, 1])),
        g=lambda X: np.zeros(len(X)), bc=lambda mid, n: "D",
        mesh=lambda n: Mesh.rectangle(nx=n, ny=n))


def case_layer(kL=1.0, kR=100.0):
    """分层（串联）高对比：精确解分段线性、通量是常向量 Q=2/(1/κ_L+1/κ_R)。

    左右定压、上下不透水 ⇒ 同时检验非齐次 Dirichlet 与本质流量边界条件。
    ★ 不要用"光滑解 + 跳跃系数"当制造解：∇·(K∇p) 会多出界面 Dirac 源。
    """
    Q = 2.0 / (1.0 / kL + 1.0 / kR)
    ph = 1.0 - Q * 0.5 / kL

    def p(X):
        x = X[:, 0]
        return np.where(x < 0.5, 1.0 - Q * x / kL, ph - Q * (x - 0.5) / kR)

    def gradp(X):
        x = X[:, 0]
        return np.column_stack([np.where(x < 0.5, -Q / kL, -Q / kR), np.zeros(len(X))])

    K = lambda X: np.where(X[:, 0] < 0.5, kL, kR)        # 标量场（分段常值）
    return dict(name=f"分层高对比 κ_L={kL:g}, κ_R={kR:g}（精确解分段线性）",
                K=K, p=p, gradp=gradp, f=lambda X: np.zeros(len(X)), g=p,
                bc=lambda mid, n: ("D" if (mid[0] < 1e-12 or mid[0] > 1 - 1e-12) else "N"),
                mesh=lambda n: Mesh.rectangle(nx=n, ny=n), Q=Q)


def case_column(kappa=2.5):
    """达西柱实验：左 p=1、右 p=0、上下不透水 ⇒ p=1-x, q=(κ,0)，出口流量 Q=κ。"""
    K = float(kappa)
    return dict(name=f"达西柱实验 κ={kappa:g}（p: 1→0，上下不透水）",
                K=K, p=lambda X: 1.0 - X[:, 0],
                gradp=lambda X: np.column_stack([-np.ones(len(X)), np.zeros(len(X))]),
                f=lambda X: np.zeros(len(X)), g=lambda X: 1.0 - X[:, 0],
                bc=lambda mid, n: ("D" if (mid[0] < 1e-12 or mid[0] > 1 - 1e-12) else "N"),
                mesh=lambda n: Mesh.rectangle(nx=n, ny=n), kappa=K)


def case_fluxbc():
    """给定流量边界：左边 q·n=-1、右边 p=0、上下不透水 ⇒ p=1-x, q=(1,0)。"""
    return dict(name="给定流量边界（左侧 q·n=-1，右侧 p=0，上下不透水）",
                K=1.0, p=lambda X: 1.0 - X[:, 0],
                gradp=lambda X: np.column_stack([-np.ones(len(X)), np.zeros(len(X))]),
                f=lambda X: np.zeros(len(X)), g=lambda X: np.zeros(len(X)),
                bc=lambda mid, n: (("flux", lambda P: -np.ones(len(P)))
                                   if mid[0] < 1e-12 else
                                   ("D" if mid[0] > 1 - 1e-12 else "N")),
                mesh=lambda n: Mesh.rectangle(nx=n, ny=n))


def case_3d(k3=4.0):
    """三维各向异性 K=diag(1,1,k3)，p=sinπx sinπy sinπz，f=π²(2+k3)p。"""
    K = np.diag([1.0, 1.0, k3])

    def p(X):
        return (np.sin(PI * X[:, 0]) * np.sin(PI * X[:, 1]) * np.sin(PI * X[:, 2]))

    def gradp(X):
        s, c = np.sin, np.cos
        return PI * np.column_stack([
            c(PI * X[:, 0]) * s(PI * X[:, 1]) * s(PI * X[:, 2]),
            s(PI * X[:, 0]) * c(PI * X[:, 1]) * s(PI * X[:, 2]),
            s(PI * X[:, 0]) * s(PI * X[:, 1]) * c(PI * X[:, 2])])

    return dict(name=f"三维四面体网格 K=diag(1,1,{k3:g})",
                K=K, p=p, gradp=gradp,
                f=lambda X: PI ** 2 * (2.0 + k3) * p(X),
                g=lambda X: np.zeros(len(X)), bc=lambda mid, n: "D",
                mesh=lambda n: Mesh.box(nx=n, ny=n, nz=n, element_type="tet"))


# =====================================================================
# 7. 主程序
# =====================================================================
def main():
    out = []

    def say(s=""):
        print(s)
        out.append(str(s))

    say("=" * 100)
    say("用 MosicaFE 的 RT0×P0 空间求解 Darcy 方程")
    say(f"   库目录（只读，未修改）：{PKG_DIR}")
    say("   A_ij = ∫ (K^{-1}φ_i)·φ_j  ；其余块与求解流程全部复用库")
    say("=" * 100)

    # ---------- ① 库接口概览 ----------
    say("")
    say("【① 库接口】网格 → 空间 → 两个块")
    mesh = Mesh.rectangle(nx=8, ny=8)
    V = FESpace.rt0(mesh)
    say(f"   网格：{mesh.summary()}")
    say(f"   空间：{V.summary()}")
    say(f"   通量自由度（每个面一个）= {V.n_flux}；压力自由度（每个单元一个）= {V.n_pressure}")
    say(f"   边界面个数（压力 Dirichlet 是自然边界条件）= {V.n_boundary_facets}")
    M = V.flux_mass_matrix()
    B = V.divergence_matrix()
    say(f"   V.flux_mass_matrix() -> {type(M).__name__} {tuple(M.shape)}"
        f"（∫φ_i·φ_j，精确闭式）")
    say(f"   V.divergence_matrix() -> {type(B).__name__} {tuple(B.shape)}"
        f"（∫∇·φ_i = ±1，无需求积）")

    # ---------- ② 加权质量矩阵自检 ----------
    say("")
    say("=" * 100)
    say("【② 自检】V.weighted_flux_mass_matrix()（库内置原语）")
    say("=" * 100)
    A1 = V.weighted_flux_mass_matrix(1.0, mode="cell")
    ref = V.flux_mass_matrix()
    d1 = abs(A1.toarray() - ref.toarray()).max() / abs(ref.toarray()).max()
    say(f"   K=I 时与库的 V.flux_mass_matrix() 相对偏差 = {d1:.3e}（应 ~1e-16）")
    Krand = np.array([[2.5, 0.7], [0.7, 1.3]])
    A_c = V.weighted_flux_mass_matrix(Krand, mode="cell").toarray()
    A_q = V.weighted_flux_mass_matrix(Krand, mode="quad", quadrature_order=6).toarray()
    say(f"   K=[[2.5,0.7],[0.7,1.3]]：闭式(cell) 与 求积(quad) 的相对偏差 = "
        f"{abs(A_c - A_q).max() / abs(A_c).max():.3e}（闭式对常值 K 精确）")
    Kfun = lambda X: 1.0 + X[:, 0]               # 光滑变系数
    A_c = V.weighted_flux_mass_matrix(Kfun, mode="cell").toarray()
    A_q = V.weighted_flux_mass_matrix(Kfun, mode="quad", quadrature_order=6).toarray()
    say(f"   K=(1+x)I（光滑）：cell 取形心值与 quad 求积的偏差 = "
        f"{abs(A_c - A_q).max() / abs(A_q).max():.3e}（差 O(h²)，两种都可用）")

    # ---------- ③ 与库的 MixedPoissonProblem 对拍 ----------
    say("")
    say("=" * 100)
    say("【③ 对拍】K=I 时，DarcyProblem 与库里的 MixedPoissonProblem 必须给同一个解")
    say("=" * 100)
    kind = case_iso()
    for n in (8, 16):
        mesh = kind["mesh"](n)
        V = FESpace.rt0(mesh)
        # 用同一个源项求积阶，两个解应当**逐位相同**（否则差的是求积误差，不是算法差异）
        lib = MixedPoissonProblem(V, f=kind["f"], quadrature_order=6)
        Q0, P0 = lib.solve()
        darcy = DarcyProblem(V, f=kind["f"], g=kind["g"], K=kind["K"], bc=kind["bc"])
        Q1, P1 = darcy.solve()
        say(f"   n={n:3d}  压力 max|ΔP| = {np.abs(P0 - P1).max():.3e}   "
            f"面通量 max|ΔQ| = {np.abs(Q0 - Q1).max():.3e}"
            f"   （守恒偏差 {darcy.check_conservation():.2e}）")

    # ---------- ④ 收敛阶 ----------
    say("")
    say("=" * 100)
    say("【④ 收敛阶】误差用库原生的 mixed_error_norms（压力 P0 / 通量 RT0 重构）")
    say("=" * 100)
    conv = []
    for case, levels in ((case_aniso(), (4, 8, 16, 32, 64)),
                         (case_3d(), (4, 8, 16))):
        say("")
        say(f"   —— {case['name']} ——")
        hs, ep, eq = [], [], []
        for n in levels:
            mesh = case["mesh"](n)
            V = FESpace.rt0(mesh)
            problem = DarcyProblem(V, f=case["f"], g=case["g"], K=case["K"],
                                   bc=case["bc"], mode="cell")
            Q, P = problem.solve()
            norms = mixed_error_norms(V, P, Q, case["p"], flux_of(case["K"], case["gradp"]))
            h = mesh.get_mesh_size()
            hs.append(h)
            ep.append(norms["p_l2"])
            eq.append(norms["q_l2"])
            say(f"      n={n:3d}  h={h:.5f}  cells={mesh.n_cells:5d}  facets={V.n_flux:5d}  "
                f"L2(p)={norms['p_l2']:.4e}  L2(q)={norms['q_l2']:.4e}  "
                f"守恒={problem.check_conservation():.1e}")
        rp = estimate_convergence_rate(hs, ep, use_last=min(3, len(hs)))
        rq = estimate_convergence_rate(hs, eq, use_last=min(3, len(hs)))
        say(f"      拟合阶：压力 {rp:.3f}（理论 1），通量 {rq:.3f}（理论 1）")
        conv.append((case["name"], hs, ep, eq))

    # ---------- ⑤ 物理算例 ----------
    say("")
    say("=" * 100)
    say("【⑤ 物理算例】")
    say("=" * 100)

    # 达西柱实验
    case = case_column()
    mesh = case["mesh"](16)
    V = FESpace.rt0(mesh)
    problem = DarcyProblem(V, f=case["f"], g=case["g"], K=case["K"], bc=case["bc"])
    Q, P = problem.solve(method="direct")
    # 出口流量 = 右边界面上外法向通量之和
    topo = V.topology
    nodes = np.asarray(mesh.nodes, float)
    out_flow = 0.0
    for c in range(mesh.n_cells):
        for i in range(mesh.dim + 1):
            fct = int(topo.cell_facets[c, i])
            if topo.counts[fct] != 1:
                continue
            mid = nodes[topo.facets[fct]].mean(axis=0)
            if abs(mid[0] - 1.0) < 1e-12:
                out_flow += Q[c, i]               # Q 已经是"外法向"通量
    say(f"   达西柱实验（n=16，直接解）：数值出口流量 ΣQ = {out_flow:.10f}，"
        f"达西定律 Q=κAΔp/L = {case['kappa']:.10f}，"
        f"相对误差 {abs(out_flow - case['kappa']) / case['kappa']:.2e}")
    nrm = mixed_error_norms(V, P, Q, case["p"], flux_of(case["K"], case["gradp"]))
    say(f"      通量误差 L2(q) = {nrm['q_l2']:.3e}（常通量场精确可表示）；"
        f"压力误差 L2(p) = {nrm['p_l2']:.3e}（P0 逼近线性压力 ⇒ O(h)）")
    pts = np.array([[0.5, 0.5], [0.25, 0.75]])
    say(f"      flux_at_points（库原生重构）q_h(0.5,0.5) = "
        f"{np.round(problem.flux_at_points(pts)[0], 8)}")

    # 给定流量边界
    case = case_fluxbc()
    mesh = case["mesh"](16)
    V = FESpace.rt0(mesh)
    problem = DarcyProblem(V, f=case["f"], g=case["g"], K=case["K"], bc=case["bc"])
    Q, P = problem.solve(method="direct")
    topo, nodes = V.topology, np.asarray(mesh.nodes, float)
    left = 0.0
    for c in range(mesh.n_cells):
        for i in range(mesh.dim + 1):
            fct = int(topo.cell_facets[c, i])
            if topo.counts[fct] != 1:
                continue
            mid = nodes[topo.facets[fct]].mean(axis=0)
            if mid[0] < 1e-12:
                left += Q[c, i]                   # 左边外面法向 = (-1,0)
    nrm = mixed_error_norms(V, P, Q, case["p"], flux_of(case["K"], case["gradp"]))
    say(f"   给定流量边界：左边界 ∫q·n ds = {left:.10f}（给定 -1）；"
        f"L2(q) = {nrm['q_l2']:.3e}（精确解 q=(1,0) 是常向量）")

    # 分层高对比
    case = case_layer()
    for n in (8, 16, 32):
        mesh = case["mesh"](n)
        V = FESpace.rt0(mesh)
        problem = DarcyProblem(V, f=case["f"], g=case["g"], K=case["K"],
                               bc=case["bc"], mode="cell")
        Q, P = problem.solve()
        topo = V.topology
        nodes = np.asarray(mesh.nodes, float)
        Phi = np.asarray(problem.solution, float)[: V.n_flux]     # 全局面自由度
        meas = topo.measure()
        normals = np.asarray(topo.normals, float)
        fl = []
        for fct in range(V.n_flux):                              # ★ 扫描**所有**面
            mid = nodes[V.facet_vertices(fct)].mean(axis=0)      # （界面是内部面）
            if abs(mid[0] - 0.5) < 1e-12:
                fl.append(Phi[fct] * np.sign(normals[fct, 0]) / meas[fct])   # 通量密度 q_x
        nrm = mixed_error_norms(V, P, Q, case["p"], flux_of(case["K"], case["gradp"]))
        Qex = case["Q"]
        say(f"   分层介质 n={n:3d}：界面通量密度 {np.mean(fl):.10f}（调和平均预测 "
            f"{Qex:.10f}，相对误差 {abs(np.mean(fl) - Qex) / Qex:.2e}）"
            f"  面数={len(fl)}  L2(q)={nrm['q_l2']:.2e}  L2(p)={nrm['p_l2']:.2e}")
    say(f"      对照：算术平均 {(1 + 100) / 2:.4f}（错）"
        f"；调和平均 {2 / (1 / 1.0 + 1 / 100.0):.10f}（对）")

    # ---------- ⑥ 注册表接口 ----------
    say("")
    say("=" * 100)
    say("【⑥ 注册表】因为它带了 @register_problem(\"darcy\")，可以直接用库的 create_problem")
    say("=" * 100)
    from MosicaFE.physics.registry import available_problems, create_problem
    say(f"   已注册的 PDE：{available_problems()}")
    mesh = Mesh.rectangle(nx=8, ny=8)
    V = FESpace.rt0(mesh)
    prob = create_problem("darcy", V, f=case_aniso()["f"], K=case_aniso()["K"])
    Q, P = prob.solve()
    say(f"   create_problem('darcy', V, f=..., K=...) -> {prob}")
    say(f"   解：max|P| = {np.abs(P).max():.4e}，max|Q| = {np.abs(Q).max():.4e}")

    # ---------- ⑦ 误差可视化 ----------
    say("")
    say("=" * 100)
    say("【⑦ 误差可视化】逐单元误差场 + 通量箭头 + 收敛曲线（Pillow 出图，不用 matplotlib）")
    say("=" * 100)
    if dp is None:
        say(f"   [跳过] 未能导入 darcy_plot（{_DP_ERR}）")
    else:
        os.makedirs(FIGDIR, exist_ok=True)
        figs = []

        # --- 各向异性：误差的空间分布 ---
        case = case_aniso()
        mesh = case["mesh"](32)
        V = FESpace.rt0(mesh)
        problem = DarcyProblem(V, f=case["f"], g=case["g"], K=case["K"], bc=case["bc"])
        Q, P = problem.solve()
        nodes, cells, ctr = cell_arrays(V)
        qh = flux_at_centroids(V, Q, ctr)
        qe = np.array([flux_of(case["K"], case["gradp"])(p) for p in ctr])
        ep, eq = cell_errors_2d(V, P, Q, case)
        f1 = dp.cell_heatmap(nodes, cells, ep, os.path.join(FIGDIR, "err_aniso_p.png"),
                             title="压力误差 |p-p_h|（单元 L2，各向异性 K）",
                             cbar_label="L2/单元", cmap="coolwarm", width=520)
        f2 = dp.cell_heatmap(nodes, cells, eq, os.path.join(FIGDIR, "err_aniso_q.png"),
                             title="通量误差 |q-q_h|（单元 L2）", cbar_label="L2/单元",
                             cmap="coolwarm", log=True, width=520)
        f3 = dp.mesh_quiver(nodes, cells, ctr, qh, os.path.join(FIGDIR, "flux_aniso.png"),
                            title="q_h（蓝）与精确 q（红）", width=520,
                            vectors_extra=qe, label="RT0 数值解", label_extra="精确解",
                            stride=quiver_stride(V))
        fig_a = dp.hstack([f1, f2, f3], os.path.join(FIGDIR, "err_aniso.png"),
                          labels=["(a) 压力误差", "(b) 通量误差", "(c) 通量对比"])
        figs.append(fig_a)
        say(f"   各向异性 n=32：单元压力误差 {ep.min():.3e}~{ep.max():.3e}，"
            f"通量误差 {eq.min():.3e}~{eq.max():.3e}")
        say(f"      min/max 比 = {eq.max()/max(eq.min(),1e-300):.1f}（误差在区域上明显不均匀）"
            f"；总 L2：{mixed_error_norms(V, P, Q, case['p'], flux_of(case['K'], case['gradp']))}")

        # --- 分层高对比：渗透率场 + 误差 ---
        case = case_layer()
        mesh = case["mesh"](32)
        V = FESpace.rt0(mesh)
        problem = DarcyProblem(V, f=case["f"], g=case["g"], K=case["K"],
                               bc=case["bc"], mode="cell")
        Q, P = problem.solve()
        nodes, cells, ctr = cell_arrays(V)
        kappa = KMat(case["K"], ctr)[:, 0, 0]
        ep, eq = cell_errors_2d(V, P, Q, case)
        g1 = dp.cell_heatmap(nodes, cells, kappa, os.path.join(FIGDIR, "layer_k.png"),
                             title="渗透率 κ（左 1 / 右 100）", cbar_label="κ",
                             cmap="viridis", show_mesh=False, width=520)
        g2 = dp.cell_heatmap(nodes, cells, ep, os.path.join(FIGDIR, "layer_p.png"),
                             title="压力误差 |p-p_h|（界面上有拐点）", cbar_label="L2/单元",
                             cmap="coolwarm", width=520)
        # 通量误差是机零（≈1e-11），画热图没信息量；改画剖面：精确解有折点，P0 是阶梯
        px = np.linspace(0.0, 1.0, 401)
        py = case["p"](np.column_stack([px, np.full_like(px, 0.5)]))
        sx, sy = p0_step_along_line(V, P, 0.5, 1.0 / 32)
        g3 = dp.profile_plot(
            [dict(label="精确解 p（分段线性，x=0.5 处有折点）", x=px, y=py,
                  color=(214, 39, 40)),
             dict(label="RT0-P0 数值解 p_h（分片常数）", x=sx, y=sy,
                  color=(31, 119, 180), dash=True)],
            os.path.join(FIGDIR, "layer_profile.png"),
            title="沿 y=0.5 的压力剖面（左半渗透率小 ⇒ 斜率大 ⇒ P0 误差大）",
            xlabel="x", ylabel="p", vlines=[0.5], width=520, height=560,
            legend_pos="tr")
        fig_b = dp.hstack([g1, g2, g3], os.path.join(FIGDIR, "err_layer.png"),
                          labels=["(a) 渗透率场", "(b) 压力误差", "(c) 剖面 p(x,0.5)"])
        figs.append(fig_b)
        say(f"   分层 n=32：通量误差 {eq.max():.2e}（机器零）；压力误差 "
            f"{ep.min():.3e}~{ep.max():.3e}（界面附近最大，符合分段线性解的特点）")

        # --- 柱实验：均匀通量 ---
        case = case_column()
        mesh = case["mesh"](16)
        V = FESpace.rt0(mesh)
        problem = DarcyProblem(V, f=case["f"], g=case["g"], K=case["K"], bc=case["bc"])
        Q, P = problem.solve(method="direct")
        nodes, cells, ctr = cell_arrays(V)
        qh = flux_at_centroids(V, Q, ctr)
        f4 = dp.mesh_quiver(nodes, cells, ctr, qh, os.path.join(FIGDIR, "flux_column.png"),
                            title="达西柱实验 q_h：水平、均匀 = (κ, 0)", width=560,
                            label="RT0 数值解")
        figs.append(f4)
        say(f"   柱实验：q_h 全部分量一致 = {np.unique(np.round(qh, 8), axis=0).tolist()}")

        # --- 收敛曲线（2D 各向异性 + 3D 各向异性） ---
        hs2, p2, q2 = [], [], []
        case = case_aniso()
        for n in (4, 8, 16, 32, 64):
            mesh = case["mesh"](n)
            V = FESpace.rt0(mesh)
            pr = DarcyProblem(V, f=case["f"], g=case["g"], K=case["K"], bc=case["bc"])
            QQ, PP = pr.solve()
            nrm = mixed_error_norms(V, PP, QQ, case["p"], flux_of(case["K"], case["gradp"]))
            hs2.append(mesh.get_mesh_size())
            p2.append(nrm["p_l2"])
            q2.append(nrm["q_l2"])
        hs3, p3, q3 = [], [], []
        case = case_3d()
        for n in (4, 8, 16):
            mesh = case["mesh"](n)
            V = FESpace.rt0(mesh)
            pr = DarcyProblem(V, f=case["f"], g=case["g"], K=case["K"], bc=case["bc"])
            QQ, PP = pr.solve()
            nrm = mixed_error_norms(V, PP, QQ, case["p"], flux_of(case["K"], case["gradp"]))
            hs3.append(mesh.get_mesh_size())
            p3.append(nrm["p_l2"])
            q3.append(nrm["q_l2"])
        f5 = dp.loglog_convergence(
            hs2, [dict(label="2D 各向异性：压力 L2", h=hs2, e=p2, color=(31, 119, 180)),
                  dict(label="2D 各向异性：通量 L2", h=hs2, e=q2, color=(31, 119, 180),
                       dash=True),
                  dict(label="3D 各向异性：压力 L2", h=hs3, e=p3, color=(214, 39, 40)),
                  dict(label="3D 各向异性：通量 L2", h=hs3, e=q3, color=(214, 39, 40),
                       dash=True)],
            os.path.join(FIGDIR, "conv_2d3d.png"),
            title="RT0-P0 求解 Darcy：误差随 h 的下降（虚线 = 通量，参考斜率 1）",
            ref_slope=1.0)
        figs.append(f5)
        say(f"   收敛曲线：2D 阶 {estimate_convergence_rate(hs2, p2, use_last=3):.3f} / "
            f"{estimate_convergence_rate(hs2, q2, use_last=3):.3f}；"
            f"3D 阶 {estimate_convergence_rate(hs3, p3, use_last=2):.3f} / "
            f"{estimate_convergence_rate(hs3, q3, use_last=2):.3f}")

        # --- 合并成一张总图 ---
        dp.hstack([f4, f5], os.path.join(FIGDIR, "err_summary.png"),
                  labels=["(a) 柱实验通量", "(b) 收敛曲线"])
        say("   图片目录：figs/mosicafe/")
        for pth in sorted(os.listdir(FIGDIR)):
            if pth.lower().endswith(".png"):
                say(f"      {os.path.relpath(os.path.join(FIGDIR, pth), HERE).replace(os.sep, '/')}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "结果_darcy_mosicafe.txt"), "w", encoding="utf-8-sig") as fp:
        fp.write("\n".join(out) + "\n")
    print("\n[结果] 已保存 结果_darcy_mosicafe.txt")


if __name__ == "__main__":
    main()
