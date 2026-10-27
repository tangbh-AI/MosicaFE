# -*- coding: utf-8 -*-
"""
rt0_darcy_2d.py —— 二维 Darcy（达西）方程的 RT0-P0 混合有限元
================================================================
问题（混合形式，q 为达西流速/通量，p 为压力/水头）：

        K^{-1} q + ∇p = 0        in  Ω          （达西定律  q = -K ∇p）
        ∇·q            = f        in  Ω          （质量守恒）
        p              = g_D      on  Γ_D        （压力 Dirichlet，本质/natural? → 见下）
        q·n            = g_N      on  Γ_N        （给定流量，本质边界条件）

弱形式：求 (q, p) ∈ V_h × Q_h，使

        (K^{-1} q, v) - (p, ∇·v) = -<g_D, v·n>_{Γ_D}   ∀ v ∈ V_h0
        (∇·q, w)                 =  (f, w)             ∀ w ∈ Q_h

        V_h  = RT0(三角网格)  ⊂ H(div;Ω)     （通量，每条内部边一个自由度）
        Q_h  = P0(分片常数)                  （压力，每个单元一个自由度）
        V_h0 = { v ∈ V_h : v·n = 0 on Γ_N }  （不透水边界上法向通量置零）

★★ 与泊松方程版本（rt0_2d.py）唯一的数学差别 ★★
   泊松：   A_ij = ∫ φ_i·φ_j dx          （质量矩阵）
   Darcy：  A_ij = ∫ (K^{-1} φ_i)·φ_j dx （渗透率加权质量矩阵）
   其余（B = ±1、方程结构、鞍点求解、RT0 基函数、散度公式）完全不变。

★ 边界条件的本质/自然之分（最容易写错的地方）
   压力 Dirichlet  p = g_D : 混合形式的**自然**边界条件，只进第一个方程的右端
                             -<g_D, v·n>_{Γ_D}，不要用它去"固定边界单元的压力自由度"。
   流量（不透水或给定流量）: **本质**边界条件，必须从通量空间里去掉/给定这些边的自由度。
   ★ 只给纯 Neumann（全不透水）时解只到相差一个常数，需要额外固定一个单元的压力。

运行：  python rt0_darcy_2d.py
输出：  终端表格；结果_darcy.txt；data_darcy.csv；fig_darcy.tex

作者：赵典
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "RT0收敛阶验证"))
from rt0_tools import (ref_quad, simplex_quad, simplex_volume,  # noqa: E402
                       fit_rate, write_csv, write_pgfplots)

try:                                    # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FIGDIR = os.path.join(_HERE, "figs", "standalone")
try:                                    # 误差可视化（Pillow 光栅，不用 matplotlib）
    import darcy_plot as dp
except Exception as _e:                 # pragma: no cover
    dp = None
    _DP_ERR = _e

REFQ_ASM = ref_quad(4, 2)      # 组装加权质量矩阵用（K 光滑时需要较高精度）
REFQ_ERR = ref_quad(12, 2)     # 误差范数用（避免求积误差污染收敛阶）


# =====================================================================
# 0. 渗透率张量的两个基本操作
# =====================================================================
# 约定：K(X) 可以返回
#         (nq,)      各向同性标量渗透率 k(x)
#         (nq, 2, 2) 对称正定张量 K(x)
def K_apply(K, X, g):
    """返回 K(x) @ g（逐点），g: (nq,2) → (nq,2)。用于算精确通量 q = -K∇p。"""
    Kx = K(X)
    if Kx.ndim == 1:
        return Kx[:, None] * g
    return np.einsum("qab,qb->qa", Kx, g)


def Kinv_dot(K, X, a, b):
    """返回 φ_i^T K^{-1} φ_j（逐点），a,b: (nq,2) → (nq,)。组装 A 用。"""
    Kx = K(X)
    if Kx.ndim == 1:                       # 标量渗透率：K^{-1} = 1/k
        return (1.0 / Kx) * np.sum(a * b, axis=1)
    return np.einsum("qa,qab,qb->q", a, np.linalg.inv(Kx), b)


# =====================================================================
# 1. 网格与边表（与 rt0_2d.py 同一套约定，改成可剖分矩形）
# =====================================================================
def mesh_rect(nx, ny, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    """矩形区域 nx×ny 结构化三角网格，每个四边形剖成 2 个逆时针三角形。"""
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    elems = []
    for j in range(ny):
        for i in range(nx):
            bl, br = idx[j, i], idx[j, i + 1]
            tl, tr = idx[j + 1, i], idx[j + 1, i + 1]
            elems.append([bl, br, tr])
            elems.append([bl, tr, tl])
    return nodes, np.array(elems, dtype=np.int64)


def build_edges(nodes, elems):
    """全局边表 + 每条边在每个单元里的局部编号与符号（与 rt0_2d.py 相同）。

    elem_sign[e,i] = n_loc·n_glob = ±1；
    全局基函数在单元 e 上的限制 = elem_sign * φ_i^local，
    且在它自己的边上 ∫ φ_k·n_glob ds = 1。
    """
    edges, edge_id = [], {}
    nK = len(elems)
    elem_edge = np.zeros((nK, 3), dtype=np.int64)
    for e, tri in enumerate(elems):
        for i in range(3):                                   # 第 i 条边相对顶点 i
            b0, b1 = tri[(i + 1) % 3], tri[(i + 2) % 3]
            key = (b0, b1) if b0 < b1 else (b1, b0)
            if key not in edge_id:
                edge_id[key] = len(edges)
                edges.append(key)
            elem_edge[e, i] = edge_id[key]
    edges = np.array(edges, dtype=np.int64)
    ne = len(edges)

    n_glob = np.zeros((ne, 2))                               # 小编号顶点→大编号顶点右旋 90°
    for k, (u0, u1) in enumerate(edges):
        d = nodes[u1] - nodes[u0]
        n_glob[k] = np.array([d[1], -d[0]]) / np.hypot(d[0], d[1])

    elem_sign = np.zeros((nK, 3))
    n_adj = np.zeros(ne, dtype=np.int64)
    bnd = {}
    for e, tri in enumerate(elems):
        for i in range(3):
            b0, b1 = tri[(i + 1) % 3], tri[(i + 2) % 3]
            d = nodes[b1] - nodes[b0]
            L = np.hypot(d[0], d[1])
            n_loc = np.array([d[1], -d[0]]) / L
            mid = 0.5 * (nodes[b0] + nodes[b1])
            if np.dot(n_loc, mid - nodes[tri[i]]) < 0.0:     # 翻成外法向
                n_loc = -n_loc
            k = elem_edge[e, i]
            elem_sign[e, i] = np.round(np.dot(n_loc, n_glob[k]))
            n_adj[k] += 1
            bnd.setdefault(k, (n_loc.copy(), L))
    boundary = []
    for k in range(ne):
        if n_adj[k] == 1:
            n_out, L = bnd[k]
            u0, u1 = edges[k]
            boundary.append((k, n_out, L, n_glob[k], u0, u1))
    return edges, elem_edge, elem_sign, n_adj, boundary


def _gauss01(n):
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (t + 1.0), 0.5 * w


def boundary_data(nodes, boundary, bc):
    """按 bc 把边界边分类。

    bc(mid, n_out) 返回
        'D'           压力 Dirichlet（自然边界条件，进右端）
        'N'           不透水 q·n = 0（本质，自由度置零）
        ('flux', gN)  给定法向流量 q·n_out = gN(x)（本质，自由度取给定值；gN 为可调用）
    返回 (dir_edges, known)：dir_edges 供右端积分；known: {dof: 给定值}
    """
    tx, tw = _gauss01(12)
    dir_edges, known = [], {}
    for (k, n_out, L, n_glob, u0, u1) in boundary:
        mid = 0.5 * (nodes[u0] + nodes[u1])
        kind = bc(mid, n_out)
        if kind == 'D':
            dir_edges.append((k, n_out, L, n_glob, u0, u1))
        elif kind == 'N':
            known[k] = 0.0
        else:
            gN = kind[1]
            c = float(np.round(np.dot(n_out, n_glob)))
            pts = nodes[u0][None, :] + tx[:, None] * (nodes[u1] - nodes[u0])[None, :]
            known[k] = c * L * float(np.sum(tw * gN(pts)))
    return dir_edges, known


# =====================================================================
# 2. 组装（与泊松版的差别只有 A 的加权）
# =====================================================================
def assemble(nodes, elems, elem_edge, elem_sign, dir_edges, f_exact, gD_exact, K):
    nK = len(elems)
    ne = int(elem_edge.max()) + 1

    rows, cols, vals = [], [], []       # A_ij = ∫ K^{-1} φ_i · φ_j
    rowsB, colsB, valsB = [], [], []    # B_{iK} = ∫_K ∇·φ_i = ±1
    F = np.zeros(nK)
    rhs_q = np.zeros(ne)

    for e, tri in enumerate(elems):
        P = nodes[tri]
        area = simplex_volume(P)
        X, w = simplex_quad(P, *REFQ_ASM)
        phi = [(X - P[i]) / (2.0 * area) for i in range(3)]           # 局部 RT0 基
        M = np.array([[np.sum(w * Kinv_dot(K, X, phi[i], phi[j]))
                       for j in range(3)] for i in range(3)])
        for i in range(3):
            gi, si = elem_edge[e, i], elem_sign[e, i]
            rowsB.append(gi); colsB.append(e); valsB.append(si)
            for j in range(3):
                rows.append(gi); cols.append(elem_edge[e, j])
                vals.append(si * elem_sign[e, j] * M[i, j])
        F[e] = np.sum(w * f_exact(X))          # (f, ψ_K)，ψ_K ≡ 1

    A = sp.coo_matrix((vals, (rows, cols)), shape=(ne, ne)).tocsr()
    B = sp.coo_matrix((valsB, (rowsB, colsB)), shape=(ne, nK)).tocsr()

    # 第一个方程的右端：-<g_D, v·n>_{Γ_D}
    if gD_exact is not None:
        tx, tw = _gauss01(12)
        for (k, n_out, L, n_glob, u0, u1) in dir_edges:
            c = float(np.round(np.dot(n_out, n_glob)))
            pts = nodes[u0][None, :] + tx[:, None] * (nodes[u1] - nodes[u0])[None, :]
            rhs_q[k] += -c * float(np.sum(tw * gD_exact(pts)))
    return A, B, F, rhs_q, ne


def restrict_system(A, B, F, rhs_q, known):
    """处理本质边界条件：已知/置零的通量自由度从未知量中消去。

    A_ff q_f = rhs_f - A_fc q_c            （已知列搬到第一个方程右端）
    B_f^T q_f = F - B_c^T q_c             （已知通量对守恒方程的贡献也要搬走！）
    行、列同时删去 = 限制到 V_h0。不透水时 q_c = 0，第二式右端不变。
    """
    ne, nK = B.shape
    known_dofs = np.array(sorted(known), dtype=np.int64)
    mask = np.ones(ne, dtype=bool)
    mask[known_dofs] = False
    free = np.flatnonzero(mask)
    if len(known_dofs) == 0:
        return A, B, F, rhs_q, free
    qc = np.array([known[k] for k in known_dofs])
    rhs = rhs_q[free] - A[free][:, known_dofs] @ qc
    F = F - B[known_dofs, :].T @ qc
    return A[free][:, free], B[free, :], F, rhs, free


# =====================================================================
# 3. 求解鞍点系统  [ A  -B ][q]   [rhs_q]
#                    [ Bᵀ  0 ][p] = [  F  ]
# =====================================================================
def solve_mixed(A, B, F, rhs_q, rtol=1e-12, maxiter=20000):
    """消去 q：q = A^{-1}(B p + rhs_q) ⇒ S p = F - BᵀA^{-1}rhs_q，S = BᵀA^{-1}B。
    A 对称正定（K^{-1} 加权质量矩阵），S 亦对称正定 ⇒ 用 CG。"""
    lu = spla.splu(A.tocsc())
    q0 = lu.solve(rhs_q)
    nK = B.shape[1]
    S = spla.LinearOperator((nK, nK), matvec=lambda p: B.T @ lu.solve(B @ p), dtype=float)
    p, info = spla.cg(S, F - B.T @ q0, rtol=rtol, atol=0.0, maxiter=maxiter)
    if info != 0:
        print(f"    [警告] CG 未收敛 (info={info})，可改用 solve_mixed_direct")
    return lu.solve(B @ p) + q0, p


def solve_mixed_direct(A, B, F, rhs_q):
    """直接解鞍点系统（小规模时用于交叉验证 Schur+CG）。"""
    nq = B.shape[0]
    K = sp.bmat([[A, -B], [B.T, None]], format="csc")
    sol = spla.spsolve(K, np.concatenate([rhs_q, F]))
    return sol[:nq], sol[nq:]


# =====================================================================
# 4. 误差范数与 Darcy 专属诊断
# =====================================================================
def compute_errors(nodes, elems, elem_edge, elem_sign, q, p, exact, K):
    e_p2 = e_q2 = e_d2 = 0.0
    for e, tri in enumerate(elems):
        P = nodes[tri]
        area = simplex_volume(P)
        X, w = simplex_quad(P, *REFQ_ERR)
        qh = np.zeros_like(X)
        for i in range(3):
            coef = elem_sign[e, i] * q[elem_edge[e, i]]
            qh += coef * (X - P[i]) / (2.0 * area)
        divqh = sum(elem_sign[e, i] * q[elem_edge[e, i]] for i in range(3)) / area
        pex = exact["p"](X)
        gex = exact["gradp"](X)
        qex = -K_apply(K, X, gex)
        e_p2 += np.sum(w * (pex - p[e]) ** 2)
        e_q2 += np.sum(w * ((qex - qh) ** 2).sum(axis=1))
        e_d2 += np.sum(w * (exact["f"](X) - divqh) ** 2)
    return np.sqrt(e_p2), np.sqrt(e_q2), np.sqrt(e_d2)


def cell_errors(nodes, elems, elem_edge, elem_sign, q, p, exact, K):
    """逐单元的 L2 误差（供误差可视化用）：``||p-p_h||_K`` 与 ``||q-q_h||_K``。"""
    e_p = np.zeros(len(elems))
    e_q = np.zeros(len(elems))
    for e, tri in enumerate(elems):
        P = nodes[tri]
        area = simplex_volume(P)
        X, w = simplex_quad(P, *REFQ_ERR)
        qh = np.zeros_like(X)
        for i in range(3):
            coef = elem_sign[e, i] * q[elem_edge[e, i]]
            qh += coef * (X - P[i]) / (2.0 * area)
        qex = -K_apply(K, X, exact["gradp"](X))
        e_p[e] = np.sqrt(np.sum(w * (exact["p"](X) - p[e]) ** 2) / area)
        e_q[e] = np.sqrt(np.sum(w * ((qex - qh) ** 2).sum(axis=1)) / area)
    return e_p, e_q


def flux_at_centroids(nodes, elems, elem_edge, elem_sign, q):
    """在单元形心重构 RT0 通量 q_h（用于箭头图与误差图）。"""
    out = np.zeros((len(elems), 2))
    for e, tri in enumerate(elems):
        P = nodes[tri]
        area = simplex_volume(P)
        ctr = P.mean(axis=0)[None, :]
        qh = np.zeros_like(ctr)
        for i in range(3):
            coef = elem_sign[e, i] * q[elem_edge[e, i]]
            qh += coef * (ctr - P[i]) / (2.0 * area)
        out[e] = qh[0]
    return out


def conservation_check(nodes, elems, elem_edge, elem_sign, q, f_exact):
    """RT0-P0 的招牌性质：**逐单元局部质量守恒**
       |K| ∇·q_h|_K = ∫_K f dx   ⇔   ∇·q_h|_K = f 的单元平均值（精确到线性方程组残差）。
    返回 max_K |∇·q_h - f̄_K|。"""
    worst = 0.0
    for e, tri in enumerate(elems):
        P = nodes[tri]
        area = simplex_volume(P)
        X, w = simplex_quad(P, *REFQ_ERR)
        fbar = np.sum(w * f_exact(X)) / area
        div = sum(elem_sign[e, i] * q[elem_edge[e, i]] for i in range(3)) / area
        worst = max(worst, abs(div - fbar))
    return worst


# =====================================================================
# 5. 算例（制造解：给定 p，算出 q = -K∇p 与 f = ∇·q）
# =====================================================================
PI = np.pi


def case_iso():
    """算例 1：K = I —— 退化为泊松方程，用来和 rt0_2d.py 的结果逐一对照。"""
    return dict(
        name=r"$K=I$（各向同性，退化为 Poisson）",
        K=lambda X: np.ones(len(X)),
        p=lambda X: np.sin(PI * X[:, 0]) * np.sin(PI * X[:, 1]),
        gradp=lambda X: PI * np.column_stack([np.cos(PI * X[:, 0]) * np.sin(PI * X[:, 1]),
                                              np.sin(PI * X[:, 0]) * np.cos(PI * X[:, 1])]),
        f=lambda X: 2 * PI ** 2 * np.sin(PI * X[:, 0]) * np.sin(PI * X[:, 1]),
        gD=lambda P: np.zeros(len(P)),
        bc=lambda mid, n: 'D',
    )


def case_aniso():
    """算例 2：常值各向异性 K = [[3,1],[1,2]]（非对角项专门检验加权质量矩阵）。"""
    Kmat = np.array([[3.0, 1.0], [1.0, 2.0]])
    tr, k12 = Kmat[0, 0] + Kmat[1, 1], Kmat[0, 1]
    return dict(
        name=r"常值各向异性 $K=\begin{pmatrix}3&1\\1&2\end{pmatrix}$",
        K=lambda X: np.broadcast_to(Kmat, (len(X), 2, 2)).copy(),
        p=lambda X: np.sin(PI * X[:, 0]) * np.sin(PI * X[:, 1]),
        gradp=lambda X: PI * np.column_stack([np.cos(PI * X[:, 0]) * np.sin(PI * X[:, 1]),
                                              np.sin(PI * X[:, 0]) * np.cos(PI * X[:, 1])]),
        # f = -∇·(K∇p) = π² tr(K) sinπx sinπy - 2π² K12 cosπx cosπy
        f=lambda X: (PI ** 2 * tr * np.sin(PI * X[:, 0]) * np.sin(PI * X[:, 1])
                     - 2 * PI ** 2 * k12 * np.cos(PI * X[:, 0]) * np.cos(PI * X[:, 1])),
        gD=lambda P: np.zeros(len(P)),
        bc=lambda mid, n: 'D',
    )


def case_var():
    """算例 3：光滑变系数 K = (1+x) I。
       q = -(1+x)∇p,  f = -π cosπx sinπy + 2(1+x)π² sinπx sinπy。"""
    return dict(
        name=r"光滑变系数 $K=(1+x)I$",
        K=lambda X: 1.0 + X[:, 0],
        p=lambda X: np.sin(PI * X[:, 0]) * np.sin(PI * X[:, 1]),
        gradp=lambda X: PI * np.column_stack([np.cos(PI * X[:, 0]) * np.sin(PI * X[:, 1]),
                                              np.sin(PI * X[:, 0]) * np.cos(PI * X[:, 1])]),
        f=lambda X: (-PI * np.cos(PI * X[:, 0]) * np.sin(PI * X[:, 1])
                     + 2 * (1.0 + X[:, 0]) * PI ** 2
                     * np.sin(PI * X[:, 0]) * np.sin(PI * X[:, 1])),
        gD=lambda P: np.zeros(len(P)),
        bc=lambda mid, n: 'D',
    )


def case_layer(kL=1.0, kR=100.0):
    """算例 4：分层（串联）高对比介质 —— 精确解分段线性、通量为常向量。

    区域左半 x<0.5 渗透率 kL，右半 kR，上下不透水，左右定压 p(0)=1, p(1)=0，f=0。
    精确解：  q = (Q, 0)，  Q = 2/(1/kL + 1/kR)  （串联电阻 ⇒ 等效渗透率取**调和平均**）
             p = 1 - Q x/kL            (x < 1/2)
             p = p(1/2) - Q(x-1/2)/kR  (x > 1/2)

    ★ 为什么不用"光滑 p + 跳跃 K"当制造解？
      因为 ∇·(K∇p) = KΔp + [K]δ(x-1/2)∂p/∂x 会多出一个**界面的 Dirac 源**，
      这个 f 不是 L² 函数，用它检验收敛阶是错的（阶数会掉到 0）。
      正确做法是让精确解满足界面条件 [K ∂p/∂x] = 0（法向通量连续），
      下面的分段线性解正是这么构造的。
    """
    Q = 2.0 / (1.0 / kL + 1.0 / kR)
    ph = 1.0 - Q * 0.5 / kL

    def p(X):
        x = X[:, 0]
        return np.where(x < 0.5, 1.0 - Q * x / kL, ph - Q * (x - 0.5) / kR)

    def gradp(X):
        x = X[:, 0]
        return np.column_stack([np.where(x < 0.5, -Q / kL, -Q / kR), np.zeros(len(X))])

    return dict(
        name=rf"分层高对比 $\kappa_L={kL:g},\ \kappa_R={kR:g}$（精确解分段线性，$q$ 精确）",
        K=lambda X: np.where(X[:, 0] < 0.5, kL, kR),
        p=p, gradp=gradp,
        f=lambda X: np.zeros(len(X)),
        gD=p,
        bc=lambda mid, n: 'D' if (mid[0] < 1e-12 or mid[0] > 1.0 - 1e-12) else 'N',
        _Q=Q, _kL=kL, _kR=kR,
    )


# ---------- 物理算例 ----------
def case_column():
    """达西柱实验：左右给定压力 1 与 0，上下不透水。K = κ I。
       精确解 p = 1-x, q = (κ, 0)（常向量，RT0 能精确表示）。
       达西定律：出口流量 Q = κ·A·(p_in-p_out)/L = κ。"""
    kap = 2.5
    return dict(
        name=r"达西柱实验：$p(0,y)=1,\ p(1,y)=0$，上下不透水，$K=2.5I$",
        K=lambda X: np.full(len(X), kap),
        p=lambda X: 1.0 - X[:, 0],
        gradp=lambda X: np.column_stack([-np.ones(len(X)), np.zeros(len(X))]),
        f=lambda X: np.zeros(len(X)),
        gD=lambda P: 1.0 - P[:, 0],
        bc=lambda mid, n: 'D' if (mid[0] < 1e-12 or mid[0] > 1.0 - 1e-12) else 'N',
        _kap=kap,
    )


def case_fluxbc():
    """给定流量边界：左边给定总入流 1（q·n_out = -1），右边 p = 0，上下不透水，K = I。
       精确解 p = 1-x, q = (1,0)。专门检验本质流量边界条件的实现。"""
    return dict(
        name=r"给定流量边界：左侧 $q\cdot n=-1$，右侧 $p=0$，上下不透水，$K=I$",
        K=lambda X: np.ones(len(X)),
        p=lambda X: 1.0 - X[:, 0],
        gradp=lambda X: np.column_stack([-np.ones(len(X)), np.zeros(len(X))]),
        f=lambda X: np.zeros(len(X)),
        gD=lambda P: np.zeros(len(P)),
        bc=lambda mid, n: (('flux', lambda P: -np.ones(len(P)))
                           if mid[0] < 1e-12 else
                           ('D' if mid[0] > 1.0 - 1e-12 else 'N')),
    )


# =====================================================================
# 6. 单次求解
# =====================================================================
def run_one(n, case, direct=False, direct_only=False, verbose=False, ny=None):
    ny = n if ny is None else ny
    nodes, elems = mesh_rect(n, ny)
    edges, elem_edge, elem_sign, n_adj, boundary = build_edges(nodes, elems)
    dir_edges, known = boundary_data(nodes, boundary, case["bc"])
    A, B, F, rhs_q, ne = assemble(nodes, elems, elem_edge, elem_sign,
                                  dir_edges, case["f"], case["gD"], case["K"])
    A, B, F, rhs_q, free = restrict_system(A, B, F, rhs_q, known)
    if direct_only:
        qf, p = solve_mixed_direct(A, B, F, rhs_q)
    else:
        qf, p = solve_mixed(A, B, F, rhs_q)
    q = np.zeros(ne)
    for k, v in known.items():
        q[k] = v
    q[free] = qf
    if direct and not direct_only and len(elems) <= 2048:
        q2, p2 = solve_mixed_direct(A, B, F, rhs_q)
        dq = np.abs(q[free] - q2).max()
        dp = np.abs(p - p2).max()
        print(f"    [自检] Schur+CG 与直接解之差：max|Δq|={dq:.2e}, max|Δp|={dp:.2e}")
    e_p, e_q, e_div = compute_errors(nodes, elems, elem_edge, elem_sign, q, p, case, case["K"])
    cons = conservation_check(nodes, elems, elem_edge, elem_sign, q, case["f"])
    if verbose:
        print(f"    n={n:4d}  单元={len(elems):6d}  边={ne:6d}  "
              f"自由度={ne + len(elems):6d}  ||p-p_h||={e_p:.4e}  ||q-q_h||={e_q:.4e}")
    return dict(n=n, h=1.0 / n, e_p=e_p, e_q=e_q, e_div=e_div,
                e_hdiv=np.hypot(e_q, e_div), cons=cons,
                nodes=nodes, elems=elems, elem_edge=elem_edge, elem_sign=elem_sign,
                q=q, p=p, ne=ne, free=free, known=known, edges=edges)


# =====================================================================
# 7. 主程序
# =====================================================================
def main():
    out = []

    def say(s=""):
        print(s)
        out.append(str(s))

    ns = [4, 8, 16, 32, 64]
    cases = [("iso", case_iso()), ("aniso", case_aniso()),
             ("var", case_var()), ("layer", case_layer())]
    store, last = {}, {}

    say("=" * 108)
    say("二维 Darcy 方程  RT0-P0 混合有限元 收敛阶验证")
    say("        K^{-1}q + ∇p = 0,  ∇·q = f  in Ω;   p = g_D on ∂Ω（压力 Dirichlet，自然边界条件）")
    say("        V_h = RT0(每条边一个法向通量自由度),  Q_h = P0(每个单元一个压力自由度)")
    say("        A_ij = ∫ K^{-1}φ_i·φ_j dx ,  B_iK = ±1 ,  rhs = -<g_D, φ_i·n>")
    say("=" * 108)

    for key, case in cases:
        say("")
        say(f"【算例 {key}】{case['name']}")
        hs, var = [], {k: [] for k in ("e_p", "e_q", "e_div", "e_hdiv")}
        cons_worst = 0.0
        for n in ns:
            r = run_one(n, case, direct=(n == 8), direct_only=(key == "layer"),
                        verbose=True)
            hs.append(r["h"])
            for k in var:
                var[k].append(r[k])
            cons_worst = max(cons_worst, r["cons"])
            last[key] = r
        store[key] = dict(hs=hs, **var)
        say("")
        cols = [(r"$||p-p_h||_{L^2}$", var["e_p"]),
                (r"$||q-q_h||_{L^2}$", var["e_q"]),
                (r"$||\nabla\cdot(q-q_h)||_{L^2}$", var["e_div"]),
                (r"$||q-q_h||_{H(div)}$", var["e_hdiv"])]
        hdr = f"{'h':>8}"
        for nm, _ in cols:
            hdr += f" | {nm:>26} {'rate':>7}"
        say(hdr)
        say("-" * len(hdr))
        for i, h in enumerate(hs):
            row = f"{h:>8.4f}"
            for _, errs in cols:
                row += f" | {errs[i]:>26.4e}"
                if i < len(hs) - 1 and errs[i] > 1e-10 and errs[i + 1] > 1e-10:
                    p = np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1])
                    row += f" {p:>7.3f}"
                else:
                    row += f" {'--':>7}"
            say(row)
        say("-" * len(hdr))
        p_p, _ = fit_rate(hs, var["e_p"], use_last=3)
        say(f"   最细 3 级最小二乘拟合：  p 的 L² 阶 ≈ {p_p:.3f}")
        if var["e_q"][0] < 1e-10:
            say("   q 的误差是机零（精确解的通量落在 RT0 空间内，可精确表示，无收敛阶可言）")
        else:
            p_q, _ = fit_rate(hs, var["e_q"], use_last=3)
            say(f"   q 的 L² 阶 ≈ {p_q:.3f}")
        say(f"   [局部守恒残差] max_K |∇·q_h - f̄_K| = {cons_worst:.3e}"
            f"（离散守恒方程 Bᵀq = F 的残差，受线性求解器容差支配；约等于通量误差/单元面积）")

    # ---------- 与泊松版本 rt0_2d.py 交叉核对 ----------
    say("")
    say("=" * 108)
    say("【交叉核对】K = I 时 Darcy 问题就是泊松问题，误差应与 RT0收敛阶验证/结果_2d.txt 完全一致")
    say("=" * 108)
    say(f"{'h':>8} {'本程序 ||p-p_h||':>20} {'泊松程序 ||u-u_h||':>20} "
        f"{'本程序 ||q-q_h||':>20} {'泊松程序 ||q-q_h||':>20}")
    ref = [(0.25, 1.2868e-01, 5.0190e-01), (0.125, 6.5174e-02, 2.5164e-01),
           (0.0625, 3.2690e-02, 1.2589e-01), (0.0312, 1.6358e-02, 6.2954e-02),
           (0.0156, 8.1807e-03, 3.1478e-02)]
    for i, (h, eu, eq) in enumerate(ref):
        say(f"{h:>8.4f} {store['iso']['e_p'][i]:>20.4e} {eu:>20.4e} "
            f"{store['iso']['e_q'][i]:>20.4e} {eq:>20.4e}")

    # ---------- 物理算例：达西柱实验 ----------
    say("")
    say("=" * 108)
    say("【物理算例 1】达西柱实验（压力驱动、上下不透水）—— 检查达西定律 Q = K·A·Δp/L")
    say("=" * 108)
    c = case_column()
    r = run_one(16, c, direct=True, verbose=True)
    kap = c["_kap"]
    q, edges = r["q"], r["edges"]
    out_flow = 0.0
    for k, (u0, u1) in enumerate(edges):
        mid = 0.5 * (r["nodes"][u0] + r["nodes"][u1])
        if abs(mid[0] - 1.0) < 1e-12:                 # x=1 出口边，外法向 (1,0)
            d = r["nodes"][u1] - r["nodes"][u0]
            n_g = np.array([d[1], -d[0]]) / np.hypot(d[0], d[1])
            out_flow += q[k] * float(np.dot(np.array([1.0, 0.0]), n_g))
    say(f"   数值出口流量 Q_h = {out_flow:.10f} ；  达西定律理论值 K·A·Δp/L = "
        f"{kap:.10f}  （相对误差 {abs(out_flow-kap)/kap:.2e}）")
    say(f"   通量误差 ||q-q_h||_L2 = {r['e_q']:.3e}（常向量场落在 RT0 空间内，应为机器零）")
    say(f"   压力误差 ||p-p_h||_L2 = {r['e_p']:.3e}（P0 只能分片常数逼近线性压力 ⇒ O(h)）")

    # ---------- 物理算例：给定流量边界 ----------
    say("")
    say("=" * 108)
    say("【物理算例 2】给定流量边界（左侧入流、右侧定压、上下不透水）—— 检查本质流量边界条件")
    say("=" * 108)
    c2 = case_fluxbc()
    r2 = run_one(16, c2, direct=True, verbose=True)
    E2 = r2["edges"]
    inflow = 0.0
    for k, (u0, u1) in enumerate(E2):
        m = 0.5 * (r2["nodes"][u0] + r2["nodes"][u1])
        if m[0] < 1e-12:                                     # 左侧边界，外法向 (-1,0)
            d = r2["nodes"][u1] - r2["nodes"][u0]
            n_g = np.array([d[1], -d[0]]) / np.hypot(d[0], d[1])
            inflow += r2["q"][k] / float(np.round(np.dot(np.array([-1.0, 0.0]), n_g)))
    say(f"   数值左边界总流量 ∫q·n ds = {inflow:.10f}（给定值 -1.0，"
        f"相对误差 {abs(inflow+1.0):.2e}）")
    say(f"   通量误差 ||q-q_h||_L2 = {r2['e_q']:.3e}（精确解 q=(1,0) 是常向量 ⇒ 机零）；"
        f"压力误差 ||p-p_h||_L2 = {r2['e_p']:.3e}")
    say(f"   局部质量守恒 max_K|∇·q_h - f̄_K| = {r2['cons']:.3e}")

    # ---------- 分层介质：界面法向通量连续性与调和平均等效渗透率 ----------
    say("")
    say("=" * 108)
    say("【诊断】分层高对比介质：界面 x=0.5 上的法向通量、串联等效渗透率（调和平均）")
    say("=" * 108)
    c4 = case_layer()
    r4 = run_one(16, c4, direct=True, verbose=False)
    E = r4["edges"]
    mid = 0.5 * (r4["nodes"][E[:, 0]] + r4["nodes"][E[:, 1]])
    d = r4["nodes"][E[:, 1]] - r4["nodes"][E[:, 0]]
    Lall = np.linalg.norm(d, axis=1)
    n_g = np.column_stack([d[:, 1], -d[:, 0]]) / Lall[:, None]
    sel = np.abs(mid[:, 0] - 0.5) < 1e-12
    # 注意：RT0 自由度 q_k = ∫_e q·n ds 是**边上法向通量的积分**，除以边长才是通量密度
    flux = r4["q"][sel] * n_g[sel, 0] / Lall[sel]
    Q = c4["_Q"]
    say(f"   界面边数 = {int(sel.sum())}；数值法向通量密度 min={flux.min():.10f}, "
        f"max={flux.max():.10f}（跨界面由同一个边自由度给出 ⇒ 连续）")
    say(f"   串联等效通量 Q = 2/(1/κ_L+1/κ_R) = {Q:.10f}，数值 = {flux.mean():.10f}，"
        f"相对误差 {abs(flux.mean()-Q)/Q:.2e}")
    say(f"   对照：算术平均 {(c4['_kL']+c4['_kR'])/2:.4f}（错），"
        f"调和平均 {2/(1/c4['_kL']+1/c4['_kR']):.4f}（对）")
    say(f"   本算例误差（n=16，直接解鞍点系统）：||p-p_h||={r4['e_p']:.3e}"
        f"（分段线性压力的 P0 逼近 ⇒ O(h)），||q-q_h||={r4['e_q']:.3e}"
        f"（常向量场 ⊂ RT0 ⇒ 机器零）")

    # ---------- 收敛图 ----------
    hs = [1.0 / n for n in ns]
    series = []
    for key in ("iso", "aniso", "var"):
        series += [(f"{key}_p", np.array(store[key]["e_p"])),
                   (f"{key}_q", np.array(store[key]["e_q"]))]
    series += [("layer_p", np.array(store["layer"]["e_p"]))]     # 分层算例 q 精确，不便画 log 图
    write_csv("data_darcy.csv", hs, series, ref_order=1)
    labs = {"iso": r"$K=I$", "aniso": "常值各向异性", "var": "光滑变系数",
            "layer": "高对比分层（仅 $p$）"}
    write_pgfplots("fig_darcy.tex", "data_darcy.csv",
                   [(f"{k}_p", rf"{labs[k]}: $\|p-p_h\|_{{L^2}}$", None)
                    for k in ("iso", "aniso", "var", "layer")]
                   + [(f"{k}_q", rf"{labs[k]}: $\|q-q_h\|_{{L^2}}$", None)
                      for k in ("iso", "aniso", "var")],
                   r"二维 Darcy 方程 RT0-P0 混合元收敛阶", ref_order=1)

    # ---------- 误差可视化（Pillow 出图，不依赖 matplotlib） ----------
    say("")
    say("=" * 108)
    say("【误差可视化】逐单元误差场 + 通量箭头 + 剖面 + 收敛曲线")
    say("=" * 108)
    if dp is None:
        say(f"   [跳过] 未能导入 darcy_plot（{_DP_ERR}）")
    else:
        os.makedirs(FIGDIR, exist_ok=True)

        # ① 各向异性：误差的空间分布 + q_h vs q
        cs = case_aniso()
        n = 32
        r = run_one(n, cs, verbose=False)
        nodes, elems = r["nodes"], r["elems"]
        ctr = nodes[elems].mean(axis=1)
        qh = flux_at_centroids(nodes, elems, r["elem_edge"], r["elem_sign"], r["q"])
        qe = -K_apply(cs["K"], ctr, cs["gradp"](ctr))
        ep, eq = cell_errors(nodes, elems, r["elem_edge"], r["elem_sign"], r["q"],
                             r["p"], cs, cs["K"])
        f1 = dp.cell_heatmap(nodes, elems, ep, os.path.join(FIGDIR, "err_aniso_p.png"),
                             title="压力误差 |p-p_h|（单元 L2，各向异性 K）",
                             cbar_label="L2/单元", cmap="coolwarm", width=520)
        f2 = dp.cell_heatmap(nodes, elems, eq, os.path.join(FIGDIR, "err_aniso_q.png"),
                             title="通量误差 |q-q_h|（单元 L2）", cbar_label="L2/单元",
                             cmap="coolwarm", log=True, width=520)
        f3 = dp.mesh_quiver(nodes, elems, ctr, qh, os.path.join(FIGDIR, "flux_aniso.png"),
                            title="q_h（蓝）与精确 q（红）", width=520,
                            vectors_extra=qe, label="RT0 数值解", label_extra="精确解",
                            stride=max(1, int(round(np.sqrt(len(elems)) / 26))))
        dp.hstack([f1, f2, f3], os.path.join(FIGDIR, "err_aniso.png"),
                  labels=["(a) 压力误差", "(b) 通量误差", "(c) 通量对比"])
        say(f"   各向异性 n=32：单元压力误差 {ep.min():.3e}~{ep.max():.3e}，"
            f"通量误差 {eq.min():.3e}~{eq.max():.3e}（min/max 比 "
            f"{eq.max()/max(eq.min(),1e-300):.0f}，误差在区域上明显不均匀）")

        # ② 分层高对比：渗透率场 + 压力误差 + 剖面（精确解有折点，P0 是阶梯）
        cs = case_layer()
        r = run_one(32, cs, direct_only=True, verbose=False)
        nodes, elems = r["nodes"], r["elems"]
        kappa = np.asarray(cs["K"](nodes[elems].mean(axis=1)), dtype=float).ravel()
        ep, eq = cell_errors(nodes, elems, r["elem_edge"], r["elem_sign"], r["q"],
                             r["p"], cs, cs["K"])
        g1 = dp.cell_heatmap(nodes, elems, kappa, os.path.join(FIGDIR, "layer_k.png"),
                             title="渗透率 κ（左 1 / 右 100）", cbar_label="κ",
                             cmap="viridis", show_mesh=False, width=520)
        g2 = dp.cell_heatmap(nodes, elems, ep, os.path.join(FIGDIR, "layer_p.png"),
                             title="压力误差 |p-p_h|（界面上有拐点）", cbar_label="L2/单元",
                             cmap="coolwarm", width=520)
        px = np.linspace(0.0, 1.0, 401)
        py = cs["p"](np.column_stack([px, np.full_like(px, 0.5)]))
        sx, sy = dp.p0_step_along_line(nodes, elems, r["p"], 0.5, 1.0 / 32)
        g3 = dp.profile_plot(
            [dict(label="精确解 p（分段线性，x=0.5 处有折点）", x=px, y=py,
                  color=(214, 39, 40)),
             dict(label="RT0-P0 数值解 p_h（分片常数）", x=sx, y=sy,
                  color=(31, 119, 180), dash=True)],
            os.path.join(FIGDIR, "layer_profile.png"),
            title="沿 y=0.5 的压力剖面（左半渗透率小 ⇒ 斜率大 ⇒ P0 误差大）",
            xlabel="x", ylabel="p", vlines=[0.5], width=520, height=560,
            legend_pos="tr")
        dp.hstack([g1, g2, g3], os.path.join(FIGDIR, "err_layer.png"),
                  labels=["(a) 渗透率场", "(b) 压力误差", "(c) 剖面 p(x,0.5)"])
        say(f"   分层 n=32：通量误差 max={eq.max():.2e}（机零，常通量场精确可表示）；"
            f"压力误差 {ep.min():.3e}~{ep.max():.3e}（左半斜率大 ⇒ 误差大）")

        # ③ 收敛曲线（把上面表格里的数据画出来）
        hs_ = [1.0 / n for n in ns]
        f4 = dp.loglog_convergence(
            hs_,
            [dict(label="K=I：压力 L2", h=hs_, e=store["iso"]["e_p"],
                  color=(31, 119, 180)),
             dict(label="K=I：通量 L2", h=hs_, e=store["iso"]["e_q"],
                  color=(31, 119, 180), dash=True),
             dict(label="各向异性：压力 L2", h=hs_, e=store["aniso"]["e_p"],
                  color=(214, 39, 40)),
             dict(label="各向异性：通量 L2", h=hs_, e=store["aniso"]["e_q"],
                  color=(214, 39, 40), dash=True),
             dict(label="光滑变系数：压力 L2", h=hs_, e=store["var"]["e_p"],
                  color=(44, 160, 44)),
             dict(label="高对比分层：压力 L2", h=hs_, e=store["layer"]["e_p"],
                  color=(148, 103, 189))],
            os.path.join(FIGDIR, "conv_2d.png"),
            title="二维 Darcy：RT0-P0 误差随 h 的下降（虚线 = 通量，参考斜率 1）",
            ref_slope=1.0, width=900, height=600)
        say("   收敛曲线（2D，4 个算例）已保存；图片目录 figs/standalone/")
        for f in sorted(os.listdir(FIGDIR)):
            if f.lower().endswith(".png"):
                say(f"      {os.path.relpath(os.path.join(FIGDIR, f), _HERE).replace(os.sep, '/')}")

    with open(os.path.join(_HERE, "结果_darcy.txt"), "w", encoding="utf-8-sig") as fp:
        fp.write("\n".join(out) + "\n")
    print("\n[结果] 已保存 结果_darcy.txt")


if __name__ == "__main__":
    main()
