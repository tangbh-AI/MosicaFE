"""虚拟元空间 (Virtual Element Method, 任意阶 ``k>=1``)。

2D/3D 支持任意阶 ``k>=1``。高阶采用增强协调空间；三维面迹递归使用二维 VEM。

虚拟元与有限元最大的不同：**基函数没有显式表达式**，只有

1. 每个顶点一个自由度；
2. 单元上可计算的量是"投影到一次多项式"的算子
   :math:`\\Pi^{\\nabla}`；
3. 因此单元双线性型拆成**一致性项 + 稳定化项**

.. math::

   K_e = \\underbrace{\\big(G\\Pi\\big)^{\\!T}\\!\\big(G\\Pi\\big)|K|}_{A_{\\rm proj}}
       + \\underbrace{\\sigma\\,(I-\\Pi_{\\rm dof})^{\\!T}(I-\\Pi_{\\rm dof})}_{S_{\\rm stab}}

这套实现适用于**任意**多边形（2D）与多面体（3D）单元：三角形、四边形、
五边形、Voronoi 多边形、四面体、六面体都是它的特例。

与早期脚本的差别（见 progress/DECISIONS.md D6）：

* 稳定化项统一取**对称半正定**形式，因此单元刚度矩阵对称正定，可以放心用 CG；
* 缩放取 :math:`\\sigma=\\mathrm{tr}(A_{\\rm proj})/\\mathrm{tr}(S)`，
  使两部分量级相当（量纲也自动对齐）；
* 传 ``stabilization="legacy"`` 可复现旧脚本的 :math:`(I-\\Pi_{\\rm dof})/n` 形式。
"""

from __future__ import annotations

from math import comb

import numpy as np

from ..core.utils import (
    HEX_FACES,
    TET_FACES,
    cell_centroid,
    cell_diameter,
    cell_measure,
    edge_normals_2d,
    simplex_jacobian,
)
from .base import BaseFESpace
from .vem_high import HighOrderVEMMixin

__all__ = ["VEMSpace"]

_EPS = 1e-14


# ---------------------------------------------------------------------------
# k>=2 高阶虚拟元的多项式工具：缩放单项式、精确单元矩、边迹基
# ---------------------------------------------------------------------------
def _monomial_indices(dim, degree):
    """总次数 ``<= degree`` 的多重指标列表（常数项在前，字典序）。"""
    degree = int(degree)
    if dim == 1:
        return [(i,) for i in range(degree + 1)]
    if dim == 2:
        out = []
        for total in range(degree + 1):
            for a in range(total + 1):
                out.append((a, total - a))
        return out
    out = []
    for total in range(degree + 1):
        for a in range(total + 1):
            for b in range(total - a + 1):
                out.append((a, b, total - a - b))
    return out


def _scaled_monomial_values(multi, points, centroid, h):
    """缩放单项式 ``((x-c)/h)^a`` 在点处的值，返回 ``(n_points, n_k)``。"""
    pts = (np.asarray(points, dtype=float) - centroid) / h
    vals = np.ones((pts.shape[0], len(multi)))
    for j, idx in enumerate(multi):
        for d, p in enumerate(idx):
            vals[:, j] *= pts[:, d] ** p
    return vals


def _scaled_monomial_grad(multi, points, centroid, h):
    """缩放单项式的梯度，返回 ``(n_points, n_k, dim)``。"""
    pts = (np.asarray(points, dtype=float) - centroid) / h
    npts, dim = pts.shape
    grad = np.zeros((npts, len(multi), dim))
    for j, idx in enumerate(multi):
        for d in range(dim):
            if idx[d] == 0:
                continue
            col = np.full(npts, idx[d] / h)
            for e in range(dim):
                p = idx[e] - (1 if e == d else 0)
                col *= pts[:, e] ** p
            grad[:, j, d] = col
    return grad


def _edge_poly_integral(u, du, v, dv, a, b):
    """精确计算 ``∫_0^1 (u + t·du)^a (v + t·dv)^b dt``（二项式展开）。"""
    total = 0.0
    for r in range(a + 1):
        term = comb(a, r) * (u ** (a - r)) * (du ** r)
        for s in range(b + 1):
            total += term * comb(b, s) * (v ** (b - s)) * (dv ** s) / (r + s + 1)
    return total


def _monomial_moments_2d(coords, exponents, centroid, h):
    """精确计算 ``∫_K ((x-c)/h)^e1 ((y-c)/h)^e2 dx dy``。

    由 ``∫_K (x-c)^e1 (y-c)^e2 = (1/(e1+1)) ∮ (x-c)^(e1+1) (y-c)^e2 dy``
    把面积分降到边界，再用二项式展开精确积分每条边。
    """
    cx, cy = centroid
    n = len(coords)
    out = np.zeros(len(exponents))
    for t, (e1, e2) in enumerate(exponents):
        total = 0.0
        for i in range(n):
            xa, ya = coords[i]
            xb, yb = coords[(i + 1) % n]
            total += (yb - ya) * _edge_poly_integral(
                xa - cx, xb - xa, ya - cy, yb - ya, e1 + 1, e2
            )
        out[t] = total / (e1 + 1) / (h ** (e1 + e2))
    return out


def _lagrange_1d_values(k, s):
    """``[0,1]`` 上等距节点 ``j/k`` 的 1 维 Lagrange 基在 ``s`` 处的值。

    返回 ``(n_s, k+1)``，第 ``m`` 列对应节点 ``m/k``。
    """
    s = np.asarray(s, dtype=float).ravel()
    nodes = np.arange(k + 1) / k
    out = np.zeros((len(s), k + 1))
    for m in range(k + 1):
        num = np.ones_like(s)
        denom = 1.0
        for j in range(k + 1):
            if j == m:
                continue
            num *= s - nodes[j]
            denom *= nodes[m] - nodes[j]
        out[:, m] = num / denom
    return out


def _duffy_triangle(order):
    """Duffy 变换构造**标准参考三角形** ``{u>=0, v>=0, u+v<=1}``（面积 1/2）
    上精确到 ``order`` 次总次数的求积规则。

    用 ``(u,v) = (xi*(1-eta), xi*eta)``（雅可比行列式 ``xi``）把 ``[0,1]^2``
    映到顶点为 ``(0,0),(1,0),(0,1)`` 的标准参考三角形——与
    :func:`~MosicaFE.core.utils.simplex_jacobian` 的仿射映射约定一致。
    """
    n = max(1, (int(order) + 3) // 2)
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    Xi, Eta = np.meshgrid(x, x, indexing="ij")
    xi = Xi.ravel()
    eta = Eta.ravel()
    u = xi * (1.0 - eta)
    v = xi * eta
    wts = (w[:, None] * w[None, :]).ravel() * xi
    return np.column_stack([u, v]), wts


class VEMSpace(HighOrderVEMMixin, BaseFESpace):
    """协调虚拟元：顶点、边节点、面矩（3D）、体矩，任意 degree>=1。"""

    family = "vem"
    has_explicit_basis = False
    _STABILIZATIONS = ("trace", "legacy", "none")

    def __init__(
        self,
        mesh,
        degree: int = 1,
        stabilization: str = "trace",
        name: str = None,
    ):
        degree = int(degree)
        if degree < 1:
            raise ValueError(f"虚拟元阶数 degree 需 >= 1，收到 {degree}")
        if stabilization not in self._STABILIZATIONS:
            raise ValueError(
                f"stabilization 需为 {self._STABILIZATIONS} 之一，收到 {stabilization!r}"
            )
        self.stabilization = stabilization
        super().__init__(mesh, degree, name or f"VEM-P{degree}")
        self._proj_cache = {}

    # ------------------------------------------------------------------
    # 自由度：每个网格顶点一个
    # ------------------------------------------------------------------
    def _build_dof_map(self):
        if self.degree == 1:
            self.n_dofs = self.mesh.n_nodes
            self.dof_map = [np.asarray(e, dtype=np.int64) for e in self.mesh.elements]
            self.dof_support = [(v,) for v in range(self.mesh.n_nodes)]
            self.dof_is_interior = np.zeros(self.n_dofs, dtype=bool)
            return
        self._build_dof_map_high()

    def _build_dof_map_high(self):
        """k>=2 的自由度：顶点 + 每条边 (k-1) 个内部点（全局共享）+ 内部矩。"""
        if self.mesh.dim == 3:
            return self._build_dof_map_3d()
        k = self.degree
        nodes = self.mesh.nodes
        book = {}
        supports = []
        interiors = []
        coords = []
        dof_map = []

        def reg(key, support, interior, coord):
            if key in book:
                return book[key]
            idx = len(supports)
            book[key] = idx
            supports.append(tuple(support))
            interiors.append(bool(interior))
            coords.append(np.asarray(coord, dtype=float))
            return idx

        n_mom = k * (k - 1) // 2  # C(k, 2)
        for c, cell in enumerate(self.mesh.elements):
            n = len(cell)
            local = []
            for v in cell:
                v = int(v)
                local.append(reg(("v", v), (v,), False, nodes[v]))
            for i in range(n):
                a = int(cell[i])
                b = int(cell[(i + 1) % n])
                lo, hi = (a, b) if a < b else (b, a)
                orient = 1 if a < b else -1
                for j in range(1, k):
                    gj = j if orient > 0 else k - j
                    t = j / k
                    local.append(reg(("e", lo, hi, gj), (lo, hi), False,
                                     (1.0 - t) * nodes[a] + t * nodes[b]))
            centroid = cell_centroid(self.mesh.get_cell_vertices(c))
            for j in range(n_mom):
                local.append(reg(("c", c, j), (), True, centroid))
            dof_map.append(np.asarray(local, dtype=np.int64))

        self.dof_map = dof_map
        self.n_dofs = len(supports)
        self.dof_support = supports
        self.dof_is_interior = np.asarray(interiors, dtype=bool)
        self.dof_coords = np.asarray(coords, dtype=float)

    def _build_dof_coordinates(self):
        if self.degree == 1:
            self.dof_coords = np.array(self.mesh.nodes, dtype=float, copy=True)
        # k>=2：dof_coords 已在 _build_dof_map_high 中填充

    def reference_nodes(self, cell_idx) -> np.ndarray:
        if self.degree == 1:
            return np.zeros((len(self.mesh.elements[cell_idx]), self.mesh.dim))
        n_local = len(self.dof_map[cell_idx])
        return np.zeros((n_local, self.mesh.dim))

    def reference_element_type(self) -> str:
        return "polygon" if self.mesh.dim == 2 else "polyhedron"

    def get_quadrature(self, order: int = None):
        raise NotImplementedError(
            "虚拟元没有参考单元求积规则；误差范数请在 postprocess 中按单元几何计算。"
        )

    # ------------------------------------------------------------------
    # 投影算子
    # ------------------------------------------------------------------
    def _cell_faces(self, cell_idx):
        """返回 ``[(local_indices, outward_normal, area), ...]``。"""
        coords = self.mesh.get_cell_vertices(cell_idx)
        d = self.mesh.dim
        if d == 2:
            normals, lengths = edge_normals_2d(coords)
            n = len(coords)
            return [
                ([i, (i + 1) % n], normals[i], lengths[i]) for i in range(n)
            ]

        n = len(coords)
        if n == 4:
            faces = TET_FACES
        elif n == 8:
            faces = HEX_FACES
        elif self.mesh._faces is not None:
            faces = self.mesh._faces
        else:
            raise ValueError(
                "3D 虚拟元需要单元的面向量；请通过 Mesh(..., faces=...) 提供，"
                "或使用四面体 / 六面体网格。"
            )

        centroid = np.mean(coords, axis=0)
        out = []
        for f in faces:
            face_coords = coords[list(f)]
            c_face = face_coords.mean(axis=0)
            if len(f) == 3:
                raw = np.cross(face_coords[1] - face_coords[0],
                               face_coords[2] - face_coords[0])
                area = 0.5 * np.linalg.norm(raw)
            else:
                raw = np.zeros(3)
                area = 0.0
                tris = [(0, i, i+1) for i in range(1, len(f)-1)]
                for (a, b, c) in tris:
                    nrm = np.cross(face_coords[b] - face_coords[a],
                                   face_coords[c] - face_coords[a])
                    raw = raw + nrm
                    area += 0.5 * np.linalg.norm(nrm)
            if np.linalg.norm(raw) < _EPS or area < _EPS:
                continue
            normal = raw / np.linalg.norm(raw)
            if np.dot(normal, c_face - centroid) < 0:
                normal = -normal
            out.append((list(f), normal, area))
        return out

    def _build_projection(self, cell_idx):
        """返回 ``(G, Pi, measure, coords, centroid)``。"""
        hit = self._proj_cache.get(cell_idx)
        if hit is not None:
            return hit

        coords = self.mesh.get_cell_vertices(cell_idx)
        n = len(coords)
        d = self.mesh.dim
        centroid = cell_centroid(coords)
        measure = cell_measure(coords)

        # 缩放单项式基 m0 = 1, m_{k+1} = x_k - c_k
        G = np.zeros((d, d + 1))
        for k in range(d):
            G[k, k + 1] = 1.0
        M = measure * (G.T @ G)

        rhs = np.zeros((d + 1, n))
        for (idx, normal, area) in self._cell_faces(cell_idx):
            for alpha in range(d + 1):
                dn = float(np.dot(G[:, alpha], normal))
                if abs(dn) < _EPS:
                    continue
                if len(idx) == 2:
                    # 边：梯形法则把 |e| dn 平均分给两个端点
                    rhs[alpha, idx[0]] += 0.5 * area * dn
                    rhs[alpha, idx[1]] += 0.5 * area * dn
                else:
                    # 面：三角形按 1/3 分给三个顶点；四边形拆成两个三角形
                    if len(idx) == 3:
                        share = area / 3.0 * dn
                        for v in idx:
                            rhs[alpha, v] += share
                    else:
                        fc = coords[idx]
                        for (a, b, c) in ((0, 1, 2), (0, 2, 3)):
                            tri_area = 0.5 * np.linalg.norm(
                                np.cross(fc[b] - fc[a], fc[c] - fc[a])
                            )
                            share = tri_area / 3.0 * dn
                            for v in (idx[a], idx[b], idx[c]):
                                rhs[alpha, v] += share

        Pi = np.linalg.pinv(M) @ rhs
        # ---- 常数模式的规范化 ----
        # 梯度部分是精确的：Pi[1:, :] 给出的就是拟合梯度。
        # 常数模式必须选成"让投影对线性自由度向量恒等"，否则稳定化项
        # S = (I-Pi_dof)^T (I-Pi_dof) 会惩罚线性函数，破坏相容性
        # （在不规则多边形上表现为收敛停滞；见 progress/VERIFICATION.md）。
        # 由 p(c) = mean(d) - grad(p)·(Vbar - c) 得到
        vbar = coords.mean(axis=0)
        Pi[0, :] = 1.0 / n - (vbar - centroid) @ Pi[1:, :]

        out = (G, Pi, measure, coords, centroid)
        self._proj_cache[cell_idx] = out
        return out

    # ------------------------------------------------------------------
    # k>=2：Π^∇ 投影与单元几何（2D）
    # ------------------------------------------------------------------
    def _projection_high(self, cell_idx):
        """构造缩放单项式基上的 Π^∇ 投影矩阵 Pi 及相关矩阵（k>=2）。

        返回 dict：``multi, mom_multi, centroid, h, measure, coords, B, H, D, Pi``，
        其中

        * ``B = ∫ ∇m_α·∇m_β``、``H = ∫ m_α m_β``（精确单元矩）；
        * ``D[i,α] = dof_i(m_α)``；
        * ``Pi = B⁺ R``，``R[α,i] = ∫_K ∇φ_i·∇m_α``（边界项 − 内部项），
          常数行按 ``Pi·D = I``（多项式再生）定标。
        """
        if self.mesh.dim == 3:
            return self._projection_3d(cell_idx)
        hit = self._proj_cache.get(cell_idx)
        if hit is not None:
            return hit

        k = self.degree
        coords = self.mesh.get_cell_vertices(cell_idx)
        n = len(coords)
        centroid = cell_centroid(coords)
        h = cell_diameter(coords)
        measure = cell_measure(coords)

        multi = _monomial_indices(2, k)          # 次数 <= k
        n_k = len(multi)
        mom_multi = _monomial_indices(2, k - 2)  # 内部矩基，次数 <= k-2
        n_mom = len(mom_multi)
        mom_idx = {e: j for j, e in enumerate(mom_multi)}

        exp_all = _monomial_indices(2, 2 * k)
        moments = _monomial_moments_2d(coords, exp_all, centroid, h)
        mom = dict(zip(exp_all, moments))

        B = np.zeros((n_k, n_k))
        H = np.zeros((n_k, n_k))
        for ai, (a1, a2) in enumerate(multi):
            for bi, (b1, b2) in enumerate(multi):
                B[ai, bi] = (
                    (a1 * b1 / h ** 2) * mom.get((a1 + b1 - 2, a2 + b2), 0.0)
                    + (a2 * b2 / h ** 2) * mom.get((a1 + b1, a2 + b2 - 2), 0.0)
                )
                H[ai, bi] = mom.get((a1 + b1, a2 + b2), 0.0)
        B = 0.5 * (B + B.T)
        H = 0.5 * (H + H.T)

        # D：自由度作用在单项式上
        n_local = n * k + n_mom
        D = np.zeros((n_local, n_k))
        D[:n, :] = _scaled_monomial_values(multi, coords, centroid, h)
        row = n
        for i in range(n):
            a = coords[i]
            b = coords[(i + 1) % n]
            for j in range(1, k):
                pt = ((1.0 - j / k) * a + (j / k) * b)[None, :]
                D[row, :] = _scaled_monomial_values(multi, pt, centroid, h)[0]
                row += 1
        for j, (b1, b2) in enumerate(mom_multi):
            for ai, (a1, a2) in enumerate(multi):
                D[n + n * (k - 1) + j, ai] = mom.get((a1 + b1, a2 + b2), 0.0) / measure

        # R = ∫_K ∇φ_i·∇m_α = 边界项 − 内部项
        R = np.zeros((n_k, n_local))
        normals, lengths = edge_normals_2d(coords)
        ng = k + 1
        xg, wg = np.polynomial.legendre.leggauss(ng)
        xg = 0.5 * (xg + 1.0)
        wg = 0.5 * wg
        L = _lagrange_1d_values(k, xg)              # (ng, k+1)
        for i in range(n):
            a = coords[i]
            b = coords[(i + 1) % n]
            pts = a[None, :] + xg[:, None] * (b - a)[None, :]
            grad = _scaled_monomial_grad(multi, pts, centroid, h)   # (ng, n_k, 2)
            dmn = grad @ normals[i]                                  # (ng, n_k)
            wL = wg * lengths[i]
            R[:, i] += dmn.T @ (wL * L[:, 0])
            R[:, (i + 1) % n] += dmn.T @ (wL * L[:, k])
            for j in range(1, k):
                R[:, n + i * (k - 1) + (j - 1)] += dmn.T @ (wL * L[:, j])
        # 内部项：∫_K φ_i Δm_α（只有内部矩自由度非零）
        # 对偶基 φ_j 满足 ∫_K φ_j q_l = |K| δ_{jl}，且 Δm_α 恰为缩放单项式组合
        #   Δm_α = a1(a1-1)/h²·q_{(a1-2,a2)} + a2(a2-1)/h²·q_{(a1,a2-2)}
        # 故 ∫_K φ_j Δm_α = |K|·[ a1(a1-1)/h² δ_{j,(a1-2,a2)} + a2(a2-1)/h² δ_{j,(a1,a2-2)} ]
        for ai, (a1, a2) in enumerate(multi):
            if a1 >= 2:
                j = mom_idx.get((a1 - 2, a2))
                if j is not None:
                    R[ai, n + n * (k - 1) + j] -= measure * (a1 * (a1 - 1) / h ** 2)
            if a2 >= 2:
                j = mom_idx.get((a1, a2 - 2))
                if j is not None:
                    R[ai, n + n * (k - 1) + j] -= measure * (a2 * (a2 - 1) / h ** 2)

        # B 在常数方向奇异；最小范数解把常数行置零，再按多项式再生修正。
        Pi = np.linalg.lstsq(B, R, rcond=None)[0]
        start = n * k
        # Fix the constant by the known cell mean, not an arbitrary left inverse.
        mean = np.zeros(n_local)
        mean[start] = 1.0
        Pi[0, :] = mean - (H[0, 1:] / measure) @ Pi[1:, :]

        out = dict(multi=multi, mom_multi=mom_multi, centroid=centroid, h=h,
                   measure=measure, coords=coords, B=B, H=H, D=D, Pi=Pi,
                   moment_start=start)
        self._complete_l2_projection(out)
        self._proj_cache[cell_idx] = out
        return out

    def _fan_quadrature(self, coords, order):
        """多边形从形心扇形剖分后数值积分，返回物理 ``(points, weights)``。"""
        tri_pts, tri_w = _duffy_triangle(order)
        centroid = cell_centroid(coords)
        pts = []
        wts = []
        for i in range(len(coords)):
            tri = np.array([centroid, coords[i], coords[(i + 1) % len(coords)]])
            J, detJ = simplex_jacobian(tri)
            pts.append(tri[0] + tri_pts @ J.T)
            wts.append(tri_w * abs(detJ))
        return np.vstack(pts), np.concatenate(wts)

    def _monomial_rhs_load(self, cell_idx, f, p, order):
        """计算 ``∫_K f m_β``（β 遍历单项式基），供投影载荷向量使用。"""
        pts, w = self._high_quadrature(cell_idx, order)
        fv = self._evaluate_function(f, pts)
        mono = _scaled_monomial_values(p["multi"], pts, p["centroid"], p["h"])
        return (mono * (w * fv)[:, None]).sum(axis=0)

    # ------------------------------------------------------------------
    # 单元矩阵
    # ------------------------------------------------------------------
    def local_stiffness(self, cell_idx) -> np.ndarray:
        """单元刚度矩阵 ``A_proj + S_stab``。"""
        if self.degree == 1:
            G, Pi, measure, coords, centroid = self._build_projection(cell_idx)
            n = len(coords)
            d = self.mesh.dim

            grad_proj = G @ Pi                       # (d, n)
            A_proj = measure * (grad_proj.T @ grad_proj)

            if self.stabilization == "none":
                return A_proj

            if self.stabilization == "legacy":
                D = np.zeros((n, d + 1))
                D[:, 0] = 1.0
                D[:, 1:] = coords - centroid
                Pi_dof = D @ Pi
                return A_proj + (np.eye(n) - Pi_dof) / n

            # 默认：对称半正定的 trace 缩放
            D = np.zeros((n, d + 1))
            D[:, 0] = 1.0
            D[:, 1:] = coords - centroid
            Pi_dof = D @ Pi
            diff = np.eye(n) - Pi_dof
            S = diff.T @ diff
            tr_A = float(np.trace(A_proj))
            tr_S = float(np.trace(S))
            if tr_S < _EPS:
                return A_proj
            sigma = tr_A / tr_S if tr_A > _EPS else 1.0 / measure
            return A_proj + sigma * S

        p = self._projection_high(cell_idx)
        Pi, B, D = p["Pi"], p["B"], p["D"]
        n_local = D.shape[0]
        A_proj = Pi.T @ B @ Pi
        A_proj = 0.5 * (A_proj + A_proj.T)

        if self.stabilization == "none":
            return A_proj

        Pi_dof = D @ Pi
        if self.stabilization == "legacy":
            return A_proj + (np.eye(n_local) - Pi_dof) / n_local

        diff = np.eye(n_local) - Pi_dof
        S = diff.T @ diff
        tr_A = float(np.trace(A_proj))
        tr_S = float(np.trace(S))
        if tr_S < _EPS:
            return A_proj
        sigma = tr_A / tr_S if tr_A > _EPS else 1.0 / p["measure"]
        return A_proj + sigma * S

    # -- 载荷 / 质量：用"扇形线性插值"把虚拟基函数几何化 -----------------
    def _face_index_triangles(self, cell_idx):
        """单元表面按三角形展开，返回局部顶点编号三元组的列表。"""
        coords = self.mesh.get_cell_vertices(cell_idx)
        n = len(coords)
        if n == 4:
            return [tuple(f) for f in TET_FACES]
        if n == 8:
            out = []
            for f in HEX_FACES:
                out.append((f[0], f[1], f[2]))
                out.append((f[0], f[2], f[3]))
            return out
        if self.mesh._faces is not None:
            out = []
            for f in self.mesh._faces:
                f = [int(i) for i in f]
                out.append(tuple(f[:3]))
                if len(f) == 4:
                    out.append((f[0], f[2], f[3]))
            return out
        raise ValueError(
            "3D 虚拟元需要单元的面向量；请通过 Mesh(..., faces=...) 提供，"
            "或使用四面体 / 六面体网格。"
        )

    def _fan_simplices(self, cell_idx):
        """把单元从形心剖成若干单纯形。

        2D：三角形 ``(形心, v_i, v_{i+1})``；
        3D：四面体 ``(形心, f_1, f_2, f_3)``（每个面先三角化）。

        Returns
        -------
        simplices : list of (coords, local_ids)
            ``coords`` 是 ``(k, dim)``，``local_ids[0] = -1`` 表示形心，
            其余元素是该单纯形其余顶点在**单元内**的局部编号。
        centroid : (dim,) ndarray
        """
        coords = self.mesh.get_cell_vertices(cell_idx)
        centroid = cell_centroid(coords)
        n = len(coords)
        simplices = []
        if self.mesh.dim == 2:
            for i in range(n):
                j = (i + 1) % n
                simplices.append((np.array([centroid, coords[i], coords[j]]), [-1, i, j]))
        else:
            for (a, b, c) in self._face_index_triangles(cell_idx):
                simplices.append(
                    (np.array([centroid, coords[a], coords[b], coords[c]]), [-1, a, b, c])
                )
        return simplices, centroid

    @staticmethod
    def _barycentric(k: int, points: np.ndarray):
        """参考单纯形上 ``k`` 个顶点的重心坐标值（k = dim+1）。"""
        if k == 3:
            s, t = points[:, 0], points[:, 1]
            return np.column_stack([1.0 - s - t, s, t])
        s, t, u = points[:, 0], points[:, 1], points[:, 2]
        return np.column_stack([1.0 - s - t - u, s, t, u])

    def _fan_integrals(self, cell_idx, f=None, order: int = 4):
        """逐扇单纯形计算 ``∫ f L_k``。

        Returns
        -------
        simplices : list of (local_ids, integrals)
        centroid : (dim,) ndarray
        """
        from ..core.quadrature import QuadratureRule
        from ..core.utils import simplex_jacobian

        simplices, centroid = self._fan_simplices(cell_idx)
        dim = self.mesh.dim
        rule = QuadratureRule.get_quadrature("triangle" if dim == 2 else "tet", order)
        L = self._barycentric(dim + 1, rule.points)
        out = []
        for (coords, ids) in simplices:
            J, detJ = simplex_jacobian(coords)
            pts = coords[0] + rule.points @ J.T
            w = rule.weights * abs(detJ)
            fv = 1.0 if f is None else self._evaluate_function(f, pts)
            out.append((ids, (L * (w * fv)[:, None]).sum(axis=0)))
        return out, centroid

    def _centroid_values(self, cell_idx):
        """每个虚拟基函数在形心处的插值。

        取"顶点自由度 → 线性最小二乘重构"在形心处的值：它保证
        (i) 对线性自由度向量精确（线性再生），
        (ii) 单位分解 ``sum_j c0[j] = 1``（因为 sum_j phi_j = 1）。
        对三角形 / 四边形分别退化到 1/3 与 1/4。
        """
        pinvA = self.reconstruction_matrix(cell_idx)
        _, centroid = self._fan_simplices(cell_idx)
        row = np.concatenate([[1.0], centroid])
        return row @ pinvA

    def local_load(self, cell_idx, f, order: int = None) -> np.ndarray:
        """单元载荷向量。

        * k=1：用**扇形三角剖分上的分片线性插值**近似（三角形网格上精确）；
        * k>=2：增强空间上的 L2 投影载荷 ``(f, Π^0_k φ_i)``，低阶矩精确。
        """
        if self.degree == 1:
            order = 4 if order is None else int(order)
            n = len(self.mesh.elements[cell_idx])
            pieces, _ = self._fan_integrals(cell_idx, f, order)
            c0 = self._centroid_values(cell_idx)
            b = np.zeros(n)
            for (ids, integ) in pieces:
                b += integ[0] * c0
                for k, lid in enumerate(ids[1:], start=1):
                    b[lid] += integ[k]
            return b

        p = self._projection_high(cell_idx)
        qorder = (2 * self.degree) if order is None else int(order)
        F = self._monomial_rhs_load(cell_idx, f, p, qorder)
        return p["Pi0"].T @ F

    def local_mass(self, cell_idx, order: int = 4) -> np.ndarray:
        """单元质量矩阵。

        * k=1：与载荷向量同套的扇形线性插值（三角形网格上精确）；
        * k>=2：L2 投影质量加核空间稳定化，保持 k 次多项式一致性。
        """
        if self.degree == 1:
            from ..core.quadrature import QuadratureRule
            from ..core.utils import simplex_jacobian

            dim = self.mesh.dim
            n = len(self.mesh.elements[cell_idx])
            simplices, _ = self._fan_simplices(cell_idx)
            c0 = self._centroid_values(cell_idx)
            rule = QuadratureRule.get_quadrature("triangle" if dim == 2 else "tet", order)
            from ..core.utils import simplex_jacobian

            L = self._barycentric(dim + 1, rule.points)
            M = np.zeros((n, n))
            coeff = np.zeros((dim + 1, n))
            for (coords, ids) in simplices:
                _, detJ = simplex_jacobian(coords)
                w = rule.weights * abs(detJ)
                coeff[:] = 0.0
                coeff[0, :] = c0
                for k, lid in enumerate(ids[1:], start=1):
                    coeff[k, lid] = 1.0
                gram = np.einsum("qk,ql,q->kl", L, L, w)
                M += coeff.T @ gram @ coeff
            return 0.5 * (M + M.T)

        p = self._projection_high(cell_idx)
        Pi0, D, H = p["Pi0"], p["D"], p["H"]
        M = Pi0.T @ H @ Pi0
        M = 0.5 * (M + M.T)
        if self.stabilization == "none":
            return M
        diff = np.eye(len(D)) - D @ Pi0
        # Geometric volume scaling; vanishes on every polynomial of degree <= k.
        return M + p["measure"] * (diff.T @ diff)

    # ------------------------------------------------------------------
    # 全局矩阵
    # ------------------------------------------------------------------
    def stiffness_matrix(self):
        return self._assemble(self.local_stiffness, symmetric=True)

    def mass_matrix(self):
        return self._assemble(self.local_mass, symmetric=True)

    def load_vector(self, f):
        return self._assemble_vector(lambda c: self.local_load(c, f))

    # ------------------------------------------------------------------
    # 自由度 → 函数值（误差分析 / 可视化用）
    # ------------------------------------------------------------------
    def reconstruction_matrix(self, cell_idx):
        """单元内重构的系数矩阵，``coef = M @ dofs`` 使 ``u_h ≈ Σ_β coef_β m_β``。

        * k=1：顶点自由度上的线性最小二乘拟合（与 ``u_h`` 同阶，O(h^2)）；
        * k>=2：Π^∇ 投影矩阵 ``Pi``（k 次多项式重构，O(h^{k+1})）。
        """
        if self.degree == 1:
            coords = self.mesh.get_cell_vertices(cell_idx)
            A = np.column_stack([np.ones(len(coords)), coords])
            return np.linalg.pinv(A)
        return self._projection_high(cell_idx)["Pi"]

    def evaluate(self, cell_idx, values, points) -> np.ndarray:
        """把单元自由度值重构成函数值并在一组物理点上求值。"""
        if self.degree == 1:
            points = np.atleast_2d(np.asarray(points, dtype=float))
            values = np.asarray(values, dtype=float).ravel()
            dofs = self.get_cell_dofs(cell_idx)
            local_vals = values[dofs]
            pinvA = self.reconstruction_matrix(cell_idx)
            coef = pinvA @ local_vals
            pts = np.column_stack([np.ones(len(points)), points])
            return pts @ coef

        p = self._projection_high(cell_idx)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        values = np.asarray(values, dtype=float).ravel()
        coef = p["Pi"] @ values[self.get_cell_dofs(cell_idx)]
        mono = _scaled_monomial_values(p["multi"], points, p["centroid"], p["h"])
        return mono @ coef

    def evaluate_gradient(self, cell_idx, values, points) -> np.ndarray:
        """重构解的梯度（k=1 单元内常数；k>=2 为 Π^∇ 重构的梯度）。"""
        if self.degree == 1:
            points = np.atleast_2d(np.asarray(points, dtype=float))
            values = np.asarray(values, dtype=float).ravel()
            dofs = self.get_cell_dofs(cell_idx)
            coef = self.reconstruction_matrix(cell_idx) @ values[dofs]
            grad = coef[1:]
            return np.tile(grad, (len(points), 1))

        p = self._projection_high(cell_idx)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        values = np.asarray(values, dtype=float).ravel()
        coef = p["Pi"] @ values[self.get_cell_dofs(cell_idx)]
        grad = _scaled_monomial_grad(p["multi"], points, p["centroid"], p["h"])
        return np.einsum("qad,a->qd", grad, coef)

    def interpolate(self, f) -> np.ndarray:
        """虚拟元的"插值"：取每个自由度泛函在 ``f`` 上的值。

        * k=1：顶点取值；
        * k>=2：顶点/边内部点取值 + 内部矩 ``(1/|K|)∫_K f·q_j``。
        """
        if self.degree == 1:
            return self._evaluate_function(f, self.mesh.nodes)

        if self.mesh.dim == 3:
            return self._interpolate_3d(f)

        vals = np.zeros(self.n_dofs)
        nonint = ~self.dof_is_interior
        vals[nonint] = self._evaluate_function(f, self.dof_coords[nonint])
        for c in range(self.mesh.n_cells):
            p = self._projection_high(c)
            pts, w = self._fan_quadrature(p["coords"], 2 * self.degree)
            fv = self._evaluate_function(f, pts)
            q = _scaled_monomial_values(p["mom_multi"], pts, p["centroid"], p["h"])
            mom_int = (q * (w * fv)[:, None]).sum(axis=0) / p["measure"]
            dofs = self.get_cell_dofs(c)
            start = p["moment_start"]
            for j in range(len(p["mom_multi"])):
                vals[dofs[start + j]] = mom_int[j]
        return vals
