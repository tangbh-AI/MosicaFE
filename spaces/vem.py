"""虚拟元空间 (Virtual Element Method, k=1)。

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

import numpy as np

from ..core.utils import HEX_FACES, TET_FACES, cell_centroid, cell_measure, edge_normals_2d
from .base import BaseFESpace

__all__ = ["VEMSpace"]

_EPS = 1e-14


class VEMSpace(BaseFESpace):
    """一次虚拟元空间（顶点型自由度）。"""

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
        if degree != 1:
            raise NotImplementedError(
                "当前只实现了 k=1 虚拟元（每个顶点一个自由度）；"
                "更高次 k 的接入点见 spaces/vem.py 的 _build_projection。"
            )
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
        self.n_dofs = self.mesh.n_nodes
        self.dof_map = [np.asarray(e, dtype=np.int64) for e in self.mesh.elements]
        self.dof_support = [(v,) for v in range(self.mesh.n_nodes)]
        self.dof_is_interior = np.zeros(self.n_dofs, dtype=bool)

    def _build_dof_coordinates(self):
        self.dof_coords = np.array(self.mesh.nodes, dtype=float, copy=True)

    def reference_nodes(self, cell_idx) -> np.ndarray:
        return np.zeros((len(self.mesh.elements[cell_idx]), self.mesh.dim))

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
                tris = [(0, 1, 2), (0, 2, 3)]
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
    # 单元矩阵
    # ------------------------------------------------------------------
    def local_stiffness(self, cell_idx) -> np.ndarray:
        """单元刚度矩阵 ``A_proj + S_stab``。"""
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

    def local_load(self, cell_idx, f, order: int = 4) -> np.ndarray:
        """单元载荷向量。

        虚拟基函数没有显式表达式，因此用**扇形三角剖分上的分片线性插值**
        近似：

        .. math::

           \\int_K f\\varphi_j \\approx
             c_0[j]\\sum_i \\!\\int_{T_i}\\! f L_0
             + \\int_{T_j}\\! f L_1 + \\int_{T_{j-1}}\\! f L_2

        对三角形网格这是精确的；对一般多边形它保持正确的几何权重。
        （旧脚本里一律用 ``|K|/n``，会让短边上的顶点被严重高估，
        在不规则多边形上直接导致收敛停滞——见 progress/VERIFICATION.md。）
        """
        n = len(self.mesh.elements[cell_idx])
        pieces, _ = self._fan_integrals(cell_idx, f, order)
        c0 = self._centroid_values(cell_idx)
        b = np.zeros(n)
        for (ids, integ) in pieces:
            b += integ[0] * c0
            for k, lid in enumerate(ids[1:], start=1):
                b[lid] += integ[k]
        return b

    def local_mass(self, cell_idx, order: int = 4) -> np.ndarray:
        """单元质量矩阵：与载荷向量使用同一套扇形线性插值。

        三角形网格上就是精确的 P1 质量矩阵；一般多边形上它对称、半正定，
        行和等于 :math:`\\int_K\\varphi_j`（与载荷向量相容）。
        """
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
        """返回单元内线性最小二乘重构的系数 ``(a, b)``，``u(x) ≈ a + b·x``。

        虚拟元没有显式基函数，但在顶点自由度上对一次多项式做最小二乘拟合，
        得到的是与 ``u_h`` 同阶（``O(h^2)``）的连续重构，可用于误差范数与绘图。
        """
        coords = self.mesh.get_cell_vertices(cell_idx)
        A = np.column_stack([np.ones(len(coords)), coords])
        # 返回伪逆，便于乘自由度向量
        return np.linalg.pinv(A)

    def evaluate(self, cell_idx, values, points) -> np.ndarray:
        """把单元自由度值重构成函数值并在一组物理点上求值。"""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        values = np.asarray(values, dtype=float).ravel()
        dofs = self.get_cell_dofs(cell_idx)
        local_vals = values[dofs]
        pinvA = self.reconstruction_matrix(cell_idx)
        coef = pinvA @ local_vals
        pts = np.column_stack([np.ones(len(points)), points])
        return pts @ coef

    def evaluate_gradient(self, cell_idx, values, points) -> np.ndarray:
        """重构解的梯度（每个单元内是常数，等于最小二乘拟合的线性系数）。"""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        values = np.asarray(values, dtype=float).ravel()
        dofs = self.get_cell_dofs(cell_idx)
        coef = self.reconstruction_matrix(cell_idx) @ values[dofs]
        grad = coef[1:]
        return np.tile(grad, (len(points), 1))

    def interpolate(self, f) -> np.ndarray:
        """虚拟元的"插值"就是取函数在顶点上的值。"""
        return self._evaluate_function(f, self.mesh.nodes)
