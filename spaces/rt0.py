"""RT0×P0 混合有限元空间（最低阶 Raviart--Thomas × 分片常数）。

与 Lagrange / VEM 不同，这里是**混合格式**：同时离散通量与压力

.. math::

   q=-\\nabla p,\\qquad \\nabla\\cdot q=f\\ \\text{in}\\ \\Omega,\\qquad
   p=0\\ \\text{on}\\ \\partial\\Omega .

最低阶 Raviart--Thomas 空间
---------------------------
在单纯形 :math:`K`（2D 三角形 / 3D 四面体，顶点 :math:`a_0,\\dots,a_d`）上
取显式基函数

.. math::

   \\varphi_i(x)=\\frac{x-a_i}{d\\,|K|},\\qquad i=0,\\dots,d,

它满足"面法向通量"自由度归一化条件

.. math::

   \\int_{f_i}\\varphi_i\\cdot n_i\\,\\mathrm ds=1,\\qquad
   \\int_{f_j}\\varphi_i\\cdot n_j\\,\\mathrm ds=0\\ (i\\ne j),
   \\qquad \\nabla\\cdot\\varphi_i=\\frac{1}{|K|},

其中 :math:`f_i` 是**与顶点 :math:`i` 相对的面**（2D 的边 / 3D 的三角面），
:math:`n_i` 是其外法向。

自由度布局（本库的约定）
------------------------
* **通量自由度** :math:`\\Phi_f=\\int_f q\\cdot n_f\\,\\mathrm ds`，
  每个**全局面**一个，由相邻单元共享——这正是
  :math:`H(\\mathrm{div})` 协调性，面拓扑见 :mod:`MosicaFE.core.facets`；
* **压力自由度**：每个单元一个常数；
* 自由度向量排成 ``[Φ (n_flux 个), P (n_cells 个)]``，
  而"单元局部自由度"是 ``[d+1 个面, 1 个单元常数]``。

齐次 Dirichlet 条件是混合格式的**自然**边界条件：边界面的通量自由度保留为
未知量即可，不做强加、也不做消去（``boundary_dofs()`` 返回空数组）。

空间层提供的矩阵原语
--------------------
与 :class:`~MosicaFE.spaces.lagrange.LagrangeSpace` 暴露 ``stiffness_matrix`` /
``mass_matrix`` / ``load_vector`` 完全平行，RT0 暴露的是**鞍点问题的两个块**：

.. code-block:: python

   V = FESpace.rt0(mesh)
   M = V.flux_mass_matrix()          # (Φ,Φ) 块   ∫_K φ_i·φ_j
   B = V.divergence_matrix()         # (P,Φ) 块   ∫_K ∇·φ_i = σ
   f = V.pressure_load_vector(src)   # (P,)       ∫_K f

物理层（:class:`~MosicaFE.physics.mixed_poisson.MixedPoissonProblem`）只用它们
拼出 ``[[M, -Bᵀ], [B, 0]]``，与 DECISIONS.md D3 约定的"空间算矩阵、物理做组合"
完全一致。要写别的混合问题（Darcy、Stokes 的同阶配对……）时不必重写单元循环。

参考文献：Raviart & Thomas (1977)；Boffi, Brezzi & Fortin,
*Mixed Finite Element Methods and Applications*, Springer (2013).
"""

from __future__ import annotations

import math

import numpy as np

from .._compat import from_triplets
from ..core.facets import build_facet_topology, local_facets, simplex_element_array
from ..core.quadrature import cell_quadrature, simplex_quadrature
from .base import BaseFESpace

__all__ = ["RT0Space", "rt0_flux_basis", "rt0_mass_matrices"]

#: 单元类型别名（与 core.mesh 的约定一致）
_TRI_ALIASES = ("tri", "triangle", "simplex")
_TET_ALIASES = ("tet", "tetra", "tetrahedron", "simplex3d")


# ---------------------------------------------------------------------------
# 单元局部量：显式基函数与质量矩阵
# ---------------------------------------------------------------------------
def _simplex_measure(vertices) -> float:
    """单纯形测度：2D 面积 / 3D 体积。"""
    v = np.asarray(vertices, dtype=float)
    d = v.shape[1]
    if v.shape[0] != d + 1:
        raise ValueError(f"顶点数组形状应为 (d+1, d)，收到 {v.shape}")
    J = np.array([v[i + 1] - v[0] for i in range(d)])
    return float(abs(np.linalg.det(J)) / math.factorial(d))


def _batch_measure(vertices: np.ndarray) -> np.ndarray:
    """批量单纯形测度，形状 ``(n_cells,)``。"""
    v = np.asarray(vertices, dtype=float)
    d = v.shape[2]
    diffs = v[:, 1:, :] - v[:, :1, :]                  # (m, d, d)
    meas = np.abs(np.linalg.det(diffs)) / math.factorial(d)
    if np.any(meas <= 0.0):
        raise ValueError("存在零测度的退化单元，请检查网格")
    return meas


def rt0_flux_basis(vertices, points) -> np.ndarray:
    """RT0 基函数值 :math:`\\varphi_i(x)=(x-a_i)/(d|K|)`。

    Parameters
    ----------
    vertices : (d+1, d)
    points : (n_points, d)

    Returns
    -------
    (n_points, d+1, d) ndarray，``[q, i]`` 是向量 :math:`\\varphi_i(x_q)`。
    """
    v = np.asarray(vertices, dtype=float)
    p = np.atleast_2d(np.asarray(points, dtype=float))
    d = v.shape[1]
    if v.shape[0] != d + 1:
        raise ValueError(f"vertices 形状应为 (d+1, d)，收到 {v.shape}")
    meas = _simplex_measure(v)
    return (p[:, None, :] - v[None, :, :]) / (d * meas)


def rt0_mass_matrices(vertices, lumped: bool = False) -> np.ndarray:
    """批量 RT0 质量矩阵 :math:`M_{ij}=\\int_K\\varphi_i\\cdot\\varphi_j`。

    Parameters
    ----------
    vertices : (n_cells, d+1, d) array_like
    lumped : bool
        ``False``（默认）用**精确闭式**（由
        :math:`\\int_K\\lambda_i\\lambda_j` 的精确值推出）

        .. math::

           M_{ij}=\\frac{S_i\\cdot S_j+G_{ij}}{d^2(d+1)(d+2)\\,|K|},\\qquad
           S_i=\\sum_k(a_k-a_i),\\quad
           G_{ij}=\\sum_k(a_k-a_i)\\cdot(a_k-a_j);

        ``True`` 用顶点（梯形）求积 :math:`w_v=|K|/(d+1)` 的集中版本，
        与文献中的"集中质量 / 有限体积"格式一致。

    Returns
    -------
    (n_cells, d+1, d+1) ndarray，对称正定，且与单元定向无关。
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 3 or v.shape[1] != v.shape[2] + 1:
        raise ValueError(f"vertices 形状应为 (n_cells, d+1, d)，收到 {v.shape}")
    n = v.shape[1]
    d = v.shape[2]
    meas = _batch_measure(v)

    if lumped:
        # 顶点求积：φ_i(a_v) = (a_v - a_i)/(d|K|)，权重 w_v = |K|/(d+1)
        phi = (v[:, :, None, :] - v[:, None, :, :]) / (
            d * meas[:, None, None, None]
        )                                              # (m, 顶点, 基函数, d)
        M = np.einsum("cvid,cvjd->cij", phi, phi) * (meas / n)[:, None, None]
        return 0.5 * (M + np.swapaxes(M, -1, -2))

    S = v.sum(axis=1)[:, None, :] - n * v              # S[c,i] = Σ_k (a_k - a_i)
    SS = np.einsum("mid,mjd->mij", S, S)
    D = v[:, :, None, :] - v[:, None, :, :]            # D[c,k,i] = a_k - a_i
    G = np.einsum("mkid,mkjd->mij", D, D)
    factor = d * d * (d + 1) * (d + 2)
    M = (SS + G) / (factor * meas[:, None, None])
    return 0.5 * (M + np.swapaxes(M, -1, -2))


# ---------------------------------------------------------------------------
# 空间
# ---------------------------------------------------------------------------
class RT0Space(BaseFESpace):
    """RT0×P0 混合空间（2D 三角形 / 3D 四面体网格）。

    Parameters
    ----------
    mesh : Mesh
        单纯形网格：``Mesh.rectangle(element_type="triangle")``（默认）或
        ``Mesh.box(element_type="tet")``。
    degree : int
        最低阶 RT0 固定为 1（保留参数以统一接口）。
    lumped : bool
        是否默认使用集中（顶点求积）质量矩阵。

    Examples
    --------
    .. code-block:: python

       V = FESpace.rt0(Mesh.rectangle(nx=16, ny=16))
       V.n_flux, V.n_pressure, V.n_dofs       # 面数、单元数、总自由度数
       problem = MixedPoissonProblem(V, f=source)
       Q, P = problem.solve()
    """

    family = "rt0"
    #: RT0 的基函数是**显式**的（只是向量值），见 :func:`rt0_flux_basis`
    has_explicit_basis = True
    is_mixed = True

    def __init__(self, mesh, degree: int = 1, name: str = None, lumped: bool = False):
        d = int(mesh.dim)
        et = str(mesh.element_type).lower()
        if d == 2 and et not in _TRI_ALIASES:
            raise ValueError(
                "2D 的 RT0 需要三角形网格（Mesh.rectangle(element_type='triangle')）；"
                f"当前单元类型为 {et!r}。"
            )
        if d == 3 and et not in _TET_ALIASES:
            raise ValueError(
                "3D 的 RT0 需要四面体网格（Mesh.box(element_type='tet')）；"
                f"当前单元类型为 {et!r}。"
            )
        if d not in (2, 3):
            raise ValueError("RT0 目前只支持 2D 三角形与 3D 四面体网格")
        self.lumped = bool(lumped)
        self._mass_cache = {}
        self._div_cache = None
        super().__init__(mesh, degree, name or "RT0-P0")

    # ------------------------------------------------------------------
    # 自由度布局
    # ------------------------------------------------------------------
    def _build_dof_map(self):
        topo = build_facet_topology(self.mesh)
        self._topology = topo
        self._facet_vertices = topo.facets
        self._cell_facet_dofs = np.asarray(topo.cell_facets, dtype=np.int64)
        self._facet_signs = np.asarray(topo.signs, dtype=float)
        self._cell_vertices = np.asarray(self.mesh.nodes, dtype=float)[
            simplex_element_array(self.mesh)
        ]

        self.n_flux = int(topo.n_facets)
        self.n_pressure = int(self.mesh.n_cells)
        self.n_dofs = self.n_flux + self.n_pressure

        # 局部自由度 = [d+1 个面通量, 1 个单元常数压力]
        self.dof_map = [
            np.concatenate([self._cell_facet_dofs[c], [self.n_flux + c]])
            for c in range(self.n_pressure)
        ]
        # 支撑集合（面 / 单元的顶点），供 boundary_dofs 与后处理使用
        self.dof_support = [tuple(int(v) for v in f) for f in self._facet_vertices]
        self.dof_support += [
            tuple(int(v) for v in np.asarray(e, dtype=np.int64).ravel())
            for e in self.mesh.elements
        ]
        # 边界面上的通量自由度不算"内部"；压力自由度是单元平均，视为内部
        self.dof_is_interior = np.concatenate(
            [topo.interior_mask, np.ones(self.n_pressure, dtype=bool)]
        )

    def _build_dof_coordinates(self):
        coords = np.zeros((self.n_dofs, self.mesh.dim))
        coords[: self.n_flux] = np.asarray(self.mesh.nodes, dtype=float)[
            self._facet_vertices
        ].mean(axis=1)
        coords[self.n_flux:] = self.mesh.cell_centroids()
        self.dof_coords = coords

    def reference_nodes(self, cell_idx: int = 0) -> np.ndarray:
        """局部自由度在参考单纯形上的坐标 ``(d+2, d)``。

        前 ``d+1`` 个是各局部面的重心，最后一个是单元重心。参考单纯形的参数化
        与 :func:`MosicaFE.core.utils.apply_affine_map` 一致
        （``x = a_0 + (ξ_1,…,ξ_d)·J``），因此可直接映射回物理单元。
        """
        d = self.mesh.dim
        ref = np.zeros((d + 2, d))
        for i, f in enumerate(local_facets(d)):
            for j in f:
                if j > 0:
                    ref[i, j - 1] = 1.0 / len(f)
        ref[-1] = 1.0 / (d + 1)
        return ref

    def boundary_dofs(self) -> np.ndarray:
        """混合格式下齐次 Dirichlet 是自然边界条件，没有需要强加的自由度。"""
        return np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def flux_dofs(self) -> np.ndarray:
        """通量自由度编号（每个面一个）。"""
        return np.arange(self.n_flux, dtype=np.int64)

    @property
    def pressure_dofs(self) -> np.ndarray:
        """压力自由度编号（每个单元一个）。"""
        return np.arange(self.n_flux, self.n_dofs, dtype=np.int64)

    @property
    def topology(self):
        """面拓扑 :class:`~MosicaFE.core.facets.FacetTopology`。"""
        return self._topology

    @property
    def n_boundary_facets(self) -> int:
        """边界面个数。"""
        return int(self._topology.n_boundary)

    def cell_flux_dofs(self, cell_idx: int) -> np.ndarray:
        """单元 ``cell_idx`` 的局部面通量自由度（按局部面序，长度 d+1）。"""
        return self._cell_facet_dofs[cell_idx]

    def cell_pressure_dof(self, cell_idx: int) -> int:
        """单元 ``cell_idx`` 的常数压力自由度编号。"""
        return int(self.n_flux + cell_idx)

    def facet_signs(self, cell_idx: int) -> np.ndarray:
        """单元局部面的外法向符号（相对全局面法向，±1）。"""
        return self._facet_signs[cell_idx]

    def facet_vertices(self, facet_idx: int) -> np.ndarray:
        """全局面 ``facet_idx`` 的顶点编号。"""
        return np.asarray(self._facet_vertices[facet_idx], dtype=np.int64)

    def cell_vertices(self, cell_idx: int) -> np.ndarray:
        """单元 ``cell_idx`` 的顶点坐标 ``(d+1, d)``。"""
        return self._cell_vertices[cell_idx]

    # ------------------------------------------------------------------
    # 单元局部量
    # ------------------------------------------------------------------
    def local_flux_mass(self, cell_idx: int, lumped: bool = None) -> np.ndarray:
        """单元 ``cell_idx`` 的 RT0 质量矩阵 ``∫_K φ_i·φ_j``，形状 ``(d+1, d+1)``。"""
        lumped = self.lumped if lumped is None else bool(lumped)
        return rt0_mass_matrices(
            self._cell_vertices[cell_idx][None, ...], lumped=lumped
        )[0]

    def eval_flux_basis(self, cell_idx: int, points) -> np.ndarray:
        """在单元内的给定点上求 RT0 基函数，返回 ``(n_points, d+1, d)``。"""
        return rt0_flux_basis(self._cell_vertices[cell_idx], points)

    def eval_basis(self, xi):
        """RT0 没有标量节点基函数（它是向量值混合元）。"""
        raise NotImplementedError(
            "RT0 的基函数是向量值的面基函数；请用 V.eval_flux_basis(cell, points) "
            "或 rt0_flux_basis(vertices, points)。"
        )

    # ------------------------------------------------------------------
    # 全局矩阵原语（物理层只做块组合）
    # ------------------------------------------------------------------
    def flux_mass_matrix(self, lumped: bool = None):
        """(Φ,Φ) 块 ``M_ij = ∫_K φ_i·φ_j``，形状 ``(n_flux, n_flux)``。

        已按外法向符号装配，因此与单元定向无关。有 scipy 时返回 CSR 稀疏矩阵，
        否则返回稠密 ndarray。
        """
        lumped = self.lumped if lumped is None else bool(lumped)
        if lumped in self._mass_cache:
            return self._mass_cache[lumped]
        n = self.mesh.dim + 1
        Mloc = rt0_mass_matrices(self._cell_vertices, lumped=lumped)
        sig = self._facet_signs
        vals = Mloc * sig[:, :, None] * sig[:, None, :]
        fid = self._cell_facet_dofs
        rows = np.repeat(fid, n, axis=1).ravel()
        cols = np.tile(fid, (1, n)).ravel()
        M = from_triplets(rows, cols, vals.ravel(), (self.n_flux, self.n_flux))
        self._mass_cache[lumped] = M
        return M

    def divergence_matrix(self):
        """(P,Φ) 块 ``B[K,i] = ∫_K ∇·φ_i = σ_{K,i}``，形状 ``(n_pressure, n_flux)``。

        散度由散度定理精确给出（:math:`\\int_K\\nabla\\cdot\\varphi_i=1`），
        不需要任何求积，因此这个块是 :math:`\\pm 1` 的稀疏矩阵。
        """
        if self._div_cache is None:
            n = self.mesh.dim + 1
            fid = self._cell_facet_dofs
            rows = np.repeat(np.arange(self.n_pressure), n)
            cols = fid.ravel()
            vals = self._facet_signs.ravel()
            self._div_cache = from_triplets(
                rows, cols, vals, (self.n_pressure, self.n_flux)
            )
        return self._div_cache

    def pressure_load_vector(self, f, quadrature_order: int = None) -> np.ndarray:
        """压力载荷 ``∫_K f``，形状 ``(n_pressure,)``。

        ``f`` 是标量源项。缺省求积阶 3（2D 4 点 / 3D 5 点），对多项式与常见
        光滑源项都足够。
        """
        order = 3 if quadrature_order is None else int(quadrature_order)
        out = np.zeros(self.n_pressure)
        for c in range(self.n_pressure):
            pts, w = cell_quadrature(self.mesh, c, order)
            out[c] = float(np.sum(w * self._evaluate_function(f, pts)))
        return out

    # ------------------------------------------------------------------
    # 插值
    # ------------------------------------------------------------------
    def interpolate_pressure(self, f) -> np.ndarray:
        """把标量函数投影到 P0：返回每个单元的平均值 ``∫_K f / |K|``。"""
        out = np.zeros(self.n_pressure)
        for c in range(self.n_pressure):
            pts, w = cell_quadrature(self.mesh, c, 3)
            out[c] = float(np.sum(w * self._evaluate_function(f, pts))) / float(np.sum(w))
        return out

    def interpolate_flux(self, q) -> np.ndarray:
        """把向量场 ``q(x) -> (d,)`` 插值成**面法向通量**自由度
        :math:`\\Phi_f=\\int_f q\\cdot n_f\\,\\mathrm ds`（全局自由度向量）。

        用 :meth:`local_flux_values` 可把它换成逐单元的外法向通量。
        """
        nodes = np.asarray(self.mesh.nodes, dtype=float)
        Phi = np.zeros(self.n_flux)
        for f in range(self.n_flux):
            verts = nodes[self._facet_vertices[f]]
            pts, w = simplex_quadrature(verts, order=3)
            vals = np.array([np.asarray(q(p), dtype=float).ravel() for p in pts])
            Phi[f] = float(np.sum(w * (vals @ self._topology.normals[f])))
        return Phi

    def interpolate(self, f):
        """混合空间没有单一的插值算子，请显式选择压力或通量。"""
        raise NotImplementedError(
            "RT0×P0 是混合空间：请用 V.interpolate_pressure(f) 得到单元平均值，"
            "或 V.interpolate_flux(q) 得到面法向通量自由度。"
        )

    # ------------------------------------------------------------------
    # 重构与误差分析
    # ------------------------------------------------------------------
    def local_flux_values(self, cell_idx: int, values) -> np.ndarray:
        """把各种形式的解向量换成单元 ``cell_idx`` 的**外法向**通量 ``(d+1,)``。

        接受逐单元通量表 ``(n_cells, d+1)``（``solve()`` 返回的 ``Q``）、
        单单元的局部通量 ``(d+1,)``、全局通量自由度 ``(n_flux,)``，
        或完整混合解 ``(n_dofs,)``（压力部分被忽略）。
        """
        arr = np.asarray(values, dtype=float)
        n = self.mesh.dim + 1
        if arr.ndim == 2:
            if arr.shape == (self.mesh.n_cells, n):
                return arr[cell_idx]
            raise ValueError(
                f"通量表形状应为 ({self.mesh.n_cells}, {n})，收到 {arr.shape}"
            )
        v = arr.ravel()
        if v.size == n:
            return v
        if v.size == self.n_dofs:
            v = v[: self.n_flux]
        if v.size != self.n_flux:
            raise ValueError(
                f"通量向量长度应为 {n}（局部）/ {self.n_flux}（全局）/ "
                f"{self.n_dofs}（混合解），收到 {v.size}"
            )
        return self._facet_signs[cell_idx] * v[self._cell_facet_dofs[cell_idx]]

    def evaluate_flux(self, cell_idx: int, values, points) -> np.ndarray:
        """在单元 ``cell_idx`` 内重构 RT0 通量 :math:`q_h=\\sum_i F_i\\varphi_i`。

        Returns
        -------
        (n_points, d) ndarray
        """
        local = self.local_flux_values(cell_idx, values)
        phi = self.eval_flux_basis(cell_idx, points)       # (nq, d+1, d)
        return np.einsum("qid,i->qd", phi, local)

    def evaluate(self, cell_idx: int, values, points) -> np.ndarray:
        """在单元内取分片常数压力（``values`` 可以是 ``P`` 或完整混合解）。"""
        v = np.asarray(values, dtype=float).ravel()
        if v.size == self.n_dofs:
            p = v[self.n_flux + cell_idx]
        elif v.size == self.n_pressure:
            p = v[cell_idx]
        else:
            raise ValueError(
                f"压力向量长度应为 {self.n_pressure} 或 {self.n_dofs}，收到 {v.size}"
            )
        nq = np.atleast_2d(np.asarray(points, dtype=float)).shape[0]
        return np.full(nq, float(p))

    def evaluate_gradient(self, cell_idx: int, values, points) -> np.ndarray:
        raise NotImplementedError(
            "混合格式里梯度就是通量：请用 V.evaluate_flux(cell, Q, points)（取负号得到 -q_h）。"
        )

    def cell_barycentric(self, cell_idx: int, points) -> np.ndarray:
        """点相对于单元顶点的重心坐标 ``(n_points, d+1)``。"""
        v = self._cell_vertices[cell_idx]
        p = np.atleast_2d(np.asarray(points, dtype=float))
        d = self.mesh.dim
        J = np.array([v[i + 1] - v[0] for i in range(d)])           # (d, d)
        # x = v0 + ξ·J^T，故 ξ = J^{-T}(x - v0)
        lam = np.linalg.solve(J.T, (p - v[0]).T).T                  # (nq, d)
        return np.column_stack([1.0 - lam.sum(axis=1), lam])

    def locate_cell(self, point, tol: float = 1e-9) -> int:
        """返回包含 ``point`` 的单元编号（单纯形网格用重心坐标判定）。"""
        for c in range(self.mesh.n_cells):
            bary = self.cell_barycentric(c, point)[0]
            if np.all(bary >= -tol) and np.all(bary <= 1.0 + tol):
                return c
        raise ValueError(f"点 {np.asarray(point, dtype=float)} 不在网格内")

    # ------------------------------------------------------------------
    # 混合格式没有单一算子接口
    # ------------------------------------------------------------------
    def _not_supported(self, what: str, hint: str):
        raise NotImplementedError(f"RT0×P0 是混合格式，没有单一的{what}。{hint}")

    def stiffness_matrix(self):
        self._not_supported(
            "刚度矩阵",
            "鞍点系统的两个块请用 V.flux_mass_matrix()（(Φ,Φ) 块）与 "
            "V.divergence_matrix()（(P,Φ) 块），或用 MixedPoissonProblem 组装完整系统。",
        )

    def mass_matrix(self):
        self._not_supported(
            "质量矩阵",
            "通量的质量矩阵是 V.flux_mass_matrix()；压力的质量矩阵是 diag(|K|)。",
        )

    def load_vector(self, f):
        self._not_supported(
            "载荷向量",
            "源项只进入压力方程：请用 V.pressure_load_vector(f)。",
        )

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------
    def __repr__(self):  # pragma: no cover - 仅展示
        return (
            f"<RT0Space {self.name} dim={self.mesh.dim} "
            f"flux_dofs={self.n_flux} pressure_dofs={self.n_pressure} "
            f"cells={self.mesh.n_cells}>"
        )
