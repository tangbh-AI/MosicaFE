"""连续 Lagrange 有限元空间（FEM 的"元"）。

支持：

=================  ===================
网格单元           可用多项式次数
=================  ===================
``interval``       1, 2
``triangle``       1, 2
``quad``           1, 2 （张量积 Q1/Q2）
``tet``            1, 2
``hex``            1, 2 （张量积 Q1/Q2）
=================  ===================

自由度按"顶点 → 棱 → 面 → 单元内部"的顺序全局共享编号，因此 P2/Q2 的
棱中点自由度在相邻单元之间是**同一个**自由度，全局空间是 H1 协调的。
"""

from __future__ import annotations

import numpy as np

from ..core.quadrature import reference_dimension
from ..core.utils import (
    apply_affine_map,
    compute_jacobian,
    simplex_jacobian,
    transform_gradient,
)
from .base import BaseFESpace
from .reference import (
    simplex_basis,
    simplex_local_descriptors,
    simplex_reference_nodes,
    tensor_basis,
    tensor_local_descriptors,
    tensor_multi_indices,
)

__all__ = ["LagrangeSpace"]

_SIMPLEX = ("interval", "triangle", "tet")
_TENSOR = ("quad", "hex")


class _DofBook:
    """全局自由度登记簿：把几何 key 映射到全局自由度编号。"""

    def __init__(self):
        self.keys = {}
        self.supports = []
        self.interiors = []

    def get(self, key, support, interior):
        if key in self.keys:
            return self.keys[key]
        idx = len(self.supports)
        self.keys[key] = idx
        self.supports.append(tuple(support))
        self.interiors.append(bool(interior))
        return idx

    @property
    def count(self):
        return len(self.supports)


class LagrangeSpace(BaseFESpace):
    """连续 Lagrange 空间 ``P_k`` / ``Q_k``。"""

    family = "lagrange"
    has_explicit_basis = True

    # ------------------------------------------------------------------
    def __init__(self, mesh, degree: int = 1, name: str = None):
        et = mesh.element_type
        if et in ("tri",):
            et = "triangle"
        if et in ("tetra",):
            et = "tet"
        if et in ("polygon", "polyhedron"):
            raise ValueError(
                f"Lagrange 空间不支持 {et} 网格（单元形状不固定）。"
                "请改用 FESpace.vem(mesh, degree=1) 虚拟元，"
                "或把网格改成 triangle/quad/tet/hex。"
            )
        if et not in _SIMPLEX + _TENSOR:
            raise ValueError(f"不支持的网格单元类型 {et!r}")
        if degree not in (1, 2):
            raise NotImplementedError(
                f"{et} 上暂时只实现了 degree=1, 2（收到 degree={degree}）；"
                "更高次元的接入点见 spaces/reference.py。"
            )
        if et == "interval" and mesh.dim != 1:
            raise ValueError("interval 空间需要 1D 网格")
        super().__init__(mesh, degree, name or f"P{degree}[{et}]")
        self._basis_cache = {}

    # ------------------------------------------------------------------
    # 参考单元信息
    # ------------------------------------------------------------------
    def reference_element_type(self) -> str:
        return self.mesh.element_type

    def _is_tensor(self) -> bool:
        return self.mesh.element_type in _TENSOR

    def _local_descriptors(self):
        if self._is_tensor():
            return tensor_local_descriptors(self.mesh.dim, self.degree)
        return simplex_local_descriptors(self.mesh.element_type, self.degree)

    def reference_nodes(self, cell_idx) -> np.ndarray:
        if self._is_tensor():
            multi = tensor_multi_indices(self.mesh.dim, self.degree)
            return np.array(multi, dtype=float) / float(self.degree)
        return simplex_reference_nodes(self.mesh.element_type, self.degree)

    # ------------------------------------------------------------------
    # 基函数
    # ------------------------------------------------------------------
    def eval_basis(self, xi):
        return self._basis(xi)[0]

    def eval_basis_grad(self, xi):
        return self._basis(xi)[1]

    def _basis(self, xi):
        xi_arr = np.asarray(xi, dtype=float)
        if xi_arr.ndim == 1:
            return self._basis_single(xi_arr)
        vals = []
        grads = []
        for row in xi_arr:
            v, g = self._basis_single(row)
            vals.append(v)
            grads.append(g)
        return np.asarray(vals), np.asarray(grads)

    def _basis_single(self, xi):
        key = tuple(np.round(np.asarray(xi, dtype=float), 14))
        hit = self._basis_cache.get(key)
        if hit is not None:
            return hit
        if self._is_tensor():
            out = tensor_basis(self.mesh.dim, self.degree, xi)
        else:
            out = simplex_basis(self.mesh.element_type, self.degree, xi)
        self._basis_cache[key] = out
        return out

    # ------------------------------------------------------------------
    # 自由度
    # ------------------------------------------------------------------
    def _build_dof_map(self):
        book = _DofBook()
        descriptors = self._local_descriptors()
        dof_map = []
        for c, cell in enumerate(self.mesh.elements):
            local = []
            for kind, data in descriptors:
                if kind == "vertex":
                    v = int(cell[data])
                    local.append(book.get(("v", v), (v,), False))
                elif kind == "edge":
                    a, b = int(cell[data[0]]), int(cell[data[1]])
                    lo, hi = (a, b) if a < b else (b, a)
                    local.append(book.get(("e", lo, hi), (lo, hi), False))
                elif kind == "face":
                    vs = tuple(sorted(int(cell[i]) for i in data))
                    local.append(book.get(("f",) + vs, vs, False))
                else:  # interior
                    local.append(book.get(("c", c), (), True))
            dof_map.append(np.asarray(local, dtype=np.int64))
        self.dof_map = dof_map
        self.n_dofs = book.count
        self.dof_support = book.supports
        self.dof_is_interior = np.asarray(book.interiors, dtype=bool)

    # ------------------------------------------------------------------
    # 单元矩阵
    # ------------------------------------------------------------------
    def default_quad_order(self) -> int:
        """刚度矩阵的求积阶。

        单纯形按**总次数** 2(p-1)；张量积单元的被积函数在每个方向上是
        2(p-1) 次，而 n 点 Gauss 规则精确到每方向 2n-1 次，
        所以取 ``order = 2p`` 才能保证 n = p 个点/方向。
        """
        if self._is_tensor():
            return max(2, 2 * self.degree)
        return max(1, 2 * (self.degree - 1))

    def mass_quad_order(self) -> int:
        if self._is_tensor():
            return 2 * self.degree + 2
        return max(1, 2 * self.degree)

    def load_quad_order(self) -> int:
        if self._is_tensor():
            return 2 * self.degree + 4
        return max(2, 2 * self.degree + 2)

    def _quadrature_data(self, order: int):
        """缓存参考单元上的基函数求值（与单元无关，只与阶数有关）。"""
        key = ("q", int(order))
        hit = self._basis_cache.get(key)
        if hit is not None:
            return hit
        quad = self.get_quadrature(order)
        n_local = len(self._local_descriptors())
        N = np.zeros((quad.n_points, n_local))
        dN = np.zeros((quad.n_points, n_local, self.mesh.dim))
        for q in range(quad.n_points):
            v, g = self._basis_single(quad.points[q])
            N[q] = v
            dN[q] = g
        out = (quad.points, quad.weights, N, dN)
        self._basis_cache[key] = out
        return out

    def _element_geometry(self, cell_idx, ref_points):
        coords = self.mesh.get_cell_vertices(cell_idx)
        phys = apply_affine_map(ref_points, coords)
        dets = np.zeros(len(ref_points))
        jac = []
        for q, xi in enumerate(ref_points):
            J, detJ = compute_jacobian(coords, xi)
            jac.append(J)
            dets[q] = abs(detJ)
        return phys, jac, dets

    def local_stiffness(self, cell_idx) -> np.ndarray:
        """单元刚度矩阵。"""
        pts, w, N, dN_ref = self._quadrature_data(self.default_quad_order())
        _, jac, dets = self._element_geometry(cell_idx, pts)
        nl = N.shape[1]
        Ke = np.zeros((nl, nl))
        for q in range(len(w)):
            dN = transform_gradient(dN_ref[q], jac[q])
            Ke += (w[q] * dets[q]) * (dN @ dN.T)
        del N
        return Ke

    def local_mass(self, cell_idx) -> np.ndarray:
        """单元质量矩阵。"""
        pts, w, N, _ = self._quadrature_data(self.mass_quad_order())
        _, _, dets = self._element_geometry(cell_idx, pts)
        nl = N.shape[1]
        Me = np.zeros((nl, nl))
        for q in range(len(w)):
            Me += (w[q] * dets[q]) * np.outer(N[q], N[q])
        return Me

    def local_load(self, cell_idx, f, order=None) -> np.ndarray:
        """单元载荷向量。"""
        order = self.load_quad_order() if order is None else int(order)
        pts, w, N, _ = self._quadrature_data(order)
        phys, _, dets = self._element_geometry(cell_idx, pts)
        fv = self._evaluate_function(f, phys)
        return (N * (w * dets * fv)[:, None]).sum(axis=0)

    # ------------------------------------------------------------------
    # 全局矩阵
    # ------------------------------------------------------------------
    def stiffness_matrix(self):
        return self._assemble(self.local_stiffness, symmetric=True)

    def mass_matrix(self):
        return self._assemble(self.local_mass, symmetric=True)

    def load_vector(self, f, order=None):
        return self._assemble_vector(lambda c: self.local_load(c, f, order))

    # ------------------------------------------------------------------
    def interpolate(self, f) -> np.ndarray:
        """按自由度坐标做 Lagrange 插值。"""
        return self._evaluate_function(f, self.dof_coords)

    # ------------------------------------------------------------------
    # 自由度 → 函数
    # ------------------------------------------------------------------
    def _inverse_map(self, cell_idx, points, iters: int = 25):
        """把物理点映回参考单元坐标。"""
        coords = self.mesh.get_cell_vertices(cell_idx)
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        d = self.mesh.dim
        if not self._is_tensor():
            J, _ = simplex_jacobian(coords)
            return (pts - coords[0]) @ np.linalg.inv(J).T
        xi = np.full((len(pts), d), 0.5)
        for _ in range(iters):
            mapped = apply_affine_map(xi, coords)
            res = pts - mapped
            if np.max(np.linalg.norm(res, axis=1)) < 1e-13:
                break
            for q in range(len(pts)):
                J, detJ = compute_jacobian(coords, xi[q])
                if abs(detJ) < 1e-14:
                    continue
                xi[q] = xi[q] + np.linalg.solve(J, res[q])
        return xi

    def evaluate(self, cell_idx, values, points):
        values = np.asarray(values, dtype=float).ravel()
        dofs = self.get_cell_dofs(cell_idx)
        local = values[dofs]
        xi = self._inverse_map(cell_idx, points)
        out = np.zeros(len(xi))
        for q in range(len(xi)):
            N, _ = self._basis_single(xi[q])
            out[q] = float(N @ local)
        return out

    def evaluate_gradient(self, cell_idx, values, points):
        values = np.asarray(values, dtype=float).ravel()
        dofs = self.get_cell_dofs(cell_idx)
        local = values[dofs]
        coords = self.mesh.get_cell_vertices(cell_idx)
        xi = self._inverse_map(cell_idx, points)
        out = np.zeros((len(xi), self.mesh.dim))
        for q in range(len(xi)):
            _, dN_ref = self._basis_single(xi[q])
            J, _ = compute_jacobian(coords, xi[q])
            dN = transform_gradient(dN_ref, J)
            out[q] = dN.T @ local
        return out

    def project(self, f) -> np.ndarray:
        """L2 投影到本空间（质量矩阵求解），返回自由度向量。"""
        from ..solvers.linear import Solver

        M = self.mass_matrix()
        b = self.load_vector(f, order=max(self.mass_quad_order(), 4))
        return Solver.direct(M, b)

    # ------------------------------------------------------------------
    @property
    def ref_dim(self) -> int:
        return reference_dimension(self.reference_element_type())
