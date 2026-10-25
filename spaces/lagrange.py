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

# 模块职责：连续 Lagrange 有限元
# 通过共享顶点、棱或面上的自由度构造连续空间，再用显式多项式基函数计算局部积分。
# 主要流程：建立自由度编号 → 缓存参考基函数 → 计算单元几何 → 求局部矩阵 → 累加为全局系统。


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
        # 建立几何支撑到编号的登记表，以及自由度支撑和内部标记列表。
        self.keys = {}
        self.supports = []
        self.interiors = []

    def get(self, key, support, interior):
        # 按几何支撑键复用或创建自由度编号，使相邻单元共享同一未知量。
        if key in self.keys:
            return self.keys[key]
        idx = len(self.supports)
        self.keys[key] = idx
        self.supports.append(tuple(support))
        self.interiors.append(bool(interior))
        return idx

    @property
    def count(self):
        # 返回登记簿中已经建立的自由度数量，用于确定全局向量长度。
        return len(self.supports)


class LagrangeSpace(BaseFESpace):
    """连续 Lagrange 空间 ``P_k`` / ``Q_k``。"""

    family = "lagrange"
    has_explicit_basis = True

    # ------------------------------------------------------------------
    def __init__(self, mesh, degree: int = 1, name: str = None):
        # 核对单元形状与一次/二次支持范围，初始化共享自由度，并准备参考基函数缓存。
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
        # 给出当前空间使用的参考单元类别，供积分及参考基函数接口识别。
        return self.mesh.element_type

    def _is_tensor(self) -> bool:
        # 区分张量积单元与单纯形，两者使用不同的参考基函数和求积阶规则。
        return self.mesh.element_type in _TENSOR

    def _local_descriptors(self):
        # 取得局部自由度依附的顶点、棱、面或内部位置，作为全局编号的几何依据。
        if self._is_tensor():
            return tensor_local_descriptors(self.mesh.dim, self.degree)
        return simplex_local_descriptors(self.mesh.element_type, self.degree)

    def reference_nodes(self, cell_idx) -> np.ndarray:
        # 返回局部自由度的参考位置或代表位置，积分矩的代表位置仅用于几何描述。
        if self._is_tensor():
            multi = tensor_multi_indices(self.mesh.dim, self.degree)
            return np.array(multi, dtype=float) / float(self.degree)
        return simplex_reference_nodes(self.mesh.element_type, self.degree)

    # ------------------------------------------------------------------
    # 基函数
    # ------------------------------------------------------------------
    def eval_basis(self, xi):
        # 提供参考点上的基函数值接口；是否可以显式计算由具体离散空间决定。
        return self._basis(xi)[0]

    def eval_basis_grad(self, xi):
        # 提供参考基函数导数接口，实际梯度还需结合单元几何变换。
        return self._basis(xi)[1]

    def _basis(self, xi):
        # 按当前单元族和阶数计算参考基函数数据，统一底层基函数的调用方式。
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
        # 在一个参考点计算函数值与梯度，为缓存多点求积数据提供基本操作。
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
    # 工作块：建立自由度的局部与全局对应关系
    # 遍历单元并登记其几何支撑上的自由度；后续组装依赖这张映射表，把局部计算结果放到正确的全局位置。
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
        # 为基函数乘积积分选择求积阶，保证质量项不会只沿用刚度项所需的较低阶数。
        if self._is_tensor():
            return 2 * self.degree + 2
        return max(1, 2 * self.degree)

    def load_quad_order(self) -> int:
        # 为源项与基函数的乘积选择求积阶，给非多项式源函数保留更高的积分精度。
        if self._is_tensor():
            return 2 * self.degree + 4
        return max(2, 2 * self.degree + 2)

    # 工作块：预计算并缓存参考积分数据
    # 同一种参考单元在不同实际单元上使用相同基函数，因此先计算求积点、权重、函数值和参考梯度，再按积分阶缓存，避免重复求值。
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

    # 工作块：准备每个求积点的实际几何信息
    # 将参考点映射到实际单元，逐点计算雅可比和行列式绝对值；返回物理坐标、梯度变换所需矩阵及积分缩放量。
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

    # 工作块：计算一个单元的扩散刚度
    # 读取参考基函数梯度，在各求积点通过雅可比转换到物理梯度；对梯度内积乘权重和 |det J| 累加，得到局部矩阵。
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

    # 工作块：计算一个单元的质量矩阵
    # 在求积点计算各基函数两两乘积，再乘积分权重和几何缩放求和；矩阵描述解函数本身的积分耦合，而不是梯度耦合。
    def local_mass(self, cell_idx) -> np.ndarray:
        """单元质量矩阵。"""
        pts, w, N, _ = self._quadrature_data(self.mass_quad_order())
        _, _, dets = self._element_geometry(cell_idx, pts)
        nl = N.shape[1]
        Me = np.zeros((nl, nl))
        for q in range(len(w)):
            Me += (w[q] * dets[q]) * np.outer(N[q], N[q])
        return Me

    # 工作块：计算一个单元的源项载荷
    # 把求积点映射到实际单元以调用源函数 f，再将函数值乘基函数、权重和几何缩放，得到每个局部自由度的右端贡献。
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
        # 将所有局部刚度贡献组装为全局矩阵；混合空间若不适用此标量接口会明确拒绝。
        return self._assemble(self.local_stiffness, symmetric=True)

    def mass_matrix(self):
        # 将局部质量贡献组装为全局矩阵；特殊空间的块结构由其专用接口提供。
        return self._assemble(self.local_mass, symmetric=True)

    def load_vector(self, f, order=None):
        # 按单元计算源项载荷，再按全局自由度编号累加成右端向量。
        return self._assemble_vector(lambda c: self.local_load(c, f, order))

    # ------------------------------------------------------------------
    def interpolate(self, f) -> np.ndarray:
        """按自由度坐标做 Lagrange 插值。"""
        return self._evaluate_function(f, self.dof_coords)

    # ------------------------------------------------------------------
    # 自由度 → 函数
    # ------------------------------------------------------------------
    # 工作块：从物理位置找回参考坐标
    # 求值接口拿到的是实际坐标，而基函数定义在参考单元上；先反解坐标映射，才能在正确位置计算近似解。
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
        # 取出当前单元的自由度，通过 L2 投影得到多项式系数，再在指定物理点计算函数值。
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
        # 由局部自由度重构采样点的物理梯度；若该空间没有此标量梯度接口则明确报错。
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
        # 返回参考单元所在的坐标维数，供基函数、映射和积分规则选择使用。
        return reference_dimension(self.reference_element_type())
