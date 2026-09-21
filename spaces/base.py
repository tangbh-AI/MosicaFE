"""有限元空间抽象基类。

设计原则（见 progress/DECISIONS.md D3）：空间向上只暴露**矩阵级**接口，

* :meth:`stiffness_matrix`  —— :math:`A_{ij}=\\int \\nabla\\phi_j\\cdot\\nabla\\phi_i`
* :meth:`mass_matrix`       —— :math:`M_{ij}=\\int \\phi_j\\phi_i`
* :meth:`load_vector`       —— :math:`b_i=\\int f\\phi_i`
* :meth:`boundary_dofs`     —— Dirichlet 自由度

这样物理层（`physics`）只需要做矩阵的线性组合，新增 PDE 时不必重复写单元循环。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .._compat import from_triplets
from ..core.quadrature import QuadratureRule, reference_dimension

__all__ = ["BaseFESpace"]


class BaseFESpace(ABC):
    """所有离散空间的公共接口。"""

    family = "base"
    has_explicit_basis = False

    def __init__(self, mesh, degree: int = 1, name: str = None):
        self.mesh = mesh
        self.degree = int(degree)
        self.name = name or f"{self.family}{self.degree}"
        self.n_dofs = 0
        self.dof_map = None          # (n_cells, n_local) 或变长 list
        self.dof_coords = None       # (n_dofs, dim)
        self.dof_support = []        # 每个自由度支撑的全局顶点编号元组
        self.dof_is_interior = np.zeros(0, dtype=bool)
        self._quad_cache = {}
        self._build()

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    def _build(self):
        self._build_dof_map()
        self._build_dof_coordinates()
        self._finalize_support()

    @abstractmethod
    def _build_dof_map(self):
        """建立 ``dof_map``（单元局部自由度 → 全局自由度）与 ``n_dofs``。"""

    def _build_dof_coordinates(self):
        """为每个自由度算一个代表坐标（用于插值与边界判定）。"""
        coords = np.zeros((self.n_dofs, self.mesh.dim))
        counts = np.zeros(self.n_dofs)
        for c in range(self.mesh.n_cells):
            dofs = self.get_cell_dofs(c)
            xi = self.reference_nodes(c)
            from ..core.utils import apply_affine_map

            phys = apply_affine_map(xi, self.mesh.get_cell_vertices(c))
            for k, dof in enumerate(dofs):
                coords[dof] += phys[k]
                counts[dof] += 1
        counts[counts == 0] = 1
        self.dof_coords = coords / counts[:, None]

    def _finalize_support(self):
        if len(self.dof_support) != self.n_dofs:
            self.dof_support = [() for _ in range(self.n_dofs)]
        if len(self.dof_is_interior) != self.n_dofs:
            self.dof_is_interior = np.zeros(self.n_dofs, dtype=bool)

    def reference_nodes(self, cell_idx):
        """该单元局部自由度的参考坐标 ``(n_local, dim)``。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 自由度查询
    # ------------------------------------------------------------------
    def get_cell_dofs(self, cell_idx: int) -> np.ndarray:
        """第 cell_idx 个单元的全局自由度编号。"""
        return np.asarray(self.dof_map[cell_idx], dtype=np.int64)

    def n_cell_dofs(self, cell_idx: int) -> int:
        return int(len(self.dof_map[cell_idx]))

    def get_dof_coordinates(self, dof: int) -> np.ndarray:
        """第 dof 个自由度的物理坐标。"""
        return self.dof_coords[dof]

    # guidebook 中使用的私有名
    def _get_dof_coordinates(self, dof: int) -> np.ndarray:
        return self.get_dof_coordinates(dof)

    def boundary_dofs(self) -> np.ndarray:
        """位于 Dirichlet 边界上的自由度编号（升序）。"""
        if not len(self.dof_support):
            return np.zeros(0, dtype=np.int64)
        bset = set(int(v) for v in self.mesh.boundary_nodes)
        out = [
            d
            for d in range(self.n_dofs)
            if (not self.dof_is_interior[d])
            and len(self.dof_support[d]) > 0
            and all(v in bset for v in self.dof_support[d])
        ]
        return np.array(out, dtype=np.int64)

    def dof_boundary_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_dofs, dtype=bool)
        mask[self.boundary_dofs()] = True
        return mask

    # ------------------------------------------------------------------
    # 求积
    # ------------------------------------------------------------------
    def reference_element_type(self) -> str:
        """对应的参考单元类型。"""
        et = self.mesh.element_type
        if et in ("polygon", "polyhedron"):
            return et
        return et

    def get_quadrature(self, order: int = None) -> QuadratureRule:
        """返回（并缓存）参考单元上的求积规则。"""
        if order is None:
            order = self.default_quad_order()
        order = int(order)
        if order not in self._quad_cache:
            self._quad_cache[order] = QuadratureRule.get_quadrature(
                self.reference_element_type(), order
            )
        return self._quad_cache[order]

    def default_quad_order(self) -> int:
        """刚度矩阵默认使用的求积阶（对总次数而言）。"""
        return max(1, 2 * (self.degree - 1))

    def mass_quad_order(self) -> int:
        return max(1, 2 * self.degree)

    def load_quad_order(self) -> int:
        return max(2, 2 * self.degree + 2)

    # ------------------------------------------------------------------
    # 基函数（仅显式基函数空间可用）
    # ------------------------------------------------------------------
    def eval_basis(self, xi):
        raise NotImplementedError(f"{type(self).__name__} 没有显式基函数")

    def eval_basis_grad(self, xi):
        raise NotImplementedError(f"{type(self).__name__} 没有显式基函数")

    # ------------------------------------------------------------------
    # 矩阵级接口
    # ------------------------------------------------------------------
    @abstractmethod
    def stiffness_matrix(self):
        """刚度矩阵 :math:`\\int\\nabla\\phi_j\\cdot\\nabla\\phi_i`。"""

    @abstractmethod
    def mass_matrix(self):
        """质量矩阵 :math:`\\int\\phi_j\\phi_i`。"""

    @abstractmethod
    def load_vector(self, f):
        """载荷向量 :math:`\\int f\\phi_i`；``f`` 是接收点坐标的标量函数。"""

    @abstractmethod
    def interpolate(self, f):
        """把标量函数 ``f`` 插值到空间的自由度上。"""

    # ------------------------------------------------------------------
    # 自由度 → 函数（误差分析 / 可视化使用）
    # ------------------------------------------------------------------
    def evaluate(self, cell_idx: int, values, points):
        """在单元 ``cell_idx`` 内把自由度向量 ``values`` 重构成函数并在点上求值。

        返回 ``(n_points,)``。子类必须实现；这是 FEM 与 VEM 在
        "解函数能不能在任意点求值"上的接缝（VEM 用最小二乘线性重构）。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 未实现 evaluate()"
        )

    def evaluate_gradient(self, cell_idx: int, values, points):
        """在单元内求重构解的梯度，返回 ``(n_points, dim)``。"""
        raise NotImplementedError(
            f"{type(self).__name__} 未实现 evaluate_gradient()"
        )

    def nodal_values(self, values) -> np.ndarray:
        """把自由度向量还原成**网格节点上的值**（可视化 / 后处理用）。

        做法：在每个单元内用 :meth:`evaluate` 在单元顶点处求值，再按节点平均。
        对 Lagrange 空间这就是精确的节点值；对 VEM 是最小二乘重构值。
        """
        values = np.asarray(values, dtype=float).ravel()
        out = np.zeros(self.mesh.n_nodes)
        cnt = np.zeros(self.mesh.n_nodes)
        for c in range(self.mesh.n_cells):
            coords = self.mesh.get_cell_vertices(c)
            vals = self.evaluate(c, values, coords)
            for k, v in enumerate(self.mesh.elements[c]):
                np.add.at(out, v, vals[k])
                np.add.at(cnt, v, 1.0)
        cnt[cnt == 0] = 1
        return out / cnt

    # ------------------------------------------------------------------
    # 组装工具
    # ------------------------------------------------------------------
    def _assemble(self, local_matrix, *, symmetric=False, desc=""):
        """逐单元调用 ``local_matrix(cell)`` 并累加到全局稀疏矩阵。"""
        n = self.n_dofs
        rows = []
        cols = []
        vals = []
        for c in range(self.mesh.n_cells):
            dofs = self.get_cell_dofs(c)
            Ke = local_matrix(c)
            nl = len(dofs)
            rows.append(np.repeat(dofs, nl))
            cols.append(np.tile(dofs, nl))
            vals.append(np.asarray(Ke, dtype=float).ravel())
        rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
        vals = np.concatenate(vals) if vals else np.zeros(0)
        A = from_triplets(rows, cols, vals, (n, n))
        if symmetric:
            A = 0.5 * (A + A.T)
        return A

    def _assemble_vector(self, local_vector):
        b = np.zeros(self.n_dofs)
        for c in range(self.mesh.n_cells):
            dofs = self.get_cell_dofs(c)
            # 必须用 np.add.at：dofs 里可能出现重复下标（网格存在重复顶点时），
            # b[dofs] += ... 会因为 numpy 的缓冲赋值而漏掉重复项的累加。
            np.add.at(b, dofs, np.asarray(local_vector(c), dtype=float))
        return b

    def _evaluate_function(self, f, points: np.ndarray) -> np.ndarray:
        """把 ``f`` 作用到 ``(n_points, dim)`` 的坐标数组上，返回 ``(n_points,)``。"""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        try:
            out = f(points)
            out = np.asarray(out, dtype=float)
            if out.shape == points.shape:
                # f 是逐分量函数（如 f(x,y) = 3x + y），做标量求和
                out = out.sum(axis=1)
            if out.shape == (points.shape[0],):
                return out
        except Exception:
            pass
        return np.array([float(f(p)) for p in points])

    def solve(self, *args, **kwargs):  # pragma: no cover - 便于交互探查
        raise NotImplementedError(
            "空间本身不求解；请构造 physics 中的问题对象后调用 problem.solve()"
        )

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------
    def __repr__(self):  # pragma: no cover - 仅展示
        return (
            f"<{type(self).__name__} {self.name} dim={self.mesh.dim} "
            f"degree={self.degree} n_dofs={self.n_dofs} "
            f"cells={self.mesh.n_cells}>"
        )

    def summary(self) -> str:
        return repr(self)

    # 便于外部按需拿到参考维数
    @property
    def dim(self) -> int:
        return int(self.mesh.dim)

    @property
    def ref_dim(self) -> int:
        return reference_dimension(self.reference_element_type())
