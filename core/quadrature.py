"""求积规则 (quadrature rules)。

参考单元与 guidebook 第 4.3 节保持一致：

===================  ==========================
单元类型              参考单元
===================  ==========================
``interval``         :math:`[0,1]`
``triangle``         :math:`\\{\\xi\\ge0,\\eta\\ge0,\\xi+\\eta\\le1\\}`
``quad``             :math:`[0,1]^2`
``tet``              标准四面体
``hex``              :math:`[0,1]^3`
===================  ==========================

用法::

    from MosicaFE.core.quadrature import QuadratureRule
    quad = QuadratureRule.get_quadrature("triangle", order=3)
    print(quad.n_points, quad.weights.sum())   # 4 0.5
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "QuadratureRule",
    "get_quadrature",
    "reference_dimension",
    "simplex_quadrature",
    "cell_quadrature",
    "TENSOR_TYPES",
]

#: 张量积型（参考单元是方体）的单元类型
TENSOR_TYPES = ("interval", "quad", "hex")

_REF_DIM = {
    "interval": 1,
    "triangle": 2,
    "quad": 2,
    "tet": 3,
    "hex": 3,
    "polygon": 2,
    "polyhedron": 3,
    "tri": 2,
    "tetra": 3,
}


def reference_dimension(element_type: str) -> int:
    """参考单元的空间维数。"""
    try:
        return _REF_DIM[str(element_type).lower()]
    except KeyError:  # pragma: no cover - 用户输入错误
        raise ValueError(
            f"未知单元类型 {element_type!r}；可用：{sorted(_REF_DIM)}"
        ) from None


def _gauss_legendre_1d(n: int):
    """[0,1] 上的 n 点 Gauss-Legendre 节点与权重。"""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _tensor_product(rule_1d, dim: int):
    """把 1D 规则张量积成 dim 维规则。"""
    nodes, weights = rule_1d
    grids = np.meshgrid(*([nodes] * dim), indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)
    w = weights.copy()
    for _ in range(dim - 1):
        w = np.outer(w, weights).ravel()
    return points, w


class QuadratureRule:
    """参考单元上的求积规则。"""

    def __init__(self, points, weights, element_type: str, order: int):
        self.points = np.atleast_2d(np.asarray(points, dtype=float))
        self.weights = np.asarray(weights, dtype=float).ravel()
        self.element_type = str(element_type).lower()
        self.order = int(order)

    # -- 基本信息 --------------------------------------------------------
    @property
    def n_points(self) -> int:
        """求积点个数。"""
        return int(self.points.shape[0])

    @property
    def dim(self) -> int:
        """参考单元维数。"""
        return int(self.points.shape[1])

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:  # pragma: no cover - 仅展示
        return (
            f"<QuadratureRule {self.element_type} order={self.order} "
            f"n_points={self.n_points} measure={self.weights.sum():.6g}>"
        )

    # -- 工厂 ------------------------------------------------------------
    @staticmethod
    def get_quadrature(element_type: str, order: int = 2) -> "QuadratureRule":
        """按单元类型与多项式精确阶返回求积规则。

        Parameters
        ----------
        element_type : {"interval","triangle","quad","tet","hex",...}
        order : int
            需被精确积分的**多项式总次数**。例如刚度矩阵含梯度内积，
            P1 单元需 ``order >= 0``，P2 单元需 ``order >= 2``。
        """
        et = str(element_type).lower()
        order = max(int(order), 1)

        if et == "interval":
            n = max(1, int(np.ceil((order + 1) / 2)))
            pts, w = _gauss_legendre_1d(n)
            return QuadratureRule(pts[:, None], w, et, order)

        if et in ("tri", "triangle"):
            return QuadratureRule(*_triangle_rule(order), et, order)

        if et in ("tet", "tetra"):
            return QuadratureRule(*_tet_rule(order), et, order)

        if et == "quad":
            n = max(1, int(np.ceil((order + 1) / 2)))
            pts, w = _tensor_product(_gauss_legendre_1d(n), 2)
            return QuadratureRule(pts, w, et, order)

        if et == "hex":
            n = max(1, int(np.ceil((order + 1) / 2)))
            pts, w = _tensor_product(_gauss_legendre_1d(n), 3)
            return QuadratureRule(pts, w, et, order)

        if et in ("polygon", "polyhedron"):
            # VEM 单元不做参考单元数值积分，取形心单点规则占位
            d = reference_dimension(et)
            return QuadratureRule(np.zeros((1, d)), [1.0], et, order)

        raise ValueError(f"未知单元类型 {element_type!r}")

    # guidebook 风格别名
    get = get_quadrature


# ---------------------------------------------------------------------------
# 具体规则表
# ---------------------------------------------------------------------------
def _triangle_rule(order: int):
    """参考三角形（面积 1/2）上的求积规则，精确到给定 total degree。"""
    if order <= 1:
        pts = [[1 / 3, 1 / 3]]
        w = [0.5]
    elif order == 2:
        pts = [[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]
        w = [1 / 6, 1 / 6, 1 / 6]
    elif order == 3:
        pts = [[1 / 3, 1 / 3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]]
        w = [-27 / 96, 25 / 96, 25 / 96, 25 / 96]
    else:
        a = 0.445948490915965
        b = 0.091576213509771
        wa = 0.223381589678011 / 2
        wb = 0.109951743655322 / 2
        pts = [
            [a, a],
            [1 - 2 * a, a],
            [a, 1 - 2 * a],
            [b, b],
            [1 - 2 * b, b],
            [b, 1 - 2 * b],
        ]
        w = [wa, wa, wa, wb, wb, wb]
    return np.array(pts, dtype=float), np.array(w, dtype=float)


def _tet_rule(order: int):
    """参考四面体（体积 1/6）上的求积规则。"""
    if order <= 1:
        pts = [[0.25, 0.25, 0.25]]
        w = [1 / 6]
    elif order == 2:
        a = 0.585410196624969
        b = 0.138196601125011
        pts = [[a, b, b], [b, a, b], [b, b, a], [b, b, b]]
        w = [1 / 24] * 4
    else:
        pts = [[0.25, 0.25, 0.25], [0.5, 1 / 6, 1 / 6],
               [1 / 6, 0.5, 1 / 6], [1 / 6, 1 / 6, 0.5], [1 / 6, 1 / 6, 1 / 6]]
        w = [-0.8 / 6] + [0.45 / 6] * 4
    return np.array(pts, dtype=float), np.array(w, dtype=float)


def get_quadrature(element_type: str, order: int = 2) -> QuadratureRule:
    """``QuadratureRule.get_quadrature`` 的函数式别名。"""
    return QuadratureRule.get_quadrature(element_type, order)


def simplex_quadrature(vertices, order: int = 2):
    """任意单纯形上的求积规则（含**嵌入**的低维单纯形）。

    参考单元规则由顶点个数决定：2 个顶点用区间规则、3 个用三角形规则、
    4 个用四面体规则，然后按仿射映射映到物理单纯形上。

    与 :func:`cell_quadrature` 的区别在于本函数不要求单纯形是满维的：

    * 三角形的一条**边**（3D 空间中的 2 个点）、四面体的一个**三角面**
      （3D 空间中的 3 个点）都可以直接积分——正是计算
      :math:`\\int_f q\\cdot n\\,\\mathrm ds` 这类边界/面泛函所需要的。

    Parameters
    ----------
    vertices : (n_vertices, d) array_like
        ``d`` 是**环境**维数，``n_vertices`` 是单纯形自身的顶点数。
    order : int
        需要精确积分的多项式总次数。

    Returns
    -------
    points : (n_q, d) ndarray
    weights : (n_q,) ndarray
        已乘上单纯形测度（长度 / 面积 / 体积）。
    """
    v = np.atleast_2d(np.asarray(vertices, dtype=float))
    dim_int = v.shape[0] - 1
    if dim_int < 1:
        raise ValueError("单纯形至少需要 2 个顶点")
    ref_type = {1: "interval", 2: "triangle", 3: "tet"}.get(dim_int)
    if ref_type is None:
        raise ValueError(f"只支持 1/2/3 维单纯形，收到 {dim_int} 维")

    rule = QuadratureRule.get_quadrature(ref_type, order)
    # 仿射映射 x = v0 + J xi，J 的列是 v_i - v_0（可能与满维不同）
    J = np.array([v[i + 1] - v[0] for i in range(dim_int)])     # (dim_int, d)
    points = v[0] + rule.points @ J                             # (n_q, d)
    gram = J @ J.T                                              # (dim_int, dim_int)
    scale = float(np.sqrt(abs(np.linalg.det(gram))))
    return points, rule.weights * scale


def cell_quadrature(mesh, cell_idx: int, order: int = 3):
    """把参考单元规则映射到第 cell_idx 个**物理单元**上。

    对任意多边形 / 多面体（VEM 网格）会自动从形心做扇形三角化 / 四面体化，
    因此本函数对三种方法（FEM / VEM / 混合）都可用。

    Returns
    -------
    points : (nq, dim) ndarray
    weights : (nq,) ndarray
        已含 ``|detJ|`` 的物理权重。
    """
    from .mesh import Mesh  # noqa: F401  (类型提示用，避免循环导入)
    from .utils import (
        HEX_FACES,
        TET_FACES,
        apply_affine_map,
        cell_centroid,
        compute_jacobian,
        simplex_jacobian,
    )

    coords = mesh.get_cell_vertices(cell_idx)
    et = mesh.element_type
    d = mesh.dim

    if et in ("interval", "triangle", "quad", "tet", "hex"):
        rule = QuadratureRule.get_quadrature(et, order)
        phys = apply_affine_map(rule.points, coords)
        weights = np.zeros(rule.n_points)
        for q, xi in enumerate(rule.points):
            _, detJ = compute_jacobian(coords, xi)
            weights[q] = rule.weights[q] * abs(detJ)
        return phys, weights

    if d == 2:
        n = len(coords)
        centroid = cell_centroid(coords)
        rule = QuadratureRule.get_quadrature("triangle", order)
        pts = []
        wts = []
        for i in range(n):
            tri = np.array([centroid, coords[i], coords[(i + 1) % n]])
            J, detJ = simplex_jacobian(tri)
            pts.append(tri[0] + rule.points @ J.T)
            wts.append(rule.weights * abs(detJ))
        return np.vstack(pts), np.concatenate(wts)

    # 3D 多面体
    centroid = cell_centroid(coords)
    faces = None
    if len(coords) == 4:
        faces = TET_FACES
    elif len(coords) == 8:
        faces = HEX_FACES
    elif getattr(mesh, "_faces", None) is not None:
        faces = mesh._faces
    if faces is None:
        raise ValueError(
            "无法自动为 3D 多面体单元生成求积点；请通过 Mesh(..., faces=...) "
            "提供有序的面（每个面是顶点局部编号列表）。"
        )

    rule = QuadratureRule.get_quadrature("tet", order)
    pts = []
    wts = []
    for f in faces:
        fv = [int(i) for i in f]
        tris = [(fv[0], fv[1], fv[2])]
        if len(fv) == 4:
            tris.append((fv[0], fv[2], fv[3]))
        for tri in tris:
            tet = np.array([centroid, coords[tri[0]], coords[tri[1]], coords[tri[2]]])
            J, detJ = simplex_jacobian(tet)
            pts.append(tet[0] + rule.points @ J.T)
            wts.append(rule.weights * abs(detJ))
    return np.vstack(pts), np.concatenate(wts)
