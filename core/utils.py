"""几何工具：单元测度、形心、法向、雅可比与梯度变换。

这里同时提供 guidebook 第 10.3 节自定义物理时用到的三个函数
(``compute_jacobian`` / ``transform_gradient`` / ``apply_affine_map``)，
以便用户按书中的写法扩展新 PDE。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "polygon_area",
    "polygon_centroid",
    "edge_normals_2d",
    "polyhedron_volume",
    "cell_measure",
    "cell_centroid",
    "cell_diameter",
    "simplex_jacobian",
    "tensor_jacobian",
    "apply_affine_map",
    "transform_gradient",
    "compute_jacobian",
]

_EPS = 1e-14

#: 六面体（8 顶点）的 6 个四边形面，顶点顺序为
#: v0..v3 位于 z=0 面逆时针，v4..v7 位于 z=1 面逆时针，且 v_{i+4} 在 v_i 正上方。
#: 每个面按"从单元外部看逆时针"排列。
HEX_FACES = (
    (0, 3, 2, 1),  # z = 0 底面
    (4, 5, 6, 7),  # z = 1 顶面
    (0, 1, 5, 4),  # y = 0
    (1, 2, 6, 5),  # x = 1
    (2, 3, 7, 6),  # y = 1
    (3, 0, 4, 7),  # x = 0
)

#: 四面体（4 顶点）的 4 个三角面，第 f 个面对顶点 f。
TET_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


def hex_faces_triangulated():
    """六面体的面按三角剖分展开，便于面积分。"""
    out = []
    for f in HEX_FACES:
        out.append((f[0], f[1], f[2]))
        out.append((f[0], f[2], f[3]))
    return out


# ---------------------------------------------------------------------------
# 2D 多边形
# ---------------------------------------------------------------------------
def polygon_area(verts: np.ndarray) -> float:
    """多边形面积（鞋带公式），顶点可顺时针或逆时针。"""
    verts = np.asarray(verts, dtype=float)
    x, y = verts[:, 0], verts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def signed_polygon_area(verts: np.ndarray) -> float:
    """带符号面积（逆时针为正）。"""
    verts = np.asarray(verts, dtype=float)
    x, y = verts[:, 0], verts[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def polygon_centroid(verts: np.ndarray) -> np.ndarray:
    """多边形形心（面积加权）；退化时退回顶点平均。"""
    verts = np.asarray(verts, dtype=float)
    x, y = verts[:, 0], verts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = 0.5 * cross.sum()
    if abs(a) < _EPS:
        return verts.mean(axis=0)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * a)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * a)
    return np.array([cx, cy])


def edge_normals_2d(verts: np.ndarray):
    """返回多边形每条边的 (外法向, 边长)。

    约定顶点按逆时针排列，则边 ``(v_i -> v_{i+1})`` 的外法向为
    ``(dy, -dx)/|e|``。
    """
    verts = np.asarray(verts, dtype=float)
    n = len(verts)
    normals = np.zeros((n, 2))
    lengths = np.zeros(n)
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        e = b - a
        length = float(np.linalg.norm(e))
        lengths[i] = length
        if length > _EPS:
            normals[i] = np.array([e[1], -e[0]]) / length
    return normals, lengths


# ---------------------------------------------------------------------------
# 3D 多面体
# ---------------------------------------------------------------------------
def polyhedron_volume(verts: np.ndarray, faces=None) -> float:
    """多面体体积。

    Parameters
    ----------
    verts : (n, 3) 顶点坐标
    faces : 可选的三角面索引列表 ``[(i, j, k), ...]``。若给定，则用
        散度定理的四面体分解精确计算；否则用"心点 + 三角面"的近似分解，
        三角面由所有顶点组合枚举（只适用于凸多面体）。
    """
    verts = np.asarray(verts, dtype=float)
    if faces is None and len(verts) == 8:
        faces = hex_faces_triangulated()
    if faces is not None and len(faces) > 0:
        vol = 0.0
        for f in faces:
            a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
            vol += float(np.dot(a, np.cross(b, c))) / 6.0
        return abs(vol)

    # 退化方案：凸包式四面体分解（顶点 0 作为参考点）
    vol = 0.0
    for i in range(1, len(verts) - 1):
        for j in range(i + 1, len(verts)):
            k = (j + 1) % len(verts)
            v1 = verts[i] - verts[0]
            v2 = verts[j] - verts[0]
            v3 = verts[k] - verts[0]
            vol += abs(float(np.dot(v1, np.cross(v2, v3)))) / 6.0
    return vol


def cell_measure(verts: np.ndarray) -> float:
    """单元测度：2D 返回面积，3D 返回体积，1D 返回长度。"""
    verts = np.asarray(verts, dtype=float)
    dim = verts.shape[1]
    if dim == 1:
        return float(verts[:, 0].max() - verts[:, 0].min())
    if dim == 2:
        return polygon_area(verts)
    if dim == 3 and len(verts) == 4:  # 四面体：精确公式
        a, b, c, d = verts
        return float(abs(np.linalg.det(np.array([b - a, c - a, d - a]))) / 6.0)
    return polyhedron_volume(verts)


def cell_centroid(verts: np.ndarray) -> np.ndarray:
    """单元形心（2D 用面积加权，其余用顶点平均，对规则单元二者一致）。"""
    verts = np.asarray(verts, dtype=float)
    if verts.shape[1] == 2 and len(verts) > 3:
        return polygon_centroid(verts)
    return verts.mean(axis=0)


def cell_diameter(verts: np.ndarray) -> float:
    """单元直径（顶点两两距离的最大值）。"""
    verts = np.asarray(verts, dtype=float)
    diff = verts[:, None, :] - verts[None, :, :]
    return float(np.sqrt((diff ** 2).sum(-1)).max())


# ---------------------------------------------------------------------------
# 雅可比与梯度
# ---------------------------------------------------------------------------
def simplex_jacobian(vertices: np.ndarray):
    """单纯形（线段/三角形/四面体）的仿射映射雅可比。

    ``x = v0 + J xi``，其中 ``J = [v1-v0, ..., vd-v0]``（列向量）。

    Returns
    -------
    J : (d, d) ndarray
    detJ : float  （带符号）
    """
    vertices = np.asarray(vertices, dtype=float)
    v0 = vertices[0]
    J = np.array([vertices[i + 1] - v0 for i in range(vertices.shape[1])]).T
    return J, float(np.linalg.det(J))


def tensor_jacobian(vertices: np.ndarray, xi: np.ndarray):
    """四边形 / 六面体（双线性 / 三线性映射）在参考点 xi 处的雅可比。

    参考单元统一取 ``[0,1]^d``；四边形顶点顺序
    ``(0,0),(1,0),(1,1),(0,1)``；六面体顶点顺序
    ``(0,0,0),(1,0,0),(1,1,0),(0,1,0)`` 之后接对应的 4 个 ``z=1`` 顶点。

    约定与 :func:`simplex_jacobian` 完全一致：**列**是物理坐标对参考坐标的偏导，

    .. math::

       J_{ab}=\\frac{\\partial x_a}{\\partial\\xi_b}
       =\\sum_i \\frac{\\partial N_i}{\\partial\\xi_b}\\,x_{i,a}
       \\qquad\\Longleftrightarrow\\qquad J=V^{\\top}\\,dN .

    这个约定很关键：:func:`transform_gradient`（用 :math:`J^{-T}`）与
    :meth:`MosicaFE.spaces.lagrange.LagrangeSpace.evaluate`（Newton 逆映射，
    用 :math:`J^{-1}`）都要求它。对轴对齐的矩形/长方体 :math:`J` 是对角阵，
    但**旋转或畸变的四边形/六面体**上 :math:`J\\ne J^{\\top}`，
    用错约定会静默算出错误的梯度。

    Returns
    -------
    J : (d, d) ndarray
    detJ : float
    """
    vertices = np.asarray(vertices, dtype=float)
    xi = np.asarray(xi, dtype=float).ravel()
    d = vertices.shape[1]
    if d == 2:
        s, t = xi
        dN = np.array(
            [
                [-(1 - t), -(1 - s)],
                [(1 - t), -s],
                [t, s],
                [-t, (1 - s)],
            ]
        )
    else:
        s, t, u = xi
        dN = np.array(
            [
                [-(1 - t) * (1 - u), -(1 - s) * (1 - u), -(1 - s) * (1 - t)],
                [(1 - t) * (1 - u), -s * (1 - u), -s * (1 - t)],
                [t * (1 - u), s * (1 - u), -s * t],
                [-t * (1 - u), (1 - s) * (1 - u), -(1 - s) * t],
                [-(1 - t) * u, -(1 - s) * u, (1 - s) * (1 - t)],
                [(1 - t) * u, -s * u, s * (1 - t)],
                [t * u, s * u, s * t],
                [-t * u, (1 - s) * u, (1 - s) * t],
            ]
        )
    # dN 的第 i 行是 (∂N_i/∂ξ_1, ..., ∂N_i/∂ξ_d)，故 J = Vᵀ dN
    # （不能用 dNᵀV，那得到的是转置雅可比；行列式不受影响，但梯度会错）
    J = vertices.T @ dN
    return J, float(np.linalg.det(J))


def apply_affine_map(ref_points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """把参考点映到物理单元。

    单纯形（线段/三角形/四面体）用仿射映射 ``x = v0 + J xi``；
    四边形/六面体用 ``[0,1]^d`` 上的双线性 / 三线性映射。
    """
    ref_points = np.atleast_2d(np.asarray(ref_points, dtype=float))
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) in (2, 3, 4) and vertices.shape[1] == len(vertices) - 1:
        J, _ = simplex_jacobian(vertices)
        return vertices[0] + ref_points @ J.T
    return _tensor_map(ref_points, vertices)


def _shape_functions_tensor(xi, d):
    xi = np.asarray(xi, dtype=float).ravel()
    if d == 2:
        s, t = xi
        return np.array(
            [(1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t]
        )
    s, t, u = xi
    return np.array(
        [
            (1 - s) * (1 - t) * (1 - u),
            s * (1 - t) * (1 - u),
            s * t * (1 - u),
            (1 - s) * t * (1 - u),
            (1 - s) * (1 - t) * u,
            s * (1 - t) * u,
            s * t * u,
            (1 - s) * t * u,
        ]
    )


def _tensor_map(ref_points, vertices):
    vertices = np.asarray(vertices, dtype=float)
    d = vertices.shape[1]
    out = np.zeros((len(ref_points), d))
    for q, xi in enumerate(ref_points):
        out[q] = _shape_functions_tensor(xi, d) @ vertices
    return out


def transform_gradient(ref_grad: np.ndarray, J: np.ndarray) -> np.ndarray:
    """把参考单元上的梯度变换到物理单元：``grad_x = J^{-T} grad_xi``。"""
    ref_grad = np.asarray(ref_grad, dtype=float)
    J = np.asarray(J, dtype=float)
    return np.linalg.solve(J.T, ref_grad.T).T


def compute_jacobian(vertices: np.ndarray, xi=None):
    """统一入口：返回 (J, detJ)。

    单纯形用仿射雅可比，四边形/六面体用 ``xi`` 处的三线性雅可比。
    默认 ``xi`` 取参考单元中心 ``(0.5, …, 0.5)``（对平行四边形 / 长方体，
    雅可比与 ``xi`` 无关，取中心与取角点结果相同）。

    两种单元的 ``J`` 用同一约定：``J[a,b] = ∂x_a/∂ξ_b``，
    可直接交给 :func:`transform_gradient`。
    """
    vertices = np.asarray(vertices, dtype=float)
    d = vertices.shape[1]
    if vertices.shape[0] == d + 1:
        return simplex_jacobian(vertices)
    if xi is None:
        xi = np.full(d, 0.5)
    return tensor_jacobian(vertices, xi)
