"""参考单元上的基函数、局部自由度布局。

本模块把"参考单元上有什么"与"全局自由度怎么编号"分开：这里只管前者，
后者由 :mod:`MosicaFE.spaces.lagrange` 中的自由度管理器负责。

局部自由度的描述符 (descriptor) 有四种：

``("vertex", (a, b, c))``
    顶点型自由度，参数为该顶点在参考单元上的坐标。
``("edge", (corner_i, corner_j))``
    边内部自由度，用参考单元的两个角点局部编号表示。
``("face", (c0, c1, c2, c3))``
    面内部自由度（3D 张量积单元），用面的 4 个角点局部编号表示。
``("interior", (i, j, k))``
    单元内部自由度，只属于本单元。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "simplex_basis",
    "tensor_basis",
    "simplex_local_descriptors",
    "tensor_local_descriptors",
    "simplex_reference_nodes",
    "TENSOR_CORNERS",
    "HEX_EDGES",
]

# 参考单元角点顺序（张量积单元）
TENSOR_CORNERS = {
    1: [(0,), (1,)],
    2: [(0, 0), (1, 0), (1, 1), (0, 1)],
    3: [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ],
}

# 六面体的 12 条棱（用角点局部编号表示）
HEX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# 六面体的 6 个面（从单元外部看逆时针）
HEX_FACES_LOCAL = [
    (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
    (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
]

_QUAD_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]

# 单纯形的棱（局部顶点编号）
_SIMPLEX_EDGES = {
    "triangle": [(0, 1), (1, 2), (2, 0)],
    "tet": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
}


# ---------------------------------------------------------------------------
# 基函数
# ---------------------------------------------------------------------------
def _simplex_lambdas(element_type: str, xi: np.ndarray):
    """参考单纯形上的重心坐标及其（参考）梯度。"""
    xi = np.asarray(xi, dtype=float).ravel()
    if element_type == "triangle":
        a, b = xi
        lam = np.array([1.0 - a - b, a, b])
        grad = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    elif element_type == "tet":
        a, b, c = xi
        lam = np.array([1.0 - a - b - c, a, b, c])
        grad = np.array(
            [
                [-1.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 1.0],
            ]
        )
    else:
        raise ValueError(f"不支持的单纯形类型 {element_type!r}")
    return lam, grad


def simplex_basis(element_type: str, degree: int, xi):
    """单纯形 Lagrange 基函数在参考点 xi 处的值与梯度。

    自由度顺序：先全部顶点，再全部棱中点（与 guidebook 第 5.2.2 节一致：
    三角形为 ``e01, e12, e20``）。

    Returns
    -------
    values : (n_local,) ndarray
    grads : (n_local, dim) ndarray
    """
    lam, dlam = _simplex_lambdas(element_type, xi)
    n_v = len(lam)
    dim = dlam.shape[0]
    values = []
    grads = []

    if degree == 1:
        for i in range(n_v):
            values.append(lam[i])
            grads.append(dlam[:, i])
    elif degree == 2:
        for i in range(n_v):
            values.append(lam[i] * (2.0 * lam[i] - 1.0))
            grads.append((4.0 * lam[i] - 1.0) * dlam[:, i])
        for (i, j) in _SIMPLEX_EDGES[element_type]:
            values.append(4.0 * lam[i] * lam[j])
            grads.append(4.0 * (lam[j] * dlam[:, i] + lam[i] * dlam[:, j]))
    else:
        raise NotImplementedError(
            f"{element_type} 上仅实现了 degree=1, 2（收到 degree={degree}）"
        )

    return np.asarray(values, dtype=float), np.asarray(grads, dtype=float).reshape(
        len(values), dim
    )


def _lagrange_1d(p: int, t: float):
    """[0,1] 上等距节点 ``i/p`` 的 1D Lagrange 基与其导数。"""
    nodes = np.arange(p + 1) / p
    values = np.ones(p + 1)
    grads = np.zeros(p + 1)
    for i in range(p + 1):
        for j in range(p + 1):
            if j == i:
                continue
            values[i] *= (t - nodes[j]) / (nodes[i] - nodes[j])
        acc = 0.0
        for k in range(p + 1):
            if k == i:
                continue
            term = 1.0 / (nodes[i] - nodes[k])
            for j in range(p + 1):
                if j == i or j == k:
                    continue
                term *= (t - nodes[j]) / (nodes[i] - nodes[j])
            acc += term
        grads[i] = acc
    return values, grads


def tensor_basis(dim: int, degree: int, xi):
    """张量积 Lagrange 基函数在参考点 xi 处的值与梯度。

    局部自由度顺序为 :func:`tensor_local_descriptors` 给出的顺序。
    """
    xi = np.asarray(xi, dtype=float).ravel()
    one_d = [_lagrange_1d(degree, xi[d]) for d in range(dim)]
    multi = tensor_multi_indices(dim, degree)

    values = []
    grads = []
    for idx in multi:
        val = 1.0
        for d in range(dim):
            val *= one_d[d][0][idx[d]]
        values.append(val)
        g = np.zeros(dim)
        for d in range(dim):
            term = one_d[d][1][idx[d]]
            for e in range(dim):
                if e != d:
                    term *= one_d[e][0][idx[e]]
            g[d] = term
        grads.append(g)
    return np.asarray(values, dtype=float), np.asarray(grads, dtype=float)


# ---------------------------------------------------------------------------
# 局部自由度布局
# ---------------------------------------------------------------------------
def _tensor_index_and_kind(dim: int, degree: int):
    """兼容旧名，返回与 :func:`tensor_multi_indices` 相同的多重指标列表。"""
    return tensor_multi_indices(dim, degree)


def tensor_multi_indices(dim: int, degree: int):
    """张量积单元局部自由度的多重指标列表（顺序 = 局部自由度顺序）。

    第 k 个自由度的参考坐标就是 ``multi[k][d] / degree``。
    """
    p = int(degree)
    if p not in (1, 2):
        raise NotImplementedError(
            f"{dim}D 张量积单元仅实现了 degree=1, 2（收到 degree={degree}）"
        )
    corners = TENSOR_CORNERS[dim]
    multi = [tuple(p * c[d] for d in range(dim)) for c in corners]
    if p == 1:
        return multi

    edges = _QUAD_EDGES if dim == 2 else HEX_EDGES
    for (a, b) in edges:
        mid = 0.5 * (np.array(corners[a], dtype=float) + np.array(corners[b], dtype=float))
        multi.append(tuple(int(round(v * p)) for v in mid))

    if dim == 3:
        for f in HEX_FACES_LOCAL:
            ctr = np.array([corners[i] for i in f], dtype=float).mean(axis=0)
            multi.append(tuple(int(round(v * p)) for v in ctr))

    multi.append(tuple([p // 2] * dim))
    return multi


def tensor_local_descriptors(dim: int, degree: int):
    """张量积单元的局部自由度描述符列表（与 :func:`tensor_multi_indices` 对齐）。"""
    p = int(degree)
    corners = TENSOR_CORNERS[dim]
    descriptors = [("vertex", k) for k in range(len(corners))]
    if p == 1:
        return descriptors
    if p != 2:
        raise NotImplementedError(
            f"{dim}D 张量积单元仅实现了 degree=1, 2（收到 degree={degree}）"
        )
    edges = _QUAD_EDGES if dim == 2 else HEX_EDGES
    descriptors += [("edge", e) for e in edges]
    if dim == 3:
        descriptors += [("face", f) for f in HEX_FACES_LOCAL]
    descriptors.append(("interior", tuple([p // 2] * dim)))
    return descriptors


def simplex_local_descriptors(element_type: str, degree: int):
    """单纯形单元的局部自由度描述符列表。"""
    if degree == 1:
        n_v = 3 if element_type == "triangle" else 4
        return [("vertex", k) for k in range(n_v)]
    if degree == 2:
        n_v = 3 if element_type == "triangle" else 4
        return [("vertex", k) for k in range(n_v)] + [
            ("edge", e) for e in _SIMPLEX_EDGES[element_type]
        ]
    raise NotImplementedError(
        f"{element_type} 上仅实现了 degree=1, 2（收到 degree={degree}）"
    )


def simplex_reference_nodes(element_type: str, degree: int):
    """单纯形局部自由度的参考坐标 ``(n_local, dim)``。"""
    if element_type == "triangle":
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    else:
        corners = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    desc = simplex_local_descriptors(element_type, degree)
    pts = []
    for kind, data in desc:
        if kind == "vertex":
            pts.append(corners[data])
        else:
            a, b = data
            pts.append(0.5 * (corners[a] + corners[b]))
    return np.asarray(pts, dtype=float)
