"""MosicaFE 可选依赖的统一入口。

MosicaFE 的硬依赖只有 numpy。scipy（稀疏矩阵与 Krylov 求解器）与
matplotlib（绘图）都是可选的：

* 装了就自动使用，走 scipy.sparse 的 CSR 与 spsolve/cg/gmres/...；
* 没装就退化成 numpy 稠密实现，小规模问题照常能跑。

本模块把这些差异收敛到一处，其它模块一律 from MosicaFE._compat import ...
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "HAS_SCIPY",
    "HAS_MATPLOTLIB",
    "sparse",
    "spsolve",
    "splu",
    "cg",
    "gmres",
    "bicgstab",
    "minres",
    "from_triplets",
    "bmat",
    "to_dense",
    "is_sparse",
    "matplotlib_or_none",
]

# ---------------------------------------------------------------------------
# scipy
# ---------------------------------------------------------------------------
try:
    import scipy.sparse as sparse
    from scipy.sparse.linalg import bicgstab as _bicgstab
    from scipy.sparse.linalg import cg as _cg
    from scipy.sparse.linalg import gmres as _gmres
    from scipy.sparse.linalg import minres as _minres
    from scipy.sparse.linalg import splu as _splu
    from scipy.sparse.linalg import spsolve as _spsolve

    HAS_SCIPY = True
except Exception:
    sparse = None
    _bicgstab = _cg = _gmres = _minres = _splu = _spsolve = None
    HAS_SCIPY = False

cg = _cg
gmres = _gmres
bicgstab = _bicgstab
minres = _minres
splu = _splu
spsolve = _spsolve


# ---------------------------------------------------------------------------
# matplotlib
# ---------------------------------------------------------------------------
try:
    import matplotlib  # noqa: F401

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


def matplotlib_or_none():
    """返回 matplotlib.pyplot；未安装时返回 None。"""
    if not HAS_MATPLOTLIB:
        return None
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------
# 稀疏 / 稠密统一接口
# ---------------------------------------------------------------------------
def from_triplets(rows, cols, vals, shape, tocsr=True):
    """由 COO 三元组构造矩阵。

    scipy 可用时返回 CSR 稀疏矩阵，否则返回 numpy 稠密数组。
    重复下标自动相加，语义与 scipy.sparse.coo_matrix 一致。
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=float)

    if HAS_SCIPY:
        A = sparse.coo_matrix((vals, (rows, cols)), shape=shape)
        return A.tocsr() if tocsr else A

    A = np.zeros(shape, dtype=float)
    if rows.size:
        np.add.at(A, (rows, cols), vals)
    return A


def to_dense(A):
    """统一转稠密 ndarray。"""
    if isinstance(A, np.ndarray):
        return np.asarray(A, dtype=float)
    if HAS_SCIPY and sparse.issparse(A):
        return np.asarray(A.todense(), dtype=float)
    return np.asarray(A, dtype=float)


def bmat(blocks):
    """由块拼装矩阵，``None`` 表示零块（鞍点系统的标准写法）。

    scipy 可用且至少有一个稀疏块时返回 CSR 稀疏矩阵，否则返回稠密 ndarray。
    与 :func:`from_triplets` 一样，把"稀疏 / 稠密"的差异收敛在这一层：
    物理层可以放心地写 ``bmat([[M, -B.T], [B, None]])``。

    .. code-block:: python

       A = bmat([[M, -B.T],
                 [B, None]])      # [[M, -B^T], [B, 0]]
    """
    rows = [list(row) for row in blocks]
    n_rows = len(rows)
    n_cols = len(rows[0])
    if any(len(r) != n_cols for r in rows):
        raise ValueError("bmat 要求每个块行的列数相同")

    row_sizes = []
    col_sizes = []
    for i, row in enumerate(rows):
        sizes = [None if b is None else _block_shape(b)[0] for b in row]
        known = [s for s in sizes if s is not None]
        row_sizes.append(known[0] if known else 0)
        _check_sizes(sizes, f"第 {i} 块行")
    for j in range(n_cols):
        sizes = [None if rows[i][j] is None else _block_shape(rows[i][j])[1]
                 for i in range(n_rows)]
        known = [s for s in sizes if s is not None]
        col_sizes.append(known[0] if known else 1)
        _check_sizes(sizes, f"第 {j} 块列")

    # 只要有块是稀疏的，就整体走稀疏路线（调用方通过块的类型来决定）
    if HAS_SCIPY and any(is_sparse(b) for row in rows for b in row if b is not None):
        dense_rows = [
            [
                None if b is None
                else (b if is_sparse(b) else sparse.csr_matrix(np.atleast_2d(b)))
                for b in row
            ]
            for row in rows
        ]
        return sparse.bmat(dense_rows, format="csr")

    shape = (int(np.sum(row_sizes)), int(np.sum(col_sizes)))
    A = np.zeros(shape, dtype=float)
    r0 = 0
    for i, row in enumerate(rows):
        c0 = 0
        for j, b in enumerate(row):
            if b is not None:
                bh, bw = _block_shape(b)
                A[r0:r0 + bh, c0:c0 + bw] = to_dense(b).reshape(bh, bw)
            c0 += col_sizes[j]
        r0 += row_sizes[i]
    return A


def _block_shape(block):
    if is_sparse(block):
        return int(block.shape[0]), int(block.shape[1])
    arr = np.asarray(block)
    if arr.ndim == 1:
        return 1, int(arr.size)
    return int(arr.shape[0]), int(arr.shape[1])


def _check_sizes(sizes, where):
    known = [s for s in sizes if s is not None]
    if len(set(known)) > 1:
        raise ValueError(f"bmat 的{where}中各块尺寸不一致：{sizes}")


def is_sparse(A):
    """判断是否为 scipy 稀疏矩阵。"""
    return bool(HAS_SCIPY and sparse.issparse(A))
