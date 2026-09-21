"""线性代数求解器（guidebook 第 7 章）。

统一入口 :class:`Solver`：::

    from MosicaFE import Solver

    u = Solver.direct(A, b)                    # 稀疏 LU / Cholesky，精确
    u = Solver.conjugate_gradient(A, b)        # SPD 系统
    u = Solver.gmres(A, b, tol=1e-10)          # 一般（非对称）系统
    u = Solver.bicgstab(A, b, tol=1e-10)
    u = Solver.minres(A, b)                    # 对称不定
    u = Solver.solve(A, b, method="auto")      # 自动选择

装了 scipy 就用 ``scipy.sparse.linalg``；没装则退化为本文件内的
纯 numpy 实现（CG / GMRES / BiCGSTAB）或稠密直接法。
"""

from __future__ import annotations

import numpy as np

from .._compat import HAS_SCIPY
from .._compat import bicgstab as _sp_bicgstab
from .._compat import cg as _sp_cg
from .._compat import gmres as _sp_gmres
from .._compat import is_sparse
from .._compat import minres as _sp_minres
from .._compat import spsolve as _sp_spsolve
from .._compat import to_dense

__all__ = ["Solver", "ConvergenceError"]


class ConvergenceError(RuntimeError):
    """迭代法未在给定迭代数内收敛。"""


class Solver:
    """线性求解器集合。"""

    METHODS = ("direct", "cg", "gmres", "bicgstab", "minres", "auto")

    # ------------------------------------------------------------------
    # 边界条件
    # ------------------------------------------------------------------
    @staticmethod
    def apply_dirichlet(A, b, dofs, values=None):
        """强加 Dirichlet 边界条件 ``u[dofs] = values``（缺省取 0）。

        采用标准的"消行消列"法，对稀疏矩阵与稠密矩阵都可用：

        1. 把边界列对内部行的贡献搬到右端项：
           :math:`b_i \\leftarrow b_i - A_{ij}\\,g_j`；
        2. 删掉涉及边界自由度的行列；
        3. 边界行在对角线上放 1、右端项放边界值 :math:`g_j`。

        第 1 步在 ``g = 0`` 时看不出来，但非齐次边界缺了它解就是错的
        （非齐次 Dirichlet 的 patch test 会立刻暴露）。
        """
        dofs = np.asarray(dofs, dtype=np.int64).ravel()
        b = np.array(b, dtype=float, copy=True)
        if dofs.size == 0:
            return A, b
        if values is None:
            vals = np.zeros(dofs.size)
        else:
            vals = np.broadcast_to(np.asarray(values, dtype=float), dofs.shape).copy()

        n = b.size
        mapping = -np.ones(n, dtype=np.int64)
        mapping[dofs] = np.arange(dofs.size)

        if is_sparse(A):
            coo = A.tocoo()
            row_bd = np.isin(coo.row, dofs)
            col_bd = np.isin(coo.col, dofs)
            # 1) 边界列 → 右端项
            lift = col_bd
            if np.any(lift):
                np.add.at(
                    b,
                    coo.row[lift],
                    -(coo.data[lift] * vals[mapping[coo.col[lift]]]),
                )
            # 2) 删行删列
            mask = ~(row_bd | col_bd)
            rows = np.concatenate([coo.row[mask], dofs])
            cols = np.concatenate([coo.col[mask], dofs])
            data = np.concatenate([coo.data[mask], np.ones(dofs.size)])
            from .._compat import sparse

            A = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        else:
            A = np.array(A, dtype=float, copy=True)
            b = b - A[:, dofs] @ vals
            A[dofs, :] = 0.0
            A[:, dofs] = 0.0
            A[dofs, dofs] = 1.0

        b[dofs] = vals
        return A, b

    # ------------------------------------------------------------------
    # 直接法
    # ------------------------------------------------------------------
    @staticmethod
    def direct(A, b):
        """直接求解 ``A x = b``（稀疏用 scipy 的 LU，稠密用 numpy）。"""
        b = np.asarray(b, dtype=float).ravel()
        if is_sparse(A):
            x = _sp_spsolve(A.tocsc(), b)
            return np.asarray(x, dtype=float).ravel()
        return np.linalg.solve(np.asarray(A, dtype=float), b)

    # ------------------------------------------------------------------
    # Krylov 迭代法
    # ------------------------------------------------------------------
    @staticmethod
    def conjugate_gradient(A, b, tol=1e-10, maxiter=None, x0=None):
        """共轭梯度法（要求 A 对称正定）。"""
        b = np.asarray(b, dtype=float).ravel()
        maxiter = int(maxiter or min(20000, 50 * b.size))
        if HAS_SCIPY and (is_sparse(A) or isinstance(A, np.ndarray)):
            x, info = _sp_cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter, x0=x0)
            if info > 0:
                raise ConvergenceError(f"CG 未收敛（info={info}, maxiter={maxiter}）")
            return np.asarray(x, dtype=float).ravel()
        return _cg_dense(to_dense(A), b, tol, maxiter, x0)

    @staticmethod
    def gmres(A, b, tol=1e-10, restart=30, maxiter=None, x0=None):
        """重启 GMRES（适用于非对称系统）。"""
        b = np.asarray(b, dtype=float).ravel()
        maxiter = int(maxiter or min(10000, 50 * b.size))
        if HAS_SCIPY:
            x, info = _sp_gmres(A, b, rtol=tol, atol=0.0, restart=restart,
                                maxiter=maxiter, x0=x0)
            if info > 0:
                raise ConvergenceError(f"GMRES 未收敛（info={info}）")
            return np.asarray(x, dtype=float).ravel()
        return _gmres_dense(to_dense(A), b, tol, restart, maxiter)

    @staticmethod
    def bicgstab(A, b, tol=1e-10, maxiter=None, x0=None):
        """稳定双共轭梯度法。"""
        b = np.asarray(b, dtype=float).ravel()
        maxiter = int(maxiter or min(20000, 50 * b.size))
        if HAS_SCIPY:
            x, info = _sp_bicgstab(A, b, rtol=tol, atol=0.0, maxiter=maxiter, x0=x0)
            if info > 0:
                raise ConvergenceError(f"BiCGSTAB 未收敛（info={info}）")
            return np.asarray(x, dtype=float).ravel()
        return _bicgstab_dense(to_dense(A), b, tol, maxiter)

    @staticmethod
    def minres(A, b, tol=1e-10, maxiter=None):
        """MINRES（适用于对称不定系统，如鞍点问题）。"""
        b = np.asarray(b, dtype=float).ravel()
        maxiter = int(maxiter or min(20000, 50 * b.size))
        if HAS_SCIPY:
            x, info = _sp_minres(A, b, rtol=tol, atol=0.0, maxiter=maxiter)
            if info > 0:
                raise ConvergenceError(f"MINRES 未收敛（info={info}）")
            return np.asarray(x, dtype=float).ravel()
        raise NotImplementedError("未安装 scipy 时没有内置 MINRES，请改用 direct 或 gmres")

    # ------------------------------------------------------------------
    # 自动选择与统一入口
    # ------------------------------------------------------------------
    @staticmethod
    def recommend(n_dofs: int, symmetric: bool = True) -> str:
        """按 guidebook 表 7.1 的规则推荐求解器。"""
        if n_dofs <= 20000:
            return "direct"
        return "cg" if symmetric else "gmres"

    @classmethod
    def solve(cls, A, b, method: str = "auto", symmetric: bool = True, **kwargs):
        """统一入口。``method="auto"`` 时按规模与对称性自动选择。"""
        method = str(method).lower()
        if method not in cls.METHODS:
            raise ValueError(f"未知求解器 {method!r}；可用：{cls.METHODS}")
        if method == "auto":
            method = cls.recommend(len(np.asarray(b).ravel()), symmetric)
        if method == "direct":
            return cls.direct(A, b)
        if method == "cg":
            return cls.conjugate_gradient(A, b, **kwargs)
        if method == "gmres":
            return cls.gmres(A, b, **kwargs)
        if method == "bicgstab":
            return cls.bicgstab(A, b, **kwargs)
        return cls.minres(A, b, **kwargs)


# ---------------------------------------------------------------------------
# 纯 numpy 回退实现（只在没有 scipy 时使用）
# ---------------------------------------------------------------------------
def _cg_dense(A, b, tol, maxiter, x0=None):
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    r = b - A @ x
    p = r.copy()
    rs = float(r @ r)
    bnorm = max(float(np.linalg.norm(b)), 1e-30)
    for _ in range(maxiter):
        if np.sqrt(rs) / bnorm < tol:
            return x
        Ap = A @ p
        denom = float(p @ Ap)
        if abs(denom) < 1e-300:
            break
        alpha = rs / denom
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        p = r + (rs_new / rs) * p
        rs = rs_new
    if np.sqrt(rs) / bnorm >= tol:
        raise ConvergenceError(f"CG（纯 numpy）未收敛，残量 {np.sqrt(rs):.3e}")
    return x


def _gmres_dense(A, b, tol, restart, maxiter):
    n = b.size
    x = np.zeros(n)
    bnorm = max(float(np.linalg.norm(b)), 1e-30)
    for _ in range(max(1, maxiter // restart)):
        r = b - A @ x
        beta = float(np.linalg.norm(r))
        if beta / bnorm < tol:
            return x
        V = np.zeros((n, restart + 1))
        H = np.zeros((restart + 1, restart))
        V[:, 0] = r / beta
        for j in range(restart):
            w = A @ V[:, j]
            for i in range(j + 1):
                H[i, j] = float(w @ V[:, i])
                w -= H[i, j] * V[:, i]
            H[j + 1, j] = float(np.linalg.norm(w))
            if H[j + 1, j] < 1e-300:
                break
            V[:, j + 1] = w / H[j + 1, j]
        k = j + 1
        g = np.zeros(k + 1)
        g[0] = beta
        for i in range(k):
            denom = np.hypot(H[i, i], H[i + 1, i])
            if denom < 1e-300:
                continue
            c, s = H[i, i] / denom, H[i + 1, i] / denom
            H[i, i], H[i + 1, i] = denom, 0.0
            g[i], g[i + 1] = c * g[i] + s * g[i + 1], -s * g[i] + c * g[i + 1]
        y = np.linalg.solve(H[:k, :k], g[:k]) if k else np.zeros(0)
        x = x + V[:, :k] @ y
        if float(np.linalg.norm(b - A @ x)) / bnorm < tol:
            return x
    raise ConvergenceError("GMRES（纯 numpy）未收敛")


def _bicgstab_dense(A, b, tol, maxiter):
    x = np.zeros_like(b)
    r = b - A @ x
    rhat = r.copy()
    rho = alpha = omega = 1.0
    v = p = np.zeros_like(b)
    bnorm = max(float(np.linalg.norm(b)), 1e-30)
    for _ in range(maxiter):
        if float(np.linalg.norm(r)) / bnorm < tol:
            return x
        rho_new = float(rhat @ r)
        if abs(rho_new) < 1e-300:
            break
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = A @ p
        denom = float(rhat @ v)
        if abs(denom) < 1e-300:
            break
        alpha = rho_new / denom
        s = r - alpha * v
        if float(np.linalg.norm(s)) / bnorm < tol:
            x += alpha * p
            return x
        t = A @ s
        tt = float(t @ t)
        omega = float(t @ s) / tt if tt > 1e-300 else 0.0
        x += alpha * p + omega * s
        r = s - omega * t
        rho = rho_new
    if float(np.linalg.norm(r)) / bnorm >= tol:
        raise ConvergenceError("BiCGSTAB（纯 numpy）未收敛")
    return x
