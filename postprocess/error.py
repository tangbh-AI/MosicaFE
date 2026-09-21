"""误差范数与误差分析（guidebook 第 8 章）。

三种范数：

* ``"L2"`` —— :math:`\\|u-u_h\\|_{L^2(\\Omega)}`，单元求积
* ``"H1"`` —— :math:`|u-u_h|_{H^1(\\Omega)}`（半范数），需要精确梯度
* ``"Linf"`` —— :math:`\\|u-u_h\\|_{L^\\infty}`，取自由度点与求积点的最大值

关于 VEM：虚拟元没有显式基函数，因此 :meth:`VEMSpace.evaluate` 用的是
**顶点自由度上的线性最小二乘重构**（与 :math:`u_h` 同阶）。
这样算出的收敛阶与真实 :math:`u_h` 是一致的，具体数值见 progress/VERIFICATION.md。

关于 RT0×P0：压力是分片常数、通量是显式的向量多项式，两者都可以逐点求值，
因此混合格式的误差范数 :func:`mixed_error_norms` 与 H1 协调空间用的是同一套
单元求积循环，只是把 :math:`u_h` 换成 :math:`(p_h,q_h)`。
"""

from __future__ import annotations

import numpy as np

from ..core.quadrature import cell_quadrature

__all__ = [
    "compute_error",
    "compute_errors",
    "l2_error",
    "h1_error",
    "linf_error",
    "mixed_error_norms",
    "estimate_convergence_rate",
    "rate_table",
]


# ---------------------------------------------------------------------------
def _eval_scalar(space, f, points) -> np.ndarray:
    return space._evaluate_function(f, points)


def _eval_vector(g, points) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return np.vstack([np.asarray(g(p), dtype=float).ravel() for p in points])


def _extract_solution(problem, solution):
    """把 ``solve()`` 的返回值统一成 "用于误差分析的标量自由度向量"。"""
    space = problem.space
    if getattr(problem, "is_mixed", False) or getattr(space, "is_mixed", False):
        if isinstance(solution, (tuple, list)) and len(solution) == 2:
            return np.asarray(solution[1], dtype=float).ravel()
        return np.asarray(solution, dtype=float).ravel()
    return np.asarray(solution, dtype=float).ravel()


# ---------------------------------------------------------------------------
def compute_errors(problem, solution, exact, exact_grad=None, quad_order=None):
    """计算 ``{"L2":..., "H1":..., "Linf":...}``。

    Parameters
    ----------
    problem : BasePhysics
    solution : ndarray 或 (Q, P)
    exact : callable
        精确解 ``exact(x)``，``x`` 是长度 dim 的坐标数组。
    exact_grad : callable, optional
        精确梯度 ``exact_grad(x) -> (dim,)``；给了才会算 H1。
    quad_order : int, optional
        单元求积阶，默认 ``max(4, 2*degree+2)``。
    """
    space = problem.space
    mesh = space.mesh

    if getattr(problem, "is_mixed", False):
        return _mixed_errors(problem, solution, exact, exact_grad)

    u = _extract_solution(problem, solution)
    if quad_order is None:
        quad_order = max(4, 2 * int(getattr(space, "degree", 1)) + 2)

    l2sq = 0.0
    h1sq = 0.0
    linf = 0.0

    for c in range(mesh.n_cells):
        pts, w = cell_quadrature(mesh, c, quad_order)
        if exact is not None:
            uh = space.evaluate(c, u, pts)
            ue = _eval_scalar(space, exact, pts)
            diff = uh - ue
            l2sq += float(np.sum(w * diff ** 2))
            linf = max(linf, float(np.max(np.abs(diff))) if diff.size else 0.0)
        if exact_grad is not None:
            gh = space.evaluate_gradient(c, u, pts)
            ge = _eval_vector(exact_grad, pts)
            h1sq += float(np.sum(w * np.sum((gh - ge) ** 2, axis=1)))

    # 自由度点上的 L∞（对 P2/VEM 更贴近原脚本的 max|u_i - u(x_i)|）
    try:
        if exact is None:
            raise RuntimeError("no exact solution given")
        ud = _eval_scalar(space, exact, space.dof_coords)
        linf = max(linf, float(np.max(np.abs(u - ud))) if u.size else 0.0)
    except Exception:
        pass

    out = {
        "L2": float(np.sqrt(max(l2sq, 0.0))) if exact is not None else float("nan"),
        "Linf": float(linf) if exact is not None else float("nan"),
        "H1": float(np.sqrt(max(h1sq, 0.0))) if exact_grad is not None else float("nan"),
    }
    return out


def _mixed_errors(problem, solution, exact, exact_flux):
    """混合格式：压力 L2/L∞ 与（可选的）通量 L2。"""
    if isinstance(solution, (tuple, list)) and len(solution) == 2:
        Q, P = solution
    else:
        Q, P = getattr(problem, "flux", None), solution
    norms = mixed_error_norms(problem.space, P, Q, exact, exact_flux)
    return {
        "L2": float(norms.get("p_l2", np.nan)),
        "Linf": float(norms.get("p_linf", np.nan)),
        "H1": float(norms.get("q_l2", np.nan)),
        "flux_L2": float(norms.get("q_l2", np.nan)),
        "flux_Linf": float(norms.get("q_linf", np.nan)),
    }


# ---------------------------------------------------------------------------
def mixed_error_norms(space, P, Q, p_exact, q_exact=None, quadrature_order=None):
    """RT0×P0 混合格式的误差范数（原生实现，不依赖任何外部后端）。

    * **压力**：:math:`p_h` 是单元常数，逐求积点与 ``p_exact`` 比较；
    * **通量**：用 :meth:`RT0Space.evaluate_flux` 重构
      :math:`q_h=\\sum_i F_i\\varphi_i` 后与 ``q_exact`` 比较；
    * 通量误差记为 ``q_l2`` / ``q_linf``，压力误差记为 ``p_l2`` / ``p_linf``。

    Parameters
    ----------
    space : RT0Space
    P : (n_cells,) ndarray    常数压力
    Q : (n_cells, d+1) ndarray 单元局部外法向通量
    p_exact : callable
    q_exact : callable, optional
        精确通量 ``q_exact(x) -> (d,)``；给了才计算 ``q_l2`` / ``q_linf``。
    quadrature_order : int, optional
        单元求积阶，缺省取 4（2D 6 点 / 3D 5 点规则）。

    Returns
    -------
    dict with keys ``{"p_l2", "p_linf", "q_l2", "q_linf"}``
        （未给 ``q_exact`` 时 ``q_*`` 为 ``nan``）。
    """
    mesh = space.mesh
    order = 4 if quadrature_order is None else int(quadrature_order)
    P = np.asarray(P, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float)
    if P.size != mesh.n_cells:
        raise ValueError(f"压力长度应为 {mesh.n_cells}，收到 {P.size}")

    p_l2sq = 0.0
    p_linf = 0.0
    q_l2sq = 0.0
    q_linf = 0.0
    for c in range(mesh.n_cells):
        pts, w = cell_quadrature(mesh, c, order)
        diff_p = _eval_scalar(space, p_exact, pts) - P[c]
        p_l2sq += float(np.sum(w * diff_p ** 2))
        p_linf = max(p_linf, float(np.max(np.abs(diff_p))) if diff_p.size else 0.0)
        if q_exact is not None:
            diff_q = _eval_vector(q_exact, pts) - space.evaluate_flux(c, Q[c], pts)
            q_l2sq += float(np.sum(w * np.sum(diff_q ** 2, axis=1)))
            q_linf = max(
                q_linf,
                float(np.max(np.linalg.norm(diff_q, axis=1))) if diff_q.size else 0.0,
            )

    return {
        "p_l2": float(np.sqrt(max(p_l2sq, 0.0))),
        "p_linf": float(p_linf),
        "q_l2": float(np.sqrt(max(q_l2sq, 0.0))) if q_exact is not None else float("nan"),
        "q_linf": float(q_linf) if q_exact is not None else float("nan"),
    }


# ---------------------------------------------------------------------------
def compute_error(problem, solution, exact, norm_type: str = "L2", exact_grad=None):
    """单个范数。``norm_type`` 取 ``"L2" / "H1" / "Linf"``（大小写不敏感）。"""
    key = str(norm_type).strip()
    mapping = {"l2": "L2", "linf": "Linf", "l∞": "Linf", "h1": "H1", "l2_flux": "flux_L2"}
    key = mapping.get(key.lower(), key)
    if key == "H1" and exact_grad is None:
        raise ValueError("计算 H1 误差需要提供 exact_grad（精确梯度函数）")
    return compute_errors(problem, solution, exact, exact_grad=exact_grad)[key]


def l2_error(problem, solution, exact):
    return compute_error(problem, solution, exact, "L2")


def h1_error(problem, solution, exact_grad):
    return compute_errors(problem, solution, None, exact_grad=exact_grad)["H1"]


def linf_error(problem, solution, exact):
    return compute_error(problem, solution, exact, "Linf")


# ---------------------------------------------------------------------------
def estimate_convergence_rate(h_values, errors, use_last: int = None) -> float:
    """由 ``(h, e)`` 序列估计收敛阶：``rate = log(e1/e2) / log(h1/h2)``。

    ``use_last=n`` 时只用最后 n 个点（更接近渐近阶）。
    """
    h = np.asarray(h_values, dtype=float).ravel()
    e = np.asarray(errors, dtype=float).ravel()
    mask = (e > 0) & (h > 0) & np.isfinite(e)
    h, e = h[mask], e[mask]
    if use_last is not None and len(h) > use_last:
        h, e = h[-use_last:], e[-use_last:]
    if len(h) < 2:
        return float("nan")
    slope = np.polyfit(np.log(h), np.log(e), 1)[0]
    return float(slope)


def rate_table(h_values, errors, labels=("L2", "H1", "Linf")) -> str:
    """生成一张 guidebook 表 8.1 那样的收敛表（纯文本）。"""
    h = np.asarray(h_values, dtype=float).ravel()
    lines = ["  level        h        " + "".join(f"{lab:>12s}{'rate':>8s}" for lab in labels)]
    for i in range(len(h)):
        row = f"  {i:5d}  {h[i]:.5f}  "
        for lab in labels:
            vals = np.asarray(errors[lab], dtype=float)
            if i < len(vals):
                row += f"{vals[i]:12.4e}"
                if i >= 1 and np.isfinite(vals[i]) and vals[i] > 0 and vals[i - 1] > 0:
                    rate = np.log(vals[i - 1] / vals[i]) / np.log(h[i - 1] / h[i])
                    row += f"{rate:8.2f}"
                else:
                    row += f"{'-':>8s}"
        lines.append(row)
    return "\n".join(lines)
