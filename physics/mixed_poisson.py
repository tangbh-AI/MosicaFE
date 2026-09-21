"""混合形式的 Poisson 问题（RT0×P0）。

.. math::

   q=-\\nabla p,\\qquad \\nabla\\cdot q=f\\ \\text{in}\\ \\Omega,
   \\qquad p=0\\ \\text{on}\\ \\partial\\Omega

弱形式（鞍点问题）：找 :math:`(q,p)\\in RT0\\times P0` 使

.. math::

   \\int\\nu\\cdot q-\\int p\\,\\nabla\\cdot\\nu=0,\\qquad
   \\int r\\,\\nabla\\cdot q=\\int f r

对任意 :math:`(\\nu,r)\\in RT0\\times P0`。离散后是

.. math::

   \\begin{pmatrix} M & -B^{T}\\\\ B & 0\\end{pmatrix}
   \\begin{pmatrix}\\mathbf Q\\\\ \\mathbf P\\end{pmatrix}
   =\\begin{pmatrix}0\\\\ \\mathbf f\\end{pmatrix},
   \\qquad
   M_{ij}=\\int_K\\varphi_i\\cdot\\varphi_j,\\quad
   B_{K,i}=\\sigma_{K,i},\\quad
   \\mathbf f_K=\\int_K f .

两个块都由空间层给出（``V.flux_mass_matrix()`` 与 ``V.divergence_matrix()``），
本类只做矩阵拼装与求解——与 :class:`~MosicaFE.physics.poisson.PoissonProblem`
对 Lagrange/VEM 的做法完全平行。

.. code-block:: python

   V = FESpace.rt0(mesh)                # 三角形 / 四面体网格
   problem = MixedPoissonProblem(V, f=source)
   Q, P = problem.solve()               # 通量、压力

注意：齐次 Dirichlet 条件是混合格式的**自然**边界条件，不需要（也不应该）
像 H1 协调空间那样做"消行消列"。
"""

from __future__ import annotations

import numpy as np

from .._compat import HAS_SCIPY, bmat, is_sparse, splu, to_dense
from ..solvers.linear import Solver
from .base import BasePhysics
from .registry import register_problem

__all__ = ["MixedPoissonProblem"]


@register_problem("mixed_poisson")
class MixedPoissonProblem(BasePhysics):
    """RT0×P0 混合 Poisson 问题。"""

    is_mixed = True
    #: 鞍点系统对称但**不定**（不是 SPD），供 Solver.recommend 参考
    symmetric_indefinite = True

    def __init__(self, space, f=None, g=None, quadrature_order: int = 3,
                 lumped: bool = None, **kwargs):
        super().__init__(space, f=f, g=g, **kwargs)
        if not getattr(space, "is_mixed", False):
            raise TypeError(
                "MixedPoissonProblem 需要混合空间 FESpace.rt0(mesh)；"
                f"收到 {type(space).__name__}"
            )
        self.quadrature_order = int(quadrature_order)
        self.lumped = None if lumped is None else bool(lumped)
        self._check_boundary_data()

        # 组装结果（assemble() 填充）
        self.flux_mass = None
        self.divergence = None
        self.source = None
        self.flux = None
        self.pressure = None

    # ------------------------------------------------------------------
    def _check_boundary_data(self):
        """混合格式只支持齐次 Dirichlet；非齐次需要额外的边界通量项。"""
        if self.g is None:
            return
        vals = self.space._evaluate_function(self.g, self.mesh.nodes)
        if np.max(np.abs(vals)) > 0.0:
            raise NotImplementedError(
                "RT0 混合格式目前只支持齐次 Dirichlet 条件 p=0"
                "（它在混合格式下是自然边界条件）；非齐次条件需要额外的边界通量项。"
            )

    # ------------------------------------------------------------------
    # 组装：空间给块，物理层只做组合（DECISIONS.md D3）
    # ------------------------------------------------------------------
    def assemble(self):
        """组装鞍点系统，返回 ``(A, b)``。

        ``A = [[M, -Bᵀ], [B, 0]]``（未知量顺序 ``[Φ, P]``），
        ``b = [0, ∫_K f]``。中间的三个块同时保存在
        ``self.flux_mass / self.divergence / self.source``，供 Schur 求解与
        通量回代使用。
        """
        V = self.space
        M = V.flux_mass_matrix(lumped=self.lumped)
        B = V.divergence_matrix()
        source = V.pressure_load_vector(self.f, quadrature_order=self.quadrature_order)

        A = bmat([[M, -B.T], [B, None]])
        b = np.concatenate([np.zeros(V.n_flux), source])

        self.flux_mass = M
        self.divergence = B
        self.source = source
        return A, b

    # ------------------------------------------------------------------
    # 求解
    # ------------------------------------------------------------------
    def solve(self, method: str = "auto", **kwargs):
        """求解，返回 ``(Q, P)``。

        ``Q[c, i]``：单元 ``c`` 第 ``i`` 个局部面上的**外法向**通量；
        ``P[c]``：单元 ``c`` 上的常数压力。

        ``method`` 可选：

        * ``"auto"``（默认）：有 scipy 时用 ``"schur"``，否则用 ``"direct"``；
        * ``"direct"``：对整个鞍点矩阵做稀疏/稠密 LU；
        * ``"schur"``：静力凝聚掉通量自由度，解 SPD 的 :math:`\\Sigma M^{-1}\\Sigma^{T}P=f_e`
          再回代通量（规模只有 ``n_cells``）；
        * ``"minres" / "gmres" / "bicgstab"``：迭代法（鞍点系统请优先 MINRES）。
        """
        A, b = self.assemble()
        self._system = (A, b)
        method = str(method).lower()
        if method == "auto":
            method = "schur" if HAS_SCIPY else "direct"
        if method == "schur":
            sol = self._solve_schur()
        elif method == "direct":
            sol = Solver.direct(A, b)
        else:
            sol = Solver.solve(A, b, method=method, symmetric=False, **kwargs)
        self.solution = sol
        return self.extract(sol)

    def _solve_schur(self) -> np.ndarray:
        """静力凝聚：``M Φ = Bᵀ P``，``B M^{-1} Bᵀ P = f_e``。"""
        V = self.space
        M = self.flux_mass
        B = self.divergence
        fe = self.source

        # 第一块行给出 M Φ = Bᵀ P，代入第二块行即得 B M⁻¹ Bᵀ P = f_e
        rhs = to_dense(B.T)                       # (n_flux, n_cells)
        if is_sparse(M) and splu is not None:
            X = splu(M.tocsc()).solve(rhs)
        else:
            X = np.linalg.solve(to_dense(M), rhs)
        S = to_dense(B @ X)
        S = 0.5 * (S + S.T)
        P = np.linalg.solve(S, fe)
        Phi = X @ P
        return np.concatenate([np.asarray(Phi).ravel(), P])

    # ------------------------------------------------------------------
    # 提取
    # ------------------------------------------------------------------
    def extract(self, sol):
        """把解向量拆成 ``(Q, P)``：``Q`` 是逐单元的**外法向**面通量。"""
        V = self.space
        s = np.asarray(sol, dtype=float).ravel()
        if s.size != V.n_dofs:
            raise ValueError(f"解向量长度 {s.size} 与空间自由度数 {V.n_dofs} 不符")
        Phi = s[: V.n_flux]
        P = s[V.n_flux:]
        Q = V._facet_signs * Phi[V._cell_facet_dofs]
        self.flux, self.pressure = Q, P
        return Q, P

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------
    def compute_error(self, solution, exact, norm_type: str = "L2", **kwargs):
        """压力误差（``solution`` 为 ``(Q, P)`` 或 ``P``）。"""
        Q, P = self._split(solution)
        norms = self.error_norms(P=P, Q=Q, p_exact=exact)
        key = {
            "L2": "p_l2", "LINF": "p_linf", "Linf": "p_linf",
            "L2_FLUX": "q_l2", "H1": "q_l2",
        }.get(str(norm_type).strip().upper(), "p_l2")
        return norms.get(key, float("nan"))

    def compute_errors(self, solution, exact, exact_flux=None, **kwargs):
        """一次给出压力/通量误差：``{"L2","Linf","H1"(=通量 L2),"flux_L2","flux_Linf"}``。"""
        from ..postprocess.error import compute_errors as _compute_errors

        return _compute_errors(self, solution, exact, exact_grad=exact_flux)

    def error_norms(self, P, Q, p_exact, q_exact=None, quadrature_order: int = None):
        """压力 L2/L∞ 与通量 L2/L∞ 误差（原生实现，见 postprocess.error）。"""
        from ..postprocess.error import mixed_error_norms

        return mixed_error_norms(
            self.space, P, Q, p_exact, q_exact,
            quadrature_order=quadrature_order,
        )

    def cell_flux(self, cell_idx: int, quadrature_order: int = None) -> np.ndarray:
        """单元 ``cell_idx`` 形心处的重构通量向量 ``q_h``。"""
        if self.flux is None:
            raise RuntimeError("请先调用 solve() 再求通量")
        return self.space.evaluate_flux(
            cell_idx, self.flux, self.mesh.get_cell_centroid(cell_idx)[None, :]
        )[0]

    def flux_at_points(self, points) -> np.ndarray:
        """在物理点上重构通量向量，返回 ``(n_points, dim)``。

        每个点先用重心坐标定位到所在单元，再用该单元的通量自由度重构
        :math:`q_h=\\sum_i F_i\\varphi_i`；点落在单元边界上时取第一个命中的单元。
        """
        if self.flux is None:
            raise RuntimeError("请先调用 solve() 再求通量")
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        V = self.space
        out = np.zeros((pts.shape[0], self.mesh.dim))
        for k, p in enumerate(pts):
            c = V.locate_cell(p)
            out[k] = V.evaluate_flux(c, self.flux, p[None, :])[0]
        return out

    def check_conservation(self, tol: float = 1e-8):
        """检查离散质量守恒 :math:`\\sum_i Q_{K,i}=\\int_K f`，返回最大偏差。"""
        if self.flux is None or self.source is None:
            raise RuntimeError("请先调用 solve()（或 assemble()）再检查守恒性")
        res = np.abs(self.flux.sum(axis=1) - self.source)
        worst = float(np.max(res))
        if worst > tol:
            raise AssertionError(
                f"离散守恒被破坏：最大偏差 {worst:.3e}（容差 {tol:.1e}）"
            )
        return worst

    # ------------------------------------------------------------------
    def _split(self, solution):
        if isinstance(solution, (tuple, list)) and len(solution) == 2:
            return np.asarray(solution[0], float), np.asarray(solution[1], float)
        P = np.asarray(solution, dtype=float).ravel()
        if P.size != self.space.n_pressure:
            raise ValueError(
                "混合问题的解应传 (Q, P)；只传数组时长度需等于压力自由度数 "
                f"({self.space.n_pressure})，收到 {P.size}"
            )
        return self.flux, P

    def __repr__(self):  # pragma: no cover - 仅展示
        return (
            f"<MixedPoissonProblem space={self.space.name} dim={self.mesh.dim} "
            f"flux={self.space.n_flux} pressure={self.space.n_pressure}>"
        )
