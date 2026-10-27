"""Darcy（达西）渗流问题（RT0×P0 混合形式）。

.. math::

   K^{-1}q+\\nabla p=0,\\qquad \\nabla\\cdot q=f\\ \\text{in}\\ \\Omega,

   p=g_D\\ \\text{on}\\ \\Gamma_D,\\qquad q\\cdot n=g_N\\ \\text{on}\\ \\Gamma_N .

与 :class:`~MosicaFE.physics.mixed_poisson.MixedPoissonProblem` 的关系
--------------------------------------------------------------------
两者共用同一套鞍点结构

.. math::

   \\begin{pmatrix} A & -B^{\\mathsf T}\\\\ B & 0\\end{pmatrix}
   \\begin{pmatrix}\\mathbf Q\\\\ \\mathbf P\\end{pmatrix}
   =\\begin{pmatrix}\\boldsymbol b_\\Phi\\\\ \\mathbf F\\end{pmatrix},

区别只有两处：

1. **通量质量矩阵带 :math:`K^{-1}` 权重**

   .. math::

      A_{ij}=\\int_\\Omega\\bigl(K^{-1}\\varphi_i\\bigr)\\cdot\\varphi_j\\,\\mathrm dx
      \\qquad(\\text{Poisson 是 }K=I\\text{ 的特例}),

   由空间层的 :meth:`~MosicaFE.spaces.rt0.RT0Space.weighted_flux_mass_matrix` 提供；
2. **支持本质（流量）边界条件**
   :math:`q\\cdot n=0`（不透水）与 :math:`q\\cdot n=g_N`（给定流量）落在通量自由度上，
   同时压力 Dirichlet 可以是**非齐次**的（它进第一个方程右端
   :math:`-\\langle g_D,\\varphi_f\\cdot n\\rangle_{\\Gamma_D}`）。

.. warning::

   **本质边界条件的消去必须做三件事**（少一件结果就是错的）：

   1. ``A`` 的已知列搬到第一个方程右端（自由自由度上的 :math:`A_{fc}q_c` 项）；
   2. ``B`` 的已知列搬到第二个方程右端（守恒方程里该给定通量的贡献）；
   3. ``A`` 的已知行/列换成单位行**并把 ``B`` 的该列清零**——
      否则鞍点矩阵的第一个块行 :math:`[A,\\,-B^{\\mathsf T}]` 里还留着
      :math:`-B^{\\mathsf T}` 对该自由度的耦合，"约束行"会变成
      :math:`\\Phi_f-\\sum_K B_{Kf}P_K=\\text{已知值}`，边界条件实际没有生效。

.. code-block:: python

   from MosicaFE import FESpace, Mesh, DarcyProblem

   mesh = Mesh.rectangle(nx=32, ny=32)                 # 三角形网格
   V = FESpace.rt0(mesh)                               # RT0 × P0

   # 上下不透水、左右定压 p=1 / p=0，渗透率 2.5
   bc = lambda mid, n: "D" if (mid[0] < 1e-12 or mid[0] > 1 - 1e-12) else "N"
   problem = DarcyProblem(V, f=lambda x: 0.0, g=lambda x: 1.0 - x[:, 0],
                          K=2.5, bc=bc)
   Q, P = problem.solve()                              # Q: (n_cells, d+1) 外法向通量
   print(problem.check_conservation())                 # 逐单元守恒偏差

各向异性的三种写法（标量 / 对角 / 一般张量，也接受可调用形式）::

   DarcyProblem(V, f=src, K=2.5)                       # 各向同性
   DarcyProblem(V, f=src, K=[1.0, 0.01])               # 对角（沿 y 方向不易流动）
   DarcyProblem(V, f=src, K=[[3.0, 1.0], [1.0, 2.0]])  # 带非对角项
   DarcyProblem(V, f=src, K=lambda x: 1.0 + x[:, 0])   # 变系数

边界条件的分类由 ``bc(mid, n_out)`` 给出（``mid`` 为面形心，``n_out`` 为单位外法向）：

===========================  =====================================  ==================
返回值                        含义                                   处理方式
===========================  =====================================  ==================
``"D"``                       压力 Dirichlet :math:`p=g_D`           自然条件，进右端
``"N"``                       不透水 :math:`q\\cdot n=0`             本质条件，自由度置零
``("flux", gN)``              给定流量 :math:`q\\cdot n=g_N`         本质条件，取已知值
===========================  =====================================  ==================

``gN`` 是接受 ``(n_points, d)`` 数组、返回 ``(n_points,)`` 的函数。

.. note::

   若整个边界都没有 Dirichlet 条件（纯 Neumann），压力的常数零空间必须由
   参考压力固定；本类会自动改用最小二乘并把压力平移到零均值。
   此时还需检查相容性 :math:`\\int_\\Omega f+\\int_{\\Gamma_N}g_N=0`。

参考文献：Darcy (1856)；Bear, *Dynamics of Fluids in Porous Media* (1972)；
Boffi, Brezzi & Fortin, *Mixed Finite Element Methods and Applications* (2013)。
"""

from __future__ import annotations

import numpy as np

from .._compat import bmat, is_sparse, sparse, splu, to_dense
from ..core.quadrature import simplex_quadrature
from ..solvers.linear import ConvergenceError, Solver
from .mixed_poisson import MixedPoissonProblem
from .registry import register_problem

__all__ = ["DarcyProblem"]


@register_problem("darcy")
class DarcyProblem(MixedPoissonProblem):
    """RT0×P0 的 Darcy 渗流问题。

    Parameters
    ----------
    space : RT0Space
        ``FESpace.rt0(mesh)``（三角形或四面体网格）。
    f : callable
        源汇强度 ``f(x)``；``x`` 是 ``(n_points, dim)`` 数组（单点也接受）。
        注水取正、采出取负。
    g : callable
        压力 Dirichlet 数据 ``g_D(x)``，默认 0（可以为非齐次）。
    K : float | (d,) | (d, d) | callable
        渗透率；含义见 :func:`~MosicaFE.spaces.rt0.rt0_permeability_matrices`。
    bc : callable, optional
        ``bc(mid, n_out) -> "D" | "N" | ("flux", gN)``；缺省全边界 ``"D"``。
    mode : {"cell", "quad"}
        加权通量质量矩阵的组装方式，见
        :meth:`~MosicaFE.spaces.rt0.RT0Space.weighted_flux_mass_matrix`。
    quadrature_order : int
        压力载荷与边界积分的求积阶（缺省 6）。

    Attributes
    ----------
    flux_mass : (n_flux, n_flux)
        消去本质自由度之后的加权通量质量矩阵（稀疏）。
    divergence : (n_cells, n_flux)
        散度块（已把给定通量的列清零）。
    source : (n_cells,)
        物理载荷 ``∫_K f``（``check_conservation`` 用的就是它）。
    effective_source : (n_cells,)
        消去已知通量后压力方程的右端 ``F - B_c^{\\mathsf T}q_c``。
    flux_rhs : (n_flux,)
        第一个方程的右端 ``-⟨g_D,φ·n⟩_{Γ_D}``，本质自由度处分量取已知值。
    dirichlet_facets, known_fluxes
        被分类为 ``"D"`` 的面编号列表与 ``{面编号: 给定通量}`` 字典。
    """

    def __init__(self, space, f=None, g=None, K=1.0, bc=None,
                 mode="cell", quadrature_order=None, **kwargs):
        if not getattr(space, "is_mixed", False):
            raise TypeError(
                "DarcyProblem 需要混合空间 FESpace.rt0(mesh)；"
                f"收到 {type(space).__name__}"
            )
        self.K = K
        self.bc = bc if bc is not None else (lambda mid, n_out: "D")
        self.mode = str(mode).lower()
        if self.mode not in ("cell", "quad"):
            raise ValueError(f"mode 只能是 'cell' 或 'quad'，收到 {mode!r}")
        self.flux_rhs = None
        self.effective_source = None
        self.dirichlet_facets = []
        self.known_fluxes = {}
        super().__init__(space, f=f, g=g,
                         quadrature_order=6 if quadrature_order is None
                         else int(quadrature_order),
                         **kwargs)

    # ------------------------------------------------------------------
    def _check_boundary_data(self):
        """Darcy 支持非齐次压力 Dirichlet（自然边界条件），故不做齐次检查。"""
        return

    # ------------------------------------------------------------------
    # 组装
    # ------------------------------------------------------------------
    def assemble(self):
        """组装鞍点系统 ``(A, b)``，未知量顺序为 ``[Φ, P]``。

        ``A = [[A_Φ, -Bᵀ], [B, 0]]``，``b = [b_Φ, F_eff]``；
        中间块同时保存在 ``flux_mass / divergence / source / effective_source``
        上供静力凝聚与守恒检查使用。
        """
        V, mesh, topo = self.space, self.mesh, self.space.topology
        nodes = np.asarray(mesh.nodes, dtype=float)
        nloc = mesh.dim + 1

        A = V.weighted_flux_mass_matrix(self.K, mode=self.mode,
                                        quadrature_order=self.quadrature_order)
        B = V.divergence_matrix()
        F = np.asarray(V.pressure_load_vector(
            self.f, quadrature_order=self.quadrature_order), dtype=float)
        if not is_sparse(A):
            B = to_dense(B)
        flux_rhs = np.zeros(V.n_flux)
        dirichlet, known = [], {}

        # ---- 逐条边界面分类（内部面自由度共享，不需要条件）----
        for c in range(mesh.n_cells):
            for i in range(nloc):
                fct = int(topo.cell_facets[c, i])
                if topo.counts[fct] != 1:
                    continue
                sign = float(topo.signs[c, i])      # +1：全局面法向指向域外
                fverts = nodes[topo.facets[fct]]
                mid = fverts.mean(axis=0)
                normal = np.asarray(topo.normals[fct], dtype=float)
                normal = sign * normal / np.linalg.norm(normal)
                kind = self.bc(mid, normal)
                pts, w = simplex_quadrature(fverts, order=self.quadrature_order)
                meas = float(np.sum(w))
                if isinstance(kind, str) and kind.upper() == "D":
                    dirichlet.append(fct)
                    # 面上 φ_f·n_f = 1/|f| ⇒ b_f = -sign·(1/|f|)∫_f g_D ds
                    gval = V._evaluate_function(self.g, pts)
                    flux_rhs[fct] = -sign * float(np.sum(w * gval)) / meas
                elif isinstance(kind, str) and kind.upper() == "N":
                    known[fct] = 0.0
                elif isinstance(kind, (tuple, list)) and str(kind[0]).lower() == "flux":
                    gN = np.asarray(kind[1](pts), dtype=float)
                    # Φ_f = ∫_f q·n_f ds = sign·∫_f g_N ds
                    known[fct] = sign * float(np.sum(w * gN))
                else:
                    raise ValueError(
                        f"未知边界条件 {kind!r}；应为 'D' / 'N' / ('flux', gN)")

        # ---- 本质边界条件：消去已知通量自由度（限制到 V_h0）----
        F_eff = F.copy()
        if known:
            A, B, flux_rhs, F_eff = self._eliminate_known(
                A, B, flux_rhs, F_eff, known)

        # ---- 子类钩子：可在组装鞍点前修改通量块（如界面接触电阻）----
        A = self._post_process_flux_mass(A)

        A_saddle = bmat([[A, -B.T], [B, None]])
        rhs = np.concatenate([flux_rhs, F_eff])

        self.flux_mass, self.divergence = A, B
        self.source = F                       # 物理载荷（守恒检查用）
        self.effective_source = F_eff         # 消去后的压力方程右端
        self.flux_rhs = flux_rhs
        self.dirichlet_facets, self.known_fluxes = dirichlet, known
        self._pressure_undetermined = not dirichlet
        return A_saddle, rhs

    def _post_process_flux_mass(self, A):
        """子类钩子：在组装鞍点系统前修改通量块 ``A``（默认不改）。

        :class:`~MosicaFE.physics.conduction.ConductionProblem` 用它加界面接触电阻。
        """
        return A

    @staticmethod
    def _eliminate_known(A, B, b, F, known):
        """把 ``Φ_f = 已知值`` 消去：搬列、搬 ``B`` 列、换单位行并清零 ``B`` 的该列。"""
        sparse_A = is_sparse(A)
        if sparse_A and sparse is not None:
            Al = sparse.lil_matrix(A)
        else:
            Al = np.array(A, dtype=float, copy=True)
        if is_sparse(B) and sparse is not None:
            Bl = sparse.lil_matrix(B)
        else:
            Bl = np.array(B, dtype=float, copy=True)

        for fct, val in known.items():
            val = float(val)
            if sparse_A and sparse is not None:
                colA = np.asarray(Al[:, fct].toarray()).ravel()
                colB = np.asarray(Bl[:, fct].toarray()).ravel()
            else:
                colA = np.asarray(Al[:, fct]).ravel()
                colB = np.asarray(Bl[:, fct]).ravel()
            b = b - colA * val                    # (1) A 的已知列 → 第一个方程右端
            F = F - colB * val                    # (2) B 的已知列 → 第二个方程右端
            b[fct] = val
            Al[fct, :] = 0.0                      # (3) 固定 Φ_f
            Al[:, fct] = 0.0
            Al[fct, fct] = 1.0
            Bl[:, fct] = 0.0                      # (3) ★ 切断 -Bᵀ 耦合

        if sparse_A and sparse is not None:
            return Al.tocsr(), Bl.tocsr(), b, F
        return Al, Bl, b, F

    # ------------------------------------------------------------------
    # 求解
    # ------------------------------------------------------------------
    def solve(self, method: str = "auto", **kwargs):
        """求解，返回 ``(Q, P)``。

        ``method``：

        * ``"auto"``（默认）：单元数不超过 2000 用 ``"schur"``，否则 ``"schur_cg"``；
        * ``"schur"``：密 Schur 凝聚（与 :class:`MixedPoissonProblem` 同风格，
          内存 :math:`O(n_{\\text{flux}}\\cdot n_{\\text{cells}})`）；
        * ``"schur_cg"``：稀疏 LU + Schur 补上的 CG（省内存，大网格用）；
        * ``"direct" / "minres" / "gmres" / "bicgstab"``：直接作用于鞍点矩阵
          （纯 Neumann 时不允许，因为矩阵奇异）。
        """
        m = str(method).lower()
        if m == "auto":
            m = "schur" if self.mesh.n_cells <= 2000 else "schur_cg"
        A, b = self.assemble()
        self._system = (A, b)
        if m == "schur":
            sol = self._solve_schur()
        elif m == "schur_cg":
            sol = self._solve_schur_cg(**kwargs)
        elif m == "direct":
            self._require_determined_pressure(m)
            sol = Solver.direct(A, b)
        else:
            self._require_determined_pressure(m)
            sol = Solver.solve(A, b, method=m, symmetric=False, **kwargs)
        self.solution = sol
        return self.extract(sol)

    def _require_determined_pressure(self, method: str):
        if getattr(self, "_pressure_undetermined", False):
            raise ValueError(
                f"纯 Neumann（无压力 Dirichlet）时鞍点矩阵奇异，不能用 method={method!r}。"
                "请改用 method='schur' / 'schur_cg'（会自动用最小二乘并平移到零均值），"
                "或在边界上给出一段压力条件。"
            )

    def _schur_parts(self):
        F = self.effective_source
        if F is None:
            F = self.source
        return self.flux_mass, self.divergence, np.asarray(F, dtype=float), self.flux_rhs

    def _solve_schur(self):
        """密 Schur：``A_Φ Φ = Bᵀ P + b`` ⇒ ``(B A_Φ^{-1} Bᵀ) P = F_eff - B A_Φ^{-1} b``。"""
        M, B, F, b = self._schur_parts()
        rhs = to_dense(B.T)
        if is_sparse(M) and splu is not None:
            lu = splu(M.tocsc())
            X = lu.solve(rhs)
            q0 = lu.solve(b)
        else:
            Md = to_dense(M)
            X = np.linalg.solve(Md, rhs)
            q0 = np.linalg.solve(Md, b)
        S = to_dense(B @ X)
        S = 0.5 * (S + S.T)
        rhsP = F - B @ q0
        if getattr(self, "_pressure_undetermined", False):
            # 纯 Neumann：S 奇异（常数零空间），用最小二乘并平移到零均值
            P = np.linalg.lstsq(S, rhsP, rcond=None)[0]
            P -= P.mean()
        else:
            P = np.linalg.solve(S, rhsP)
        Phi = X @ P + q0
        return np.concatenate([np.asarray(Phi).ravel(), P])

    def _solve_schur_cg(self, tol: float = 1e-12, maxiter: int = 20000):
        """稀疏 Schur：对 ``A_Φ`` 做一次 LU，``B A_Φ^{-1} Bᵀ`` 用 LinearOperator + CG。"""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        M, B, F, b = self._schur_parts()
        lu = spla.splu(sp.csc_matrix(M))
        q0 = lu.solve(b)
        n = B.shape[0]
        op = spla.LinearOperator((n, n),
                                 matvec=lambda P: B @ lu.solve(B.T @ P), dtype=float)
        rhsP = F - B @ q0
        if getattr(self, "_pressure_undetermined", False):
            P = np.linalg.lstsq(to_dense(B @ lu.solve(to_dense(B.T))), rhsP,
                                rcond=None)[0]
            P -= P.mean()
        else:
            P, info = spla.cg(op, rhsP, rtol=tol, atol=0.0, maxiter=maxiter)
            if info != 0:
                raise ConvergenceError(
                    f"Darcy 的 Schur CG 未收敛（info={info}，maxiter={maxiter}）")
        Phi = lu.solve(B.T @ P + b)
        return np.concatenate([np.asarray(Phi).ravel(), P])

    # ------------------------------------------------------------------
    def total_flux(self, facets) -> float:
        """给定面集合上的总法向通量 ``Σ_f Φ_f``（例如出口流量）。

        ``facets`` 可以是面编号序列，或 ``"boundary"`` 表示全部边界面。
        """
        if self.flux is None:
            raise RuntimeError("请先调用 solve()")
        Phi = np.asarray(self.solution, dtype=float)[: self.space.n_flux]
        if isinstance(facets, str) and facets.lower() == "boundary":
            idx = np.flatnonzero(self.space.topology.boundary_mask)
        else:
            idx = np.asarray(list(facets), dtype=np.int64)
        return float(Phi[idx].sum())

    def __repr__(self):  # pragma: no cover - 仅展示
        return (
            f"<DarcyProblem space={self.space.name} dim={self.mesh.dim} "
            f"flux={self.space.n_flux} pressure={self.space.n_pressure} "
            f"mode={self.mode}>"
        )
