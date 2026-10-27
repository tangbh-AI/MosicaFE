"""定常电流场（DC conduction / 接地电阻）问题（RT0×P0 混合形式）。

方程
----
欧姆定律 + 电荷守恒（定常）::

    J = -sigma grad(phi)          （欧姆定律）
    div J = f                     （电荷守恒；f 为体注入电流密度）

其中 ``phi`` 是电势（V）、``J`` 是电流密度（A/m²）、``sigma`` 是电导率（S/m，
对称正定张量）。与 :class:`~MosicaFE.physics.darcy.DarcyProblem` **完全同构**：
``sigma <-> K``、``phi <-> p``、``J <-> q``，因此复用同一套鞍点装配与 Schur 求解。

边界条件
--------
=====================  ================================  =====================
``bc(mid, n_out)``     物理含义                           在混合格式中的处理
=====================  ================================  =====================
``"D"``                电极电势给定 ``phi = g_D``        自然边界，进右端
``"N"``                绝缘 ``J·n = 0``                  本质边界，自由度置零
``("flux", gN)``       注入电流密度 ``J·n = g_N``        本质边界，取已知值
=====================  ================================  =====================

界面接触电阻
------------
在内部面 :math:`\\Gamma_c` 上允许电势跳变，用**面接触电阻率**
:math:`R_c`（Ω·m²，二维按单位厚度计）描述::

    [[phi]] = R_c (J·n)      on  Γ_c

混合格式里这正好是一个**对角修正**。由分部积分

.. math::

   (\\sigma^{-1}J,v)-(\\phi,\\nabla\\cdot v)+\\langle[[\\phi]],v\\cdot n\\rangle
   +\\langle\\phi,v\\cdot n\\rangle_{\\partial\\Omega}=0,

其中 :math:`\\langle[[\\phi]],v_f\\cdot n\\rangle=\\phi^- - \\phi^+`（面上 :math:`v_f\\cdot n=1/|f|`）。
标准方法把这一项当作 0（电势连续）；有接触电阻时它等于

.. math::

   \\langle R_c(J\\cdot n),v_f\\cdot n\\rangle=\\frac{R_c}{|f|}\\Phi_f,

所以只需在通量块的该自由度上加 :math:`R_c/|f|`：

.. math::

   A_{ff}\\;\\leftarrow\\;A_{ff}+\\frac{R_c(f)}{|f|}.

``contact`` 是一个可调用对象 ``contact(mid) -> R_c``（返回 ``None`` 或 ``0`` 表示理想接触）。

后处理
------
* :meth:`total_current`：某个面集合上的总电流 :math:`I=\\sum_f\\Phi_f`（法向朝外为正）；
* :meth:`resistance`：:math:`R=\\Delta\\phi/I`；
* :meth:`joule_heating`：焦耳热 :math:`P=(\\sigma^{-1}J,J)`（含接触电阻上的耗散）。

.. code-block:: python

   from MosicaFE import FESpace, Mesh, ConductionProblem

   mesh = Mesh.rectangle(nx=32, ny=32)
   V = FESpace.rt0(mesh)

   # 左电极 1 V、右电极 0 V、上下绝缘，电导率 2.0：J=(2,0)，R = 1/2
   bc = lambda mid, n: "D" if (mid[0] < 1e-12 or mid[0] > 1 - 1e-12) else "N"
   problem = ConductionProblem(V, g=lambda x: 1.0 - x[:, 0], sigma=2.0, bc=bc)
   J, phi = problem.solve()
   print("总电流", problem.total_current("right"))     # ≈ 2.0
   print("电阻", problem.resistance(1.0, "right"))      # ≈ 0.5

参见 Bear (1972) 的渗流–导电类比；混合元用于电流场见 Boffi--Brezzi--Fortin (2013)。
"""

from __future__ import annotations

import numpy as np

from .._compat import is_sparse, sparse
from .darcy import DarcyProblem
from .registry import register_problem

__all__ = ["ConductionProblem"]


@register_problem("conduction")
class ConductionProblem(DarcyProblem):
    """RT0×P0 的定常电流场问题。

    Parameters
    ----------
    space : RT0Space
        ``FESpace.rt0(mesh)``。
    f : callable
        体注入电流密度 ``f(x)``（源汇），默认 0。
    g : callable
        电极电势 ``g_D(x)``，默认 0。
    sigma : float | (d,) | (d, d) | callable
        电导率（标量 / 对角 / 张量 / 空间变系数）。
    bc : callable, optional
        ``bc(mid, n_out) -> "D" | "N" | ("flux", gN)``。
    contact : callable, optional
        ``contact(mid) -> R_c``，内部面上的接触电阻率；``None``/``0`` 为理想接触。
    mode : {"cell", "quad"}
        加权通量质量矩阵的组装方式。
    """

    def __init__(self, space, f=None, g=None, sigma=1.0, bc=None, contact=None,
                 mode="cell", quadrature_order=None, **kwargs):
        self.sigma = sigma
        self.contact = contact
        super().__init__(space, f=f, g=g, K=sigma, bc=bc, mode=mode,
                         quadrature_order=quadrature_order, **kwargs)
        self.contact_facets = self._find_contact_facets()

    # ------------------------------------------------------------------
    # 语义别名：电流 / 电势
    # ------------------------------------------------------------------
    @property
    def current(self):
        """电流密度面积分（逐单元外法向通量 ``(n_cells, d+1)``）。"""
        return self.flux

    @property
    def potential(self):
        """电势（``P0``，每单元一个常数）。"""
        return self.pressure

    # ------------------------------------------------------------------
    def _find_contact_facets(self):
        """扫描内部面，收集 ``(面编号, 接触电阻率)``。"""
        if self.contact is None:
            return []
        topo = self.space.topology
        nodes = np.asarray(self.mesh.nodes, dtype=float)
        out = []
        for f in range(topo.n_facets):
            if topo.counts[f] != 2:                 # 只处理内部面
                continue
            mid = nodes[topo.facets[f]].mean(axis=0)
            rc = self.contact(mid)
            if rc is None:
                continue
            rc = float(rc)
            if rc > 0.0:
                out.append((int(f), rc))
        return out

    def _post_process_flux_mass(self, A):
        """在通量块的对角加上 ``R_c/|f|``（接触电阻的弱形式）。"""
        if not self.contact_facets:
            return A
        meas = self.space.topology.measure()
        add = [(f, rc / float(meas[f])) for f, rc in self.contact_facets]
        if is_sparse(A) and sparse is not None:
            Al = sparse.lil_matrix(A)
            for f, val in add:
                Al[f, f] = Al[f, f] + val
            return Al.tocsr()
        A2 = np.array(A, dtype=float, copy=True)
        for f, val in add:
            A2[f, f] += val
        return A2

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------
    def _facet_cell_local(self, facet):
        """面 → (相邻单元, 局部面号)；边界面唯一，内部面取第一个相邻单元。"""
        topo = self.space.topology
        for c in topo.facet_cells[facet]:
            i = int(np.flatnonzero(np.asarray(topo.cell_facets[c]) == facet)[0])
            return int(c), i
        raise KeyError(facet)

    def total_current(self, facets="boundary") -> float:
        """面集合上的总电流 ``I``（用**逐单元外法向**通量求和，符号明确）。

        ``facets`` 可以是

        * 面编号序列；
        * 可调用对象 ``pred(mid) -> bool``（按面中点筛选，例如右电极）；
        * ``"boundary"``（全部边界面，封闭域上应为 0，可作守恒检查）。
        """
        topo = self.space.topology
        nodes = np.asarray(self.mesh.nodes, dtype=float)
        if callable(facets):
            ids = [f for f in range(topo.n_facets)
                   if topo.counts[f] == 1
                   and bool(facets(nodes[topo.facets[f]].mean(axis=0)))]
        elif isinstance(facets, str):
            ids = list(topo.boundary_facets())
        else:
            ids = [int(f) for f in facets]
        total = 0.0
        for f in ids:
            c, i = self._facet_cell_local(f)
            total += float(self.flux[c, i])          # 该单元的外法向通量
        return float(total)

    def resistance(self, delta_phi: float, facets) -> float:
        """电阻 ``R = Δφ / |I|``。

        ``delta_phi`` 为电极间电势差，``facets`` 为出流电极（必需，含义同
        :meth:`total_current`；不要用 ``"boundary"``，封闭域上净电流为零）。
        """
        i = abs(self.total_current(facets))
        if i == 0.0:
            raise ZeroDivisionError("总电流为零，无法定义电阻")
        return float(delta_phi) / i

    def joule_heating(self) -> float:
        """焦耳热功率 ``P = (σ^{-1}J, J)``（通量块二次型，含接触电阻耗散）。"""
        if self.solution is None:
            raise RuntimeError("请先调用 solve()")
        Phi = np.asarray(self.solution, dtype=float)[: self.space.n_flux]
        A = self.flux_mass
        APhi = A @ Phi if not isinstance(A, np.ndarray) else A.dot(Phi)
        return float(Phi @ np.asarray(APhi).ravel())

    def check_charge_conservation(self, tol: float = 1e-8) -> float:
        """逐单元电荷守恒偏差（与 ``check_conservation`` 相同）。"""
        return self.check_conservation(tol=tol)

    def __repr__(self):  # pragma: no cover - 仅展示
        return (f"<ConductionProblem dim={self.mesh.dim} cells={self.mesh.n_cells} "
                f"flux={self.space.n_flux} potential={self.space.n_pressure} "
                f"contacts={len(self.contact_facets)}>")
