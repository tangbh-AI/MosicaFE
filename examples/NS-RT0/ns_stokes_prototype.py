"""定常不可压 Navier--Stokes 方程（Stokes / Oseen / Picard 非线性）。

方程
----
.. math::

   \\rho(\\boldsymbol u\\cdot\\nabla)\\boldsymbol u-\\mu\\Delta\\boldsymbol u+\\nabla p=\\boldsymbol f,
   \\qquad \\nabla\\cdot\\boldsymbol u=0 \\quad\\text{in }\\Omega,

边界：速度 Dirichlet :math:`\\boldsymbol u=\\boldsymbol g` 于 :math:`\\Gamma_D`
（壁面 :math:`\\boldsymbol g=0`、入口剖面、顶盖驱动），"do-nothing" 出流
:math:`\\mu\\partial_n\\boldsymbol u-p\\boldsymbol n=0` 于 :math:`\\Gamma_N`。

.. warning::

   **速度不能用 RT0**。RT0 场形如 :math:`\\boldsymbol u=\\boldsymbol a+b\\boldsymbol x`，于是

   .. math::

      \\nabla\\boldsymbol u=b\\boldsymbol I,\\qquad \\nabla\\times\\boldsymbol u\\equiv 0 ,

   它**无旋**，表达不了任何剪切（Poiseuille 剖面、边界层、涡）；再叠加
   :math:`\\nabla\\cdot\\boldsymbol u=0` 还会把速度逼成分片常数。
   RT0 的合适位置是**无旋通量**（Darcy 的 :math:`\\boldsymbol q=-K\\nabla p`）或混合法的
   应力/梯度。粘性速度要用含剪切的 :math:`H^1` 型空间。

   因此本模块取 **Taylor--Hood 对**：速度 :math:`(P_2)^d`、压力 :math:`P_1`（连续），
   无条件满足 inf-sup 稳定；RT0 只承担
   :meth:`NavierStokesProblem.flux_rt0` 的**守恒通量后处理**。

离散
----
.. math::

   \\left\\{
   \\begin{aligned}
   &\\rho\\,(\\boldsymbol b\\cdot\\nabla\\boldsymbol u_h,\\boldsymbol v_h)
    +\\mu(\\nabla\\boldsymbol u_h,\\nabla\\boldsymbol v_h)
    -(p_h,\\nabla\\cdot\\boldsymbol v_h)=(\\boldsymbol f,\\boldsymbol v_h),\\\\
   &(\\nabla\\cdot\\boldsymbol u_h,q_h)=0,
   \\end{aligned}
   \\right.

非线性用 Picard 迭代（:math:`\\boldsymbol b` 取上一迭代的速度），每步解鞍点系统
:math:`\\begin{pmatrix}A&-B^{\\mathsf T}\\\\B&0\\end{pmatrix}`。

.. code-block:: python

   from MosicaFE import FESpace, Mesh, NavierStokesProblem

   mesh = Mesh.rectangle(nx=32, ny=32)
   V = FESpace.rt0(mesh)                       # 仅用于通量后处理
   bc = lambda mid, n: "D" if (mid[1] < 1e-12 or mid[1] > 1 - 1e-12) else "N"
   problem = NavierStokesProblem(V, f=lambda x: np.array([1.0, 0.0]),
                                 mu=1.0, bc=bc)          # 通道 Poiseuille
   U, P = problem.solve()                      # U: (n_u,), P: (n_p,)
   Phi = problem.flux_rt0()                    # RT0 面通量（跨单元连续、局部守恒）

若需要"纯 RT0"的粘性格式，必须把速度换成 BDM/RT1（div 值域到 P1）或应力型
（PEERS）空间——报告里给出了对应组装，本库当前没有这些速度空间。

参考文献：Taylor & Hood (1973)；Girault & Raviart, *Finite Element Methods for
Navier--Stokes Equations* (1986)；Ern & Guermond, *Theory and Practice of FEM* (2004)。
"""

from __future__ import annotations

import numpy as np

from MosicaFE.core.quadrature import QuadratureRule, cell_quadrature
from MosicaFE.solvers.linear import Solver
from MosicaFE.spaces.lagrange import LagrangeSpace

__all__ = ["NavierStokesProblem"]


class NavierStokesProblem:
    """定常不可压 NS（Stokes / Oseen / Picard 非线性），Taylor--Hood 离散。

    Parameters
    ----------
    space : BaseFESpace | Mesh
        只需要它的 ``mesh``（通常传 ``FESpace.rt0(mesh)``，这样还能用 RT0 通量后处理）。
    f : callable
        体积力 ``f(x)``，返回长度 ``dim`` 的向量（也接受 ``(nq, dim)`` 数组）。
    g : callable
        速度 Dirichlet 数据；默认 0。接受单点或 ``(nq, dim)`` 数组。
    bc : callable
        ``bc(mid, n_out) -> "D" | "N"``：``"D"`` 速度给定、``"N"`` do-nothing 出流。
    mu, rho : float
        动力粘度、密度。
    degree : (int, int)
        速度/压力的多项式次数，缺省 ``(2, 1)``（Taylor--Hood）。
    """

    is_mixed = True

    def __init__(self, space, f=None, g=None, bc=None, mu: float = 1.0,
                 rho: float = 1.0, degree=(2, 1), quadrature_order: int = 6,
                 **_kwargs):
        self.mesh = getattr(space, "mesh", space)
        self.space = space
        self.dim = int(self.mesh.dim)
        self.n_cells = int(self.mesh.n_cells)
        self.mu = float(mu)
        self.rho = float(rho)
        self.degree = (int(degree[0]), int(degree[1]))
        self.order = int(quadrature_order)
        self.f = f if f is not None else (lambda x: np.zeros(self.dim))
        self.g = g if g is not None else (lambda x: np.zeros(self.dim))
        self.bc = bc if bc is not None else (lambda mid, n_out: "D")

        self.Vu = LagrangeSpace(self.mesh, degree=self.degree[0])
        self.Vp = LagrangeSpace(self.mesh, degree=self.degree[1])
        self.n_u = self.dim * self.Vu.n_dofs
        self.n_p = self.Vp.n_dofs
        self.n_dofs = self.n_u + self.n_p
        self.n_loc_u = int(self.Vu.dof_map[0].size)
        self.n_loc_p = int(self.Vp.dof_map[0].size)

        self.solution = None
        self.velocity = None
        self.pressure = None
        self.picard_history = []
        self._bc_data = self._build_dirichlet()

    # ------------------------------------------------------------------
    def _eval_vec(self, func, pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        try:
            out = np.asarray(func(pts), dtype=float)
            if out.shape == (pts.shape[0], self.dim):
                return out
            if out.shape == (self.dim,):
                return np.broadcast_to(out, (pts.shape[0], self.dim))
        except Exception:
            pass
        return np.array([np.asarray(func(p), dtype=float).ravel() for p in pts])

    def _topology(self):
        if not hasattr(self, "_topo"):
            from MosicaFE.core.facets import build_facet_topology

            self._topo = build_facet_topology(self.mesh)
        return self._topo

    def _build_dirichlet(self):
        """Γ_D 上速度自由度（nodal）与给定值。"""
        nodes = np.asarray(self.mesh.nodes, dtype=float)
        topo = self._topology()
        dofs, vals = [], []
        for f in topo.boundary_facets():
            fverts = nodes[topo.facets[f]]
            n_out = np.asarray(topo.normals[f], dtype=float)
            n_out = n_out / np.linalg.norm(n_out)
            if str(self.bc(fverts.mean(axis=0), n_out)).upper() != "D":
                continue
            for v in topo.facets[f]:
                gv = np.asarray(self.g(nodes[v]), dtype=float).ravel()
                for i in range(self.dim):
                    dofs.append(i * self.Vu.n_dofs + int(v))
                    vals.append(float(gv[i]))
        if not dofs:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        dofs = np.asarray(dofs, dtype=np.int64)
        vals = np.asarray(vals)
        uniq, idx = np.unique(dofs, return_index=True)
        return uniq, vals[idx]

    # ------------------------------------------------------------------
    def assemble(self, u_current=None):
        """组装鞍点系统 ``(A_saddle, b)``；``u_current`` 非零时含对流项。"""
        dim, mu, rho = self.dim, self.mu, self.rho
        nu, npp = self.n_u, self.n_p
        nodes = np.asarray(self.mesh.nodes, dtype=float)
        etype = "triangle" if dim == 2 else "tet"
        ref_q = QuadratureRule.get_quadrature(etype, self.order)
        xi = ref_q.points

        rows, cols, vals = [], [], []
        Mp_r, Mp_c, Mp_v = [], [], []
        N_r, N_c, N_v = [], [], []
        rhs = np.zeros(nu)

        def put(r, c, v):
            rows.append(r); cols.append(c); vals.append(v)

        for c in range(self.n_cells):
            tri = np.asarray(self.mesh.elements[c], dtype=np.int64)
            verts = nodes[tri]
            J = np.column_stack([verts[i + 1] - verts[0] for i in range(dim)])
            detJ = abs(np.linalg.det(J))
            Jinv = np.linalg.inv(J)
            X = verts[0] + xi @ J.T
            w = ref_q.weights * detJ

            phi_u = self.Vu.eval_basis(xi)                      # (nq, nu_loc)
            gxi_u = self.Vu.eval_basis_grad(xi)                  # (nq, nu_loc, d)
            grad_u = np.einsum("jk,qbj->qbk", Jinv, gxi_u)
            phi_p = self.Vp.eval_basis(xi)                       # (nq, np_loc)

            ud = np.asarray(self.Vu.dof_map[c], dtype=np.int64)
            pd = np.asarray(self.Vp.dof_map[c], dtype=np.int64)

            Kmat = np.einsum("q,qak,qbk->ab", w, grad_u, grad_u)
            for i in range(dim):
                for a in range(self.n_loc_u):
                    ga = i * self.Vu.n_dofs + ud[a]
                    for b in range(self.n_loc_u):
                        put(ga, i * self.Vu.n_dofs + ud[b], mu * Kmat[a, b])

            if u_current is not None and rho != 0.0:
                uc = np.asarray(u_current, dtype=float)
                bq = np.zeros((len(w), dim))
                for a in range(self.n_loc_u):
                    for i in range(dim):
                        bq[:, i] += uc[i * self.Vu.n_dofs + ud[a]] * phi_u[:, a]
                Cm = np.einsum("q,qj,qbj,qa->ab", w, bq, grad_u, phi_u)
                for i in range(dim):
                    for a in range(self.n_loc_u):
                        ga = i * self.Vu.n_dofs + ud[a]
                        for b in range(self.n_loc_u):
                            put(ga, i * self.Vu.n_dofs + ud[b], rho * Cm[a, b])

            fv = self._eval_vec(self.f, X)
            for i in range(dim):
                for a in range(self.n_loc_u):
                    rhs[i * self.Vu.n_dofs + ud[a]] += np.sum(w * fv[:, i] * phi_u[:, a])

            for i in range(dim):
                Ci = np.einsum("q,qa,qb->ab", w, phi_p, grad_u[:, :, i])
                for a in range(self.n_loc_p):
                    for b in range(self.n_loc_u):
                        Mp_r.append(i * self.Vu.n_dofs + ud[b])
                        Mp_c.append(pd[a])
                        Mp_v.append(-Ci[a, b])
                        N_r.append(pd[a])
                        N_c.append(i * self.Vu.n_dofs + ud[b])
                        N_v.append(Ci[a, b])

        A = self._mat(rows, cols, vals, (nu, nu))
        M = self._mat(Mp_r, Mp_c, Mp_v, (nu, npp))
        N = self._mat(N_r, N_c, N_v, (npp, nu))
        b = np.concatenate([rhs, np.zeros(npp)])
        from MosicaFE._compat import bmat

        self.A_velocity, self.M_pressure, self.N_pressure = A, M, N
        self.rhs = rhs
        return bmat([[A, M], [N, None]]), b

    @staticmethod
    def _mat(r, c, v, shape):
        from MosicaFE._compat import from_triplets

        if not r:
            return from_triplets(np.zeros(0, int), np.zeros(0, int), np.zeros(0), shape)
        return from_triplets(np.asarray(r), np.asarray(c), np.asarray(v), shape)

    # ------------------------------------------------------------------
    def solve(self, solver: str = "direct", max_iter: int = 50, tol: float = 1e-10,
              verbose: bool = False):
        """Picard 迭代求解，返回 ``(U, P)``。"""
        U = np.zeros(self.n_u)
        P = np.zeros(self.n_p)
        self.picard_history = []
        dofs, vals = self._bc_data
        pin = self._needs_pin()
        for it in range(max_iter + 1):
            A, b = self.assemble(u_current=(U if self.rho != 0.0 else None))
            if dofs.size:
                A, b = Solver.apply_dirichlet(A, b, dofs, vals)
            if pin:
                A, b = self._add_pin(A, b)
            sol = Solver.direct(A, b) if solver == "direct" else Solver.solve(
                A, b, method=solver, symmetric=False)
            U_new = sol[: self.n_u]
            P_new = sol[self.n_u: self.n_u + self.n_p]
            if pin:
                P_new = P_new - P_new.mean()
            res = np.linalg.norm(U_new - U) / max(np.linalg.norm(U_new), 1e-30)
            self.picard_history.append(float(res))
            U, P = U_new, P_new
            if verbose:
                print(f"    Picard {it:2d}: |Δu|/|u| = {res:.3e}")
            if res < tol:
                break
        self.velocity, self.pressure = U, P
        self.solution = np.concatenate([U, P])
        return U, P

    def _needs_pin(self):
        """全边界都是速度 Dirichlet 时，压力只定到常数，需要固定。"""
        nodes = np.asarray(self.mesh.nodes, dtype=float)
        topo = self._topology()
        for f in topo.boundary_facets():
            n_out = np.asarray(topo.normals[f], dtype=float)
            n_out = n_out / np.linalg.norm(n_out)
            mid = nodes[topo.facets[f]].mean(axis=0)
            if str(self.bc(mid, n_out)).upper() == "N":
                return False
        return True

    def _add_pin(self, A, b):
        """为压力常数零空间追加 Lagrange 乘子（约束 ∫p = 0）。"""
        from MosicaFE._compat import sparse

        n = self.n_u + self.n_p
        if sparse is not None:
            col = sparse.vstack([sparse.csr_matrix((self.n_u, 1)),
                                 sparse.csr_matrix(np.ones((self.n_p, 1)))])
            row = col.T.tocsr()
            A3 = sparse.bmat([[A, col], [row, None]], format="csr")
        else:
            A3 = np.zeros((n + 1, n + 1))
            A3[:n, :n] = A
            A3[self.n_u:n, n] = 1.0
            A3[n, self.n_u:n] = 1.0
        return A3, np.concatenate([b, [0.0]])

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------
    def velocity_at(self, cell, points):
        """单元 ``cell`` 内速度 ``(nq, dim)``。"""
        nodes = np.asarray(self.mesh.nodes, dtype=float)
        verts = nodes[np.asarray(self.mesh.elements[cell], dtype=np.int64)]
        J = np.column_stack([verts[i + 1] - verts[0] for i in range(self.dim)])
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        xi = np.linalg.solve(J, (pts - verts[0]).T).T
        phi = self.Vu.eval_basis(xi)
        ud = np.asarray(self.Vu.dof_map[cell], dtype=np.int64)
        out = np.zeros((pts.shape[0], self.dim))
        for i in range(self.dim):
            for a in range(self.n_loc_u):
                out[:, i] += self.velocity[i * self.Vu.n_dofs + ud[a]] * phi[:, a]
        return out

    def pressure_at(self, cell, points):
        """单元 ``cell`` 内压力 ``(nq,)``。"""
        nodes = np.asarray(self.mesh.nodes, dtype=float)
        verts = nodes[np.asarray(self.mesh.elements[cell], dtype=np.int64)]
        J = np.column_stack([verts[i + 1] - verts[0] for i in range(self.dim)])
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        xi = np.linalg.solve(J, (pts - verts[0]).T).T
        phi = self.Vp.eval_basis(xi)
        pd = np.asarray(self.Vp.dof_map[cell], dtype=np.int64)
        return phi @ self.pressure[pd]

    def velocity_divergence(self, cell):
        """单元内某点的 ``∇·u_h``（取参考重心处）。"""
        nodes = np.asarray(self.mesh.nodes, dtype=float)
        verts = nodes[np.asarray(self.mesh.elements[cell], dtype=np.int64)]
        J = np.column_stack([verts[i + 1] - verts[0] for i in range(self.dim)])
        Jinv = np.linalg.inv(J)
        gxi = self.Vu.eval_basis_grad(np.full((1, self.dim), 1.0 / (self.dim + 1)))
        grad = np.einsum("jk,qbj->qbk", Jinv, gxi)
        ud = np.asarray(self.Vu.dof_map[cell], dtype=np.int64)
        return float(sum(self.velocity[i * self.Vu.n_dofs + ud[a]] * grad[0, a, i]
                         for i in range(self.dim) for a in range(self.n_loc_u)))

    def errors(self, exact_u, exact_p=None, order: int = 8):
        """``(||u-u_h||_L2, ||p-p_h||_L2, max|div u_h|)``；压力比较前先扣除常数差。"""
        eu = 0.0
        for c in range(self.n_cells):
            pts, w = cell_quadrature(self.mesh, c, order)
            uh = self.velocity_at(c, pts)
            ue = self._eval_vec(exact_u, pts)
            eu += np.sum(w * np.sum((uh - ue) ** 2, axis=1))
        div = max(abs(self.velocity_divergence(c)) for c in range(self.n_cells))
        ep = float("nan")
        if exact_p is not None:
            num = den = 0.0
            for c in range(self.n_cells):
                pts, w = cell_quadrature(self.mesh, c, order)
                pe = np.asarray([float(np.asarray(exact_p(p)).ravel()[0]) for p in pts])
                num += np.sum(w * (self.pressure_at(c, pts) - pe))
                den += np.sum(w)
            shift = num / max(den, 1e-30)          # p 只定到常数
            ep2 = 0.0
            for c in range(self.n_cells):
                pts, w = cell_quadrature(self.mesh, c, order)
                pe = np.asarray([float(np.asarray(exact_p(p)).ravel()[0]) for p in pts])
                ep2 += np.sum(w * (self.pressure_at(c, pts) - pe - shift) ** 2)
            ep = float(np.sqrt(ep2))
        return float(np.sqrt(eu)), ep, float(div)

    def flux_rt0(self):
        """把速度投影到 RT0 的面法向通量自由度（跨单元法向连续、逐单元守恒）。"""
        if not hasattr(self.space, "interpolate_flux"):
            raise TypeError("flux_rt0 需要传入 RT0 空间（FESpace.rt0(mesh)）")

        def q(x):
            x = np.asarray(x, dtype=float).ravel()
            return self.velocity_at(self._locate(x), x[None, :])[0]

        return self.space.interpolate_flux(q)

    def _locate(self, x):
        nodes = np.asarray(self.mesh.nodes, dtype=float)
        best, bestc = -np.inf, 0
        for c in range(self.n_cells):
            verts = nodes[np.asarray(self.mesh.elements[c], dtype=np.int64)]
            J = np.column_stack([verts[i + 1] - verts[0] for i in range(self.dim)])
            xi = np.linalg.solve(J, np.asarray(x, float) - verts[0])
            m = float(min(xi.min(), 1.0 - xi.sum()))
            if m > best:
                best, bestc = m, c
            if m >= -1e-12:
                return c
        return bestc

    def __repr__(self):  # pragma: no cover
        return (f"<NavierStokesProblem dim={self.dim} cells={self.n_cells} "
                f"n_u={self.n_u} n_p={self.n_p} mu={self.mu} rho={self.rho} "
                f"degree={self.degree}>")
