# -*- coding: utf-8 -*-
"""物理问题、线性求解器、高层 API 与扩展点的测试。"""

from __future__ import annotations

import numpy as np
import pytest

import MosicaFE as mf
from MosicaFE.physics.base import BasePhysics
from MosicaFE.physics.registry import available_problems, register_problem
from MosicaFE.solvers.linear import Solver


def exact2(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def source2(x):
    return 2.0 * np.pi ** 2 * exact2(x)


def grad2(x):
    return np.array(
        [
            np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]),
            np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
        ]
    )


def zero(x):
    return 0.0


# ---------------------------------------------------------------------------
# 求解器
# ---------------------------------------------------------------------------
def test_dirichlet_rhs_correction():
    """非齐次 Dirichlet 必须把边界列的贡献搬到右端项。"""
    A = np.array([[2.0, -1.0], [-1.0, 2.0]])
    b = np.zeros(2)
    Abc, bbc = Solver.apply_dirichlet(A, b, [0], [3.0])
    sol = np.linalg.solve(Abc, bbc)
    assert sol[0] == pytest.approx(3.0)
    assert sol[1] == pytest.approx(1.5)


def test_dirichlet_sparse_matches_dense():
    rng = np.random.default_rng(0)
    A = rng.random((6, 6))
    A = A + A.T + 6 * np.eye(6)
    b = rng.random(6)
    dofs = np.array([0, 4])
    vals = np.array([1.5, -2.0])
    As, bs = Solver.apply_dirichlet(A, b, dofs, vals)
    sol = np.linalg.solve(As, bs)
    assert sol[0] == pytest.approx(1.5)
    assert sol[4] == pytest.approx(-2.0)

    import scipy.sparse as sp

    A2, b2 = Solver.apply_dirichlet(sp.csr_matrix(A), b, dofs, vals)
    sol2 = np.linalg.solve(A2.toarray(), b2)
    assert np.abs(sol - sol2).max() < 1e-12


def test_solver_agreement():
    mesh = mf.Mesh.rectangle(nx=16, ny=16)
    V = mf.make_space(mesh, "P2")
    problem = mf.PoissonProblem(V, f=source2, g=zero)
    A, b = problem.assemble()
    x_direct = Solver.direct(A, b)
    x_cg = Solver.conjugate_gradient(A, b, tol=1e-12)
    assert np.abs(x_direct - x_cg).max() < 1e-7


def test_solver_recommendation():
    assert Solver.recommend(100, True) == "direct"
    assert Solver.recommend(10 ** 6, True) == "cg"
    assert Solver.recommend(10 ** 6, False) == "gmres"
    with pytest.raises(ValueError):
        Solver.solve(np.eye(2), np.ones(2), method="nope")


def test_gmres_solves_nonsymmetric_dense():
    rng = np.random.default_rng(1)
    A = rng.random((30, 30)) + 5 * np.eye(30)
    b = rng.random(30)
    x = Solver.gmres(A, b, tol=1e-12)
    assert np.abs(A @ x - b).max() < 1e-8


# ---------------------------------------------------------------------------
# 物理问题
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("elem,expected_l2", [("P1", 1.9), ("P2", 2.8), ("VEM", 1.8)])
def test_convergence_2d(elem, expected_l2):
    hs, l2 = [], []
    for n in (4, 8, 16):
        mesh = mf.Mesh.rectangle(nx=n, ny=n)
        V = mf.make_space(mesh, elem)
        problem = mf.PoissonProblem(V, f=source2, g=zero)
        u = problem.solve()
        hs.append(mesh.get_mesh_size())
        l2.append(problem.compute_error(u, exact2, "L2"))
    assert mf.estimate_convergence_rate(hs, l2, use_last=2) > expected_l2


def test_convergence_vem_polygon():
    """VEM 在 Voronoi 多边形网格上同样保持二阶 L2 收敛。"""
    hs, l2 = [], []
    for npts in (50, 200, 800):
        mesh = mf.Mesh.voronoi_polygon(n_points=npts, seed=7)
        V = mf.FESpace.vem(mesh)
        problem = mf.PoissonProblem(V, f=source2, g=zero)
        u = problem.solve()
        hs.append(float(np.mean([mesh.get_cell_diameter(c) for c in range(mesh.n_cells)])))
        l2.append(problem.compute_error(u, exact2, "L2"))
    assert mf.estimate_convergence_rate(hs, l2, use_last=2) > 1.6


def test_error_norm_types():
    mesh = mf.Mesh.rectangle(nx=8, ny=8)
    V = mf.make_space(mesh, "P2")
    problem = mf.PoissonProblem(V, f=source2, g=zero)
    u = problem.solve()
    assert problem.compute_error(u, exact2, "L2") > 0
    assert problem.compute_error(u, exact2, "Linf") > 0
    assert problem.compute_error(u, exact2, "H1", exact_grad=grad2) > 0
    with pytest.raises(ValueError):
        problem.compute_error(u, exact2, "H1")


def test_reaction_diffusion_recovers_poisson():
    """c → 0 时反应扩散退化为 Poisson。"""
    mesh = mf.Mesh.rectangle(nx=16, ny=16)
    V = mf.make_space(mesh, "P1")
    a = mf.PoissonProblem(V, f=source2, g=zero).solve()
    V2 = mf.make_space(mesh, "P1")
    b = mf.ReactionDiffusionProblem(V2, f=source2, g=zero, reaction_coeff=0.0).solve()
    assert np.abs(a - b).max() < 1e-12


# ---------------------------------------------------------------------------
# 亥姆霍兹方程（内置方程）
# ---------------------------------------------------------------------------
def helmholtz_source(k):
    """制造解 u = sin(pi x) sin(pi y) 对应的源项：f = (2 pi^2 - k^2) u。"""
    return lambda x: (2.0 * np.pi ** 2 - k ** 2) * exact2(x)


def test_helmholtz_registered():
    assert "helmholtz" in available_problems()
    assert mf.get_problem_class("helmholtz") is mf.HelmholtzProblem


def test_helmholtz_recovers_poisson_and_scales_with_kappa():
    """k = 0 退化为 Poisson；kappa 只是常数因子。"""
    mesh = mf.Mesh.rectangle(nx=16, ny=16)
    a = mf.PoissonProblem(mf.make_space(mesh, "P1"), f=source2, g=zero).solve()
    b = mf.HelmholtzProblem(mf.make_space(mesh, "P1"), f=source2, g=zero,
                            k=0.0).solve()
    assert np.abs(a - b).max() < 1e-12

    c = mf.HelmholtzProblem(mf.make_space(mesh, "P1"), f=source2, g=zero,
                            k=0.0, kappa=2.5).solve()
    d = mf.PoissonProblem(mf.make_space(mesh, "P1"), f=source2, g=zero,
                          kappa=2.5).solve()
    assert np.abs(c - d).max() < 1e-12


@pytest.mark.parametrize("elem,expected_l2", [("P1", 1.9), ("P2", 2.8), ("VEM", 1.8)])
def test_helmholtz_convergence_2d(elem, expected_l2):
    """低频（k = 1）时亥姆霍兹的收敛阶与 Poisson 相同。"""
    k = 1.0
    hs, l2 = [], []
    for n in (4, 8, 16):
        mesh = mf.Mesh.rectangle(nx=n, ny=n)
        V = mf.make_space(mesh, elem)
        problem = mf.HelmholtzProblem(V, f=helmholtz_source(k), g=zero, k=k)
        u = problem.solve(method="direct")
        hs.append(mesh.get_mesh_size())
        l2.append(problem.compute_error(u, exact2, "L2"))
    assert mf.estimate_convergence_rate(hs, l2, use_last=2) > expected_l2


def test_helmholtz_definiteness_switches_at_resonance():
    """低频 k^2 < lambda_1 时 A 对称正定；越过第一特征值后出现负特征值。"""
    mesh = mf.Mesh.rectangle(nx=16, ny=16)
    V = mf.make_space(mesh, "P1")

    def smallest_eigenvalue(k):
        A, _ = mf.HelmholtzProblem(V, f=source2, g=zero, k=k).assemble()
        dense = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
        assert np.abs(dense - dense.T).max() < 1e-12      # 对称
        return float(np.linalg.eigvalsh(dense)[0])

    assert smallest_eigenvalue(0.0) > 0.0                  # k = 0：就是 Poisson
    assert smallest_eigenvalue(3.0) > 0.0                  # 低频
    assert smallest_eigenvalue(5.0) < 0.0                  # 5^2 = 25 > 2 pi^2


def test_helmholtz_rejects_non_real_or_rt0():
    mesh = mf.Mesh.rectangle(nx=4, ny=4)
    V = mf.make_space(mesh, "P1")
    with pytest.raises(TypeError):
        mf.HelmholtzProblem(V, f=source2, k=1.0j)
    with pytest.raises(TypeError):
        mf.HelmholtzProblem(V, f=source2, k=1.0, kappa="big")
    with pytest.raises(ValueError):
        mf.solve_helmholtz(mesh, element="RT0", f=source2, k=1.0)


@pytest.mark.parametrize("elem", ["P1", "P2", "Q1", "Q2", "VEM"])
def test_solve_helmholtz_api(elem):
    et = "quad" if elem.startswith("Q") else "triangle"
    mesh = mf.Mesh.rectangle(nx=6, ny=6, element_type=et)
    k = 1.0
    u = mf.solve_helmholtz(mesh, element=elem, f=helmholtz_source(k), g=zero, k=k)
    assert u.shape[0] > 0 and np.isfinite(u).all()

    out = mf.solve_helmholtz(mesh, element=elem, f=helmholtz_source(k), g=zero,
                             k=k, return_all=True)
    assert out["u"].size == out["space"].n_dofs
    assert isinstance(out["problem"], mf.HelmholtzProblem)


def test_helmholtz_by_name_api():
    mesh = mf.Mesh.rectangle(nx=8, ny=8)
    k = 1.0
    u1 = mf.run("helmholtz", mesh, element="P2", f=helmholtz_source(k), g=zero, k=k)
    V = mf.make_space(mesh, "P2")
    u2 = mf.create_problem("helmholtz", V, f=helmholtz_source(k), g=zero, k=k).solve()
    assert np.abs(u1 - u2).max() < 1e-12


# ---------------------------------------------------------------------------
# RT0 混合元
# ---------------------------------------------------------------------------
def test_rt0_mixed_solve_2d():
    mesh = mf.Mesh.rectangle(nx=16, ny=16)
    V = mf.FESpace.rt0(mesh)
    assert V.n_dofs == V.n_flux + V.n_pressure
    problem = mf.MixedPoissonProblem(V, f=source2)
    Q, P = problem.solve()
    assert Q.shape[0] == mesh.n_cells == P.size
    err = problem.compute_errors((Q, P), exact2, lambda x: -grad2(x))
    assert err["L2"] < 0.1
    assert err["flux_L2"] < 0.3
    assert problem.check_conservation() < 1e-8


def test_rt0_mixed_solve_3d():
    mesh = mf.Mesh.box(nx=4, ny=4, nz=4, element_type="tet")
    V = mf.FESpace.rt0(mesh)
    problem = mf.MixedPoissonProblem(V, f=lambda x: 1.0)
    Q, P = problem.solve()
    assert P.size == mesh.n_cells
    assert P.max() > 0


def test_rt0_rejects_non_simplex_mesh():
    with pytest.raises(ValueError):
        mf.FESpace.rt0(mf.Mesh.rectangle(nx=4, ny=4, element_type="quad"))


# ---------------------------------------------------------------------------
# 高层 API 与扩展点
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("elem", ["P1", "P2", "Q1", "VEM", "RT0"])
def test_solve_poisson_api(elem):
    et = "quad" if elem.startswith("Q") else "triangle"
    mesh = mf.Mesh.rectangle(nx=6, ny=6, element_type=et)
    out = mf.solve_poisson(mesh, element=elem, f=source2, g=zero)
    if elem == "RT0":
        assert isinstance(out, tuple) and len(out) == 2
    else:
        assert out.shape[0] > 0


def test_solve_poisson_return_all():
    mesh = mf.Mesh.rectangle(nx=6, ny=6)
    out = mf.solve_poisson(mesh, element="P1", f=source2, g=zero, return_all=True)
    assert set(["u", "space", "problem", "mesh"]).issubset(out)
    assert out["space"].n_dofs == out["u"].size


def test_element_alias_resolution():
    assert mf.resolve_element("p2") == ("lagrange", 2)
    assert mf.resolve_element("VEM") == ("vem", 1)
    assert mf.resolve_element("RT0-P0") == ("rt0", 1)
    with pytest.raises(ValueError):
        mf.resolve_element("XYZ")


def test_registry_and_custom_pde():
    name = "test_custom_pde"
    if name not in available_problems():

        @register_problem(name)
        class _CustomPDE(BasePhysics):
            def __init__(self, space, f=None, g=None, k=1.0, **kw):
                super().__init__(space, f=f, g=g, **kw)
                self.k = k

            def assemble(self):
                A = self.space.stiffness_matrix() - self.k ** 2 * self.space.mass_matrix()
                b = self.space.load_vector(self.f)
                return self.apply_boundary_conditions(A, b)

    assert name in available_problems()
    mesh = mf.Mesh.rectangle(nx=8, ny=8)
    V = mf.make_space(mesh, "P1")
    problem = mf.create_problem(name, V, f=source2, g=zero, k=1.0)
    u = problem.solve(method="direct")
    assert np.isfinite(u).all()


def test_convergence_study_runner():
    def mesh_factory(level):
        return mf.Mesh.rectangle(nx=4 * 2 ** level, ny=4 * 2 ** level)

    def problem_factory(mesh):
        return mf.PoissonProblem(mf.FESpace.lagrange(mesh, 1), f=source2, g=zero)

    h, errors = mf.convergence_study(
        mesh_factory, problem_factory, exact2, grad2,
        refinement_levels=3, verbose=False,
    )
    assert len(h) == 3
    assert errors["L2"][-1] < errors["L2"][0]


def test_nodal_values_reproduces_dofs():
    mesh = mf.Mesh.rectangle(nx=4, ny=4)
    V = mf.make_space(mesh, "P1")
    lin = lambda x: 1.0 + x[0] - 2.0 * x[1]  # noqa: E731
    u = V.interpolate(lin)
    nodal = V.nodal_values(u)
    assert np.abs(nodal - V._evaluate_function(lin, mesh.nodes)).max() < 1e-12


K_ANISO = np.array([[3.0, 1.0], [1.0, 2.0]])


def aniso_source(x):
    """各向异性 K 下 p = sin(πx)sin(πy) 对应的源项（单点与批量都接受）。"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        s = np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        c = np.cos(np.pi * x[0]) * np.cos(np.pi * x[1])
    else:
        s = np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        c = np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
    tr, k12 = K_ANISO[0, 0] + K_ANISO[1, 1], K_ANISO[0, 1]
    return np.pi ** 2 * tr * s - 2.0 * np.pi ** 2 * k12 * c


def aniso_flux(x):
    return -K_ANISO @ grad2(x)


def _bc_column(mid, n):
    """左右定压、上下不透水。"""
    return "D" if (mid[0] < 1e-12 or mid[0] > 1.0 - 1e-12) else "N"


def _layered_exact(kL=1.0, kR=100.0):
    """层状（串联）精确解：q=(Q,0)，p 分段线性，Q = 2/(1/kL+1/kR)。"""
    Q = 2.0 / (1.0 / kL + 1.0 / kR)
    ph = 1.0 - Q * 0.5 / kL

    def p(x):
        xx = x[:, 0]
        return np.where(xx < 0.5, 1.0 - Q * xx / kL, ph - Q * (xx - 0.5) / kR)

    return Q, p


def test_darcy_registered():
    assert "darcy" in mf.available_problems()
    assert mf.get_problem_class("darcy") is mf.DarcyProblem


def test_darcy_recovers_mixed_poisson():
    """K = I 时 Darcy 与 MixedPoissonProblem 必须给同一个解。"""
    V = mf.FESpace.rt0(mf.Mesh.rectangle(nx=8, ny=8))
    Q0, P0 = mf.MixedPoissonProblem(V, f=source2, quadrature_order=6).solve()
    V2 = mf.FESpace.rt0(mf.Mesh.rectangle(nx=8, ny=8))
    darcy = mf.DarcyProblem(V2, f=source2, K=1.0)
    Q1, P1 = darcy.solve()
    assert np.abs(P0 - P1).max() < 1e-12
    assert np.abs(Q0 - Q1).max() < 1e-12
    assert darcy.check_conservation() < 1e-12


def test_weighted_flux_mass_unit_permeability_equals_plain():
    """K=I 时加权质量矩阵就是库里的闭式质量矩阵（同一份公式）。"""
    V = mf.FESpace.rt0(mf.Mesh.rectangle(nx=6, ny=6))
    A = V.weighted_flux_mass_matrix(1.0).toarray()
    M = V.flux_mass_matrix().toarray()
    assert np.abs(A - M).max() == 0.0


def test_weighted_mass_closed_form_matches_quadrature():
    """分片常数张量：闭式（cell）与逐点求积（quad）应一致。"""
    V = mf.FESpace.rt0(mf.Mesh.rectangle(nx=6, ny=6))
    A1 = V.weighted_flux_mass_matrix(K_ANISO, mode="cell").toarray()
    A2 = V.weighted_flux_mass_matrix(K_ANISO, mode="quad",
                                     quadrature_order=8).toarray()
    assert np.abs(A1 - A2).max() / np.abs(A1).max() < 1e-13


def test_darcy_permeability_validation():
    V = mf.FESpace.rt0(mf.Mesh.rectangle(nx=4, ny=4))
    with pytest.raises(ValueError):          # 非正定
        V.weighted_flux_mass_matrix([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError):          # 形状不对
        V.weighted_flux_mass_matrix([1.0, 2.0, 3.0])


def test_darcy_anisotropic_convergence():
    """常值各向异性：压力与通量都是一阶。"""
    hs, ep, eq = [], [], []
    for n in (4, 8, 16):
        V = mf.FESpace.rt0(mf.Mesh.rectangle(nx=n, ny=n))
        problem = mf.DarcyProblem(V, f=aniso_source, K=K_ANISO)
        Q, P = problem.solve()
        err = problem.compute_errors((Q, P), exact2, aniso_flux)
        hs.append(V.mesh.get_mesh_size())
        ep.append(err["L2"])
        eq.append(err["flux_L2"])
        assert problem.check_conservation() < 1e-10
    assert mf.estimate_convergence_rate(hs, ep, use_last=2) > 0.9
    assert mf.estimate_convergence_rate(hs, eq, use_last=2) > 0.9


def test_darcy_layered_harmonic_mean():
    """层状串联介质：通量精确可表示，且等于调和平均 2/(1/kL+1/kR)。"""
    kL, kR = 1.0, 100.0
    Qex, pex = _layered_exact(kL, kR)
    mesh = mf.Mesh.rectangle(nx=16, ny=16)
    V = mf.FESpace.rt0(mesh)
    K = lambda x: np.where(x[:, 0] < 0.5, kL, kR)   # noqa: E731
    problem = mf.DarcyProblem(V, f=lambda x: np.zeros(len(x)), g=pex,
                              K=K, bc=_bc_column)
    Q, P = problem.solve()
    err = problem.compute_errors((Q, P), pex, lambda x: np.array([Qex, 0.0]))
    assert err["flux_L2"] < 1e-9                     # 常通量场落在 RT0 内
    # 右边界（x=1）的外法向通量之和 = 串联等效通量
    nodes = np.asarray(mesh.nodes, dtype=float)
    total = 0.0
    for c in range(mesh.n_cells):
        for i in range(mesh.dim + 1):
            fct = int(V.topology.cell_facets[c, i])
            if V.topology.counts[fct] != 1:
                continue
            if abs(nodes[V.topology.facets[fct]].mean(axis=0)[0] - 1.0) < 1e-12:
                total += Q[c, i]
    assert total == pytest.approx(Qex, rel=1e-10)
    assert Qex == pytest.approx(200.0 / 101.0)


def test_darcy_prescribed_flux_boundary():
    """左边界给定 q·n=-1、右边界 p=0、上下不透水 ⇒ q=(1,0)。"""
    def bc(mid, n):
        if mid[0] < 1e-12:
            return ("flux", lambda pts: -np.ones(len(pts)))
        return "D" if mid[0] > 1.0 - 1e-12 else "N"

    V = mf.FESpace.rt0(mf.Mesh.rectangle(nx=8, ny=8))
    p_ex = lambda x: 1.0 - x[:, 0]                    # noqa: E731
    problem = mf.DarcyProblem(V, f=lambda x: np.zeros(len(x)), g=lambda x: np.zeros(len(x)),
                              K=1.0, bc=bc)
    Q, P = problem.solve()
    err = problem.compute_errors((Q, P), p_ex, lambda x: np.array([1.0, 0.0]))
    assert err["flux_L2"] < 1e-10
    mesh = V.mesh
    nodes = np.asarray(mesh.nodes, dtype=float)
    left = 0.0
    for c in range(mesh.n_cells):
        for i in range(mesh.dim + 1):
            fct = int(V.topology.cell_facets[c, i])
            if V.topology.counts[fct] != 1:
                continue
            if nodes[V.topology.facets[fct]].mean(axis=0)[0] < 1e-12:
                left += Q[c, i]
    assert left == pytest.approx(-1.0, rel=1e-10)


def test_darcy_pure_neumann_guards_and_solves():
    """纯 Neumann：压力只到相差常数；Schur 路径自动取零均值，直接法被拦下。"""
    f_zero_mean = lambda x: np.sin(2 * np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1])  # noqa: E731
    V = mf.FESpace.rt0(mf.Mesh.rectangle(nx=8, ny=8))
    problem = mf.DarcyProblem(V, f=f_zero_mean, bc=lambda mid, n: "N")
    with pytest.raises(ValueError):
        problem.solve(method="direct")
    Q, P = problem.solve()
    assert abs(P.mean()) < 1e-10
    assert np.isfinite(Q).all()


def test_solve_darcy_api():
    mesh = mf.Mesh.rectangle(nx=8, ny=8)
    Q, P = mf.solve_darcy(mesh, f=source2, K=1.0)
    assert Q.shape == (mesh.n_cells, mesh.dim + 1)
    assert P.size == mesh.n_cells
    out = mf.solve_darcy(mesh, f=source2, K=K_ANISO, return_all=True)
    assert set(["u", "flux", "pressure", "space", "problem", "mesh"]).issubset(out)
    with pytest.raises(ValueError):                  # Darcy 只支持 RT0
        mf.solve_darcy(mesh, element="P1", f=source2)


# ---------------------------------------------------------------------------
# 高层 API 与扩展点
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("elem", ["P1", "P2", "Q1", "VEM", "RT0"])
def test_solve_poisson_api(elem):
    et = "quad" if elem.startswith("Q") else "triangle"
    mesh = mf.Mesh.rectangle(nx=6, ny=6, element_type=et)
    out = mf.solve_poisson(mesh, element=elem, f=source2, g=zero)
    if elem == "RT0":
        assert isinstance(out, tuple) and len(out) == 2
    else:
        assert out.shape[0] > 0


def test_solve_poisson_return_all():
    mesh = mf.Mesh.rectangle(nx=6, ny=6)
    out = mf.solve_poisson(mesh, element="P1", f=source2, g=zero, return_all=True)
    assert set(["u", "space", "problem", "mesh"]).issubset(out)
    assert out["space"].n_dofs == out["u"].size


def test_element_alias_resolution():
    assert mf.resolve_element("p2") == ("lagrange", 2)
    assert mf.resolve_element("VEM") == ("vem", 1)
    assert mf.resolve_element("RT0-P0") == ("rt0", 1)
    with pytest.raises(ValueError):
        mf.resolve_element("XYZ")


def test_registry_and_custom_pde():
    name = "test_helmholtz"
    if name not in available_problems():

        @register_problem(name)
        class _Helmholtz(BasePhysics):
            def __init__(self, space, f=None, g=None, k=1.0, **kw):
                super().__init__(space, f=f, g=g, **kw)
                self.k = k

            def assemble(self):
                A = self.space.stiffness_matrix() - self.k ** 2 * self.space.mass_matrix()
                b = self.space.load_vector(self.f)
                return self.apply_boundary_conditions(A, b)

    assert name in available_problems()
    mesh = mf.Mesh.rectangle(nx=8, ny=8)
    V = mf.make_space(mesh, "P1")
    problem = mf.create_problem(name, V, f=source2, g=zero, k=1.0)
    u = problem.solve(method="direct")
    assert np.isfinite(u).all()


def test_convergence_study_runner():
    def mesh_factory(level):
        return mf.Mesh.rectangle(nx=4 * 2 ** level, ny=4 * 2 ** level)

    def problem_factory(mesh):
        return mf.PoissonProblem(mf.FESpace.lagrange(mesh, 1), f=source2, g=zero)

    h, errors = mf.convergence_study(
        mesh_factory, problem_factory, exact2, grad2,
        refinement_levels=3, verbose=False,
    )
    assert len(h) == 3
    assert errors["L2"][-1] < errors["L2"][0]


def test_nodal_values_reproduces_dofs():
    mesh = mf.Mesh.rectangle(nx=4, ny=4)
    V = mf.make_space(mesh, "P1")
    lin = lambda x: 1.0 + x[0] - 2.0 * x[1]  # noqa: E731
    u = V.interpolate(lin)
    nodal = V.nodal_values(u)
    assert np.abs(nodal - V._evaluate_function(lin, mesh.nodes)).max() < 1e-12

# ---------------------------------------------------------------------------
# 定常电流场（DC conduction，RT0×P0）
# ---------------------------------------------------------------------------
def _bc_electrodes(mid, n):
    """左右电极定压、上下绝缘。"""
    return "D" if (mid[0] < 1e-12 or mid[0] > 1.0 - 1e-12) else "N"


def _right_electrode(mid):
    return mid[0] > 1.0 - 1e-12


def test_conduction_registered():
    assert "conduction" in mf.available_problems()
    assert mf.get_problem_class("conduction") is mf.ConductionProblem


def test_conduction_bar():
    """均质条形电阻器：I=σAΔφ/L，R=L/(σA)，P=Δφ·I，逐单元电荷守恒。"""
    mesh = mf.Mesh.rectangle(nx=8, ny=8)
    V = mf.FESpace.rt0(mesh)
    problem = mf.ConductionProblem(V, g=lambda x: 1.0 - x[:, 0], sigma=2.0,
                                   bc=_bc_electrodes)
    J, phi = problem.solve()
    assert problem.total_current(_right_electrode) == pytest.approx(2.0, rel=1e-10)
    assert problem.resistance(1.0, _right_electrode) == pytest.approx(0.5, rel=1e-10)
    assert problem.joule_heating() == pytest.approx(2.0, rel=1e-10)
    assert problem.check_charge_conservation() < 1e-10


def test_conduction_layered_series():
    """串联分层导体：R = Σ L_i/σ_i（与 Darcy 的调和平均同一公式）。"""
    mesh = mf.Mesh.rectangle(nx=8, ny=8)
    V = mf.FESpace.rt0(mesh)
    sigma = lambda x: np.where(x[:, 0] < 0.5, 1.0, 100.0)   # noqa: E731
    problem = mf.ConductionProblem(V, g=lambda x: 1.0 - x[:, 0], sigma=sigma,
                                   bc=_bc_electrodes)
    problem.solve()
    assert problem.resistance(1.0, _right_electrode) == pytest.approx(0.505, rel=1e-10)


def test_conduction_contact_resistance():
    """界面接触电阻：R = R_bulk + R_c/|Γ_c|，电势跳变 = R_c·I。"""
    mesh = mf.Mesh.rectangle(nx=8, ny=8)
    V = mf.FESpace.rt0(mesh)
    problem = mf.ConductionProblem(
        V, g=lambda x: 1.0 - x[:, 0], sigma=1.0, bc=_bc_electrodes,
        contact=lambda mid: 0.5 if abs(mid[0] - 0.5) < 1e-12 else None)
    J, phi = problem.solve()
    I = problem.total_current(_right_electrode)
    assert problem.resistance(1.0, _right_electrode) == pytest.approx(1.5, rel=1e-10)
    ctr = np.array([mesh.get_cell_centroid(c) for c in range(mesh.n_cells)])
    h = 1.0 / 8.0
    left = int(np.abs(ctr[:, 0] - (0.5 - h / 2)).argmin())
    right = int(np.abs(ctr[:, 0] - (0.5 + h / 2)).argmin())
    jump = (phi[left] - phi[right]) - I * h          # 扣掉形心间体压降
    assert jump == pytest.approx(0.5 * I, rel=1e-9)


def test_conduction_matches_darcy():
    """σ ↔ K 时与 DarcyProblem 给同一个解。"""
    K = np.array([[3.0, 1.0], [1.0, 2.0]])
    V1 = mf.FESpace.rt0(mf.Mesh.rectangle(nx=8, ny=8))
    p1 = mf.ConductionProblem(V1, g=lambda x: 1.0 - x[:, 0], sigma=K,
                              bc=_bc_electrodes)
    J1, phi1 = p1.solve()
    V2 = mf.FESpace.rt0(mf.Mesh.rectangle(nx=8, ny=8))
    p2 = mf.DarcyProblem(V2, g=lambda x: 1.0 - x[:, 0], K=K, bc=_bc_electrodes)
    J2, phi2 = p2.solve()
    assert np.abs(J1 - J2).max() < 1e-14
    assert np.abs(phi1 - phi2).max() < 1e-14


def test_solve_conduction_api():
    mesh = mf.Mesh.rectangle(nx=8, ny=8)
    J, phi = mf.solve_conduction(mesh, g=lambda x: 1.0 - x[:, 0], sigma=1.0,
                                 bc=_bc_electrodes)
    assert J.shape == (mesh.n_cells, mesh.dim + 1)
    assert phi.size == mesh.n_cells
    out = mf.solve_conduction(mesh, g=lambda x: 1.0 - x[:, 0], sigma=1.0,
                              bc=_bc_electrodes, return_all=True)
    assert {"current", "potential", "problem", "space"} <= set(out)
    with pytest.raises(ValueError):
        mf.solve_conduction(mesh, element="P1", g=lambda x: 0.0)
