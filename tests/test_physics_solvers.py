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
