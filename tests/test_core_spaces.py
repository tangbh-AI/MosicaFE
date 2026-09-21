# -*- coding: utf-8 -*-
"""网格、求积规则、离散空间基本性质的测试。"""

from __future__ import annotations

import numpy as np
import pytest

import MosicaFE as mf
from MosicaFE.core.quadrature import QuadratureRule


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
def test_mesh_topology_rectangle():
    mesh = mf.Mesh.rectangle(nx=3, ny=2)
    assert mesh.n_cells == 12
    assert mesh.n_nodes == 12
    assert len(mesh.boundary_nodes) == 10
    assert mesh.get_mesh_size() > 0


@pytest.mark.parametrize(
    "mesh,n_cells,n_nodes",
    [
        (mf.Mesh.rectangle(nx=3, ny=3), 18, 16),
        (mf.Mesh.rectangle(nx=3, ny=3, element_type="quad"), 9, 16),
        (mf.Mesh.box(nx=2, ny=2, nz=2), 48, 27),
        (mf.Mesh.box(nx=2, ny=2, nz=2, element_type="hex"), 8, 27),
        (mf.Mesh.pentagon(nx=3, ny=3), 9, None),
        (mf.Mesh.voronoi_polygon(25, seed=1), 25, None),
    ],
)
def test_mesh_families(mesh, n_cells, n_nodes):
    assert mesh.n_cells == n_cells
    if n_nodes is not None:
        assert mesh.n_nodes == n_nodes
    assert len(mesh.boundary_facets) > 0
    assert len(mesh.boundary_nodes) > 0
    assert np.all(mesh.cell_measures() > 0)


def test_mesh_refine_uniform():
    coarse = mf.Mesh.rectangle(nx=2, ny=2)
    fine = coarse.refine_uniform()
    assert fine.n_cells == 4 * coarse.n_cells
    assert fine.get_mesh_size() < coarse.get_mesh_size()
    with pytest.raises(ValueError):
        mf.Mesh.voronoi_polygon(20, seed=0).refine_uniform()


def test_mesh_from_arrays():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mesh = mf.Mesh.from_arrays(nodes, [[0, 1, 2]])
    assert mesh.element_type == "triangle"
    assert mesh.n_cells == 1
    assert len(mesh.boundary_nodes) == 3


# ---------------------------------------------------------------------------
def test_quadrature_weights():
    assert QuadratureRule.get_quadrature("interval", 3).weights.sum() == pytest.approx(1.0)
    assert QuadratureRule.get_quadrature("triangle", 2).weights.sum() == pytest.approx(0.5)
    assert QuadratureRule.get_quadrature("quad", 3).weights.sum() == pytest.approx(1.0)
    assert QuadratureRule.get_quadrature("tet", 3).weights.sum() == pytest.approx(1.0 / 6.0)
    assert QuadratureRule.get_quadrature("hex", 3).weights.sum() == pytest.approx(1.0)


def test_quadrature_polynomial_exactness():
    quad = QuadratureRule.get_quadrature("triangle", 3)
    f = quad.points[:, 0] ** 2 + quad.points[:, 1] ** 2
    assert float(np.sum(quad.weights * f)) == pytest.approx(1.0 / 6.0, abs=1e-12)


def test_cell_quadrature_integrates_area():
    for mesh in (
        mf.Mesh.rectangle(nx=3, ny=3),
        mf.Mesh.rectangle(nx=3, ny=3, element_type="quad"),
        mf.Mesh.pentagon(nx=3, ny=3),
        mf.Mesh.voronoi_polygon(20, seed=2),
    ):
        total = 0.0
        for c in range(mesh.n_cells):
            _, w = mf.cell_quadrature(mesh, c, 3)
            total += float(w.sum())
        assert total == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mesh,elements",
    [
        (mf.Mesh.rectangle(nx=3, ny=3), ("P1", "P2")),
        (mf.Mesh.rectangle(nx=3, ny=3, element_type="quad"), ("Q1", "Q2")),
        (mf.Mesh.box(nx=2, ny=2, nz=2), ("P1", "P2")),
        (mf.Mesh.box(nx=2, ny=2, nz=2, element_type="hex"), ("Q1", "Q2")),
    ],
)
def test_lagrange_nodal_property(mesh, elements):
    """基函数在节点上满足 Kronecker delta，并且构成单位分解。"""
    for elem in elements:
        V = mf.make_space(mesh, elem)
        for x in V.reference_nodes(0):
            N, _ = V._basis_single(np.asarray(x, dtype=float))
            assert np.max(N) == pytest.approx(1.0, abs=1e-10)
            assert np.sum(N) == pytest.approx(1.0, abs=1e-10)


def test_lagrange_stiffness_symmetry():
    mesh = mf.Mesh.rectangle(nx=3, ny=3)
    V = mf.make_space(mesh, "P2")
    A = V.stiffness_matrix().toarray()
    assert np.abs(A - A.T).max() < 1e-13
    assert np.abs(A @ np.ones(V.n_dofs)).max() < 1e-12


# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mesh,elem",
    [
        (mf.Mesh.rectangle(nx=4, ny=4), "P1"),
        (mf.Mesh.rectangle(nx=4, ny=4), "P2"),
        (mf.Mesh.rectangle(nx=4, ny=4, element_type="quad"), "Q1"),
        (mf.Mesh.rectangle(nx=4, ny=4), "VEM"),
        (mf.Mesh.rectangle(nx=4, ny=4, element_type="quad"), "VEM"),
        (mf.Mesh.pentagon(nx=4, ny=4), "VEM"),
        (mf.Mesh.voronoi_polygon(30, seed=3), "VEM"),
    ],
)
def test_patch_test(mesh, elem):
    """线性精确解 + 线性 Dirichlet ⇒ 离散解必须精确。"""
    lin = lambda x: 1.0 + 2.0 * x[0] - 0.5 * x[1]  # noqa: E731
    V = mf.make_space(mesh, elem)
    problem = mf.PoissonProblem(V, f=zero, g=lin)
    u = problem.solve()
    assert np.abs(u - V.interpolate(lin)).max() < 1e-9


# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mesh",
    [
        mf.Mesh.rectangle(nx=3, ny=3),
        mf.Mesh.rectangle(nx=3, ny=3, element_type="quad"),
        mf.Mesh.pentagon(nx=3, ny=3),
        mf.Mesh.voronoi_polygon(25, seed=5),
        mf.Mesh.box(nx=2, ny=2, nz=2),
    ],
)
def test_vem_stiffness_properties(mesh):
    V = mf.FESpace.vem(mesh)
    A = V.stiffness_matrix().toarray()
    assert np.abs(A - A.T).max() < 1e-12
    assert V.n_dofs == mesh.n_nodes
    w = np.linalg.eigvalsh(0.5 * (A + A.T))
    assert w.min() > -1e-10


def test_vem_mass_row_sum_matches_load():
    mesh = mf.Mesh.voronoi_polygon(30, seed=9)
    V = mf.FESpace.vem(mesh)
    b = V.load_vector(lambda x: 1.0)
    M = V.mass_matrix().toarray()
    assert np.abs(b - M.sum(axis=1)).max() < 1e-12
