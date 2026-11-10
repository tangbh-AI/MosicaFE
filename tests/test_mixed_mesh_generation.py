"""自动混合网格的覆盖、协调性、形状和 VEM 求解检查。"""

import numpy as np
import pytest

from MosicaFE import FESpace, Mesh, PoissonProblem


def _coverage(mesh, bbox):
    bounds = np.asarray(bbox).reshape(mesh.dim, 2)
    expected = np.prod(bounds[:, 1]-bounds[:, 0])
    measures = mesh.cell_measures()
    assert np.all(measures > 0)
    assert abs(measures.sum()-expected) < 1e-10*expected
    # 无孤立节点；每个内部边/面恰好共享两次，没有区域内部的伪边界。
    assert set(np.concatenate(mesh.elements)) == set(range(mesh.n_nodes))
    for face in set(mesh.facets):
        assert len(mesh.facet_owner(face)) in (1, 2)
    for face in mesh.boundary_facets:
        points = mesh.nodes[list(face)]
        assert any(np.allclose(points[:, a], b, rtol=0, atol=1e-11)
                   for a in range(mesh.dim) for b in bounds[a])
    # 星形条件和闭合面检查由 VEM 本身执行。
    return FESpace.vem(mesh, degree=2)


@pytest.mark.parametrize('seed,concave', [(seed, concave) for seed in (0, 7, 42)
                                        for concave in (False, True)])
def test_mixed_polygon_coverage(seed, concave):
    bbox = (-2., 3., 4., 6.)
    mesh = Mesh.mixed_polygon(nx=6, ny=5, seed=seed, bbox=bbox, concave=concave)
    _coverage(mesh, bbox)
    assert min(mesh.vertices_per_cell) >= 3
    assert max(mesh.vertices_per_cell) <= 8
    assert len(set(mesh.vertices_per_cell)) >= 3
    if not concave:
        for ring in mesh.elements:
            edges = np.roll(mesh.nodes[ring], -1, axis=0)-mesh.nodes[ring]
            nxt = np.roll(edges, -1, axis=0)
            assert np.all(edges[:, 0]*nxt[:, 1]-edges[:, 1]*nxt[:, 0] >= -1e-12)


def test_mixed_polygon_shapes_and_reproducibility():
    a, b = Mesh.mixed_polygon(), Mesh.mixed_polygon()
    assert np.array_equal(a.nodes, b.nodes)
    assert all(np.array_equal(x, y) for x, y in zip(a.elements, b.elements))
    assert a.n_cells == b.n_cells
    assert set(a.vertices_per_cell) == set(range(3, 9))
    concave = False
    for ring in a.elements:
        edges = np.roll(a.nodes[ring], -1, axis=0)-a.nodes[ring]
        nxt = np.roll(edges, -1, axis=0)
        concave |= bool(np.any(edges[:, 0]*nxt[:, 1]-edges[:, 1]*nxt[:, 0] < -1e-12))
    assert concave


@pytest.mark.parametrize('split', [0., .35, 1.])
def test_mixed_polyhedron_coverage(split):
    bbox = (-1., 2., 3., 5., -2., 2.)
    mesh = Mesh.mixed_polyhedron(nx=4, ny=3, nz=2, split=split, bbox=bbox)
    _coverage(mesh, bbox)
    for c in range(mesh.n_cells):
        # 闭合曲面中，每条边必须恰好出现两次。
        edges = {}
        for face in mesh.get_cell_faces(c):
            for i, a in enumerate(face):
                key = tuple(sorted((a, face[(i+1) % len(face)])))
                edges[key] = edges.get(key, 0)+1
        assert set(edges.values()) == {2}


@pytest.mark.parametrize('dim,k', [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_mixed_mesh_polynomial_poisson(dim, k):
    mesh = (Mesh.mixed_polygon(nx=3, ny=3, seed=7) if dim == 2 else
            Mesh.mixed_polyhedron(nx=2, ny=2, nz=1, split=.5, seed=7))
    space = FESpace.vem(mesh, degree=k)
    exact = lambda x: 1.+np.sum(x**k)
    source = lambda x: -k*(k-1)*np.sum(x**(k-2))
    problem = PoissonProblem(space, f=source, g=exact)
    u = problem.solve(method='direct')
    assert np.max(abs(u-space.interpolate(exact))) < 2e-8
    D, H, G, energy, l2 = space.cells[0].data
    assert np.max(abs(energy@D-np.eye(len(H)))) < 2e-8
    assert np.max(abs(l2@D-np.eye(len(H)))) < 2e-8


def test_mixed_polyhedron_shapes_and_reproducibility():
    a, b = Mesh.mixed_polyhedron(), Mesh.mixed_polyhedron()
    assert np.array_equal(a.nodes, b.nodes)
    assert len(a.elements) == len(b.elements)
    assert all(np.array_equal(x, y) for x, y in zip(a.elements, b.elements))
    assert all(a.get_cell_faces(c) == b.get_cell_faces(c) for c in range(a.n_cells))
    assert {4, 5, 6, 8, 10, 12, 14} <= set(a.vertices_per_cell)
    assert max(len(face) for c in range(a.n_cells) for face in a.get_cell_faces(c)) >= 5


def test_mixed_mesh_minimum_size():
    for mesh in (Mesh.mixed_polygon(nx=1, ny=1, max_sides=3, jitter=0),
                 Mesh.mixed_polyhedron(nx=1, ny=1, nz=1, max_sides=3, split=1, jitter=0)):
        _coverage(mesh, (0., 1.)*mesh.dim)


@pytest.mark.parametrize('kwargs', [
    dict(nx=0), dict(ny=-1), dict(nx=True), dict(nx=1.5),
    dict(max_sides=2), dict(max_sides=True), dict(jitter=-.1),
    dict(jitter=.25), dict(jitter=float('nan')),
    dict(bbox=(0., 1., 0., 0.)), dict(bbox=(0., float('inf'), 0., 1.)),
])
def test_mixed_polygon_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        Mesh.mixed_polygon(**kwargs)


@pytest.mark.parametrize('kwargs', [
    dict(nz=0), dict(nz=True), dict(split=-.1), dict(split=1.1),
    dict(split=float('nan')), dict(bbox=(0., 1., 0., 1.)),
])
def test_mixed_polyhedron_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        Mesh.mixed_polyhedron(**kwargs)
