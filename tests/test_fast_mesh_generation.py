"""批量网格的非等距区域、最小 Voronoi 及三角形邻接验证。"""
import numpy as np
import pytest
from MosicaFE import Mesh, FESpace
from MosicaFE.core.mesh import rectangle_arrays, _triangle_neighbors


@pytest.mark.parametrize('kind,pattern,count', [
    ('hex', 'kuhn', 24), ('tet', 'kuhn', 144), ('tet', 'center', 288),
])
def test_batch_box_nonuniform_region(kind, pattern, count):
    mesh = Mesh.box(xmin=-2., xmax=1., ymin=4., ymax=6., zmin=-1., zmax=3.,
                    nx=3, ny=2, nz=4, element_type=kind, pattern=pattern)
    assert mesh.n_cells == count
    measures = mesh.cell_measures()
    assert np.all(measures > 0)
    assert abs(measures.sum()-24.) < 1e-11
    assert mesh.refine_uniform().n_cells == 8*count


@pytest.mark.parametrize('dim,count', [(2, 1), (2, 2), (3, 1), (3, 2)])
def test_minimal_mirrored_voronoi(dim, count):
    mesh = (Mesh.voronoi_polygon(n_points=count, seed=7, bbox=(-1., 2., 4., 6.))
            if dim == 2 else Mesh.voronoi_polyhedron(
                n_points=count, seed=7, bbox=(-1., 2., 4., 6., 0., 2.)))
    assert mesh.n_cells == count
    assert abs(mesh.cell_measures().sum()-(6. if dim == 2 else 12.)) < 1e-10
    FESpace.vem(mesh, degree=2)  # 星形条件、面共面及闭合性检查。


def test_batch_triangle_neighbors():
    _, quads = rectangle_arrays(4, 3, (0., 1., 0., 1.))
    triangles = quads[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
    neighbors = _triangle_neighbors(triangles)
    assert np.count_nonzero(neighbors < 0) == 2*(4+3)
    for cell, row in enumerate(neighbors):
        for other in row[row >= 0]:
            assert cell in neighbors[other]
            assert len(set(triangles[cell]) & set(triangles[other])) == 2


def test_batch_orientation_and_input_ownership():
    nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    cells = np.array([[0, 3, 2, 1], [0, 1, 2, 3]])
    original = cells.copy()
    mesh = Mesh.from_arrays(nodes, cells, element_type='quad')
    assert all(np.array_equal(row, [0, 1, 2, 3]) or
               np.array_equal(row, [1, 2, 3, 0]) for row in mesh.elements)
    assert np.array_equal(cells, original)


def test_interval_batch_generation():
    mesh = Mesh.interval(a=-2., b=3., nx=5)
    assert mesh.n_nodes == 6 and mesh.n_cells == 5
    assert np.array_equal(mesh.nodes[:, 0], np.arange(-2., 4.))
    assert np.array_equal(mesh.uniform_vertices, np.column_stack((np.arange(5), np.arange(1, 6))))
