"""VEM 网格普适性：混合拓扑、真正多边形面、星形凹单元及三维 Voronoi。"""

# 模块职责：核对变长多面体拓扑、共享自由度、星形几何和真实截面的覆盖及求值。


import numpy as np
import pytest

from MosicaFE import Mesh, FESpace, PoissonProblem
from MosicaFE.core.quadrature import cell_quadrature


def check_quadratic(mesh):
    # 求解具有非零边界的二次多项式 Poisson 问题，同时核对自由度与两种投影的一致性。
    space = FESpace.vem(mesh, degree=2)
    exact = lambda x: 1.+np.sum(x*x)
    problem = PoissonProblem(space, f=lambda x: -2.*mesh.dim, g=exact)
    u = problem.solve(method='direct')
    assert np.max(np.abs(u-space.interpolate(exact))) < 1e-8
    for cell in space.cells:
        D, H, G, energy, l2 = cell.data
        assert np.max(np.abs(energy@D-np.eye(len(H)))) < 1e-8
        assert np.max(np.abs(l2@D-np.eye(len(H)))) < 1e-8
    return space


@pytest.mark.parametrize('seed', [3, 42])
def test_voronoi_3d_volume_topology_and_poisson(seed):
    # 验证目标：三维 Voronoi 单元体积和、外边界识别与非零边界多项式求解同时正确。
    mesh = Mesh.voronoi_polyhedron(n_points=12, seed=seed)
    assert mesh.n_cells == 12
    assert len(set(mesh.vertices_per_cell)) > 1
    assert len({len(mesh.get_cell_faces(c)) for c in range(12)}) > 1
    assert max(len(f) for c in range(12) for f in mesh.get_cell_faces(c)) > 4
    assert abs(sum(mesh.cell_measures())-1.) < 1e-10
    for face in mesh.boundary_facets:
        xyz = mesh.nodes[list(face)]
        # 区域内部不能出现伪边界；外边界面必须位于立方体六个平面之一。
        assert any(np.max(abs(xyz[:, a]-b)) < 1e-9 for a in range(3) for b in (0., 1.))
    for c in range(12):
        assert abs(cell_quadrature(mesh, c, 6)[1].sum()-mesh.get_cell_measure(c)) < 1e-10
    check_quadratic(mesh)


def test_mixed_hex_and_triangular_prisms():
    # 左半立方体是八顶点六面体；右半由两个六顶点三棱柱组成。
    bases = [[(0., 0.), (.5, 0.), (.5, 1.), (0., 1.)],
             [(.5, 0.), (1., 0.), (.5, 1.)],
             [(1., 0.), (1., 1.), (.5, 1.)]]
    nodes, lookup, elements, faces = [], {}, [], []
    for base in bases:
        vertices = []
        for z in (0., 1.):
            for x, y in base:
                p = (x, y, z)
                if p not in lookup:
                    lookup[p] = len(nodes)
                    nodes.append(p)
                vertices.append(lookup[p])
        n = len(base)
        elements.append(vertices)
        faces.append([list(range(n)), list(range(n, 2*n))] +
                     [[i, (i+1)%n, (i+1)%n+n, i+n] for i in range(n)])
    mesh = Mesh.from_arrays(nodes, elements, element_type='polyhedron', faces=faces)
    assert mesh.vertices_per_cell == [8, 6, 6]
    assert [len(mesh.get_cell_faces(c)) for c in range(3)] == [6, 5, 5]
    assert len(mesh.interior_facets) == 2
    assert abs(sum(mesh.cell_measures())-1.) < 1e-12
    space = check_quadratic(mesh)
    assert len(set(space.dof_map[0]) & set(space.dof_map[1])) == 9


def test_eight_vertices_do_not_imply_hexahedron():
    # 七棱锥也有八个顶点，但有八个面，不能根据“八顶点”套六面体模板。
    theta = np.arange(7)*2*np.pi/7
    nodes = np.vstack([np.column_stack([np.cos(theta), np.sin(theta), np.zeros(7)]), [0., 0., 1.]])
    faces = [list(range(7))] + [[i, (i+1)%7, 7] for i in range(7)]
    mesh = Mesh.from_arrays(nodes, [list(range(8))], element_type='polyhedron', faces=faces)
    assert len(mesh.get_cell_faces(0)) == 8
    assert len(mesh.boundary_facets) == 8
    expected = 7*np.sin(2*np.pi/7)/6
    assert abs(mesh.get_cell_measure(0)-expected) < 1e-12
    assert abs(cell_quadrature(mesh, 0, 4)[1].sum()-expected) < 1e-12
    check_quadratic(mesh)


def test_star_shaped_concave_polygon_and_polyhedron():
    # 验证目标：满足星形核条件的凹多边形和凹多面体能够通过投影及 Poisson 一致性检查。
    base = np.array([[0., 0.], [2., 0.], [2., 2.], [1., 1.4], [0., 2.]])
    mesh2 = Mesh.from_arrays(base, [list(range(5))], element_type='polygon')
    check_quadratic(mesh2)
    nodes = np.vstack([np.column_stack([base, np.zeros(5)]),
                       np.column_stack([base, np.ones(5)])])
    faces = [list(range(5)), list(range(5, 10))] + [[i, (i+1)%5, (i+1)%5+5, i+5] for i in range(5)]
    faces[2] = faces[2][::-1]  # 面的输入方向不必预先统一，由邻接关系确定外法向。
    mesh3 = Mesh.from_arrays(nodes, [list(range(10))], element_type='polyhedron', faces=faces)
    assert abs(mesh3.get_cell_measure(0)-3.4) < 1e-12
    check_quadratic(mesh3)


def test_polyhedron_open_boundary_is_rejected():
    # 验证目标：缺少面而未闭合的三维单元必须报错，不能继续组装一个不完整区域。
    mesh = Mesh.box(nx=1, ny=1, nz=1, element_type='hex')
    invalid = Mesh.from_arrays(mesh.nodes, mesh.elements, element_type='polyhedron',
                               faces=mesh.get_cell_faces(0)[:-1])
    with pytest.raises(ValueError):
        FESpace.vem(invalid)


@pytest.mark.parametrize('kind', ['hex', 'voronoi'])
def test_true_slice_area_and_projected_values(kind):
    # 验证目标：真实截面无遗漏和重复覆盖，且显示值来自正确高度处的多项式投影。
    try:
        import matplotlib.tri
    except ImportError:
        return  # 绘图依赖可选；没有安装时仍可运行数值求解测试。
    from MosicaFE.postprocess.visualization import _vem_slice
    mesh = (Mesh.box(nx=2, ny=2, nz=2, element_type='hex') if kind == 'hex'
            else Mesh.voronoi_polyhedron(8, seed=3))
    space = FESpace.vem(mesh, degree=2)
    u = space.interpolate(lambda x: 1.+np.sum(x*x))
    tri, values = _vem_slice(space, u, .5)
    xy = np.column_stack([tri.x, tri.y])[tri.triangles]
    a, b = xy[:, 1]-xy[:, 0], xy[:, 2]-xy[:, 0]
    area = np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]).sum()/2
    assert abs(area-1.) < 1e-10  # 不漏画，也不能把恰好落在单元面上的截面画两次。
    assert np.max(abs(values-(1.25+tri.x**2+tri.y**2))) < 1e-9
