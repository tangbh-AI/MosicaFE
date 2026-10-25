"""高阶 VEM：验证真实多项式再生、非零边界矩、共享自由度及正弦解收敛。"""

# 模块职责：从求积精度、多项式投影、非零边界和收敛性四个方面验证高阶 VEM。


import math

import numpy as np
import pytest

from MosicaFE import FESpace, Mesh, PoissonProblem, make_space, resolve_element
from MosicaFE.core.quadrature import QuadratureRule


# 工作块：积分必须随 k 提高，不能继续沿用固定的低阶三角/四面体规则。
@pytest.mark.parametrize('dim,degree', [(2, 10), (3, 9)])
def test_high_order_simplex_quadrature(dim, degree):
    # 验证目标：高阶三角形和四面体规则能够积分超过原固定低阶规则的多项式。
    rule = QuadratureRule.get_quadrature('triangle' if dim == 2 else 'tet', degree)
    alpha = [degree-2, 1] if dim == 2 else [degree-3, 1, 1]
    value = np.sum(rule.weights * np.prod(rule.points**alpha, axis=1))
    expected = math.prod(math.factorial(a) for a in alpha) / math.factorial(sum(alpha)+dim)
    assert abs(value-expected) < 1e-13


@pytest.mark.parametrize('kind,k', [
    ('quad', 1), ('quad', 2), ('quad', 3), ('quad', 4), ('quad', 5),
    ('triangle', 3), ('pentagon', 3), ('voronoi', 3),
    ('hex', 1), ('hex', 2), ('hex', 3), ('hex', 4), ('hex', 5), ('tet', 3),
])
def test_vem_polynomial_projection_and_consistency(kind, k):
    # 验证目标：H1/L2 投影重现 k 次多项式，刚度保持一致性且常数函数处于零能量方向。
    if kind in ('tet', 'hex'):
        mesh = Mesh.box(nx=1, ny=1, nz=1, element_type=kind)
    elif kind == 'pentagon':
        mesh = Mesh.pentagon(nx=2, ny=2)
    elif kind == 'voronoi':
        mesh = Mesh.voronoi_polygon(8, seed=5)
    else:
        mesh = Mesh.rectangle(nx=1, ny=1, element_type=kind)
    space = FESpace.vem(mesh, degree=k)
    for cell in space.cells:
        D, H, G, energy, l2 = cell.data
        identity = np.eye(len(H))
        assert np.max(np.abs(energy@D-identity)) < 2e-8
        assert np.max(np.abs(l2@D-identity)) < 2e-8
        A = cell.stiffness('trace')
        assert np.max(np.abs(A-A.T)) < 1e-8
        assert np.max(np.abs(D.T@A@D-G)) < 2e-8
        # 常数自由度中的高次矩不一定是 1，必须用 D[:,0] 表示常数函数。
        assert np.linalg.norm(A@D[:, 0]) < 1e-6
        assert np.linalg.eigvalsh(A).min() > -1e-7


# 工作块：非零边界让三维面的矩自由度真正参与测试。
# 以 k 次多项式作精确解，源项为其负 Laplace；离散解应精确再现所有自由度。
@pytest.mark.parametrize('dim,k', [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_vem_polynomial_poisson_with_nonzero_boundary(dim, k):
    # 验证目标：非零边界上的点值和三维面矩都按定义施加，从而精确重现多项式解。
    mesh = (Mesh.rectangle(nx=2, ny=2, element_type='quad') if dim == 2
            else Mesh.box(nx=2, ny=2, nz=2, element_type='hex'))
    space = FESpace.vem(mesh, degree=k)
    exact = lambda x: 1.0 + np.sum(x**k)
    source = lambda x: -k*(k-1)*np.sum(x**(k-2))
    problem = PoissonProblem(space, f=source, g=exact)
    u = problem.solve(method='direct')
    assert np.max(np.abs(u-space.interpolate(exact))) < 2e-8


@pytest.mark.parametrize('dim,k', [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_vem_sine_poisson_convergence(dim, k):
    # 验证目标：二维、三维高阶 VEM 在正弦制造解下达到预期的误差下降速度。
    exact = lambda x: np.prod(np.sin(np.pi*x))
    source = lambda x: dim*np.pi**2*exact(x)
    grad = lambda x: np.array([
        np.pi*np.cos(np.pi*x[j])*np.prod(np.sin(np.pi*np.delete(x, j)))
        for j in range(dim)
    ])
    errors = []
    for n in (2, 4):
        mesh = (Mesh.rectangle(nx=n, ny=n, element_type='quad') if dim == 2
                else Mesh.box(nx=n, ny=n, nz=n, element_type='hex'))
        space = FESpace.vem(mesh, degree=k)
        problem = PoissonProblem(space, f=source)
        u = problem.solve(method='direct')
        errors.append(problem.compute_errors(u, exact, grad))
    assert np.log2(errors[0]['L2']/errors[1]['L2']) > k + 0.5
    assert np.log2(errors[0]['H1']/errors[1]['H1']) > k - 0.5


def test_vem_shared_dofs_and_element_alias():
    # 验证目标：相邻单元共享完整高阶边/面自由度，VEMk 别名解析得到正确阶数。
    mesh = Mesh.rectangle(nx=2, ny=1, element_type='quad')
    space = make_space(mesh, 'VEM3')
    # 两个单元共享一条边：两个端点、两个边内点，总共四个自由度。
    shared = set(space.dof_map[0]) & set(space.dof_map[1])
    assert len(shared) == 4
    assert resolve_element('VEM12') == ('vem', 12)
    assert resolve_element('VEM', degree=4) == ('vem', 4)
    space3 = FESpace.vem(Mesh.box(nx=2, ny=1, nz=1, element_type='hex'), degree=3)
    # 共享四边形面：4 个顶点、8 个边内点、3 个面矩。
    assert len(set(space3.dof_map[0]) & set(space3.dof_map[1])) == 15


def test_vem_general_polyhedron_and_rotated_geometry():
    # 五棱柱含五边形面，不能用四面体/六面体的特例公式冒充通用三维实现。
    theta = 2*np.pi*np.arange(5)/5
    xy = np.column_stack([np.cos(theta), np.sin(theta)])
    nodes = np.vstack([np.column_stack([xy, np.zeros(5)]),
                       np.column_stack([xy, np.ones(5)])])
    faces = [list(range(5)), list(range(5, 10))]
    faces += [[i, (i+1) % 5, (i+1) % 5+5, i+5] for i in range(5)]
    # 非轴对齐的仿射变换，同时检验面局部坐标和物理梯度的转换。
    transform = np.array([[1., .2, .1], [.3, 1.2, -.2], [.1, .2, .9]])
    mesh = Mesh.from_arrays(nodes@transform.T, [list(range(10))],
                            element_type='polyhedron', faces=faces)
    space = FESpace.vem(mesh, degree=3)
    cell = space.cells[0]
    D, H, G, energy, l2 = cell.data
    assert np.max(np.abs(energy@D-np.eye(len(H)))) < 1e-8
    assert np.max(np.abs(l2@D-np.eye(len(H)))) < 1e-8
    exact = lambda x: 1.+np.sum(x**3)
    u = PoissonProblem(space, f=lambda x: -6.*np.sum(x), g=exact).solve(method='direct')
    assert np.max(np.abs(u-space.interpolate(exact))) < 1e-8


@pytest.mark.parametrize('k', [0, -1, 1.5, True])
def test_vem_invalid_degree(k):
    # 验证目标：拒绝零、负数、非整数及布尔类型，保证次数参数表示真正的正整数。
    with pytest.raises(ValueError):
        FESpace.vem(Mesh.rectangle(nx=1, ny=1), degree=k)
