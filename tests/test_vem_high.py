"""High-order VEM consistency, topology, quadrature and PDE regression tests."""
import math
import numpy as np
import pytest
import MosicaFE as mf
from MosicaFE.core.quadrature import QuadratureRule


@pytest.mark.parametrize('dim,k,kind', [(2,2,'triangle'), (2,3,'quad'),
    (2,4,'triangle'), (3,2,'tet'), (3,3,'hex'), (3,4,'tet')])
def test_high_order_projectors_and_bilinear_forms(dim, k, kind):
    mesh = (mf.Mesh.rectangle(nx=1, ny=1, element_type=kind) if dim == 2 else
            mf.Mesh.box(nx=1, ny=1, nz=1, element_type=kind))
    # A shear and rotation exercise face frames away from coordinate planes.
    transform = np.eye(dim)
    transform[0, 1] = .27
    if dim == 3:
        transform[2, 0] = -.19
    mesh.nodes = mesh.nodes @ transform.T + .13
    v = mf.FESpace.vem(mesh, degree=k)
    p = v._projection_high(0)
    D, H, B = p['D'], p['H'], p['B']
    eye = np.eye(D.shape[1])
    assert np.max(abs(p['Pi'] @ D-eye)) < 2e-8
    assert np.max(abs(p['Pi0'] @ D-eye)) < 2e-8
    K, M = v.local_stiffness(0), v.local_mass(0)
    assert np.max(abs(K-K.T)) < 1e-9
    assert np.max(abs(M-M.T)) < 1e-9
    assert np.max(abs(D.T @ K @ D-B)) < 1e-8
    assert np.max(abs(D.T @ M @ D-H)) < 1e-8
    assert np.linalg.eigvalsh(K)[0] > -1e-8
    assert np.linalg.eigvalsh(M)[0] > 0
    # Full polynomial load consistency (also tests enhanced higher moments).
    from MosicaFE.spaces.vem import _scaled_monomial_values
    for j in (0, len(p['multi'])-1):
        f = lambda x: _scaled_monomial_values(p['multi'], np.atleast_2d(x),
                                              p['centroid'], p['h'])[:, j]
        assert np.max(abs(v.local_load(0, f)-M @ D[:, j])) < 2e-9


@pytest.mark.parametrize('dim,k,kind', [(2,2,'triangle'),(2,3,'quad'),
                                      (3,2,'tet'),(3,3,'hex')])
def test_high_order_nonzero_boundary_polynomial_patch(dim, k, kind):
    mesh = (mf.Mesh.rectangle(nx=2, ny=2, element_type=kind) if dim == 2 else
            mf.Mesh.box(nx=2, ny=2, nz=2, element_type=kind))
    v = mf.FESpace.vem(mesh, degree=k)
    exact = lambda x: 1 + np.sum(np.asarray(x)**k, axis=-1)
    lap = lambda x: k*(k-1)*np.sum(np.asarray(x)**(k-2), axis=-1)
    for wave in (0.0, 1.0):
        f = lambda x: -lap(x)-wave**2*exact(x)
        problem = mf.HelmholtzProblem(v, f=f, g=exact, k=wave)
        u = problem.solve(method='direct')
        assert np.max(abs(u-v.interpolate(exact))) < 2e-8


def test_boundary_topology_does_not_constrain_interior_diagonal():
    v = mf.FESpace.vem(mf.Mesh.rectangle(nx=1, ny=1), degree=2)
    # Four vertices plus four boundary edge midpoints, excluding the diagonal.
    assert len(v.boundary_dofs()) == 8


@pytest.mark.parametrize('k', [2,3])
def test_irregular_polygon_patch(k):
    mesh = mf.Mesh.voronoi_polygon(n_points=16, seed=7)
    v = mf.FESpace.vem(mesh, degree=k)
    exact = lambda x: np.sum(np.asarray(x)**k, axis=-1)
    source = lambda x: -k*(k-1)*np.sum(np.asarray(x)**(k-2), axis=-1)
    u = mf.PoissonProblem(v, f=source, g=exact).solve(method='direct')
    assert np.max(abs(u-v.interpolate(exact))) < 2e-8


def test_polyhedron_with_pentagonal_faces():
    angle = np.arange(5)*2*np.pi/5
    xy = np.column_stack((np.cos(angle),np.sin(angle)))
    xyz = np.vstack((np.column_stack((xy,np.zeros(5))),np.column_stack((xy,np.ones(5)))))
    faces = [list(range(5)),list(range(5,10))]
    faces += [[i,(i+1)%5,(i+1)%5+5,i+5] for i in range(5)]
    mesh = mf.Mesh(xyz,[list(range(10))],element_type='polyhedron',faces=faces)
    v = mf.FESpace.vem(mesh,degree=2)
    p = v._projection_high(0)
    assert abs(p['Pi'] @ p['D']-np.eye(10)).max() < 1e-9
    exact = lambda x: np.sum(np.asarray(x)**2,axis=-1)
    u = mf.PoissonProblem(v,f=lambda x:-6.,g=exact).solve(method='direct')
    assert abs(u-v.interpolate(exact)).max() < 1e-9


@pytest.mark.parametrize('dim,order', [(2,6),(2,10),(3,4),(3,8)])
def test_high_order_simplex_quadrature(dim, order):
    from MosicaFE.spaces.vem import _monomial_indices
    q = QuadratureRule.get_quadrature('triangle' if dim == 2 else 'tet', order)
    assert np.min(q.weights) > 0
    for a in _monomial_indices(dim, order):
        value = np.prod(q.points ** np.array(a), axis=1) @ q.weights
        target = math.prod(math.factorial(i) for i in a)/math.factorial(sum(a)+dim)
        assert abs(value-target) < 2e-14


@pytest.mark.parametrize('dim,k', [(2,2),(2,3),(3,2),(3,3)])
def test_high_order_convergence(dim, k):
    exact = lambda x: np.prod(np.sin(np.pi*np.asarray(x)), axis=-1)
    def gradient(x):
        s = np.sin(np.pi*x)
        return np.array([np.pi*np.cos(np.pi*x[d])*np.prod(np.delete(s,d)) for d in range(dim)])
    errors = []
    for n in (2,4,8) if dim == 2 else (2,4):
        mesh = (mf.Mesh.rectangle(nx=n,ny=n) if dim == 2 else
                mf.Mesh.box(nx=n,ny=n,nz=n))
        v = mf.FESpace.vem(mesh, degree=k)
        p = mf.HelmholtzProblem(v, f=lambda x:(dim*np.pi**2-1)*exact(x), k=1)
        u = p.solve(method='direct')
        e = p.compute_errors(u, exact, gradient)
        errors.append((e['L2'], e['H1']))
        assert np.isfinite(e['Linf'])
        assert np.linalg.norm(p._system[0] @ u-p._system[1]) < 1e-8
    rates = np.log2(np.array(errors[-2])/np.array(errors[-1]))
    assert rates[0] > k+.5
    assert rates[1] > k-.4
