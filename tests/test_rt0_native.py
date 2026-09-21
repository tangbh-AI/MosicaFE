# -*- coding: utf-8 -*-
"""RT0×P0 混合元作为**库原生实现**的测试。

覆盖四层：

* :mod:`MosicaFE.core.facets` —— 面拓扑、共享面、外法向定向；
* :mod:`MosicaFE.spaces.rt0` —— 显式基函数、精确质量矩阵、通量/压力自由度；
* :mod:`MosicaFE.physics.mixed_poisson` —— 鞍点组装、三种求解方式、守恒性；
* :mod:`MosicaFE.postprocess.error` —— 混合格式的误差范数。

这些测试全部只依赖 MosicaFE 自身（numpy + 可选 scipy），
不依赖任何外部有限元后端。
"""

from __future__ import annotations

import numpy as np
import pytest

import MosicaFE as mf
from MosicaFE.core.facets import build_facet_topology, local_facets
from MosicaFE.core.quadrature import simplex_quadrature
from MosicaFE.postprocess.error import mixed_error_norms
from MosicaFE.spaces.rt0 import rt0_flux_basis, rt0_mass_matrices


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


def flux2(x):
    return -grad2(x)


# ---------------------------------------------------------------------------
# 面拓扑
# ---------------------------------------------------------------------------
def test_facet_topology_counts_2d():
    """4×4 三角形网格：边数 = 内部边 40 + 边界边 16。"""
    topo = build_facet_topology(mf.Mesh.rectangle(nx=4, ny=4))
    assert topo.n_facets == 56
    assert (topo.n_interior, topo.n_boundary) == (40, 16)
    # 每个单元 3 条边，每条内部边被两个单元记录
    assert topo.cell_facets.shape == (32, 3)
    assert int(topo.counts.sum()) == 32 * 3


def test_facet_topology_counts_3d():
    """2×2×2 四面体网格（Kuhn 剖分）：面 120 = 内部 72 + 边界 48。"""
    topo = build_facet_topology(mf.Mesh.box(nx=2, ny=2, nz=2, element_type="tet"))
    assert topo.n_facets == 120
    assert (topo.n_interior, topo.n_boundary) == (72, 48)
    assert topo.cell_facets.shape == (48, 4)


def test_facet_topology_orientation():
    """内部面两侧符号相反，且 signs·normals 确实指向单元外部。"""
    mesh = mf.Mesh.box(nx=2, ny=2, nz=2, element_type="tet")
    topo = build_facet_topology(mesh)
    nodes = np.asarray(mesh.nodes, dtype=float)

    for f, cells in enumerate(topo.facet_cells):
        # 符号一致性：内部面两侧相反（两侧共享同一个全局未知量）
        signs = [topo.signs[c][int(np.where(topo.cell_facets[c] == f)[0][0])]
                 for c in cells]
        if len(cells) == 2:
            assert signs[0] * signs[1] == pytest.approx(-1.0)
        else:
            assert signs[0] in (-1.0, 1.0)
        # 几何一致性：signs·normals 指向单元外侧
        for c in cells:
            i = int(np.where(topo.cell_facets[c] == f)[0][0])
            outward = topo.signs[c, i] * topo.normals[f]
            cell_center = nodes[mesh.elements[c]].mean(axis=0)
            facet_center = nodes[topo.facets[f]].mean(axis=0)
            assert float(np.dot(outward, facet_center - cell_center)) > 0.0


def test_facet_topology_rejects_non_simplex():
    with pytest.raises(ValueError):
        build_facet_topology(mf.Mesh.rectangle(nx=2, ny=2, element_type="quad"))
    with pytest.raises(ValueError):
        build_facet_topology(mf.Mesh.pentagon(nx=2, ny=2))


def test_local_facets_are_opposite_to_vertices():
    assert local_facets(2) == ((1, 2), (0, 2), (0, 1))
    assert local_facets(3) == ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


# ---------------------------------------------------------------------------
# 单元基函数与质量矩阵
# ---------------------------------------------------------------------------
def _random_simplex(dim, rng):
    """构造一个（近）正定的随机单纯形。"""
    v = rng.random((dim + 1, dim))
    v[1:] = v[0] + 1.6 * (v[1:] - v[0]) + 0.25
    # 保证正定向（与 Mesh._orient_elements 的约定一致）
    J = np.array([v[i + 1] - v[0] for i in range(dim)])
    if np.linalg.det(J) < 0:
        v[[0, 1]] = v[[1, 0]]
    return v


def test_rt0_basis_face_flux_is_one():
    """基函数满足 ∫_{f_i} φ_i·n_i ds = 1（面通量自由度归一化）。"""
    rng = np.random.default_rng(1)
    for dim in (2, 3):
        v = _random_simplex(dim, rng)
        for i in range(dim + 1):
            face = v[[j for j in range(dim + 1) if j != i]]
            pts, w = simplex_quadrature(face, order=3)
            # 面的外法向（由物理几何给出，与拓扑排序无关）
            if dim == 2:
                t = face[1] - face[0]
                n = np.array([t[1], -t[0]])
            else:
                n = np.cross(face[1] - face[0], face[2] - face[0])
            center = v.mean(axis=0)
            if np.dot(n, face.mean(axis=0) - v[i]) < 0:
                n = -n
            n = n / np.linalg.norm(n)
            phi_i = rt0_flux_basis(v, pts)[:, i, :]
            assert float(np.sum(w * (phi_i @ n))) == pytest.approx(1.0, rel=1e-10)


def test_rt0_mass_matrix_closed_form_matches_quadrature():
    """精确闭式质量矩阵 vs 高阶求积（相对误差 ~1e-15）。"""
    rng = np.random.default_rng(2)
    for dim in (2, 3):
        for _ in range(5):
            v = _random_simplex(dim, rng)
            M = rt0_mass_matrices(v[None, ...])[0]
            pts, w = simplex_quadrature(v, order=3 if dim == 2 else 2)
            phi = rt0_flux_basis(v, pts)
            Mq = np.einsum("q,qid,qjd->ij", w, phi, phi)
            assert np.abs(M - Mq).max() < 1e-13 * max(np.abs(M).max(), 1.0)
            # 对称正定
            assert np.abs(M - M.T).max() < 1e-14 * np.abs(M).max()
            assert np.linalg.eigvalsh(M).min() > 0.0


def test_rt0_mass_matrix_is_orientation_independent():
    """交换两个顶点（翻转定向）不改变局部质量矩阵。"""
    v = np.array([[0.0, 0.0], [1.2, 0.1], [0.3, 1.1]])
    M = rt0_mass_matrices(v[None, ...])[0]
    w = np.array([[0.0, 0.0], [1.2, 0.1], [0.3, 1.1]])
    perm = [0, 2, 1]
    M2 = rt0_mass_matrices(w[perm][None, ...])[0]
    assert np.abs(M - M2[np.ix_(perm, perm)]).max() < 1e-14 * np.abs(M).max()


def test_lumped_mass_is_diagonally_dominant():
    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    M = rt0_mass_matrices(v[None, ...], lumped=True)[0]
    assert np.all(np.diag(M) > 0)
    assert np.abs(M - M.T).max() < 1e-14
    # 集中版本与精确版本量级一致
    Me = rt0_mass_matrices(v[None, ...])[0]
    assert 0.2 < (np.trace(M) / np.trace(Me)) < 5.0


# ---------------------------------------------------------------------------
# 空间层
# ---------------------------------------------------------------------------
def test_space_dof_layout_and_coordinates():
    mesh = mf.Mesh.rectangle(nx=4, ny=4)
    V = mf.FESpace.rt0(mesh)
    assert V.n_flux == mesh.facet_topology.n_facets
    assert V.n_pressure == mesh.n_cells
    assert V.n_dofs == V.n_flux + V.n_pressure
    # 局部自由度 = d+1 个面 + 1 个压力
    for c in range(mesh.n_cells):
        dofs = V.get_cell_dofs(c)
        assert dofs.size == 4
        assert V.cell_pressure_dof(c) == dofs[-1]
    # 边界是自然条件：没有需要强加的自由度
    assert V.boundary_dofs().size == 0
    # 自由度坐标：通量在面心、压力在单元形心
    assert np.abs(V.dof_coords[0] - mesh.nodes[V.facet_vertices(0)].mean(axis=0)).max() < 1e-14
    assert np.abs(V.dof_coords[V.n_flux] - mesh.get_cell_centroid(0)).max() < 1e-14


def test_space_reference_nodes_map_back_to_dofs():
    """参考坐标经仿射映射应回到自由度的物理坐标。"""
    from MosicaFE.core.utils import apply_affine_map

    mesh = mf.Mesh.box(nx=1, ny=1, nz=1, element_type="tet")
    V = mf.FESpace.rt0(mesh)
    c = 0
    phys = apply_affine_map(V.reference_nodes(c), V.cell_vertices(c))
    dofs = V.get_cell_dofs(c)
    assert np.abs(phys - V.dof_coords[dofs]).max() < 1e-12


def test_global_blocks_have_expected_structure():
    mesh = mf.Mesh.rectangle(nx=6, ny=6)
    V = mf.FESpace.rt0(mesh)
    M = V.flux_mass_matrix()
    B = V.divergence_matrix()
    assert M.shape == (V.n_flux, V.n_flux)
    assert B.shape == (V.n_pressure, V.n_flux)
    Md = np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M)
    assert np.abs(Md - Md.T).max() < 1e-12 * np.abs(Md).max()
    assert np.linalg.eigvalsh(Md).min() > 0.0
    Bd = np.asarray(B.todense()) if hasattr(B, "todense") else np.asarray(B)
    # 每行恰好 d+1 个 ±1
    assert np.all(np.abs(Bd).sum(axis=1) == 3)
    assert np.all(np.isin(Bd, [-1.0, 0.0, 1.0]))


def test_interpolate_flux_matches_face_integrals():
    """interpolate_flux 得到的自由度等于 ∫_f q·n_f ds。"""
    mesh = mf.Mesh.rectangle(nx=4, ny=4)
    V = mf.FESpace.rt0(mesh)
    q = lambda x: np.array([1.0 + x[1], 2.0 - x[0]])       # noqa: E731
    Phi = V.interpolate_flux(q)
    nodes = np.asarray(mesh.nodes, dtype=float)
    for f in range(V.n_flux):
        verts = nodes[V.facet_vertices(f)]
        pts, w = simplex_quadrature(verts, order=3)
        vals = np.array([q(p) for p in pts])
        expect = float(np.sum(w * (vals @ V.topology.normals[f])))
        assert Phi[f] == pytest.approx(expect, rel=1e-10, abs=1e-12)


def test_interpolate_pressure_is_cell_average():
    mesh = mf.Mesh.rectangle(nx=3, ny=3)
    V = mf.FESpace.rt0(mesh)
    lin = lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1]          # noqa: E731
    P = V.interpolate_pressure(lin)
    exact_avg = np.array(
        [float(lin(mesh.get_cell_centroid(c))) for c in range(mesh.n_cells)]
    )
    assert np.abs(P - exact_avg).max() < 1e-12


def test_mixed_space_rejects_single_operator_api():
    V = mf.FESpace.rt0(mf.Mesh.rectangle(nx=2, ny=2))
    for call in (V.stiffness_matrix, V.mass_matrix, lambda: V.load_vector(source2)):
        with pytest.raises(NotImplementedError):
            call()
    with pytest.raises(NotImplementedError):
        V.interpolate(source2)


# ---------------------------------------------------------------------------
# 物理层
# ---------------------------------------------------------------------------
def test_direct_and_schur_agree():
    mesh = mf.Mesh.rectangle(nx=8, ny=8)
    V = mf.FESpace.rt0(mesh)
    p1 = mf.MixedPoissonProblem(V, f=source2)
    Q1, P1 = p1.solve(method="direct")
    p2 = mf.MixedPoissonProblem(V, f=source2)
    Q2, P2 = p2.solve(method="schur")
    assert np.abs(P1 - P2).max() < 1e-10 * max(np.abs(P1).max(), 1.0)
    assert np.abs(Q1 - Q2).max() < 1e-10 * max(np.abs(Q1).max(), 1.0)
    # 两种方式都应该满足离散守恒
    assert p1.check_conservation() < 1e-9
    assert p2.check_conservation() < 1e-9


def test_conservation_on_3d_mesh():
    mesh = mf.Mesh.box(nx=2, ny=2, nz=2, element_type="tet")
    V = mf.FESpace.rt0(mesh)
    problem = mf.MixedPoissonProblem(V, f=lambda x: 1.0 + x[0] * x[1])
    problem.solve()
    resid = problem.check_conservation()
    assert resid < 1e-12
    # 守恒式与右端项逐单元一致
    assert np.abs(problem.flux.sum(axis=1) - problem.source).max() < 1e-12


def test_flux_at_points_reproduces_local_reconstruction():
    mesh = mf.Mesh.rectangle(nx=6, ny=6)
    V = mf.FESpace.rt0(mesh)
    problem = mf.MixedPoissonProblem(V, f=source2)
    Q, P = problem.solve()
    centers = np.vstack([mesh.get_cell_centroid(c) for c in range(mesh.n_cells)])
    got = problem.flux_at_points(centers)
    for c in range(mesh.n_cells):
        assert np.abs(got[c] - V.evaluate_flux(c, Q[c], centers[c][None, :])[0]).max() < 1e-12


def test_mixed_error_norms_native_implementation():
    """原生误差范数与 compute_errors 的返回值一致，且通量为一阶收敛。"""
    hs, p_l2, q_l2 = [], [], []
    for n in (4, 8, 16):
        mesh = mf.Mesh.rectangle(nx=n, ny=n)
        V = mf.FESpace.rt0(mesh)
        problem = mf.MixedPoissonProblem(V, f=source2)
        Q, P = problem.solve()
        norms = mixed_error_norms(V, P, Q, exact2, flux2)
        assert set(["p_l2", "p_linf", "q_l2", "q_linf"]).issubset(norms)
        errs = problem.compute_errors((Q, P), exact2, flux2)
        assert errs["L2"] == pytest.approx(norms["p_l2"], rel=1e-12)
        assert errs["flux_L2"] == pytest.approx(norms["q_l2"], rel=1e-12)
        # 未给 q_exact 时通量误差为 nan
        only_p = mixed_error_norms(V, P, Q, exact2)
        assert np.isnan(only_p["q_l2"])
        hs.append(mesh.get_mesh_size())
        p_l2.append(norms["p_l2"])
        q_l2.append(norms["q_l2"])
    assert mf.estimate_convergence_rate(hs, p_l2, use_last=2) > 0.9
    assert mf.estimate_convergence_rate(hs, q_l2, use_last=2) > 0.9


def test_nonhomogeneous_dirichlet_is_rejected():
    V = mf.FESpace.rt0(mf.Mesh.rectangle(nx=2, ny=2))
    with pytest.raises(NotImplementedError):
        mf.MixedPoissonProblem(V, f=source2, g=lambda x: 1.0)


def test_solve_poisson_api_uses_native_rt0():
    mesh = mf.Mesh.rectangle(nx=6, ny=6)
    out = mf.solve_poisson(mesh, element="RT0", f=source2)
    assert isinstance(out, tuple) and len(out) == 2
    Q, P = out
    assert P.size == mesh.n_cells
    with pytest.raises(ValueError):
        mf.solve_poisson(mesh, element="RT0", f=source2, solver="cg")


def test_rt0_space_has_no_external_backend_import():
    """确认 RT0 是库自身实现：相关模块不再引用任何 backends 包。"""
    import pathlib

    import MosicaFE.physics.mixed_poisson as mixed
    import MosicaFE.spaces.rt0 as rt0

    for mod in (rt0, mixed):
        src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        assert "backends" not in src
        assert "rt0fem" not in src
