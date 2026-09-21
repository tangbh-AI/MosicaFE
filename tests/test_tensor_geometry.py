# -*- coding: utf-8 -*-
"""四边形 / 六面体的几何变换（雅可比与梯度）回归测试。

背景（2026-09-21 修掉的第 7 号 bug）
-----------------------------------
``core/utils.tensor_jacobian`` 原来返回的是**转置**雅可比
（``dN.T @ vertices``，即 :math:`\\partial x_b/\\partial\\xi_a`），
而 :func:`~MosicaFE.core.utils.transform_gradient` 与
``LagrangeSpace._inverse_map`` 用的都是标准约定
:math:`J_{ab}=\\partial x_a/\\partial\\xi_b`（列 = 对参考坐标的偏导）。

对**轴对齐**的矩形 / 长方体，雅可比是对角阵、转置不变，所以这个错误一直
藏得住；但网格一旦旋转或畸变，程序就会静默给出错误的梯度与错误的解
（实测：正方形网格旋转 15° 后 Q1 的 L2 收敛阶从 2 掉到 0.03，45° 时误差到 1e5）。

本文件把这些情形固定成测试：雅可比必须与有限差分一致、梯度变换对线性函数
必须精确、旋转 / 畸变网格上的 patch test 必须精确、收敛阶必须达到理论值。
"""

from __future__ import annotations

import numpy as np
import pytest

import MosicaFE as mf
from MosicaFE.core.quadrature import cell_quadrature
from MosicaFE.core.utils import (
    _shape_functions_tensor,
    apply_affine_map,
    compute_jacobian,
    simplex_jacobian,
    tensor_jacobian,
    transform_gradient,
)


# ---------------------------------------------------------------------------
# 辅助：旋转 / 畸变网格与标定的制造解
# ---------------------------------------------------------------------------
def rotation(angle_deg, dim=2):
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    if dim == 2:
        return np.array([[c, -s], [s, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotated_mesh(n, angle_deg, element_type):
    """把结构化网格整体旋转：单元形状不变，只改变取向。"""
    if element_type in ("triangle", "quad"):
        base = mf.Mesh.rectangle(nx=n, ny=n, element_type=element_type)
        R = rotation(angle_deg, 2)
    else:
        base = mf.Mesh.box(nx=n, ny=n, nz=n, element_type=element_type)
        R = rotation(angle_deg, 3)
    nodes = np.asarray(base.nodes, dtype=float) @ R.T
    return mf.Mesh.from_arrays(nodes, base.elements, element_type, name="rotated")


def distorted_quad_mesh(n, sigma=0.08, seed=0):
    """随机扰动内部节点：得到非平行四边形的四边形网格。"""
    base = mf.Mesh.rectangle(nx=n, ny=n, element_type="quad")
    nodes = np.asarray(base.nodes, dtype=float)
    inner = (
        (nodes[:, 0] > 1e-12) & (nodes[:, 0] < 1 - 1e-12)
        & (nodes[:, 1] > 1e-12) & (nodes[:, 1] < 1 - 1e-12)
    )
    rng = np.random.default_rng(seed)
    nodes[inner] += rng.normal(0.0, sigma / n, size=(int(inner.sum()), 2))
    return mf.Mesh.from_arrays(nodes, base.elements, "quad", name="distorted")


def distorted_hex_mesh(n, sigma=0.04, seed=0):
    """随机扰动内部节点：得到非长方体形状的六面体网格。"""
    base = mf.Mesh.box(nx=n, ny=n, nz=n, element_type="hex")
    nodes = np.asarray(base.nodes, dtype=float)
    interior = np.all((nodes > 1e-12) & (nodes < 1 - 1e-12), axis=1)
    rng = np.random.default_rng(seed)
    nodes[interior] += rng.normal(0.0, sigma / n, size=(int(interior.sum()), 3))
    return mf.Mesh.from_arrays(nodes, base.elements, "hex", name="distorted_hex")


def pulled_back_solution(R, dim):
    """把 sin(πξ₁)… 沿 R 拉回：在**旋转后**的区域边界上恰好为 0。"""
    def exact(x):
        xi = R.T @ np.asarray(x, dtype=float)
        if dim == 2:
            return np.sin(np.pi * xi[0]) * np.sin(np.pi * xi[1])
        return np.sin(np.pi * xi[0]) * np.sin(np.pi * xi[1]) * np.sin(np.pi * xi[2])

    return exact, (lambda x: dim * np.pi ** 2 * exact(x))


def fd_jacobian(vertices, xi, eps=1e-6):
    """中心差分给出的真雅可比 ``J[a, b] = ∂x_a/∂ξ_b``。"""
    xi = np.asarray(xi, dtype=float)
    d = xi.size
    J = np.zeros((d, d))
    for k in range(d):
        step = np.zeros(d)
        step[k] = eps
        J[:, k] = (
            apply_affine_map(xi + step, vertices)[0]
            - apply_affine_map(xi - step, vertices)[0]
        ) / (2 * eps)
    return J


def ref_gradient_fd(vertices, xi, eps=1e-7):
    """参考单元上的形函数梯度（有限差分），形状 ``(n_basis, d)``。"""
    d = np.asarray(xi, dtype=float).size
    n = len(vertices)
    out = np.zeros((n, d))
    for k in range(d):
        step = np.zeros(d)
        step[k] = eps
        out[:, k] = (
            _shape_functions_tensor(np.asarray(xi) + step, d)
            - _shape_functions_tensor(np.asarray(xi) - step, d)
        ) / (2 * eps)
    return out


# ---------------------------------------------------------------------------
# 1. 雅可比本身
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,vertices,xi",
    [
        ("轴对齐正方形", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [0.3, 0.7]),
        ("旋转正方形", [[0.0, 0.0], [0.7, 0.7], [0.0, 1.4], [-0.7, 0.7]], [0.3, 0.7]),
        ("平行四边形", [[0.0, 0.0], [1.0, 0.2], [0.8, 1.1], [-0.2, 0.9]], [0.3, 0.7]),
        ("畸变四边形", [[0.0, 0.0], [1.0, 0.1], [1.2, 1.0], [0.1, 0.9]], [0.35, 0.6]),
        ("畸变六面体",
         [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.1, 1.0, 0.05], [0.1, 1.0, 0.0],
          [0.0, 0.0, 1.0], [1.0, 0.1, 1.05], [1.05, 1.0, 1.0], [0.1, 1.0, 0.95]],
         [0.3, 0.6, 0.4]),
    ],
)
def test_tensor_jacobian_matches_finite_difference(name, vertices, xi):
    """张量积单元的 J 必须与有限差分一致（列 = ∂x/∂ξ）。"""
    v = np.asarray(vertices, dtype=float)
    xi = np.asarray(xi, dtype=float)
    J, detJ = tensor_jacobian(v, xi)
    J_fd = fd_jacobian(v, xi)
    assert np.abs(J - J_fd).max() < 1e-6 * max(np.abs(J_fd).max(), 1.0)
    assert detJ == pytest.approx(np.linalg.det(J_fd), rel=1e-6, abs=1e-9)


def test_jacobian_convention_consistent_between_simplex_and_tensor():
    """两个分支必须用同一约定：J[:, k] = ∂x/∂ξ_k。"""
    tri = np.array([[0.0, 0.0], [1.1, 0.2], [0.3, 1.0]])
    J_tri, _ = simplex_jacobian(tri)
    assert np.abs(J_tri - fd_jacobian(tri, [0.0, 0.0])).max() < 1e-6

    quad = np.array([[0.0, 0.0], [1.0, 0.1], [1.2, 1.0], [0.1, 0.9]])
    J_quad, _ = compute_jacobian(quad, np.array([0.4, 0.4]))
    assert np.abs(J_quad - fd_jacobian(quad, [0.4, 0.4])).max() < 1e-6


def test_compute_jacobian_defaults_to_reference_center():
    quad = np.array([[0.0, 0.0], [1.0, 0.1], [1.2, 1.0], [0.1, 0.9]])
    J_default, det_default = compute_jacobian(quad)
    J_center, det_center = compute_jacobian(quad, np.array([0.5, 0.5]))
    assert np.allclose(J_default, J_center)
    assert det_default == pytest.approx(det_center)


# ---------------------------------------------------------------------------
# 2. 梯度变换与逆映射（出错时用户直接能看到的地方）
# ---------------------------------------------------------------------------
def test_gradient_transform_is_exact_for_linear_function():
    """畸变四边形上，线性函数的梯度必须被精确还原。"""
    mesh = distorted_quad_mesh(4, sigma=0.35, seed=3)
    V = mf.FESpace.lagrange(mesh, 1)
    lin = lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1]      # noqa: E731
    exact_grad = np.array([2.0, -3.0])
    u = V.interpolate(lin)
    for c in range(mesh.n_cells):
        pts, _ = cell_quadrature(mesh, c, 3)
        g = V.evaluate_gradient(c, u, pts)
        assert np.abs(g - exact_grad).max() < 1e-9


def test_local_stiffness_matches_reference_on_distorted_quad():
    """单元刚度矩阵与"有限差分梯度 + 同一求积规则"的参考值一致。"""
    mesh = distorted_quad_mesh(2, sigma=0.35, seed=5)
    V = mf.FESpace.lagrange(mesh, 1)
    quad = V.get_quadrature(V.default_quad_order())
    for c in range(mesh.n_cells):
        v = mesh.get_cell_vertices(c)
        nl = len(mesh.elements[c])
        K_ref = np.zeros((nl, nl))
        for q, xi in enumerate(quad.points):
            J = fd_jacobian(v, xi, eps=1e-7)
            dN = transform_gradient(ref_gradient_fd(v, xi), J)
            K_ref += quad.weights[q] * abs(np.linalg.det(J)) * (dN @ dN.T)
        assert np.abs(V.local_stiffness(c) - K_ref).max() < 1e-5


def test_inverse_map_round_trip_on_distorted_quad():
    """物理点映回参考单元再映回来应回到原点，参考坐标落在 [0,1]² 内。"""
    mesh = distorted_quad_mesh(3, sigma=0.3, seed=7)
    V = mf.FESpace.lagrange(mesh, 1)
    for c in range(mesh.n_cells):
        pts, _ = cell_quadrature(mesh, c, 4)
        ref = V._inverse_map(c, pts)
        assert np.isfinite(ref).all()
        assert ref.min() > -1e-8 and ref.max() < 1 + 1e-8
        back = apply_affine_map(ref, mesh.get_cell_vertices(c))
        assert np.abs(back - pts).max() < 1e-10


# ---------------------------------------------------------------------------
# 3. 端到端：patch test 与收敛阶
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("degree", [1, 2])
def test_patch_test_on_distorted_and_rotated_quad(degree):
    """线性精确解在畸变 / 旋转四边形网格上仍应被精确重现。"""
    lin = lambda x: 1.0 + 2.0 * x[0] - 0.5 * x[1]      # noqa: E731
    meshes = {
        "distorted": distorted_quad_mesh(4, sigma=0.25, seed=11),
        "rotated15": rotated_mesh(4, 15, "quad"),
        "rotated45": rotated_mesh(4, 45, "quad"),
    }
    for name, mesh in meshes.items():
        V = mf.FESpace.lagrange(mesh, degree)
        problem = mf.PoissonProblem(V, f=lambda x: 0.0, g=lin)
        u = problem.solve()
        err = float(np.abs(u - V.interpolate(lin)).max())
        assert err < 1e-9, f"{name} Q{degree} patch test 失败：{err:.3e}"


def test_patch_test_on_distorted_hex():
    lin = lambda x: 1.0 + x[0] - 2.0 * x[1] + 0.5 * x[2]   # noqa: E731
    mesh = distorted_hex_mesh(2, sigma=0.25, seed=13)
    V = mf.FESpace.lagrange(mesh, 1)
    problem = mf.PoissonProblem(V, f=lambda x: 0.0, g=lin)
    u = problem.solve()
    assert float(np.abs(u - V.interpolate(lin)).max()) < 1e-9


def _l2_sequence(mesh_factory, degree, dim, levels):
    hs, errs = [], []
    for n in levels:
        mesh, R = mesh_factory(n)
        exact, src = pulled_back_solution(R, dim)
        V = mf.FESpace.lagrange(mesh, degree)
        problem = mf.PoissonProblem(V, f=src, g=lambda x: 0.0)
        u = problem.solve()
        hs.append(mesh.get_mesh_size())
        errs.append(problem.compute_error(u, exact, "L2"))
    return hs, errs


def test_q1_is_invariant_under_rotation_and_converges():
    """Q1 在旋转区域上的误差必须与轴对齐时相同，且保持二阶收敛。"""
    def factory(angle):
        return lambda n: (rotated_mesh(n, angle, "quad"), rotation(angle, 2))

    hs0, e0 = _l2_sequence(factory(0), 1, 2, (4, 8, 16))
    assert mf.estimate_convergence_rate(hs0, e0, use_last=2) > 1.8
    for angle in (15, 30, 45):
        hs, e = _l2_sequence(factory(angle), 1, 2, (4, 8, 16))
        assert np.abs(np.array(e) - np.array(e0)).max() < 1e-10
        assert mf.estimate_convergence_rate(hs, e, use_last=2) > 1.8


def test_q2_converges_on_rotated_quad_mesh():
    def factory(n):
        return rotated_mesh(n, 30, "quad"), rotation(30, 2)

    hs, e = _l2_sequence(factory, 2, 2, (4, 8))
    assert mf.estimate_convergence_rate(hs, e, use_last=2) > 2.7


def test_q1_hex_converges_on_rotated_box():
    def factory(n):
        return rotated_mesh(n, 30, "hex"), rotation(30, 3)

    hs, e = _l2_sequence(factory, 1, 3, (2, 4, 8))
    assert mf.estimate_convergence_rate(hs, e, use_last=2) > 1.8


def test_p1_triangle_is_unaffected():
    """单纯形分支本来就正确，旋转后误差同样应当相同。"""
    def factory(angle):
        return lambda n: (rotated_mesh(n, angle, "triangle"), rotation(angle, 2))

    hs0, e0 = _l2_sequence(factory(0), 1, 2, (4, 8))
    hs, e = _l2_sequence(factory(30), 1, 2, (4, 8))
    assert np.abs(np.array(e) - np.array(e0)).max() < 1e-10
    assert mf.estimate_convergence_rate(hs, e, use_last=2) > 1.7
