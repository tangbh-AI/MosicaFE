"""MosicaFE 的一行式高层接口。

主文件真的只需要几行：

.. code-block:: python

   import numpy as np
   from MosicaFE import Mesh, solve_poisson

   f = lambda x: 2 * np.pi ** 2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
   u = solve_poisson(Mesh.rectangle(nx=32, ny=32), element="P2", f=f)

同一个问题换成亥姆霍兹方程 ``-Laplacian(u) - k**2 u = f`` 也只是换个函数：

.. code-block:: python

   from MosicaFE import solve_helmholtz
   u = solve_helmholtz(Mesh.rectangle(nx=32, ny=32), element="P2", f=f, k=1.0)

``element`` 可选（大小写不敏感）：

===============  =========================================
``"P1" / "P2"``  经典 Lagrange 有限元（三角/四面体）
``"Q1" / "Q2"``  张量积 Lagrange（四边形/六面体）
``"VEM" / "VEM1"``虚拟元（任意多边形 / 多面体）
``"RT0"``        RT0×P0 混合元（返回 ``(Q, P)``）
===============  =========================================

``solver`` 可选 ``"auto" / "direct" / "cg" / "gmres" / "bicgstab" / "minres"``；
RT0 的鞍点系统是对称**不定**的，只接受 ``"auto" / "direct" / "schur" / "minres"
/ "gmres" / "bicgstab"``。
"""

from __future__ import annotations

import numpy as np

from .core.mesh import Mesh
from .physics.conduction import ConductionProblem
from .physics.darcy import DarcyProblem
from .physics.helmholtz import HelmholtzProblem
from .physics.mixed_poisson import MixedPoissonProblem
from .physics.poisson import PoissonProblem
from .physics.registry import create_problem
from .spaces.factory import FESpace

__all__ = [
    "ELEMENT_ALIASES",
    "resolve_element",
    "make_space",
    "solve_poisson",
    "solve_helmholtz",
    "solve_darcy",
    "solve_conduction",
    "run",
]

#: RT0 的鞍点系统可用的求解方式
_MIXED_SOLVERS = ("auto", "direct", "schur", "minres", "gmres", "bicgstab")

#: 元的名字 → (空间族, 默认次数)
ELEMENT_ALIASES = {
    "P1": ("lagrange", 1),
    "Q1": ("lagrange", 1),
    "P2": ("lagrange", 2),
    "Q2": ("lagrange", 2),
    "LAGRANGE": ("lagrange", None),
    "FEM": ("lagrange", None),
    "VEM": ("vem", 1),
    "VEM1": ("vem", 1),
    "VIRTUAL": ("vem", 1),
    "RT0": ("rt0", 1),
    "RT0-P0": ("rt0", 1),
    "MIXED": ("rt0", 1),
}


def resolve_element(element, degree: int = None):
    """把 ``element`` 字符串解析成 ``(family, degree)``。"""
    key = str(element).strip().upper().replace(" ", "")
    if key in ELEMENT_ALIASES:
        family, default_degree = ELEMENT_ALIASES[key]
    elif key.startswith("VEM"):
        family, default_degree = "vem", 1
    elif key.startswith("RT0") or key.startswith("MIXED"):
        family, default_degree = "rt0", 1
    elif key[:1] in ("P", "Q") and key[1:].isdigit():
        family, default_degree = "lagrange", int(key[1:])
    else:
        raise ValueError(
            f"未知元 {element!r}；可用：{sorted(ELEMENT_ALIASES)}"
        )
    return family, int(degree if degree is not None else (default_degree or 1))


def make_space(mesh, element: str = "P1", degree: int = None, **kwargs):
    """按名字构造离散空间。"""
    family, deg = resolve_element(element, degree)
    return FESpace.create(mesh, family=family, degree=deg, **kwargs)


def _coerce_mesh(mesh):
    if isinstance(mesh, Mesh):
        return mesh
    if isinstance(mesh, dict):
        return Mesh(**mesh)
    raise TypeError(f"无法识别的网格 {mesh!r}；请传 Mesh 对象或 Mesh 工厂参数字典")


def solve_poisson(
    mesh=None,
    element: str = "P1",
    f=None,
    g=None,
    kappa: float = 1.0,
    degree: int = None,
    solver: str = "auto",
    region=None,
    return_all: bool = False,
    **kwargs,
):
    """一行求解 Poisson 方程。

    Parameters
    ----------
    mesh : Mesh | dict | None
        ``Mesh`` 对象，或 ``Mesh.rectangle`` 之类的工厂参数字典；
        省略时用 ``region``。
    element : str
        ``"P1" / "P2" / "Q1" / "Q2" / "VEM" / "RT0"``。
    f : callable
        源项 ``f(x)``，``x`` 是坐标数组（与 guidebook 一致）。
    g : callable
        Dirichlet 边界值 ``g(x)``，默认 0。
    solver : str
        线性求解器。
    return_all : bool
        True 时返回 ``dict(u=..., space=..., problem=..., mesh=...)``。

    Returns
    -------
    u : ndarray
        H1 协调空间（P/Q/VEM）返回自由度向量；
        ``element="RT0"`` 返回 ``(Q, P)`` 二元组。
    """
    if mesh is None:
        if region is None:
            raise ValueError("请提供 mesh（Mesh 对象或工厂参数字典），或用 region=... 指定区域")
        mesh = region if isinstance(region, Mesh) else _coerce_mesh(region)
    else:
        mesh = _coerce_mesh(mesh)

    family, deg = resolve_element(element, degree)
    space = FESpace.create(mesh, family=family, degree=deg, **kwargs)

    if family == "rt0":
        problem = MixedPoissonProblem(space, f=f, g=g)
        method = "auto" if solver is None else str(solver).lower()
        if method not in _MIXED_SOLVERS:
            raise ValueError(
                f"RT0×P0 是混合格式（对称不定系统），solver 不能取 {solver!r}；"
                f"可用：{_MIXED_SOLVERS}"
            )
        Q, P = problem.solve(method=method)
        if return_all:
            return {"u": (Q, P), "flux": Q, "pressure": P,
                    "space": space, "problem": problem, "mesh": mesh}
        return Q, P

    problem = PoissonProblem(space, f=f, g=g, kappa=kappa)
    u = problem.solve(method=solver)
    if return_all:
        return {"u": u, "space": space, "problem": problem, "mesh": mesh}
    return u


def solve_helmholtz(
    mesh=None,
    element: str = "P1",
    f=None,
    g=None,
    k: float = 1.0,
    kappa: float = 1.0,
    degree: int = None,
    solver: str = "auto",
    region=None,
    return_all: bool = False,
    **kwargs,
):
    """一行求解亥姆霍兹方程 ``-div(kappa grad u) - k**2 u = f``。

    参数与 :func:`solve_poisson` 完全平行，只多了一个实常数波数 ``k``
    （``kappa`` 仍默认 1）：

    .. code-block:: python

       from MosicaFE import Mesh, solve_helmholtz

       f = lambda x: (2*np.pi**2 - 1.0) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
       u = solve_helmholtz(Mesh.rectangle(nx=32, ny=32), element="P2", f=f, k=1.0)

    ``k**2`` 必须小于 ``-Laplacian`` 的 Dirichlet 第一特征值 ``lambda_1``
    （单位正方形上为 ``2*pi**2``），否则问题进入共振区，矩阵对称不定，
    应改用 ``solver="direct"`` 或 ``"minres"``。

    ``element`` 接受 ``"P1" / "P2" / "Q1" / "Q2" / "VEM"``；
    ``"RT0"`` 是混合格式、没有对应的亥姆霍兹实现，会被明确拒绝。
    """
    if mesh is None:
        if region is None:
            raise ValueError("请提供 mesh（Mesh 对象或工厂参数字典），或用 region=... 指定区域")
        mesh = region if isinstance(region, Mesh) else _coerce_mesh(region)
    else:
        mesh = _coerce_mesh(mesh)

    family, deg = resolve_element(element, degree)
    if family == "rt0":
        raise ValueError(
            "RT0×P0 是混合（鞍点）格式，不支持亥姆霍兹方程；"
            "请用 element='P1' / 'P2' / 'Q1' / 'Q2' / 'VEM'。"
        )
    space = FESpace.create(mesh, family=family, degree=deg, **kwargs)
    problem = HelmholtzProblem(space, f=f, g=g, k=k, kappa=kappa)
    u = problem.solve(method=solver)
    if return_all:
        return {"u": u, "space": space, "problem": problem, "mesh": mesh}
    return u


def solve_darcy(
    mesh=None,
    element: str = "RT0",
    f=None,
    g=None,
    K=1.0,
    bc=None,
    mode: str = "cell",
    solver: str = "auto",
    region=None,
    return_all: bool = False,
    **kwargs,
):
    """一行求解 Darcy（达西）渗流问题。

    .. math::

       K^{-1}q+\\nabla p=0,\\qquad \\nabla\\cdot q=f\\ \\text{in}\\ \\Omega,
       \\qquad p=g_D\\ \\text{on}\\ \\Gamma_D,\\quad q\\cdot n=g_N\\ \\text{on}\\ \\Gamma_N .

    Parameters
    ----------
    mesh : Mesh | dict | None
        ``Mesh`` 对象或工厂参数字典；省略时用 ``region``。
    element : str
        只支持 ``"RT0"``（RT0×P0 混合元）。
    f : callable
        源汇 ``f(x)``。
    g : callable
        压力 Dirichlet 数据 ``g_D(x)``（可以为非齐次，默认 0）。
    K : float | (d,) | (d, d) | callable
        渗透率（标量 / 对角 / 张量 / 变系数）。
    bc : callable, optional
        ``bc(mid, n_out) -> "D" | "N" | ("flux", gN)``；缺省全边界定压。
    mode : {"cell", "quad"}
        加权通量质量矩阵的组装方式。
    solver : str
        ``"auto" / "schur" / "schur_cg" / "direct" / "minres" / "gmres" / "bicgstab"``。
    return_all : bool
        True 时返回 ``dict(flux=..., pressure=..., space=..., problem=..., mesh=...)``。

    Returns
    -------
    (Q, P) : tuple
        ``Q`` 为 ``(n_cells, d+1)`` 的逐单元外法向通量，``P`` 为 ``(n_cells,)`` 常数压力。

    Examples
    --------
    .. code-block:: python

       from MosicaFE import Mesh, solve_darcy

       # 左右定压、上下不透水，渗透率 2.5：解应是 q=(2.5, 0)
       bc = lambda mid, n: "D" if (mid[0] < 1e-12 or mid[0] > 1 - 1e-12) else "N"
       Q, P = solve_darcy(Mesh.rectangle(nx=16, ny=16),
                          f=lambda x: 0.0, g=lambda x: 1.0 - x[:, 0],
                          K=2.5, bc=bc)
    """
    if mesh is None:
        if region is None:
            raise ValueError("请提供 mesh（Mesh 对象或工厂参数字典），或用 region=... 指定区域")
        mesh = region if isinstance(region, Mesh) else _coerce_mesh(region)
    else:
        mesh = _coerce_mesh(mesh)

    family, deg = resolve_element(element)
    if family != "rt0":
        raise ValueError(
            f"Darcy 问题需要 RT0×P0 混合元，element 应为 'RT0'；收到 {element!r}"
        )
    space = FESpace.create(mesh, family=family, degree=deg, **kwargs)

    method = "auto" if solver is None else str(solver).lower()
    if method not in _MIXED_SOLVERS + ("schur_cg",):
        raise ValueError(
            f"Darcy 问题是鞍点系统，solver 不能取 {solver!r}；"
            f"可用：{_MIXED_SOLVERS + ('schur_cg',)}"
        )
    problem = DarcyProblem(space, f=f, g=g, K=K, bc=bc, mode=mode)
    Q, P = problem.solve(method=method)
    if return_all:
        return {"u": (Q, P), "flux": Q, "pressure": P,
                "space": space, "problem": problem, "mesh": mesh}
    return Q, P


def solve_conduction(
    mesh=None,
    element: str = "RT0",
    f=None,
    g=None,
    sigma=1.0,
    bc=None,
    contact=None,
    mode: str = "cell",
    solver: str = "auto",
    region=None,
    return_all: bool = False,
    **kwargs,
):
    """一行求解定常电流场（DC conduction）问题。

    .. math::

       \\boldsymbol J=-\\sigma\\nabla\\varphi,\\qquad
       \\nabla\\cdot\\boldsymbol J=f .

    Parameters
    ----------
    mesh : Mesh | dict | None
        ``Mesh`` 对象或工厂参数字典；省略时用 ``region``。
    element : str
        只支持 ``"RT0"``（RT0×P0 混合元）。
    f : callable
        体注入电流密度 ``f(x)``。
    g : callable
        电极电势 ``g_D(x)``。
    sigma : float | (d,) | (d, d) | callable
        电导率（标量 / 对角 / 张量 / 变系数）。
    bc : callable, optional
        ``bc(mid, n_out) -> "D" | "N" | ("flux", gN)``。
    contact : callable, optional
        ``contact(mid) -> R_c``，内部面上的接触电阻率。
    mode : {"cell", "quad"}
        加权通量质量矩阵的组装方式。
    solver : str
        同 :func:`solve_darcy`。
    return_all : bool
        True 时返回 ``dict(current=..., potential=..., space=..., problem=..., mesh=...)``。

    Returns
    -------
    (J, phi) : tuple
        ``J`` 为 ``(n_cells, d+1)`` 逐单元外法向电流，``phi`` 为 ``(n_cells,)`` 电势。

    Examples
    --------
    .. code-block:: python

       from MosicaFE import Mesh, solve_conduction

       # 左右电极、上下绝缘、σ=2：J=(2,0)，R=1/2
       bc = lambda mid, n: "D" if (mid[0] < 1e-12 or mid[0] > 1 - 1e-12) else "N"
       J, phi = solve_conduction(Mesh.rectangle(nx=16, ny=16),
                                 g=lambda x: 1.0 - x[:, 0], sigma=2.0, bc=bc)
    """
    if mesh is None:
        if region is None:
            raise ValueError("请提供 mesh（Mesh 对象或工厂参数字典），或用 region=... 指定区域")
        mesh = region if isinstance(region, Mesh) else _coerce_mesh(region)
    else:
        mesh = _coerce_mesh(mesh)

    family, deg = resolve_element(element)
    if family != "rt0":
        raise ValueError(
            f"电流场问题需要 RT0×P0 混合元，element 应为 'RT0'；收到 {element!r}"
        )
    space = FESpace.create(mesh, family=family, degree=deg, **kwargs)

    method = "auto" if solver is None else str(solver).lower()
    if method not in _MIXED_SOLVERS + ("schur_cg",):
        raise ValueError(
            f"电流场问题是鞍点系统，solver 不能取 {solver!r}；"
            f"可用：{_MIXED_SOLVERS + ('schur_cg',)}"
        )
    problem = ConductionProblem(space, f=f, g=g, sigma=sigma, bc=bc,
                                contact=contact, mode=mode)
    J, phi = problem.solve(method=method)
    if return_all:
        return {"u": (J, phi), "current": J, "potential": phi,
                "space": space, "problem": problem, "mesh": mesh}
    return J, phi


def run(
    problem: str = "poisson",
    mesh=None,
    element: str = "P1",
    solver: str = "auto",
    degree: int = None,
    return_all: bool = False,
    **kwargs,
):
    """按名字求解任意已注册的 PDE。

    ``run("reaction_diffusion", mesh, element="VEM", f=src, g=zero, c=10.0)``
    """
    if mesh is None:
        raise ValueError("请提供 mesh")
    mesh = _coerce_mesh(mesh)
    family, deg = resolve_element(element, degree)
    space = FESpace.create(mesh, family=family, degree=deg)
    obj = create_problem(problem, space, **kwargs)
    sol = obj.solve(method=solver)
    if return_all:
        return {"u": sol, "space": space, "problem": obj, "mesh": mesh}
    return sol
