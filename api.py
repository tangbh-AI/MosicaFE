"""MosicaFE 的一行式高层接口。

主文件真的只需要几行：

.. code-block:: python

   import numpy as np
   from MosicaFE import Mesh, solve_poisson

   f = lambda x: 2 * np.pi ** 2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
   u = solve_poisson(Mesh.rectangle(nx=32, ny=32), element="P2", f=f)

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
from .physics.mixed_poisson import MixedPoissonProblem
from .physics.poisson import PoissonProblem
from .physics.registry import create_problem
from .spaces.factory import FESpace

__all__ = [
    "ELEMENT_ALIASES",
    "resolve_element",
    "make_space",
    "solve_poisson",
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
