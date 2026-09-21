"""MosicaFE —— 一个统一调用 FEM / VEM / RT0 求解偏微分方程的 Python 库。

把原来三套独立程序整合成**同一套接口**：选"元"、选"求解器"、选"可视化"，
主文件只剩几行。

快速开始
--------
.. code-block:: python

   import numpy as np
   from MosicaFE import Mesh, FESpace, PoissonProblem, plot_solution

   def exact(x):
       return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

   def source(x):
       return 2 * np.pi ** 2 * exact(x)

   mesh = Mesh.rectangle(nx=32, ny=32)          # 选网格
   V = FESpace.lagrange(mesh, degree=2)         # 选元（也可以 FESpace.vem / FESpace.rt0）
   u = PoissonProblem(V, f=source, g=lambda x: 0.0).solve()   # 选求解器
   plot_solution(u, mesh, V, filename="solution.png")          # 选可视化

或者更短的"一行流"：

.. code-block:: python

   from MosicaFE import Mesh, solve_poisson
   u = solve_poisson(Mesh.rectangle(nx=32, ny=32), element="VEM", f=source)

模块结构（与 guidebook 第 1.4 节一致）：

``core``        网格、面拓扑、求积规则、几何工具
``spaces``      离散空间（Lagrange / VEM / RT0）
``physics``     PDE 弱形式与组装
``solvers``     线性求解器
``postprocess`` 误差分析、收敛性研究、可视化

FEM、VEM、RT0 三类元都是库自身的实现：RT0×P0 的单元基函数、精确质量矩阵、
面拓扑与鞍点组装分别在 :mod:`MosicaFE.spaces.rt0`、
:mod:`MosicaFE.core.facets`、:mod:`MosicaFE.physics.mixed_poisson` 中，
不存在需要额外接入的外部数值后端。
"""

from __future__ import annotations

from .api import ELEMENT_ALIASES, make_space, resolve_element, run, solve_poisson
from .core.facets import FacetTopology, build_facet_topology
from .core.mesh import Mesh
from .core.quadrature import (
    QuadratureRule,
    cell_quadrature,
    get_quadrature,
    simplex_quadrature,
)
from .physics.base import BasePhysics
from .physics.mixed_poisson import MixedPoissonProblem
from .physics.poisson import PoissonProblem
from .physics.reaction_diffusion import ReactionDiffusionProblem
from .physics.registry import (
    available_problems,
    create_problem,
    get_problem_class,
    register_problem,
)
from .postprocess.convergence import convergence_study, estimate_convergence_rate
from .postprocess.error import compute_error, compute_errors, l2_error, linf_error
from .postprocess.visualization import (
    compare_solutions,
    plot_convergence,
    plot_error,
    plot_flux_2d,
    plot_mesh,
    plot_pressure_2d,
    plot_solution,
)
from .solvers.linear import ConvergenceError, Solver
from .spaces.base import BaseFESpace
from .spaces.factory import FESpace
from .spaces.lagrange import LagrangeSpace
from .spaces.rt0 import RT0Space, rt0_flux_basis, rt0_mass_matrices
from .spaces.vem import VEMSpace

__version__ = "1.0.0"
__author__ = "MosicaFE contributors"

__all__ = [
    "__version__",
    # core
    "Mesh",
    "FacetTopology",
    "build_facet_topology",
    "QuadratureRule",
    "get_quadrature",
    "cell_quadrature",
    "simplex_quadrature",
    # spaces
    "FESpace",
    "BaseFESpace",
    "LagrangeSpace",
    "VEMSpace",
    "RT0Space",
    "rt0_flux_basis",
    "rt0_mass_matrices",
    # physics
    "BasePhysics",
    "PoissonProblem",
    "MixedPoissonProblem",
    "ReactionDiffusionProblem",
    "register_problem",
    "get_problem_class",
    "available_problems",
    "create_problem",
    # solvers
    "Solver",
    "ConvergenceError",
    # postprocess
    "compute_error",
    "compute_errors",
    "l2_error",
    "linf_error",
    "convergence_study",
    "estimate_convergence_rate",
    "plot_solution",
    "plot_mesh",
    "plot_error",
    "plot_convergence",
    "plot_pressure_2d",
    "plot_flux_2d",
    "compare_solutions",
    # api
    "solve_poisson",
    "run",
    "make_space",
    "resolve_element",
    "ELEMENT_ALIASES",
]
