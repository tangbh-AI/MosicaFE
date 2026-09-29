"""PDE 弱形式与组装。

内置问题：:class:`PoissonProblem`、:class:`MixedPoissonProblem`、
:class:`ReactionDiffusionProblem`、:class:`HelmholtzProblem`。
用 :mod:`MosicaFE.physics.registry` 可以注册自己的新方程。
"""

from .base import BasePhysics
from .helmholtz import HelmholtzProblem
from .mixed_poisson import MixedPoissonProblem
from .poisson import PoissonProblem
from .reaction_diffusion import ReactionDiffusionProblem
from .registry import (
    available_problems,
    create_problem,
    get_problem_class,
    register_problem,
)

__all__ = [
    "BasePhysics",
    "PoissonProblem",
    "MixedPoissonProblem",
    "ReactionDiffusionProblem",
    "HelmholtzProblem",
    "register_problem",
    "get_problem_class",
    "available_problems",
    "create_problem",
]
