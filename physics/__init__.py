"""PDE 弱形式与组装。

内置问题：:class:`PoissonProblem`、:class:`MixedPoissonProblem`、
:class:`ReactionDiffusionProblem`、:class:`HelmholtzProblem`。
用 :mod:`MosicaFE.physics.registry` 可以注册自己的新方程。
"""

# 模块职责：公共接口的集中导出
# 把本层常用类和函数导入统一命名空间；用户可从包入口导入接口，无需逐个寻找内部实现文件。
# __all__ 指定公开名称；计算逻辑保存在被导入的模块中，这个文件负责组织调用入口。


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
