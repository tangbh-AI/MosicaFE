"""PDE 注册表：让"选择方程"和"选择元"一样简单。

.. code-block:: python

   from MosicaFE.physics.registry import available_problems, create_problem

   print(available_problems())          # ['poisson', 'reaction_diffusion', ...]
   problem = create_problem("poisson", V, f=source, g=zero)
"""

from __future__ import annotations

__all__ = [
    "register_problem",
    "get_problem_class",
    "available_problems",
    "create_problem",
    "clear_registry",
]

_REGISTRY = {}


def register_problem(name: str, cls=None, *, override: bool = False):
    """注册一个 PDE 实现。既可当装饰器，也可直接调用。"""

    def _register(target):
        key = str(name).lower()
        if key in _REGISTRY and not override:
            raise ValueError(
                f"PDE 名字 {key!r} 已注册为 {_REGISTRY[key].__name__}；"
                "如需覆盖请传 override=True"
            )
        _REGISTRY[key] = target
        target.problem_name = key
        return target

    return _register(cls) if cls is not None else _register


def get_problem_class(name: str):
    """按名字取回 PDE 类。"""
    key = str(name).lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"未注册的 PDE {name!r}；当前可用：{available_problems()}"
        )
    return _REGISTRY[key]


def available_problems():
    """已注册的 PDE 名字（排序）。"""
    return sorted(_REGISTRY)


def create_problem(name: str, space, **kwargs):
    """按名字实例化 PDE。"""
    return get_problem_class(name)(space, **kwargs)


def clear_registry():
    """清空注册表（测试用）。"""
    _REGISTRY.clear()
