"""PDE 注册表：让"选择方程"和"选择元"一样简单。

.. code-block:: python

   from MosicaFE.physics.registry import available_problems, create_problem

   print(available_problems())          # ['poisson', 'reaction_diffusion', ...]
   problem = create_problem("poisson", V, f=source, g=zero)
"""

# 模块职责：方程名称到实现类的注册表
# 装饰器将短名称与问题类关联，调用方可以按名称创建问题，而无需知道类的导入位置。
# 注册、查找和创建分开处理；未知名称和重复注册的限制在这里统一检查。


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
        # 把装饰器接收的问题类放入注册表，并返回原类，使类定义和按名称创建接口同时可用。
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
