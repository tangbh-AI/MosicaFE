# -*- coding: utf-8 -*-
"""极简 pytest 兼容层（只为在没有 pytest 的环境里跑自带测试）。

只实现本库测试用到的四个东西：

* ``pytest.approx``
* ``pytest.raises``
* ``pytest.mark.parametrize``
* ``pytest.fixture``（空实现，保证 import 不失败）

装好 pytest 之后完全不需要它——测试文件本身就是标准 pytest 文件。
"""

from __future__ import annotations

import numpy as np

__all__ = ["approx", "raises", "mark", "fixture", "skip"]


class _Approx:
    def __init__(self, expected, rel: float = 1e-6, abs: float = None):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _abs_tol(self):
        if self.abs is not None:
            return float(self.abs)
        expected = np.asarray(self.expected, dtype=float)
        return float(self.rel * np.maximum(np.abs(expected), 1e-300))

    def __eq__(self, other):
        other = np.asarray(other, dtype=float)
        expected = np.asarray(self.expected, dtype=float)
        tol = np.maximum(self._abs_tol(), 0.0)
        self.ok = bool(np.all(np.abs(other - expected) <= tol))
        self.actual = other
        return self.ok

    def __ne__(self, other):
        return not self.__eq__(other)

    def __req__(self, other):  # pragma: no cover - 反射比较
        return self.__eq__(other)

    def __repr__(self):
        return f"approx({self.expected!r}, rel={self.rel}, abs={self.abs})"


def approx(expected, rel: float = 1e-6, abs: float = None):
    """比较包装：``value == pytest.approx(target, rel=..., abs=...)``。"""
    return _Approx(expected, rel=rel, abs=abs)


class _Raises:
    def __init__(self, expected):
        self.expected = expected
        self.value = None
        self.type = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"期望抛出 {self.expected.__name__}，但没有异常")
        if issubclass(exc_type, self.expected):
            self.type = exc_type
            self.value = exc
            return True
        return False


def raises(expected):
    """上下文管理器：``with pytest.raises(ValueError): ...``。"""
    return _Raises(expected)


class _Mark:
    def parametrize(self, argnames, argvalues, **kwargs):
        names = [n.strip() for n in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

        def decorator(func):
            cases = []
            for values in argvalues:
                if len(names) == 1:
                    values = (values,)
                cases.append(dict(zip(names, values)))
            func._parametrize = (names, cases)
            return func

        return decorator

    def skip(self, *args, **kwargs):
        def decorator(func):
            func._skip = True
            return func

        return decorator

    def __getattr__(self, item):  # 其它 mark 一律忽略
        def decorator(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]

            def inner(func):
                return func

            return inner

        return decorator


mark = _Mark()


def fixture(func=None, **kwargs):
    """空 fixture：直接返回被装饰的函数。"""
    if func is None:
        return lambda f: f
    return func


def skip(reason="skipped"):
    raise AssertionError(f"skip: {reason}")
