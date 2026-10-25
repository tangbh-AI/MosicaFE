# -*- coding: utf-8 -*-
"""极简 pytest 兼容层（只为在没有 pytest 的环境里跑自带测试）。

只实现本库测试用到的四个东西：

* ``pytest.approx``
* ``pytest.raises``
* ``pytest.mark.parametrize``
* ``pytest.fixture``（空实现，保证 import 不失败）

装好 pytest 之后完全不需要它——测试文件本身就是标准 pytest 文件。
"""

# 模块职责：测试框架的最小兼容实现
# 为本项目测试提供近似比较、异常断言和参数化等小范围 pytest 接口。
# 它只覆盖本项目需要的能力；参数化信息由运行器读取，浮点比较和预期异常由辅助对象处理。


from __future__ import annotations

import numpy as np

__all__ = ["approx", "raises", "mark", "fixture", "skip"]


class _Approx:
    # 包装预期数值与浮点容差，使测试可以用等号表达近似相等。
    def __init__(self, expected, rel: float = 1e-6, abs: float = None):
        # 保存预期数值和绝对/相对容限，实际比较留到相等运算时执行。
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _abs_tol(self):
        # 优先使用显式绝对容限，否则根据预期数值和相对容限计算允许的差值。
        if self.abs is not None:
            return float(self.abs)
        expected = np.asarray(self.expected, dtype=float)
        return float(self.rel * np.maximum(np.abs(expected), 1e-300))

    def __eq__(self, other):
        # 比较实际值与预期值的差是否落在容限内，并记录实际值便于定位失败原因。
        other = np.asarray(other, dtype=float)
        expected = np.asarray(self.expected, dtype=float)
        tol = np.maximum(self._abs_tol(), 0.0)
        self.ok = bool(np.all(np.abs(other - expected) <= tol))
        self.actual = other
        return self.ok

    def __ne__(self, other):
        # 复用近似相等判断并取反，保证两种比较使用相同的浮点容限。
        return not self.__eq__(other)

    def __req__(self, other):  # pragma: no cover - 反射比较
        # 将反向比较转交相等判断，保持比较时的容差规则一致。
        return self.__eq__(other)

    def __repr__(self):
        # 返回对象的可读说明，方便交互查看；此方法只组织展示信息。
        return f"approx({self.expected!r}, rel={self.rel}, abs={self.abs})"


def approx(expected, rel: float = 1e-6, abs: float = None):
    """比较包装：``value == pytest.approx(target, rel=..., abs=...)``。"""
    return _Approx(expected, rel=rel, abs=abs)


class _Raises:
    # 记录预期异常类型，供 with 块结束时核对是否确实抛出相应异常。
    def __init__(self, expected):
        # 保存预期异常类型，并将实际异常及其类型留待 with 块退出时填写。
        self.expected = expected
        self.value = None
        self.type = None

    def __enter__(self):
        # 进入异常检查的 with 块，返回保存异常信息的对象供测试读取。
        return self

    def __exit__(self, exc_type, exc, tb):
        # 离开 with 块时核对异常类型；未抛异常则失败，类型匹配时记录异常并抑制继续传播。
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
    # 提供测试标记的最小兼容接口，将参数化信息交给本项目的测试运行器。
    def parametrize(self, argnames, argvalues, **kwargs):
        # 把参数名称和参数取值组合保存为函数元数据；测试运行器随后按这些组合逐次调用。
        names = [n.strip() for n in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

        def decorator(func):
            # 给函数附加本兼容层需要的测试元数据，或在无需处理的标记下直接保留原函数。
            cases = []
            for values in argvalues:
                if len(names) == 1:
                    values = (values,)
                cases.append(dict(zip(names, values)))
            func._parametrize = (names, cases)
            return func

        return decorator

    def skip(self, *args, **kwargs):
        # 在兼容层中记录跳过标记，或通过明确的 skip 异常信息结束当前调用；由下面的实际入口区分。
        def decorator(func):
            # 给函数附加本兼容层需要的测试元数据，或在无需处理的标记下直接保留原函数。
            func._skip = True
            return func

        return decorator

    def __getattr__(self, item):  # 其它 mark 一律忽略
        # 兼容当前测试不需要实际处理的其他 mark，返回保持原函数的装饰器。
        def decorator(*args, **kwargs):
            # 给函数附加本兼容层需要的测试元数据，或在无需处理的标记下直接保留原函数。
            if args and callable(args[0]):
                return args[0]

            def inner(func):
                # 返回原测试函数，使本兼容层忽略的标记不改变函数行为。
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
    # 在兼容层中记录跳过标记，或通过明确的 skip 异常信息结束当前调用；由下面的实际入口区分。
    raise AssertionError(f"skip: {reason}")
