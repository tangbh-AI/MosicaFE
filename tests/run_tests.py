# -*- coding: utf-8 -*-
"""不依赖 pytest 的测试运行器。

它把 ``minipytest`` 注册成 ``pytest`` 模块，然后收集 ``test_*.py`` 里的
``test_*`` 函数逐个执行（支持 parametrize）。

用法::

    & "D:\\anaconda\\envs\\fealpy\\python.exe" MosicaFE\\tests\\run_tests.py

装了 pytest 的话，直接用 ``python -m pytest MosicaFE/tests -q`` 即可，
两者跑的是同一套测试。
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

# 用 minipytest 冒充 pytest（本环境没装 pytest）
try:
    import pytest  # noqa: F401

    HAVE_PYTEST = True
except ImportError:
    import minipytest

    sys.modules["pytest"] = minipytest
    HAVE_PYTEST = False


def _iter_cases(func):
    spec = getattr(func, "_parametrize", None)
    if spec is None:
        yield None, {}
        return
    names, cases = spec
    for case in cases:
        label = ", ".join(f"{k}={v!r}" for k, v in case.items())
        yield label, case


def main():
    modules = [
        name[:-3]
        for name in sorted(os.listdir(_HERE))
        if name.startswith("test_") and name.endswith(".py")
    ]
    passed = 0
    failed = []
    print(f"MosicaFE 测试运行器（pytest 可用：{HAVE_PYTEST}）")
    print("=" * 72)
    for mod_name in modules:
        module = importlib.import_module(mod_name)
        for attr in sorted(dir(module)):
            if not attr.startswith("test_"):
                continue
            func = getattr(module, attr)
            if not callable(func):
                continue
            for label, kwargs in _iter_cases(func):
                title = f"{mod_name}.{attr}" + (f"[{label}]" if label else "")
                try:
                    func(**kwargs)
                except Exception as exc:  # noqa: BLE001
                    failed.append((title, exc, traceback.format_exc()))
                    print(f"  FAIL  {title}\n        {type(exc).__name__}: {exc}")
                else:
                    passed += 1
                    print(f"  ok    {title}")
    print("=" * 72)
    print(f"通过 {passed}，失败 {len(failed)}")
    if failed:
        print("\n失败详情：")
        for title, _exc, tb in failed:
            print("-" * 72)
            print(title)
            print(tb)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
