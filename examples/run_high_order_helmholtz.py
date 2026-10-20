"""Run arbitrary-order Lagrange FEM for the Helmholtz equation.

Problem: ``-Delta u - k^2 u = f`` on the unit square, ``u = 0`` on the
boundary.  The manufactured solution is ``u = sin(pi*x)*sin(pi*y)``.

Run from the repository's parent directory:

    python MosicaFE/examples/run_high_order_helmholtz.py

Examples
--------
    python MosicaFE/examples/run_high_order_helmholtz.py --degrees 1 2 3 4 5
    python MosicaFE/examples/run_high_order_helmholtz.py --degrees 6 --sizes 8 16 24
    python MosicaFE/examples/run_high_order_helmholtz.py --wave-number 2.0
"""

# 启用延迟注解求值，避免运行时解析类型注解
from __future__ import annotations

# 标准库导入：命令行参数解析、路径操作、计时、数值计算
import argparse
import os
import time

import numpy as np

# 从公共工具模块导入共享常量与辅助函数
from _high_order_common import (
    OUT_DIR,               # 结果输出目录
    add_common_arguments,  # 统一的命令行参数（含 --wave-number 选项）
    exact,                 # 精确（制造）解 u
    run_problem,           # 运行一组阶数/网格的收敛性测试
    validated_degrees_and_sizes,  # 校验并返回阶数与网格尺寸
)
# 从主包导入有限元空间与 Helmholtz 问题类
from MosicaFE import FESpace, HelmholtzProblem


# 源项 f = -Δu - k²u，由制造解 u 与给定波数 k 推导得到
def helmholtz_source(x, wave_number):
    """Source term ``f = -Delta u - k^2 u``."""
    return (2.0 * np.pi**2 - wave_number**2) * exact(x)


# 为固定波数 k 返回一个"问题构造工厂"函数
def make_problem_factory(wave_number):
    """Return a problem factory for a fixed Helmholtz wave number."""

    # 内部函数：为给定网格和阶数构造 Helmholtz 问题对象
    def make_problem(mesh, degree):
        # 构造 Lagrange 有限元空间
        space = FESpace.lagrange(mesh, degree=degree)
        # 组装 Helmholtz 问题：源项 f、齐次 Dirichlet 边界条件 g=0、波数 k
        return HelmholtzProblem(
            space,
            f=lambda x: helmholtz_source(x, wave_number),
            g=lambda x: 0.0,
            k=wave_number,
        )

    # 返回闭包，使波数在 run_problem 调用中保持固定
    return make_problem


# 主入口：解析参数、运行收敛性测试、输出计时与结果目录
def main(argv=None):
    # 创建参数解析器，使用模块 docstring 作为帮助文本
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 添加公共命令行参数，并额外启用 --wave-number 选项
    add_common_arguments(parser, include_wave_number=True)
    # 解析参数（argv 允许测试时传入自定义参数）
    args = parser.parse_args(argv)

    # 校验并获取阶数列表与网格尺寸列表
    degrees, sizes = validated_degrees_and_sizes(args)
    # 记录开始时间，用于统计总耗时
    started = time.perf_counter()
    # 运行 Helmholtz 问题：对每组 (degree, size) 求解并计算误差/收敛率
    run_problem(
        "helmholtz",
        degrees,
        sizes,
        make_problem_factory(args.wave_number),
        save_plots=not args.no_plots,
    )

    # 打印总耗时与输出目录的绝对路径
    print(f"\nTotal elapsed: {time.perf_counter() - started:.3f}s")
    print("Output directory:", os.path.abspath(OUT_DIR))


# 作为脚本直接运行时执行 main
if __name__ == "__main__":
    main()