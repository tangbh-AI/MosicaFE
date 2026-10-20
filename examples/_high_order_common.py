"""Shared runner helpers for high-order Lagrange convergence scripts.

This module intentionally contains no PDE-specific weak form.  Each equation
gets its own small entry script under ``examples/``; new equations can reuse
:func:`run_problem` by supplying their own ``problem_factory``.
"""

# 启用延迟注解求值，避免运行时解析类型注解
from __future__ import annotations

# 标准库导入：CSV 写入、路径操作、模块搜索路径、计时、数值计算
import csv
import os
import sys
import time

import numpy as np


# 计算仓库根目录：当前文件位于 MosicaFE/examples/ 下，向上三级即为仓库根
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# 若仓库根不在 sys.path 中，则插入，以便导入 MosicaFE 包
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 从主包导入网格、误差计算、收敛率估计与收敛图绘制工具
from MosicaFE import Mesh, compute_errors, estimate_convergence_rate, plot_convergence  # noqa: E402


# 结果输出目录：examples/output/，若不存在则自动创建
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


# 制造解 u = sin(πx)·sin(πy)，即单位正方形上的精确解
def exact(x):
    """Manufactured exact solution on the unit square."""
    x = np.asarray(x, dtype=float)
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


# 制造解的精确梯度 ∇u = (π·cos(πx)·sin(πy), π·sin(πx)·cos(πy))
def exact_grad(x):
    """Exact gradient of the manufactured solution."""
    x = np.asarray(x, dtype=float)
    return np.array(
        [
            np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]),
            np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
        ]
    )


# 为某一层网格生成单位正方形三角形网格（nx×nx）
def make_mesh(nx):
    """Create the triangular unit-square mesh for one refinement level."""
    return Mesh.rectangle(nx=nx, ny=nx, element_type="triangle")


# 为所有方程脚本添加共用的命令行参数（阶数、网格尺寸、输出选项）
def add_common_arguments(parser, *, include_wave_number=False):
    """Add degree/mesh/output arguments shared by all equation scripts."""
    # --degrees：要运行的多项式阶数列表，默认 P1~P5
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Polynomial degrees P_k to run (default: 1 2 3 4 5).",
    )
    # --sizes：单位正方形每边的网格划分数列表，默认 4 8 16 32
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="Grid sizes nx=ny for the unit square (default: 4 8 16 32).",
    )
    # 若为 Helmholtz 脚本，额外添加 --wave-number（波数 k）
    if include_wave_number:
        parser.add_argument(
            "--wave-number",
            type=float,
            default=1.0,
            help="Helmholtz wave number k; keep k^2 < 2*pi^2 (~19.74).",
        )
    # --no-plots：跳过生成收敛图 PNG 文件
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not save convergence PNG files.",
    )


# 从解析后的参数中取出并校验阶数与网格尺寸：去重、排序、检查合法性
def validated_degrees_and_sizes(args):
    """Return sorted, validated ``(degrees, sizes)`` from parsed arguments."""
    # 阶数去重并排序，要求均 >= 1
    degrees = sorted(set(int(d) for d in args.degrees))
    if any(d < 1 for d in degrees):
        raise SystemExit("degrees must be >= 1")

    # 网格尺寸去重并排序，要求均 >= 1
    sizes = sorted(set(int(n) for n in args.sizes))
    if any(n < 1 for n in sizes):
        raise SystemExit("sizes must be >= 1")

    return degrees, sizes


# 核心运行函数：对指定方程遍历所有阶数与网格尺寸，求解、统计误差、绘图并写 CSV
def run_problem(kind, degrees, sizes, problem_factory, *, save_plots=True):
    """Run one equation for all requested degrees and mesh sizes.

    Parameters
    ----------
    kind : str
        Short name used in filenames and printed output, e.g. ``"poisson"``.
    degrees : iterable of int
        Polynomial degrees to run.
    sizes : iterable of int
        Grid sizes ``nx=ny`` to run.
    problem_factory : callable
        ``problem_factory(mesh, degree) -> BasePhysics``.
    save_plots : bool
        Whether to write ``convergence_{kind}_P{degree}.png`` files.
    """
    # 收集所有结果行，最终写入 CSV
    rows = []

    # 打印标题分隔线
    print(f"\n{'=' * 78}")
    print(f"{kind.upper()}")
    print("=" * 78)

    # 外层循环：遍历每个多项式阶数
    for degree in degrees:
        h_values = []
        errors = {"L2": [], "H1": [], "Linf": []}

        print(f"\n--- {kind} P{degree} ---")
        # 内层循环：遍历每个网格尺寸（即每一层网格加密）
        for nx in sizes:
            # 构造当前层网格，并调用工厂创建对应问题对象
            mesh = make_mesh(nx)
            problem = problem_factory(mesh, degree)
            space = problem.space

            # 计时：用直接法求解线性系统
            t0 = time.perf_counter()
            u = problem.solve(method="direct")
            solve_time = time.perf_counter() - t0

            # 计算 L2、H1、L∞ 误差，并获取当前网格尺寸 h
            norms = compute_errors(problem, u, exact, exact_grad=exact_grad)
            h = mesh.get_mesh_size()

            # 记录本层数据，供后续收敛率估计与绘图使用
            h_values.append(h)
            errors["L2"].append(norms["L2"])
            errors["H1"].append(norms["H1"])
            errors["Linf"].append(norms["Linf"])

            # 追加一行 CSV 结果
            rows.append(
                {
                    "problem": kind,
                    "degree": degree,
                    "nx": nx,
                    "h": h,
                    "n_dofs": int(space.n_dofs),
                    "L2": norms["L2"],
                    "H1": norms["H1"],
                    "Linf": norms["Linf"],
                    "solve_time_s": solve_time,
                }
            )

            # 打印本层摘要信息
            print(
                f"  n={nx:2d}  h={h:.4f}  dofs={space.n_dofs:5d}  "
                f"L2={norms['L2']:.4e}  H1={norms['H1']:.4e}  "
                f"Linf={norms['Linf']:.4e}  t={solve_time:.3f}s"
            )

        # 若至少有 2 层网格，估计收敛率（最多取最后 3 层）
        if len(sizes) >= 2:
            use_last = min(3, len(sizes))
            print(
                "  estimated rates (last 3 levels if available): "
                f"L2={estimate_convergence_rate(h_values, errors['L2'], use_last=use_last):.3f}, "
                f"H1={estimate_convergence_rate(h_values, errors['H1'], use_last=use_last):.3f}, "
                f"Linf={estimate_convergence_rate(h_values, errors['Linf'], use_last=use_last):.3f}"
            )

        # 若需要绘图且至少有 2 层网格，则绘制本阶数的收敛图
        if save_plots and len(sizes) >= 2:
            filename = os.path.join(OUT_DIR, f"convergence_{kind}_P{degree}.png")
            plot_convergence(
                h_values,
                errors,
                filename=filename,
                reference_rates=(1, 2, 3, degree, degree + 1),
            )

    # 将所有结果写入 CSV 文件
    csv_path = os.path.join(OUT_DIR, f"high_order_{kind}_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "problem",
            "degree",
            "nx",
            "h",
            "n_dofs",
            "L2",
            "H1",
            "Linf",
            "solve_time_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 打印 CSV 保存路径并返回结果行
    print(f"\nSaved results to {csv_path}")
    return rows