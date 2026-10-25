"""同一问题的"分步"写法：显式体现"选元 / 选求解器 / 选可视化"，

所有空间族都换成 3 行即可，符合 guidebook 的调用习惯。
"""

# 模块职责：分步比较三种离散方法
# 先定义同一个已知解及源项，再分别建立 FEM、VEM 和 RT0 问题，比较误差并输出图像。
# 可从 main 开始阅读：共享网格 → 调用三种示例 → 绘图；再回看各个 demo 中空间与问题的创建过程。


import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MosicaFE import (  # noqa: E402
    FESpace,
    Mesh,
    MixedPoissonProblem,
    PoissonProblem,
    plot_solution,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


def exact(x):
    # 提供已知精确解，用于制造源项并评估数值误差；求解过程中无需把精确解作为未知量输入。
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def source(x):
    # 给出制造解对应的源项 f=-Δu，供单元载荷积分调用。
    return 2.0 * np.pi ** 2 * exact(x)


def exact_grad(x):
    # 返回精确解各坐标方向的偏导数，供 H1 半范数误差评估使用。
    return np.array(
        [
            np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]),
            np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
        ]
    )


def zero(x):
    # 统一提供零函数，作为齐次边界值或零源项传入求解接口。
    return 0.0


# 工作块：求解同一算例的 FEM 版本
# 选二次 Lagrange 空间，绑定源项和零边界，求解后与精确解及梯度比较；返回解和空间供绘图使用。
def demo_lagrange(mesh):
    """经典有限元：P2 三角形。"""
    V = FESpace.lagrange(mesh, degree=2)
    problem = PoissonProblem(V, f=source, g=zero)
    u = problem.solve()
    print("FEM-P2 :", problem.compute_errors(u, exact, exact_grad))
    return u, V


# 工作块：求解同一算例的 VEM 版本
# 用一次虚拟元空间替换 FEM 空间，其余源项与边界数据保持同一算例的定义；误差评估仍由问题对象统一提供。
def demo_vem(mesh):
    """虚拟元：同一段代码，只换一个空间。"""
    V = FESpace.vem(mesh, degree=1)
    problem = PoissonProblem(V, f=source, g=zero)
    u = problem.solve()
    print("VEM-P1 :", problem.compute_errors(u, exact, exact_grad))
    return u, V


# 工作块：求解同一算例的混合版本
# 选 RT0 空间和 MixedPoissonProblem，同时求通量 Q 与压力 P；精确通量使用精确解的负梯度，不能直接使用正梯度比较。
def demo_rt0(mesh):
    """RT0×P0 混合元：换空间 + 换问题类，解是 (通量, 压力)。"""
    V = FESpace.rt0(mesh)
    problem = MixedPoissonProblem(V, f=source)
    Q, P = problem.solve()
    print("RT0-P0 :", problem.compute_errors((Q, P), exact, lambda x: -exact_grad(x)))
    return P, None


# 工作块：组织三种空间的对照计算
# 在同一三角形网格上运行 FEM、VEM 和 RT0，再分别绘制结果；每次绘图使用对应的空间对象解释解向量，不能混用空间。
def main():
    # 在同一网格上分别执行 FEM、VEM 与 RT0 示例，再使用各自空间解释和显示结果。
    mesh = Mesh.rectangle(nx=32, ny=32)
    u_fem, V = demo_lagrange(mesh)
    u_vem, V_vem = demo_vem(mesh)
    u_rt0, _ = demo_rt0(mesh)

    plot_solution(u_fem, mesh, V, title="FEM P2",
                  filename=os.path.join(OUT, "demo_fem_p2.png"))
    plot_solution(u_vem, mesh, V_vem, title="VEM P1",
                  filename=os.path.join(OUT, "demo_vem_p1.png"))
    print("RT0 pressure range:", float(np.min(u_rt0)), float(np.max(u_rt0)))


if __name__ == "__main__":
    main()
