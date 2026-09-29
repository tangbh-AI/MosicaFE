# MosicaFE

**一个统一调用 FEM / VEM / RT0 求解偏微分方程的 Python 库。**

MosicaFE 把三套原本互相独立的程序整合成同一套接口：**选网格 → 选元 → 选求解器 → 选可视化**，
主文件通常只有几行代码。

```python
import numpy as np
from MosicaFE import Mesh, solve_poisson

source = lambda x: 2 * np.pi**2 * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
u = solve_poisson(Mesh.rectangle(nx=32, ny=32), element="P2", f=source)
```

换一门方程同样只改一行（内置方程：`poisson` / `reaction_diffusion` /
`helmholtz` / `mixed_poisson`）：

```python
from MosicaFE import Mesh, solve_helmholtz

source = lambda x: (2 * np.pi**2 - 1.0) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
u = solve_helmholtz(Mesh.rectangle(nx=32, ny=32), element="P2", f=source, k=1.0)
```

---

## 目录

* [安装](#安装)
* [快速开始](#快速开始)
* [可选的"元"](#可选的元)
* [完整的调用方式](#完整的调用方式)
* [RT0 混合元](#rt0-混合元)
* [亥姆霍兹方程](#亥姆霍兹方程)
* [误差与收敛性](#误差与收敛性)
* [可视化](#可视化)
* [扩展到其它 PDE](#扩展到其它-pde)
* [包结构](#包结构)
* [验证结果](#验证结果)
* [已知限制](#已知限制)

---

## 安装

只需要 `numpy`；`scipy`（稀疏矩阵与 Krylov 求解器）与 `matplotlib`（绘图）是**可选**的，
没装会自动退化成纯 numpy 实现。

```bash
cd MosicaFE
python -m pip install -e .
```

不安装也可以，只要把**父目录**加进 `PYTHONPATH`：

```powershell
# PowerShell（本机推荐用 conda 环境 fealpy）
cd D:\mosica
$env:PYTHONPATH = "D:\mosica"
& "D:\anaconda\envs\fealpy\python.exe" MosicaFE\examples\poisson_demo.py
```

---

## 快速开始

三种写法，从最短到最清楚：

**① 一行流**

```python
from MosicaFE import Mesh, solve_poisson
u = solve_poisson(Mesh.rectangle(nx=32, ny=32), element="VEM", f=source)
```

**② 完整风格（推荐）**

```python
import numpy as np
from MosicaFE import Mesh, FESpace, PoissonProblem, plot_solution

def exact(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

def source(x):
    return 2 * np.pi ** 2 * exact(x)

mesh    = Mesh.rectangle(nx=32, ny=32)              # 选网格
V       = FESpace.lagrange(mesh, degree=2)          # 选元
problem = PoissonProblem(V, f=source, g=lambda x: 0.0)
u       = problem.solve()                            # 选求解器（默认自动）
plot_solution(u, mesh, V, filename="solution.png")  # 选可视化
```

**③ 换一种元，只改一行**

```python
V = FESpace.vem(mesh)        # 虚拟元（任意多边形/多面体）
V = FESpace.rt0(mesh)        # RT0×P0 混合元（此时解是 (Q, P)）
V = FESpace.lagrange(mesh, degree=1)
```

**换一门方程，同样只改一行**（内置方程：`poisson` / `helmholtz` /
`reaction_diffusion` / `mixed_poisson`）

```python
from MosicaFE import HelmholtzProblem, solve_helmholtz

u = solve_helmholtz(mesh, element="P2", f=source, k=1.0)         # 一行流
u = HelmholtzProblem(V, f=source, g=lambda x: 0.0, k=1.0).solve()  # 分步写法
```

---

## 可选的"元"

| `element=` | 空间 | 支持网格 | 收敛阶（L2 / H1） |
| --- | --- | --- | --- |
| `"P1"` | 线性 Lagrange | 三角形 / 四面体 | 2 / 1 |
| `"P2"` | 二次 Lagrange | 三角形 / 四面体 | 3 / 2 |
| `"Q1"` | 双线性 Lagrange | 四边形 / 六面体 | 2 / 1 |
| `"Q2"` | 双二次 Lagrange | 四边形 / 六面体 | 3 / 2 |
| `"VEM"` | 一次虚拟元 | **任意**多边形 / 多面体 | 2 / 1 |
| `"RT0"` | RT0×P0 混合元 | 三角形 / 四面体 | 1 / 1（压力 / 通量） |

网格由 `MosicaFE.Mesh` 生成：

| 工厂 | 区域 | 单元 |
| --- | --- | --- |
| `Mesh.interval(a, b, nx)` | 一维区间 | 线段 |
| `Mesh.rectangle(..., element_type="triangle")` | 矩形 | 三角形 |
| `Mesh.rectangle(..., element_type="quad")` | 矩形 | 四边形 |
| `Mesh.box(..., element_type="tet")` | 长方体 | 四面体（Kuhn 剖分，或 `pattern="center"`） |
| `Mesh.box(..., element_type="hex")` | 长方体 | 六面体 |
| `Mesh.pentagon(nx, ny)` | 正方形 | 五边形（棋盘格） |
| `Mesh.voronoi_polygon(n_points, seed)` | 正方形 | Voronoi 多边形 |
| `Mesh.from_arrays(nodes, elements)` | 自定义 | 自动推断 |

---

## 完整的调用方式

### 求解器

```python
from MosicaFE import Solver

Solver.direct(A, b)                       # 稀疏 LU，精确
Solver.conjugate_gradient(A, b, tol=1e-10)  # SPD 系统
Solver.gmres(A, b, tol=1e-10, restart=30)   # 非对称
Solver.bicgstab(A, b)                     # 折中
Solver.minres(A, b)                       # 对称不定（鞍点）
Solver.solve(A, b, method="auto")         # 按规模和对称性自动选择
```

在问题层直接用名字即可：

```python
u = problem.solve(method="cg")     # "auto" / "direct" / "cg" / "gmres" / "bicgstab" / "minres"
```

### 边界条件

`PoissonProblem(V, f=..., g=...)` 中的 `g(x)` 是 Dirichlet 数据（默认 0）。
库内部按"消行消列 + 右端项修正"实现，非齐次边界也能通过 patch test。

### 方程

```python
PoissonProblem(V, f=..., g=..., kappa=1.0)                # -∇·(κ∇u) = f
ReactionDiffusionProblem(V, f=..., reaction_coeff=c)      # -Δu + c u = f
HelmholtzProblem(V, f=..., g=..., k=1.0, kappa=1.0)       # -∇·(κ∇u) - k²u = f
MixedPoissonProblem(V_rt0, f=...)                         # RT0 混合形式
```

也可以按名字建方程：`create_problem("helmholtz", V, f=..., k=2.0)`、
`run("helmholtz", mesh, element="P2", f=..., k=2.0)`、
`available_problems()`。

---

## RT0 混合元

混合格式同时解出通量与压力：

```python
from MosicaFE import FESpace, Mesh, MixedPoissonProblem, plot_flux_2d, plot_pressure_2d

mesh = Mesh.rectangle(nx=32, ny=32)        # 三角形网格
V = FESpace.rt0(mesh)                       # RT0×P0，和其他"元"地位相同
problem = MixedPoissonProblem(V, f=source)
Q, P = problem.solve()                      # Q: (n_cells, 3) 面通量, P: (n_cells,) 常数压力

print(problem.error_norms(P, Q, exact, lambda x: -exact_grad(x)))
print("质量守恒误差：", problem.check_conservation())
plot_pressure_2d(mesh, P, filename="pressure.png")
plot_flux_2d(space=V, Q=Q, filename="flux.png")
```

三维同理（`Mesh.box(..., element_type="tet")`）。

RT0 是**库自身的实现**（不依赖任何外部数值后端），由三部分组成：

| 位置 | 内容 |
| --- | --- |
| `core/facets.py` | 面拓扑：全局面编号、共享面、外法向符号（`H(div)` 协调性的基础） |
| `spaces/rt0.py` | 显式基函数 `φ_i=(x-a_i)/(d|K|)`、**精确闭式**质量矩阵、自由度布局与重构 |
| `physics/mixed_poisson.py` | 鞍点系统 `[[M, -Bᵀ], [B, 0]]` 的拼装（`M`、`B` 来自空间层） |

空间层给的是"块"，写别的混合问题（Darcy、Stokes 的同阶配对……）时可以直接复用：

```python
M = V.flux_mass_matrix()        # (Φ,Φ)  ∫_K φ_i·φ_j，精确闭式，无需求积
B = V.divergence_matrix()       # (P,Φ)  ∫_K ∇·φ_i = σ（±1，精确）
f = V.pressure_load_vector(src) # (P,)   ∫_K f
```

也可用 `V.interpolate_pressure(p)` / `V.interpolate_flux(q)`、
`V.evaluate_flux(cell, Q, points)`、`V.locate_cell(x)` 等原生工具。

---

## 亥姆霍兹方程

时谐波的模型问题（常数波数 $k$、常数扩散系数 $\kappa$）：

$$\nabla\cdot(\kappa\nabla u) - k^2u = f,\qquad u=g\ \text{on}\ \partial\Omega$$

它就是"刚度矩阵 $-$ $k^2\times$ 质量矩阵"，与 Poisson 完全共用一套元、求解器和后处理：

```python
import numpy as np
from MosicaFE import FESpace, HelmholtzProblem, Mesh, plot_solution, solve_helmholtz

k = 1.0
exact = lambda x: np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
source = lambda x: (2*np.pi**2 - k**2) * exact(x)

# 一行流
u = solve_helmholtz(Mesh.rectangle(nx=32, ny=32), element="P2", f=source, k=k)

# 分步写法（选网格 → 选元 → 选求解器 → 选可视化）
mesh = Mesh.rectangle(nx=32, ny=32)
V = FESpace.lagrange(mesh, degree=2)          # 换成 FESpace.vem(mesh) 也行
problem = HelmholtzProblem(V, f=source, g=lambda x: 0.0, k=k)
u = problem.solve(method="direct")
print(problem.compute_errors(u, exact))
plot_solution(u, mesh, V, filename="helmholtz.png")
```

要点：

* **低频就是"不共振"**：$k^2$ 小于 $-\Delta$ 的 Dirichlet 第一特征值
  $\lambda_1$ 时，矩阵 $A=\kappa K-k^2M$ 对称正定，解存在唯一。
  单位正方形上 $\lambda_1=2\pi^2\approx19.74$，即 $k<4.443$；
  实测离散判据 $k^2/\lambda_{1,h}$（P2、32×32）为 $0.0507$。
  $k^2>\lambda_1$ 后矩阵对称**不定**（实测 $k=5$ 时最小特征值变负），
  要用 `method="direct"` 或 `"minres"`，误差也会被放大。
* **网格要分辨波**：需要 $kh\lesssim1$（实测 $k=1$、$n=4\ldots32$ 对应
  $kh=0.25\ldots0.031$），否则看到的是污染（色散）误差而不是收敛阶。
* `k` 接受**实常数**（复数波数需要复数求解器，库会明确报错）；
  `-Δu + k²u = f` 这一支（屏蔽 Poisson 方程）请直接用
  `ReactionDiffusionProblem(..., reaction_coeff=k**2)`。
* RT0×P0 是混合格式，目前**没有**亥姆霍兹的实现，
  `solve_helmholtz(..., element="RT0")` 会明确拒绝而不是静默算错。

---

## 误差与收敛性

```python
u = problem.solve()
problem.compute_error(u, exact, "L2")          # L2
problem.compute_error(u, exact, "Linf")        # L∞
problem.compute_error(u, exact, "H1", exact_grad=exact_grad)   # H1 半范数
problem.compute_errors(u, exact, exact_grad)   # 三个一起返回
```

自动收敛性研究：

```python
from MosicaFE import convergence_study, estimate_convergence_rate

def mesh_factory(level):
    n = 4 * 2 ** level
    return Mesh.rectangle(nx=n, ny=n)

def problem_factory(mesh):
    return PoissonProblem(FESpace.lagrange(mesh, 2), f=source, g=lambda x: 0.0)

h, errors = convergence_study(mesh_factory, problem_factory,
                              exact, exact_grad, refinement_levels=5, plot=True)
print(estimate_convergence_rate(h, errors["L2"], use_last=3))
```

---

## 可视化

```python
from MosicaFE.postprocess import (
    plot_solution,      # 解（2D 等值线/曲面，3D 切片）
    plot_mesh,          # 网格
    plot_error,         # 逐点误差
    plot_convergence,   # log-log 收敛曲线
    plot_pressure_2d,   # RT0 压力
    plot_flux_2d,       # 通量箭头
    compare_solutions,  # 并排比较
)
plot_solution(u, mesh, V, title="VEM", filename="vem.png", show=False)
```

没有安装 matplotlib 时这些函数只会打印一条提示，不会抛异常。

---

## 扩展到其它 PDE

空间层已经把单元矩阵算好了，物理层只需要做"矩阵的线性组合"：

```python
import numpy as np
from MosicaFE.physics.base import BasePhysics
from MosicaFE.physics.registry import register_problem

@register_problem("point_source_poisson")
class PointSourcePoissonProblem(BasePhysics):
    """-Δu = f + δ(x - x0)（点源没法写成 f(x)，所以在 assemble 里自己加一项）"""
    def __init__(self, space, x0, f=None, g=None, **kw):
        super().__init__(space, f=f, g=g, **kw)
        self.x0 = np.asarray(x0, dtype=float)

    def assemble(self):
        A = self.space.stiffness_matrix()
        b = self.space.load_vector(self.f)
        # 对 P1/Q1/VEM 这类顶点型空间，φ_i(x0) = δ_ij（x0 取节点时），
        # 于是 ∫δ(x-x0)φ_i dx 就是"把 1 加到该自由度"。
        dist = np.linalg.norm(self.space.dof_coords - self.x0, axis=1)
        b[int(np.argmin(dist))] += 1.0
        return self.apply_boundary_conditions(A, b)
```

然后 `create_problem("point_source_poisson", V, x0=(0.5, 0.5), f=zero, g=zero)` 或
`MosicaFE.run("point_source_poisson", mesh, element="P1", x0=(0.5, 0.5))` 就能用了
（完整可运行版本见 `examples/custom_pde.py`）。

内置的 `HelmholtzProblem` 就是同一模式的另一个例子：它只比 Poisson 多一行
`A = stiffness_matrix() - k**2 * mass_matrix()`。

可用的空间原语：`stiffness_matrix()`、`mass_matrix()`、`load_vector(f)`、
`boundary_dofs()`、`interpolate(f)`、`evaluate(cell, u, points)`、
`nodal_values(u)`。

混合格式（RT0×P0）的空间原语与上面**平行**，只是换成了两个块：
`flux_mass_matrix()`、`divergence_matrix()`、`pressure_load_vector(f)`，
再配合 `MixedPoissonProblem` 拼出鞍点系统。

---

## 包结构

```
MosicaFE/
  core/        网格（mesh.py）、面拓扑（facets.py）、求积（quadrature.py）、几何工具
  spaces/      离散空间（Lagrange / VEM / RT0）与工厂 FESpace
  physics/     PDE 弱形式与组装
               （Poisson / Helmholtz / ReactionDiffusion / MixedPoisson + 注册表）
  solvers/     线性求解器
  postprocess/ 误差分析（含混合格式）、收敛性研究、可视化
  examples/    可运行示例
  tests/       测试与自检脚本
```

FEM、VEM、RT0 三类元都在库内实现，没有任何需要额外接入的外部数值后端。

---

## 验证结果

制造解 `u = sin(πx)sin(πy)sin(πz)`（2D 去掉 z 部分），实测收敛阶见
`tests/run_checks.py` 与本仓库 `progress/VERIFICATION.md`：

| 元 / 网格 | L2 | H1 |
| --- | --- | --- |
| P1 / 三角 | 1.98 | 0.99 |
| P2 / 三角 | 3.00 | 1.99 |
| Q1 / 四边形 | 2.00 | 1.00 |
| Q2 / 四边形 | 3.00 | 2.00 |
| VEM / 三角 | 1.98 | 0.99 |
| VEM / 四边形 | 1.99 | 1.00 |
| VEM / 五边形 | 2.00 | 1.00 |
| VEM / Voronoi 多边形 | 1.97 | 1.01 |
| P1 / 四面体 | 1.95 | 0.98 |
| Q1 / 六面体 | 2.01 | 1.00 |
| VEM / 六面体 | 1.97 | 0.97 |
| RT0-P0 / 三角（压力、通量） | 1.00 | 1.00 |

亥姆霍兹方程 `-Δu - k²u = f`（$k=1$ 常数波数，制造解同 `u = sin(πx)sin(πy)`，
由 `tests/run_checks.py` 的 `check_helmholtz()` 复现）与 Poisson **同阶**：

| 元 / 网格 | L2 | H1 |
| --- | --- | --- |
| P1 / 三角 | 1.98 | 0.99 |
| P2 / 三角 | 3.01 | 1.99 |
| Q1 / 四边形 | 2.00 | 1.00 |
| VEM / 三角 | 1.98 | 0.99 |

同时验证低频判据：离散第一特征值 $\lambda_{1,h}$（内点上 $Kv=\lambda Mv$ 的最小特征值）
实测 P1/VEM 为 19.7868、P2 为 19.7394，均从上方逼近理论值 $2\pi^2=19.7392$，
故 $k=1$ 时 $k^2/\lambda_{1,h}=0.0507$（远小于 1，远离共振）。

一条命令复现：

```powershell
& "D:\anaconda\envs\fealpy\python.exe" MosicaFE\tests\run_checks.py
& "D:\anaconda\envs\fealpy\python.exe" MosicaFE\tests\run_tests.py
```

---

## 已知限制

* Lagrange 只实现到 2 次（P2 / Q2）；更高次的接入点在 `spaces/reference.py`。
* VEM 只实现 k=1（每顶点一个自由度），且 3D 通用多面体需要显式给出面。
* RT0 只支持齐次 Dirichlet（在混合格式里是自然边界条件）；
  网格必须是 `H(div)` 协调的单纯形剖分（三角形 / 四面体），
  不协调的剖分会被 `core/facets.py` 直接拦下而不是静默算错。
* `HelmholtzProblem` 的 `k`、`kappa` 都只支持**实常数**（复波数/吸收介质需要
  复数求解器，构造时会明确报错）；`-Δu + k²u = f` 请用
  `ReactionDiffusionProblem(reaction_coeff=k**2)`。
* 亥姆霍兹没有混合（RT0×P0）实现，`solve_helmholtz(element="RT0")` 会明确拒绝。
* `VEMSpace(stabilization="none")` 仅供诊断，纯投影项是奇异的，不要用于求解。
* `Mesh.pentagon` 的单元含 180° 内角（退化多边形），实测收敛阶正常，
  但严格来说在 VEM 的星形正则性假设之外。
* 误差范数中，VEM 的解函数用"顶点自由度上的线性最小二乘重构"代替
  （虚拟元没有显式基函数），收敛阶与 `u_h` 一致。

---

## 许可

MosicaFE 按 MIT 发布，全部代码都是库自身实现。

RT0×P0 部分（`core/facets.py`、`spaces/rt0.py`、`physics/mixed_poisson.py`）
由作者自己的 `rt0fem` 程序（`D:\mosica\RT0`，MIT）重写并入本库，
不再以内嵌第三方包的形式存在；原始 `rt0fem` 包仍可独立使用。
