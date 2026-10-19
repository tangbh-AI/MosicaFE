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

---

## 目录

* [安装](#安装)
* [快速开始](#快速开始)
* [可选的"元"](#可选的元)
* [完整的调用方式](#完整的调用方式)
* [RT0 混合元](#rt0-混合元)
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
PoissonProblem(V, f=..., g=..., kappa=1.0)               # -∇·(κ∇u) = f
ReactionDiffusionProblem(V, f=..., reaction_coeff=c)      # -Δu + c u = f
MixedPoissonProblem(V_rt0, f=...)                         # RT0 混合形式（K=I）
DarcyProblem(V_rt0, f=..., K=2.5, bc=...)                 # RT0 混合形式的 Darcy 渗流
ConductionProblem(V_rt0, g=..., sigma=2.0, bc=...,
                  contact=...)                            # 定常电流场（欧姆定律 + 接触电阻）
```

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

### Darcy 渗流（内置）

达西定律 + 质量守恒的一阶系统，渗透率可以是标量、对角、一般对称正定张量或变系数：

```python
from MosicaFE import FESpace, Mesh, DarcyProblem

mesh = Mesh.rectangle(nx=32, ny=32)
V = FESpace.rt0(mesh)

# 上下不透水、左右定压 p=1 / p=0，渗透率 2.5 ⇒ 精确解 q=(2.5, 0)
bc = lambda mid, n: "D" if (mid[0] < 1e-12 or mid[0] > 1 - 1e-12) else "N"
problem = DarcyProblem(V, f=lambda x: 0.0, g=lambda x: 1.0 - x[:, 0], K=2.5, bc=bc)
Q, P = problem.solve()
print(problem.check_conservation())      # 逐单元守恒偏差

# 也可以一行调用
from MosicaFE import solve_darcy
Q, P = solve_darcy(mesh, f=src, K=[[3.0, 1.0], [1.0, 2.0]])
```

与 `MixedPoissonProblem` 相比只有两处不同：

1. **通量质量矩阵带 `K^{-1}` 权重** `A_ij = ∫ (K^{-1}φ_i)·φ_j`，
   由空间层的 `V.weighted_flux_mass_matrix(K)` 提供（`K=I` 时与 `flux_mass_matrix()` 逐位相同）；
   两种组装方式：`mode="cell"`（每单元常数 `K`，**精确闭式**，分片常数渗透率时无误差）
   与 `mode="quad"`（求积点逐点取值，光滑 `K` 更准）；
2. **支持本质（流量）边界条件**：`bc(mid, n_out)` 返回
   `"D"`（压力 Dirichlet，可为非齐次）/ `"N"`（不透水 `q·n=0`）/
   `("flux", gN)`（给定流量），压力条件仍走右端（自然边界条件）。

鞍点求解与 `MixedPoissonProblem` 同风格（`"schur"` 密凝聚、`"schur_cg"` 稀疏 LU+CG、
`"direct"`/`"minres"` 等），纯 Neumann 时自动取零均值并拦下会解奇异矩阵的直接法。

### 定常电流场（内置）

欧姆定律 + 电荷守恒，与 Darcy **完全同构**（$\sigma\leftrightarrow K$、$\varphi\leftrightarrow p$、
$\boldsymbol J\leftrightarrow\boldsymbol q$），所以直接复用同一套鞍点装配：

$$
\boldsymbol J=-\sigma\nabla\varphi,\qquad \nabla\cdot\boldsymbol J=f .
$$

```python
from MosicaFE import FESpace, Mesh, ConductionProblem

mesh = Mesh.rectangle(nx=32, ny=32)
V = FESpace.rt0(mesh)

# 左右电极 1 V / 0 V，上下绝缘，电导率 σ=2 ⇒ J=(2,0)，R=0.5
bc = lambda mid, n: "D" if (mid[0] < 1e-12 or mid[0] > 1 - 1e-12) else "N"
problem = ConductionProblem(V, g=lambda x: 1.0 - x[:, 0], sigma=2.0, bc=bc)
J, phi = problem.solve()

print(problem.total_current(lambda m: m[0] > 1 - 1e-12))   # 总电流 I = 2
print(problem.resistance(1.0, lambda m: m[0] > 1 - 1e-12)) # 电阻  R = 0.5
print(problem.joule_heating())                             # 焦耳热 P = Δφ·I = 2
```

**接触电阻**（电极/材料界面的面接触电阻率 $R_c$）在混合格式里是对角修正：
由分部积分

$$
(\sigma^{-1}\boldsymbol J,\boldsymbol v)-(\varphi,\nabla\cdot\boldsymbol v)
+\langle\llbracket\varphi\rrbracket,\boldsymbol v\cdot\boldsymbol n\rangle
+\langle\varphi,\boldsymbol v\cdot\boldsymbol n\rangle_{\partial\Omega}=0,
$$

标准方法把 $\llbracket\varphi\rrbracket$ 当 0（电势连续）；界面有接触电阻时
$\llbracket\varphi\rrbracket=R_c\,\boldsymbol J\cdot\boldsymbol n$，于是

$$
\bigl\langle R_c(\boldsymbol J\cdot\boldsymbol n),v_f\cdot\boldsymbol n\bigr\rangle
=\frac{R_c}{|f|}\Phi_f
\quad\Longrightarrow\quad
A_{ff}\leftarrow A_{ff}+\frac{R_c(f)}{|f|}.
$$

```python
# 在 x=0.5 的界面上加 R_c = 0.5（二维按单位厚度）：总电阻 = 1 + 0.5 = 1.5
problem = ConductionProblem(V, g=lambda x: 1.0 - x[:, 0], sigma=1.0, bc=bc,
                            contact=lambda mid: 0.5 if abs(mid[0] - 0.5) < 1e-12 else None)
J, phi = problem.solve()
print(problem.resistance(1.0, lambda m: m[0] > 1 - 1e-12))  # 1.5
```

边界条件与 Darcy 相同：`"D"` 电极电势（自然边界，可为非齐次）、`"N"` 绝缘
（$\boldsymbol J\cdot\boldsymbol n=0$）、`("flux", gN)` 注入电流密度（本质边界）。
也可一行调用 `solve_conduction(mesh, g=..., sigma=2.0, bc=...)`。

验证（`RT0-Conduction/conduction_rt0.py`）：均质条形 $I=2,R=0.5,P=2$ 精确；
串联分层 $R=\sum L_i/\sigma_i$ 相对误差 $7\times10^{-15}$；接触电阻 $R=1.5$、
电势跳变 $=R_c I$ 精确到 $10^{-10}$；$\sigma=K$ 时与 `DarcyProblem` 逐位同解。


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
from MosicaFE.physics.base import BasePhysics
from MosicaFE.physics.registry import register_problem

@register_problem("helmholtz")
class HelmholtzProblem(BasePhysics):
    """-Δu - k²u = f"""
    def __init__(self, space, f=None, g=None, k=1.0, **kw):
        super().__init__(space, f=f, g=g, **kw)
        self.k = k

    def assemble(self):
        A = self.space.stiffness_matrix() - self.k**2 * self.space.mass_matrix()
        b = self.space.load_vector(self.f)
        return self.apply_boundary_conditions(A, b)
```

然后 `create_problem("helmholtz", V, f=..., k=2.0)` 或
`MosicaFE.run("helmholtz", mesh, element="VEM", f=..., k=2.0)` 就能用了。

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
  physics/     PDE 弱形式与组装（Poisson / MixedPoisson / Darcy / Conduction / ReactionDiffusion + 注册表）
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
| RT0-P0 / Darcy（压力、通量；各向异性 K） | 1.00 | 1.00 |
| RT0-P0 / 电流场（电阻、接触电阻；与 Darcy 逐位同解） | 精确 | — |

一条命令复现：

```powershell
& "D:\anaconda\envs\fealpy\python.exe" MosicaFE\tests\run_checks.py
& "D:\anaconda\envs\fealpy\python.exe" MosicaFE\tests\run_tests.py
```

---

## 已知限制

* Lagrange 只实现到 2 次（P2 / Q2）；更高次的接入点在 `spaces/reference.py`。
* VEM 只实现 k=1（每顶点一个自由度），且 3D 通用多面体需要显式给出面。
* RT0 的网格必须是 `H(div)` 协调的单纯形剖分（三角形 / 四面体），
  不协调的剖分会被 `core/facets.py` 直接拦下而不是静默算错。
* `MixedPoissonProblem` 只支持**齐次** Dirichlet（混合格式里的自然边界条件）；
  需要非齐次压力、不透水或给定流量边界时用 `DarcyProblem`。
* `DarcyProblem` 的渗透率按**单元常数**（`mode="cell"`）或**求积点取值**（`mode="quad"`）
  离散，不支持跨越单元间断的复杂分片函数解析积分；
  纯 Neumann 时它只能走 Schur 路径（压力差一个常数，自动取零均值）。
* `VEMSpace(stabilization="none")` 仅供诊断，纯投影项是奇异的，不要用于求解。
* `Mesh.pentagon` 的单元含 180° 内角（退化多边形），实测收敛阶正常，
  但严格来说在 VEM 的星形正则性假设之外。
* 误差范数中，VEM 的解函数用"顶点自由度上的线性最小二乘重构"代替
  （虚拟元没有显式基函数），收敛阶与 `u_h` 一致。

---

## 许可

MosicaFE 按 MIT 发布，全部代码都是库自身实现。

RT0×P0 部分（`core/facets.py`、`spaces/rt0.py`、`physics/mixed_poisson.py`、
`physics/darcy.py`）由作者自己的 `rt0fem` 程序（`D:\mosica\RT0`，MIT）重写并入本库，
不再以内嵌第三方包的形式存在；原始 `rt0fem` 包仍可独立使用。
