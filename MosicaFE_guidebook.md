# MosicaFE 使用手册

FEM / VEM / RT0 混合元与 PyTorch 深度学习 PDE 求解框架

本手册沿用参考 guidebook 的分层结构、章节组织和简洁主程序风格，说明本仓库实际实现的接口。
运行环境统一为 conda **AI4Sci**。深度学习模块位于 `MosicaFE/deeplearning/`，仅使用 PyTorch。

## 目录

1. [概述与主程序](#1-概述与主程序)
2. [安装与环境](#2-安装与环境)
3. [数学约定](#3-数学约定)
4. [网格与求积](#4-网格与求积)
5. [离散空间](#5-离散空间)
6. [物理问题](#6-物理问题)
7. [传统求解器](#7-传统求解器)
8. [误差与收敛](#8-误差与收敛)
9. [可视化](#9-可视化)
10. [完整示例](#10-完整示例)
11. [进阶工作流](#11-进阶工作流)
12. [扩展接口](#12-扩展接口)
13. [深度学习](#13-深度学习)
14. [日志、进度与恢复铁则](#14-日志进度与恢复铁则)
15. [接口速查与限制](#15-接口速查与限制)

## 1. 概述与主程序

### 1.1 设计哲学

传统计算流程为“PDE → 弱形式 → 离散空间 → 线性系统 → 求解 → 后处理”。
深度学习流程为“PDE/数据 → 神经函数 → 残差/数据损失 → 优化 → 后处理”。
两者共用网格、物理坐标、求积、自由度插值和误差分析；神经训练保留自己的 PyTorch 计算图。

### 1.2 传统主程序

```python
import numpy as np
from MosicaFE import Mesh, FESpace, PoissonProblem, plot_solution

exact = lambda x: np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
source = lambda x: 2*np.pi**2 * exact(x)
mesh = Mesh.rectangle(nx=16, ny=16)
V = FESpace.lagrange(mesh, degree=2)
problem = PoissonProblem(V, f=source)
uh = problem.solve()
print(problem.compute_errors(uh, exact))
plot_solution(uh, mesh, V, filename="solution.png")
```

### 1.3 神经主程序

```python
from MosicaFE.deeplearning import MLP, PINNProblem

network = MLP(2, hidden_layers=(32, 32, 32), bounds=[[0, 1], [0, 1]])
problem_nn = PINNProblem(mesh, network, f=source)
u = problem_nn.solve(epochs=2000, lr=.003)
print(u.compute_errors(exact))
plot_solution(u.to_fe(V), mesh, V, filename=str(u.run_dir / "solution.png"))
```

`uh` 是传统自由度向量；`u` 是可求值的 `NeuralSolution`。通过 `to_fe(V)` 获得兼容传统后处理的向量。

### 1.4 包结构

| 层 | 职责 |
| --- | --- |
| `core` | Mesh、几何、面拓扑、求积 |
| `spaces` | Lagrange / VEM / RT0 空间与矩阵原语 |
| `physics` | Poisson / ReactionDiffusion / Helmholtz / MixedPoisson 弱形式 |
| `solvers` | 线性系统求解 |
| `postprocess` | 积分误差、收敛、绘图 |
| `deeplearning` | PyTorch MLP、PINN、监督学习、训练与数据适配 |
| `examples` / `tests` | 可运行主程序与回归验证 |

## 2. 安装与环境

Python >= 3.9，传统核心仅要求 numpy；scipy 和 matplotlib 为可选依赖。
`deeplearning` 要求 torch >= 2.0；本次实际验证环境为 Python 3.12、torch 2.4.0，CPU 与 CUDA 均通过验证。

```powershell
cd D:\code\dcm_pro
$env:PYTHONPATH = "D:\code\dcm_pro"
& "D:\anaconda\envs\AI4Sci\python.exe" -m pip install -e './MosicaFE[deeplearning]'
```

环境中已安装 PyTorch 时，无需再次安装即可通过父目录导入运行。
常规 `import MosicaFE` 不导入 torch；显式导入 `MosicaFE.deeplearning` 或根级神经 API 时才加载。
CUDA 训练指定 `device="cuda"`；未提供 CUDA 时显式报错，默认采用 CPU。

## 3. 数学约定

### 3.1 传统弱形式

Poisson：$-\nabla\cdot(\kappa\nabla u)=f$，$u=g$ 于边界。
常系数弱形式为 $\kappa(\nabla u,\nabla v)=(f,v)$。
反应扩散为 $-\Delta u+c u=f$；Helmholtz 为 $-\kappa\Delta u-k^2u=f$。
RT0 混合格式使用 $q=-\nabla p$，压力为 P0 单元常数。

### 3.2 PINN 损失

$$L=\lambda_r\frac{\sum_i w_i |R[u_\theta](x_i)|^2}{\sum_i w_i}
 +\sum_j\lambda_j\frac{\sum_i w_{ji}|B_j[u_\theta](x_{ji})-g_j(x_{ji})|^2}{\sum_i w_{ji}}
 +\lambda_d L_{data}.$$

默认内部残差权重为 1，Dirichlet 权重为 10，数据权重为 1。
多输出损失在通道方向取平均。日志中每个分项已乘对应损失权重，`total` 是分项之和。
盒区域内部点使用均匀随机采样；网格内部点使用带物理权重的单元求积。
训练 MSE 禁止负权重，默认 order=2。常规积分误差沿用原库求积规则。

### 3.3 全连接网络与 PDE

`MLP` 是普通全连接函数近似器。以残差和边界训练它构成 PINN；以 PDE 解标签训练它构成监督模型。
监督方法需要内部解数据，来自 FEM/VEM/混合元、观测或制造解；仅给源项和边界不是监督标签。

## 4. 网格与求积

```python
mesh1 = Mesh.interval(a=0, b=1, nx=32)
mesh2 = Mesh.rectangle(nx=16, ny=16, element_type="triangle")
meshq = Mesh.rectangle(nx=16, ny=16, element_type="quad")
mesh3 = Mesh.box(nx=4, ny=4, nz=4, element_type="tet")
meshh = Mesh.box(nx=4, ny=4, nz=4, element_type="hex")
meshp = Mesh.pentagon(nx=4, ny=4)
```

`mesh.nodes` 为 `(n_nodes,dim)`；`mesh.elements` 为变长顶点编号列表。
`boundary_facets` 给真实边界面，`cell_centroids()`、`cell_measures()` 给逐单元几何数据。
`mesh.refine_uniform()` 支持结构化工厂生成的网格。

```python
from MosicaFE import cell_quadrature
points, weights = cell_quadrature(mesh2, cell_idx=0, order=2)
```

`Domain.from_mesh(mesh)` 共用以上数据，不把非矩形网格的包围盒当作实际区域。
一般多边形/多面体的求积适用范围继承原库的扇形分解约定；凹或非星形单元需先提供可靠网格。

## 5. 离散空间

| 元 | 空间创建 | 网格 |
| --- | --- | --- |
| P1/P2 | `FESpace.lagrange(mesh, degree=1/2)` | 三角形/四面体 |
| Q1/Q2 | `FESpace.lagrange(mesh, degree=1/2)` | 四边形/六面体 |
| VEM1 | `FESpace.vem(mesh)` | 多边形/多面体 |
| RT0×P0 | `FESpace.rt0(mesh)` | 三角形/四面体 |

H1 空间原语为 `stiffness_matrix`、`mass_matrix`、`load_vector`、`interpolate`、`evaluate`、`evaluate_gradient`。
RT0 原语为 `flux_mass_matrix`、`divergence_matrix`、`pressure_load_vector`、`interpolate_pressure`、`interpolate_flux`。
RT0 面法向自由度是积分通量；全局法向在拓扑中以向量面积保存。插值积分时使用单位法向，面测度只乘一次。

## 6. 物理问题

```python
from MosicaFE import ReactionDiffusionProblem, HelmholtzProblem, MixedPoissonProblem

poisson = PoissonProblem(V, f=source, g=lambda x: 0., kappa=1.)
reaction = ReactionDiffusionProblem(V, f=source, reaction_coeff=2.)
wave = HelmholtzProblem(V, f=source, k=1.)
mixed = MixedPoissonProblem(FESpace.rt0(mesh), f=source)
Q, P = mixed.solve()
```

混合格式 `Q` 为 `(n_cells,dim+1)` 的局部外法向积分通量，`P` 为 `(n_cells,)`。
传统物理注册表通过 `create_problem` 和 `run` 扩展；神经问题通过自定义可微残差扩展。

## 7. 传统求解器

`problem.solve(method="auto")` 或 `"direct"/"cg"/"gmres"/"bicgstab"/"minres"`。
SPD 问题可用 CG；对称不定系统可用 MINRES；混合问题另提供 Schur 方式。
深度学习的优化器由 `TrainingConfig` 管理，与线性求解器的 `method` 含义分开。

## 8. 误差与收敛

```python
problem.compute_errors(uh, exact, exact_grad)
problem.compute_error(uh, exact, "L2")
u.compute_errors(exact, exact_grad, mesh=mesh)
```

L2 为积分范数，H1 为梯度半范数，Linf 为离散采样点最大误差。
`NeuralSolution.compute_errors` 在单元求积点调用神经函数及自动微分梯度。
它和 `PoissonProblem(V).compute_errors(u.to_fe(V),...)` 的含义不同：后者测量离散插值。
神经误差未必随网格加密单调变化；训练损失不能替代独立评估误差。

## 9. 可视化

```python
from MosicaFE import plot_solution, plot_mesh, plot_pressure_2d, plot_flux_2d
plot_solution(u.to_fe(V), mesh, V, filename=str(u.run_dir / "solution.png"), show=False)
plot_mesh(mesh, filename="mesh.png")
plot_pressure_2d(mesh, P, filename="pressure.png")
plot_flux_2d(space=mixed.space, Q=Q, filename="flux.png")
```

二维采用等值图，三维采用切片。已有 `plot_solution` 用于网格解的展示；深度学习输出先插值到空间。
后台运行设置 `$env:MPLBACKEND="Agg"`，图片保存至训练目录。

## 10. 完整示例

| 文件 | 内容 |
| --- | --- |
| `examples/poisson_demo.py` | 传统 Poisson |
| `examples/rt0_mixed_demo.py` | RT0 混合元 |
| `examples/custom_pde.py` | 自定义传统 PDE |
| `examples/deeplearning_poisson.py` | 二维 Poisson PINN、积分误差、绘图、续训 |
| `examples/deeplearning_fem_data.py` | 普通全连接网络学习 FEM 解、独立点评估 |
| `examples/deeplearning_heat.py` | 时空热方程 PINN、初值与空间边界 |

在 `D:\code\dcm_pro` 运行：

```powershell
& "D:\anaconda\envs\AI4Sci\python.exe" MosicaFE/examples/deeplearning_poisson.py --epochs 2000 --run-dir logs/my_poisson 2>&1 |
    Tee-Object -FilePath logs/my_poisson_command.log
```

示例支持 `--device cuda`、`--run-dir` 和 `--resume <检查点路径>`；`--epochs` 为累计训练目标。
新任务必须使用新的目录；同目录续训必须显式提供检查点。

## 11. 进阶工作流

### 11.1 传统解到普通网络

```python
from MosicaFE.deeplearning import PDEData, solve_supervised
data = PDEData.from_fe(V, uh)
u_nn = solve_supervised(data, domain=mesh, epochs=2000, lr=.003)
uh_nn = u_nn.to_fe(V)
```

训练点来自单元内部求积，不局限于节点。VEM 标签使用现有线性最小二乘重构。
`PDEData(points, values, weights=None)` 也接收外部数据；points 为 `(N,dim)`，values 为 `(N,C)` 或 `(N,)`。

### 11.2 数据辅助 PINN

```python
network = MLP(mesh.dim, bounds=[[0, 1], [0, 1]])
u_hybrid = PINNProblem(mesh, network, f=source, data=data, data_weight=.1).solve(epochs=2000)
```

这是“PDE 残差 + 边界 + 传统解数据”的联合损失，可作为后续混合算法的基础。
`u.tensor(x)` 返回保留计算图的 torch 输出；NumPy 传统矩阵/插值适配不保留对参数的梯度。
若以后需要弱形式或离散残差的可微联合训练，应另写 torch 稀疏矩阵/积分算子，通过自定义残差接入。

### 11.3 RT0 压力与通量

```python
Vrt = FESpace.rt0(mesh)
Q, P = MixedPoissonProblem(Vrt, f=source).solve()
p_data = PDEData.from_fe(Vrt, (Q, P), field="pressure")
q_data = PDEData.from_fe(Vrt, (Q, P), field="flux")
p_nn = solve_supervised(p_data, domain=mesh, epochs=2000)
q_nn = solve_supervised(q_data, domain=mesh, epochs=2000)
P_nn = p_nn.to_pressure(Vrt)
Phi_nn = q_nn.to_flux(Vrt, components=[0, 1])
Q_nn = np.vstack([Vrt.local_flux_values(c, Phi_nn, global_values=True) for c in range(mesh.n_cells)])
```

标量势函数 `u` 的 `to_flux(Vrt,kappa=...)` 默认生成 $q=-\kappa\nabla u$ 的全局面通量。
`to_pressure` 计算 P0 单元平均，`to_flux` 计算面法向积分，均不以单点值代替积分。
转换全局通量时显式指定 `global_values=True`，避免单单元网格的局部/全局向量长度歧义。

## 12. 扩展接口

### 12.1 传统 PDE

继承 `physics.base.BasePhysics`，实现 `assemble()`，并用 `register_problem` 注册。
空间层计算矩阵，物理层组合并施加边界条件；完整模板见 `examples/custom_pde.py`。

### 12.2 可微残差

```python
from MosicaFE.deeplearning import laplacian
def residual(x, u):
    return -laplacian(u, x) + u**3 - 1.
nonlinear = PINNProblem(Domain.rectangle(), MLP(2), residual=residual)
```

残差接收 `(N,dim)` 的可求导 torch 坐标与 `(N,C)` 输出，返回 `(N,)` 或 `(N,R)` torch 张量。
不要在残差内转换成 NumPy、使用 `detach()` 或 `torch.no_grad()`。
多输出 PDE 可用 `gradient(u,x,component=i)`，每个分量构建残差后 `torch.cat`。
自定义网络可传任何样本独立的 `torch.nn.Module`；它需要被原样重建以续训。

## 13. 深度学习

### 13.1 区域、网络与配置

```python
from MosicaFE.deeplearning import Domain, TrainingConfig, Trainer
domain = Domain.rectangle()
network = MLP(2, output_dim=1, hidden_layers=(64, 64), activation="tanh", bounds=domain.bounds)
config = TrainingConfig(epochs=2000, lr=.001, optimizer="adam", dtype="float64", device="cpu")
u = Trainer(config).fit(PINNProblem(domain, network, f=source))
```

`Domain.interval`、`Domain.rectangle`、`Domain.box(bounds)` 提供无网格盒区域。
盒区域 `n_interior=512`、`n_boundary=128`；每张边界面都必须采样，因此 `n_boundary>=2*dim`。
Mesh 区域用全部 order-2 求积点，`n_interior/n_boundary` 在这种模式下不控制点数，网格密度控制规模。
MLP 可用 tanh/sigmoid/silu/softplus；ReLU 可用于监督学习，但当前二阶 PINN 会拒绝。
`bounds` 在网络内将物理坐标映射到 [-1,1]，自动微分仍对原物理坐标求导，包含链式法则。
不要加入 BatchNorm 或跨样本 attention；本模块的 pointwise 导数假设样本之间独立。

### 13.2 内置方程

```python
from MosicaFE.deeplearning import solve_pinn
u = solve_pinn(domain, equation="poisson", f=source)
u = solve_pinn(domain, equation="reaction_diffusion", f=source, reaction=2.)
u = solve_pinn(domain, equation="helmholtz", f=source, k=1.)
```

统一 elliptic residual 为 $-\nabla\cdot(\kappa\nabla u)+c u-f$。
默认 `backend="numpy"`，源项与边界是传统 `f(point)` / `g(point)` 函数，或标量常数。
`backend="torch"` 时函数接收整批 `(N,dim)` torch 张量，返回标量、`(N,)` 或 `(N,1)`。
变扩散系数必须可微，不能走 NumPy 分支：

```python
u = solve_pinn(domain, backend="torch",
               f=lambda x: torch.ones_like(x[:, :1]),
               kappa=lambda x: 1 + x[:, :1]**2)
```

内置椭圆残差仅支持标量网络输出。系统或其他算子请传自定义 residual。

### 13.3 边界与初值

```python
from MosicaFE.deeplearning import DirichletBC, NeumannBC, InitialCondition, PointCondition
left = np.column_stack([np.zeros(32), np.linspace(0, 1, 32)])
right = np.column_stack([np.ones(32), np.linspace(0, 1, 32)])
conditions = [DirichletBC(left, 0., name="left", weight=10.),
              NeumannBC(right, np.tile([1., 0.], (32, 1)), 0., name="right", weight=10.)]
u = PINNProblem(domain, MLP(2), f=source, conditions=conditions).solve(epochs=2000)
```

条件名称必须唯一，不能使用 `residual/data/total`。`conditions=None` 自动为整个边界生成 Dirichlet；
显式列表则完全替代默认边界条件，`conditions=[]` 可用于自行处理约束的自定义网络。
`NeumannBC` 指定 $\partial u/\partial n$，法向必须为单位向量；物理通量 $-\kappa\partial u/\partial n$ 用自定义 operator 表达。
纯 Neumann 问题还需解的归一化约束和源项兼容性，库不自动添加。
Robin 条件可用 `PointCondition(...,operator=lambda x,u: alpha*u+beta*...)`。

### 13.4 热方程

以 `(x,t)` 为输入，最后一维默认为时间，只对空间方向计算 Laplacian。

```python
domain = Domain.box([[0., 1.], [0., .5]])
t = np.linspace(0, .5, 64)
bx = np.vstack([np.column_stack([np.zeros(64), t]), np.column_stack([np.ones(64), t])])
ix = np.column_stack([np.linspace(0, 1, 64), np.zeros(64)])
conditions = [DirichletBC(bx, 0., weight=10.),
              InitialCondition(ix, lambda p: np.sin(np.pi*p[0]), weight=10.)]
u = solve_pinn(domain, equation="heat", conditions=conditions, epochs=2500, lr=.003)
```

终止时间没有 Dirichlet 条件。`HeatResidual` 当前支持常数标量扩散系数。
完整可运行版本与独立时空误差评估见 `examples/deeplearning_heat.py`。

### 13.5 Adam 与 L-BFGS

默认全批 Adam。`optimizer="lbfgs"` 使用 PyTorch closure 和 strong-Wolfe 线搜索，
每个外层 epoch 最多 `lbfgs_max_iter=20` 次内部迭代；Adam epoch 是一次参数更新。
Adam 训练完成后可用同一模型启动新的 L-BFGS 训练目录：

```python
polished = PINNProblem(domain, u.model, f=source).solve(
    optimizer="lbfgs", lr=1., epochs=20)
```

这是新阶段，不是改变原优化器的断点续训。`tolerance>0` 可按总损失提前停止。
`callback(trainer,metrics)` 每个完成步后调用，可扩展外部监测或调用 `trainer.save_checkpoint("custom.pt")`。
函数回调抛出的异常也会触发状态/检查点保存。

## 14. 日志、进度与恢复铁则

### 14.1 工作记录

**铁则：所有运行日志与进度必须保存；每次主动或被动结束前要尽可能预记当前任务、检查点和恢复命令，
使下次能够继续上次进度。**

本工作区记录位置固定为 `D:\code\dcm_pro`：

| 文件/目录 | 用途 |
| --- | --- |
| `AGENTS.md` | 后续协作必须遵守的持续日志铁则 |
| `WORK_PROGRESS.md` | 已完成事项、验证、当前任务、下一步和阻塞 |
| `logs/<编号>_<任务>.log` | 命令输出，包括错误 |
| `logs/deeplearning/<唯一运行目录>/` | 默认训练目录 |

开始先读进度；关键步骤前后更新进度；交付、暂停或等待输入前先记录。
长任务运行前写下恢复路径，并定期保存。系统强制杀进程时无法执行 Python 回调，不能只依赖退出钩子。
本项目通过预先记录、初始/周期/退出检查点应对此情况。

### 14.2 每次训练的持久化产物

| 产物 | 内容 |
| --- | --- |
| `config_<会话>.json` | 本次配置、问题摘要、torch 版本、恢复来源 |
| `training_<会话>.log` | 开始、周期日志、异常堆栈和结束 |
| `metrics_<会话>.jsonl` | 每个完成步的残差、条件、数据和总损失 |
| `status.json` | 最近状态、完成步、目标、检查点和错误 |
| `last.pt` | 初始、每 `checkpoint_every` 步以及正常结束的原子检查点 |
| `interrupted.pt` / `failure.pt` | 可捕获中断/错误时尽力保存的检查点 |

文件写入临时文件并通过 `os.replace` 原子替换，防止半写入破坏已有检查点。
指标每步 flush，续训新增会话文件，不删除历史日志。
默认每 50 步保存，可配置 `checkpoint_every=1`；强制终止时最多损失一个周期的参数更新。
中断可能发生在优化器内部时，退出检查点用于尽力恢复；需要从完整步严格恢复可选择最近的 `last.pt`。
内置框架不接管操作系统的不可捕获终止事件。

### 14.3 续训与推理

```python
u = solve_pinn(mesh, f=source, epochs=3000, lr=.003,
               resume_from="logs/my_poisson/last.pt")
from MosicaFE.deeplearning import NeuralSolution
loaded = NeuralSolution.load("logs/my_poisson/last.pt", device="cpu")
print(loaded([[.5, .5]]))
```

续训 `epochs=3000` 表示总目标 3000 步，不是在原有步数上再训练 3000 步。
网络、条件、问题常数、权重、优化器、学习率、dtype 和 L-BFGS 内部迭代参数须匹配。
保存并恢复采样点、条件值、数据、网络、优化器、随机状态、指标历史。
必须重建完全相同的 Python 残差、算子和物理回调；它们不通过 pickle 保存。
不同函数代码即使名称相同也不是同一 PDE，用户应核对原定义。
相同环境和确定性算子下，Adam 的拆分/连续训练在测试中达到参数逐位一致。
CUDA 结果还取决于相应算子的确定性和环境，不承诺跨硬件逐位一致。
加载检查点使用 PyTorch 序列化，只加载本项目自行产生的可信文件。

## 15. 接口速查与限制

| 接口 | 结果/约定 |
| --- | --- |
| `solve_pinn(domain,...)` | 返回 NeuralSolution；domain 为 Domain/Mesh/bounds |
| `solve_supervised(PDEData,...)` | 返回普通 MLP 监督解 |
| `PINNProblem(...).solve(config)` | 显式 TrainingConfig |
| `SupervisedProblem(model,data,domain).solve(...)` | 自选网络和数据 |
| `u(points)` / `u.predict(points)` | NumPy；单标量点为标量，批量为 `(N,)` 或 `(N,C)` |
| `u.tensor(points)` | torch `(N,C)`，保留计算图 |
| `u.gradient(points,component=0)` | NumPy `(N,input_dim)` |
| `u.to_fe(V)` | Lagrange/VEM 插值自由度 |
| `u.to_pressure(Vrt)` | RT0 P0 平均值 |
| `u.to_flux(Vrt)` | 全局面通量，转局部用 `Vrt.local_flux_values` |
| `u.compute_errors(...)` | scalar L2/H1/Linf，需有 Mesh |
| `NeuralSolution.load(path)` | 内置 MLP 推理；原 Mesh 需另传给误差函数 |

一维批量点必须写为 `(N,1)`，例如 `x[:,None]`；长度 dim 的一维数组表示单个点。
当前训练采用固定采样与全批损失，没有 minibatch、动态重采样、分布式训练、神经算子或自适应损失权重。
FEM/VEM/RT0 适配目前是数值交换，不是可微传统求解器；不自动实现新的混合离散算法。
传统 Lagrange 仅到二次，VEM 为一次，RT0 仅支持单纯形网格且混合 PDE 的 Dirichlet 边界仍限齐次。
深度学习模块为后续方法组合提供接口；需要额外实现的新算法应独立验证其一致性、稳定性和收敛性。

运行全部验证（输出另存日志）：

```powershell
& "D:\anaconda\envs\AI4Sci\python.exe" -m pytest MosicaFE/tests -q 2>&1 |
    Tee-Object -FilePath logs/all_tests.log
```

本次实现的详细验证结果和恢复位置以工作区 `WORK_PROGRESS.md` 为准。
PyTorch 自动微分与优化器约定参见 [torch.autograd.grad](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)
和 [L-BFGS](https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html)。
