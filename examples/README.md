# MosicaFE · RT0 求解器分支

本分支（`rt0-darcy-conduction`）在 MosicaFE 主体之上加入了 **RT0 混合元求解器**与三个方程的
完整算例：**Darcy 渗流**、**定常电流场**、**Navier–Stokes（分析报告）**。

> 上游仓库：<https://github.com/tangbh-AI/MosicaFE>　·　作者：赵典（<https://github.com/ZhaoDian-3>）

## 目录

```
.
├── MosicaFE/                     # 库主体（RT0/Nédélec、混合元、VEM、求解器、后处理）
└── examples/
    ├── RT0-Darcy/                # ① Darcy 渗流（RT0×P0，各向异性/分层/流量边界）
    │   ├── rt0_darcy_2d.py       #    独立实现（手写 RT0，不依赖库）
    │   ├── darcy_mosicafe.py     #    调用库的版本（与独立实现相互印证）
    │   ├── darcy_plot.py         #    Pillow 绘图工具（无 matplotlib 依赖）
    │   ├── 讲解_RT0求解Darcy方程.md
    │   └── figs/                 #    20 张图（两套实现各一份，6 张逐字节相同）
    ├── RT0-Conduction/           # ② 定常电流场（欧姆定律 + 接触电阻）
    │   ├── conduction_rt0.py     #    验证脚本：电阻、分层串联、接触电阻、与 Darcy 对拍
    │   ├── 结果_电流场.txt
    │   └── figs/
    └── NS-RT0/                   # ③ Navier–Stokes：RT0 可行性的分析报告
        ├── 报告_NS方程的RT0求解.pdf / .tex
        ├── ns_rt0_验证.py         #    复现报告中的全部数字
        ├── ns_stokes_prototype.py #    Taylor–Hood 原型（未通过线性精确解测试，见报告 §5）
        └── 结果_ns_rt0.txt
```

## 这个分支加了什么

| 位置 | 内容 | 状态 |
| --- | --- | --- |
| `MosicaFE/spaces/rt0.py` | `weighted_flux_mass_matrix(K)` 与加权质量矩阵原语（闭式） | ✅ 已测 |
| `MosicaFE/physics/darcy.py` | `DarcyProblem`：各向异性/变系数 `K`、压力 Dirichlet、流量边界、纯 Neumann 兜底 | ✅ 已测 |
| `MosicaFE/physics/conduction.py` | `ConductionProblem`：电流场、**界面接触电阻**、电阻/电流/焦耳热后处理 | ✅ 已测 |
| `MosicaFE/api.py` | `solve_darcy(...)`、`solve_conduction(...)` 一行式接口 | ✅ 已测 |
| `MosicaFE/tests/` | Darcy 10 项 + 电流场 6 项测试 | ✅ 109 项全通过 |
| `examples/RT0-Darcy` | RT0 求解 Darcy 的完整推导、实现与误差可视化 | ✅ |
| `examples/RT0-Conduction` | 电流场求解与四项验证 | ✅ |
| `examples/NS-RT0` | NS 方程用 RT0 的**可行性分析**（两条否定结论 + 正确路线与装配公式） | 📄 分析报告 |

## 三个方程的方程与验证

### ① Darcy 渗流（RT0×P0）

$$\boldsymbol q=-K\nabla p,\qquad \nabla\cdot\boldsymbol q=f$$

混合弱形式：$A_{ij}=\int (K^{-1}\boldsymbol\varphi_i)\cdot\boldsymbol\varphi_j$、
$B_{Ki}=\int_K\nabla\cdot\boldsymbol\varphi_i$，鞍点系统
$\begin{pmatrix}A&-B^{\mathsf T}\\B&0\end{pmatrix}$。压力 Dirichlet 是自然边界条件
（右端 $-\langle g_D,\boldsymbol\varphi\cdot\boldsymbol n\rangle$），流量边界是本质边界条件。

验证：各向异性 $K=\begin{pmatrix}3&1\\1&2\end{pmatrix}$ 收敛阶 $0.999/1.000$；
分层介质界面通量 $=2/(1/\kappa_L+1/\kappa_R)=1.9801980198$；逐单元守恒 $10^{-15}$；
与库内 `MixedPoissonProblem` 在 $K=I$ 时逐位相同。

### ② 定常电流场（RT0×P0）

$$\boldsymbol J=-\sigma\nabla\varphi,\qquad \nabla\cdot\boldsymbol J=f$$

**界面接触电阻** $\llbracket\varphi\rrbracket=R_c\,\boldsymbol J\cdot\boldsymbol n$ 在混合格式里
是对角修正 $A_{ff}\leftarrow A_{ff}+R_c(f)/|f|$（报告见 `MosicaFE/physics/conduction.py` 文档）。

验证（`examples/RT0-Conduction/结果_电流场.txt`）：

| 算例 | 结果 | 理论 |
| --- | --- | --- |
| 均质条形 $\sigma=2$ | $I=2,\ R=0.5,\ P=2$ | 同 |
| 逐单元电荷守恒 | $8.2\times10^{-15}$ | 0 |
| 串联分层 $\sigma=1/100$ | $R=0.5050000000$ | $0.5/1+0.5/100$ |
| 接触电阻 $R_c=0.5$ | $R=1.5$，跳变 $0.3333333333$ | $1+R_c/|f|$，$R_cI$ |
| $\sigma\leftrightarrow K$ 对拍 `DarcyProblem` | 逐位相同 | 同一解 |

### ③ Navier–Stokes：RT0 能不能用？

结论：**RT0 不能当速度**。RT0 场 $\boldsymbol u=\boldsymbol a+b\boldsymbol x$ 满足
$\nabla\boldsymbol u=b\boldsymbol I$、$\nabla\times\boldsymbol u\equiv0$，**无旋**，
表达不了剪切（Poiseuille、边界层、涡）；再配 $P_0$ 压力还会把速度逼成分片常数。
数值证据：随机 RT0 场 $\max|\nabla\times\boldsymbol u_h|=0.00\mathrm{e}{+}00$。

"RT0 放梯度 + $P_1$ 放速度 + $P_0$ 放压力"的三场格式也**数值奇异**
（零空间维数 $6\to14\to30$，条件数 $10^{33}$），因为 $(\boldsymbol u,\nabla\cdot\boldsymbol\tau)$
只看得见速度的单元均值。正确路线（应力型 PEERS、$\mathrm{BDM}_1$/RT1、$H(\mathrm{div})$-DG）
及其逐块装配公式见报告 §3–§4。

## 快速开始

```bash
git clone -b rt0-darcy-conduction https://github.com/<你的账号>/MosicaFE.git
cd MosicaFE
pip install -e MosicaFE            # 可选；也可以直接 import（examples 里已用 sys.path 处理）
```

```python
from MosicaFE import FESpace, Mesh, DarcyProblem, ConductionProblem

mesh = Mesh.rectangle(nx=32, ny=32)
bc = lambda mid, n: "D" if (mid[0] < 1e-12 or mid[0] > 1 - 1e-12) else "N"

# Darcy：q = (2.5, 0)
V = FESpace.rt0(mesh)
print(DarcyProblem(V, g=lambda x: 1.0 - x[:, 0], K=2.5, bc=bc).solve())

# 电流场：I = 2, R = 0.5
problem = ConductionProblem(V, g=lambda x: 1.0 - x[:, 0], sigma=2.0, bc=bc)
problem.solve()
print(problem.total_current(lambda m: m[0] > 1 - 1e-12))
```

跑算例（每个脚本都会写出 `结果_*.txt` 与 `figs/`）：

```bash
python examples/RT0-Darcy/darcy_mosicafe.py
python examples/RT0-Conduction/conduction_rt0.py
python examples/NS-RT0/ns_rt0_验证.py
```

测试：

```bash
python -m pytest MosicaFE/tests -q        # 109 passed
```

## RT0 的适用面（一句话判别）

RT0 适合"**主未知量是守恒通量、且通量是势的梯度**"的一阶系统
$A\boldsymbol q+\nabla u=0,\ \nabla\cdot\boldsymbol q=f$（$A$ 对称正定）：
渗流、导热、导电、静电场、压力方程、两相流的压力步、Stokes–Darcy 的 Darcy 侧，
以及任意速度场的守恒通量后处理。需要速度有旋（NS/Stokes）、需要 $H(\mathrm{curl})$
（三维电磁）或需要 $O(h^2)$ 以上精度的场合，应改用 $\mathrm{BDM}_k$/$\mathrm{RT}_k$
（$k\ge1$）或 Nédélec 元。

## 许可

继承上游 MosicaFE 的许可与致谢。
