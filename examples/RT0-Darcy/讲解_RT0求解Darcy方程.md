# 用 RT0 元求解 Darcy 方程：从泊松方程版本出发的完整路线

> 配套代码：`rt0_darcy_2d.py`（二维 RT0-P0 混合有限元，自带收敛阶与物理算例验证）
> 配套结果：`结果_darcy.txt`、`data_darcy.csv`、`fig_darcy.tex`
> 前置材料：`RT0收敛阶验证/rt0_2d.py`（泊松方程版）、`_rt0_notes.txt`

---

## 一、Darcy 方程是什么

### 1.1 从砂柱实验到微分方程

1856 年 Darcy 用砂柱做渗流实验，发现通过砂柱的体积流量与横截面积成正比、与水头差成正比、与柱长成反比：

$$Q = K\,A\,\frac{h_1-h_2}{L}.$$

把"差商"换成"梯度"，就得到**达西定律**（Darcy's law）的微分形式。以压力 $p$、渗透率 $K$、流体粘度 $\mu$ 写：

$$\boxed{\;\boldsymbol q = -\frac{K}{\mu}\bigl(\nabla p-\rho\,\boldsymbol g\bigr)\;}$$

- $\boldsymbol q$：**达西流速**（单位时间穿过单位**总**面积的体积，m/s），注意它不等于孔隙中的真实流速，后者是 $\boldsymbol q/\phi$（$\phi$ 为孔隙度）；
- $K$：渗透率张量（m²），工程上也用 Darcy（$1\,\mathrm{D}\approx 0.9869\times10^{-12}\,\mathrm{m^2}$），各向异性介质里 $K$ 是对称正定矩阵；
- $\mu$：流体动力粘度，$\rho\boldsymbol g$ 是重力项（写成水头 $h=p/(\rho g)+z$ 时可以吸收掉）。

再配上**质量守恒**（不可压缩、稳态、无体积源以外的源汇）

$$\nabla\cdot\boldsymbol q = f,$$

其中 $f$ 是单位体积的源汇强度（注水井取正、采油井取负）。把达西定律代入守恒方程，消去 $\boldsymbol q$ 就得到**压力方程**

$$-\nabla\cdot\Bigl(\frac{K}{\mu}\nabla p\Bigr)=f .$$

> **与泊松方程的关系**：若 $K/\mu=\kappa I$ 是常数，上式就是 $-\kappa\Delta p=f$，正是你已经在做的泊松方程。
> 所以 Darcy 问题可以看成"**变系数、可能高对比、可能各向异性的泊松问题**"。
> 真正的困难全部来自系数：$\kappa$ 可以跨几个数量级、可以在单元之间跳跃、可以是非对角张量。

### 1.2 为什么要写成混合形式

上面的压力方程是二阶方程，用 P1 有限元也能解。但对 Darcy 问题，**一阶（混合）系统**

$$\begin{cases}
K^{-1}\boldsymbol q+\nabla p=\boldsymbol 0,&\text{（达西定律）}\\[2pt]
\nabla\cdot\boldsymbol q=f,&\text{（质量守恒）}
\end{cases}$$

才是自然的写法，原因有四条：

1. **主要物理量是通量不是压力。** 地下水、油藏、CO₂ 封存里关心的是流量、突破时间、井产量；
2. **局部质量守恒。** 混合元在每个单元上精确满足 $\int_K\nabla\cdot\boldsymbol q_h=\int_K f$，而 $P_1$ Galerkin 只有整体守恒。做输运耦合时，不守恒的通量会让示踪物凭空产生或消失；
3. **通量是 $H(\mathrm{div})$ 协调的。** 跨单元法向通量单值连续，正是守恒格式需要的；
4. **inf-sup 稳定性与系数对比度无关。** RT0-P0 的 LBB 常数只依赖网格形状，不依赖 $K$ 的高对比，这对高对比介质特别关键。

### 1.3 弱形式

取试探空间

$$V=H(\mathrm{div};\Omega)=\{\boldsymbol v\in L^2(\Omega)^d:\ \nabla\cdot\boldsymbol v\in L^2(\Omega)\},\qquad Q=L^2(\Omega),$$

乘试探函数分部积分（$\int\nabla p\cdot\boldsymbol v=-\int p\,\nabla\cdot\boldsymbol v+\int_{\partial\Omega}p\,\boldsymbol v\cdot\boldsymbol n$），得到：求 $(\boldsymbol q,p)\in V\times Q$ 使

$$\begin{cases}
\displaystyle\int_\Omega K^{-1}\boldsymbol q\cdot\boldsymbol v\,\mathrm dx-\int_\Omega p\,\nabla\cdot\boldsymbol v\,\mathrm dx
=-\int_{\Gamma_D}g_D\,\boldsymbol v\cdot\boldsymbol n\,\mathrm ds,&\forall\boldsymbol v\in V_0,\\[6pt]
\displaystyle\int_\Omega(\nabla\cdot\boldsymbol q)\,w\,\mathrm dx=\int_\Omega f w\,\mathrm dx,&\forall w\in Q,
\end{cases}$$

其中 $V_0=\{\boldsymbol v\in V:\boldsymbol v\cdot\boldsymbol n=0\ \text{on}\ \Gamma_N\}$。

### 1.4 边界条件：**本质/自然的归属和泊松方程正好相反**（最容易写错的地方）

| 边界条件 | 在混合形式中的地位 | 怎么加 |
|---|---|---|
| 压力 $p=g_D$（Dirichlet） | **自然**（natural） | 只进第一个方程右端 $-{\langle g_D,\boldsymbol v\cdot\boldsymbol n\rangle}_{\Gamma_D}$ |
| 流量 $\boldsymbol q\cdot\boldsymbol n=g_N$ | **本质**（essential） | 落在通量空间 $V_h$ 上：无流动置零自由度；给定流量把该自由度取成已知值并搬到右端 |

也就是说：

- $p$ 是**没有法向迹**的变量（$L^2$ 里没有边界值），它在边界上的数据只能"弱"地进右端——这跟泊松方程里 $u=0$ 要强加在节点上是反的；
- $\boldsymbol q\cdot\boldsymbol n$ 在 $H(\mathrm{div})$ 里**有迹**，所以它是可以强加的自由度——这跟泊松方程里 $\partial u/\partial n$ 是自然边界条件也是反的。

> **不要用"把边界单元的压力自由度置零"来施加 Dirichlet！** 那等于在解一个 Neumann 问题，而且会把边界单元的通量一起压成零（你的笔记旧代码里踩过这个坑）。

两个补充：

- **相容性条件**：全 $\Gamma_N$ 且给定流量时，必须有 $\int_\Omega f+\int_{\Gamma_N}g_N=0$，否则无解；
- **纯 Neumann 时的奇异性**：此时 $p$ 只确定到相差一个常数（$Q_h$ 里含常数），必须额外固定一个参考压力（或约束 $\int p_h=0$），否则 $S=B^{\mathsf T}A^{-1}B$ 奇异。

---

## 二、RT0 元为什么适合 Darcy

最低阶 Raviart–Thomas 空间（三角形）

$$\mathrm{RT}_0(K)=\{\boldsymbol v(\boldsymbol x)=\boldsymbol a+b\,\boldsymbol x:\ \boldsymbol a\in\mathbb R^2,\ b\in\mathbb R\},$$

- **自由度**：每条边上的法向通量 $\ell_i(\boldsymbol v)=\displaystyle\int_{e_i}\boldsymbol v\cdot\boldsymbol n_i\,\mathrm ds$；
- **基函数**：$\boldsymbol\varphi_i(\boldsymbol x)=\dfrac{\boldsymbol x-\boldsymbol a_i}{2|K|}$（$\boldsymbol a_i$ 是第 $i$ 条边相对的顶点），满足 $\int_{e_j}\boldsymbol\varphi_i\cdot\boldsymbol n_j\,\mathrm ds=\delta_{ij}$；
- **散度是常数**：$\nabla\cdot\boldsymbol\varphi_i=\dfrac{1}{|K|}$，于是 $B_{iK}=\int_K\nabla\cdot\boldsymbol\varphi_i\,\mathrm dx=\pm1$，**组装 $B$ 不需要任何求积**；
- **全局自由度**：每条**内部边只有一个**自由度 $\Rightarrow$ 跨单元法向通量单值 $\Rightarrow V_h\subset H(\mathrm{div};\Omega)$；
- 压力用 $P_0$（分片常数），每个单元一个自由度。

稳定性与收敛性（$p\in H^1$，$\boldsymbol q\in H(\mathrm{div})\cap H^1$）：

$$\|\boldsymbol q-\boldsymbol q_h\|_{L^2}+\|\nabla\cdot(\boldsymbol q-\boldsymbol q_h)\|_{L^2}+\|p-p_h\|_{L^2}\le C\,h\,(\cdots),$$

且 $C$ 与 $K$ 的对比度无关（只依赖网格正则性）。**只有一阶**——因为 $P_0$ 压力和 $\mathrm{RT}_0$ 通量都是最低阶。

---

## 三、离散系统

取基展开 $\boldsymbol q_h=\sum_i q_i\boldsymbol\varphi_i,\ p_h=\sum_K p_K\psi_K$（$\psi_K$ 是单元 $K$ 的指示函数），得到鞍点系统

$$\begin{pmatrix}A & -B\\ B^{\mathsf T} & 0\end{pmatrix}
\begin{pmatrix}\boldsymbol q\\ \boldsymbol p\end{pmatrix}=
\begin{pmatrix}\boldsymbol b\\ \boldsymbol F\end{pmatrix},$$

$$\boxed{\,A_{ij}=\int_\Omega \bigl(K^{-1}\boldsymbol\varphi_i\bigr)\cdot\boldsymbol\varphi_j\,\mathrm dx\,},\qquad
B_{iK}=\int_K\nabla\cdot\boldsymbol\varphi_i\,\mathrm dx=\pm1,\qquad
F_K=\int_K f\,\mathrm dx,\qquad
b_i=-\int_{\Gamma_D}g_D\,\boldsymbol\varphi_i\cdot\boldsymbol n\,\mathrm ds .$$

矩阵性质：

- $A$ 对称正定（$K$ 对称正定 $\Rightarrow$ 加权质量矩阵 SPD），**可逆**；
- 用 $A$ 消去 $\boldsymbol q$：$\boldsymbol q=A^{-1}(B\boldsymbol p+\boldsymbol b)$，代入第二式得 **Schur 补**方程
  $$\underbrace{B^{\mathsf T}A^{-1}B}_{=S}\ \boldsymbol p=\boldsymbol F-B^{\mathsf T}A^{-1}\boldsymbol b,$$
  $S$ 对称正定（Dirichlet 情形），可用 CG；$S$ 的物理意义是**单元中心（cell-centered）的扩散算子**。
- **与有限体积的关系**：$S=B^{\mathsf T}A^{-1}B$ 恰好是 $B^{\mathsf T}A^{-1}B$ 型的多点通量格式；当网格关于 $K$ 正交（$K$ 各向同性、或 $K$ 的主方向与网格边平行）时，它退化为**两点通量近似（TPFA）**，边上的传导率自动取**调和平均**——这是混合元"免费"给出的结果，不需要 MPFA 那套复杂构造。

---

## 四、你实际要改的代码：只有一处

你现有的 `rt0_2d.py` 结构（网格 → `build_edges` → `assemble` → `solve_mixed` → 误差）可以**原样保留**。数学上唯一的差别是组装 $A$ 时被积函数多乘一个 $K^{-1}$：

```python
# ---------- 泊松版（rt0_2d.py）----------
phi = [(X - P[i]) / (2.0 * area) for i in range(3)]
M = np.array([[np.sum(w * (phi[i] * phi[j]).sum(axis=1))     # ∫ φ_i·φ_j
               for j in range(3)] for i in range(3)])

# ---------- Darcy 版（rt0_darcy_2d.py）----------
phi = [(X - P[i]) / (2.0 * area) for i in range(3)]
M = np.array([[np.sum(w * Kinv_dot(K, X, phi[i], phi[j]))    # ∫ (K^{-1}φ_i)·φ_j
               for j in range(3)] for i in range(3)])
```

其中（两种写法都支持：标量渗透率或张量渗透率）

```python
def Kinv_dot(K, X, a, b):
    """逐点计算 aᵀ K(x)^{-1} b，a,b: (nq,2) → (nq,)"""
    Kx = K(X)
    if Kx.ndim == 1:                       # 各向同性标量渗透率 k(x)
        return (1.0 / Kx) * np.sum(a * b, axis=1)
    return np.einsum("qa,qab,qb->q", a, np.linalg.inv(Kx), b)
```

**除此外完全不变**：$B$ 还是 $\pm1$、鞍点结构不变、`solve_mixed`（Schur + CG）不变、RT0 基函数与 `elem_sign` 约定不变。

但**边界条件处理要升级**，因为 Darcy 多了一类本质边界条件：

```python
def boundary_data(nodes, boundary, bc):
    """bc(mid, n_out) 返回 'D'（压力）| 'N'（不透水）| ('flux', gN)（给定流量）"""
    dir_edges, known = [], {}
    for (k, n_out, L, n_glob, u0, u1) in boundary:
        kind = bc(0.5*(nodes[u0]+nodes[u1]), n_out)
        if kind == 'D':
            dir_edges.append(...)          # 进第一个方程右端 -<g_D, v·n>
        elif kind == 'N':
            known[k] = 0.0                 # 不透水：自由度置零
        else:
            c = round(n_out·n_glob)
            known[k] = c * ∫_e gN ds       # 给定流量：自由度取已知值
    return dir_edges, known
```

再把已知/置零的自由度从方程组里删掉（等价于限制到 $V_{h0}$）：

```python
A_ff = A[free][:, free]
rhs_f = rhs_q[free] - A[free][:, known_dofs] @ qc    # 已知列搬到右端
```

---

## 五、实现清单与常见坑

1. **内部边必须共享自由度**（每条内部边一个未知量）。
   - 你在泊松版里已经验证过：逐元独立编号（$3N_K$ 个自由度）会让方程组逐元解耦，误差不随 $h$ 下降（实测阶 0.000）。
   - 这个要求在 Darcy 里更要命：它同时破坏 $H(\mathrm{div})$ 协调性与**法向通量连续**这一物理条件。
2. **边界条件不要搞反**：压力 Dirichlet 走右端（自然），流量走自由度（本质）；别用"边界单元压力置零"。
3. **系数跳跃时不要用"光滑解 + 跳跃系数"当制造解**。
   因为 $\nabla\cdot(K\nabla p)=K\Delta p+[K]\,\delta(x-x_I)\,\partial_x p$ 会多出一个**界面 Dirac 源**，这个 $f$ 不是 $L^2$ 函数。用它检验，收敛阶会掉到 0（我第一次跑就是这么错的）。
   正确做法：让精确解满足界面条件 $[\![K\partial p/\partial n]\!]=0$（法向通量连续），例如分层介质里的**分段线性解**（见 `case_layer`）。
4. **求积阶数**：$K$ 分片常数时被积函数是二次多项式，Duffy + 3 点 Gauss 已精确；$K$ 光滑时要用更高阶求积（本代码用 4 点 $\times$ 4 点，共 16 点），否则质量矩阵的误差会污染收敛阶。
5. **高对比导致病态**：$K$ 跨 $10^6$ 量级时 $A$ 的条件数同步放大。实务做法：用参考渗透率 $K_0$ 无量纲化（$\tilde K=K/K_0$），并对 Schur 系统加预条件（例如 $B^{\mathsf T}\mathrm{diag}(A)^{-1}B$ 型对角预条件、AMG）；否则 CG 迭代步数会爆炸。
6. **对称性**：$A$ 对称正定要求 $K$ 对称正定。若 $K$ 非对称（对流占优、裂隙介质），$A$ 不再对称，鞍点系统要用 GMRES/BiCGStab，且通常需要迎风处理。
7. **纯 Neumann 要固定参考压力**（去掉 $p$ 的常数零空间），并检查相容性 $\int f+\int_{\Gamma_N}g_N=0$。
8. **三维**：你的 `rt0_3d.py` 同样只改 $A$（$\boldsymbol\varphi_i=(\boldsymbol x-\boldsymbol a_i)/(3|K|)$，$\nabla\cdot\boldsymbol\varphi_i=1/|K|$，四面体 4 个面自由度），其它完全不变。

---

## 六、验证清单（`rt0_darcy_2d.py` 全部自动做了）

| 检验 | 手段 | 期望结果 |
|---|---|---|
| 退化一致性 | $K=I$ 与 `rt0_2d.py` 的泊松结果逐位对照 | 误差完全相同 |
| 收敛阶 | 4 个算例（$K=I$、常值各向异性、光滑变系数、高对比分层）加密网格 | $\|p-p_h\|,\|\boldsymbol q-\boldsymbol q_h\|=O(h)$ |
| 局部质量守恒 | $\max_K\|\nabla\cdot\boldsymbol q_h-\bar f_K\|$ | 线性求解器容差 / 单元面积（方程本身逐单元精确） |
| 达西定律 | 达西柱实验：出口流量 vs $Q=\kappa A\Delta p/L$ | 精确到机器零（常向量场 $\subset\mathrm{RT}_0$） |
| 给定流量边界 | 左侧 $q\cdot n=-1$、右侧定压、上下不透水 | $\boldsymbol q_h=(1,0)$ 精确 |
| 界面通量连续 | 分层介质界面上的法向通量 | 单值、等于调和平均等效通量 $2/(1/\kappa_L+1/\kappa_R)$ |
| 鞍点求解自检 | Schur+CG vs 直接解 `spsolve` | 差 $\sim10^{-13}$ |

---

## 七、实测结果

`python rt0_darcy_2d.py`（本机约 37 s，含 n=64 的三次 Schur+CG 与一次 2 万自由度直接解）。

### 7.1 收敛阶（$\|p-p_h\|_{L^2}$ / $\|q-q_h\|_{L^2}$）

| $h$ | $K=I$：p / q | 常值各向异性：p / q | 光滑变系数：p / q | 高对比分层：p / q |
|---|---|---|---|---|
| 0.2500 | 1.287e-01 / 5.019e-01 | 1.290e-01 / 1.430e+00 | 1.287e-01 / 7.800e-01 | 8.251e-02 / 2.19e-14 |
| 0.1250 | 6.517e-02 / 2.516e-01 | 6.522e-02 / 7.135e-01 | 6.517e-02 / 3.922e-01 | 4.126e-02 / 4.25e-15 |
| 0.0625 | 3.269e-02 / 1.259e-01 | 3.270e-02 / 3.563e-01 | 3.269e-02 / 1.964e-01 | 2.063e-02 / 2.67e-14 |
| 0.0312 | 1.636e-02 / 6.295e-02 | 1.636e-02 / 1.781e-01 | 1.636e-02 / 9.822e-02 | 1.031e-02 / 1.18e-14 |
| 0.0156 | 8.181e-03 / 3.148e-02 | 8.181e-03 / 8.904e-02 | 8.181e-03 / 4.912e-02 | 5.157e-03 / 6.83e-14 |
| **拟合阶** | **1.000 / 1.000** | **1.000 / 1.000** | **1.000 / 1.000** | **1.000 / 机零** |

- 三个光滑系数算例都是干净的 $O(h)$，与 RT0-P0 的理论阶一致；
- 分层算例的**通量是机器零误差**：精确解 $\boldsymbol q=(Q,0)$ 是常向量，落在 $\mathrm{RT}_0$ 空间内，有限元把它**精确**表示出来了（压力的分段线性解只能被 $P_0$ 以 $O(h)$ 逼近）。

### 7.2 与泊松版本的交叉核对（$K=I$）

| $h$ | 本程序 $\|p-p_h\|$ | `rt0_2d.py` 的 $\|u-u_h\|$ | 本程序 $\|q-q_h\|$ | `rt0_2d.py` 的 $\|q-q_h\|$ |
|---|---|---|---|---|
| 0.2500 | 1.2868e-01 | 1.2868e-01 | 5.0190e-01 | 5.0190e-01 |
| 0.1250 | 6.5174e-02 | 6.5174e-02 | 2.5164e-01 | 2.5164e-01 |
| 0.0625 | 3.2690e-02 | 3.2690e-02 | 1.2589e-01 | 1.2589e-01 |
| 0.0312 | 1.6358e-02 | 1.6358e-02 | 6.2954e-02 | 6.2954e-02 |
| 0.0156 | 8.1807e-03 | 8.1807e-03 | 3.1478e-02 | 3.1478e-02 |

（数值与 `RT0收敛阶验证/结果_2d.txt` 逐位相同——说明推广没有破坏原有实现。）

### 7.3 Darcy 专属的物理检验

| 检验 | 结果 |
|---|---|
| 达西柱实验：出口流量（$K=2.5$，$\Delta p=1$，$A=1$，$L=1$） | $Q_h=2.5000000000$，理论值 $2.5$，相对误差 $6.4\times10^{-13}$ |
| 达西柱实验：通量误差 | $3.0\times10^{-11}$（常通量场精确可表示） |
| 给定流量边界：左边界总流量 | $\int q\cdot n\,\mathrm ds=-1.0000000000$（给定 $-1$，误差 $1.1\times10^{-16}$） |
| 给定流量边界：通量误差 | $2.0\times10^{-13}$ |
| 分层介质界面通量密度 | $1.9801980198$；调和平均 $2/(1/1+1/100)=1.9802$ ✓；算术平均 $50.5$ ✗ |
| 鞍点求解自检（Schur+CG vs 直接解） | $\max|\Delta q|\sim10^{-13}$，$\max|\Delta p|\sim10^{-14}$ |
| 局部守恒残差 $\max_K\|\nabla\cdot\boldsymbol q_h-\bar f_K\|$ | $10^{-11}\sim10^{-6}$（= 线性求解器容差 / 单元面积） |

收敛图见 `fig_darcy.pdf`（由 `fig_darcy.tex` 用 xelatex 编译，数据在 `data_darcy.csv`）。

---

## 八、可以继续做的方向

1. **自适应**：RT0-P0 的后验误差估计子（基于 $\|\boldsymbol q_h+K\nabla p_h\|$ 的残差型估计子）+ 网格自适应，高对比介质上收益明显；
2. **杂交混合元（hybrid RT0）**：把边通量在界面上的连续性用 Lagrange 乘子放松，鞍点系统变成**对称正定**系统，并可直接静力凝聚出单元中心的压力方程；
3. **耦合输运**：$\partial_t c+\nabla\cdot(\boldsymbol qc)=0$ 用 DG 或有限体积离散，直接吃 $\boldsymbol q_h$ 的守恒性；
4. **Stokes–Darcy 耦合**：自由流用 Taylor–Hood/RT-P0，多孔介质用 RT0-P0，界面条件（Beavers–Joseph）正好用通量自由度写；
5. **高阶元**：$\mathrm{RT}_1$–$P_1^{\mathrm{disc}}$ 得到 $O(h^2)$ 通量；BDM/BDFM 家族；
6. **非结构网格与一般各向异性**：此时混合元给出的多点格式比 TPFA 更可靠（TPFA 在非 $K$-正交网格上不收敛）。

---

## 九、附：Darcy 求解器并入 MosicaFE（库内实现）

> **状态更新**：Darcy 求解器已经从"库外扩展示例"**正式并入 MosicaFE**：
>
> | 位置 | 内容 |
> |---|---|
> | `MosicaFE/spaces/rt0.py` | `rt0_permeability_matrices(K, …)`、`rt0_weighted_mass_matrices(…)、`RT0Space.weighted_flux_mass_matrix(K, mode=…)` |
> | `MosicaFE/physics/darcy.py` | `DarcyProblem`（注册名 `darcy`）：鞍点组装、本质/自然边界条件、Schur 求解、纯 Neumann 处理 |
> | `MosicaFE/api.py` | `solve_darcy(mesh, f=…, K=…, bc=…)` 一行式接口 |
> | `MosicaFE/tests/test_physics_solvers.py` | 10 个 Darcy 测试（对拍、加权质量矩阵、各向异性收敛、调和平均、给定流量、纯 Neumann） |
>
> 因此 `from MosicaFE import DarcyProblem` 或 `MosicaFE.run("darcy", …)` 都可直接使用；
> `RT0-Darcy/darcy_mosicafe.py` 现在只是"用库 + 验证 + 出图"的演示脚本
> （结果：`结果_darcy_mosicafe.txt`）。

下面 9.1–9.5 保留的是**当时推导与踩坑的记录**——它解释了库内那 200 行代码为什么这样写，
也是把一个混合元问题接进通用库的完整思路。

### 9.1 MosicaFE 把 RT0 拆成了三层

| 文件 | 职责 |
|---|---|
| `core/facets.py` | 面拓扑：全局面编号、共享面、外法向符号 `signs[c,i]=±1`、`normals[f]`（**模长 = 面测度**）——$H(\mathrm{div})$ 协调性的基础 |
| `spaces/rt0.py` | 显式基 $\varphi_i=(x-a_i)/(d\lvert K\rvert)$、**精确闭式**局部质量矩阵 `rt0_mass_matrices`、自由度布局、三个块原语 `flux_mass_matrix / divergence_matrix / pressure_load_vector`、重构 `evaluate_flux` |
| `physics/mixed_poisson.py` | 只做块拼装 $\begin{pmatrix}M&-B^{\mathsf T}\\B&0\end{pmatrix}$ 与求解（Schur 凝聚 / direct / Krylov） |

所以"写一个新混合问题"= **只写物理层的组装**，这正是 README 里"扩展到其它 PDE"和"要写别的混合问题（Darcy、Stokes 的同阶配对……）时不必重写单元循环"的意思。

### 9.2 从 Poisson 到 Darcy 要补的三件事

**(1) 通量质量矩阵加权** $A_{ij}=\int (K^{-1}\varphi_i)\cdot\varphi_j$

关键洞察：库里的闭式公式只用到"对每个自变量线性"，把点积换成 $u^{\mathsf T}Wv$ **逐字成立**：

$$A_{ij}=\frac{S_i^{\mathsf T}WS_j+\sum_k(a_k-a_i)^{\mathsf T}W(a_k-a_j)}{d^2(d+1)(d+2)\lvert K\rvert},\qquad S_i=\sum_k(a_k-a_i),\ W=K^{-1}.$$

- 分片常数 $K$：上式**精确**（常数 $W$ 提出积分号）；
- 光滑 $K$：用求积点逐点取 $K(x_q)^{-1}$（`cell_quadrature` + `eval_flux_basis`），或直接取形心值（两者差 $O(h^2)$，实测 $8.5\times10^{-3}$，都不影响 $O(h)$ 通量阶）。

**(2) 本质边界条件**（`MixedPoissonProblem` 只支持齐次压力 Dirichlet）

不透水 $\Phi_f=0$；给定流量 $\Phi_f=\mathrm{sign}\cdot\int_f g_N\,\mathrm ds$（$\Phi_f=\int_f q\cdot n_f\,\mathrm ds$ 是库的**全局**面自由度）。消去要做**三件事**：

```python
colA = A[:, f]                 # (1) A 的已知列 → 第一个方程右端
flux_rhs -= colA * val
colB = B[:, f]                 # (2) B 的已知列 → 第二个方程右端
F_eff    -= colB * val
A[f,:] = 0; A[:,f] = 0; A[f,f] = 1; flux_rhs[f] = val   # (3) 固定 Φ_f
B[:, f] = 0                                            # (3) ★ 必须切断 -Bᵀ 耦合
```

> **★ 这里有个坑（我第一次就是这么错的）**：鞍点矩阵第一个块行是 $[A,\;-B^{\mathsf T}]$。只把 $A$ 的已知行/列换成单位行是**不够**的——$B$ 的该列还在，那"约束行"实际是
> $$\Phi_f-\sum_K B_{Kf}P_K=\text{已知值},$$
> 边界条件根本没生效（达西柱实验算出 $q_h=(2.26,-0.27)$ 而不是 $(2.5,0)$）。必须同时把 $B$ 的该列清零，并把 $B_c^{\mathsf T}q_c$ 搬到压力方程右端。
> 另外：守恒检查要用**物理**载荷 $F$（含给定流量），而 Schur 的右端要用消去后的 $F_{\text{eff}}=F-B_c^{\mathsf T}q_c$。

**(3) 非齐次压力 Dirichlet 的右端项**：面 $f$ 上 $\varphi_f\cdot n_f=1/\lvert f\rvert$，于是

$$\int_f g_D\,\varphi_f\cdot n_{\text{out}}\,\mathrm ds=\frac{\mathrm{sign}}{\lvert f\rvert}\int_f g_D\,\mathrm ds
\quad\Longrightarrow\quad b_f=-\frac{\mathrm{sign}}{\lvert f\rvert}\int_f g_D\,\mathrm ds .$$

### 9.3 骨架（库外约 30 行，完整版见脚本）

```python
@register_problem("darcy")                       # 注册进库的 PDE 表
class DarcyProblem(MixedPoissonProblem):         # 继承库里的混合 Poisson
    def _check_boundary_data(self):              # 库只许齐次 Dirichlet，这里放开
        return
    def assemble(self):
        A = weighted_flux_mass_matrix(V, self.K)  # ← 唯一的新矩阵（带 K^{-1}）
        B = V.divergence_matrix()                 # 复用库：±1，精确
        F = V.pressure_load_vector(self.f)        # 复用库：∫_K f
        ...                                       # 边界分类 + 上面的"三件事"
        return bmat([[A, -B.T], [B, None]]), np.concatenate([b_phi, F_eff])
    def _solve_schur(self):
        ...                                       # 库的静力凝聚把 b_phi 当成 0，要带上
```

于是 `create_problem("darcy", V, f=..., K=...)` 就能用；`extract / compute_errors / check_conservation / flux_at_points` 全部继承自库。

### 9.4 库现有约定的绕开方式（都不改库）

| 库的限制 | 绕开方式 |
|---|---|
| `MixedPoissonProblem._check_boundary_data` 拒绝 $g\ne0$ | 子类覆盖为 no-op |
| `RT0Space.boundary_dofs()` 返回空数组（自然边界条件） | 本质条件在物理层自己消去 |
| `stiffness_matrix/mass_matrix/load_vector` 对 RT0 抛 `NotImplementedError` | 用混合原语 `flux_mass_matrix/divergence_matrix/pressure_load_vector` |
| `_solve_schur` 假定 $b_\Phi=0$ | 覆盖，带上 Dirichlet 右端项 |
| Schur 用**稠密** $S$（内存 $O(n_{\text{flux}}\cdot n_{\text{cells}})$，$n=64$ 时约 800 MB） | 脚本加了 `method="schur_cg"`：$M$ 一次稀疏 LU + $S$ 上的 CG，只存稀疏量 |
| 目录名 `MosicaFE-master` 不能 `import MosicaFE` | 脚本用 `importlib` 按路径注册包名；也可 `pip install -e .` 或把目录改名 |
| import 会在库里生成 `__pycache__` | 脚本开头 `sys.dont_write_bytecode = True`（并已清掉此前生成的那些） |

### 9.5 实测（`python darcy_mosicafe.py`）

| 检验 | 结果 |
|---|---|
| $K=I$ 与库 `MixedPoissonProblem` 对拍 | $\max\lvert\Delta P\rvert=\max\lvert\Delta Q\rvert=\mathbf{0}$（逐位相同） |
| 加权质量矩阵 $K=I$ vs 库的 `flux_mass_matrix()` | 相对偏差 $0$ |
| 常值各向异性：闭式(cell) vs 求积(quad) | $1.2\times10^{-15}$（闭式对分片常数 $K$ 精确） |
| 2D 各向异性 $K=\begin{pmatrix}3&1\\1&2\end{pmatrix}$，$n=4\ldots64$ | 压力阶 $0.999$，通量阶 $1.000$；误差值与手写版 `rt0_darcy_2d.py` **完全一致** |
| 3D 四面体 $K=\mathrm{diag}(1,1,4)$，$n=4,8,16$（$n{=}16$ 为 24576 单元） | 压力阶 $0.992$，通量阶 $0.982$（粗网格上偏低的 $0.93$ 只是尚未进入渐近区） |
| 达西柱实验 | $\sum Q=2.5000000000$（达西定律 $2.5$，误差 $5.3\times10^{-16}$），`flux_at_points` 给出 $(2.5,0)$ |
| 给定流量边界 | $\int q\cdot n\,\mathrm ds=-1.0000000000$，$\lVert q-q_h\rVert=2.8\times10^{-15}$ |
| 分层高对比（界面通量**密度**） | $1.9801980198$ = 调和平均 $2/(1/\kappa_L+1/\kappa_R)$，误差 $\sim10^{-15}$ |
| `problem.check_conservation()` | 逐单元 $10^{-15}\sim10^{-14}$ |

两个诊断上的小坑（与库无关，但很容易看错）：

- **界面是内部面**：扫 `topo.counts == 1` 的边界面会一个都找不到，要扫 `V.n_flux` 全部面（用 `V.facet_vertices(f)` + `topo.measure()`）；
- RT0 自由度 $\Phi_f=\int_f q\cdot n\,\mathrm ds$ 是**通量积分**，要得到通量**密度**（如 $q_x$）必须除以面测度 $\lvert f\rvert$（第一次忘了除，界面值显示成 $0.1238$ 而不是 $1.9802$）。

### 9.6 两条路怎么选（并入后的现状）

| | 手写 `rt0_darcy_2d.py` | 库内置 `DarcyProblem` |
|---|---|---|
| 适合 | 摸清算法细节、教学、完全可控的验证 | 复用拓扑/闭式质量矩阵/误差/可视化，快速搭模型 |
| 网格 | 自己的结构化矩形网格 | 库的 `Mesh.rectangle / box(tet) / Voronoi / pentagon` |
| 后处理 | 自己写（pgfplots 出图） | 库原生 `mixed_error_norms / flux_at_points / plot_flux_2d` |
| 代价 | 所有细节自己写 | 无（本质边界条件、非齐次 Dirichlet、纯 Neumann 都已内置） |

两条路对同一算例（$K=\begin{pmatrix}3&1\\1&2\end{pmatrix}$）给出的误差数值完全一致，可以互为校验。

### 9.7 并入时补的两处"库级"细节

1. **纯 Neumann 的奇异处理**。没有压力 Dirichlet 时 $S=B^{\mathsf T}A^{-1}B$ 有常数零空间。
   库内做法：`"schur"/"schur_cg"` 用最小二乘求解并把压力平移到零均值；
   `"direct"/"minres"` 会直接抛 `ValueError` 并给出提示（否则会静默解一个奇异矩阵）。
2. **渗透率合法性检查**。`rt0_permeability_matrices` 统一把标量 / 对角 / 张量 / 可调用场
   转成 `(nq,d,d)`，并检查**对称正定**（最小特征值 > 0）与有限性，报错信息直接指出是哪一项。

另外，`DarcyProblem` 提供 `total_flux(facets)` 便利函数（例如求出口流量：
`problem.total_flux(右边界的面编号)` 或 `total_flux("boundary")`）。

---

## 十、误差可视化（`darcy_plot.py`）

### 10.1 为什么自己写一个绘图模块

本机 `matplotlib 3.7` 与 `numpy 2.x` ABI 不匹配，`savefig` 直接抛
`ValueError: object __array__ method not producing an array`
——所以**库自带的 `plot_error / plot_pressure_2d / plot_flux_2d` 在这台机器上都跑不了**。

`RT0-Darcy/darcy_plot.py` 只用 **Pillow + numpy** 自己实现了 colormap、坐标轴、颜色条、
对数坐标与箭头，因此一定能出图（中文用微软雅黑渲染）。想用回库里的绘图，
把 matplotlib 升到 ≥3.8（支持 numpy 2）即可，例如 `pip install -U "matplotlib>=3.8"`。

| 函数 | 画什么 |
|---|---|
| `cell_heatmap(nodes, cells, values, path, …)` | **逐单元标量热图**：压力/通量误差场、渗透率场（P0 数据的自然画法；`log=True` 支持跨数量级） |
| `mesh_quiver(nodes, cells, pts, vec, …)` | **通量箭头图**；给 `vectors_extra=精确解` 时半透明叠加，重合处呈紫色、分离处一眼可见 |
| `profile_plot(series, path, …)` | **一维剖面折线**：精确解 vs 数值解，可标材料界面 `vlines=[0.5]` |
| `p0_step_along_line(nodes, cells, P, y, h)` | 把 P0 分片常数解拼成**阶梯折线**，专门用来画剖面 |
| `loglog_convergence(hs, series, path, …)` | **log-log 收敛曲线** + 参考斜率（支持 2D/3D 各自一套 $h$、虚线区分通量） |
| `hstack(paths, out, labels=…)` | 多图拼成一行（子图标签 (a)(b)(c)） |

### 10.2 逐单元误差怎么算（可视化真正需要的量）

整体 $L^2$ 误差只有一个数，看不出误差**在哪里**。要画图就得用逐单元误差

$$e_p^K=\Bigl(\frac{1}{\lvert K\rvert}\int_K(p-p_h)^2\Bigr)^{1/2},\qquad
e_q^K=\Bigl(\frac{1}{\lvert K\rvert}\int_K\lvert\boldsymbol q-\boldsymbol q_h\rvert^2\Bigr)^{1/2},$$

两个脚本里分别是 `cell_errors(...)`（手写版用 Duffy 高精度求积）和 `cell_errors_2d(...)`
（MosicaFE 版用库的 `cell_quadrature`）。

### 10.3 出的图（`figs/standalone/` 与 `figs/mosicafe/`）

| 图 | 内容 | 看到什么 |
|---|---|---|
| `err_aniso.png` | (a) $\lvert p-p_h\rvert$ (b) $\lvert\boldsymbol q-\boldsymbol q_h\rvert$（对数） (c) $\boldsymbol q_h$ vs $\boldsymbol q$ | 误差在区域上**很不均匀**（$\max/\min\approx29$）：角点与边界最大，中心最小 |
| `err_layer.png` | (a) 渗透率场 (b) 压力误差 (c) 剖面 $p(x,0.5)$ | 左半（$\kappa=1$，斜率大）误差 $1.46\times10^{-2}$，右半（$\kappa=100$）只有 $1.5\times10^{-4}$；剖面上精确解在 $x=0.5$ 有**折点**、$p_h$ 是**阶梯** |
| `flux_aniso.png` | $\boldsymbol q_h$（蓝）叠加 $\boldsymbol q$（红） | 全场几乎都是紫色 ⇒ 二者处处重合 |
| `flux_column.png` | 达西柱实验的 $\boldsymbol q_h$ | 全部水平、等长 = $(\kappa,0)$，就是达西定律的图示 |
| `conv_2d.png` / `conv_2d3d.png` | log-log 收敛曲线 + 参考斜率 | 压力/通量曲线都与 $h^1$ 平行；2D、3D 画在一起 |
| `err_summary.png` | 柱实验通量 + 收敛曲线 | 汇报用的一张总图 |

### 10.4 顺手得到的一个交叉验证

两套**完全独立**的实现（手写组装 `rt0_darcy_2d.py` vs 调用 MosicaFE 的 `darcy_mosicafe.py`）
产出的 9 张图中，**7 张字节完全相同**（`err_layer.png`、`layer_profile.png`、
`flux_aniso.png`、`err_aniso_p.png` …）。剩下 2 张（各向异性通量误差图）只在末位有差，
原因是逐单元误差积分用的求积规则不同（Duffy 144 点 vs 库里 6 点三角形规则），
整体 $L^2$ 误差两者是逐位相同的。

### 10.5 用法

```python
import darcy_plot as dp                       # 放在 RT0-Darcy/ 下，和脚本同级

# 逐单元误差热图（P0 场）
dp.cell_heatmap(nodes, cells, ep, "err_p.png",
                title="压力误差 |p-p_h|", cbar_label="L2/单元", cmap="coolwarm")

# 通量箭头：数值解 + 精确解半透明叠加
dp.mesh_quiver(nodes, cells, centroids, q_h, "flux.png",
               vectors_extra=q_exact, label="RT0 数值解", label_extra="精确解")

# 剖面：精确解（有折点）vs P0 阶梯
sx, sy = dp.p0_step_along_line(nodes, cells, P, y_target=0.5, h=1/32)
dp.profile_plot([dict(label="精确解", x=xs, y=ps, color=(214,39,40)),
                 dict(label="p_h", x=sx, y=sy, color=(31,119,180), dash=True)],
                "profile.png", vlines=[0.5], legend_pos="tr")

# 多图拼一行
dp.hstack(["err_p.png", "err_q.png", "profile.png"], "panel.png",
          labels=["(a) 压力误差", "(b) 通量误差", "(c) 剖面"])
```

两个脚本运行时会自动把图写进各自的 `figs/` 子目录（互不覆盖），并在终端与
`结果_darcy*.txt` 里列出文件名。
