# Lagrange 有限元矩阵组装优化

## 1. 优化概述

本次工作基于 `wen` 分支，在个人开发分支 `gxl` 中优化 Lagrange 有限元的质量矩阵与刚度矩阵组装。`wen` 作为优化前实现的参考，优化后的核心实现位于 `spaces/lagrange.py`，验证及性能对比位于 `examples/check_lagrange_assembly.py`。

优化目标是减少 Python 层循环和临时数组分配，在不改变有限元离散形式、积分规则、自由度编号及对外接口的前提下，提高组装效率。通用基类、其他有限元空间及载荷向量组装路径不在本次修改范围内。

| 修改位置 | 优化内容 | 目的 |
| --- | --- | --- |
| `local_stiffness` | 张量单元采用批量 `np.linalg.solve` | 避免逐积分点梯度变换 |
| `local_mass` | 单纯形单元复用参考质量矩阵 | 避免逐单元重复积分 |
| `_reference_mass` | 利用 `_basis_cache` 缓存参考矩阵 | 减少重复计算 |
| `_assemble_lagrange` | COO 索引预分配、单纯形分块组装 | 减少 Python 循环和临时数组 |
| `stiffness_matrix`、`mass_matrix` | 调用专用组装入口 | 保持公共接口不变 |
| `from_triplets` | 统一构造稀疏矩阵 | 累加共享自由度的局部贡献 |

## 2. 数学基础与优化实现

### 2.1 参考单元映射与梯度变换

设参考单元为 $\widehat K$、物理单元为 $K$，映射 $x=F_K(\xi)$ 的 Jacobian 为 $J$。体积元和基函数梯度满足

$$
\mathrm dx=|\det J|\,\mathrm d\xi,
\qquad
\nabla_x\phi_i=J^{-T}\nabla_\xi N_i.
$$

张量单元（四边形、六面体）的 $J$ 可随积分点变化。原 `local_stiffness` 对每个积分点调用一次梯度变换；现在批量求解

$$
J_q^T G_q^T=\widehat G_q^T,
$$

其中 $\widehat G_q$ 和 $G_q$ 分别是第 $q$ 个积分点的参考梯度、物理梯度（按行排列）。核心代码为：

```python
grads = np.linalg.solve(
    np.swapaxes(jac, -1, -2), np.swapaxes(grads_ref, -1, -2)
).swapaxes(-1, -2)
```

若积分点数为 $Q$、局部自由度数为 $L$、空间维数为 $d$，则 `grads_ref` 和结果 `grads` 均为 `(Q, L, d)`。批量求解与逐点应用梯度变换在数学上等价，但减少了 Python 层调用。

### 2.2 局部质量矩阵与参考矩阵缓存

质量矩阵定义为

$$
M_{ij}^K=\int_K\phi_i\phi_j\,\mathrm dx
\approx\sum_q w_q|\det J_q|N_i(\xi_q)N_j(\xi_q).
$$

设 $N\in\mathbb R^{Q\times L}$ 为积分点上的基函数值矩阵，则

$$
M^K=N^T\operatorname{diag}(w_q|\det J_q|)N.
$$

张量单元通过矩阵乘法计算，无须显式构造对角矩阵：

```python
return N.T @ ((w * dets)[:, None] * N)
```

对于仿射单纯形（三角形、四面体），$J_K$ 在单元内为常数，因而

$$
M^K=|\det J_K|\,\widehat M,
\qquad
\widehat M_{ij}=\int_{\widehat K}N_iN_j\,\mathrm d\xi.
$$

`local_mass` 因此只需按几何行列式缩放参考矩阵。新增的 `_reference_mass` 以 `("mass", mass_quad_order())` 为键复用 `_basis_cache`，避免对每个物理单元重复计算 $\widehat M$。

### 2.3 局部刚度矩阵与单纯形批量计算

刚度矩阵定义为

$$
K_{ij}^K=\int_K\nabla\phi_i\cdot\nabla\phi_j\,\mathrm dx
\approx\sum_q w_q|\det J_q|\sum_{a=1}^{d}G_{qia}G_{qja}.
$$

单纯形的仿射映射允许按块计算 Jacobian、行列式和物理梯度。对于一块单元，使用 NumPy 广播及 `einsum` 批量计算局部矩阵：

```python
grads = grads_ref[None, ...] @ np.linalg.inv(jac)[:, None, ...]
values[start:stop] = np.einsum(
    "q,cqia,cqja->cij", weights, grads, grads
) * dets[:, None, None]
```

这里 `c` 为块内单元索引，`q` 为积分点，`i,j` 为局部自由度，`a` 为空间维度。质量矩阵使用 `dets[:, None, None] * reference_mass` 批量缩放。质量矩阵块大小为 256；刚度矩阵按参考梯度数组大小调整块大小，以控制临时数组规模。

### 2.4 全局 COO 组装

设单元数为 $E$，每单元局部自由度数为 $L$，局部到全局自由度映射 `dof_map` 的形状为 `(E, L)`。全局矩阵由各单元贡献累加：

$$
A_{IJ}=\sum_{K}\sum_{i,j:\,\mathrm{dof}_K(i)=I,\,\mathrm{dof}_K(j)=J}A^K_{ij}.
$$

原通用组装路径逐单元生成索引并拼接；新增 `_assemble_lagrange` 一次生成 COO 行、列索引，预分配局部矩阵数组：

```python
rows = np.repeat(dofs, n_local, axis=1).ravel()
cols = np.tile(dofs, (1, n_local)).ravel()
values = np.empty((n_cells, n_local, n_local))
```

最后通过 `from_triplets(rows, cols, values.ravel(), ...)` 构造矩阵并累加重复索引，保留原有对称化处理 `0.5 * (matrix + matrix.T)`。SciPy 可用时采用 COO/CSR 路径，否则使用 NumPy 回退实现。

这些优化不改变基函数、求积规则或离散矩阵的数学定义；质量矩阵仍为一致质量矩阵，并未引入质量集中。

## 3. 正确性验证

`examples/check_lagrange_assembly.py` 在内存中构造优化前的 `ReferenceLagrangeSpace`，与当前 `LagrangeSpace` 对比质量矩阵和刚度矩阵。参考实现基于原文所记录的提交 `4cc162e`，不需要生成或读取历史 JSON/NPZ 基准文件。

验证包括：

- 全局最大绝对误差 $<10^{-10}$，相对 Frobenius 误差 $<10^{-11}$；
- 第 0 单元局部矩阵误差 $<10^{-10}$；
- 刚度矩阵作用于常数向量的残差 $<10^{-10}$；
- 质量矩阵元素和与网格总测度之差 $<10^{-10}$；
- 小网格质量矩阵正定、刚度矩阵半正定；
- P1 解析矩阵与 NumPy 回退路径检查。

在面积为 $1/2$ 的参考三角形上，P1 局部解析矩阵为

$$
M=\frac1{24}\begin{bmatrix}2&1&1\\1&2&1\\1&1&2\end{bmatrix},
\qquad
K=\begin{bmatrix}1&-\tfrac12&-\tfrac12\\-\tfrac12&\tfrac12&0\\-\tfrac12&0&\tfrac12\end{bmatrix}.
$$

从项目根目录运行：

```powershell
python -B examples/check_lagrange_assembly.py
```

原文记录的验证结束信息为：

```text
Validation passed: 48 matrix comparisons, P1 analytic and NumPy fallback checks
```

## 4. 性能测试

原文测试环境：Windows 11、Python 3.12.7、NumPy 1.26.4、SciPy 1.13.1。每项预热一次，组装并转 CSR 五次，取中位数。`old` 表示优化前参考实现，`new` 表示优化后实现，加速比为 `old / new`。下表保留三种网格规模和多项式阶数的代表性结果，均来自原文记录，并非重新测量。

| 网格 | 阶数 | 自由度 | 质量 old/new (ms) | 质量加速 | 刚度 old/new (ms) | 刚度加速 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tri_n16 | P1 | 289 | 32.390 / 0.946 | 34.24× | 17.027 / 0.999 | 17.04× |
| tri_n16 | P2 | 1089 | 53.075 / 1.851 | 28.67× | 18.180 / 3.148 | 5.78× |
| tri_n16 | P3 | 2401 | 139.608 / 3.198 | 43.66× | 25.192 / 8.765 | 2.87× |
| tri_n32 | P1 | 1089 | 129.565 / 2.302 | 56.27× | 63.072 / 3.601 | 17.52× |
| tri_n32 | P2 | 4225 | 213.876 / 6.225 | 34.36× | 79.431 / 12.452 | 6.38× |
| tri_n32 | P3 | 9409 | 514.965 / 11.686 | 44.07× | 103.780 / 34.905 | 2.97× |
| tri_n64 | P1 | 4225 | 517.083 / 9.099 | 56.83× | 270.481 / 14.292 | 18.93× |
| tri_n64 | P2 | 16641 | 876.283 / 18.009 | 48.66× | 309.265 / 44.184 | 7.00× |
| tri_n64 | P3 | 37249 | 4375.222 / 52.794 | 82.87× | 972.639 / 272.148 | 3.57× |

在这些规则三角网格案例中，质量矩阵组装加速约 **28.67–82.87 倍**，刚度矩阵组装加速约 **2.87–18.93 倍**。实际耗时受硬件、依赖版本和运行负载影响；其他几何和维数的案例用于正确性验证，不纳入上述性能统计。
