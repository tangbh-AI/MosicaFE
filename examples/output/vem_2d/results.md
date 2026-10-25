# 2D VEM 细网格复算结果

算例：单位区域上的 -Δu=f，零 Dirichlet 边界。二维精确解为 sin(πx)sin(πy)，三维再乘 sin(πz)。
网格为 Voronoi 多边形/多面体，seed=42。增加单元数会重新生成网格，粗细网格不要求嵌套。

| k | 单元数 | 几何节点 | 自由度 | 最大直径 h | L2 投影误差 | H1 投影半范数误差 | 计算时间/秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 256 | 514 | 1539 | 0.168481 | 1.952003e-04 | 1.327659e-02 | 0.94 |
| 2 | 1024 | 2050 | 6147 | 0.089563 | 2.547782e-05 | 3.377731e-03 | 3.41 |
| 3 | 1024 | 2050 | 11268 | 0.089563 | 4.038294e-07 | 6.825137e-05 | 5.34 |

时间包括建网格、自由度构造、组装、直接求解及误差计算，不包括绘图。
L2 使用可计算的 L2 多项式投影，H1 使用 H1 投影的梯度；均非不可见虚拟函数本身的精确误差。

主程序结构保持：问题数据 → 选网格 → 选元并求解 → 可视化。

- k=2，256 单元：[解图](D:/MosicaFE/examples/output/vem_2d/solution_k2_n256.png)、[网格图](D:/MosicaFE/examples/output/vem_2d/mesh_n256.png)。
- k=2，1024 单元：[解图](D:/MosicaFE/examples/output/vem_2d/solution_k2_n1024.png)、[网格图](D:/MosicaFE/examples/output/vem_2d/mesh_n1024.png)。
- k=3，1024 单元：[解图](D:/MosicaFE/examples/output/vem_2d/solution_k3_n1024.png)、[网格图](D:/MosicaFE/examples/output/vem_2d/mesh_n1024.png)。

在项目根目录复现各组结果：

```powershell
python -X utf8 -c "from examples.vem_poisson import main; main(n_points=256, degree=2)"
python -X utf8 -c "from examples.vem_poisson import main; main(n_points=1024, degree=2)"
python -X utf8 -c "from examples.vem_poisson import main; main(n_points=1024, degree=3)"
```
