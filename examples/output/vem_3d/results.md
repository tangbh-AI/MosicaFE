# 3D VEM 细网格复算结果

算例：单位区域上的 -Δu=f，零 Dirichlet 边界。二维精确解为 sin(πx)sin(πy)，三维再乘 sin(πz)。
网格为 Voronoi 多边形/多面体，seed=42。增加单元数会重新生成网格，粗细网格不要求嵌套。

| k | 单元数 | 几何节点 | 自由度 | 最大直径 h | L2 投影误差 | H1 投影半范数误差 | 计算时间/秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 96 | 544 | 2361 | 0.645336 | 7.294387e-03 | 1.780506e-01 | 10.27 |
| 2 | 192 | 1150 | 4977 | 0.548668 | 4.014975e-03 | 1.212173e-01 | 22.12 |
| 3 | 192 | 1150 | 10527 | 0.548668 | 9.960969e-04 | 1.918771e-02 | 66.43 |

时间包括建网格、自由度构造、组装、直接求解及误差计算，不包括绘图。
L2 使用可计算的 L2 多项式投影，H1 使用 H1 投影的梯度；均非不可见虚拟函数本身的精确误差。

主程序结构保持：问题数据 → 选网格 → 选元并求解 → 可视化。

- k=2，96 单元：[解图](D:/MosicaFE/examples/output/vem_3d/solution_k2_n96.png)、[网格图](D:/MosicaFE/examples/output/vem_3d/mesh_n96.png)。
- k=2，192 单元：[解图](D:/MosicaFE/examples/output/vem_3d/solution_k2_n192.png)、[网格图](D:/MosicaFE/examples/output/vem_3d/mesh_n192.png)。
- k=3，192 单元：[解图](D:/MosicaFE/examples/output/vem_3d/solution_k3_n192.png)、[网格图](D:/MosicaFE/examples/output/vem_3d/mesh_n192.png)。

在项目根目录复现各组结果：

```powershell
python -X utf8 -c "from examples.vem_poisson_3d import main; main(n_points=96, degree=2)"
python -X utf8 -c "from examples.vem_poisson_3d import main; main(n_points=192, degree=2)"
python -X utf8 -c "from examples.vem_poisson_3d import main; main(n_points=192, degree=3)"
```
