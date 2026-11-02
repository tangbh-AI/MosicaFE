# -*- coding: utf-8 -*-
# =============================================================================
#  RT0 混合元求解 Darcy 方程 —— 详解版（注释很多，写法尽量朴素）
# =============================================================================
#
#  这份脚本的目标是"能一行一行看懂"，所以：
#    * 网格、基函数与误差计算保留教学写法
#    * 矩阵组装采用本文件内的 NumPy 批量计算与 COO 稀疏组装
#    * 每个函数、每段循环前面都写清楚"为什么这样做"
#
#  方程（Darcy 渗流的混合形式）：
#
#        q = -K * grad(p)          （达西定律：流速正比于压力梯度）
#        div(q) = f                （质量守恒：流入减流出等于源汇）
#
#  未知量：q（通量，RT0 空间）、p（压力，分片常数 P0 空间）
#
#  弱形式（混合形式）：求 q_h ∈ RT0、p_h ∈ P0，使得对任意 v_h ∈ RT0、r_h ∈ P0
#
#        (K^{-1} q_h, v_h) - (p_h, div v_h) = -<g_D, v_h·n>      （达西定律）
#        (div q_h, r_h)                     = (f, r_h)          （质量守恒）
#
#  离散后是一个"鞍点系统"：
#
#        [ A   -B^T ] [ Q ]   [ b ]
#        [ B    0   ] [ P ] = [ F ]
#
#  其中  A[e,e'] = ∫ (K^{-1} φ_e) · φ_e'        （通量质量矩阵）
#        B[c,e]  = ∫_c div φ_e = ±1              （散度矩阵，只有 ±1！）
#        F[c]    = ∫_c f                         （源项）
#        b[e]    = -∫_e g_D                      （压力 Dirichlet 进右端）
#
#  脚本结构：
#    第 1 部分  网格生成（make_rect_mesh）
#    第 2 部分  网格加密（refine_uniform）★ 本脚本的教学重点
#    第 3 部分  面（棱）的信息：编号、相邻单元、法向符号
#    第 4 部分  组装矩阵 A、B 和右端 b、F
#    第 5 部分  解鞍点系统
#    第 6 部分  误差计算
#    第 7 部分  算例 1：均匀流（有精确解，用来验证代码没错）
#    第 8 部分  算例 2：正弦解 + 网格加密 → 看误差怎么按 O(h) 下降
#    第 9 部分  画图（网格加密过程、收敛曲线、通量箭头）
#
#  运行： python rt0_darcy_2d.py
#  默认输出：压力、通量与守恒残差；run_report() 运行完整收敛实验并生成图片
# =============================================================================

# =============================================================================
#  符号对照表（看公式时随手查：左边是数学，右边是代码）
# =============================================================================
#   数学符号            代码里的名字                    说明
#   ------------------------------------------------------------------------
#   Ω                   （隐式）                        区域，本文是单位正方形
#   K                   cells[c]                        第 c 个三角形（三个顶点编号）
#   |K|                 triangle_area(verts)           三角形面积
#   h                   h = 1 / nx                      网格尺寸（越小越细）
#   q_h                 Q                               通量自由度，每条棱一个
#   p_h                 P                               每个单元一个常数
#   φ_e                 rt0_basis_at(...)               RT0 基函数 (x-a_k)/(2|K|)
#   σ（±1）             facets["signs"][c][k]           全局法向 vs 单元外法向
#   A_{ee'}             A                               通量质量矩阵 ∫ K^{-1} φ_e·φ_e'
#   B_{ce}              B                               散度矩阵，只有 ±1
#   f                   source(x, y)                    源项（正数=注入）
#   g_D                 g_dir(x, y)                     边界压力（Dirichlet 数据）
#   -B^T                -B.T                            鞍点系统右上块
#   A^{-1}              （由 solve_saddle 内部完成）     不显式求逆，直接解方程
#   S = B A^{-1} B^T    （未显式组装）                   Schur 补；本脚本解大系统
#
#   自由度编号约定（非常重要，看懂它就懂了一半）：
#     * 未知量向量 = [Q(0), Q(1), ..., Q(nE-1), P(0), ..., P(nC-1)]
#       前 nE 个是"每条棱的通量"，后 nC 个是"每个单元的压力"；
#     * 棱 e 的编号由 build_facets 决定，规则是"先遍历单元、再遍历该单元的 3 条边"；
#     * 单元 c 的第 k 条棱 = 顶点 k 的对边，编号存在 facets["cell_edges"][c][k]。

import os
import sys
import math

import numpy as np

# 不要在目录里生成 __pycache__（保持文件夹干净）
sys.dont_write_bytecode = True

# 输出图片放在脚本旁边的 figs_detailed 文件夹
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figs_detailed")

# 尝试导入 scipy：有就用稀疏矩阵（网格可以大一些），没有就用 numpy 稠密矩阵
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# =============================================================================
#  第 1 部分：生成矩形区域的三角形网格
# =============================================================================
def make_rect_mesh(nx, ny):
    """
    把单位正方形 [0,1]×[0,1] 切成 nx×ny 个小矩形，
    再把每个小矩形沿对角线切成 2 个三角形。

    参数
    ----
    nx, ny : int
        横向、纵向的"小格子"个数。nx 越大，网格越细，h = 1/nx 越小。

    返回
    ----
    nodes : (n_nodes, 2) 的数组
        每个节点的坐标。
    cells : (n_cells, 3) 的整数数组
        每个三角形三个顶点的编号，按逆时针排列。

    小贴士
    ------
    为什么三角形要按"逆时针"排？
      这样三角形的面积算出来是正的，外法向的公式也不用加负号，
      后面所有符号（±1）都能统一。
    """
    # ---- 1) 先生成所有节点坐标 ----
    # 一共有 (nx+1)*(ny+1) 个节点，排列方式是"先横后纵"
    nodes = []
    for j in range(ny + 1):                 # 纵坐标 j = 0,1,...,ny
        for i in range(nx + 1):             # 横坐标 i = 0,1,...,nx
            x = i / nx                      # 归一化到 [0,1]
            y = j / ny
            nodes.append([x, y])
    nodes = np.array(nodes, dtype=float)

    # 节点编号的小工具：给 (i,j)，返回它在 nodes 里的行号
    # 因为上面是"外层 j、内层 i"，所以编号 = j*(nx+1) + i
    def node_id(i, j):
        return j * (nx + 1) + i

    # ---- 2) 再生成每个小矩形里的两个三角形 ----
    cells = []
    for j in range(ny):
        for i in range(nx):
            # 一个小矩形的四个角（顺时针看是 左下→右下→右上→左上）
            a = node_id(i,     j)       # 左下
            b = node_id(i + 1, j)       # 右下
            c = node_id(i,     j + 1)   # 左上
            d = node_id(i + 1, j + 1)   # 右上

            # 沿 a-d 这条对角线切成两个三角形
            # 第一个：a -> b -> d（逆时针）
            cells.append([a, b, d])
            # 第二个：a -> d -> c（逆时针）
            cells.append([a, d, c])

    return nodes, np.array(cells, dtype=int)


# =============================================================================
#  第 2 部分：网格加密（一致加密 / red refinement）★ 重点
# =============================================================================
def refine_uniform(nodes, cells):
    """
    一致加密：把每个三角形"一分为四"。

    做法（这就是有限元里最标准的"red refinement"）：
      1. 取三角形三条边的中点；
      2. 把原三角形和三个中点连起来，得到 4 个全等的小三角形。

    文字示意图（顶点 a、b、c，边中点 ab、bc、ca）：

            c
            |\\
            | \\
           ca   bc
            |    \\
            |_____\\
            a  ab  b

      4 个小三角形：
        (a,  ab, ca)     左下角
        (ab, b,  bc)     右下角
        (ca, bc, c)      顶角
        (ab, bc, ca)     中间那个（倒过来）

    关键点（也是初学者最容易写错的地方）：
      相邻两个三角形共享一条边，这条边的中点只能创建 **一次**。
      所以要用一个字典 mid_of_edge 记录"这条边的中点已经建过了，
      编号是几"。字典的键用 (小号, 大号) 这样排好序，避免 (3,7) 和 (7,3)
      被当成两条不同的边。

    加密后：
      * 单元数变成原来的 4 倍；
      * 每个三角形的直径（h）变成原来的 1/2；
      * 节点数大约变成 4 倍（准确说是 4V + E 换成 4E 的关系）。
    """
    # 新节点列表：先把原来的节点全部抄进去（编号保持不变，方便对照）
    new_nodes = []
    for p in nodes:
        new_nodes.append([float(p[0]), float(p[1])])

    # 字典：键是 (小端号, 大端号)，值是"中点节点的新编号"
    mid_of_edge = {}

    def edge_midpoint(i, j):
        """
        返回节点 i、j 之间那条边的中点编号。
        如果这个中点还没建，就新建一个节点，并记进字典。
        """
        # 排序，保证 (i,j) 和 (j,i) 拿到同一个键
        if i < j:
            key = (i, j)
        else:
            key = (j, i)

        # 已经建过：直接返回旧编号
        if key in mid_of_edge:
            return mid_of_edge[key]

        # 没建过：取两点平均，作为新节点追加到 new_nodes 末尾
        p1 = nodes[i]
        p2 = nodes[j]
        mx = (float(p1[0]) + float(p2[0])) / 2.0
        my = (float(p1[1]) + float(p2[1])) / 2.0
        new_id = len(new_nodes)                 # 新节点的编号 = 当前长度
        new_nodes.append([mx, my])
        mid_of_edge[key] = new_id               # 记进字典
        return new_id

    # 新单元列表：每个老三角形变 4 个
    new_cells = []
    for tri in cells:
        a = int(tri[0])
        b = int(tri[1])
        c = int(tri[2])

        # 三条边的中点
        ab = edge_midpoint(a, b)
        bc = edge_midpoint(b, c)
        ca = edge_midpoint(c, a)

        # 拼出 4 个小三角形（顺序仍然保持逆时针）
        new_cells.append([a,  ab, ca])
        new_cells.append([ab, b,  bc])
        new_cells.append([ca, bc, c])
        new_cells.append([ab, bc, ca])

    return np.array(new_nodes, dtype=float), np.array(new_cells, dtype=int)


def refine_times(nodes, cells, times):
    """
    连续加密 times 次。

    每加密一次 h 减半、单元数乘 4，所以：
        nx  ->  nx * 2^times
        h   ->  h / 2^times
        M   ->  M * 4^times      （M 是单元数）
    """
    for _ in range(times):
        nodes, cells = refine_uniform(nodes, cells)
    return nodes, cells


# =============================================================================
#  第 3 部分：面（2D 里就是"棱"）的信息
# =============================================================================
def build_facets(nodes, cells):
    """
    把网格里的所有棱整理成"面"（facet），并记录：

      edge_nodes[e]  = [i, j]    第 e 条棱的两个端点
      edge_cells[e]  = [c1, c2]  第 e 条棱相邻的单元（边界棱只有 1 个）
      cell_edges[c][k] = e       单元 c 的第 k 条棱（k = 顶点 k 的"对边"）
      signs[c][k]    = +1 / -1   单元 c 在这条棱上的外法向，
                                 与"全局法向"同向取 +1，反向取 -1

    为什么要 signs：
      RT0 的自由度是"穿过这条棱的法向通量"。为了让相邻两个单元共用
      同一个自由度，我们必须给每条棱规定一个**统一的**法向（全局法向）。
      但每个单元自己的"外法向"可能与此相同、也可能相反，
      这个差别就是 ±1，记作 σ（代码里叫 sign）。
      它出现在两处：
        * 散度矩阵   B[c,e] = σ
        * 单元内基函数 φ_e = σ * φ̂_k （φ̂_k 是"按单元外法向"的局部基函数）

    说明：局部编号 k 表示"第 k 个顶点的对边"。
      三角形三个顶点 tri[0], tri[1], tri[2]；
      顶点 0 的对边 = (tri[1], tri[2])
      顶点 1 的对边 = (tri[2], tri[0])
      顶点 2 的对边 = (tri[0], tri[1])
    """
    edge_id = {}                # (i,j) -> 全局棱编号
    edge_nodes = []             # 每条棱的两个端点
    edge_cells = []             # 每条棱的相邻单元
    cell_edges = []             # cell_edges[c][k] = 棱编号

    # ---- 1) 先给所有棱编号 ----
    for c in range(len(cells)):
        tri = cells[c]
        cell_edges.append([-1, -1, -1])
        for k in range(3):
            # 顶点 k 的对边：另外两个顶点
            i = int(tri[(k + 1) % 3])
            j = int(tri[(k + 2) % 3])
            # 排序做键
            if i < j:
                key = (i, j)
            else:
                key = (j, i)
            # 新棱就新建编号
            if key not in edge_id:
                edge_id[key] = len(edge_nodes)
                edge_nodes.append([key[0], key[1]])
                edge_cells.append([])
            e = edge_id[key]
            cell_edges[c][k] = e
            edge_cells[e].append(c)     # 顺便记下相邻单元

    edge_nodes = np.array(edge_nodes, dtype=int)

    # ---- 2) 算每条棱的"全局单位法向" ----
    # 约定：把"从 i 指向 j 的切向"逆时针转 90°，作为全局法向。
    # 逆时针转 90°：(tx, ty) -> (-ty, tx)
    edge_normals = []
    for e in range(len(edge_nodes)):
        i = edge_nodes[e][0]
        j = edge_nodes[e][1]
        p1 = nodes[i]
        p2 = nodes[j]
        tx = float(p2[0]) - float(p1[0])
        ty = float(p2[1]) - float(p1[1])
        length = math.sqrt(tx * tx + ty * ty)
        nx_ = -ty / length
        ny_ = tx / length
        edge_normals.append([nx_, ny_])
    edge_normals = np.array(edge_normals, dtype=float)

    # ---- 3) 算每个单元、每条棱的 sign = ±1 ----
    signs = []
    for c in range(len(cells)):
        tri = cells[c]
        signs_row = [0, 0, 0]
        for k in range(3):
            e = cell_edges[c][k]
            i = edge_nodes[e][0]
            j = edge_nodes[e][1]

            # 棱的中点
            mid = (nodes[i] + nodes[j]) / 2.0
            # 对边顶点（顶点 k）
            opposite = nodes[int(tri[k])]

            # 棱的切向
            tx = float(nodes[j][0]) - float(nodes[i][0])
            ty = float(nodes[j][1]) - float(nodes[i][1])

            # 候选法向（垂直于切向）
            nx1 = -ty
            ny1 = tx

            # 判断它指向"远离对边顶点"的方向（那就是外法向）
            # 用向量 (对边顶点 - 棱中点) 点乘候选法向：
            #   如果是正的，说明候选法向指向三角形内部，要反过来
            vx = float(opposite[0]) - float(mid[0])
            vy = float(opposite[1]) - float(mid[1])
            if nx1 * vx + ny1 * vy > 0.0:
                nx1 = -nx1
                ny1 = -ny1

            # 归一化
            norm = math.sqrt(nx1 * nx1 + ny1 * ny1)
            nx1 = nx1 / norm
            ny1 = ny1 / norm

            # 与全局法向比较：同向 +1，反向 -1
            g = edge_normals[e]
            dot = nx1 * float(g[0]) + ny1 * float(g[1])
            if dot > 0.0:
                signs_row[k] = 1
            else:
                signs_row[k] = -1
        signs.append(signs_row)

    return {
        "edge_nodes": edge_nodes,
        "edge_cells": edge_cells,
        "cell_edges": cell_edges,
        "edge_normals": edge_normals,
        "signs": signs,
    }


# =============================================================================
#  第 4 部分：组装矩阵
# =============================================================================
def triangle_area(verts):
    """
    三角形面积 = |叉积| / 2。
    顶点按逆时针排时叉积是正的，所以这里再取绝对值更保险。
    """
    x0, y0 = verts[0]
    x1, y1 = verts[1]
    x2, y2 = verts[2]
    cross = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    return abs(cross) / 2.0


def quadrature_triangle(verts):
    """
    三角形上的 7 点求积公式（Dunavant，代数精度 5）。

    为什么不用"3 点中点公式"？
      3 点公式精度只有 2，算 sin 这类源项的积分会有额外误差，
      会污染收敛阶。7 点公式对 5 次多项式精确，够用了。

    返回：
      points  : (7, 2)  积分点坐标
      weights : (7,)    权重（已经乘过面积，所以可以直接 sum(w*f)）
    """
    # 重心坐标 (λ1, λ2, λ3)，注意 λ1+λ2+λ3 = 1
    bary = [
        (0.333333333333333, 0.333333333333333, 0.333333333333333),
        (0.059715871789770, 0.470142064105115, 0.470142064105115),
        (0.470142064105115, 0.059715871789770, 0.470142064105115),
        (0.470142064105115, 0.470142064105115, 0.059715871789770),
        (0.797426985353087, 0.101286507323456, 0.101286507323456),
        (0.101286507323456, 0.797426985353087, 0.101286507323456),
        (0.101286507323456, 0.101286507323456, 0.797426985353087),
    ]
    # 对应的权重（和为 1）
    ref_w = [
        0.225000000000000,
        0.132394152788506,
        0.132394152788506,
        0.132394152788506,
        0.125939180544827,
        0.125939180544827,
        0.125939180544827,
    ]

    area = triangle_area(verts)
    points = []
    weights = []
    for k in range(7):
        l1 = bary[k][0]
        l2 = bary[k][1]
        l3 = bary[k][2]
        # 用重心坐标把参考点映射到真实三角形：
        #   x = l1*v0 + l2*v1 + l3*v2
        x = l1 * verts[0][0] + l2 * verts[1][0] + l3 * verts[2][0]
        y = l1 * verts[0][1] + l2 * verts[1][1] + l3 * verts[2][1]
        points.append([x, y])
        weights.append(ref_w[k] * area)
    return np.array(points), np.array(weights)


def rt0_basis_at(verts, point):
    """
    在一个单元上，算三个 RT0 基函数在 point 处的值。

    RT0 的局部基函数（按"单元外法向"定向）：
        φ̂_k(x) = (x - a_k) / (d * |K|)，  d = 2（二维）
    其中 a_k 是第 k 个顶点（也就是第 k 条边的对顶点）。

    性质（可以自己用笔验证）：
        ∫_{第 k 条边} φ̂_k · n_out ds = 1
        ∫_{第 j 条边} φ̂_k · n_out ds = 0   (j ≠ k)
        div φ̂_k = d/(d|K|) = 1/|K|          （常数！）

    返回：长度为 3 的列表，第 k 个是二维向量
    """
    d = 2.0
    area = triangle_area(verts)
    out = []
    for k in range(3):
        ax = verts[k][0]
        ay = verts[k][1]
        vx = (point[0] - ax) / (d * area)
        vy = (point[1] - ay) / (d * area)
        out.append([vx, vy])
    return out


# =============================================================================
#  批量组装与统一求解接口（本文件内实现，需要 SciPy）
# =============================================================================
from scipy import sparse
from scipy.sparse.linalg import spsolve


def rectangle_mesh(n=16):
    """单位正方形 n×n 网格，每格沿左下至右上分成两个三角形。"""
    x, y = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
    nodes = np.column_stack((x.ravel(), y.ravel()))
    a = (np.arange(n)[:, None] * (n + 1) + np.arange(n)).ravel()
    cells = np.stack((np.column_stack((a, a + 1, a + n + 2)),
                      np.column_stack((a, a + n + 2, a + n + 1))), axis=1)
    return nodes, cells.reshape(-1, 3)


def _values(function, points):
    if not callable(function):
        return np.full(points.shape[:-1], float(function))
    coords = np.moveaxis(points, -1, 0)
    try:
        return np.broadcast_to(np.asarray(function(*coords), dtype=float), points.shape[:-1])
    except (TypeError, ValueError):
        return np.vectorize(function, otypes=[float])(*coords)


def _geometry(mesh, dim, facets=None):
    nodes, cells = np.asarray(mesh[0], dtype=float), np.asarray(mesh[1])
    if dim not in (2, 3) or nodes.ndim != 2 or nodes.shape[1] != dim:
        raise ValueError('dim 必须是 2 或 3，且与节点坐标维数一致')
    if cells.ndim != 2 or cells.shape[1] != dim + 1 or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError('cells 必须是整数单纯形连接数组')
    if np.any(cells < 0) or np.any(cells >= len(nodes)):
        raise ValueError('单元节点编号越界')
    vertices = nodes[cells]
    volume = np.abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1])) / math.factorial(dim)
    if not np.all(np.isfinite(vertices)) or np.any(volume <= 0):
        raise ValueError('网格含退化单元或非有限坐标')
    if facets is not None:
        return nodes, cells, vertices, volume, facets
    local_faces = np.array([[j for j in range(dim + 1) if j != k] for k in range(dim + 1)])
    faces = np.sort(cells[:, local_faces], axis=-1)
    face_nodes, inverse, counts = np.unique(faces.reshape(-1, dim), axis=0,
                                           return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('非流形网格：一个面邻接超过两个单元')
    face_ids = inverse.reshape(-1, dim + 1)
    fv = nodes[face_nodes]
    tangent = fv[:, 1] - fv[:, 0]
    if dim == 2:
        normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    else:
        normals = np.cross(tangent, fv[:, 2] - fv[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    centers = fv.mean(axis=1)
    # 对顶点位于内部；全局法向指向对顶点时，与外法向反向。
    dot = np.einsum('ckd,ckd->ck', vertices - centers[face_ids], normals[face_ids])
    signs = np.where(dot > 0, -1, 1)
    facets = dict(edge_nodes=face_nodes, cell_edges=face_ids, signs=signs,
                  edge_normals=normals, boundary=np.flatnonzero(counts == 1))
    return nodes, cells, vertices, volume, facets


def assemble_rt0(mesh, *, dim=2, permeability=1.0, source=0.0, pressure=0.0,
                 noflow_edges=(), pressure_edges=None, facets=None):
    """返回 A,B,b,F。质量矩阵用精确几何矩批量计算，无三重 Python 循环。"""
    nodes, cells, v, vol, facets = _geometry(mesh, dim, facets)
    ids = np.asarray(facets['cell_edges'])
    signs = np.asarray(facets['signs'])
    nf, nc = len(facets['edge_nodes']), len(cells)
    K = np.asarray(permeability, dtype=float)
    K = np.eye(dim) * K if K.ndim == 0 else K
    if K.shape != (dim, dim) or not np.all(np.isfinite(K)) or not np.allclose(K, K.T):
        raise ValueError('permeability 必须是正常数或常值对称正定张量')
    np.linalg.cholesky(K)
    invK = np.linalg.inv(K)
    # E[x]=顶点平均值；E[xx^T]=(Σ a_i a_i^T + S S^T)/((d+1)(d+2))。
    total = v.sum(axis=1)
    mean = v.mean(axis=1)
    second = (np.einsum('cid,cie->cde', v, v)
              + np.einsum('cd,ce->cde', total, total)) / ((dim + 1) * (dim + 2))
    constant = np.einsum('de,cde->c', invK, second)
    cross = np.einsum('cid,de,ce->ci', v, invK, mean)
    local = (constant[:, None, None] - cross[:, :, None] - cross[:, None, :]
             + np.einsum('cid,de,cje->cij', v, invK, v)) / (dim**2 * vol[:, None, None])
    local *= signs[:, :, None] * signs[:, None, :]
    shape = local.shape
    rows = np.broadcast_to(ids[:, :, None], shape).ravel()
    cols = np.broadcast_to(ids[:, None, :], shape).ravel()
    A = sparse.coo_matrix((local.ravel(), (rows, cols)), shape=(nf, nf)).tocsr()
    B = sparse.coo_matrix((signs.ravel(),
                          (np.repeat(np.arange(nc), dim + 1), ids.ravel())), shape=(nc, nf)).tocsr()
    # Duffy 变换 + 4 点 Gauss：三角形/四面体上批量积分源项。
    g, w = np.polynomial.legendre.leggauss(4)
    g, w = (g + 1) / 2, w / 2
    grids = np.meshgrid(*([g] * dim), indexing='ij')
    weights = np.prod(np.stack(np.meshgrid(*([w] * dim), indexing='ij')), axis=0).ravel()
    u = [a.ravel() for a in grids]
    if dim == 2:
        bary = np.column_stack(((1-u[0])*(1-u[1]), u[0], (1-u[0])*u[1]))
        weights *= 2 * (1-u[0])
    else:
        bary = np.column_stack(((1-u[0])*(1-u[1])*(1-u[2]), u[0],
                                (1-u[0])*u[1], (1-u[0])*(1-u[1])*u[2]))
        weights *= 6 * (1-u[0])**2 * (1-u[1])
    points = np.einsum('qi,cid->cqd', bary, v)
    F = vol * (_values(source, points) @ weights)
    boundary = facets.get('boundary')
    if boundary is None:
        boundary = np.array([e for e, cs in enumerate(facets['edge_cells']) if len(cs) == 1])
    noflow = np.asarray(noflow_edges, dtype=int)
    pe = np.setdiff1d(boundary, noflow) if pressure_edges is None else np.asarray(pressure_edges, dtype=int)
    if not np.all(np.isin(noflow, boundary)) or not np.all(np.isin(pe, boundary)):
        raise ValueError('边界条件只能施加于边界面')
    if np.intersect1d(pe, noflow).size or not np.array_equal(np.union1d(pe, noflow), np.sort(boundary)):
        raise ValueError('压力边界与不透水边界必须不重叠并覆盖全部边界')
    if not len(pe):
        raise ValueError('当前接口要求至少一个压力边界面以确定压力常数')
    # 边界外法向符号取自唯一邻接单元，不能遗漏全局法向转换。
    owner_sign = np.zeros(nf)
    owner_sign[ids.ravel()] = signs.ravel()
    centers = nodes[np.asarray(facets['edge_nodes'])[pe]].mean(axis=1)
    b = np.zeros(nf)
    b[pe] = -owner_sign[pe] * _values(pressure, centers)
    if len(noflow):
        free = np.ones(nf); free[noflow] = 0
        D = sparse.diags(free)
        A = (D @ A @ D + sparse.diags(1-free)).tocsr()
        B = (B @ D).tocsr()
    return A, B, b, F


def solve_rt0(mesh, *, element='RT0', dim=2, refinements=0, **data):
    """组装并求解 Darcy；refinements 为二维网格逐级加密次数，默认 0。

    返回 Q、P、最终网格、边信息、加密次数与守恒残差。
    如显式传入边界面编号，编号必须对应最终网格。
    三维网格目前只接受 refinements=0。
    """
    if str(element).upper() != 'RT0':
        raise ValueError('此接口仅支持 RT0-P0')
    if isinstance(refinements, (bool, np.bool_)) or not isinstance(refinements, (int, np.integer)) or refinements < 0:
        raise ValueError('refinements 必须是非负整数，0 表示不加密')
    if refinements and dim != 2:
        raise ValueError('当前逐级加密仅支持二维三角形；三维请传入已加密网格并设 refinements=0')
    nodes, cells, _, _, facets = _geometry(mesh, dim)
    if refinements:
        nodes, cells = refine_times(nodes, cells, refinements)
        nodes, cells, _, _, facets = _geometry((nodes, cells), dim)
    A, B, b, F = assemble_rt0((nodes, cells), dim=dim, facets=facets, **data)
    system = sparse.bmat([[A, -B.T], [B, None]], format='csc')
    sol = spsolve(system, np.concatenate((b, F)))
    if not np.all(np.isfinite(sol)):
        raise RuntimeError('求解失败：检查网格连通性和压力边界')
    Q, P = sol[:A.shape[0]], sol[A.shape[0]:]
    return dict(Q=Q, P=P, nodes=nodes, cells=cells, facets=facets,
                refinements=int(refinements),
                conservation=float(np.max(np.abs(B @ Q-F))))



def assemble(nodes, cells, facets, K_value, source, g_dir, dirichlet_edges,
             noflow_edges):
    """
    组装线性系统需要的所有东西。

    参数说明
    --------
    K_value : float
        渗透率（这里先用常数；要做各向异性/变系数，把 A 里的 K^{-1}
        换成 1/K(x) 或矩阵乘法即可，见文末"怎么扩展"的说明）。
    source : 函数 f(x, y) -> 实数
        源项。返回正数表示"这里在注入"。
    g_dir : 函数 g(x, y) -> 实数
        边界上的压力值（Dirichlet 数据）。
    dirichlet_edges : 列表
        哪些边界棱是"压力给定"。
    noflow_edges : 列表
        哪些边界棱是"不透水"（q·n = 0）。

    返回
    ----
    A, B, rhs_q, rhs_p
        拼成鞍点系统用的矩阵与右端。
    批量实现见本文件的 assemble_rt0；保留传入网格的全局边编号。
    """
    return assemble_rt0(
        (nodes, cells), dim=2, permeability=K_value, source=source,
        pressure=g_dir, noflow_edges=noflow_edges,
        pressure_edges=dirichlet_edges, facets=facets,
    )


# =============================================================================
#  第 5 部分：解鞍点系统
# =============================================================================
def solve_saddle(A, B, rhs_q, rhs_p):
    """
    解
        [ A   -B^T ] [ Q ]   [ rhs_q ]
        [ B    0   ] [ P ] = [ rhs_p ]

    这里直接把整个大矩阵拼出来解（最直白）。
    网格很大时可以改用 Schur 补（先解 P，再回代 Q），会快很多。
    """
    n_q = A.shape[0]
    n_p = B.shape[0]

    if HAS_SCIPY:
        # A、B 在 assemble 里已经转成 csr，这里直接拼装即可
        big = sp.bmat([[A, -B.T],
                       [B, None]], format="csr")
        rhs = np.concatenate([rhs_q, rhs_p])
        sol = spla.spsolve(big, rhs)        # 稀疏 LU 直接解
    else:
        # 稠密矩阵：只适合小网格（nx 不超过 16 左右）
        big = np.zeros((n_q + n_p, n_q + n_p))
        big[0:n_q, 0:n_q] = A
        big[0:n_q, n_q:] = -B.T
        big[n_q:, 0:n_q] = B
        rhs = np.concatenate([rhs_q, rhs_p])
        sol = np.linalg.solve(big, rhs)

    Q = sol[0:n_q]          # 每条棱上的通量
    P = sol[n_q:]           # 每个单元的压力
    return Q, P


def flux_at_point(nodes, cells, facets, c, Q, point):
    """
    在单元 c 内某点重构通量向量：
        q_h(x) = Σ_k Φ_{e_k} * sign_k * φ̂_k(x)
    """
    tri = cells[c]
    verts = nodes[tri]
    basis = rt0_basis_at(verts, point)      # 三个局部基函数的值

    vx = 0.0
    vy = 0.0
    for k in range(3):
        e = facets["cell_edges"][c][k]
        s = facets["signs"][c][k]
        vx += Q[e] * s * basis[k][0]
        vy += Q[e] * s * basis[k][1]
    return vx, vy


def cell_center(nodes, cells, c):
    """单元重心（三个顶点坐标的平均）。"""
    tri = cells[c]
    x = (float(nodes[tri[0]][0]) + float(nodes[tri[1]][0]) + float(nodes[tri[2]][0])) / 3.0
    y = (float(nodes[tri[0]][1]) + float(nodes[tri[1]][1]) + float(nodes[tri[2]][1])) / 3.0
    return x, y


# =============================================================================
#  第 6 部分：误差计算
# =============================================================================
def compute_errors(nodes, cells, facets, Q, P, exact_p, exact_q):
    """
    算两个 L2 误差：
        ||q_h - q||   （通量，向量）
        ||p_h - p||   （压力，分片常数）
    以及最大散度误差（检查"逐单元守恒"有没有满足）。

    exact_p(x, y) -> 实数
    exact_q(x, y) -> (vx, vy)
    """
    err_q2 = 0.0        # 通量误差平方
    err_p2 = 0.0        # 压力误差平方（口径 A：重心点）
    err_p_l2_sq = 0.0   # 压力误差平方（口径 B：单元内真 L2）

    for c in range(len(cells)):
        tri = cells[c]
        verts = nodes[tri]
        area = triangle_area(verts)

        # 压力：这里给两种口径，含义不同，写报告时别混用。
        #   口径 A（重心点）：拿 P0 常数与"重心处的精确值"比。
        #       三角形重心是二阶精确的中点公式点，所以这个误差是 O(h^2)。
        #   口径 B（真 L2）：∫_K (P_K - p(x))^2 dx，用 7 点公式积分。
        #       P0 表达不了单元内 p 的线性变化，所以这个误差是 O(h)。
        #   两版脚本的差别就在这里（旧版用 B，新版原来只给 A）。
        cx, cy = cell_center(nodes, cells, c)
        p_ex = exact_p(cx, cy)
        err_p2 += area * (P[c] - p_ex) ** 2          # 口径 A

        qpts_p, qw_p = quadrature_triangle(verts)
        for m in range(len(qw_p)):
            px = float(qpts_p[m][0])
            py = float(qpts_p[m][1])
            err_p_l2_sq += qw_p[m] * (P[c] - exact_p(px, py)) ** 2   # 口径 B

        # 通量：在 7 个求积点上逐点比较
        qpts, qw = quadrature_triangle(verts)
        for m in range(len(qw)):
            x = float(qpts[m][0])
            y = float(qpts[m][1])
            qx, qy = flux_at_point(nodes, cells, facets, c, Q, [x, y])
            ex, ey = exact_q(x, y)
            err_q2 += qw[m] * ((qx - ex) ** 2 + (qy - ey) ** 2)

    err_q = math.sqrt(err_q2)
    err_p = math.sqrt(err_p2)          # 重心点口径（O(h^2)，见下）

    # 逐单元守恒：Σ_e sign * Φ_e 应该等于 ∫_c f。
    # 这里没有存 f 的积分，所以改看"散度是否分片常数"的另一个等价检查：
    # 直接算每个单元里 div q_h 的值，看它是否等于 -源项之外的常数……
    # 简单起见，这里只报告 B@Q 的最大绝对值（对无源情形应为 0）。
    err_p_l2 = math.sqrt(err_p_l2_sq)
    return err_q, err_p, err_p_l2


def max_divergence(B, Q):
    """
    B @ Q 的第 c 个分量就是 ∫_c div q_h。
    无源（f=0）时它应该精确等于 0（这是 RT0 的看家本领：逐单元守恒）。
    """
    r = B.dot(Q)
    return float(np.max(np.abs(r)))


# =============================================================================
#  第 7 部分：算例 1 —— 均匀流（验证代码对不对）
# =============================================================================
def case1_uniform_flow():
    """
    左右两边给定压力 p = 1 - x（左边 1、右边 0），上下不透水。
    渗透率 K = 2.5，源项 f = 0。

    精确解：
        p = 1 - x
        q = -K grad p = (2.5, 0)      （水平均匀流）

    这个算例的多项式次数很低（p 是线性的、q 是常数），
    RT0 应该能**精确**表示它，所以误差应该在 1e-14 量级。
    如果误差很大，说明组装或边界条件写错了。
    """
    print("=" * 74)
    print("算例 1：均匀流（左右定压、上下不透水，K=2.5）")
    print("=" * 74)

    nx = 8
    ny = 8
    K = 2.5
    nodes, cells = make_rect_mesh(nx, ny)
    facets = build_facets(nodes, cells)

    # 找出边界棱：只属于 1 个单元的那些
    boundary_edges = []
    for e in range(len(facets["edge_nodes"])):
        if len(facets["edge_cells"][e]) == 1:
            boundary_edges.append(e)

    # 分类：x≈0 或 x≈1 的是定压边；y≈0 或 y≈1 的是不透水边
    dirichlet_edges = []
    noflow_edges = []
    for e in boundary_edges:
        i = facets["edge_nodes"][e][0]
        j = facets["edge_nodes"][e][1]
        mx = (float(nodes[i][0]) + float(nodes[j][0])) / 2.0
        if mx < 1e-12 or mx > 1.0 - 1e-12:
            dirichlet_edges.append(e)       # 左右：给定压力
        else:
            noflow_edges.append(e)          # 上下：不透水

    def source(x, y):
        return 0.0

    def g_dir(x, y):
        return 1.0 - x                      # 左边 1、右边 0

    A, B, rhs_q, rhs_p = assemble(nodes, cells, facets, K, source, g_dir,
                                  dirichlet_edges, noflow_edges)
    Q, P = solve_saddle(A, B, rhs_q, rhs_p)

    # 精确解
    def exact_p(x, y):
        return 1.0 - x

    def exact_q(x, y):
        return K, 0.0                       # q = -K * dp/dx = -K * (-1) = K

    err_q, err_p, err_p_l2 = compute_errors(nodes, cells, facets, Q, P,
                                            exact_p, exact_q)

    print("  单元数 = %d，棱数 = %d，未知量 = %d" % (len(cells), len(facets["edge_nodes"]),
                                                 len(facets["edge_nodes"]) + len(cells)))
    print("  ||q_h - q||_L2 = %.3e   （应该是机器零）" % err_q)
    print("  ||p_h - p||     = %.3e   （重心点口径，应为机器零）" % err_p)
    print("  ||p_h - p||_L2  = %.3e   （单元内真 L2；p 是线性的，"
          "P0 表达不了单元内变化，所以这里是 O(h) 量级）" % err_p_l2)

    # 看一眼通量是不是 (2.5, 0)
    vx, vy = flux_at_point(nodes, cells, facets, 0, Q, cell_center(nodes, cells, 0))
    print("  第 0 个单元里的 q_h = (%.10f, %.10f)   （应为 (2.5, 0)）" % (vx, vy))

    # 检查逐单元守恒：B@Q 应该全为 0（因为 f=0）
    print("  max |∫_c div q_h| = %.2e   （应为 0）" % max_divergence(B, Q))
    print()
    return err_q, err_p


# =============================================================================
#  第 8 部分：算例 2 —— 正弦解 + 网格加密（看收敛阶）
# =============================================================================
def case2_convergence():
    """
    取精确解
        p = sin(pi x) * sin(pi y)
    则
        q = -grad p = ( -pi cos(pi x) sin(pi y),  -pi sin(pi x) cos(pi y) )
        f = div q = -Laplacian(p) = 2 pi^2 sin(pi x) sin(pi y)

    四周都给 p = 0（自然边界条件，进右端）。

    然后从 4×4 的网格开始，每加密一次（一分为四）算一次误差，
    看误差是不是大致按 h^1 下降（RT0 + P0 的理论阶是 1）。

    ★ 这就是"网格加密 → 误差下降"的完整演示：
       加密 1 次：h 减半
       加密 2 次：h 再减半
       ……
       如果误差也每次大约减半，就说明收敛阶 ≈ 1。
    """
    print("=" * 74)
    print("算例 2：正弦解 + 网格加密（看收敛阶）")
    print("=" * 74)

    pi = math.pi

    def exact_p(x, y):
        return math.sin(pi * x) * math.sin(pi * y)

    def exact_q(x, y):
        qx = -pi * math.cos(pi * x) * math.sin(pi * y)
        qy = -pi * math.sin(pi * x) * math.cos(pi * y)
        return qx, qy

    def source(x, y):
        # f = div q = -Laplacian(p) = 2 pi^2 sin sin
        return 2.0 * pi * pi * math.sin(pi * x) * math.sin(pi * y)

    def g_dir(x, y):
        return 0.0                          # 边界上 p = 0

    # 记录每一层的：网格尺寸 h、棱数、单元数、两个误差
    rows = []

    # 基准网格 16×16，然后加密 0、1、2、3 次
    #   -> 实际跑的是 16、32、64、128 四档（h = 1/16, 1/32, 1/64, 1/128）
    #   网格太粗时（例如 4、8）误差本身很小，收敛阶容易被离散化的"前渐近区"带偏；
    #   从 16 起四档都是渐近区，阶数会稳稳贴在 1 上。
    base_n = 16
    # 初始网格只生成一次，后续各层直接细分上一层的网格。
    nodes, cells = make_rect_mesh(base_n, base_n)
    for level in range(4):
        if level > 0:
            nodes, cells = refine_uniform(nodes, cells)

        facets = build_facets(nodes, cells)

        # 本层网格的 h：相邻节点间距（基准 1/base_n，每加密一次减半）
        h = 1.0 / (base_n * (2 ** level))

        # 所有边界棱都给压力 Dirichlet
        dirichlet_edges = []
        for e in range(len(facets["edge_nodes"])):
            if len(facets["edge_cells"][e]) == 1:
                dirichlet_edges.append(e)
        noflow_edges = []

        A, B, rhs_q, rhs_p = assemble(nodes, cells, facets, 1.0, source, g_dir,
                                      dirichlet_edges, noflow_edges)
        Q, P = solve_saddle(A, B, rhs_q, rhs_p)
        err_q, err_p, err_p_l2 = compute_errors(nodes, cells, facets, Q, P,
                                                exact_p, exact_q)

        n_edges = len(facets["edge_nodes"])
        n_cells = len(cells)
        rows.append({
            "level": level,
            "h": h,
            "edges": n_edges,
            "cells": n_cells,
            "unknowns": n_edges + n_cells,
            "err_q": err_q,
            "err_p": err_p,
            "err_p_l2": err_p_l2,
        })

        print("  加密 %d 次: h=1/%-3d  单元=%5d  棱=%5d  未知量=%5d  "
              "||q||=%.3e  ||p||=%.3e（L2 口径 %.3e）"
              % (level, int(round(1.0 / h)), n_cells, n_edges,
                 n_edges + n_cells, err_q, err_p, err_p_l2))

    # ---- 算收敛阶 ----
    # 公式：  rate = log(e1/e2) / log(h1/h2)
    # 如果 h 每次减半，那么 rate = log2(e1/e2)。
    print()
    print("  收敛阶（相邻两层之间）：")
    print("  %-8s %-10s %-16s %-16s"
          % ("层数", "q 的阶", "p 的阶(重心点)", "p 的阶(真 L2)"))
    for k in range(1, len(rows)):
        e1 = rows[k - 1]
        e2 = rows[k]
        lg = math.log(e1["h"] / e2["h"])
        rate_q = math.log(e1["err_q"] / e2["err_q"]) / lg
        rate_p = math.log(e1["err_p"] / e2["err_p"]) / lg
        rate_pl2 = math.log(e1["err_p_l2"] / e2["err_p_l2"]) / lg
        print("  %d -> %-4d %-10.3f %-16.3f %-16.3f"
              % (k - 1, k, rate_q, rate_p, rate_pl2))

    print()
    print("  理论：q 是 1 阶（O(h)）；p 也是 1 阶。")
    print("  下表里 p 有两个口径：")
    print("    * 重心点口径 ≈2：重心是二阶精确的中点公式点，所以多赚一阶；")
    print("    * 真 L2 口径  ≈1：P0 表达不了单元内 p 的线性变化，这才"
          "对应理论阶。")
    print("  判据：q 的阶稳定在 0.95~1.05 就说明组装与边界条件都对了。")
    print()
    return rows, exact_p, exact_q


# =============================================================================
#  第 9 部分：画图（用 Pillow，不依赖 matplotlib）
# =============================================================================
def draw_mesh(nodes, cells, path, size=460, title=""):
    """
    把一个网格画成 PNG。
    做法：把节点坐标线性映射到画布像素，然后逐条边画线段。
    （三角形之间的公共边会被画两次，视觉上没影响。）
    """
    from PIL import Image, ImageDraw, ImageFont

    margin = 34
    # 画布是正方形，坐标范围 [0,1]×[0,1]
    scale = size - 2 * margin

    def to_px(x, y):
        # y 轴要翻转：数学里向上是正，图像里向下是正
        px = margin + x * scale
        py = margin + (1.0 - y) * scale
        return px, py

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # 先画所有三角形
    for tri in cells:
        pts = []
        for v in tri:
            pts.append(to_px(float(nodes[v][0]), float(nodes[v][1])))
        pts.append(pts[0])                  # 闭合
        draw.line(pts, fill=(60, 90, 160), width=1)

    # 标题
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 8), title, fill=(20, 20, 20), font=font)

    img.save(path)


def draw_refinement_figure(base_n=2, levels=3, size=440):
    """
    画一张"网格加密过程"的图：一行四格，分别是加密 0、1、2、3 次。
    """
    from PIL import Image, ImageDraw, ImageFont

    images = []
    # 与收敛实验一致：保留当前网格，每次只向下一层加密一次。
    nodes, cells = make_rect_mesh(base_n, base_n)
    for level in range(levels + 1):
        if level > 0:
            nodes, cells = refine_uniform(nodes, cells)
        path = os.path.join(OUTDIR, "_tmp_level%d.png" % level)
        draw_mesh(nodes, cells, path, size=size,
                  title="加密 %d 次：单元 %d 个" % (level, len(cells)))
        images.append(Image.open(path))

    # 横向拼起来
    total_w = size * len(images)
    canvas = Image.new("RGB", (total_w, size), "white")
    for k in range(len(images)):
        canvas.paste(images[k], (k * size, 0))
    out = os.path.join(OUTDIR, "refinement.png")
    canvas.save(out)

    # 删掉临时文件
    for level in range(levels + 1):
        p = os.path.join(OUTDIR, "_tmp_level%d.png" % level)
        if os.path.exists(p):
            os.remove(p)
    return out


def draw_convergence(rows, path, size=520):
    """
    画 log-log 收敛曲线：横轴 log10(h)，纵轴 log10(误差)。
    同时画两条参考斜率线（斜率 1 和斜率 2）方便比较。
    """
    from PIL import Image, ImageDraw, ImageFont

    margin_left = 70
    margin_bottom = 60
    margin_top = 46
    margin_right = 24

    plot_w = size - margin_left - margin_right
    plot_h = size - margin_top - margin_bottom

    hs = [r["h"] for r in rows]
    eq = [r["err_q"] for r in rows]
    ep = [r["err_p"] for r in rows]

    # log10 坐标
    xs = [math.log10(h) for h in hs]
    ys_q = [math.log10(e) for e in eq]
    ys_p = [math.log10(e) for e in ep]

    x_min = min(xs) - 0.05
    x_max = max(xs) + 0.05
    y_min = min(min(ys_q), min(ys_p)) - 0.25
    y_max = max(max(ys_q), max(ys_p)) + 0.25

    def to_px(x, y):
        px = margin_left + (x - x_min) / (x_max - x_min) * plot_w
        py = margin_top + (y_max - y) / (y_max - y_min) * plot_h
        return px, py

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # 坐标轴
    draw.line([(margin_left, margin_top),
               (margin_left, margin_top + plot_h),
               (margin_left + plot_w, margin_top + plot_h)],
              fill=(40, 40, 40), width=2)

    # 刻度（按整数 log10）
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
        font_small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # 横轴刻度直接标出 h 的分数形式（1/4、1/8……），比 10^-1 直观
    for k in range(len(xs)):
        px, _ = to_px(xs[k], y_min)
        draw.line([(px, margin_top + plot_h), (px, margin_top + plot_h + 4)],
                  fill=(40, 40, 40), width=1)
        label = "1/%d" % int(round(1.0 / hs[k]))
        draw.text((px - 10, margin_top + plot_h + 8), label,
                  fill=(40, 40, 40), font=font_small)
    for yk in range(int(math.floor(y_min)), int(math.ceil(y_max)) + 1):
        if y_min <= yk <= y_max:
            _, py = to_px(x_min, yk)
            draw.line([(margin_left - 4, py), (margin_left, py)],
                      fill=(40, 40, 40), width=1)
            draw.text((10, py - 7), "10^%d" % yk, fill=(40, 40, 40), font=font_small)

    # 数据点与折线
    def draw_series(ys, color, label):
        pts = []
        for k in range(len(xs)):
            pts.append(to_px(xs[k], ys[k]))
        draw.line(pts, fill=color, width=2)
        for p in pts:
            draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=color)
        # 图例
        return pts

    pts_q = draw_series(ys_q, (200, 60, 60), "q")
    pts_p = draw_series(ys_p, (60, 110, 200), "p")

    # 参考斜率 1 的虚线（穿过最后一个数据点）
    x0 = xs[-1]
    y0 = ys_q[-1]
    x1 = xs[0]
    y1 = y0 + (x1 - x0) * 1.0
    p0 = to_px(x0, y0)
    p1 = to_px(x1, y1)
    # 手动画虚线
    steps = 40
    for s in range(steps):
        if s % 2 == 0:
            t0 = s / steps
            t1 = (s + 1) / steps
            draw.line([(p0[0] + (p1[0] - p0[0]) * t0, p0[1] + (p1[1] - p0[1]) * t0),
                       (p0[0] + (p1[0] - p0[0]) * t1, p0[1] + (p1[1] - p0[1]) * t1)],
                      fill=(120, 120, 120), width=1)

    # 文字
    draw.text((margin_left + 10, margin_top - 30), "log-log 收敛曲线", fill=(20, 20, 20), font=font)
    ly = margin_top + plot_h - 66
    # 先画一块白底，免得图例和数据线叠在一起看不清
    draw.rectangle([margin_left + 10, ly - 6, margin_left + 190, ly + 58],
                   fill="white", outline=(200, 200, 200))
    draw.text((margin_left + 16, ly), "红：通量误差 ||q_h-q||", fill=(200, 60, 60), font=font_small)
    draw.text((margin_left + 16, ly + 18), "蓝：压力误差 ||p_h-p||", fill=(60, 110, 200), font=font_small)
    draw.text((margin_left + 16, ly + 36), "灰虚线：参考斜率 1", fill=(120, 120, 120), font=font_small)
    draw.text((margin_left + plot_w // 2 - 20, size - 22), "log10(h)", fill=(20, 20, 20), font=font_small)
    draw.text((8, margin_top + plot_h // 2), "log10(误差)", fill=(20, 20, 20), font=font_small)

    img.save(path)
    return path


def draw_flux(nodes, cells, facets, Q, path, size=520, title=""):
    """
    画通量箭头：每个单元中心画一个小箭头，方向就是 q_h 的方向。
    """
    from PIL import Image, ImageDraw, ImageFont

    margin = 34
    scale = size - 2 * margin

    def to_px(x, y):
        return margin + x * scale, margin + (1.0 - y) * scale

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # 背景网格（很淡）
    for tri in cells:
        pts = []
        for v in tri:
            pts.append(to_px(float(nodes[v][0]), float(nodes[v][1])))
        pts.append(pts[0])
        draw.line(pts, fill=(225, 230, 240), width=1)

    # 每个单元一个箭头
    for c in range(len(cells)):
        cx, cy = cell_center(nodes, cells, c)
        vx, vy = flux_at_point(nodes, cells, facets, c, Q, [cx, cy])
        # 箭头长度固定一点，方便看方向
        length = 0.55 * math.sqrt(1.0 / len(cells))
        norm = math.sqrt(vx * vx + vy * vy)
        if norm < 1e-14:
            continue
        ux = vx / norm * length
        uy = vy / norm * length
        x1, y1 = to_px(cx - ux / 2, cy - uy / 2)
        x2, y2 = to_px(cx + ux / 2, cy + uy / 2)
        draw.line([(x1, y1), (x2, y2)], fill=(30, 90, 160), width=1)
        # 箭头那个小三角
        ang = math.atan2(y2 - y1, x2 - x1)
        for da in (2.6, -2.6):
            xa = x2 + 4.0 * math.cos(ang + da)
            ya = y2 + 4.0 * math.sin(ang + da)
            draw.line([(x2, y2), (xa, ya)], fill=(30, 90, 160), width=1)

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 8), title, fill=(20, 20, 20), font=font)

    img.save(path)
    return path


# =============================================================================
#  主程序：把上面各部分串起来
# =============================================================================
# =============================================================================
#  主程序
# =============================================================================
#  自检清单（改完代码建议逐条过一遍，能抓住 90% 的低级错误）
# =============================================================================
#   1) A 对称吗？          np.allclose(A, A.T)             应为 True
#   2) B 的行和是 0 吗？   np.abs(B.sum(axis=1)).max()     应为 0
#                          （因为 ∫_K div(常向量场) = 0）
#   3) 有精确解时，把精确解插值到自由度上，检查
#        A Φ - B^T P = b      与     B Φ = F
#      的残差，应为 1e-14 量级。不满足就说明组装或边界项错了。
#   4) 无源（f = 0）时 B @ Q 应为 0 —— 这是 RT0 的逐单元守恒性质。
#   5) 均匀流算例的误差应该是"机器零"（1e-14），而不是"看起来很小"（1e-3）。
#   6) 加密后 q 的收敛阶要稳定在 1 附近；掉到 0.5 时先查边界条件和求积精度。
#   7) 压力 Dirichlet 项是 -g_D（棱中点值），**不要乘棱长**（见文档第 8 节）。
# =============================================================================
def _run_all():
    # 建输出目录
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    print()
    print("RT0 混合元求解 Darcy 方程（详解版）")
    print("  scipy 可用：%s   （有 scipy 用稀疏矩阵，没有就用稠密矩阵）" % HAS_SCIPY)
    print()

    # ---- 先看网格长什么样 ----
    print("【网格】")
    for n in (16, 32, 64, 128):
        nodes, cells = make_rect_mesh(n, n)
        print("  %3d×%-3d 网格：节点 %6d 个，单元 %6d 个"
              % (n, n, len(nodes), len(cells)))
    print()

    # 演示"加密一次"对网格规模的影响
    nodes, cells = make_rect_mesh(16, 16)
    print("  加密前：单元 %d" % len(cells))
    for k in range(1, 4):
        nodes, cells = refine_uniform(nodes, cells)
        # 每加密一次，单元数 ×4（因为 h 减半，面积按 h² 缩小）
        print("  加密 %d 次：单元 %d（= 4^%d = %d 倍）"
              % (k, len(cells), k, 4 ** k))
    print()

    # ---- 算例 1 ----
    case1_uniform_flow()

    # ---- 算例 2 ----
    rows, exact_p, exact_q = case2_convergence()

    # ---- 画图 ----
    print("【画图】")
    p1 = draw_refinement_figure(base_n=2, levels=3, size=420)
    print("  网格加密过程：", p1)

    p2 = draw_convergence(rows, os.path.join(OUTDIR, "convergence.png"))
    print("  收敛曲线：    ", p2)

    # 用最细那层画通量箭头
    nodes, cells = make_rect_mesh(4, 4)
    nodes, cells = refine_times(nodes, cells, 2)
    facets = build_facets(nodes, cells)

    def source(x, y):
        pi = math.pi
        return 2.0 * pi * pi * math.sin(pi * x) * math.sin(pi * y)

    dirichlet_edges = []
    for e in range(len(facets["edge_nodes"])):
        if len(facets["edge_cells"][e]) == 1:
            dirichlet_edges.append(e)

    A, B, rhs_q, rhs_p = assemble(nodes, cells, facets, 1.0, source,
                                  lambda x, y: 0.0, dirichlet_edges, [])
    Q, P = solve_saddle(A, B, rhs_q, rhs_p)
    p3 = draw_flux(nodes, cells, facets, Q,
                   os.path.join(OUTDIR, "flux.png"),
                   title="通量 q_h 的方向（正弦解，16×16）")
    print("  通量箭头：    ", p3)
    print()
    print("全部完成。")


def run_report():
    """
    外层入口：把 _run_all() 打印的内容全部捕获下来，
    既显示在屏幕上，也存成一份 UTF-8 文本（方便粘进报告）。
    """
    import io as _io
    import contextlib

    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run_all()
    text = buf.getvalue()

    # 显示
    print(text)

    # 存档（utf-8-sig 带 BOM，Windows 记事本/Excel 打开都不会乱码）
    result_path = os.path.join(HERE, "结果_详解版.txt")
    with open(result_path, "w", encoding="utf-8-sig") as fp:
        fp.write(text)
    print("[结果] 已保存：", result_path)


def main():
    result = solve_rt0(make_rect_mesh(16, 16), element="RT0", dim=2, refinements=0,
                       source=lambda x, y: 2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y))
    print("压力 P =", result["P"])
    print("通量 Q =", result["Q"], "守恒残差 =", result["conservation"])


# 直接运行本文件时执行 main()
if __name__ == "__main__":
    main()
