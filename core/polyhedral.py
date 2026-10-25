"""三维 Voronoi 网格生成：镜像种子把单元限制在长方体内。

保留 Voronoi 的多边形面，并合并全局顶点；不会把多面体转换成求解用四面体。
返回 nodes、elements 和 faces[单元][面][局部顶点]，连接原有 Mesh 接口。
"""

# 模块职责：输入盒子范围和种子点，依次建立镜像点、Voronoi 面、共享顶点以及逐单元局部面列表。


from itertools import product

import numpy as np


def voronoi_arrays(n_points, seed, bbox):
    # 输入种子数量、随机种子和盒子边界，返回全局节点、变长单元以及逐单元局部面列表。
    from scipy.spatial import Voronoi

    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)) or n_points < 1:
        raise ValueError('n_points 必须是正整数')
    # bbox 按 xmin,xmax,ymin,ymax,zmin,zmax 排列；固定随机种子使同一组参数能重建同一张网格。
    bounds = np.asarray(bbox, dtype=float).reshape(3, 2)
    lo, hi = bounds[:, 0], bounds[:, 1]
    if np.any(hi <= lo):
        raise ValueError('bbox 每个方向的上界必须大于下界')
    rng = np.random.default_rng(seed)
    sites = lo + (.05 + .9*rng.random((n_points, 3))) * (hi-lo)

    # 原种子在前，随后添加各坐标平面的镜像，保证原单元恰好被盒子边界截断。
    blocks = [sites]
    for signs in product((-1, 0, 1), repeat=3):
        if signs == (0, 0, 0):
            continue
        block = sites.copy()
        for axis, side in enumerate(signs):
            if side:
                plane = lo[axis] if side < 0 else hi[axis]
                block[:, axis] = 2*plane-sites[:, axis]
        blocks.append(block)
    # 原始种子排在前 n_points 个位置，镜像点只负责截断盒子边界；下面只为原始种子收集求解单元。
    vor = Voronoi(np.vstack(blocks))
    global_faces = [[] for _ in range(n_points)]
    nodes, lookup = [], {}
    scale = np.max(hi-lo)

    def node_id(vertex):
        # 将 Voronoi 顶点转换为唯一全局编号；只修正边界附近的舍入误差，明显越界时立即报错。
        point = vor.vertices[vertex]
        if np.any(point < lo-1e-8*scale) or np.any(point > hi+1e-8*scale):
            raise ValueError('Voronoi 单元超出边界，请更换 seed')
        point = np.clip(point, lo, hi)
        key = tuple(np.round((point-lo)/scale, 11))
        if key not in lookup:
            lookup[key] = len(nodes)
            nodes.append(point)
        return lookup[key]

    # ridge 是两个种子之间的一张公共多边形面；若两个种子都在原区域，使用同一组全局顶点登记到两个单元。
    for pair, ridge in zip(vor.ridge_points, vor.ridge_vertices):
        owners = [int(i) for i in pair if i < n_points]
        if not owners:
            continue
        if -1 in ridge or len(ridge) < 3:
            raise ValueError('Voronoi 出现未闭合面，请更换 seed')
        ids = list(dict.fromkeys(node_id(i) for i in ridge))
        if len(ids) < 3:
            raise ValueError('Voronoi 出现退化面，请更换 seed')
        xyz = np.array([nodes[i] for i in ids])
        # Voronoi 返回面的顶点集合；先投影到面内二维坐标，再按极角排序，得到 VEM 所需的有序边界环。
        centered = xyz-xyz.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        xy = centered @ vt[:2].T
        order = np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))
        face = [ids[i] for i in order]
        for owner in owners:
            global_faces[owner].append(face)

    # 先收集每个体单元使用的全局顶点，再把各面的全局编号改写为该单元内的局部编号；相邻单元仍通过 nodes 共享顶点。
    elements, faces = [], []
    for cell_faces in global_faces:
        vertices = sorted({v for f in cell_faces for v in f})
        local = {v: i for i, v in enumerate(vertices)}
        elements.append(vertices)
        faces.append([[local[v] for v in f] for f in cell_faces])
    return np.array(nodes), elements, faces
