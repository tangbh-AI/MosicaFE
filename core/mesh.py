"""网格数据结构与常见区域的结构化/多边形网格生成。

设计要点
--------
* ``Mesh.elements`` 是**变长**的顶点索引列表的列表（``list[np.ndarray]``），
  因此三角形、四边形、五边形、Voronoi 多边形、四面体、六面体可以共存于同一套
  组装循环里。若所有单元顶点数相同，可以额外取出 ``mesh.uniform_vertices``
  这个 ``(n_cells, k)`` 的 ndarray 看。
* ``facets`` 是"余维 1 的单元面"：1D 是端点、2D 是边、3D 是三角/四边形面。
  只属于一个单元的 facet 就是边界，其顶点集合即 Dirichlet 边界自由度所在节点。

生成的网格全部使用 **逆时针（2D）/ 正体积（3D）** 取向。
"""

from __future__ import annotations

import numpy as np

from .utils import (
    HEX_FACES,
    TET_FACES,
    cell_centroid,
    cell_diameter,
    cell_measure,
    signed_polygon_area,
)

__all__ = ["Mesh"]

_TRI_ALIASES = ("tri", "triangle", "simplex")
_TET_ALIASES = ("tet", "tetra", "tetrahedron", "simplex3d")


class Mesh:
    """网格。

    Parameters
    ----------
    nodes : (n_nodes, dim) array_like
    elements : sequence of sequence of int
        每个单元的顶点索引。变长。
    element_type : str
        ``"interval"`` / ``"triangle"`` / ``"quad"`` / ``"tet"`` / ``"hex"`` /
        ``"polygon"`` / ``"polyhedron"``。
    boundary_nodes : array_like, optional
        显式指定边界节点；缺省时由 facet 拓扑自动判定。
    faces : sequence, optional
        仅用于 3D 任意多面体：每个单元的面（顶点索引子列表）。
    name : str
        便于在日志里识别。
    """

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    def __init__(
        self,
        nodes,
        elements,
        element_type: str = "polygon",
        boundary_nodes=None,
        faces=None,
        name: str = "",
    ):
        self.nodes = np.ascontiguousarray(np.asarray(nodes, dtype=float))
        if self.nodes.ndim != 2:
            raise ValueError("nodes 必须是 (n_nodes, dim) 的二维数组")
        self.elements = [
            np.asarray(e, dtype=np.int64).ravel() for e in elements
        ]
        if len(self.elements) == 0:
            raise ValueError("网格至少需要一个单元")
        self.element_type = str(element_type).lower()
        self.name = name
        self._faces = faces
        self._spec = None
        self._facets = None
        self._facet_to_cells = None
        self._facet_topology = None

        self._orient_elements()

        self._boundary_nodes = (
            np.asarray(boundary_nodes, dtype=np.int64).ravel()
            if boundary_nodes is not None
            else None
        )

    # -- 取向修正 --------------------------------------------------------
    def _orient_elements(self):
        """把 2D 单元统一为逆时针、3D 四面体统一为正体积。"""
        et = self.element_type
        if self.dim == 2:
            for k, e in enumerate(self.elements):
                if signed_polygon_area(self.nodes[e]) < 0:
                    self.elements[k] = e[::-1].copy()
        elif self.dim == 3 and (et in _TET_ALIASES) or (
            self.dim == 3 and et == "polyhedron"
        ):
            for k, e in enumerate(self.elements):
                if len(e) == 4:
                    a, b, c, d = self.nodes[e]
                    if np.linalg.det(np.array([b - a, c - a, d - a])) < 0:
                        self.elements[k] = e[[0, 2, 1, 3]].copy()

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        """空间维数。"""
        return int(self.nodes.shape[1])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    #: guidebook / VEM 老代码里的叫法
    @property
    def n_vertices(self) -> int:
        return self.n_nodes

    @property
    def n_cells(self) -> int:
        return len(self.elements)

    #: VEM 老代码里的叫法
    @property
    def n_elements(self) -> int:
        return self.n_cells

    @property
    def vertices_per_cell(self):
        """每个单元的顶点数（list）。"""
        return [int(len(e)) for e in self.elements]

    @property
    def uniform_vertices(self):
        """所有单元顶点数相同时返回 ``(n_cells, k)`` 数组，否则 ``None``。"""
        k = len(self.elements[0])
        if all(len(e) == k for e in self.elements):
            return np.vstack([np.asarray(e, dtype=np.int64) for e in self.elements])
        return None

    # ------------------------------------------------------------------
    # facet 拓扑与边界
    # ------------------------------------------------------------------
    def _cell_facets(self, cell):
        """返回单元 cell 的 facet 列表（每个 facet 是排序后的顶点元组）。"""
        et = self.element_type
        if self.dim == 1:
            return [(int(v),) for v in cell]
        if self.dim == 2:
            n = len(cell)
            return [tuple(sorted((int(cell[i]), int(cell[(i + 1) % n])))) for i in range(n)]
        # 3D
        if len(cell) == 4:
            return [tuple(sorted(int(cell[i]) for i in f)) for f in TET_FACES]
        if len(cell) == 8 and et in ("hex", "hexahedron", "polyhedron"):
            return [tuple(sorted(int(cell[i]) for i in f)) for f in HEX_FACES]
        if self._faces is not None:
            return [tuple(sorted(int(cell[i]) for i in f)) for f in self._faces]
        raise ValueError(
            "无法自动推导该 3D 单元的 facet 拓扑；请通过 faces=... 显式给出每个单元的面，"
            "或使用受支持的六面体/四面体顶点顺序。"
        )

    def _build_facets(self):
        if self._facets is not None:
            return
        facets = []
        owners = {}
        for c, cell in enumerate(self.elements):
            for f in self._cell_facets(cell):
                facets.append(f)
                owners.setdefault(f, []).append(c)
        self._facets = facets
        self._facet_to_cells = owners

    @property
    def facets(self):
        """所有 facet（顶点索引元组的列表）。"""
        self._build_facets()
        return self._facets

    def facet_owner(self, facet):
        """返回共享某个 facet 的单元编号列表。"""
        self._build_facets()
        return self._facet_to_cells[facet]

    @property
    def boundary_facets(self):
        """只属于一个单元的 facet。"""
        self._build_facets()
        return [f for f, cells in self._facet_to_cells.items() if len(cells) == 1]

    @property
    def interior_facets(self):
        """被两个单元共享的 facet。"""
        self._build_facets()
        return [f for f, cells in self._facet_to_cells.items() if len(cells) == 2]

    @property
    def boundary_nodes(self):
        """边界节点索引（升序，去重）。"""
        if self._boundary_nodes is None:
            idx = set()
            for f in self.boundary_facets:
                idx.update(int(v) for v in f)
            self._boundary_nodes = np.array(sorted(idx), dtype=np.int64)
        return self._boundary_nodes

    @property
    def boundary_edges(self):
        """2D 边界边 ``(n, 2)``；等价于边界 facet。"""
        if self.dim != 2:
            raise AttributeError("boundary_edges 只对 2D 网格有定义")
        return np.array([list(f) for f in self.boundary_facets], dtype=np.int64)

    @property
    def facet_topology(self):
        """面拓扑与定向（惰性构建并缓存，见 :mod:`MosicaFE.core.facets`）。

        RT0 这类 :math:`H(\\mathrm{div})` 协调空间用它拿到
        "全局面编号 + 相邻单元 + 外法向符号"。只对单纯形网格可用。
        """
        if self._facet_topology is None:
            from .facets import build_facet_topology

            self._facet_topology = build_facet_topology(self)
        return self._facet_topology

    def vertex_boundary_mask(self) -> np.ndarray:
        """``(n_nodes,)`` 布尔数组，True 表示该节点在边界上。"""
        mask = np.zeros(self.n_nodes, dtype=bool)
        mask[self.boundary_nodes] = True
        return mask

    # ------------------------------------------------------------------
    # 几何量
    # ------------------------------------------------------------------
    def get_cell_vertices(self, cell_idx: int) -> np.ndarray:
        """第 cell_idx 个单元的顶点坐标。"""
        return self.nodes[self.elements[cell_idx]]

    def get_cell_measure(self, cell_idx: int) -> float:
        """单元面积 / 体积 / 长度。"""
        return cell_measure(self.get_cell_vertices(cell_idx))

    def get_cell_diameter(self, cell_idx: int) -> float:
        """单元直径。"""
        return cell_diameter(self.get_cell_vertices(cell_idx))

    def get_cell_centroid(self, cell_idx: int) -> np.ndarray:
        """单元形心。"""
        return cell_centroid(self.get_cell_vertices(cell_idx))

    def cell_measures(self) -> np.ndarray:
        return np.array([self.get_cell_measure(c) for c in range(self.n_cells)])

    def cell_centroids(self) -> np.ndarray:
        return np.vstack([self.get_cell_centroid(c) for c in range(self.n_cells)])

    def get_mesh_size(self) -> float:
        """最大单元直径 h。"""
        return max(self.get_cell_diameter(c) for c in range(self.n_cells))

    def summary(self) -> str:
        """一行文字摘要，便于日志。"""
        return (
            f"<Mesh {self.name or self.element_type} dim={self.dim} "
            f"nodes={self.n_nodes} cells={self.n_cells} "
            f"boundary_nodes={len(self.boundary_nodes)} h={self.get_mesh_size():.4g}>"
        )

    def __repr__(self):  # pragma: no cover - 仅展示
        return self.summary()

    # ------------------------------------------------------------------
    # 细化
    # ------------------------------------------------------------------
    def refine_uniform(self) -> "Mesh":
        """均匀加密（把每个方向的网格数翻倍）。

        仅对结构化网格（interval / rectangle / box）可用；
        对 Voronoi 等非结构网格会抛出 ``ValueError``。
        """
        if not self._spec:
            raise ValueError(
                "该网格不是由结构化工厂函数生成的，无法自动 refine_uniform()；"
                "请直接生成更细的网格。"
            )
        spec = dict(self._spec)
        fn = spec.pop("factory")
        for key in ("nx", "ny", "nz"):
            if key in spec:
                spec[key] = 2 * spec[key]
        return getattr(Mesh, fn)(**spec)

    # ------------------------------------------------------------------
    # 工厂：结构化网格
    # ------------------------------------------------------------------
    @classmethod
    def interval(cls, a: float = 0.0, b: float = 1.0, nx: int = 10) -> "Mesh":
        """一维区间 ``[a,b]`` 上的均分网格。"""
        x = np.linspace(a, b, nx + 1)
        nodes = x[:, None]
        elements = [[i, i + 1] for i in range(nx)]
        mesh = cls(nodes, elements, "interval", name=f"interval({nx})")
        mesh._spec = {"factory": "interval", "a": a, "b": b, "nx": nx}
        return mesh

    @classmethod
    def rectangle(
        cls,
        xmin: float = 0.0,
        xmax: float = 1.0,
        ymin: float = 0.0,
        ymax: float = 1.0,
        nx: int = 8,
        ny: int = 8,
        element_type: str = "triangle",
    ) -> "Mesh":
        """矩形区域网格。``element_type`` 取 ``"triangle"``(默认) 或 ``"quad"``。"""
        et = str(element_type).lower()
        xs = np.linspace(xmin, xmax, nx + 1)
        ys = np.linspace(ymin, ymax, ny + 1)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        nodes = np.column_stack([X.ravel(), Y.ravel()])

        def nid(i, j):
            return j * (nx + 1) + i

        elements = []
        if et in _TRI_ALIASES:
            for j in range(ny):
                for i in range(nx):
                    n1, n2 = nid(i, j), nid(i + 1, j)
                    n3, n4 = nid(i, j + 1), nid(i + 1, j + 1)
                    elements.append([n1, n2, n4])
                    elements.append([n1, n4, n3])
            et = "triangle"
        elif et in ("quad", "quadrilateral"):
            for j in range(ny):
                for i in range(nx):
                    elements.append(
                        [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
                    )
            et = "quad"
        else:
            raise ValueError("rectangle 的 element_type 只能是 'triangle' 或 'quad'")

        mesh = cls(nodes, elements, et, name=f"rectangle[{et}]({nx}x{ny})")
        mesh._spec = {
            "factory": "rectangle",
            "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
            "nx": nx, "ny": ny, "element_type": et,
        }
        return mesh

    @classmethod
    def box(
        cls,
        xmin: float = 0.0,
        xmax: float = 1.0,
        ymin: float = 0.0,
        ymax: float = 1.0,
        zmin: float = 0.0,
        zmax: float = 1.0,
        nx: int = 4,
        ny: int = 4,
        nz: int = 4,
        element_type: str = "tet",
        pattern: str = "kuhn",
    ) -> "Mesh":
        """长方体区域网格。

        ``element_type="tet"`` 时每个小立方体剖成 6 个四面体（``pattern="kuhn"``，
        与 RT0 的 ``box_tet_mesh`` 一致）或 12 个四面体（``pattern="center"``，
        额外插入立方体中心点，与 VEM 3D 老脚本一致）。
        ``element_type="hex"`` 时每个小立方体就是一个六面体单元。
        """
        et = str(element_type).lower()
        xs = np.linspace(xmin, xmax, nx + 1)
        ys = np.linspace(ymin, ymax, ny + 1)
        zs = np.linspace(zmin, zmax, nz + 1)
        grids = np.meshgrid(xs, ys, zs, indexing="ij")
        nodes = np.column_stack([g.ravel() for g in grids])

        def nid(i, j, k):
            return (k * (ny + 1) + j) * (nx + 1) + i

        elements = []
        if et in _TET_ALIASES:
            if pattern == "center":
                nodes, elements = _box_tet_center(
                    nodes, nid, nx, ny, nz
                )
            else:
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            elements.extend(_kuhn_tets(nid, i, j, k))
            et = "tet"
        elif et in ("hex", "hexahedron"):
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        elements.append(
                            [
                                nid(i, j, k), nid(i + 1, j, k),
                                nid(i + 1, j + 1, k), nid(i, j + 1, k),
                                nid(i, j, k + 1), nid(i + 1, j, k + 1),
                                nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                            ]
                        )
            et = "hex"
        else:
            raise ValueError("box 的 element_type 只能是 'tet' 或 'hex'")

        mesh = cls(nodes, elements, et, name=f"box[{et}]({nx}x{ny}x{nz})")
        mesh._spec = {
            "factory": "box",
            "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
            "zmin": zmin, "zmax": zmax,
            "nx": nx, "ny": ny, "nz": nz, "element_type": et, "pattern": pattern,
        }
        return mesh

    # ------------------------------------------------------------------
    # 工厂：多边形网格
    # ------------------------------------------------------------------
    @classmethod
    def voronoi_polygon(
        cls,
        n_points: int = 60,
        seed: int = 42,
        bbox=(0.0, 1.0, 0.0, 1.0),
    ) -> "Mesh":
        """矩形区域内的 Voronoi 多边形网格（VEM 的理想网格）。

        通过在 ``bbox`` 外做 8 方向反射来保证边界处的 Voronoi 单元闭合，
        再裁剪回区域并对顶点排序，得到逆时针多边形。
        """
        try:
            from scipy.spatial import Voronoi
        except Exception as exc:  # pragma: no cover - 取决于环境
            raise ImportError(
                "生成 Voronoi 网格需要 scipy，请先安装 scipy 或改用 "
                "Mesh.pentagon()/Mesh.rectangle()。"
            ) from exc

        xmin, xmax, ymin, ymax = bbox
        rng = np.random.default_rng(seed)
        interior = rng.random((n_points, 2))
        interior[:, 0] = xmin + interior[:, 0] * (xmax - xmin)
        interior[:, 1] = ymin + interior[:, 1] * (ymax - ymin)

        reflected = []
        for x, y in interior:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    rx = x if dx == 0 else (2 * xmin - x if dx < 0 else 2 * xmax - x)
                    ry = y if dy == 0 else (2 * ymin - y if dy < 0 else 2 * ymax - y)
                    reflected.append([rx, ry])
        all_points = np.vstack([interior, np.asarray(reflected)])
        vor = Voronoi(all_points)

        vert_map = {}
        all_verts = []

        def get_or_add(v):
            key = (round(float(v[0]), 10), round(float(v[1]), 10))
            if key not in vert_map:
                vert_map[key] = len(all_verts)
                all_verts.append(v)
            return vert_map[key]

        polygons = []
        for i in range(n_points):
            region = vor.regions[vor.point_region[i]]
            if len(region) < 3 or -1 in region:
                continue
            clipped = _clip_polygon(vor.vertices[region], bbox)
            clipped = _dedupe_consecutive(clipped)
            if len(clipped) < 3:
                continue
            center = clipped.mean(axis=0)
            angles = np.arctan2(clipped[:, 1] - center[1], clipped[:, 0] - center[0])
            clipped = clipped[np.argsort(angles)]
            polygons.append([get_or_add(v) for v in clipped])

        nodes = np.asarray(all_verts)
        mesh = cls(nodes, polygons, "polygon", name=f"voronoi({len(polygons)})")
        mesh._spec = None  # 非结构网格，不支持自动细化
        return mesh

    @classmethod
    def pentagon(cls, nx: int = 8, ny: int = 8) -> "Mesh":
        """棋盘格式五边形网格（每个矩形交替切上边 / 下边）。

        水平边中点按需创建，保证相邻单元严格协调、没有孤立节点。
        """
        nodes = []

        def grid_id(i, j):
            return j * (nx + 1) + i

        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes.append([i / nx, j / ny])

        mp_map = {}

        def get_midpoint(i, j_row):
            key = (i, j_row)
            if key not in mp_map:
                mp_map[key] = len(nodes)
                nodes.append([(i + 0.5) / nx, j_row / ny])
            return mp_map[key]

        elements = []
        for j in range(ny):
            for i in range(nx):
                n1, n2 = grid_id(i, j), grid_id(i + 1, j)
                n3, n4 = grid_id(i + 1, j + 1), grid_id(i, j + 1)
                if (i + j) % 2 == 0:
                    m = get_midpoint(i, j + 1)
                    elements.append([n1, n2, n3, m, n4])
                else:
                    m = get_midpoint(i, j)
                    elements.append([n1, m, n2, n3, n4])

        mesh = cls(np.asarray(nodes), elements, "polygon", name=f"pentagon({nx}x{ny})")
        mesh._spec = {"factory": "pentagon", "nx": nx, "ny": ny}
        return mesh

    # ------------------------------------------------------------------
    # 从数组构造
    # ------------------------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        nodes,
        elements,
        element_type: str = None,
        boundary_nodes=None,
        faces=None,
        name: str = "custom",
    ) -> "Mesh":
        """从用户给定的节点/单元数组构造网格（自动推断单元类型）。"""
        nodes = np.asarray(nodes, dtype=float)
        elements = [np.asarray(e, dtype=np.int64).ravel() for e in elements]
        if element_type is None:
            k = len(elements[0])
            d = nodes.shape[1]
            element_type = {
                (1, 2): "interval",
                (2, 3): "triangle",
                (2, 4): "quad",
                (3, 4): "tet",
                (3, 8): "hex",
            }.get((d, k), "polygon" if d == 2 else "polyhedron")
        return cls(
            nodes, elements, element_type,
            boundary_nodes=boundary_nodes, faces=faces, name=name,
        )


# ---------------------------------------------------------------------------
# 模块级辅助函数
# ---------------------------------------------------------------------------
def _kuhn_tets(nid, i, j, k):
    """把立方体 (i,j,k) 剖成 6 个共享主对角线的四面体（Kuhn 剖分，面协调）。"""
    from itertools import permutations

    corner = nid(i, j, k)
    top = nid(i + 1, j + 1, k + 1)
    base = np.array([i, j, k])
    tets = []
    for perm in permutations(range(3)):
        cur = base.copy()
        path = [corner]
        for axis in perm:
            cur = cur.copy()
            cur[axis] += 1
            path.append(nid(int(cur[0]), int(cur[1]), int(cur[2])))
        path[-1] = top
        tets.append(path)
    return tets


def _box_tet_center(nodes, nid, nx, ny, nz):
    """每个小立方体插入中心点并剖成 12 个四面体（与 VEM 3D 老脚本一致）。"""
    from itertools import product

    nodes = list(nodes)
    n_grid = (nx + 1) * (ny + 1) * (nz + 1)

    def center_id(i, j, k):
        return n_grid + (k * ny + j) * nx + i

    corners_of = lambda i, j, k: {
        (di, dj, dk): nid(i + di, j + dj, k + dk)
        for di, dj, dk in product((0, 1), repeat=3)
    }

    for k, j, i in product(range(nz), range(ny), range(nx)):
        c = corners_of(i, j, k)
        nodes.append(
            np.mean([nodes[idx] for idx in c.values()], axis=0).tolist()
        )

    elements = []
    for k, j, i in product(range(nz), range(ny), range(nx)):
        c = corners_of(i, j, k)
        ctr = center_id(i, j, k)
        for tri in _cube_face_triangles(c):
            elements.append([ctr, tri[0], tri[1], tri[2]])
    return np.asarray(nodes), elements


def _cube_face_triangles(c):
    """立方体 6 个面各自剖成 2 个三角形（对角方向一致，保证相邻单元协调）。"""
    return [
        [c[(0, 0, 0)], c[(1, 0, 0)], c[(1, 1, 0)]],
        [c[(0, 0, 0)], c[(1, 1, 0)], c[(0, 1, 0)]],
        [c[(0, 0, 1)], c[(1, 0, 1)], c[(1, 1, 1)]],
        [c[(0, 0, 1)], c[(1, 1, 1)], c[(0, 1, 1)]],
        [c[(0, 0, 0)], c[(1, 0, 0)], c[(1, 0, 1)]],
        [c[(0, 0, 0)], c[(1, 0, 1)], c[(0, 0, 1)]],
        [c[(0, 1, 0)], c[(1, 1, 0)], c[(1, 1, 1)]],
        [c[(0, 1, 0)], c[(1, 1, 1)], c[(0, 1, 1)]],
        [c[(0, 0, 0)], c[(0, 1, 0)], c[(0, 1, 1)]],
        [c[(0, 0, 0)], c[(0, 1, 1)], c[(0, 0, 1)]],
        [c[(1, 0, 0)], c[(1, 1, 0)], c[(1, 1, 1)]],
        [c[(1, 0, 0)], c[(1, 1, 1)], c[(1, 0, 1)]],
    ]


def _clip_polygon(poly, bbox):
    """Sutherland-Hodgman 多边形裁剪到 ``bbox``。"""
    xmin, xmax, ymin, ymax = bbox
    out = np.asarray(poly, dtype=float)

    def clip_edge(pts, edge):
        if len(pts) == 0:
            return pts
        result = []
        n = len(pts)
        for i in range(n):
            p1, p2 = pts[i], pts[(i + 1) % n]
            in1, in2 = _inside(p1, edge, bbox), _inside(p2, edge, bbox)
            if in1 and in2:
                result.append(p2)
            elif in1 and not in2:
                result.append(_intersect(p1, p2, edge, bbox))
            elif not in1 and in2:
                result.append(_intersect(p1, p2, edge, bbox))
                result.append(p2)
        return np.asarray(result) if result else np.zeros((0, 2))

    for edge in range(4):
        out = clip_edge(out, edge)
        if len(out) == 0:
            break
    return out


def _inside(p, edge, bbox):
    """点是否在裁剪框某条边的内侧。"""
    xmin, xmax, ymin, ymax = bbox
    return (
        p[0] >= xmin if edge == 0 else
        p[0] <= xmax if edge == 1 else
        p[1] >= ymin if edge == 2 else
        p[1] <= ymax
    )


def _dedupe_consecutive(poly, tol: float = 1e-9):
    """去掉多边形里相邻（含首尾）重合的顶点，避免退化单元。"""
    pts = np.asarray(poly, dtype=float)
    if len(pts) == 0:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > tol:
            keep.append(i)
    out = pts[keep]
    while len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= tol:
        out = out[:-1]
    return out


def _intersect(p1, p2, edge, bbox):
    xmin, xmax, ymin, ymax = bbox
    if edge in (0, 1):
        x = xmin if edge == 0 else xmax
        denom = p2[0] - p1[0]
        t = (x - p1[0]) / denom if abs(denom) > 1e-14 else 0.0
        return np.array([x, p1[1] + t * (p2[1] - p1[1])])
    y = ymin if edge == 2 else ymax
    denom = p2[1] - p1[1]
    t = (y - p1[1]) / denom if abs(denom) > 1e-14 else 0.0
    return np.array([p1[0] + t * (p2[0] - p1[0]), y])
