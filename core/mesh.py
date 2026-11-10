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

# 模块职责：网格的存储、拓扑和生成
# nodes 保存物理坐标，elements 保存单元的节点编号；面、邻接关系和边界由这些数据推导。
# 本文件集中存放 Mesh 类、所有网格工厂及其数组生成函数。
# 阅读顺序：Mesh.xxx 工厂 → 文件后部的同名 *_arrays 算法 → 公共数组工具。
# 基础几何量仍复用 utils.py；SciPy 只在 Voronoi 生成时导入。


from __future__ import annotations

import numbers
from functools import lru_cache

import numpy as np

from .utils import (
    HEX_FACES,
    TET_FACES,
    cell_centroid,
    cell_diameter,
    cell_measure,
)

__all__ = ["Mesh"]

_TRI_ALIASES = ("tri", "triangle", "simplex")
_TET_ALIASES = ("tet", "tetra", "tetrahedron", "simplex3d")


def _index_array(row):
    """将一行连接表转换为 int64；相同类型的数组不复制。"""
    return np.asarray(row, dtype=np.int64).ravel()


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
        三维面的局部顶点编号，可给公共模板 faces[面][顶点]，
        或逐单元给 faces[单元][面][顶点]，允许单元顶点数、面数不同。
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
        # 把坐标和变长连接表整理成数值数组，识别逐单元面列表，再准备拓扑缓存和边界信息。
        self.nodes = np.ascontiguousarray(np.asarray(nodes, dtype=float))
        if self.nodes.ndim != 2:
            raise ValueError("nodes 必须是 (n_nodes, dim) 的二维数组")
        # 规则生成器返回等长二维数组：直接按行建立视图，不再逐行转换类型。
        # 自定义、Voronoi 和混合网格可能为变长列表，保留兼容转换路径。
        if isinstance(elements, np.ndarray) and elements.ndim == 2:
            self.elements = list(np.asarray(elements, dtype=np.int64))
        else:
            self.elements = list(map(_index_array, elements))
        if len(self.elements) == 0:
            raise ValueError("网格至少需要一个单元")
        self.element_type = str(element_type).lower()
        self.name = name
        self._faces = faces
        self._per_cell_faces = None
        if faces is not None:
            if not len(faces) or not len(faces[0]):
                raise ValueError('faces 不能为空')
            if not np.isscalar(faces[0][0]):
                if len(faces) != len(self.elements):
                    raise ValueError('逐单元 faces 的长度必须等于单元数')
                self._per_cell_faces = faces
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
    # 工作块：统一几何取向
    # 二维单元按逆时针排列，三维四面体调整为正体积方向；这样后续面积符号、外法向和积分公式可以使用一致约定。
    def _orient_elements(self):
        """批量统一二维边界方向和三维四面体正体积。"""
        lengths = np.fromiter(map(len, self.elements), dtype=int, count=len(self.elements))
        if self.dim == 2:
            # 变长多边形先打平；reduceat 对每段累计鞋带公式。
            flat = np.concatenate(self.elements)
            offsets = np.r_[0, np.cumsum(lengths)[:-1]]
            following = np.arange(len(flat))+1
            following[np.cumsum(lengths)-1] = offsets
            points = self.nodes[flat]
            origins = np.repeat(points[offsets], lengths, axis=0)
            a, b = points-origins, points[following]-origins
            area = np.add.reduceat(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0], offsets)
            # 仅对顺时针段反转索引；不逐单元求面积。
            position = np.arange(len(flat))-np.repeat(offsets, lengths)
            reverse = np.repeat(offsets+lengths-1, lengths)-position
            source = np.where(np.repeat(area < 0, lengths), reverse, np.arange(len(flat)))
            self.elements = list(np.split(flat[source], np.cumsum(lengths)[:-1]))
        elif self.dim == 3 and self._faces is None:
            if self.element_type not in _TET_ALIASES + ('polyhedron',):
                return
            selected = np.flatnonzero(lengths == 4)
            if not len(selected):
                return
            # 一次计算所有四面体的有向体积；仅写回需要换向的单元。
            rows = np.stack(list(map(self.elements.__getitem__, selected)))
            xyz = self.nodes[rows]
            negative = np.linalg.det(xyz[:, 1:]-xyz[:, :1]) < 0
            # 全部为四面体时批量修正；一般混合拓扑仅写回需要改变的条目。
            rows[negative] = rows[negative][:, [0, 2, 1, 3]]
            if len(selected) == len(self.elements):
                self.elements = list(rows)
            else:
                for position in np.flatnonzero(negative):
                    self.elements[selected[position]] = rows[position]

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        """空间维数。"""
        return int(self.nodes.shape[1])

    @property
    def n_nodes(self) -> int:
        # 返回几何顶点数量；高阶空间增加的边自由度和积分矩不计入这里。
        return int(self.nodes.shape[0])

    #: guidebook / VEM 老代码里的叫法
    @property
    def n_vertices(self) -> int:
        # 兼容旧接口的顶点计数名称，数值与 n_nodes 一致。
        return self.n_nodes

    @property
    def n_cells(self) -> int:
        # 返回单元数量，二维表示多边形数量，三维表示体单元数量。
        return len(self.elements)

    #: VEM 老代码里的叫法
    @property
    def n_elements(self) -> int:
        # 兼容旧接口的单元计数名称，数值与 n_cells 一致。
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
    def get_cell_faces(self, cell_idx):
        """三维单元的面，返回单元内局部顶点编号；显式 faces 优先于形状推断。"""
        cell = self.elements[cell_idx]
        if self._per_cell_faces is not None:
            faces = self._per_cell_faces[cell_idx]
        elif self._faces is not None:
            faces = self._faces
        elif len(cell) == 4:
            faces = TET_FACES
        elif len(cell) == 8:
            faces = HEX_FACES
        else:
            raise ValueError('三维多面体需提供 faces[单元][面][局部顶点]')
        result = [tuple(int(i) for i in f) for f in faces]
        if any(len(f) < 3 or len(set(f)) != len(f) or min(f) < 0 or max(f) >= len(cell)
               for f in result):
            raise ValueError('每个面至少含三个不同的有效局部顶点，且沿边界排列')
        return result

    def _cell_facets(self, cell, cell_idx=None):
        """返回单元 cell 的 facet 列表（每个 facet 是排序后的顶点元组）。"""
        et = self.element_type
        if self.dim == 1:
            return [(int(v),) for v in cell]
        if self.dim == 2:
            n = len(cell)
            return [tuple(sorted((int(cell[i]), int(cell[(i + 1) % n])))) for i in range(n)]
        # 3D
        if cell_idx is None:
            cell_idx = next(i for i, e in enumerate(self.elements) if np.array_equal(e, cell))
        return [tuple(sorted(int(cell[i]) for i in f)) for f in self.get_cell_faces(cell_idx)]

    # 工作块：从单元连接关系建立面邻接表
    # 枚举各单元的面，将同一个几何面归为一组并记录所属单元；只出现一次的面构成边界，出现两次的面连接相邻单元。
    def _build_facets(self):
        if self._facets is not None:
            return
        facets = []
        owners = {}
        for c, cell in enumerate(self.elements):
            for f in self._cell_facets(cell, c):
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
        coords = self.get_cell_vertices(cell_idx)
        if self.dim == 3:
            # 用实际面剖分，不能从顶点数量猜测任意多面体的体积。
            center = coords.mean(axis=0)
            volume = 0.0
            for face in self.get_cell_faces(cell_idx):
                fv = coords[list(face)]
                fc = fv.mean(axis=0)
                for i in range(len(fv)):
                    J = np.array([fc-center, fv[i]-center, fv[(i+1)%len(fv)]-center])
                    volume += abs(np.linalg.det(J))/6.0
            return float(volume)
        return cell_measure(coords)

    def get_cell_diameter(self, cell_idx: int) -> float:
        """单元直径。"""
        return cell_diameter(self.get_cell_vertices(cell_idx))

    def get_cell_centroid(self, cell_idx: int) -> np.ndarray:
        """单元形心。"""
        return cell_centroid(self.get_cell_vertices(cell_idx))

    def cell_measures(self) -> np.ndarray:
        # 逐单元计算面积或体积，结果按单元编号排列，可用于积分及总区域测度检查。
        return np.array([self.get_cell_measure(c) for c in range(self.n_cells)])

    def cell_centroids(self) -> np.ndarray:
        # 逐单元收集形心坐标，供几何查询、显示与单元数据定位使用。
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
        # 返回对象的可读说明，方便交互查看；此方法只组织展示信息。
        return self.summary()

    # ------------------------------------------------------------------
    # 细化
    # ------------------------------------------------------------------
    # 工作块：创建更细的网格
    # 根据网格保存的生成信息增加细分数量，重新构造同一区域的网格；它返回新的网格，供网格加密实验使用。
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
    def interval(cls, a=0., b=1., nx=10):
        """批量生成区间节点与相邻端点，无逐单元循环。"""
        _integer(nx, 'nx')
        _bounds((a, b), 1)
        nodes = np.linspace(a, b, nx+1)[:, None]
        elements = np.column_stack((np.arange(nx), np.arange(1, nx+1)))
        mesh = cls(nodes, elements, 'interval', name=f'interval({nx})')
        mesh._spec = dict(factory='interval', a=a, b=b, nx=nx)
        return mesh

    # 工作块：生成矩形区域的结构网格
    # 先在两个坐标方向生成节点，再按网格小块构造连接表；根据 element_type 保留四边形或拆为三角形，最后返回 Mesh 对象。
    @classmethod
    def rectangle(cls, xmin=0., xmax=1., ymin=0., ymax=1., nx=8, ny=8,
                  element_type='triangle'):
        """四个数组切片建立全部四边形；三角形由固定模板批量拆分。"""
        nodes, elements = rectangle_arrays(nx, ny, (xmin, xmax, ymin, ymax))
        et = str(element_type).lower()
        if et in _TRI_ALIASES:
            elements = elements[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
            et = 'triangle'
        elif et in ('quad', 'quadrilateral'):
            et = 'quad'
        else:
            raise ValueError("rectangle 的 element_type 只能是 'triangle' 或 'quad'")
        mesh = cls(nodes, elements, et, name=f'rectangle[{et}]({nx}x{ny})')
        mesh._spec = dict(factory='rectangle', xmin=xmin, xmax=xmax, ymin=ymin,
                          ymax=ymax, nx=nx, ny=ny, element_type=et)
        return mesh

    # 工作块：生成长方体区域的结构网格
    # 先生成三维节点，再按所选单元和剖分模式建立六面体或四面体连接；相邻小块必须使用兼容的面剖分。
    @classmethod
    def box(cls, xmin=0., xmax=1., ymin=0., ymax=1., zmin=0., zmax=1.,
            nx=4, ny=4, nz=4, element_type='tet', pattern='kuhn'):
        """八角点切片生成六面体；按模板生成六/十二个四面体。"""
        et = str(element_type).lower()
        if et in _TET_ALIASES:
            et = 'tet'
        elif et in ('hex', 'hexahedron'):
            et = 'hex'
        else:
            raise ValueError("box 的 element_type 只能是 'tet' 或 'hex'")
        nodes, elements = box_arrays(nx, ny, nz, (xmin, xmax, ymin, ymax, zmin, zmax), et, pattern)
        mesh = cls(nodes, elements, et, name=f'box[{et}]({nx}x{ny}x{nz})')
        mesh._spec = dict(factory='box', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                          zmin=zmin, zmax=zmax, nx=nx, ny=ny, nz=nz, element_type=et, pattern=pattern)
        return mesh

    # ------------------------------------------------------------------
    # 工厂：多边形网格
    # ------------------------------------------------------------------
    # 工作块：通过 Voronoi 划分生成多边形网格
    # 从区域内种子点构造各自的近邻区域，并将多边形限制在目标边界内；合并共享顶点后建立统一节点编号。
    @classmethod
    def voronoi_polygon(cls, n_points=60, seed=42, bbox=(0., 1., 0., 1.)):
        """单平面镜像闭合单元；全局批量去重，保留一次单元遍历。"""
        nodes, elements = voronoi_polygon_arrays(n_points, seed, bbox)
        return cls(nodes, elements, 'polygon', name=f'voronoi({len(elements)})')

    @classmethod
    def pentagon(cls, nx=8, ny=8):
        """批量建立棋盘式五顶点单元及唯一共享中点，含共线边点。"""
        nodes, elements = pentagon_arrays(nx, ny)
        mesh = cls(nodes, elements, 'polygon', name=f'pentagon({nx}x{ny})')
        mesh._spec = dict(factory='pentagon', nx=nx, ny=ny)
        return mesh

    # ------------------------------------------------------------------
    # 三维 Voronoi 多面体：每个单元保存自己的面，不要求公共拓扑模板。
    @classmethod
    def voronoi_polyhedron(cls, n_points=24, seed=42, bbox=(0., 1., 0., 1., 0., 1.)):
        """生成长方体内协调的非结构三维 Voronoi 网格，需要 SciPy。"""
        nodes, elements, faces = voronoi_arrays(n_points, seed, bbox)
        return cls(nodes, elements, 'polyhedron', faces=faces,
                   name=f'voronoi3d({n_points})')

    @classmethod
    def mixed_polygon(cls, nx=8, ny=8, max_sides=8, jitter=.15, seed=42,
                      bbox=(0., 1., 0., 1.), concave=True):
        """生成可直接用于 VEM 的混合多边形网格，仅需 NumPy。

        nx、ny 为合并前的矩形分格数；实际单元数由随机合并决定。
        max_sides >= 3 控制每个单元最多的边数，3 表示纯三角形。
        jitter 为内部节点扰动占格宽的比例，范围 [0, .25)。
        concave=True 允许星形凹单元；False 只接受凸单元（含共线边点）。
        固定 seed 可复现；不同边数按随机合并产生，不保证每次出现所有边数。
        """
        nodes, cells = polygon_arrays(nx, ny, max_sides, jitter, seed, bbox, concave)
        return cls(nodes, cells, 'polygon', name=f'mixed2d({nx}x{ny})')

    @classmethod
    def mixed_polyhedron(cls, nx=4, ny=4, nz=2, max_sides=8, jitter=.15,
                         split=.35, seed=42, bbox=(0., 1., 0., 1., 0., 1.),
                         concave=True):
        """生成协调的棱柱、棱锥、四面体混合网格，仅需 NumPy。

        挤出 mixed_polygon 的多边形，再将比例约为 split 的母单元分锥。
        split=0 保留全部棱柱；split=1 全部分锥。共享面不会被独立切开。
        单元及面的顶点数可变，均满足当前 VEM 所需的星形几何条件。
        其余参数与 mixed_polygon 相同；nz 为竖直层数。
        """
        nodes, cells, faces = polyhedron_arrays(
            nx, ny, nz, max_sides, jitter, split, seed, bbox, concave)
        return cls(nodes, cells, 'polyhedron', faces=faces,
                   name=f'mixed3d({nx}x{ny}x{nz})')

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
        # 等长数组在 Mesh 构造阶段一次处理；变长输入在这里整理以便推断类型。
        if not (isinstance(elements, np.ndarray) and elements.ndim == 2):
            elements = list(map(_index_array, elements))
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




# ==========================================================================
# 公共数组工具与规则网格生成（零显式循环）
# ==========================================================================

def _integer(value, name, minimum=1):
    """分格数必须是整数；排除容易被误当作整数的布尔值。"""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} 必须是大于等于 {minimum} 的整数')


def _bounds(bbox, dim):
    """把 xmin,xmax,... 整理为各方向下界 lo、上界 hi。"""
    data = np.asarray(bbox, dtype=float)
    if data.shape != (2*dim,) or not np.isfinite(data).all():
        raise ValueError(f'bbox 必须包含 {2*dim} 个有限实数')
    data = data.reshape(dim, 2)
    if np.any(data[:, 1] <= data[:, 0]):
        raise ValueError('bbox 每个方向的上界必须大于下界')
    return data[:, 0], data[:, 1]


def rectangle_arrays(nx, ny, bbox):
    """meshgrid 生成坐标；四个切片同时取出每个小格的角点。"""
    _integer(nx, 'nx')
    _integer(ny, 'ny')
    lo, hi = _bounds(bbox, 2)
    # 坐标广播成 (ny+1, nx+1) 的节点阵列，展平后 x 下标变化最快。
    x, y = np.meshgrid(np.linspace(lo[0], hi[0], nx+1),
                       np.linspace(lo[1], hi[1], ny+1), indexing='xy')
    nodes = np.column_stack((x.ravel(), y.ravel()))
    ids = np.arange(len(nodes)).reshape(ny+1, nx+1)
    # 每个切片形状均为 (ny,nx)。stack 后为 (ny,nx,4)，
    # reshape 后每行就是一个单元：逆时针左下、右下、右上、左上。
    cells = np.stack((ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]),
                     axis=-1).reshape(-1, 4)
    return nodes, cells


def box_arrays(nx, ny, nz, bbox, element_type, pattern):
    """八个切片生成六面体；局部模板批量拆成四面体。"""
    _integer(nx, 'nx')
    _integer(ny, 'ny')
    _integer(nz, 'nz')
    lo, hi = _bounds(bbox, 3)
    x, y, z = np.meshgrid(np.linspace(lo[0], hi[0], nx+1),
                          np.linspace(lo[1], hi[1], ny+1),
                          np.linspace(lo[2], hi[2], nz+1), indexing='ij')
    nodes = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    ids = np.arange(len(nodes)).reshape(nx+1, ny+1, nz+1)
    # 八个切片代替 i/j/k 三层循环，前四点是底面，后四点是对应顶面。
    cubes = np.stack((ids[:-1, :-1, :-1], ids[1:, :-1, :-1],
                      ids[1:, 1:, :-1], ids[:-1, 1:, :-1],
                      ids[:-1, :-1, 1:], ids[1:, :-1, 1:],
                      ids[1:, 1:, 1:], ids[:-1, 1:, 1:]), axis=-1)
    # 保持原代码 k→j→i 的单元编号顺序，节点仍按 i→j→k 编号。
    cubes = cubes.transpose(2, 1, 0, 3).reshape(-1, 8)
    if element_type == 'hex':
        return nodes, cubes
    if pattern != 'center':
        # 每行是一个四面体在母六面体中的局部顶点编号。
        # cubes[:,template] 同时处理所有单元，结果形状为 (体数,6,4)。
        template = np.array([[0,1,2,6], [0,1,5,6], [0,3,2,6],
                             [0,3,7,6], [0,4,5,6], [0,4,7,6]])
        return nodes, cubes[:, template].reshape(-1, 4)
    # 十二个边界三角形分别连接母单元中心。
    template = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],
                         [0,1,5],[0,5,4],[3,2,6],[3,6,7],
                         [0,3,7],[0,7,4],[1,2,6],[1,6,5]])
    centers = .5*(nodes[cubes[:, 0]]+nodes[cubes[:, 6]])
    center_ids = len(nodes)+np.arange(len(cubes))
    # broadcast_to 不复制中心编号；连接时扩展成 (体数,12,1)，
    # 与十二个三角面拼接为 (体数,12,4)，即十二个四面体。
    apex = np.broadcast_to(center_ids[:, None, None], (len(cubes), 12, 1))
    cells = np.concatenate((apex, cubes[:, template]), axis=-1).reshape(-1, 4)
    return np.vstack((nodes, centers)), cells


def pentagon_arrays(nx, ny):
    """批量建立共享边中点；棋盘奇偶掩码选择五顶点连接模板。"""
    nodes, quads = rectangle_arrays(nx, ny, (0., 1., 0., 1.))
    j, i = np.divmod(np.arange(nx*ny), nx)
    # (i+j) 的奇偶决定在上边还是下边插入中点；共享边的键必须相同。
    even = (i+j) % 2 == 0
    edge_row = j+even
    # 同一水平边只分配一个中点；按首次出现排序，保持旧节点编号。
    keys, first, inverse = np.unique(edge_row*nx+i, return_index=True, return_inverse=True)
    # unique 批量去重。first 用于恢复旧算法的首次出现编号顺序，
    # inverse 将每个单元映射回对应的共享中点，避免逐边字典登记。
    order = np.argsort(first)
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    mids = len(nodes)+rank[inverse]
    row, col = np.divmod(keys[order], nx)
    points = np.column_stack(((col+.5)/nx, row/ny))
    cells = np.column_stack((quads[:, 0], np.where(even, quads[:, 1], mids),
                             np.where(even, quads[:, 2], quads[:, 1]),
                             np.where(even, mids, quads[:, 2]), quads[:, 3]))
    return np.vstack((nodes, points)), cells


def split_rows(flat, lengths):
    """扁平编号按段长度恢复为变长单元/面表，避免逐项 append。"""
    return np.split(flat, np.cumsum(lengths)[:-1])


def compact(nodes, cells):
    """一次删除孤立节点并重排全局编号，避免零刚度自由度。"""
    lengths = np.fromiter(map(len, cells), dtype=int, count=len(cells))
    used, inverse = np.unique(np.concatenate(cells), return_inverse=True)
    # used 是保留节点的旧编号；inverse 是全部连接表的新编号。
    # 节点顺序改变只影响全局编号，变长面表中的单元局部位置保持不变。
    return nodes[used], split_rows(inverse, lengths)


def mirrored_sites(sites, lo, hi):
    """批量生成各边界平面的镜像：二维 5N 点、三维 7N 点。

    自身种子的单平面镜像已经把单元限制在盒内，组合方向的镜像不再
    缩小盒内原始单元，因此不需要原来的二维 9N / 三维 27N 个点。
    """
    dim, count = sites.shape[1], len(sites)
    axes = np.repeat(np.arange(dim), 2)
    planes = np.column_stack((lo, hi)).ravel()
    blocks = np.broadcast_to(sites, (2*dim, count, dim)).copy()
    # 第 2*a 和 2*a+1 块分别沿坐标轴 a 的下、上边界反射。
    # 每个反射只修改一个坐标分量，其余坐标直接沿用原种子。
    blocks[np.arange(2*dim)[:, None], np.arange(count)[None, :], axes[:, None]] = (
        2*planes[:, None]-sites[:, axes].T)
    return np.vstack((sites, blocks.reshape(-1, dim)))


def bounded_vertices(vertices, lo, hi):
    """只登记盒内顶点，归一化舍入后批量去重，并建立旧→新编号。"""
    scale = np.max(hi-lo)
    mask = np.all((vertices >= lo-1e-8*scale) & (vertices <= hi+1e-8*scale), axis=1)
    raw = np.clip(vertices[mask], lo, hi)
    _, first, inverse = np.unique(np.round((raw-lo)/scale, 11), axis=0,
                                  return_index=True, return_inverse=True)
    mapping = np.full(len(vertices), -1, dtype=int)
    # -1 标记区域外顶点；原始单元引用到这些点时会明确报错。
    mapping[mask] = inverse
    return raw[first], mapping


# ==========================================================================
# 二维、三维 Voronoi 生成（SciPy 按需导入）
# ==========================================================================

def _voronoi(sites, lo, hi):
    """仅生成 Voronoi 时才导入可选的 SciPy。"""
    try:
        from scipy.spatial import Voronoi
    except ImportError as exc:
        raise ImportError('生成 Voronoi 网格需要 scipy') from exc
    return Voronoi(mirrored_sites(sites, lo, hi))


def _face_ids(ridge, mapping):
    """验证有限面并批量去掉重复几何顶点。"""
    raw = np.asarray(ridge, dtype=int)
    if not len(raw) or np.any(raw < 0):
        raise ValueError('Voronoi 出现未闭合单元，请更换 seed')
    ids = np.unique(mapping[raw])
    if ids[0] < 0:
        raise ValueError('Voronoi 单元超出边界，请更换 seed')
    return ids


def voronoi_polygon_arrays(n_points, seed, bbox):
    """原种子 + 四个单平面镜像；仅遍历变长单元边界。"""
    _integer(n_points, 'n_points')
    lo, hi = _bounds(bbox, 2)
    sites = lo+np.random.default_rng(seed).random((n_points, 2))*(hi-lo)
    vor = _voronoi(sites, lo, hi)
    nodes, mapping = bounded_vertices(vor.vertices, lo, hi)
    cells = []
    for region in vor.point_region[:n_points]:
        ids = _face_ids(vor.regions[region], mapping)
        if len(ids) < 3:
            raise ValueError('Voronoi 出现退化多边形')
        xy = nodes[ids]-nodes[ids].mean(axis=0)
        cells.append(ids[np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))])
    return compact(nodes, cells)


def _pack_cell(cell_faces):
    """扁平全局面编号转换为单元顶点集合和面局部编号。"""
    lengths = np.fromiter(map(len, cell_faces), dtype=int, count=len(cell_faces))
    vertices, local = np.unique(np.concatenate(cell_faces), return_inverse=True)
    # inverse 给出每个全局面顶点在 vertices 中的位置，恢复分段后
    # 就是 faces[单元][面][局部顶点]；不再逐点查询字典。
    return vertices, split_rows(local, lengths)


def _pack_cells(global_faces):
    """每个体单元仅整理一次，公共面已共享。"""
    elements, faces = [], []
    for cell_faces in global_faces:
        vertices, local_faces = _pack_cell(cell_faces)
        elements.append(vertices)
        faces.append(local_faces)
    return elements, faces


def voronoi_arrays(n_points, seed, bbox):
    """三维保留真实多边形面，返回节点、单元、局部面表。"""
    _integer(n_points, 'n_points')
    lo, hi = _bounds(bbox, 3)
    sites = lo+(.05+.9*np.random.default_rng(seed).random((n_points, 3)))*(hi-lo)
    vor = _voronoi(sites, lo, hi)
    nodes, mapping = bounded_vertices(vor.vertices, lo, hi)
    # 每个原始种子分配一个独立的面列表；共享面只在下面构造一次。
    global_faces = list(map(list, np.empty((n_points, 0), dtype=int)))
    # 批量过滤：只遍历至少连接一个原始种子的面，排除纯镜像区域。
    selected = np.flatnonzero(np.any(vor.ridge_points < n_points, axis=1))
    for number in selected:
        pair = vor.ridge_points[number]
        ids = _face_ids(vor.ridge_vertices[number], mapping)
        if len(ids) < 3:
            raise ValueError('Voronoi 出现退化面')
        # 两种子连线为公共面的法向；删除法向最大分量对应的坐标，
        # 即得稳定面内投影，省去逐面的 SVD。
        normal = vor.points[pair[1]]-vor.points[pair[0]]
        xy = np.delete(nodes[ids]-nodes[ids].mean(axis=0), np.argmax(abs(normal)), axis=1)
        face = ids[np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))]
        # 一张面最多连接两个原单元，不需要内层遍历。
        if pair[0] < n_points:
            global_faces[pair[0]].append(face)
        if pair[1] < n_points:
            global_faces[pair[1]].append(face)
    elements, faces = _pack_cells(global_faces)
    nodes, elements = compact(nodes, elements)
    return nodes, elements, faces


# ==========================================================================
# 混合多边形、多面体生成（局部合并与模板缓存）
# ==========================================================================

def _merge(left, right):
    """集合批量消去反向公共边，再沿唯一后继恢复闭合边界。"""
    left, right = list(left), list(right)
    a = set(zip(left, left[1:]+left[:1]))
    b = set(zip(right, right[1:]+right[:1]))
    ar = set(zip(left[1:]+left[:1], left))
    br = set(zip(right[1:]+right[:1], right))
    edges = (a-br) | (b-ar)
    following = dict(edges)
    if len(edges) < 3 or len(following) != len(edges) or len(set(following.values())) != len(edges):
        return None
    start = min(following)
    current, ring, seen = start, [], set()
    for _ in range(len(edges)):
        if current in seen or current not in following:
            return None
        seen.add(current)
        ring.append(current)
        current = following[current]
    return ring if current == start else None


def _admissible(points, concave):
    """中心位于所有有向边左侧就是星形核条件；按需再检查凸性。"""
    xy = points-points.mean(axis=0)
    nxt = np.roll(xy, -1, axis=0)
    tolerance = 1e-10*np.sum(np.ptp(xy, axis=0)**2)
    if np.any(xy[:, 0]*nxt[:, 1]-xy[:, 1]*nxt[:, 0] <= tolerance):
        return False
    if not concave:
        edges = nxt-xy
        after = np.roll(edges, -1, axis=0)
        if np.any(edges[:, 0]*after[:, 1]-edges[:, 1]*after[:, 0] < -tolerance):
            return False
    return True


def _triangle_neighbors(triangles):
    """全局无向边排序，一次找到公共边两侧三角形，无逐边字典。"""
    # 一个三角形的三条有向边依次为 (v0,v1)、(v1,v2)、(v2,v0)。
    edges = np.stack((triangles, np.roll(triangles, -1, axis=1)), axis=-1).reshape(-1, 2)
    edges.sort(axis=1)
    # 基础三角剖分中一条边最多出现两次。排序后，公共边恰好相邻，
    # 无需 unique 的逆映射和第二次排序；边记录编号 // 3 就是三角形编号。
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered = edges[order]
    shared = np.all(ordered[:-1] == ordered[1:], axis=1)
    a, b = order[:-1][shared], order[1:][shared]
    neighbors = np.full(3*len(triangles), -1, dtype=int)
    neighbors[a], neighbors[b] = b//3, a//3
    return neighbors.reshape(-1, 3)


def _try_neighbor(ring, candidates, triangles, points, max_sides, concave):
    """候选邻单元逐个检查；只计算当前边界的小规模几何。"""
    for t in candidates:
        merged = _merge(ring, triangles[t])
        if (merged is not None and len(merged) <= max_sides
                and _admissible(points[merged], concave)):
            return merged, t
    return None


def _grow(start, target, triangles, neighbors, assigned, points, rng, max_sides, concave):
    """只维护当前单元的局部前沿；已占用三角形不会再次使用。"""
    assigned[start] = True
    ring = triangles[start].tolist()
    frontier = set(neighbors[start][neighbors[start] >= 0])
    for _ in range(target-1):
        candidates = np.asarray(sorted(frontier), dtype=int)
        candidates = candidates[~assigned[candidates]]
        found = _try_neighbor(ring, rng.permutation(candidates), triangles, points, max_sides, concave)
        if found is None:
            break
        ring, t = found
        assigned[t] = True
        frontier.discard(t)
        frontier.update(neighbors[t][neighbors[t] >= 0])
    return ring


def polygon_arrays(nx, ny, max_sides, jitter, seed, bbox, concave):
    """批量三角化与邻接 + 必要的局部合并；输出紧凑变长连接表。"""
    _integer(max_sides, 'max_sides', 3)
    if not np.isfinite(jitter) or not 0 <= jitter < .25:
        raise ValueError('jitter 必须满足 0 <= jitter < 0.25')
    lo, hi = _bounds(bbox, 2)
    points, quads = rectangle_arrays(nx, ny, (0., 1., 0., 1.))
    rng = np.random.default_rng(seed)
    interior = ((points > 0) & (points < 1)).all(axis=1)
    points[interior] += rng.uniform(-jitter, jitter, (interior.sum(), 2))/[nx, ny]
    # 随机对角线的两种模板一次应用到全部基础小格。
    templates = np.array([[[0,1,3],[1,2,3]], [[0,1,2],[0,2,3]]])
    choice = rng.integers(2, size=len(quads))
    triangles = np.take_along_axis(quads[:, None, :], templates[choice], axis=2).reshape(-1, 3)
    neighbors = _triangle_neighbors(triangles)
    assigned = np.zeros(len(triangles), dtype=bool)
    cells = []
    for start in rng.permutation(len(triangles)):
        if not assigned[start]:
            target = int(rng.integers(1, max_sides-1))
            cells.append(_grow(start, target, triangles, neighbors, assigned, points,
                               rng, max_sides, concave))
    return compact(lo+points*(hi-lo), cells)


@lru_cache(maxsize=128)
def _prism_faces(n):
    """相同底面边数复用局部面模板，不为每个棱柱重建拓扑。"""
    i = np.arange(n)
    sides = np.column_stack((i, (i+1) % n, (i+1) % n+n, i+n))
    return (i[::-1], i+n, *sides)


@lru_cache(maxsize=128)
def _pyramid_faces(n):
    """一个 n 边形底面和 n 个三角形侧面，锥顶局部编号为 n。"""
    i = np.arange(n)
    sides = np.column_stack((i, (i+1) % n, np.full(n, n)))
    return (i, *sides)


def _split_prism(vertices, boundary, center, elements, faces):
    """公共底面保持不变；分锥只在母单元内部增加三角面。"""
    for face in boundary:
        elements.append(np.r_[vertices[face], center])
        faces.append(_pyramid_faces(len(face)))


def polyhedron_arrays(nx, ny, nz, max_sides, jitter, split, seed, bbox, concave):
    """批量复制层节点、预分配中心；一次母单元遍历生成混合多面体。"""
    _integer(nz, 'nz')
    if not np.isfinite(split) or not 0 <= split <= 1:
        raise ValueError('split 必须满足 0 <= split <= 1')
    lo, hi = _bounds(bbox, 3)
    xy, polygons = polygon_arrays(nx, ny, max_sides, jitter, seed, bbox[:4], concave)
    z = np.linspace(lo[2], hi[2], nz+1)
    base = np.column_stack((np.tile(xy, (nz+1, 1)), np.repeat(z, len(xy))))
    count = nz*len(polygons)
    selected = np.random.default_rng(seed).random(count) < split
    # 所有中心一次分配内存，避免反复扩展坐标列表。
    nodes = np.empty((len(base)+np.count_nonzero(selected), 3))
    nodes[:len(base)] = base
    center_ids = len(base)+np.cumsum(selected)-1
    # 同一底面在各层的中心 xy 坐标相同，先按变长分段批量求均值，
    # 再一次写入所有被选中母单元的中心坐标。主循环只处理连接关系。
    lengths = np.fromiter(map(len, polygons), dtype=int, count=len(polygons))
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    centers_xy = np.add.reduceat(xy[np.concatenate(polygons)], offsets, axis=0)/lengths[:, None]
    chosen_layers, chosen_cells = np.divmod(np.flatnonzero(selected), len(polygons))
    nodes[len(base):, :2] = centers_xy[chosen_cells]
    nodes[len(base):, 2] = .5*(z[chosen_layers]+z[chosen_layers+1])
    elements, faces = [], []
    for parent in range(count):
        layer, c = divmod(parent, len(polygons))
        ring = np.asarray(polygons[c])
        bottom = ring+layer*len(xy)
        vertices = np.r_[bottom, bottom+len(xy)]
        boundary = _prism_faces(len(ring))
        if selected[parent]:
            center = center_ids[parent]
            _split_prism(vertices, boundary, center, elements, faces)
        else:
            elements.append(vertices)
            faces.append(boundary)
    return nodes, elements, faces
