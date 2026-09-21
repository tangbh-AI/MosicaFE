"""面 (facet) 拓扑与定向：H(div) 型空间的基础设施。

为什么需要单独的一层
--------------------
连续 Lagrange 元与虚拟元只需要"顶点自由度"：单元之间通过共享顶点耦合。
而最低阶 Raviart--Thomas 元（RT0）的自由度是**面法向通量**

.. math::

   \\Phi_f=\\int_f q\\cdot n_f\\,\\mathrm ds,

它必须满足 :math:`H(\\mathrm{div})` 协调性，即**相邻两个单元共享同一个未知量**。
因此"哪些单元共享同一个面""这个面的全局法向指向谁"必须显式给出，
这就是本模块提供的 :class:`FacetTopology`。

约定（与 :mod:`MosicaFE.core.utils` 一致）
------------------------------------------
* 单元的**第 i 个局部面** = "不含第 i 个顶点"的那个面
  （2D 是边，3D 是三角面）；于是 2D 有 3 个面、3D 有 4 个面；
* 每个面用**升序顶点表**作为全局唯一键，这个顺序同时定义了全局法向
  :math:`n_f`（由 :func:`facet_vector_normals` 给出，模长 = 面的测度）；
* ``signs[c, i] = +1`` 表示全局法向恰好指向单元 ``c`` 的外部，
  ``-1`` 表示相反。于是单元内部的外法向通量与全局未知量满足
  :math:`F_{c,i}=\\sigma_{c,i}\\,\\Phi_{f(c,i)}`。

库内使用方式
------------
.. code-block:: python

   from MosicaFE.core.facets import build_facet_topology

   topo = build_facet_topology(mesh)      # 只支持单纯形网格（三角/四面体）
   topo.n_facets, topo.n_boundary         # 面总数、边界面数
   topo.cell_facets[0], topo.signs[0]     # 单元 0 的局部面编号与外法向符号

拓扑是惰性构建的：:class:`~MosicaFE.core.mesh.Mesh` 也提供 ``mesh.facet_topology``
这一入口，同一个网格只构建一次。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .utils import TET_FACES

__all__ = [
    "FacetTopology",
    "build_facet_topology",
    "local_facets",
    "simplex_element_array",
    "facet_vector_normals",
]

#: 单纯形单元的局部面：第 i 个面 = "不含第 i 个顶点" 的面。
_LOCAL_FACETS = {
    2: ((1, 2), (0, 2), (0, 1)),   # 3 条边，与顶点 0/1/2 相对
    3: tuple(TET_FACES),           # 4 个三角面，与顶点 0/1/2/3 相对
}


def local_facets(dim: int) -> Tuple[Tuple[int, ...], ...]:
    """返回 ``dim`` 维单纯形的局部面连接表。

    ``dim=2``：``((1,2),(0,2),(0,1))``（第 i 个面是与顶点 i 相对的边）；
    ``dim=3``：``((1,2,3),(0,2,3),(0,1,3),(0,1,2))``（与顶点 i 相对的面）。

    与 :data:`MosicaFE.core.utils.TET_FACES` 完全一致，因此 Lagrange/VEM
    与 RT0 使用同一套面编号约定。
    """
    try:
        return _LOCAL_FACETS[int(dim)]
    except KeyError:  # pragma: no cover - 防御性分支
        raise ValueError(
            f"面拓扑目前只支持单纯形网格（dim=2 三角形 / dim=3 四面体），收到 dim={dim}"
        ) from None


def simplex_element_array(mesh) -> np.ndarray:
    """把网格的连接表取成 ``(n_cells, d+1)`` 的整型数组。

    非单纯形网格（四边形、多边形、六面体……）会抛出 ``ValueError``：
    RT0 要求 :math:`H(\\mathrm{div})` 协调的单纯形剖分。
    """
    d = int(mesh.dim)
    expected = d + 1
    cells = [np.asarray(e, dtype=np.int64).ravel() for e in mesh.elements]
    if any(c.size != expected for c in cells):
        sizes = sorted({int(c.size) for c in cells})
        raise ValueError(
            f"{d}D 单纯形网格的每个单元应有 {expected} 个顶点，"
            f"当前单元顶点数为 {sizes}；请使用 Mesh.rectangle(element_type='triangle') "
            f"或 Mesh.box(element_type='tet')。"
        )
    return np.vstack(cells)


def facet_vector_normals(facet_points: np.ndarray) -> np.ndarray:
    """由**有序**顶点表给出面的向量面积 :math:`\\int_f n\\,\\mathrm ds`。

    参数
    ----
    facet_points : (n_facets, d, d)
        2D 时每个面是 2 个点，3D 时是 3 个点。

    返回
    ----
    (n_facets, d) ndarray，方向为面法向、模长等于面的测度（边长 / 面积）。

    该定义对顶点顺序是**交替**的（交换两个顶点，法向变号）；由于全局面
    统一用升序顶点，相邻单元得到的方向恰好相反，这正是确定 ``signs`` 的依据。
    """
    p = np.asarray(facet_points, dtype=float)
    if p.ndim != 3 or p.shape[1] != p.shape[2]:
        raise ValueError(f"facet_points 形状应为 (n_facets, d, d)，收到 {p.shape}")
    d = p.shape[2]
    if d == 2:
        t = p[:, 1, :] - p[:, 0, :]
        return np.stack([-t[:, 1], t[:, 0]], axis=1)
    if d == 3:
        return 0.5 * np.cross(
            p[:, 1, :] - p[:, 0, :], p[:, 2, :] - p[:, 0, :]
        )
    raise ValueError(f"只支持 2D / 3D，收到维度 {d}")


class FacetTopology:
    """单纯形网格的面拓扑与定向。

    Attributes
    ----------
    facets : (n_facets, d) int
        每个面的**升序**顶点索引（全局唯一键）。
    cell_facets : (n_cells, d+1) int
        ``cell_facets[c, i]`` = 单元 ``c`` 第 ``i`` 个局部面的全局编号。
    signs : (n_cells, d+1) float, ±1
        ``+1`` 表示全局法向指向单元外部；``-1`` 表示指向内部。
    normals : (n_facets, d) float
        未归一化的全局法向（模长 = 面测度）。
    counts : (n_facets,) int
        相邻单元个数：内部面 2，边界面 1。
    facet_cells : tuple[tuple[int, ...], ...]
        每个面的相邻单元编号。
    """

    __slots__ = (
        "facets", "cell_facets", "signs", "normals", "counts",
        "facet_cells", "n_cells", "dim", "n_facets", "n_flux",
    )

    def __init__(
        self,
        facets: np.ndarray,
        cell_facets: np.ndarray,
        signs: np.ndarray,
        normals: np.ndarray,
        counts: np.ndarray,
        facet_cells: Tuple[Tuple[int, ...], ...],
        n_cells: int,
        dim: int,
    ) -> None:
        self.facets = facets
        self.cell_facets = cell_facets
        self.signs = signs
        self.normals = normals
        self.counts = counts
        self.facet_cells = facet_cells
        self.n_cells = int(n_cells)
        self.dim = int(dim)
        self.n_facets = int(facets.shape[0])
        #: RT0 的通量自由度与面一一对应，保留这个别名便于对照文献
        self.n_flux = self.n_facets

    # -- 便捷属性 --------------------------------------------------------
    @property
    def boundary_mask(self) -> np.ndarray:
        """边界面（只属于一个单元）的布尔掩码，长度 ``n_facets``。"""
        return self.counts == 1

    @property
    def interior_mask(self) -> np.ndarray:
        """内部面（被两个单元共享）的布尔掩码。"""
        return self.counts == 2

    @property
    def n_boundary(self) -> int:
        return int(np.count_nonzero(self.counts == 1))

    @property
    def n_interior(self) -> int:
        return int(np.count_nonzero(self.counts == 2))

    def boundary_facets(self) -> np.ndarray:
        """边界面的编号数组（升序）。"""
        return np.flatnonzero(self.counts == 1)

    def interior_facets(self) -> np.ndarray:
        """内部面的编号数组（升序）。"""
        return np.flatnonzero(self.counts == 2)

    def measure(self) -> np.ndarray:
        """每个面的测度（2D 周长 / 3D 面积）。"""
        return np.linalg.norm(self.normals, axis=1)

    def summary(self) -> str:
        return (
            f"FacetTopology(dim={self.dim}, cells={self.n_cells}, "
            f"facets={self.n_facets}, interior={self.n_interior}, "
            f"boundary={self.n_boundary})"
        )

    def __repr__(self) -> str:  # pragma: no cover - 仅展示
        return self.summary()


def build_facet_topology(mesh) -> FacetTopology:
    """由单纯形 :class:`~MosicaFE.core.mesh.Mesh` 构建面拓扑。

    步骤与每一步的理由：

    1. 把每个单元的局部面写成**升序顶点表**，以此作为全局唯一键
       （升序使其与单元的顶点顺序无关，相邻单元自然配对）；
    2. 由升序顶点表定义全局法向；因为它是交替的，
       两个相邻单元得到的法向恰好相反；
    3. 用"面中心 - 单元中心"与全局法向的点积判定 ``signs``；
    4. 统计每个面的相邻单元个数，超过 2 说明网格不协调，直接报错
       （RT0 在不协调网格上会**静默给出错误解**，所以必须拦住）。
    """
    elements = simplex_element_array(mesh)
    nodes = np.asarray(mesh.nodes, dtype=float)
    d = int(mesh.dim)
    n = d + 1
    n_cells = elements.shape[0]

    # 1) 局部面 -> 升序全局顶点表
    keys = np.empty((n_cells, n, d), dtype=np.int64)
    for i, cols in enumerate(local_facets(d)):
        keys[:, i, :] = np.sort(elements[:, cols], axis=1)
    flat = keys.reshape(-1, d)
    facets, inverse = np.unique(flat, axis=0, return_inverse=True)
    cell_facets = np.asarray(inverse).reshape(-1).reshape(n_cells, n)

    # 2) 全局法向
    normals = facet_vector_normals(nodes[facets])

    # 3) 定向符号
    cell_centers = nodes[elements].mean(axis=1)                  # (n_cells, d)
    facet_centers = nodes[facets].mean(axis=1)                   # (n_facets, d)
    outward = facet_centers[cell_facets] - cell_centers[:, None, :]
    dots = np.einsum("cfd,cfd->cf", normals[cell_facets], outward)
    scale = np.maximum(np.linalg.norm(outward, axis=-1), 1e-300) * np.maximum(
        np.linalg.norm(normals[cell_facets], axis=-1), 1e-300
    )
    if np.any(np.abs(dots) <= 1e-12 * scale):
        bad = np.argwhere(np.abs(dots) <= 1e-12 * scale)
        raise ValueError(
            "网格含退化单元（面中心与单元中心重合，无法判定外法向），"
            f"位置（单元, 局部面）= {bad[:5].tolist()}"
        )
    signs = np.where(dots > 0.0, 1.0, -1.0)

    # 4) 相邻单元统计
    counts = np.bincount(cell_facets.ravel(), minlength=facets.shape[0])
    if np.any(counts > 2):
        raise ValueError(
            "网格不协调：存在被 3 个及以上单元共用的面，RT0 需要 H(div) 协调的单纯形剖分"
        )
    order = np.argsort(cell_facets.ravel(), kind="stable")
    flat_cells = np.repeat(np.arange(n_cells), n)[order]
    groups = np.split(flat_cells, np.cumsum(counts)[:-1])
    facet_cells = tuple(tuple(int(c) for c in g) for g in groups)

    return FacetTopology(
        facets=facets,
        cell_facets=cell_facets,
        signs=signs,
        normals=normals,
        counts=counts,
        facet_cells=facet_cells,
        n_cells=n_cells,
        dim=d,
    )
