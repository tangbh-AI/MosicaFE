"""连续 Lagrange 有限元空间（FEM 的"元"）。

支持：

=================  ===================
网格单元           可用多项式次数
=================  ===================
``interval``       任意 k>=1
``triangle``       任意 k>=1
``quad``           任意 k>=1（张量积 Qk）
``tet``            任意 k>=1
``hex``            任意 k>=1（张量积 Qk）
=================  ===================

自由度通过顶点编号及节点的整数权重在相邻单元之间共享，
因此任意阶 Pk/Qk 空间在协调网格上是 H1 协调的。
"""

from __future__ import annotations

import numpy as np

# reference_dimension 返回参考单元的空间维数；下面四个工具负责参考域到物理域的几何转换。
# apply_affine_map: 参考坐标 xi -> 物理坐标 x；compute_jacobian: 计算映射雅可比矩阵 J。
# simplex_jacobian: 单纯形仿射映射的 J；transform_gradient: 根据 J^{-T} 把参考梯度转换到物理梯度。
from ..core.quadrature import reference_dimension
from ..core.utils import (
    apply_affine_map,
    compute_jacobian,
    simplex_jacobian,
    transform_gradient,
)
# 继承基础有限元空间：复用全局组装、自由度坐标、函数求值等公共接口。
from .base import BaseFESpace
# reference.py 专门处理“单个参考单元”：Pk/Qk 基函数、局部节点及其顺序。
# 这里的 lagrange.py 则把参考单元与网格、全局自由度、单元矩阵连接起来。
from .reference import (
    simplex_basis,
    simplex_multi_indices,
    simplex_local_descriptors,
    simplex_reference_nodes,
    tensor_basis,
    tensor_local_descriptors,
    tensor_multi_indices,
)

__all__ = ["LagrangeSpace"]

# 单纯形: 线段/三角形/四面体，采用总次数 <= k 的 Pk 多项式。
# 张量积: 四边形/六面体，采用各坐标方向次数 <= k 的 Qk 多项式。
_SIMPLEX = ("interval", "triangle", "tet")
_TENSOR = ("quad", "hex")


# ===== 全局自由度登记簿 =====
# “局部自由度”是某个单元内的节点编号；“全局自由度”是整个网格上的统一编号。
# 相邻单元公共顶点或公共边上的同一节点必须共用一个全局编号，才能形成连续有限元空间。
class _DofBook:
    """全局自由度登记簿：把几何 key 映射到全局自由度编号。"""

    def __init__(self):
        # keys[key]=idx: 几何身份 key 到全局编号 idx 的字典；重复 key 不重复分配。
        self.keys = {}
        # supports[idx]: 标识该自由度所依附的网格顶点集合，用于判断边界自由度。
        self.supports = []
        # interiors[idx]: 布尔值，记录是否属于单元内部自由度。
        self.interiors = []

    # key: 节点的全局几何身份；support: 支撑该节点的网格顶点编号；interior: 是否内部点。
    def get(self, key, support, interior):
        # 已经在相邻单元登记过，直接复用原来的全局自由度编号。
        if key in self.keys:
            return self.keys[key]
        # 尚未出现的新节点：当前已登记的自由度个数就是下一个可用编号（从 0 开始）。
        idx = len(self.supports)
        self.keys[key] = idx
        self.supports.append(tuple(support))
        self.interiors.append(bool(interior))
        return idx

    @property
        # @property 使 book.count 像属性一样访问，无需写 book.count()。
    def count(self):
        return len(self.supports)


# ===== 连续 Lagrange 有限元空间 =====
# 用户调用 FESpace.lagrange(mesh, degree=k) 时，最终使用类似本类的接口构造空间。
# 本类不仅保存基函数，还包含自由度映射、积分规则及单元矩阵/载荷向量计算。
class LagrangeSpace(BaseFESpace):
    """连续 Lagrange 空间 ``P_k`` / ``Q_k``。"""

    # 标记有限元家族，供统一的空间工厂/问题类识别。
    family = "lagrange"
    # 表示能够显式评估基函数及其梯度；与部分虚拟元方法不同。
    has_explicit_basis = True

    # ------------------------------------------------------------------
    # mesh: 网格对象；degree: 多项式阶数 k；name: 可选的空间名称。
    # 示例: degree=3 且 element_type="triangle" 表示二维三角形 P3 元（每单元 10 个节点）。
    def __init__(self, mesh, degree: int = 1, name: str = None):
        # 从 mesh 读取单元类型。下面将一些简写规范化，但暂不直接改写 mesh.element_type。
        et = mesh.element_type
        if et in ("tri",):
            et = "triangle"
        if et in ("tetra",):
            et = "tet"
        # 一般多边形/多面体没有本类预设的固定参考单元，不能直接按 Pk/Qk 处理。
        if et in ("polygon", "polyhedron"):
            raise ValueError(
                f"Lagrange 空间不支持 {et} 网格（单元形状不固定）。"
                "请改用 FESpace.vem(mesh, degree=1) 虚拟元，"
                "或把网格改成 triangle/quad/tet/hex。"
            )
        if et not in _SIMPLEX + _TENSOR:
            raise ValueError(f"不支持的网格单元类型 {et!r}")
        # 阶数 k 应为正整数，例如 1、2、3；拒绝 0、负数和非整数。
        if int(degree) != degree or degree < 1:
            raise ValueError("Lagrange degree 必须是正整数")
        if et == "interval" and mesh.dim != 1:
            raise ValueError("interval 空间需要 1D 网格")
        # 调用父类初始化；父类负责通用属性/缓存/自由度初始化等生命周期工作。
        # 注意显示名称里的 P 并不改变 quad/hex 实际使用 Qk 的事实。
        super().__init__(mesh, degree, name or f"P{degree}[{et}]")
        # 缓存“参考点 -> 基函数值和梯度”及“求积阶 -> 各求积点基函数数据”，避免重复计算。
        self._basis_cache = {}

    # ------------------------------------------------------------------
    # 参考单元信息
    # ------------------------------------------------------------------
    # 返回网格中保存的单元类型，供参考单元维度等公共接口使用。
    def reference_element_type(self) -> str:
        return self.mesh.element_type

    # quad / hex 返回 True；triangle / tet / interval 返回 False。
    def _is_tensor(self) -> bool:
        return self.mesh.element_type in _TENSOR

    # 描述一个单元内每个局部自由度的“类别/身份”。
    # 例如顶点 vertex、边 edge、面 face 或单元内部 interior。
    # 列表顺序必须与 reference.py 返回的基函数顺序一一对应。
    def _local_descriptors(self):
        # Qk: 多个一维 Lagrange 基函数作张量积；Pk: 重心坐标构造单纯形基函数。
        if self._is_tensor():
            # 这里将高阶张量积局部节点临时标为 interior，仅用于占位与取得正确长度。
            # 真正的跨单元共享不是据此判断，而由 _build_dof_map 内的权重 key 决定。
            if self.degree > 1:
                return [("interior", idx) for idx in tensor_multi_indices(self.mesh.dim, self.degree)]
            return tensor_local_descriptors(self.mesh.dim, self.degree)
        return simplex_local_descriptors(self.mesh.element_type, self.degree)

    # 返回参考单元插值节点的坐标数组，形状 (n_local, dim)。
    # cell_idx 保留用于统一接口；此函数的参考节点本身不依赖物理单元编号。
    def reference_nodes(self, cell_idx) -> np.ndarray:
        if self._is_tensor():
            multi = tensor_multi_indices(self.mesh.dim, self.degree)
            # 张量积节点整数指标 (i,j,...) / k，得到 [0,1]^d 中的参考坐标。
            return np.array(multi, dtype=float) / float(self.degree)
        return simplex_reference_nodes(self.mesh.element_type, self.degree)

    # ------------------------------------------------------------------
    # 基函数
    # ------------------------------------------------------------------
    # 在参考点 xi 求每个局部基函数值 N_i(xi)；可能一次输入多个点。
    def eval_basis(self, xi):
        return self._basis(xi)[0]

    # 返回参考梯度 dN/dxi，而不是物理梯度 dN/dx。
    def eval_basis_grad(self, xi):
        return self._basis(xi)[1]

    # 统一处理单个点 xi.shape=(dim,) 和一批点 xi.shape=(n_points,dim)。
    def _basis(self, xi):
        xi_arr = np.asarray(xi, dtype=float)
        # 一维数组代表“一个空间点”，交给 _basis_single；二维数组代表一批点。
        if xi_arr.ndim == 1:
            return self._basis_single(xi_arr)
        vals = []
        grads = []
        # 批量评估：按行逐个调用单点基函数，最后堆叠为 numpy 数组。
        for row in xi_arr:
            v, g = self._basis_single(row)
            vals.append(v)
            grads.append(g)
        return np.asarray(vals), np.asarray(grads)

    # 求单个参考点处的 (values, gradients)，采用缓存避免重复的多项式计算。
    def _basis_single(self, xi):
        # 将浮点坐标保留约 14 位小数后作为字典键，减少数值误差造成的缓存重复。
        key = tuple(np.round(np.asarray(xi, dtype=float), 14))
        hit = self._basis_cache.get(key)
        if hit is not None:
            return hit
        if self._is_tensor():
            out = tensor_basis(self.mesh.dim, self.degree, xi)
        else:
            out = simplex_basis(self.mesh.element_type, self.degree, xi)
        self._basis_cache[key] = out
        return out

    # ------------------------------------------------------------------
    # 自由度
    # ------------------------------------------------------------------
    # ===== 建立 局部自由度 -> 全局自由度 的映射 =====
    # 示例：两个三角形共用一条边，它们在公共边上的 P3 节点必须获得同一全局编号。
    # 最终 self.dof_map[c][j] = 第 c 个单元局部节点 j 对应的全局自由度编号。
    def _build_dof_map(self):
        """统一建立任意阶 P_k / Q_k 的局部到全局自由度映射。

        P_k（线段、三角形、四面体）统一使用重心整数指标 alpha；
        Q_k（四边形、六面体）使用参考角点的张量积整数权重。
        两种元都通过排序后的“(全局顶点编号, 权重)”识别共享节点。
        """
        from .reference import TENSOR_CORNERS

        book = _DofBook()
        tensor = self._is_tensor()
        if tensor:
            indices = tensor_multi_indices(self.mesh.dim, self.degree)
            corners = TENSOR_CORNERS[self.mesh.dim]
        else:
            # 一个 P_k 局部节点由 alpha=(a_0,...,a_d) 标识，sum(alpha)=k。
            indices = simplex_multi_indices(self.mesh.element_type, self.degree)

        self.dof_map = []
        for cell in self.mesh.elements:
            local = []
            for alpha in indices:
                if tensor:
                    # Q_k: 角点参考坐标取 0/1，权重为各方向 (k-a) 或 a 的乘积。
                    weights = []
                    for v, corner in enumerate(corners):
                        w = 1
                        for a, x in zip(alpha, corner):
                            w *= self.degree - a if x == 0 else a
                        if w:
                            weights.append((int(cell[v]), int(w)))
                    key_type = "q"
                else:
                    # P_k: 重心权重直接对应单纯形各顶点；所有维度/阶数用同一规则。
                    weights = [(int(v), int(a)) for v, a in zip(cell, alpha) if a]
                    key_type = "p"

                # 排序消除相邻单元局部顶点顺序的影响，使公共节点共享全局编号。
                weights.sort()
                support = tuple(v for v, _ in weights)
                interior = len(support) == len(cell)
                local.append(book.get((key_type, tuple(weights)), support, interior))

            # 第 e 个单元的第 i 个局部基函数对应全局编号 dof_map[e][i]。
            self.dof_map.append(np.asarray(local, dtype=np.int64))

        self.n_dofs = book.count
        self.dof_support = book.supports
        self.dof_is_interior = np.asarray(book.interiors, dtype=bool)

    # ===== 边界自由度提取 =====
    # 不能仅用“在某个单元边上”来判断边界：两个三角形之间的公共对角线属于内部边。
    # mesh.boundary_facets 提供真正的外边界边/面。
    def boundary_dofs(self) -> np.ndarray:
        """按照真正的边界 facet 判定高阶节点，避免内部对角线误判。"""
        # 将每个外边界 facet 的顶点转为集合，便于后面做子集测试。
        boundary = [set(int(v) for v in f) for f in self.mesh.boundary_facets]
        return np.asarray([
            i for i, support in enumerate(self.dof_support)
            if support and not self.dof_is_interior[i]
            # support 完全包含在某条边界边/面上，才把该自由度判为边界自由度。
            and any(set(support).issubset(face) for face in boundary)
        ], dtype=np.int64)

    # ------------------------------------------------------------------
    # 单元矩阵
    # ------------------------------------------------------------------
    # “求积阶”控制数值积分精度，而不是有限元多项式阶数本身。
    # 刚度矩阵 K_ij = integral_K grad(phi_i)·grad(phi_j) dx，需要足够高精度的积分规则。
    def default_quad_order(self) -> int:
        """刚度矩阵的求积阶。

        单纯形按**总次数** 2(p-1)；张量积单元的被积函数在每个方向上是
        2(p-1) 次，而 n 点 Gauss 规则精确到每方向 2n-1 次，
        所以取 ``order = 2p`` 才能保证 n = p 个点/方向。
        """
        if self._is_tensor():
            return max(2, 2 * self.degree)
        return max(1, 2 * (self.degree - 1))

    # 质量矩阵 M_ij = integral_K phi_i phi_j dx；乘积总次数一般达到 2k。
    def mass_quad_order(self) -> int:
        if self._is_tensor():
            return 2 * self.degree + 2
        return max(1, 2 * self.degree)

    # 载荷 b_i = integral_K f(x) phi_i(x) dx；f 可能不是多项式，因此选较高的数值积分阶。
    def load_quad_order(self) -> int:
        if self._is_tensor():
            return 2 * self.degree + 4
        return max(2, 2 * self.degree + 2)

    # ===== 取得参考域求积点和求积权重 =====
    # 积分近似 integral_K g dx ~= sum_q w_q |det(J_q)| g(x_q)。
    # 较低阶调用父类已有规则；高阶三角形/四面体用 Duffy 映射 + Gauss 积分。
    def get_quadrature(self, order: int = None):
        """高阶单纯形的 Duffy-Gauss 求积，仅影响 Lagrange FEM。"""
        if order is None:
            order = self.default_quad_order()
        order = int(order)
        et = self.mesh.element_type
        # 非单纯形或积分阶较低时，直接复用公共 quadrature 模块的规则。
        # 这里的 order 表示希望精确积分的多项式次数，不是积分点个数。
        # 对三角形 Pk 刚度矩阵，被积函数次数至多 2k-2；载荷还涉及 f，
        # 若 f 不是多项式（例如 sin(pi*x)sin(pi*y)），一般只可做到高精度近似。
        if et not in ("triangle", "tet") or order <= (4 if et == "triangle" else 3):
            return super().get_quadrature(order)
        if order in self._quad_cache:
            return self._quad_cache[order]
        from ..core.quadrature import QuadratureRule
        from itertools import product
        d = self.mesh.dim
        # Duffy map's Jacobian adds polynomial degrees (d-1, ..., 0).
        # 在每个参数方向选择 n 个高斯点；总求积点数为 n^d。
        n = max(1, (order + d + 1)//2)
        # leggauss(n) 给出 [-1,1] 上 n 点 Gauss-Legendre 节点与权重。
        nodes, weights = np.polynomial.legendre.leggauss(n)
        # 线性变换到 [0,1]：t=(s+1)/2，dt=ds/2，故权重也需除以 2。
        nodes, weights = (nodes+1)/2, weights/2
        pts, ww = [], []
        # 枚举 d 维张量积高斯积分点，每组下标确定一个积分点。
        for ind in product(range(n), repeat=d):
            t = [nodes[i] for i in ind]
            wt = float(np.prod([weights[i] for i in ind]))
            x = []
            # Duffy 映射示例(三角形)：xi_1=t_1，xi_2=(1-t_1)t_2。
            # 雅可比权重因子为 (1-t_1)，因此把方形上的积分转化为三角形上的积分。
            remaining = 1.0
            for j in range(d):
                x.append(remaining*t[j])
                if j < d-1:
                    # 乘上 Duffy 变换的雅可比因子，不能漏掉，否则积分面积/体积会错误。
                    wt *= (1-t[j]) ** (d-j-1)
                remaining *= 1-t[j]
            pts.append(x)
            ww.append(wt)
        # 封装为项目的 QuadratureRule 对象，并缓存到 _quad_cache。
        rule = QuadratureRule(pts, ww, et, order)
        self._quad_cache[order] = rule
        return rule

    # ===== 参考单元上预计算求积数据 =====
    # N[q,i] = 第 q 个求积点处基函数 phi_i 的值。
    # dN[q,i,a] = 同一点处第 i 个基函数对参考坐标 xi_a 的导数。
    def _quadrature_data(self, order: int):
        """缓存参考单元上的基函数求值（与单元无关，只与阶数有关）。"""
        key = ("q", int(order))
        hit = self._basis_cache.get(key)
        if hit is not None:
            return hit
        quad = self.get_quadrature(order)
        # 局部节点个数，例如三角形 P3 为 (3+1)(3+2)/2 = 10。
        n_local = len(self._local_descriptors())
        # 形状 (n_quad, n_local)，用于载荷向量与质量矩阵。
        N = np.zeros((quad.n_points, n_local))
        # 形状 (n_quad, n_local, dim)，用于刚度矩阵的梯度内积。
        dN = np.zeros((quad.n_points, n_local, self.mesh.dim))
        for q in range(quad.n_points):
            v, g = self._basis_single(quad.points[q])
            N[q] = v
            dN[q] = g
        out = (quad.points, quad.weights, N, dN)
        self._basis_cache[key] = out
        return out

    # ===== 当前物理单元几何信息 =====
    # 根据单元顶点坐标，将参考求积点映射到物理单元，计算 J 和 |det(J)|。
    def _element_geometry(self, cell_idx, ref_points):
        # 提取第 cell_idx 个单元的物理顶点坐标（而不是全局顶点编号）。
        coords = self.mesh.get_cell_vertices(cell_idx)
        # 得到物理积分点 x_q = F_K(xi_q)，用于计算 f(x_q)。
        phys = apply_affine_map(ref_points, coords)
        dets = np.zeros(len(ref_points))
        jac = []
        for q, xi in enumerate(ref_points):
            # J=dF_K/dxi，detJ 负责参考积分体积因子；某些单元 J 可能随 xi 变化。
            J, detJ = compute_jacobian(coords, xi)
            jac.append(J)
            dets[q] = abs(detJ)
        return phys, jac, dets

    # ===== 单元刚度矩阵 =====
    # Ke[i,j] = integral_K grad(phi_i)·grad(phi_j) dx。
    # 对 Poisson 方程 -Delta u=f，Ke 是主要的单元矩阵；输出形状 (n_local,n_local)。
    def local_stiffness(self, cell_idx) -> np.ndarray:
        """计算局部刚度矩阵 K_ij = ∫ grad(phi_i)·grad(phi_j) dx。"""
        pts, weights, _, grads_ref = self._quadrature_data(self.default_quad_order())
        vertices = self.mesh.get_cell_vertices(cell_idx)

        if self._is_tensor():
            # Q_k: 四边形/六面体映射的 J 可能随参考积分点变化。
            _, jac, dets = self._element_geometry(cell_idx, pts)
            grads = np.asarray([
                transform_gradient(grads_ref[q], jac[q]) for q in range(len(pts))
            ])
        else:
            # 所有单纯形 P_k（线段/三角形/四面体）共用常数雅可比 J。
            J, detJ = simplex_jacobian(vertices)
            grads = grads_ref @ np.linalg.inv(J)
            dets = np.full(len(weights), abs(detJ))

        # q: 积分点，i/j: 局部基函数，a: 空间坐标方向。
        # K_ij = Σ_q (w_q |detJ_q|) Σ_a grad_i[q,a] grad_j[q,a]
        return np.einsum("q,qia,qja->ij", weights * dets, grads, grads)

    # ===== 单元质量矩阵 =====
    # Me[i,j] = integral_K phi_i phi_j dx；Helmholtz、时间依赖 PDE 常用。
    def local_mass(self, cell_idx) -> np.ndarray:
        """单元质量矩阵。"""
        pts, w, N, _ = self._quadrature_data(self.mass_quad_order())
        _, _, dets = self._element_geometry(cell_idx, pts)
        nl = N.shape[1]
        Me = np.zeros((nl, nl))
        for q in range(len(w)):
            # np.outer(N[q],N[q]) 的 (i,j) 项就是 phi_i(x_q)*phi_j(x_q)。
            Me += (w[q] * dets[q]) * np.outer(N[q], N[q])
        return Me

    # ===== 单元载荷向量 =====
    # Fe[i] = integral_K f(x) phi_i(x) dx，其中 f 是 PDE 的右端项。
    def local_load(self, cell_idx, f, order=None) -> np.ndarray:
        """计算局部载荷向量 F_i = ∫ f(x) phi_i(x) dx。"""
        order = self.load_quad_order() if order is None else int(order)
        pts, weights, basis, _ = self._quadrature_data(order)
        vertices = self.mesh.get_cell_vertices(cell_idx)

        if self._is_tensor():
            # Q_k: 为各积分点分别计算几何变换。
            xy, _, dets = self._element_geometry(cell_idx, pts)
        else:
            # 所有单纯形 P_k 都使用同一个仿射变换 x = x0 + J xi。
            J, detJ = simplex_jacobian(vertices)
            xy = vertices[0] + pts @ J.T
            dets = np.full(len(weights), abs(detJ))

        source = self._evaluate_function(f, xy)
        # basis[q,i] = phi_i(xi_q)；矩阵乘法一次算出所有局部载荷分量。
        return basis.T @ (weights * dets * source)

    # ------------------------------------------------------------------
    # 全局矩阵
    # ------------------------------------------------------------------
    # 全局刚度矩阵：将每个 Ke 根据 dof_map 加到全局矩阵的对应行列位置。
    def stiffness_matrix(self):
        # 不在这里重新写双重 for 循环组装：base.py 的 _assemble 已完成。
        # 对每个单元 e，读取局部 Ke 及其 dof_map[e]，执行
        # A[dof_map[e][i], dof_map[e][j]] += Ke[i,j]。
        # symmetric=True 声明 Poisson 刚度矩阵在正确组装时为对称矩阵。
        return self._assemble(self.local_stiffness, symmetric=True)

    # 全局质量矩阵：将所有单元 Me 累加组装。
    def mass_matrix(self):
        return self._assemble(self.local_mass, symmetric=True)

    # 全局载荷向量：将各单元 Fe 根据局部到全局映射累加。
    def load_vector(self, f, order=None):
        return self._assemble_vector(lambda c: self.local_load(c, f, order))

    # ------------------------------------------------------------------
    # 对 Lagrange 节点插值：每个自由度对应坐标 x_i，返回 u_i=f(x_i)。
    def interpolate(self, f) -> np.ndarray:
        """按自由度坐标做 Lagrange 插值。"""
        return self._evaluate_function(f, self.dof_coords)

    # ------------------------------------------------------------------
    # 自由度 → 函数
    # ------------------------------------------------------------------
    # ===== 物理点 -> 参考点 =====
    # 已知物理坐标 x，计算相应参考坐标 xi=F_K^{-1}(x)，才能用参考基函数评估数值解。
    def _inverse_map(self, cell_idx, points, iters: int = 25):
        """把物理点映回参考单元坐标。"""
        coords = self.mesh.get_cell_vertices(cell_idx)
        # 将单个点也转为 (1,dim) 形状，统一后面批量处理。
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        d = self.mesh.dim
        # 单纯形的几何映射是仿射映射 x=x0+J*xi，可直接求逆。
        if not self._is_tensor():
            J, _ = simplex_jacobian(coords)
            # 对每个物理点计算 xi=J^{-1}(x-x0)；此处以“行向量”形式实现。
            return (pts - coords[0]) @ np.linalg.inv(J).T
        # quad/hex 几何映射可能是非线性的，无法用一个常数 J 直接求逆。
        # 以参考单元中心 xi=(0.5,...,0.5) 作 Newton 迭代初值。
        xi = np.full((len(pts), d), 0.5)
        for _ in range(iters):
            mapped = apply_affine_map(xi, coords)
            # 残差 r = 目标物理点 x - 当前映射 F_K(xi)，希望迭代使 r->0。
            res = pts - mapped
            if np.max(np.linalg.norm(res, axis=1)) < 1e-13:
                break
            for q in range(len(pts)):
                J, detJ = compute_jacobian(coords, xi[q])
                if abs(detJ) < 1e-14:
                    continue
                # Newton 更新：xi_new=xi_old+J^{-1}r；避免直接计算逆矩阵。
                xi[q] = xi[q] + np.linalg.solve(J, res[q])
        return xi

    # ===== 根据自由度系数计算数值解在物理点的值 =====
    # u_h(x)=sum_{j=1}^{n_local} U_{I_j} phi_j(F_K^{-1}(x))。
    def evaluate(self, cell_idx, values, points):
        values = np.asarray(values, dtype=float).ravel()
        # 当前单元的全局自由度编号数组 I_j。
        dofs = self.get_cell_dofs(cell_idx)
        # 从全局自由度向量 U 中挑出本单元的局部系数。
        local = values[dofs]
        xi = self._inverse_map(cell_idx, points)
        out = np.zeros(len(xi))
        for q in range(len(xi)):
            N, _ = self._basis_single(xi[q])
            # N 是各局部基函数在 xi_q 的值，点积 N@local 即 u_h(x_q)。
            out[q] = float(N @ local)
        return out

    # ===== 计算数值解的物理梯度 =====
    # grad(u_h)(x)=sum_j U_{I_j} J^{-T} grad_xi(phi_j)(xi)。
    def evaluate_gradient(self, cell_idx, values, points):
        values = np.asarray(values, dtype=float).ravel()
        dofs = self.get_cell_dofs(cell_idx)
        local = values[dofs]
        coords = self.mesh.get_cell_vertices(cell_idx)
        xi = self._inverse_map(cell_idx, points)
        out = np.zeros((len(xi), self.mesh.dim))
        for q in range(len(xi)):
            _, dN_ref = self._basis_single(xi[q])
            J, _ = compute_jacobian(coords, xi[q])
            dN = transform_gradient(dN_ref, J)
            # dN.T 形状 (dim,n_local)，与局部系数相乘得到 dim 维梯度向量。
            out[q] = dN.T @ local
        return out

    # ===== L2 正交投影 =====
    # 寻找 p_h in V_h，使 integral (p_h-f)*v_h dx =0，对所有 v_h in V_h。
    # 写成矩阵就是 M*U=b，其中 M_ij=int phi_i phi_j，b_i=int f phi_i。
    def project(self, f) -> np.ndarray:
        """L2 投影到本空间（质量矩阵求解），返回自由度向量。"""
        from ..solvers.linear import Solver

        M = self.mass_matrix()
        b = self.load_vector(f, order=max(self.mass_quad_order(), 4))
        # 解质量矩阵线性方程组，返回投影函数对应的全局自由度系数。
        return Solver.direct(M, b)

    # ------------------------------------------------------------------
    @property
        # 返回参考单元空间维度，例如 triangle->2，tet->3。
    def ref_dim(self) -> int:
        return reference_dimension(self.reference_element_type())
