"""二维/三维任意整数 k≥1 的协调增强型虚拟元。

接口保持 FESpace.vem(mesh, degree=k)，局部投影公式位于 vem_local.py。
支持顶点均值在星形核内的多边形/多面体，三维各面必须共面且满足同样条件。
后处理使用可计算的多项式投影，不显式构造虚拟基函数。
"""

# 模块职责：负责全局自由度与边界的管理；局部积分和投影由 LocalVEM 计算，全局累加复用 BaseFESpace。


import numbers

import numpy as np

from .base import BaseFESpace
from .vem_local import LocalVEM, scalar_values

__all__ = ['VEMSpace']


class VEMSpace(BaseFESpace):
    # 协调增强型虚拟元空间，负责全局共享自由度、边界数据和局部投影算子的连接。
    family = 'vem'
    has_explicit_basis = False

    def __init__(self, mesh, degree=1, stabilization='trace', name=None):
        # 检查维度、正整数阶数和稳定化选项，再交给基类按统一顺序建立全局自由度及几何信息。
        if isinstance(degree, bool) or not isinstance(degree, numbers.Integral) or degree < 1:
            raise ValueError('VEM 的 degree 必须是正整数 k≥1')
        if mesh.dim not in (2, 3):
            raise ValueError('VEM 支持二维和三维网格')
        if stabilization not in ('trace', 'none', 'legacy'):
            raise ValueError('stabilization 请选择 trace 或 none')
        # 兼容旧诊断脚本的参数；legacy 也使用对称的投影核稳定化。
        self.stabilization = stabilization
        super().__init__(mesh, degree, name or f'VEM-P{degree}')

    # 工作块：递归登记共享自由度。
    # 顶点编号保持网格编号；相邻单元共享同一条边、同一个面及其全部自由度。
    # 面/体内部矩是积分值，代表坐标仅供展示，不能当作点值使用。
    def _build_dof_map(self):
        self._functionals = [('point', p.copy()) for p in self.mesh.nodes]
        self._entities = {}
        self.cells = []
        self._boundary_counts = {}

        def allocate(item):
            # 把点值或积分矩的定义追加到全局登记表，返回该未知量在全局向量中的编号。
            self._functionals.append(item)
            return len(self._functionals) - 1

        def entity(vertices, dim, faces=None, cell_number=None):
            # 递归建立边、面和单元对象；相同几何实体复用缓存，单元内部矩保持各单元独有。
            # 边和面以排序后的全局顶点集合为共享键，方向可以相反；体单元额外带 cell_number，内部矩因此不会被邻单元误共享。
            key = (dim, tuple(sorted(vertices)), cell_number)
            if key in self._entities:
                shared = self._entities[key]
                if dim == 2:
                    def edge_set(ids):
                        # 把一个面的边界环转换为无向边集合，用来确认共享面的两种顶点顺序表示同一边界。
                        return {tuple(sorted((ids[i], ids[(i+1)%len(ids)]))) for i in range(len(ids))}
                    if edge_set(vertices) != edge_set(shared.vertices):
                        raise ValueError('共享面的顶点必须按同一个边界环排列，可反向')
                return shared
            if dim == 1:
                vertices = tuple(sorted(vertices))
                children = []
            elif dim == 2:
                children = [entity((vertices[i], vertices[(i+1) % len(vertices)]), 1)
                            for i in range(len(vertices))]
            else:
                children = [entity(tuple(vertices[i] for i in face), 2) for face in faces]
            local = LocalVEM(vertices, self.mesh.nodes[list(vertices)], dim, self.degree,
                             children, allocate, self._functionals)
            self._entities[key] = local
            return local

        for c, vertices in enumerate(self.mesh.elements):
            vertices = tuple(int(i) for i in vertices)
            faces = None
            if self.mesh.dim == 3:
                faces = self.mesh.get_cell_faces(c)
            local = entity(vertices, self.mesh.dim, faces, c)
            self.cells.append(local)
            for child in local.children:
                self._boundary_counts[child] = self._boundary_counts.get(child, 0) + 1

        if any(count > 2 for count in self._boundary_counts.values()):
            raise ValueError('非流形网格：同一条边/同一个面不能被超过两个单元共享')

        self.dof_map = [cell.dofs for cell in self.cells]
        self.n_dofs = len(self._functionals)
        # 统计中只出现一次的边界实体属于区域外边界；取其全部自由度，不只取顶点，才能覆盖高阶边点和面矩。
        self._boundary = np.array(sorted({int(i) for face, count in self._boundary_counts.items()
                                         if count == 1 for i in face.dofs}), dtype=int)
        self.dof_support = [() for _ in range(self.n_dofs)]
        self.dof_is_interior = np.ones(self.n_dofs, dtype=bool)
        self.dof_is_interior[self._boundary] = False

    def _build_dof_coordinates(self):
        # 建立自由度的代表坐标；点自由度位于实际点上，积分矩使用所属实体中心作代表。
        # 矩自由度没有单一点坐标，这里只给它一个所属实体中心作代表；插值和施加边界时仍必须调用积分定义。
        self.dof_coords = np.array([item[1] if item[0] == 'point' else item[1].center
                                    for item in self._functionals])

    def boundary_dofs(self):
        # 返回位于区域外边界的全局自由度编号，包含高阶空间需要施加约束的边和面自由度。
        return self._boundary.copy()

    # 工作块：按自由度定义对函数取值。
    # 边界点取 g(x)，边界面的矩取归一化积分，确保非零 Dirichlet 边界也正确。
    def _functional_value(self, item, f):
        if item[0] == 'point':
            return float(f(item[1]))
        entity, j = item[1:]
        x, w = entity.quadrature
        return float((w * entity.polynomial(x)[:, j]) @ scalar_values(f, x) / entity.measure)

    def boundary_values(self, f, dofs):
        # 只计算给定边界自由度上的函数数据，正确区分节点值和三维边界面的积分矩。
        return np.array([self._functional_value(self._functionals[i], f) for i in dofs])

    def interpolate(self, f):
        # 按每个自由度的定义离散已知函数；VEM 的内部矩需要积分，不能用中心点值替代。
        return np.array([self._functional_value(item, f) for item in self._functionals])

    def reference_nodes(self, cell_idx):
        # 返回局部自由度的参考位置或代表位置，积分矩的代表位置仅用于几何描述。
        cell = self.cells[cell_idx]
        return (self.dof_coords[cell.dofs] - cell.center) / cell.h

    def reference_element_type(self):
        # 给出当前空间使用的参考单元类别，供积分及参考基函数接口识别。
        return 'polygon' if self.mesh.dim == 2 else 'polyhedron'

    def get_quadrature(self, order=None):
        # 查询积分数据；VEM 没有统一参考单元规则，需要改用各实际单元的积分接口。
        raise NotImplementedError('VEM 按实际单元积分；请用 space.cell_quadrature(cell_idx)')

    def cell_quadrature(self, cell_idx, order=None):
        # 返回指定实际单元的积分点和物理权重；请求高于缓存精度的阶数时重新生成。
        cell = self.cells[cell_idx]
        if order is not None and order > 2*self.degree+2:
            return cell.integrate(order)
        return cell.quadrature

    # 工作块：局部投影公式与原来的全局组装框架连接。
    def local_stiffness(self, cell_idx):
        return self.cells[cell_idx].stiffness(self.stabilization)

    def local_mass(self, cell_idx, order=None):
        # 返回指定单元的质量矩阵，描述离散函数乘积的积分贡献。
        return self.cells[cell_idx].mass()

    def local_load(self, cell_idx, f, order=None):
        # 对源项和局部测试函数的乘积积分，得到该单元对全局右端的贡献。
        return self.cells[cell_idx].load(f)

    def stiffness_matrix(self):
        # 将所有局部刚度贡献组装为全局矩阵；混合空间若不适用此标量接口会明确拒绝。
        return self._assemble(self.local_stiffness, symmetric=True)

    def mass_matrix(self):
        # 将局部质量贡献组装为全局矩阵；特殊空间的块结构由其专用接口提供。
        return self._assemble(self.local_mass, symmetric=True)

    def load_vector(self, f):
        # 按单元计算源项载荷，再按全局自由度编号累加成右端向量。
        return self._assemble_vector(lambda c: self.local_load(c, f))

    # 工作块：后处理使用可计算投影，兼容原来的误差与绘图接口。
    # 点值来自 Pi_L2 u，梯度来自 grad(Pi_grad u)，不宣称是虚拟函数的精确求值。
    def reconstruction_matrix(self, cell_idx):
        return self.cells[cell_idx].data[4]

    def evaluate(self, cell_idx, values, points):
        # 取出当前单元的自由度，通过 L2 投影得到多项式系数，再在指定物理点计算函数值。
        cell = self.cells[cell_idx]
        coefficients = cell.data[4] @ np.asarray(values)[cell.dofs]
        return cell.polynomial(points) @ coefficients

    def evaluate_gradient(self, cell_idx, values, points):
        # 由局部自由度重构采样点的物理梯度；若该空间没有此标量梯度接口则明确报错。
        cell = self.cells[cell_idx]
        coefficients = cell.data[3] @ np.asarray(values)[cell.dofs]
        return np.einsum('qid,i->qd', cell.gradient(points), coefficients)

    def nodal_values(self, values):
        # 顶点本身就是点自由度，绘图直接取顶点值。
        return np.asarray(values)[:self.mesh.n_nodes].copy()
