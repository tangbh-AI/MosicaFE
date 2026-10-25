"""k 阶协调增强型 VEM 的局部计算，二维面和三维单元共用同一套公式。

自由度：顶点值、每条边 k-1 个内部点值、面/体中不超过 k-2 次的归一化矩。
先用分部积分构造 H1 投影，再用增强约束补出高阶矩，得到 L2 投影。
三维单元的边界积分使用二维面的 L2 投影，不把面内函数假装成已知多项式。
参考：Ahmad et al., Equivalent projectors for virtual element methods (2013).
"""

# 模块职责：按边、面、体递归计算局部投影，缓存几何积分；D、H、G、energy、l2 连接多项式和自由度。


from functools import cached_property
from itertools import product

import numpy as np

from ..core.quadrature import simplex_quadrature


def powers(dim, degree):
    """总次数不超过 degree 的多重指标；低次排在前面，常数是第 0 项。"""
    if degree < 0:
        return []
    return sorted((a for a in product(range(degree + 1), repeat=dim)
                   if sum(a) <= degree), key=lambda a: (sum(a), a))


def scalar_values(f, points):
    """逐点调用，直接兼容现有算例的 f(x)，避免猜测其向量化约定。"""
    return np.array([float(f(x)) for x in points])


class LocalVEM:
    """一个边、面或体的几何、自由度和投影；children 是它的边界实体。"""

    def __init__(self, vertices, coords, dim, degree, children, allocate, functionals):
        # 保存实体几何与子实体，建立内禀坐标轴并检查星形条件，随后登记边点和内部矩，形成局部自由度布局。
        self.vertices = tuple(vertices)
        self.coords = np.asarray(coords, dtype=float)
        self.dim = dim
        self.degree = degree
        self.children = children
        self.functionals = functionals
        self.center = self.coords.mean(axis=0)
        self.h = np.max(np.linalg.norm(self.coords[:, None] - self.coords, axis=2))
        if self.h <= 0:
            raise ValueError("VEM 单元或面退化：直径为零")
        # dim 是实体自身维数，ambient 是所在物理空间维数；三维中的二维面需要先建立面内正交坐标轴。
        ambient = self.coords.shape[1]
        if dim == ambient:
            self.axes = np.eye(dim)
        elif dim == 1:
            direction = self.coords[1] - self.coords[0]
            self.axes = (direction / np.linalg.norm(direction))[:, None]
        else:
            _, _, vt = np.linalg.svd(self.coords - self.center, full_matrices=False)
            self.axes = vt[:dim].T
            off_plane = self.coords - self.center - ((self.coords-self.center) @ self.axes) @ self.axes.T
            if np.max(np.linalg.norm(off_plane, axis=1)) > 1e-10 * self.h:
                raise ValueError("三维 VEM 的每个面必须共面")
        if dim == 2:
            # 扇形积分要求中心在多边形的星形核内；同时检查顺/逆时针均可。
            xy = (self.coords - self.center) @ self.axes
            nxt = np.roll(xy, -1, axis=0)
            cross = xy[:, 0]*nxt[:, 1] - xy[:, 1]*nxt[:, 0]
            if abs(cross.sum()) < 1e-12*self.h**2 or np.any(cross*np.sign(cross.sum()) <= 1e-13*self.h**2):
                raise ValueError('VEM 多边形/面的顶点均值必须严格位于星形核内，顶点需沿边界排序')
        elif dim == 3:
            self._normals = self._orient_boundary(children)
        # exponents 描述完整 P_k 多项式，low 描述真实内部矩的 P_(k-2)；两者按相同顺序排列，所以 low 是完整基的前缀。
        self.exponents = powers(dim, degree)
        self.low = powers(dim, degree - 2) if dim > 1 else []

        # 顶点在前，随后合并子实体的共享自由度，最后追加本实体的内部矩。
        dofs = list(vertices)
        for child in children:
            for dof in child.dofs:
                if dof not in dofs:
                    dofs.append(dof)
        if dim == 1:
            for j in range(1, degree):
                point = self.coords[0] + j / degree * (self.coords[1] - self.coords[0])
                dofs.append(allocate(('point', point)))
        self.moment_start = len(dofs)
        for j in range(len(self.low)):
            dofs.append(allocate(('moment', self, j)))
        self.dofs = np.array(dofs, dtype=int)
        # 子实体的全局自由度在父实体局部向量中的位置先缓存，边界积分时可直接把面贡献写到正确的列。
        self.child_positions = [np.array([dofs.index(i) for i in ch.dofs]) for ch in children]

    def polynomial(self, points, derivative=None):
        """缩放单项式 m_alpha=((x-c)/h)^alpha；可求内禀坐标方向的一、二阶导数。"""
        z = (np.atleast_2d(points) - self.center) @ self.axes / self.h
        out = np.ones((len(z), len(self.exponents)))
        for j, alpha in enumerate(self.exponents):
            a = list(alpha)
            factor = 1.0
            for axis in derivative or ():
                factor *= a[axis] / self.h
                a[axis] = max(0, a[axis] - 1)
            out[:, j] = factor * np.prod(z ** np.array(a), axis=1)
        return out

    def gradient(self, points):
        # 先对内禀坐标的缩放单项式求导，再乘局部坐标轴，将梯度转换为物理坐标分量。
        intrinsic = np.stack([self.polynomial(points, (a,)) for a in range(self.dim)], axis=2)
        return intrinsic @ self.axes.T

    def outward(self, child):
        """取当前实体切空间中、垂直于子实体且指向外部的单位法向。"""
        if self.dim == 3:
            return self._normals[child]
        radial = child.center - self.center
        normal = self.axes @ (self.axes.T @ radial)
        normal -= child.axes @ (child.axes.T @ normal)
        length = np.linalg.norm(normal)
        if length <= 1e-12 * self.h:
            raise ValueError("VEM 要求顶点均值位于单元及各面的星形核内")
        return normal / length

    def _orient_boundary(self, faces):
        """沿共享边统一面的取向，再检查中心是否位于所有面的内半空间。

        由闭合面确定外法向，允许星形凹多面体；不能用“所有顶点在同侧”限制为凸体。
        """
        edges = {}
        raw = []
        for j, face in enumerate(faces):
            ids = face.vertices
            for i, a in enumerate(ids):
                b = ids[(i+1)%len(ids)]
                edges.setdefault(tuple(sorted((a, b))), []).append((j, 1 if a < b else -1))
            cross = np.cross(face.coords-self.center,
                             np.roll(face.coords, -1, axis=0)-self.center).sum(axis=0)/2
            raw.append(cross)
        if any(len(owners) != 2 for owners in edges.values()):
            raise ValueError('多面体表面未闭合：每条边必须恰好被两个面共享')
        # 闭合曲面的相邻面必须沿共享边反向行走；用边方向建立符号关系，再从一个面传播到整个连通表面。
        adjacency = [[] for _ in faces]
        for (a, da), (b, db) in edges.values():
            adjacency[a].append((b, -da*db))
            adjacency[b].append((a, -da*db))
        signs = {0: 1}
        pending = [0]
        while pending:
            j = pending.pop()
            for other, relative in adjacency[j]:
                value = signs[j]*relative
                if other in signs and signs[other] != value:
                    raise ValueError('多面体面的顶点顺序不一致或表面不可定向')
                if other not in signs:
                    signs[other] = value
                    pending.append(other)
        if len(signs) != len(faces):
            raise ValueError('单个多面体的表面必须连通')
        # 前一步只确定相对方向，整体仍可同时反转；用有向体积的符号选定朝外方向，再核对中心处于每个面的内侧。
        orientation = np.sign(sum(signs[j]*np.dot(raw[j], face.center-self.center)
                                  for j, face in enumerate(faces)))
        normals = {}
        for j, face in enumerate(faces):
            normal = orientation*signs[j]*raw[j]/np.linalg.norm(raw[j])
            if np.dot(normal, face.center-self.center) <= 1e-12*self.h:
                raise ValueError('多面体顶点均值必须严格位于星形核内')
            normals[face] = normal
        return normals

    @cached_property
    def quadrature(self):
        """只对已知多项式与源项求积；几何剖分不引入有限元未知量。"""
        return self.integrate(2 * self.degree + 2)

    def integrate(self, order):
        """按指定阶数生成物理积分点，供局部计算和误差评估共同使用。"""
        if len(self.coords) == self.dim + 1:
            return simplex_quadrature(self.coords, order)
        # 中心位于星形核内时，中心与各边/面组成的积分分片覆盖原实体；返回的权重已含物理长度、面积或体积缩放。
        pieces = []
        if self.dim == 2:
            for edge in self.children:
                pieces.append(np.vstack([self.center, edge.coords]))
        else:
            for face in self.children:
                for edge in face.children:
                    pieces.append(np.vstack([self.center, face.center, edge.coords]))
        rules = [simplex_quadrature(piece, order) for piece in pieces]
        return np.vstack([p for p, w in rules]), np.concatenate([w for p, w in rules])

    @cached_property
    def measure(self):
        # 积分常数函数得到实体测度；一维为长度，二维为面积，三维为体积，计算后缓存。
        return float(self.quadrature[1].sum())

    @cached_property
    def data(self):
        """返回 D、H、G、Pi_grad、Pi_L2；它们连接多项式系数与虚拟自由度。

        D：多项式 → 自由度；H/G：多项式质量/梯度 Gram 矩阵。
        Pi_grad/Pi_L2：自由度 → 投影多项式系数。无需构造虚拟基函数。
        """
        x, w = self.quadrature
        m = self.polynomial(x)
        # m 的行为求积点、列为单项式；加权内积形成多项式质量 Gram 矩阵 H，后续 L2 投影依赖它。
        H = m.T @ (w[:, None] * m)
        n = len(self.dofs)
        # D 的每行对应一个虚拟自由度、每列对应一个单项式；对点行求值，对矩行按其实体上的归一化积分求值。
        D = np.zeros((n, len(self.exponents)))
        for row, dof in enumerate(self.dofs):
            item = self.functionals[dof]
            if item[0] == 'point':
                D[row] = self.polynomial(item[1])[0]
            else:
                entity, j = item[1:]
                q, weights = entity.quadrature
                test = entity.polynomial(q)[:, j]
                D[row] = (weights * test) @ self.polynomial(q) / entity.measure

        # 边上的迹就是 k 次多项式，k+1 个点值唯一决定它。
        if self.dim == 1:
            projection = np.linalg.solve(D, np.eye(n))
            grad = self.gradient(x)
            G = np.einsum('qid,qjd,q->ij', grad, grad, w)
            return D, H, G, projection, projection

        # 分部积分：∫ grad(v)·grad(m) = ∫边界 v ∂n(m) - ∫ v Δm。
        # 边界迹由子实体的 L2 投影积分；Δm 的次数 ≤ k-2，内部矩足以精确计算。
        grad = self.gradient(x)
        G = np.einsum('qid,qjd,q->ij', grad, grad, w)
        rhs = np.zeros((len(self.exponents), n))
        for child, positions in zip(self.children, self.child_positions):
            q, weights = child.quadrature
            dn = self.gradient(q) @ self.outward(child)
            trace = child.polynomial(q) @ child.data[4]
            rhs[:, positions] += dn.T @ (weights[:, None] * trace)
        low_index = {a: i for i, a in enumerate(self.low)}
        for row, alpha in enumerate(self.exponents):
            for axis in range(self.dim):
                if alpha[axis] >= 2:
                    beta = list(alpha)
                    beta[axis] -= 2
                    col = self.moment_start + low_index[tuple(beta)]
                    rhs[row, col] -= self.measure * alpha[axis] * (alpha[axis]-1) / self.h**2

        # 梯度无法决定常数：k≥2 用单元均值固定，k=1 用顶点均值固定。
        system = G.copy()
        if self.low:
            system[0] = H[0]
            rhs[0] = 0.0
            rhs[0, self.moment_start] = self.measure
        else:
            system[0] = D[:len(self.vertices)].mean(axis=0)
            rhs[0] = 0.0
            rhs[0, :len(self.vertices)] = 1.0 / len(self.vertices)
        energy = np.linalg.solve(system, rhs)

        # 增强条件令 k-1、k 次矩等于 H1 投影的对应矩；低阶矩仍取真实自由度。
        moments = H @ energy
        for j in range(len(self.low)):
            moments[j] = 0.0
            moments[j, self.moment_start+j] = self.measure
        l2 = np.linalg.solve(H, moments)
        return D, H, G, energy, l2

    def stiffness(self, stabilization):
        # 组合 H1 投影的一致性刚度与投影核上的稳定化，并显式对称化以减小浮点不对称。
        D, H, G, energy, l2 = self.data
        consistent = energy.T @ G @ energy
        consistent = 0.5 * (consistent + consistent.T)
        if stabilization == 'none':
            return consistent
        # 若自由度来自 P_k 多项式，则 energy 恢复系数、D 恢复原自由度，残差为零；稳定化因此只作用于未被投影决定的方向。
        residual = np.eye(len(self.dofs)) - D @ energy
        stable = residual.T @ residual
        scale = np.trace(consistent) / max(np.trace(stable), 1e-30)
        if np.linalg.norm(residual) < 1e-10:
            return consistent
        result = consistent + scale * stable
        return 0.5 * (result + result.T)

    def mass(self):
        # 以 L2 投影构造多项式乘积积分，在其投影核上加入与实体测度同量纲的稳定化项。
        D, H, G, energy, l2 = self.data
        residual = np.eye(len(self.dofs)) - D @ l2
        result = l2.T @ H @ l2 + self.measure * (residual.T @ residual)
        return 0.5 * (result + result.T)

    def load(self, f):
        # 先积分源项与各单项式的乘积，再用 L2 投影转置映射成局部自由度载荷。
        x, w = self.quadrature
        integral = self.polynomial(x).T @ (w * scalar_values(f, x))
        return self.data[4].T @ integral
