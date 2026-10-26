"""参考单元上的基函数、局部自由度布局。

本模块把"参考单元上有什么"与"全局自由度怎么编号"分开：这里只管前者，
后者由 :mod:`MosicaFE.spaces.lagrange` 中的自由度管理器负责。

局部自由度的描述符 (descriptor) 有四种：

``("vertex", (a, b, c))``
    顶点型自由度，参数为该顶点在参考单元上的坐标。
``("edge", (corner_i, corner_j))``
    边内部自由度，用参考单元的两个角点局部编号表示。
``("face", (c0, c1, c2, c3))``
    面内部自由度（3D 张量积单元），用面的 4 个角点局部编号表示。
``("interior", (i, j, k))``
    单元内部自由度，只属于本单元。
"""

# 允许类型注解延迟求值；不改变下面有限元算法的计算过程。
from __future__ import annotations

import numpy as np


__all__ = [
    "simplex_basis",
    "tensor_basis",
    "simplex_local_descriptors",
    "tensor_local_descriptors",
    "simplex_reference_nodes",
    "TENSOR_CORNERS",
    "HEX_EDGES",
]

# 参考单元角点顺序（张量积单元）
# 张量积参考单元：1D=[0,1]，2D=[0,1]^2，3D=[0,1]^3。
# 注意字典键 1/2/3 是“空间维度”，不是多项式阶数。
# 每组坐标按局部顶点编号依次排列，后面边和面的编号均依赖该顺序。
TENSOR_CORNERS = {
    1: [(0,), (1,)],
    2: [(0, 0), (1, 0), (1, 1), (0, 1)],
    3: [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ],
}

# 六面体的 12 条棱（用角点局部编号表示）
# 六面体的每一条棱由两个顶点编号组成：底面4条、顶面4条、竖向4条。
HEX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# 六面体的 6 个面（从单元外部看逆时针）
# 六面体每个面的4个顶点编号，供三维面自由度识别。
HEX_FACES_LOCAL = [
    (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
    (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
]

_QUAD_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]

# 单纯形的棱（局部顶点编号）
# 单纯形包括线段（1D）、三角形（2D）、四面体（3D）。
# 这里存储三角形的3条边和四面体的6条棱；线段的边在函数中单独定义。
_SIMPLEX_EDGES = {
    "triangle": [(0, 1), (1, 2), (2, 0)],
    "tet": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
}


# ---------------------------------------------------------------------------
# 基函数
# ---------------------------------------------------------------------------
# 【重心坐标】在参考三角形上：lambda=(1-a-b, a, b)；
# 在参考四面体上：lambda=(1-a-b-c, a, b, c)。
# 输出 lam 是重心坐标值，grad 的“行”是参考坐标方向，“列”是各 lambda。
def _simplex_lambdas(element_type: str, xi: np.ndarray): #xi是参考单元坐标，形状为 (2,) 或 (3,)。
    """参考单纯形上的重心坐标及其（参考）梯度。"""
   
    xi = np.asarray(xi, dtype=float).ravel() # 将 xi 转为一维浮点数组，例如 [[0.2,0.3]] → [0.2,0.3]。
    if element_type == "triangle":  #判断是不是三角形
        # 三角形参考顶点为 (0,0)、(1,0)、(0,1)。
        a, b = xi  #提取参考点xi的坐标
        # lam=[lambda_0,lambda_1,lambda_2]，且三个重心坐标之和为1。
        lam = np.array([1.0 - a - b, a, b])
        # 第一行 d(lambda_0,lambda_1,lambda_2)/da=(-1,1,0)；
        # 第二行 d(lambda_0,lambda_1,lambda_2)/db=(-1,0,1)。
        grad = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    elif element_type == "tet":
        # 四面体参考顶点为 (0,0,0)、(1,0,0)、(0,1,0)、(0,0,1)。
        a, b, c = xi
        lam = np.array([1.0 - a - b - c, a, b, c])
        # 3×4 矩阵：第1/2/3行分别是对 a/b/c 的偏导数。
        grad = np.array(
            [
                [-1.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 1.0],
            ]
        )
    else:
        raise ValueError(f"不支持的单纯形类型 {element_type!r}")
    # 后面通过 grad @ grad_l 应用链式法则，将重心坐标导数转为参考坐标导数。
    return lam, grad


# 【多重指标】对于 P_k 三角形，alpha=(a0,a1,a2)，各项>=0，且 a0+a1+a2=k。
# 节点坐标为 (a1/k,a2/k)；P_k 三角形一共有 (k+1)(k+2)/2 个节点。
# 三阶例子：顶点(3,0,0)，边节点(2,1,0)，内部节点(1,1,1)。
def simplex_multi_indices(element_type: str, degree: int):
    """P_k 重心整指标，按顶点、边、面、内部排序（任意 k>=1）。"""
    # p 就是有限元多项式次数 k。
    p = int(degree)
    if p < 1:
        raise ValueError("Lagrange degree 必须 >= 1")
    # d=几何维数；n=d+1 是单纯形的顶点数（线段2/三角形3/四面体4）。
    d = {"interval": 1, "triangle": 2, "tet": 3}[element_type]
    n = d + 1
    # 准备遍历所有棱，以便生成边内部插值节点。
    edges = [(0, 1)] if d == 1 else _SIMPLEX_EDGES[element_type]
    indices = []
    # 第一组：生成所有顶点指标。顶点 v 对应 alpha_v=p，其余分量=0。
    for v in range(n):
        # 先创建长度 n 的全零数组，例如三角形 n=3 时 [0,0,0]。
        a = [0] * n
        # 将指定顶点分量设为 p，例如 p=3,v=1 时 [0,3,0]。
        a[v] = p
        indices.append(tuple(a))
    # 第二组：每条边上生成 p-1 个内部节点；p=1 则没有边内部节点。
    for i, j in edges:
        # m=1,...,p-1，让两个端点的重心权重分别为 (p-m)/p 和 m/p。
        for m in range(1, p):
            a = [0] * n
            # 例如 p=3，边(0,1)的两个指标是 (2,1,0) 和 (1,2,0)。
            a[i], a[j] = p-m, m
            indices.append(tuple(a))
    # 第三组（仅四面体）：生成三角形面内部节点。
    # 面内部节点的4个指标恰有1个为0，另外3个都>0。
    if d == 3:
        # excluded 指不属于当前面的那个顶点，其重心坐标固定为0。
        for excluded in range(4):
            # active 保存当前面的另3个顶点编号。
            active = [i for i in range(4) if i != excluded]
            for a in range(1, p):
                for b in range(1, p-a):
                    # a+b+c=p，且面内部必须 a,b,c 都至少为1。
                    c = p-a-b
                    if c >= 1:
                        # 四元指标中排除的那一项保持0。
                        z = [0] * 4
                        for i, v in zip(active, (a, b, c)):
                            z[i] = v
                        indices.append(tuple(z))
    # 第四组：单元内部节点。三角形的全部3个指标必须>0。
    if d == 2:
        for a in range(1, p):
            for b in range(1, p-a):
                # 例如 p=3 时唯一内部节点 alpha=(1,1,1)，坐标=(1/3,1/3)。
                indices.append((a, b, p-a-b))
    # 四面体内部节点要求4个指标全>0。
    elif d == 3:
        for a in range(1, p):
            for b in range(1, p-a):
                for c in range(1, p-a-b):
                    indices.append((a, b, c, p-a-b-c))
    # 排序依次是：顶点 → 边内部 → （四面体）面内部 → 单元内部。
    # 这个顺序还会被基函数求值、参考节点生成等函数共同使用。
    return indices


def triangle_lagrange_basis(k, xi):
    """在二维参考三角形上计算全部 k 阶 Lagrange 基函数及其参考梯度。

    对应 PDF“任意阶 Lagrange 基函数”一节。

    1. 参考三角形：K̂={(ξ,η): ξ>=0, η>=0, ξ+η<=1}。
    2. 重心坐标：λ1=1-ξ-η，λ2=ξ，λ3=η。
    3. 每个三元整指标 (a,b,c) 满足 a+b+c=k，对应插值点 (b/k,c/k)。
    4. 该点对应的基函数为
          φ_abc = [∏_{m=0}^{a-1}(kλ1-m)/(a-m)]
                  [∏_{m=0}^{b-1}(kλ2-m)/(b-m)]
                  [∏_{m=0}^{c-1}(kλ3-m)/(c-m)]。
       代码中分母使用 m+1，所有因子相乘后分母仍是 a!、b!、c!，
       因而与文档中的乘积公式完全等价，只是因子顺序略有不同。

    参数
    ----
    k : int
        Lagrange 多项式次数（k>=1），与网格划分次数无关。
    xi : array-like，长度为 2
        一个参考坐标点 (ξ,η)，不是所有插值节点的集合。

    返回
    ----
    values : ndarray，形状 (Nk,)
        在参考点 (ξ,η) 处所有局部基函数的值。
    grads : ndarray，形状 (Nk,2)
        每行 [∂φ_i/∂ξ, ∂φ_i/∂η]，仍是参考梯度。
        真正用于物理三角形刚度矩阵时，还要做 J^(-T) 梯度变换。

    重要：两个返回数组的第 i 项严格对应 simplex_multi_indices 给出的
    第 i 个节点；切勿随意改变节点顺序，否则组装会产生错误。
    """
    # 把输入点整理成两个标量；例如 xi=[0.2,0.3]。
    xi, eta = np.asarray(xi, dtype=float).ravel()

    # PDF 的重心坐标 λ1、λ2、λ3。
    # 例如 (ξ,η)=(0.2,0.3)，则 λ=(0.5,0.2,0.3)，并且和为 1。
    lam = (1.0 - xi - eta, xi, eta)

    # values 保存每个 φ_abc；grads 保存每个 φ_abc 的两个参考偏导数。
    values, grads = [], []

    def factor(t, a):
        """计算一维乘积 F_a(t) 及其导数 F'_a(t)。

        F_a(t) = ∏_{m=0}^{a-1} (k*t-m)/(m+1)。
        例如 k=3,a=2 时：F_2(t)=(3t)(3t-1)/2。

        a=0 时，空乘积约定为 1，导数为 0。
        这里用逐次乘法及乘积求导法则，不用 F'/F，因此当 F=0
        （恰好在其他插值节点）时，也不会出现除以零的问题。
        """
        # 初始尚未乘任何因子，所以 F=1，F'=0。
        value, derivative = 1.0, 0.0

        # 对 m=0,...,a-1 逐项计算乘积；若 a=0，循环不会执行。
        for m in range(a):
            # 本次乘上的因子 z(t)=(k*t-m)/(m+1)。
            z = (k * t - m) / (m + 1)

            # z'(t)=k/(m+1)，它是一个常数。
            # 新乘积 = 旧乘积 * z。
            # 新导数 = 旧导数 * z + 旧乘积 * z'（乘积求导）。
            # Python 同时赋值会先使用右侧的“旧值”，因此不会覆盖错误。
            derivative, value = (
                derivative * z + value * k / (m + 1),
                value * z,
            )

        # 分别返回 F_a(t) 和 dF_a(t)/dt。
        return value, derivative

    # 逐一处理全部 (a,b,c)：顶点、边内部、三角形内部节点均包含在内。
    # Nk=(k+1)(k+2)/2；例如 k=3 时共 10 个三元整指标。
    for a, b, c in simplex_multi_indices("triangle", k):
        # φ_abc = A(λ1)*B(λ2)*C(λ3)。
        # A,dA：第一组因子 A(λ1) 及对 λ1 的导数；B、C 同理。
        A, dA = factor(lam[0], a)
        B, dB = factor(lam[1], b)
        C, dC = factor(lam[2], c)

        # 当前基函数在参考点 (ξ,η) 的值。
        values.append(A * B * C)

        # 下面利用链式法则计算对 ξ、η 的导数：
        # λ1=1-ξ-η，λ2=ξ，λ3=η，所以
        # ∂λ1/∂ξ=-1，∂λ2/∂ξ=1，∂λ3/∂ξ=0；
        # ∂λ1/∂η=-1，∂λ2/∂η=0，∂λ3/∂η=1。
        #
        # ∂φ/∂ξ = (∂φ/∂λ1)*(-1) + (∂φ/∂λ2)*(1)
        #        = -dA*B*C + A*dB*C；
        # ∂φ/∂η = (∂φ/∂λ1)*(-1) + (∂φ/∂λ3)*(1)
        #        = -dA*B*C + A*B*dC。
        dphi_dxi = -dA * B * C + A * dB * C
        dphi_deta = -dA * B * C + A * B * dC
        grads.append((dphi_dxi, dphi_deta))

    # 第 i 个 values 和 grads 对应同一个局部插值节点。
    # 转成 NumPy 数组便于后续单元刚度矩阵数值积分。
    return np.asarray(values), np.asarray(grads)


# 【P_k 基函数核心】给定参考点 xi，返回所有局部 Lagrange 基函数值与梯度。
# 数学公式：L_alpha(lambda)=Π_i Π_{j=0}^{alpha_i-1} (p*lambda_i-j)/(j+1)。
# 每个多重指标 alpha 对应一个基函数，并满足 L_alpha(x_beta)=delta_{alpha,beta}。
def simplex_basis(element_type: str, degree: int, xi):
    """任意 P_k 单纯形等距节点 Lagrange 基与参考梯度。"""
    # 二维三角形 Pk 使用与文档完全一致的重心坐标乘积公式。
    if element_type == "triangle":
        return triangle_lagrange_basis(int(degree), xi)
    p = int(degree)
    # 线段单独复用一维插值函数；结果重新排序成“端点在前，内部点在后”。
    if element_type == "interval":
        t = float(np.asarray(xi).ravel()[0])
        vals, der = _lagrange_1d(p, t)
        indices = simplex_multi_indices(element_type, p)
        # 一维 multi-index (p-m,m) 对应坐标 t=m/p，故取第二个分量定位节点。
        positions = [a[1] for a in indices]
        # der[positions,None] 把一维导数从 (p+1,) 变为 (p+1,1)。
        return vals[positions], der[positions, None]
    # lam：各重心坐标的值；dlam：重心坐标关于参考变量的导数矩阵。
    lam, dlam = _simplex_lambdas(element_type, xi)
    # L_alpha(lambda) = product_i product_{j=0}^{alpha_i-1}(p*lambda_i-j)/(alpha_i!)
    # 分别积累所有基函数的数值和各基函数的参考梯度。
    values, grads = [], []
    # 例如 P3 三角形有10个 alpha，因此最后 values 长度为10、grads 为10×2。
    for alpha in simplex_multi_indices(element_type, p):
        # 对每个 alpha：将 L_alpha 分解为 F_0(lambda_0)*F_1(lambda_1)*...。
        # factors 存 F_i 的值；derivatives 存 F_i 对 lambda_i 的导数。
        factors = []
        derivatives = []
        # l 是某个重心坐标 lambda_i；a 是对应多重指标 alpha_i。
        for l, a in zip(lam, alpha):
            # 累乘的初始乘积 f=1，初始导数 df=0（常数1的导数）。
            f, df = 1.0, 0.0
            # 逐个乘上因子 t_j=(p*l-j)/(j+1)，共 a 个因子。a=0 时空乘积=1。
            for j in range(a):
                t = (p*l-j)/(j+1)
                # dt=d(t_j)/dl=p/(j+1)，因为 j、p 都是常数。
                dt = p/(j+1)
                # 乘积求导：(f*t)′=f′*t+f*t′。Python 同时赋值使用旧 f、df。
                df, f = df*t+f*dt, f*t
            # 当前重心坐标方向的因子 F_i 及导数 F_i′ 计算完成。
            factors.append(f)
            derivatives.append(df)
        # L_alpha=所有 F_i 相乘，就是当前基函数在 xi 处的值。
        values.append(float(np.prod(factors)))
        # 求 L_alpha 对各重心坐标的导数：dL/dlambda_i=F_i′*Π_{j!=i}F_j。
        # grad_l 是重心坐标梯度，还不是对参考坐标 (a,b)/(a,b,c) 的梯度。
        grad_l = np.array([derivatives[i]*np.prod([factors[j] for j in range(len(alpha)) if j != i]) for i in range(len(alpha))])
        # 链式法则：grad_xi(L)=dlam @ grad_lambda(L)。
        # 例如三角形 d/da=-d/dlambda_0+d/dlambda_1。
        grads.append(dlam @ grad_l)
    # 返回数组形状：(局部节点数,), (局部节点数, 空间维数)。
    # 注意仍是“参考单元”梯度，物理单元上需再做 J^{-T} 变换。
    return np.asarray(values), np.asarray(grads)



# 【一维插值】区间 [0,1] 上节点 x_i=i/p，共 p+1 个。
# 基函数 l_i(t)=Π_{j!=i}(t-x_j)/(x_i-x_j)，具有 l_i(x_j)=delta_ij。
def _lagrange_1d(p: int, t: float):
    """[0,1] 上等距节点 ``i/p`` 的 1D Lagrange 基与其导数。"""
    # 例如 p=2：nodes=[0,0.5,1]；p=3：[0,1/3,2/3,1]。
    nodes = np.arange(p + 1) / p
    # 每个 l_i 是一个连乘，初值1；导数数组初值0。
    values = np.ones(p + 1)
    grads = np.zeros(p + 1)
    # 外层 i：依次构造第 i 个基函数 l_i。
    for i in range(p + 1):
        # 内层 j：遍历除 i 外的节点，构造插值乘积。
        for j in range(p + 1):
            # 因为分母 nodes[i]-nodes[i] 为0，公式本来就要求跳过 j=i。
            if j == i:
                continue
            # 累乘 (t-x_j)/(x_i-x_j) 得到 l_i(t)。
            values[i] *= (t - nodes[j]) / (nodes[i] - nodes[j])
        # 开始计算 l_i′：对乘积中每一个因子分别求导，再把结果相加。
        acc = 0.0
        # k 表示当前对第 k 个因子求导（其他因子不求导）。
        for k in range(p + 1):
            if k == i:
                continue
            # d[(t-x_k)/(x_i-x_k)]/dt = 1/(x_i-x_k)。
            term = 1.0 / (nodes[i] - nodes[k])
            # 将没有被求导的其余因子相乘，跳过 i 和 k。
            for j in range(p + 1):
                # j=i 不属于插值公式；j=k 的因子已求导，不能重复乘。
                if j == i or j == k:
                    continue
                term *= (t - nodes[j]) / (nodes[i] - nodes[j])
            # 根据乘积求导法则累加各项，得到 l_i′(t)。
            acc += term
        # 存储第 i 个一维基函数在 t 处的导数。
        grads[i] = acc
    # 两个数组长度均为 p+1：函数值和一维导数。
    return values, grads


# 【Q_k 张量积】二维：phi_{ij}(x,y)=l_i(x)*l_j(y)。
# 三维：phi_{ijk}(x,y,z)=l_i(x)*l_j(y)*l_k(z)。
# 注意这里只能说明“基函数支持任意阶”，局部自由度描述符另有限制。
def tensor_basis(dim: int, degree: int, xi):
    """张量积 Lagrange 基函数在参考点 xi 处的值与梯度。

    局部自由度顺序为 :func:`tensor_local_descriptors` 给出的顺序。
    """
    # 将 xi 转为一维浮点数组，例如 [[0.2,0.3]] → [0.2,0.3]。
    xi = np.asarray(xi, dtype=float).ravel()
    # 每个坐标方向单独调用一维插值：one_d[d]=(该方向基函数值,该方向导数)。
    one_d = [_lagrange_1d(degree, xi[d]) for d in range(dim)]
    # Q_k 节点指标：二维 (i,j) 共 (p+1)^2 个，三维 (i,j,k) 共 (p+1)^3 个。
    multi = tensor_multi_indices(dim, degree)

    values = []
    grads = []
    # 当前 idx=(i,j) 时，计算 phi_ij 及其两个偏导。
    for idx in multi:
        # 基函数值：将各方向的一维基函数相乘。
        val = 1.0
        for d in range(dim):
            # [d]选坐标方向；[0]选函数值；[idx[d]]选该方向第几个一维基函数。
            val *= one_d[d][0][idx[d]]
        values.append(val)
        # g[d] 保存当前张量积基函数对第 d 个参考坐标的偏导数。
        g = np.zeros(dim)
        for d in range(dim):
            # 对第 d 个方向求导：该方向取 l′，其余方向取 l。
            term = one_d[d][1][idx[d]]
            for e in range(dim):
                if e != d:
                    # 例如 d=0 的二维情形：l_i′(x)*l_j(y)。
                    term *= one_d[e][0][idx[e]]
            # 存好一个偏导数，再处理下一个坐标方向。
            g[d] = term
        grads.append(g)
    # 返回 Q_k 的所有基函数值和参考梯度；形状分别为 (N,) 和 (N,dim)。
    return np.asarray(values, dtype=float), np.asarray(grads, dtype=float)

# ---------------------------------------------------------------------------
# 局部自由度布局
# ---------------------------------------------------------------------------
# 为兼容旧代码保留的入口；实际直接转交给 tensor_multi_indices。
def _tensor_index_and_kind(dim: int, degree: int):
    """兼容旧名，返回与 :func:`tensor_multi_indices` 相同的多重指标列表。"""
    return tensor_multi_indices(dim, degree)


# 【张量积节点顺序】生成所有整数组合，并排序成：顶点、边、面、体内部。
def tensor_multi_indices(dim: int, degree: int):
    """Q_k 所有张量积节点：先顶点，再棱、面、内部。"""
    # itertools.product 相当于多个方向做笛卡尔积。
    from itertools import product
    # p 就是有限元多项式次数 k。
    p = int(degree)
    if p < 1:
        raise ValueError("Lagrange degree 必须 >= 1")
    corners = TENSOR_CORNERS[dim]
    # 二阶二维示例：先得到 (0,0),(0,1),...,(2,2) 共9个指标。
    all_idx = list(product(range(p+1), repeat=dim))
    # 排序关键字：按节点位于几何体的哪个维度，从低维实体到高维实体。
    def codim_key(idx):
        # 计算有多少坐标指标在端点 0 或 p。二维 boundary=2 是顶点，=1 是边，=0 是内部。
        boundary = sum(i in (0, p) for i in idx)
            # 全部坐标为端点，说明这是顶点：使用 TENSOR_CORNERS 规定的角点顺序。
        if boundary == dim:
            return (0, corners.index(tuple(i//p for i in idx)), ())
        # 非顶点：先按余维排序，再根据整数坐标 idx 排序。
        return (dim-boundary, 0, idx)
    # 警告：该函数的非顶点排序与后面的 tensor_local_descriptors 在 Q2 时
    # 未必逐项匹配（例如二维 Q2 的4个边中点顺序不同），组装前应核对。
    return sorted(all_idx, key=codim_key)


# 【张量积自由度身份】描述节点属于哪个顶点/棱/面/内部，供全局自由度管理。
# 注意：此函数仅实现 p=1,2，不能用于 p>=3 的高阶 Q_k 完整组装。
def tensor_local_descriptors(dim: int, degree: int):
    """张量积单元的局部自由度描述符列表（与 :func:`tensor_multi_indices` 对齐）。"""
    # p 就是有限元多项式次数 k。
    p = int(degree)
    corners = TENSOR_CORNERS[dim]
    # 先保存所有顶点，例如二维四边形有4个：("vertex",0),...,3。
    descriptors = [("vertex", k) for k in range(len(corners))]
    # 一次 Q1 单元只有角点自由度，直接返回。
    if p == 1:
        return descriptors
    # 原始代码的限制：其他阶数直接抛出异常，不是完整任意阶 Q_k 描述符。
    if p != 2:
        raise NotImplementedError(
            f"{dim}D 张量积单元仅实现了 degree=1, 2（收到 degree={degree}）"
        )
    # 二维使用四边形4条边；三维使用六面体12条棱。
    edges = _QUAD_EDGES if dim == 2 else HEX_EDGES
    # 对每条边加入一个边内部自由度。仅适用于 p=2，每条边只有1个中点。
    descriptors += [("edge", e) for e in edges]
    # 三维 Q2 还有6个面中心自由度；二维无面内部自由度这一层。
    if dim == 3:
        descriptors += [("face", f) for f in HEX_FACES_LOCAL]
    # 增加单元中心：二维 Q2 是 (1,1)，三维 Q2 是 (1,1,1)。
    descriptors.append(("interior", tuple([p // 2] * dim)))
    # 输出的是“局部身份标签”，不是全局编号，也不是节点坐标。
    return descriptors


# 【单纯形自由度身份】描述每个 P_k 节点属于顶点、边、面或内部。
# 为支持 k>2 的每条边多个点，会额外存储这些点在边上的整数权重。
def simplex_local_descriptors(element_type: str, degree: int):
    """任意阶单纯形自由度描述符；高阶包含每条边/面上的多个节点。"""
    # P1 仅顶点；P2 在顶点之外每条边恰有一个中点，可用简单标签。
    if degree <= 2:
        n_v = {"interval": 2, "triangle": 3, "tet": 4}[element_type]
        edges = [(0, 1)] if element_type == "interval" else _SIMPLEX_EDGES[element_type]
        return [("vertex", i) for i in range(n_v)] + ([("edge", e) for e in edges] if degree == 2 else [])
    # 对 k>2：用多重指标 alpha 判断节点的几何归属。
    out = []
    for alpha in simplex_multi_indices(element_type, degree):
        # active 记录 alpha 中非零分量的位置。
        # 三角形：1个非零→顶点；2个→边；3个→内部。
        # 四面体：1个→顶点；2个→棱；3个→面；4个→内部。
        active = tuple(i for i, a in enumerate(alpha) if a)
            # 仅一个重心分量非零，位置必是一个顶点。
        if len(active) == 1:
            out.append(("vertex", active[0]))
            # 恰有两个非零分量：边内部节点。保存边端点与沿边权重，区分多个点。
        elif len(active) == 2:
            out.append(("edge", (active, tuple(alpha[i] for i in active))))
            # 四面体中三个非零分量：节点在三角形面内部。
        elif element_type == "tet" and len(active) == 3:
            out.append(("face", (active, tuple(alpha[i] for i in active))))
            # 三角形的三个分量均非零，或四面体的四个分量均非零：单元内部。
        else:
            out.append(("interior", alpha))
    # 返回顺序与 simplex_multi_indices 完全一致，便于基函数和节点配对。
    return out


# 【参考节点】将重心整数指标 alpha 转为实际的参考坐标。
# 对三角形，alpha=(a0,a1,a2) → (a1/p,a2/p)。
# 例如 p=3,alpha=(2,1,0) → (1/3,0)。
def simplex_reference_nodes(element_type: str, degree: int):
    """P_k 节点的参考坐标，顺序与 simplex_basis 完全一致。"""
    # 丢掉首个重心指标（由其余指标和 p 可以恢复），再除以 p。
    # 返回 N×d 数组，与 simplex_basis 的 N 个基函数按同一顺序对应。
    return np.asarray([alpha[1:] for alpha in simplex_multi_indices(element_type, degree)], dtype=float) / degree