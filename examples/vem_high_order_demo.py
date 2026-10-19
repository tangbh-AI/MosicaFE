"""Run unchanged sine examples with high-order VEM in 2D and 3D.

Usage: python examples/vem_high_order_demo.py
Outputs: examples/output/vem_high_order (CSV, NPZ, solution/error PNGs, report).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path
import re
import sys
import time

# Small dense element matrices benefit from a single BLAS thread.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _source_import import load_local
mf = load_local()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Reuse the existing example data verbatim, including the existing 3D example.
spec = importlib.util.spec_from_file_location('original_checks', ROOT/'tests'/'run_checks.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def sample_projection(space, u, points):
    """Sample cell energy projections without interpolating vertex errors."""
    values = np.full(len(points), np.nan)
    for c in range(space.mesh.n_cells):
        xyz = space.mesh.get_cell_vertices(c)
        candidates = np.flatnonzero(np.isnan(values) &
            np.all(points >= xyz.min(axis=0)-1e-11, axis=1) &
            np.all(points <= xyz.max(axis=0)+1e-11, axis=1))
        if not len(candidates):
            continue
        if len(xyz) == space.mesh.dim+1:
            bary = np.linalg.solve((xyz[1:]-xyz[0]).T,
                                   (points[candidates]-xyz[0]).T).T
            inside = (bary.min(axis=1) >= -1e-10) & (bary.sum(axis=1) <= 1+1e-10)
            candidates = candidates[inside]
        # Demo meshes are simplices or axis-aligned boxes.
        values[candidates] = space.evaluate(c, u, points[candidates])
    if not np.isfinite(values).all():
        raise RuntimeError('Plot sampling left points outside the mesh')
    return values


def plot_fields(space, u, exact, name, out):
    dim, degree = space.mesh.dim, space.degree
    grid = np.linspace(0,1,129 if dim == 2 else 57)
    X,Y = np.meshgrid(grid,grid)
    pts = np.column_stack((X.ravel(),Y.ravel()))
    if dim == 3:
        pts = np.column_stack((pts, np.full(len(pts),.5)))
    uh = sample_projection(space,u,pts).reshape(X.shape)
    ue = np.array([exact(p) for p in pts]).reshape(X.shape)
    error = abs(uh-ue)
    fig, axes = plt.subplots(1,3,figsize=(14.2,4.2),layout='constrained')
    for ax,data,label,cmap in zip(axes,(ue,uh,error),
            ('Exact solution','VEM energy projection', 'Absolute error'),
            ('viridis','viridis','magma')):
        limits = dict(vmin=0,vmax=1) if label != 'Absolute error' else dict(vmin=0)
        im=ax.pcolormesh(X,Y,data,shading='auto',cmap=cmap,**limits)
        ax.set(xlabel='x',ylabel='y',title=label,aspect='equal')
        fig.colorbar(im,ax=ax,shrink=.85)
    extra = ', z = 0.5' if dim==3 else ''
    fig.suptitle(f'{name.capitalize()}, {dim}D VEM degree {degree}{extra}')
    fig.savefig(out/f'{name}_{dim}d_k{degree}_fields.png',dpi=170)
    plt.close(fig)
    if dim==2:
        return
    # Actual three-dimensional interior cutplanes; boundary-only plots would
    # be uninformative because these examples have homogeneous boundary data.
    planes=[]
    for fixed in range(3):
        coords=[X.copy(),Y.copy()]
        xyz=[]
        for d in range(3):
            xyz.append(np.full_like(X,.5) if d==fixed else coords.pop(0))
        points=np.column_stack([a.ravel() for a in xyz])
        numerical=sample_projection(space,u,points).reshape(X.shape)
        truth=np.array([exact(p) for p in points]).reshape(X.shape)
        planes.append((xyz,truth,numerical,abs(numerical-truth)))
    vmax=max(float(p[3].max()) for p in planes)
    fig=plt.figure(figsize=(15,4.8),layout='constrained')
    for j,title in enumerate(('Exact solution','VEM energy projection','Absolute error')):
        ax=fig.add_subplot(1,3,j+1,projection='3d')
        cmap=plt.get_cmap('magma' if j==2 else 'viridis')
        norm=Normalize(0,vmax if j==2 else 1)
        for xyz,*data in planes:
            ax.plot_surface(*xyz,facecolors=cmap(norm(data[j])),shade=False,
                            rstride=1,cstride=1,linewidth=0,antialiased=False)
        ax.set(xlabel='x',ylabel='y',zlabel='z',title=title)
        ax.set_box_aspect((1,1,1))
        fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.6,pad=.05)
    fig.suptitle(f'{name.capitalize()}, 3D VEM degree {degree}: interior cutplanes')
    fig.savefig(out/f'{name}_3d_k{degree}_cutplanes.png',dpi=170)
    plt.close(fig)


def plot_convergence(rows,out):
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for d in (2,3):
        for j,name in enumerate(('poisson','helmholtz')):
            ax=axes[d-2,j]
            for degree in sorted({r['degree'] for r in rows}):
                data=[r for r in rows if r['dim']==d and r['problem']==name and r['degree']==degree]
                if not data:
                    continue
                h=[r['h'] for r in data]
                ax.loglog(h,[r['L2'] for r in data],'o-',label=f'k={degree}, L2')
                ax.loglog(h,[r['H1'] for r in data],'s--',label=f'k={degree}, H1 seminorm')
            ax.set(title=f'{name.capitalize()} {d}D',xlabel='h',ylabel='Projected error norm')
            ax.grid(True,which='both',alpha=.25)
            ax.legend(fontsize=8)
    fig.savefig(out/'convergence.png',dpi=170)
    plt.close(fig)


def write_report(rows,out):
    lines=['# 高阶 VEM 运行结果', '',
        '运行命令：`python examples/vem_high_order_demo.py`。', '',
        '## 算例与框架', '',
        '二维沿用指导书和 `examples/poisson_demo.py`：单位正方形，'
        '`u=sin(pi*x)sin(pi*y)`，齐次 Dirichlet 边界。三维沿用 '
        '`tests/run_checks.py`：单位立方体，`u=sin(pi*x)sin(pi*y)sin(pi*z)`。', '',
        'Poisson 为 `-Delta u=f`，`f=d*pi^2*u`；Helmholtz 为 '
        '`-Delta u-wave_number^2*u=f`，波数保持指导书的 1，'
        '`f=(d*pi^2-1)*u`。三维 Helmholtz 在原三维正弦算例上采用同一方程，'
        '未替换精确解、区域或边界。这里 k 表示离散多项式次数，不是波数。', '',
        '原有 core → spaces → physics → solvers → postprocess 分层不变。'
        '调用仍为 `FESpace.vem(mesh, degree=k)`，物理类和求解接口不变。'
        '支持任意整数 k>=1，k=1 保留原路径；本报告验证并运行 k=2、3。', '',
        '## 方法与代码阅读', '',
        '- `core/mesh.py`：网格、边界面与几何；`core/quadrature.py`：'
        '积分。高阶单纯形规则已补齐，避免把三/四阶规则当成任意高阶规则。',
        '- `spaces/vem.py`：二维能量投影、稳定化、刚度、质量、载荷和重构。'
        '自由度是顶点值、每条边 k-1 个内部点值、次数 <=k-2 的归一化单元矩。',
        '- `spaces/vem_high.py`：三维增加共享面矩和体矩，面迹递归构造二维增强 VEM。'
        '同一面使用统一的局部坐标和自由度方向；面矩边界条件按积分赋值。',
        '- 能量投影由分部积分计算，并用已知单元均值固定常数。'
        '增强约束提供 k-1、k 次矩，低次矩使用真实自由度，构造完整 L2 投影。'
        '质量矩阵为投影质量加核空间稳定化；载荷为 L2 投影载荷。',
        '- `physics/poisson.py` 与 `physics/helmholtz.py`：继续组合空间矩阵，'
        '直接法解稀疏方程；`postprocess/error.py`：数值积分误差。'
        '矩自由度不是点值，已从点值 Linf 比较中排除。', '',
        '### 与指导书一阶实现的对应关系', '',
        '设 D 是多项式到自由度的矩阵，B 是多项式梯度 Gram 矩阵，H 是多项式质量矩阵。'
        '高阶仍沿用指导书的“一致性项 + 稳定化项”：', '',
        '`B Pi = R`，其中 `R[alpha,i] = integral_boundary(phi_i * dn(m_alpha)) '
        '- integral_cell(phi_i * Delta(m_alpha))`；常数模式用单元均值补足。', '',
        '`K_local = Pi.T B Pi + sigma (I-D Pi).T (I-D Pi)`；'
        '`M_local = Pi0.T H Pi0 + |K| (I-D Pi0).T (I-D Pi0)`；'
        '`b_local = Pi0.T [integral_cell(f*m_alpha)]`。', '',
        '二维单元自由度数为 `Nv + Ne*(k-1) + k*(k-1)/2`；'
        '三维为 `Nv + Ne*(k-1) + Nf*k*(k-1)/2 + (k+1)*k*(k-1)/6`。'
        '例如三阶四面体每单元 32 个局部自由度；共享顶点、边和面自由度后，'
        '全局自由度数小于局部数之和。', '',
        '## 最细网格结果', '',
        '| 问题 | 维数 | k | n | 单元数 | 自由度 | L2 | H1 半范数 | 采样 Linf | 相对残差 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    log_path = ROOT/'tests/regression_results.txt'
    if log_path.exists():
        summaries = re.findall(r'通过\s+\d+，失败\s+\d+', log_path.read_text(encoding='utf-8-sig'))
        if summaries:
            lines[2:2] = ['验证记录：' + summaries[-1] + '（`python tests/run_tests.py`）。', '']
    finest=[]
    for dim in (2,3):
        for name in ('poisson','helmholtz'):
            for degree in sorted({r['degree'] for r in rows}):
                data=[r for r in rows if (r['dim'],r['problem'],r['degree'])==(dim,name,degree)]
                if not data: continue
                r=data[-1]; finest.append(r)
                lines.append(f"| {name} | {dim} | {degree} | {r['n']} | {r['cells']} | {r['dofs']} | "
                             f"{r['L2']:.6e} | {r['H1']:.6e} | {r['Linf']:.6e} | {r['relative_residual']:.2e} |")
    lines += ['', '## 最后两级网格收敛阶', '',
              '| 问题 | 维数 | k | L2 阶（预期 k+1） | H1 阶（预期 k） |',
              '|---|---:|---:|---:|---:|']
    for r in finest:
        lines.append(f"| {r['problem']} | {r['dim']} | {r['degree']} | {r['L2_rate']:.3f} | {r['H1_rate']:.3f} |")
    lines += ['', '## 误差与图的含义', '',
        'VEM 内部基函数不显式可得，L2 和 H1 分别比较能量投影 '
        '`Pi^nabla_k u_h` 及其梯度与精确解，图也使用同一投影。'
        '报告的是可计算的投影误差，不能声称为虚拟函数本体的精确范数。'
        'Linf 是求积点、顶点和边节点上的采样最大误差，不是严格的连续上确界。', '',
        '二维图包含精确解、数值投影解和绝对误差；三维同时提供 z=0.5 截面与 '
        'x/y/z=0.5 三个内部切面的空间云图。齐次边界上的解为零，'
        '因此不采用只展示外表面的云图。色标是实际数值，没有对误差进行归一化。', '',
        '主运行保持已有三角形与四面体网格。高阶投影、刚度、质量与非零边界多项式 '
        'patch tests 还覆盖四边形和六面体；积分检验覆盖更高次数。', '',
        '完整逐级数据见 [results.csv](results.csv)，最细网格的自由度、'
        '单元连接和投影系数保存在各个 NPZ 文件。', '',
        '实现参考：[The Virtual Element Method, Acta Numerica 2023](https://iris.cnr.it/bitstream/20.500.14243/538888/1/the-virtual-element-method.pdf)，'
        '其中增强空间与三维递归面构造用于补充指导书的一阶方法。', '',
        '![收敛曲线](convergence.png)']
    for r in finest:
        stem=f"{r['problem']}_{r['dim']}d_k{r['degree']}"
        lines += ['', f"### {r['problem']} {r['dim']}D, k={r['degree']}", '',
                  f'![解与误差]({stem}_fields.png)']
        if r['dim']==3:
            lines += ['', f'![三维内部切面云图]({stem}_cutplanes.png)']
    (out/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degrees',nargs='+',type=int,default=[2,3])
    parser.add_argument('--levels2',nargs='+',type=int,default=[4,8,16,32])
    parser.add_argument('--levels3',nargs='+',type=int,default=[2,4,8])
    parser.add_argument('--output',type=Path,default=ROOT/'examples/output/vem_high_order')
    args=parser.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for dim,levels in ((2,args.levels2),(3,args.levels3)):
        exact=getattr(original,f'exact_{dim}d')
        gradient=getattr(original,f'grad_{dim}d')
        base_source=getattr(original,f'source_{dim}d')
        for degree in args.degrees:
            previous={}
            for n in levels:
                mesh=(mf.Mesh.rectangle(nx=n,ny=n) if dim==2 else mf.Mesh.box(nx=n,ny=n,nz=n))
                space=mf.FESpace.vem(mesh,degree=degree)
                for name in ('poisson','helmholtz'):
                    started=time.perf_counter()
                    problem=(mf.PoissonProblem(space,f=base_source,g=original.ZERO) if name=='poisson' else
                        mf.HelmholtzProblem(space,f=lambda x:base_source(x)-exact(x),g=original.ZERO,k=1.0))
                    u=problem.solve(method='direct')
                    errors=problem.compute_errors(u,exact,gradient)
                    A,b=problem._system
                    residual=float(np.linalg.norm(A@u-b)/max(np.linalg.norm(b),1e-30))
                    row=dict(problem=name,dim=dim,degree=degree,n=n,cells=mesh.n_cells,
                             dofs=space.n_dofs,h=mesh.get_mesh_size(),**errors,
                             relative_residual=residual,seconds=time.perf_counter()-started)
                    for norm in ('L2','H1'):
                        old=previous.get(name)
                        row[norm+'_rate']=(float(np.log(old[norm]/row[norm])/np.log(old['h']/row['h']))
                                          if old else float('nan'))
                    if not np.isfinite([row['L2'],row['H1'],residual]).all() or residual>1e-8:
                        raise RuntimeError(f'Invalid solve: {row}')
                    previous[name]=row; rows.append(row)
                    print(f"{name:9s} {dim}D k={degree} n={n:2d} dofs={space.n_dofs:6d} "
                          f"L2={row['L2']:.5e} H1={row['H1']:.5e} "
                          f"rates={row['L2_rate']:.2f}/{row['H1_rate']:.2f} ({row['seconds']:.1f}s)",flush=True)
                    if n==levels[-1]:
                        plot_fields(space,u,exact,name,out)
                        coefficients=np.array([space.reconstruction_matrix(c) @ u[space.get_cell_dofs(c)]
                                               for c in range(mesh.n_cells)])
                        np.savez_compressed(out/f'{name}_{dim}d_k{degree}.npz',nodes=mesh.nodes,
                            elements=np.asarray(mesh.elements),dof_map=np.asarray(space.dof_map),
                            monomial_exponents=np.array(space._projection_high(0)['multi']),
                            dof_coordinates=space.dof_coords,solution_dofs=u,projection_coefficients=coefficients,
                            centroids=np.array([space._projection_high(c)['centroid'] for c in range(mesh.n_cells)]),
                            diameters=np.array([space._projection_high(c)['h'] for c in range(mesh.n_cells)]))
                    with (out/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
                        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot_convergence(rows,out)
    write_report(rows,out)
    print(f'Finished: {out}',flush=True)


if __name__=='__main__':
    main()
