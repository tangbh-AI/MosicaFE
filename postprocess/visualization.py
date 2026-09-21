"""可视化工具（guidebook 第 9 章）。

设计约定：

* 所有函数在**没有安装 matplotlib** 时打印一条提示并返回 ``None``，不抛异常；
* 传 ``filename=...`` 时自动保存 PNG（``dpi`` 可调），传 ``show=True`` 时弹窗；
* 三角形/四边形/多边形/四面体/六面体网格都能画：2D 用扇形三角化，
  3D 用切片（``z_slice=...``）。

.. code-block:: python

   from MosicaFE.postprocess import plot_solution, plot_mesh, plot_convergence

   plot_solution(u, mesh, V, title="VEM solution", filename="vem.png")
   plot_mesh(mesh, filename="mesh.png")
"""

from __future__ import annotations

import os

import numpy as np

from .._compat import HAS_MATPLOTLIB, matplotlib_or_none

__all__ = [
    "plot_mesh",
    "plot_solution",
    "plot_convergence",
    "plot_pressure_2d",
    "plot_flux_2d",
    "plot_error",
    "compare_solutions",
    "save_figure",
]


def _require_pyplot():
    plt = matplotlib_or_none()
    if plt is None:
        print(
            "[MosicaFE] 未安装 matplotlib，跳过绘图。"
            "安装方式：pip install matplotlib"
        )
    return plt


def save_figure(fig, filename, show=False, dpi=200):
    """保存并（可选）显示图像，返回被保存的绝对路径。"""
    path = None
    if filename:
        fig.tight_layout()
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        path = os.path.abspath(filename)
        print(f"[MosicaFE] 图像已保存：{path}")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return path


# ---------------------------------------------------------------------------
# 网格 → 三角剖分（绘图用）
# ---------------------------------------------------------------------------
def _triangulate_2d(mesh, nodal_values, cell_values=None):
    """把任意 2D 多边形网格转成 matplotlib 可用的三角剖分。

    Returns
    -------
    xy : (n_pts, 2) ndarray
    triangles : (n_tri, 3) ndarray
    values : (n_pts,) ndarray
    """
    xy = list(np.asarray(mesh.nodes, dtype=float))
    values = list(np.asarray(nodal_values, dtype=float))
    triangles = []
    for c, cell in enumerate(mesh.elements):
        if cell_values is None:
            cval = float(np.mean([values[v] for v in cell]))
        else:
            cval = float(cell_values[c])
        n = len(cell)
        if n == 3:
            triangles.append([int(cell[0]), int(cell[1]), int(cell[2])])
            continue
        centroid = np.asarray(mesh.nodes[cell], dtype=float).mean(axis=0)
        cidx = len(xy)
        xy.append(centroid)
        values.append(cval)
        for i in range(n):
            triangles.append([int(cell[i]), int(cell[(i + 1) % n]), cidx])
    return np.asarray(xy), np.asarray(triangles, dtype=int), np.asarray(values)


def _slice_3d(mesh, nodal_values, z_slice, tol=None):
    """取出 3D 网格在 ``z = z_slice`` 附近的节点，返回 2D 三角剖分。"""
    z = np.asarray(mesh.nodes, dtype=float)[:, 2]
    tol = tol if tol is not None else 0.35 * mesh.get_mesh_size()
    ids = np.where(np.abs(z - z_slice) < tol)[0]
    if ids.size < 3:
        return None
    xy = np.asarray(mesh.nodes, dtype=float)[ids][:, :2]
    vals = np.asarray(nodal_values, dtype=float)[ids]
    import matplotlib.tri as mtri

    return mtri.Triangulation(xy[:, 0], xy[:, 1]), vals


# ---------------------------------------------------------------------------
def plot_mesh(mesh, title=None, filename=None, show=False, ax=None, dpi=200):
    """画网格结构（2D 直接画线；3D 画 z 中截面）。"""
    plt = _require_pyplot()
    if plt is None:
        return None
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6, 5.4))

    if mesh.dim == 3:
        z_mid = 0.5 * (mesh.nodes[:, 2].min() + mesh.nodes[:, 2].max())
        tol = 0.35 * mesh.get_mesh_size()
        ids = np.where(np.abs(mesh.nodes[:, 2] - z_mid) < tol)[0]
        if ids.size < 3:
            raise ValueError("切面节点太少，无法绘图")
        ax.triplot(
            mesh.nodes[ids, 0], mesh.nodes[ids, 1],
            color="k", linewidth=0.6,
        )
        ax.set_title(title or f"Mesh (z={z_mid:.3g} slice)")
    else:
        for cell in mesh.elements:
            v = np.asarray(mesh.nodes[cell], dtype=float)
            loop = np.vstack([v, v[:1]])
            ax.plot(loop[:, 0], loop[:, 1], "k-", linewidth=0.6)
        ax.scatter(mesh.nodes[:, 0], mesh.nodes[:, 1], s=8, c="tab:red", zorder=5)
        ax.set_title(
            title or f"{mesh.element_type} mesh: {mesh.n_cells} cells, {mesh.n_nodes} nodes"
        )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if own:
        save_figure(ax.figure, filename, show, dpi)
    return ax


# ---------------------------------------------------------------------------
def _plot_2d_field(ax, mesh, values, title, cmap="viridis", levels=50,
                   cell_values=None):
    import matplotlib.tri as mtri

    xy, tris, vals = _triangulate_2d(mesh, values, cell_values=cell_values)
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
    tcf = ax.tricontourf(triang, vals, levels=levels, cmap=cmap)
    ax.figure.colorbar(tcf, ax=ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)


def plot_solution(
    u,
    mesh=None,
    space=None,
    title="Solution",
    filename=None,
    show=False,
    mode="contour",
    z_slice=0.5,
    levels=50,
    ax=None,
    dpi=200,
):
    """画数值解。

    Parameters
    ----------
    u : ndarray
        自由度向量（或网格节点值）。
    mesh : Mesh
        省略时从 ``space.mesh`` 取。
    space : BaseFESpace, optional
        给了就用 ``space.nodal_values(u)`` 还原到节点上。
    mode : {"contour", "surface"}
        2D 时可选等值线填充或三维曲面。
    """
    plt = _require_pyplot()
    if plt is None:
        return None
    if mesh is None and space is not None:
        mesh = space.mesh
    if mesh is None:
        raise ValueError("plot_solution 需要 mesh 或 space 之一")
    if space is not None:
        nodal = space.nodal_values(u)
    else:
        nodal = np.asarray(u, dtype=float).ravel()
        if nodal.size != mesh.n_nodes:
            raise ValueError(
                f"解的长度 {nodal.size} 与网格节点数 {mesh.n_nodes} 不一致；"
                "自由度向量需要同时传 space= 才能还原到节点（例如 P2 的边中点自由度）。"
            )

    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6, 5.4))

    if mesh.dim == 1:
        x = mesh.nodes[:, 0]
        ax.plot(x, nodal, "o-", lw=1.2, ms=3)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(title)
    elif mesh.dim == 2:
        if mode == "surface":
            import matplotlib.tri as mtri

            xy, tris, vals = _triangulate_2d(mesh, nodal)
            triang = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
            ax.remove()
            ax = ax.figure.add_subplot(111, projection="3d")
            ax.plot_trisurf(triang, vals, cmap="viridis", linewidth=0.1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("u")
            ax.set_title(title)
        else:
            _plot_2d_field(ax, mesh, nodal, title, levels=levels)
    else:
        out = _slice_3d(mesh, nodal, z_slice)
        if out is None:
            raise ValueError(f"z={z_slice} 附近节点太少，无法绘制切片")
        triang, vals = out
        ax.tricontourf(triang, vals, levels=levels, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title}  (z={z_slice:g} slice)")

    if own:
        save_figure(ax.figure, filename, show, dpi)
    return ax


# ---------------------------------------------------------------------------
def plot_error(u, mesh=None, space=None, exact=None, title="Error",
               filename=None, show=False, levels=50, ax=None, dpi=200):
    """画逐点误差 ``|u_h - u|``。"""
    plt = _require_pyplot()
    if plt is None:
        return None
    if mesh is None and space is not None:
        mesh = space.mesh
    nodal = space.nodal_values(u) if space is not None else np.asarray(u, float)
    if exact is None:
        raise ValueError("plot_error 需要 exact=精确解函数")
    if space is not None:
        exact_vals = space._evaluate_function(exact, mesh.nodes)
    else:
        exact_vals = np.array([float(exact(p)) for p in mesh.nodes])
    err = np.abs(nodal - exact_vals)

    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6, 5.4))
    if mesh.dim == 1:
        ax.plot(mesh.nodes[:, 0], err, "r.-")
        ax.set_xlabel("x")
        ax.set_title(title)
    elif mesh.dim == 2:
        _plot_2d_field(ax, mesh, err, title, cmap="YlOrRd", levels=levels)
    else:
        out = _slice_3d(mesh, err, 0.5)
        ax.tricontourf(out[0], out[1], levels=levels, cmap="YlOrRd")
        ax.set_aspect("equal")
        ax.set_title(title)
    if own:
        save_figure(ax.figure, filename, show, dpi)
    return ax


# ---------------------------------------------------------------------------
def plot_pressure_2d(mesh, P, title="Pressure (P0)", filename=None, show=False,
                     levels=50, ax=None, dpi=200):
    """画 RT0-P0 混合格式下的分片常数压力。"""
    plt = _require_pyplot()
    if plt is None:
        return None
    if mesh.dim != 2:
        raise ValueError("plot_pressure_2d 只支持 2D")
    P = np.asarray(P, dtype=float).ravel()
    nodal = np.zeros(mesh.n_nodes)
    cnt = np.zeros(mesh.n_nodes)
    for c, cell in enumerate(mesh.elements):
        for v in cell:
            np.add.at(nodal, v, P[c])
            np.add.at(cnt, v, 1.0)
    cnt[cnt == 0] = 1
    nodal /= cnt
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6, 5.4))
    _plot_2d_field(ax, mesh, nodal, title, levels=levels, cell_values=P)
    if own:
        save_figure(ax.figure, filename, show, dpi)
    return ax


def plot_flux_2d(mesh=None, Q=None, space=None, cell_flux_fn=None, title="Flux (RT0)",
                 filename=None, show=False, ax=None, dpi=200):
    """画 RT0 通量场（每个单元形心处的箭头）。

    Parameters
    ----------
    mesh : Mesh
        省略时从 ``space.mesh`` 取。
    Q : (n_cells, 3) ndarray
        单元局部外法向通量（``MixedPoissonProblem.solve()`` 的第一个返回值）。
    space : RT0Space, optional
        给了就用 ``space.evaluate_flux`` 在单元形心重构通量——**推荐用法**，
        这样画面上的箭头是真正的 :math:`q_h`，而不是简单的面通量平均。
    cell_flux_fn : callable, optional
        ``cell_flux_fn(cell_idx) -> (2,)``，自定义通量取值方式（优先级最高）。

    Examples
    --------
    .. code-block:: python

       V = FESpace.rt0(mesh)
       problem = MixedPoissonProblem(V, f=source)
       Q, P = problem.solve()
       plot_pressure_2d(mesh, P, filename="p.png")
       plot_flux_2d(space=V, Q=Q, filename="q.png")
    """
    plt = _require_pyplot()
    if plt is None:
        return None
    if mesh is None:
        if space is None:
            raise ValueError("plot_flux_2d 需要 mesh 或 space 之一")
        mesh = space.mesh
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6, 5.4))

    for cell in mesh.elements:
        v = np.asarray(mesh.nodes[cell], dtype=float)
        loop = np.vstack([v, v[:1]])
        ax.plot(loop[:, 0], loop[:, 1], "k-", linewidth=0.4, alpha=0.5)

    if cell_flux_fn is None:
        if space is None or Q is None:
            raise ValueError(
                "请给出 space=RT0 空间与 Q=(n_cells,3) 面通量（或用 cell_flux_fn 自定义）。"
            )
        Q = np.asarray(Q, dtype=float)

        def cell_flux_fn(c):
            ctr = mesh.get_cell_centroid(c)[None, :]
            return space.evaluate_flux(c, Q[c], ctr)[0]

    xs, ys, us, vs = [], [], [], []
    for c in range(mesh.n_cells):
        ctr = mesh.get_cell_centroid(c)
        vec = np.asarray(cell_flux_fn(c), dtype=float).ravel()
        xs.append(ctr[0])
        ys.append(ctr[1])
        us.append(vec[0])
        vs.append(vec[1])
    ax.quiver(xs, ys, us, vs, color="tab:blue")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    if own:
        save_figure(ax.figure, filename, show, dpi)
    return ax


# ---------------------------------------------------------------------------
def plot_convergence(h_values, errors, filename=None, show=False,
                     reference_rates=(1, 2, 3), dpi=200):
    """画 log-log 收敛曲线，并叠加 O(h^k) 参考斜率。"""
    plt = _require_pyplot()
    if plt is None:
        return None
    h = np.asarray(h_values, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    if "L2" in errors and np.any(np.isfinite(errors["L2"])):
        ax.loglog(h, errors["L2"], "o-", label="L2")
    if "H1" in errors and np.any(np.isfinite(errors["H1"])):
        ax.loglog(h, errors["H1"], "s-", label="H1")
    if "Linf" in errors and np.any(np.isfinite(errors["Linf"])):
        ax.loglog(h, errors["Linf"], "^--", label="Linf")

    anchor = None
    for key in ("L2", "H1"):
        vals = errors.get(key)
        if vals is not None and np.any(np.isfinite(vals)):
            vals = np.asarray(vals, dtype=float)
            if np.all(np.isfinite(vals)) and np.all(vals > 0):
                anchor = vals[0] / h[0] ** reference_rates[0]
                break
    if anchor is not None:
        hh = np.array([h.min(), h.max()])
        for k in reference_rates:
            ax.loglog(hh, anchor * hh ** k, ":", color="gray", linewidth=0.8)
            ax.annotate(f"$O(h^{k})$", (hh[-1], anchor * hh[-1] ** k),
                        textcoords="offset points", xytext=(4, -2), color="gray")

    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("mesh size $h$")
    ax.set_ylabel("error")
    ax.set_title("Convergence study")
    ax.legend()
    save_figure(fig, filename, show, dpi)
    return ax


# ---------------------------------------------------------------------------
def compare_solutions(entries, filename=None, show=False, dpi=200,
                      title="Comparison"):
    """并排比较多组解。

    ``entries`` 是 ``[(label, u, mesh, space), ...]`` 的列表。
    """
    plt = _require_pyplot()
    if plt is None:
        return None
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.0))
    if n == 1:
        axes = [axes]
    for (label, u, mesh, space), ax in zip(entries, axes):
        plot_solution(u, mesh, space, title=label, ax=ax)
    fig.suptitle(title)
    save_figure(fig, filename, show, dpi)
    return axes
