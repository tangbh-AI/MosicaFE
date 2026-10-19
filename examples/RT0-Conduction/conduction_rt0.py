# -*- coding: utf-8 -*-
"""conduction_rt0.py —— 定常电流场的 RT0×P0 混合元求解与验证

库内实现：``MosicaFE/physics/conduction.py`` 的 ``ConductionProblem``
（注册名 ``conduction``，一行式接口 ``MosicaFE.solve_conduction``）。

方程::

    J = -sigma grad(phi)        （欧姆定律）
    div J = f                   （定常电荷守恒）

边界：``"D"`` 电极电势 / ``"N"`` 绝缘（J·n=0）/ ``("flux", gN)`` 注入电流密度；
内部面可加**接触电阻**：[[phi]] = R_c (J·n)。

验证四项：
  (1) 均质条形电阻器（R = L/(σA)）；
  (2) 分层串联（R = L1/σ1 + L2/σ2，与 Darcy 的调和平均一致）；
  (3) 界面接触电阻（R = R_bulk + R_c/|Γ_c|，电势跳变 = R_c·I）；
  (4) 能量关系 P = ΔV·I 与逐单元电荷守恒；
另附与 DarcyProblem 的等价性对拍（σ ↔ K）。

运行： python conduction_rt0.py
输出： 结果_电流场.txt；图 figs/*.png（若..\RT0-Darcy\darcy_plot.py 可用）
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "..", "MosicaFE")
spec = importlib.util.spec_from_file_location(
    "MosicaFE", os.path.join(PKG, "__init__.py"), submodule_search_locations=[PKG])
_mod = importlib.util.module_from_spec(spec)
sys.modules["MosicaFE"] = _mod
spec.loader.exec_module(_mod)

import MosicaFE as mf                                        # noqa: E402
from MosicaFE.physics.conduction import ConductionProblem    # noqa: E402

# 画图模块在隔壁 RT0-Darcy 目录（纯 Pillow，不依赖 matplotlib）
sys.path.insert(0, os.path.join(HERE, "..", "RT0-Darcy"))
try:
    import darcy_plot as dp
except Exception:                                            # pragma: no cover
    dp = None

FIGDIR = os.path.join(HERE, "figs")
out = []


def say(s=""):
    print(s)
    out.append(str(s))


def bc_column(mid, n):
    """左右电极（定压），上下绝缘。"""
    return "D" if (mid[0] < 1e-12 or mid[0] > 1.0 - 1e-12) else "N"


def right_electrode(mid):
    return mid[0] > 1.0 - 1e-12


def run(mesh, **kw):
    V = mf.FESpace.rt0(mesh)
    prob = ConductionProblem(V, **kw)
    J, phi = prob.solve()
    return prob, J, phi


def main():
    say("=" * 92)
    say("定常电流场（DC conduction）的 RT0×P0 混合元验证")
    say("    J = -σ∇φ,  ∇·J = f;  界面接触电阻 [[φ]] = R_c (J·n)")
    say("=" * 92)

    # ---------- (1) 均质条形电阻器 ----------
    say("")
    say("【(1) 均质条形电阻器】左右电极 1 V / 0 V，上下绝缘，σ=2，面积/长度=1")
    mesh = mf.Mesh.rectangle(nx=16, ny=16)
    prob, J, phi = run(mesh, g=lambda x: 1.0 - x[:, 0], sigma=2.0, bc=bc_column)
    I = prob.total_current(right_electrode)
    R = prob.resistance(1.0, right_electrode)
    P = prob.joule_heating()
    say(f"   总电流 I = {I:.10f}（理论 σ·A·Δφ/L = 2）")
    say(f"   电阻   R = {R:.10f}（理论 0.5）")
    say(f"   焦耳热 P = {P:.10f}（理论 Δφ·I = 2）")
    say(f"   逐单元电荷守恒偏差 = {prob.check_charge_conservation():.2e}")
    say(f"   逐单元面电流（局部自由度，前 3 个）= "
        f"{np.round(np.unique(np.round(J, 8), axis=0)[:3], 6).tolist()}")

    # ---------- (2) 分层串联 ----------
    say("")
    say("【(2) 分层串联导体】σ=1（x<0.5）、σ=100（x>0.5）：R = 0.5/1 + 0.5/100")
    mesh = mf.Mesh.rectangle(nx=16, ny=16)
    sig = lambda x: np.where(x[:, 0] < 0.5, 1.0, 100.0)
    prob, J, phi = run(mesh, g=lambda x: 1.0 - x[:, 0], sigma=sig, bc=bc_column)
    I = prob.total_current(right_electrode)
    R = prob.resistance(1.0, right_electrode)
    R_ex = 0.5 / 1.0 + 0.5 / 100.0
    say(f"   I = {I:.10f}，R = {R:.10f}（理论 {R_ex:.10f}，相对误差 "
        f"{abs(R - R_ex) / R_ex:.2e}）")
    say("   （与 Darcy 的串联调和平均是同一个公式：R = Σ L_i/σ_i）")

    # ---------- (3) 界面接触电阻 ----------
    say("")
    say("【(3) 界面接触电阻】均质 σ=1，在 x=0.5 上加 R_c=0.5（单位厚度）")
    mesh = mf.Mesh.rectangle(nx=16, ny=16)
    prob, J, phi = run(mesh, g=lambda x: 1.0 - x[:, 0], sigma=1.0, bc=bc_column,
                       contact=lambda mid: 0.5 if abs(mid[0] - 0.5) < 1e-12 else None)
    I = prob.total_current(right_electrode)
    R = prob.resistance(1.0, right_electrode)
    say(f"   接触面数 = {len(prob.contact_facets)}；I = {I:.10f}，R = {R:.10f}"
        f"（理论 1 + 0.5 = 1.5）")
    # 界面两侧单元的电势跳变
    nodes = np.asarray(mesh.nodes, dtype=float)
    ctr = np.array([mesh.get_cell_centroid(c) for c in range(mesh.n_cells)])
    left = np.abs(ctr[:, 0] - (0.5 - 1.0 / 32)).argmin()
    right = np.abs(ctr[:, 0] - (0.5 + 1.0 / 32)).argmin()
    h = 1.0 / 16.0                              # 相邻单元形心到界面的距离 h/2
    bulk = I * (0.5 * h + 0.5 * h) / 1.0        # 两形心之间的体电阻压降（σ=1）
    jump = (phi[left] - phi[right]) - bulk
    say(f"   界面电势跳变（扣掉形心间体压降）= {jump:.10f}"
        f"（理论 R_c·I = {0.5 * I:.10f}）")
    say(f"   焦耳热 P = {prob.joule_heating():.10f}（理论 Δφ·I = {I:.10f}）")

    # ---------- (4) 与 Darcy 的等价性 ----------
    say("")
    say("【(4) 与 DarcyProblem 对拍（σ ↔ K，φ ↔ p）】")
    mesh = mf.Mesh.rectangle(nx=8, ny=8)
    V = mf.FESpace.rt0(mesh)
    Kmat = np.array([[3.0, 1.0], [1.0, 2.0]])
    cond = ConductionProblem(V, g=lambda x: 1.0 - x[:, 0], sigma=Kmat, bc=bc_column)
    Jc, phic = cond.solve()
    V2 = mf.FESpace.rt0(mf.Mesh.rectangle(nx=8, ny=8))
    darcy = mf.DarcyProblem(V2, g=lambda x: 1.0 - x[:, 0], K=Kmat, bc=bc_column)
    Qd, pd = darcy.solve()
    say(f"   max|ΔJ| = {np.abs(Jc - Qd).max():.3e}， max|Δφ| = {np.abs(phic - pd).max():.3e}"
        "（应为机器零）")

    # ---------- 图 ----------
    if dp is not None:
        os.makedirs(FIGDIR, exist_ok=True)
        mesh = mf.Mesh.rectangle(nx=32, ny=32)
        prob, J, phi = run(mesh, g=lambda x: 1.0 - x[:, 0], sigma=1.0, bc=bc_column)
        nodes = np.asarray(mesh.nodes, dtype=float)
        cells = np.asarray(mesh.elements, dtype=np.int64)
        ctr = np.array([mesh.get_cell_centroid(c) for c in range(mesh.n_cells)])
        V = prob.space
        qh = np.array([V.evaluate_flux(c, J[c], ctr[c][None, :])[0]
                       for c in range(mesh.n_cells)])
        f1 = dp.cell_heatmap(nodes, cells, phi, os.path.join(FIGDIR, "potential.png"),
                             title="电势 φ_h（P0，左右电极 1 V/0 V）",
                             cbar_label="V", cmap="coolwarm", show_mesh=False, width=520)
        f2 = dp.cell_heatmap(nodes, cells, np.sqrt((qh ** 2).sum(axis=1)),
                             os.path.join(FIGDIR, "current_mag.png"),
                             title="电流密度 |J_h|", cbar_label="A/m²",
                             cmap="viridis", show_mesh=False, width=520)
        f3 = dp.mesh_quiver(nodes, cells, ctr, qh, os.path.join(FIGDIR, "current.png"),
                            title="电流密度 J_h（均匀水平）", width=520)
        dp.hstack([f1, f2, f3], os.path.join(FIGDIR, "conduction.png"),
                  labels=["(a) 电势", "(b) 电流密度模", "(c) 电流方向"])
        say("")
        say(f"   图已保存：figs/{os.listdir(FIGDIR)}")

    with open(os.path.join(HERE, "结果_电流场.txt"), "w", encoding="utf-8-sig") as fp:
        fp.write("\n".join(out) + "\n")
    print("\n[结果] 已保存 结果_电流场.txt")


if __name__ == "__main__":
    main()
