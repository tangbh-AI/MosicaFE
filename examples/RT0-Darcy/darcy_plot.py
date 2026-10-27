# -*- coding: utf-8 -*-
"""darcy_plot.py —— 不依赖 matplotlib 的误差可视化工具（Pillow 光栅）

为什么不直接用 matplotlib：本机 matplotlib 3.7 与 numpy 2.x ABI 不匹配，
``savefig`` 会抛 ``ValueError: object __array__ method not producing an array``。
本模块只用 Pillow + numpy，自己实现 colormap、坐标轴、对数收敛图与箭头图，
所以 RT0 / Darcy 的误差图在这台机器上一定能出图。

主要接口
--------
cell_heatmap(nodes, cells, values, path, ...)  逐单元标量（P0 场）热图：
                                               压力误差、通量误差、渗透率……
mesh_quiver(nodes, cells, points, vectors, ...) 单元中心通量箭头图（q_h）
loglog_convergence(hs, series, path, ...)       log-log 收敛曲线 + 参考斜率
hstack(paths, out, labels=...)                  把几张图拼成一行（对比用）

颜色约定：误差图用 ``coolwarm``（蓝=小、红=大），标量/渗透率场用 ``viridis``。

作者：赵典
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["cell_heatmap", "mesh_quiver", "loglog_convergence", "profile_plot",
           "hstack", "make_cmap", "draw_domain_frame"]

# ---------------------------------------------------------------------------
# 字体（中文优先，找不到就退回 Pillow 内置位图字体）
# ---------------------------------------------------------------------------
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",        # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",      # 黑体
    r"C:\Windows\Fonts\simsun.ttc",      # 宋体
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


@lru_cache(maxsize=None)
def font(size: int = 14, bold: bool = False):
    """返回一个可显示中文的字体对象。"""
    cands = ([r"C:\Windows\Fonts\msyhbd.ttc"] + _FONT_CANDIDATES) if bold \
        else _FONT_CANDIDATES
    for p in cands:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# colormap
# ---------------------------------------------------------------------------
_CMAPS = {
    "coolwarm": [(0.000, (59, 76, 192)), (0.125, (98, 130, 234)),
                 (0.250, (141, 176, 254)), (0.375, (184, 208, 249)),
                 (0.500, (221, 221, 221)), (0.625, (245, 196, 173)),
                 (0.750, (244, 154, 123)), (0.875, (222, 96, 77)),
                 (1.000, (180, 4, 38))],
    "viridis": [(0.000, (68, 1, 84)), (0.125, (72, 40, 120)),
                (0.250, (62, 74, 137)), (0.375, (49, 104, 142)),
                (0.500, (38, 130, 142)), (0.625, (31, 158, 137)),
                (0.750, (53, 183, 121)), (0.875, (109, 205, 89)),
                (1.000, (253, 231, 37))],
    "gray": [(0.0, (20, 20, 20)), (1.0, (245, 245, 245))],
}


def make_cmap(name: str = "coolwarm"):
    """返回 ``f(t) -> (n,3) uint8``，``t`` 为 [0,1] 数组（线性插值）。"""
    if name not in _CMAPS:
        raise ValueError(f"未知 colormap {name!r}；可用：{sorted(_CMAPS)}")
    stops = _CMAPS[name]
    ts = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], dtype=float)

    def cmap(t):
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        out = np.empty(t.shape + (3,), dtype=np.uint8)
        for k in range(3):
            out[..., k] = np.interp(t, ts, cs[:, k]).astype(np.uint8)
        return out
    return cmap


# ---------------------------------------------------------------------------
# 工具：坐标变换、颜色条、文字
# ---------------------------------------------------------------------------
class _Canvas:
    """把物理坐标映射到像素坐标的小工具（y 轴翻转）。"""

    def __init__(self, nodes, width=760, margin=0.05, pad_top=46, pad_bottom=14,
                 pad_left=14, pad_right=132):
        nodes = np.asarray(nodes, dtype=float)
        self.x0, self.y0 = nodes.min(axis=0)
        self.x1, self.y1 = nodes.max(axis=0)
        dx = max(self.x1 - self.x0, 1e-12)
        dy = max(self.y1 - self.y0, 1e-12)
        mx, my = margin * dx, margin * dy
        self.plot_w = int(width)
        self.plot_h = max(60, int(self.plot_w * (dy + 2 * my) / (dx + 2 * mx)))
        self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = \
            pad_top, pad_bottom, pad_left, pad_right
        self.W = self.plot_w + self.pad_left + self.pad_right
        self.H = self.plot_h + self.pad_top + self.pad_bottom
        self.x0p, self.x1p = self.x0 - mx, self.x1 + mx
        self.y0p, self.y1p = self.y0 - my, self.y1 + my

    def px(self, xy):
        """物理坐标 -> 像素坐标（浮点）。"""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        u = (xy[:, 0] - self.x0p) / (self.x1p - self.x0p)
        v = (xy[:, 1] - self.y0p) / (self.y1p - self.y0p)
        px = self.pad_left + u * self.plot_w
        py = self.pad_top + (1.0 - v) * self.plot_h       # y 翻转
        return np.column_stack([px, py])


def draw_domain_frame(draw, canvas, color=(90, 90, 90), width=1):
    """画区域外框。"""
    x0, y0 = canvas.px([[canvas.x0, canvas.y0]])[0]
    x1, y1 = canvas.px([[canvas.x1, canvas.y1]])[0]
    draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                   outline=color, width=width)


def _colorbar(img, cmap, vmin, vmax, canvas, label="", log=False):
    """在右侧画颜色条（顶部 = 最大值，含 min/mid/max 刻度文字）。"""
    draw = ImageDraw.Draw(img)
    bar_w, bar_h = 18, int(canvas.plot_h * 0.72)
    bx = canvas.W - canvas.pad_right + 26
    by = canvas.pad_top + (canvas.plot_h - bar_h) // 2
    t = np.linspace(1.0, 0.0, bar_h)[:, None]        # 顶端 t=1（最大值颜色）
    strip = np.repeat(cmap(t), bar_w, axis=1).astype(np.uint8)
    img.paste(Image.fromarray(strip, "RGB"), (bx, by))
    draw.rectangle([bx, by, bx + bar_w, by + bar_h], outline=(60, 60, 60))
    f = font(13)
    lo, mid, hi = _cbar_values(vmin, vmax, log)
    for frac, txt in zip((1.0, 0.5, 0.0), (hi, mid, lo)):     # 顶端 = 最大值
        y = by + (1 - frac) * bar_h
        draw.line([bx + bar_w, y, bx + bar_w + 5, y], fill=(60, 60, 60))
        draw.text((bx + bar_w + 9, y - 8), txt, fill=(20, 20, 20), font=f)
    if label:                                                  # 标签放在色条下方
        tw = draw.textlength(label, font=font(13))
        draw.text((bx + bar_w / 2 - tw / 2, by + bar_h + 8), label,
                  fill=(20, 20, 20), font=font(13))


def _cbar_values(vmin, vmax, log):
    def fmt(v):
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1e4 or a < 1e-3:
            e = int(math.floor(math.log10(a)))
            m = v / 10.0 ** e
            return f"{m:.1f}e{e}" if abs(m - 1) > 1e-9 else f"1e{e}"
        return f"{v:.3g}"
    if log:
        lo, hi = math.log10(vmin), math.log10(vmax)
        mid = 10 ** (0.5 * (lo + hi))
        return [fmt(vmin), fmt(mid), fmt(vmax)]
    return [fmt(vmin), fmt(0.5 * (vmin + vmax)), fmt(vmax)]


def _to_t(values, vmin, vmax, log):
    v = np.asarray(values, dtype=float)
    if log:
        floor = max(vmin, 1e-300)
        v = np.clip(v, floor, None)
        lo, hi = math.log10(floor), math.log10(vmax)
        t = (np.log10(v) - lo) / max(hi - lo, 1e-30)
    else:
        t = (v - vmin) / max(vmax - vmin, 1e-30)
    return np.clip(t, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ① 逐单元标量热图（P0 误差场 / 渗透率场）
# ---------------------------------------------------------------------------
def cell_heatmap(nodes, cells, values, path, *, title="", cbar_label="",
                 cmap="coolwarm", vmin=None, vmax=None, log=False,
                 width=760, show_mesh=None, margin=0.05):
    """把逐单元标量画成 PNG 热图（每个单元一个颜色，P0 数据的自然画法）。

    Parameters
    ----------
    nodes : (n_nodes, 2)；cells : (n_cells, 3) 顶点编号
    values : (n_cells,) 或 ``(n_cells, k)``（多列时只取第 0 列，或传 ``values=`` 的均值）
    log : bool  颜色按 log10 分布（标注仍用真实值），误差跨数量级时很有用
    show_mesh : bool or None  ``None`` = 单元数 ≤ 4000 时自动画网格线
    """
    nodes = np.asarray(nodes, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    values = np.asarray(values, dtype=float).ravel()
    if values.size != cells.shape[0]:
        raise ValueError(f"values 长度 {values.size} 与单元数 {cells.shape[0]} 不符")
    if show_mesh is None:
        show_mesh = cells.shape[0] <= 4000

    vmin = float(np.min(values)) if vmin is None else float(vmin)
    vmax = float(np.max(values)) if vmax is None else float(vmax)
    # 常值场（数据跨度在舍入级别）：统一取中间色，避免把 1e-11 的抖动放大成噪声
    _scale = max(abs(vmin), abs(vmax), 1e-30)
    uniform = (vmax - vmin) <= 1e-9 * _scale
    if uniform:
        vmin, vmax = vmin - 0.05 * _scale, vmax + 0.05 * _scale
    if vmax <= vmin:
        vmax = vmin + max(abs(vmin), 1.0) * 1e-12

    canvas = _Canvas(nodes, width=width, margin=margin)
    cmapf = make_cmap(cmap)
    t = np.full(values.shape, 0.5) if uniform else _to_t(values, vmin, vmax, log)
    rgb = cmapf(t)                                    # (n_cells, 3)

    img_arr = np.full((canvas.plot_h, canvas.plot_w, 3), 255, dtype=np.uint8)
    # 逐单元光栅化：单元凸且小，直接做像素包围盒 + 重心坐标内测
    for c in range(cells.shape[0]):
        tri = nodes[cells[c]]
        pix = canvas.px(tri) - np.array([canvas.pad_left, canvas.pad_top])
        xlo = max(int(np.floor(pix[:, 0].min())), 0)
        xhi = min(int(np.ceil(pix[:, 0].max())) + 1, canvas.plot_w)
        ylo = max(int(np.floor(pix[:, 1].min())), 0)
        yhi = min(int(np.ceil(pix[:, 1].max())) + 1, canvas.plot_h)
        if xhi <= xlo or yhi <= ylo:
            continue
        xs = np.arange(xlo, xhi) + 0.5
        ys = np.arange(ylo, yhi) + 0.5
        XX, YY = np.meshgrid(xs, ys)
        p0 = pix[0]
        e1 = pix[1] - p0
        e2 = pix[2] - p0
        det = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(det) < 1e-12:
            continue
        rx = XX - p0[0]
        ry = YY - p0[1]
        w1 = (rx * e2[1] - ry * e2[0]) / det
        w2 = (e1[0] * ry - e1[1] * rx) / det
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not np.any(inside):
            continue
        sub = img_arr[ylo:yhi, xlo:xhi]
        sub[inside] = rgb[c]

    img = Image.fromarray(img_arr, "RGB")

    if show_mesh:
        ov = Image.new("L", (canvas.plot_w, canvas.plot_h), 0)
        dov = ImageDraw.Draw(ov)
        edges = set()
        for tri in cells:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                key = (min(tri[a], tri[b]), max(tri[a], tri[b]))
                if key in edges:
                    continue
                edges.add(key)
                pa, pb = canvas.px([nodes[key[0]], nodes[key[1]]])
                dov.line([tuple(pa - [canvas.pad_left, canvas.pad_top]),
                          tuple(pb - [canvas.pad_left, canvas.pad_top])],
                         fill=70, width=1)
        black = Image.new("RGB", img.size, (0, 0, 0))
        img = Image.composite(black, img, ov)

    full = Image.new("RGB", (canvas.W, canvas.H), (255, 255, 255))
    full.paste(img, (canvas.pad_left, canvas.pad_top))
    draw = ImageDraw.Draw(full)
    draw_domain_frame(draw, canvas)
    if title:
        tw = draw.textlength(title, font=font(17, bold=True))
        draw.text(((canvas.W - tw) / 2, 14), title, fill=(10, 10, 10),
                  font=font(17, bold=True))
    _colorbar(full, cmapf, vmin, vmax, canvas, cbar_label, log)
    full.save(path)
    return path


# ---------------------------------------------------------------------------
# ② 通量箭头图
# ---------------------------------------------------------------------------
def _nice_number(v):
    """取一个不超过 ``v`` 的'整'数值（1/2/5×10^k），用于比例尺。"""
    if v <= 0:
        return 1.0
    e = math.floor(math.log10(v))
    for m in (5.0, 2.0, 1.0):
        if m * 10.0 ** e <= v * (1 + 1e-12):
            return m * 10.0 ** e
    return 10.0 ** e


def mesh_quiver(nodes, cells, points, vectors, path, *, title="", width=760,
                color=(20, 60, 160), scale=None, background=None, margin=0.05,
                ref_arrow=True, cbar_label="", vectors_extra=None,
                color_extra=(214, 39, 40), label=None, label_extra=None,
                stride=1):
    """在单元中心画 ``q_h`` 箭头（可选叠在热图 ``background`` 上）。

    ``vectors_extra``：再画一组箭头（例如精确解 ``q``）用于对比，颜色 ``color_extra``。
    ``scale``：物理单位 → 物理长度的放大系数；``None`` 时自动取"最长箭头 ≈ 半格"。
    ``ref_arrow``：左下角画一根参考箭头（长度 = 最长数据箭头）并标出它的数值。
    """
    nodes = np.asarray(nodes, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    canvas = _Canvas(nodes, width=width, margin=margin)

    if background is not None:
        base = np.asarray(background, dtype=np.uint8)
        if base.shape[:2] != (canvas.plot_h, canvas.plot_w):
            base = np.array(Image.fromarray(base).resize(
                (canvas.plot_w, canvas.plot_h), Image.BILINEAR))
        img = Image.fromarray(base, "RGB")
    else:
        img = Image.new("RGB", (canvas.plot_w, canvas.plot_h), (255, 255, 255))
    full = Image.new("RGB", (canvas.W, canvas.H), (255, 255, 255))
    full.paste(img, (canvas.pad_left, canvas.pad_top))
    draw = ImageDraw.Draw(full)

    def arrow(target, a, b, col, width=2):
        d = b - a
        L = float(np.hypot(*d))
        if L < 2.0:
            return False
        u = d / L
        n = np.array([-u[1], u[0]])
        head = min(7.0, 0.45 * L)
        base = b - head * u
        target.line([tuple(a), tuple(base)], fill=col, width=width)
        target.polygon([tuple(b), tuple(base + 0.55 * head * n),
                        tuple(base - 0.55 * head * n)], fill=col)
        return True

    mag = np.linalg.norm(vectors, axis=1)
    vmax = float(mag.max()) if mag.size else 1.0
    h = np.hypot(np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1])) / max(
        math.sqrt(cells.shape[0]), 1.0)
    if scale is None:
        scale = (0.85 * h) / max(vmax, 1e-300)
    px_per_unit = canvas.plot_w / (canvas.x1p - canvas.x0p)
    step = max(int(stride), 1)

    for p, v in zip(points[::step], vectors[::step]):
        arrow(draw, canvas.px([p])[0], canvas.px([p + scale * v])[0], color)
    if vectors_extra is not None:
        # 半透明覆盖：数值解（蓝）与精确解（红）重合处呈紫色，分离处一眼可见
        ov = Image.new("RGBA", full.size, (0, 0, 0, 0))
        dov = ImageDraw.Draw(ov)
        ve = np.atleast_2d(np.asarray(vectors_extra, float))[::step]
        for p, v in zip(points[::step], ve):
            arrow(dov, canvas.px([p])[0], canvas.px([p + scale * v])[0],
                  tuple(color_extra) + (135,), width=2)
        full = Image.alpha_composite(full.convert("RGBA"), ov).convert("RGB")
        draw = ImageDraw.Draw(full)

    draw_domain_frame(draw, canvas)
    # 两组箭头的图例
    ly = canvas.pad_top + 12
    for lab, col in ((label, color), (label_extra, color_extra)):
        if not lab:
            continue
        x0 = canvas.pad_left + 14
        draw.line([(x0, ly), (x0 + 30, ly)], fill=col, width=3)
        draw.polygon([(x0 + 30, ly), (x0 + 22, ly - 4), (x0 + 22, ly + 4)], fill=col)
        draw.text((x0 + 38, ly - 8), lab, fill=(30, 30, 30), font=font(13))
        ly += 20

    if ref_arrow:
        # 参考箭头：取一个"整"数值，长度严格按同一比例换算
        vref = _nice_number(vmax)
        dxpix = max(px_per_unit * scale * vref, 6.0)
        a = np.array([canvas.pad_left + 18.0, canvas.H - canvas.pad_bottom - 34.0])
        box = [a[0] - 10, a[1] - 30, a[0] + dxpix + 74, a[1] + 12]
        draw.rectangle(box, fill=(255, 255, 255), outline=(190, 190, 190))
        draw.line([tuple(a), (a[0] + dxpix, a[1])], fill=(110, 110, 110), width=2)
        draw.polygon([(a[0] + dxpix, a[1]), (a[0] + dxpix - 8, a[1] - 4),
                      (a[0] + dxpix - 8, a[1] + 4)], fill=(110, 110, 110))
        draw.text((a[0], a[1] - 26), f"|q| = {vref:.3g}（最大 {vmax:.3g}）",
                  fill=(90, 90, 90), font=font(12))
    if title:
        tw = draw.textlength(title, font=font(17, bold=True))
        draw.text(((canvas.W - tw) / 2, 14), title, fill=(10, 10, 10),
                  font=font(17, bold=True))
    if cbar_label:
        draw.text((canvas.W - canvas.pad_right + 10, canvas.pad_top),
                  cbar_label, fill=(20, 20, 20), font=font(13))
    full.save(path)
    return path


# ---------------------------------------------------------------------------
# ③ log-log 收敛曲线
# ---------------------------------------------------------------------------
def loglog_convergence(hs, series, path, *, title="", xlabel="网格尺寸 h",
                       ylabel="误差", ref_slope=1.0, width=820, height=560):
    """log-log 收敛曲线。

    ``series`` 每项可以是

    * ``(label, 误差数组, 颜色)``  —— 共用 ``hs``；
    * ``(label, h数组, 误差数组, 颜色)`` —— 各自一套 ``h``（2D/3D 混画时用）；
    * ``dict(label=..., e=..., h=..., color=..., dash=True)``。

    颜色为 ``None`` 时按调色板自动分配；``dash=True`` 画虚线。
    """
    norm = []
    for it in series:
        if isinstance(it, dict):
            d = dict(it)
        elif len(it) == 4:
            d = dict(label=it[0], h=it[1], e=it[2], color=it[3])
        else:
            d = dict(label=it[0], h=hs, e=it[1], color=it[2])
        d["h"] = np.asarray(hs if d.get("h") is None else d["h"], dtype=float)
        d["e"] = np.asarray(d["e"], dtype=float)
        norm.append(d)
    series = norm
    palette = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (255, 127, 14),
               (148, 103, 189), (140, 86, 75), (23, 190, 207)]
    pad_l, pad_r, pad_t, pad_b = 96, 24, 52, 62
    W, H = width, height
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b

    allh = np.concatenate([d["h"] for d in series])
    alle = np.concatenate([d["e"] for d in series])
    allh, alle = allh[allh > 0], alle[alle > 0]
    lx0 = math.floor(math.log10(allh.min()) - 0.05)
    lx1 = math.ceil(math.log10(allh.max()) + 0.05)
    ly0 = math.floor(math.log10(alle.min()) - 0.05)
    ly1 = math.ceil(math.log10(alle.max()) + 0.05)

    def X(v):
        return pad_l + (math.log10(v) - lx0) / (lx1 - lx0) * pw

    def Y(v):
        return pad_t + (ly1 - math.log10(v)) / (ly1 - ly0) * ph

    def dashed(pts, col, width=3, on=9, off=6):
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 1e-9:
                continue
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            s = 0.0
            while s < L:
                e = min(s + on, L)
                draw.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)],
                          fill=col, width=width)
                s = e + off

    for k in range(lx0, lx1 + 1):
        x = X(10.0 ** k)
        draw.line([(x, pad_t), (x, pad_t + ph)], fill=(232, 232, 232))
        draw.text((x - 14, pad_t + ph + 8), f"1e{k}", fill=(60, 60, 60), font=font(12))
    for k in range(ly0, ly1 + 1):
        y = Y(10.0 ** k)
        draw.line([(pad_l, y), (pad_l + pw, y)], fill=(232, 232, 232))
        draw.text((10, y - 8), f"1e{k}", fill=(60, 60, 60), font=font(12))
    draw.rectangle([pad_l, pad_t, pad_l + pw, pad_t + ph], outline=(60, 60, 60))

    # 参考斜率（画在数据上方，避免与曲线重叠）
    xr = np.array([allh.min() * 1.02, allh.max() * 0.98])
    e0 = alle.max() * 1.12
    yr = e0 * (xr / xr[1]) ** ref_slope
    dashed([(X(xr[0]), Y(yr[0])), (X(xr[1]), Y(yr[1]))], (140, 140, 140), 2, 7, 5)
    draw.text((X(xr[0]) + 6, Y(yr[0]) - 20), f"斜率 {ref_slope:g}", fill=(120, 120, 120),
              font=font(13))

    for k, d in enumerate(series):
        col = d.get("color") or palette[k % len(palette)]
        pts = [(X(h), Y(e)) for h, e in zip(d["h"], d["e"]) if e > 0]
        if not pts:
            continue
        if d.get("dash"):
            dashed(pts, col)
        else:
            draw.line(pts, fill=col, width=3)
        for (x, y) in pts:
            r = 4
            draw.ellipse([x - r, y - r, x + r, y + r], fill=col,
                         outline=(255, 255, 255))

    # 图例
    lh = 21
    wid = 12
    for d in series:
        wid = max(wid, draw.textlength(d["label"], font=font(13)))
    box_w = int(wid) + 78
    box_h = lh * len(series) + 12
    draw.rectangle([pad_l + 12, pad_t + 12, pad_l + 12 + box_w, pad_t + 12 + box_h],
                   fill=(255, 255, 255), outline=(200, 200, 200))
    for k, d in enumerate(series):
        col = d.get("color") or palette[k % len(palette)]
        y = pad_t + 24 + k * lh
        if d.get("dash"):
            dashed([(pad_l + 22, y), (pad_l + 60, y)], col, 3, 8, 5)
        else:
            draw.line([(pad_l + 22, y), (pad_l + 60, y)], fill=col, width=3)
        draw.ellipse([pad_l + 37, y - 4, pad_l + 45, y + 4], fill=col,
                     outline=(255, 255, 255))
        draw.text((pad_l + 68, y - 8), d["label"], fill=(30, 30, 30), font=font(13))
    if title:
        tw = draw.textlength(title, font=font(17, bold=True))
        draw.text(((W - tw) / 2, 14), title, fill=(10, 10, 10), font=font(17, bold=True))
    draw.text((W / 2 - 30, H - 26), xlabel, fill=(30, 30, 30), font=font(13))
    draw.text((12, pad_t - 32), ylabel, fill=(30, 30, 30), font=font(13))
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# ④ 一维剖面折线图（沿一条线比较精确解与数值解）
# ---------------------------------------------------------------------------
def p0_step_along_line(nodes, cells, P, y_target, h):
    """取贴着 ``y = y_target`` **上方**那一排三角形，拼出 P0 的阶梯折线。

    返回 ``(x, p)``：相邻单元的 x 区间首尾相接，折线在接缝处自动竖直跳变，
    正好画出分片常数解的"台阶"。
    """
    nodes = np.asarray(nodes, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    P = np.asarray(P, dtype=float).ravel()
    sizes = nodes[cells].max(axis=1) - nodes[cells].min(axis=1)
    cy = nodes[cells][:, :, 1].mean(axis=1)
    x0s = nodes[cells][:, :, 0].min(axis=1)
    x1s = nodes[cells][:, :, 0].max(axis=1)
    yrow = y_target + h / 3.0                      # 该排三角形形心的 y
    segs = sorted((float(x0s[c]), float(x1s[c]), float(P[c]))
                  for c in range(len(cells)) if abs(cy[c] - yrow) < 1e-9)
    del sizes
    x, y = [], []
    for a, b, v in segs:
        x += [a, b]
        y += [v, v]
    return np.array(x), np.array(y)


def _nice_ticks(lo, hi, n=5):
    """给 [lo, hi] 选 n 个"整"刻度。"""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(n, 1)
    e = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = m * e
        if step >= raw:
            break
    t0 = math.ceil(lo / step) * step
    out = []
    v = t0
    while v <= hi + 1e-12 and len(out) < 40:
        out.append(round(v, 12))
        v += step
    return out


def profile_plot(series, path, *, title="", xlabel="x", ylabel="p", width=800,
                 height=470, vlines=(), ylim=None, legend_pos="tl"):
    """一维剖面折线图（线性坐标）。

    ``series`` 每项为 ``dict(label=…, x=…, y=…, color=…, dash=False)``
    或 ``(label, x, y, color)``；``vlines`` 是竖直参考线的 x 位置（如材料界面）；
    ``legend_pos``：``"tl" / "tr" / "bl" / "br"``。
    """
    norm = []
    for it in series:
        d = dict(it) if isinstance(it, dict) else dict(label=it[0], x=it[1],
                                                       y=it[2], color=it[3])
        d["x"] = np.asarray(d["x"], dtype=float)
        d["y"] = np.asarray(d["y"], dtype=float)
        norm.append(d)
    series = norm
    palette = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (255, 127, 14),
               (148, 103, 189)]
    pad_l, pad_r, pad_t, pad_b = 92, 26, 52, 60
    W, H = width, height
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b

    xs = np.concatenate([d["x"] for d in series])
    ys = np.concatenate([d["y"] for d in series])
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = (float(ys.min()), float(ys.max())) if ylim is None else ylim
    if y1 == y0:
        y1 = y0 + 1.0

    def X(v):
        return pad_l + (v - x0) / (x1 - x0) * pw

    def Y(v):
        return pad_t + (y1 - v) / (y1 - y0) * ph

    for t in _nice_ticks(x0, x1, 6):
        x = X(t)
        draw.line([(x, pad_t), (x, pad_t + ph)], fill=(236, 236, 236))
        draw.text((x - 12, pad_t + ph + 8), f"{t:g}", fill=(60, 60, 60), font=font(12))
    for t in _nice_ticks(y0, y1, 5):
        y = Y(t)
        draw.line([(pad_l, y), (pad_l + pw, y)], fill=(236, 236, 236))
        draw.text((14, y - 8), f"{t:g}", fill=(60, 60, 60), font=font(12))
    for xv in vlines:
        draw.line([(X(xv), pad_t), (X(xv), pad_t + ph)], fill=(150, 150, 150))
    draw.rectangle([pad_l, pad_t, pad_l + pw, pad_t + ph], outline=(60, 60, 60))

    def dashed(pts, col, width=3, on=9, off=6):
        for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
            L = math.hypot(xb - xa, yb - ya)
            if L < 1e-9:
                continue
            ux, uy = (xb - xa) / L, (yb - ya) / L
            s = 0.0
            while s < L:
                e = min(s + on, L)
                draw.line([(xa + ux * s, ya + uy * s), (xa + ux * e, ya + uy * e)],
                          fill=col, width=width)
                s = e + off

    for k, d in enumerate(series):
        col = d.get("color") or palette[k % len(palette)]
        pts = [(X(a), Y(b)) for a, b in zip(d["x"], d["y"])]
        if len(pts) == 1:
            continue
        if d.get("dash"):
            dashed(pts, col, 2)
        else:
            draw.line(pts, fill=col, width=3 if not d.get("thin") else 1)
        if d.get("marker"):
            for (x, y) in pts:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)

    wid = max([draw.textlength(d["label"], font=font(13)) for d in series] + [60])
    box_h = 21 * len(series) + 12
    box_w = int(wid) + 80
    lx = pad_l + 12 if legend_pos in ("tl", "bl") else pad_l + pw - box_w - 12
    ly = pad_t + 12 if legend_pos in ("tl", "tr") else pad_t + ph - box_h - 12
    draw.rectangle([lx, ly, lx + box_w, ly + box_h],
                   fill=(255, 255, 255), outline=(200, 200, 200))
    for k, d in enumerate(series):
        col = d.get("color") or palette[k % len(palette)]
        y = ly + 12 + k * 21
        draw.line([(lx + 10, y), (lx + 46, y)], fill=col, width=3)
        draw.text((lx + 54, y - 8), d["label"], fill=(30, 30, 30), font=font(13))
    if title:
        tw = draw.textlength(title, font=font(17, bold=True))
        draw.text(((W - tw) / 2, 14), title, fill=(10, 10, 10), font=font(17, bold=True))
    draw.text((W / 2 - 20, H - 26), xlabel, fill=(30, 30, 30), font=font(13))
    draw.text((16, pad_t - 32), ylabel, fill=(30, 30, 30), font=font(13))
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# ⑤ 拼图
# ---------------------------------------------------------------------------
def hstack(paths, out, labels=None, gap=14, bg=(255, 255, 255)):
    """把若干 PNG 横向拼成一行（上方可加子图标签）。"""
    ims = [Image.open(p).convert("RGB") for p in paths]
    H = max(im.height for im in ims) + (24 if labels else 0)
    W = sum(im.width for im in ims) + gap * (len(ims) - 1)
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)
    x = 0
    for k, im in enumerate(ims):
        y = H - im.height
        canvas.paste(im, (x, y))
        if labels and k < len(labels):
            draw.text((x + 8, 4), labels[k], fill=(20, 20, 20), font=font(14, bold=True))
        x += im.width + gap
    canvas.save(out)
    return out
