"""有限元空间的工厂入口 :class:`FESpace`。

用法与 guidebook 第 5.2.3 节一致，只是把"元"扩成了三种族：

.. code-block:: python

   from MosicaFE import Mesh, FESpace

   V1 = FESpace.lagrange(mesh, degree=1)   # 经典 FEM: P1/Q1
   V2 = FESpace.lagrange(mesh, degree=2)   # 经典 FEM: P2/Q2
   Vv = FESpace.vem(mesh, degree=1)        # 虚拟元（任意多边形/多面体）
   Vr = FESpace.rt0(mesh)                  # RT0×P0 混合元
"""

from __future__ import annotations

from .lagrange import LagrangeSpace
from .rt0 import RT0Space
from .vem import VEMSpace

__all__ = ["FESpace"]


class FESpace:
    """离散空间工厂。"""

    #: 已注册的空间族
    FAMILIES = {
        "lagrange": LagrangeSpace,
        "fem": LagrangeSpace,
        "p1": LagrangeSpace,
        "p2": LagrangeSpace,
        "vem": VEMSpace,
        "virtual": VEMSpace,
        "rt0": RT0Space,
        "mixed": RT0Space,
    }

    # ------------------------------------------------------------------
    @staticmethod
    def lagrange(mesh, degree: int = 1, **kwargs) -> LagrangeSpace:
        """连续 Lagrange 空间（经典有限元）。"""
        return LagrangeSpace(mesh, degree=degree, **kwargs)

    @staticmethod
    def vem(mesh, degree: int = 1, **kwargs) -> VEMSpace:
        """虚拟元空间（k=1，任意多边形 / 多面体）。"""
        return VEMSpace(mesh, degree=degree, **kwargs)

    @staticmethod
    def rt0(mesh, degree: int = 1, **kwargs) -> RT0Space:
        """RT0×P0 混合空间（2D 三角形 / 3D 四面体）。"""
        return RT0Space(mesh, degree=degree, **kwargs)

    # ------------------------------------------------------------------
    @staticmethod
    def discontinuous_lagrange(mesh, degree: int = 0, **kwargs):
        """DG 空间（**尚未实现**）。

        这是 guidebook 第 12.1 节给出的扩展范例。要接入 DG，只需
        新建 ``spaces/dg.py``，实现 ``_build_dof_map / eval_basis /
        eval_basis_grad / stiffness_matrix / mass_matrix / load_vector``，
        再在这里注册即可（组装循环与 Lagrange 完全兼容）。
        """
        raise NotImplementedError(
            "DG 空间还未实现。可用的空间族：FESpace.lagrange / FESpace.vem / FESpace.rt0。"
        )

    # ------------------------------------------------------------------
    @classmethod
    def create(cls, mesh, family: str = "lagrange", degree: int = 1, **kwargs):
        """按名字创建空间（``solve_poisson(element=...)`` 内部使用）。"""
        key = str(family).lower()
        if key not in cls.FAMILIES:
            raise ValueError(
                f"未知空间族 {family!r}；可用：{sorted(set(cls.FAMILIES))}"
            )
        ctor = cls.FAMILIES[key]
        if ctor is LagrangeSpace and key in ("p1", "p2"):
            degree = int(key[1])
        return ctor(mesh, degree=degree, **kwargs)

    # guidebook 里出现过 space.get_quadrature() 的写法
    @staticmethod
    def available() -> list:
        """返回可用空间族名字列表。"""
        return sorted(set(FESpace.FAMILIES))
