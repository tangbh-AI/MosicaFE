"""离散空间：Lagrange 有限元、虚拟元 (VEM)、RT0×P0 混合元。"""

from .base import BaseFESpace
from .factory import FESpace
from .lagrange import LagrangeSpace
from .rt0 import RT0Space
from .vem import VEMSpace

__all__ = [
    "BaseFESpace",
    "FESpace",
    "LagrangeSpace",
    "VEMSpace",
    "RT0Space",
]
