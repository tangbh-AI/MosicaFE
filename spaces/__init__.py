"""离散空间：Lagrange 有限元、虚拟元 (VEM)、RT0×P0 混合元。"""

# 模块职责：公共接口的集中导出
# 把本层常用类和函数导入统一命名空间；用户可从包入口导入接口，无需逐个寻找内部实现文件。
# __all__ 指定公开名称；计算逻辑保存在被导入的模块中，这个文件负责组织调用入口。


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
