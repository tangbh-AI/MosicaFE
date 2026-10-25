"""核心模块：网格、面拓扑、求积规则与几何工具。"""

# 模块职责：公共接口的集中导出
# 把本层常用类和函数导入统一命名空间；用户可从包入口导入接口，无需逐个寻找内部实现文件。
# __all__ 指定公开名称；计算逻辑保存在被导入的模块中，这个文件负责组织调用入口。


from .facets import FacetTopology, build_facet_topology, local_facets
from .mesh import Mesh
from .quadrature import (
    QuadratureRule,
    cell_quadrature,
    get_quadrature,
    reference_dimension,
    simplex_quadrature,
)
from .utils import (
    apply_affine_map,
    cell_diameter,
    cell_measure,
    compute_jacobian,
    polygon_area,
    polyhedron_volume,
    simplex_jacobian,
    tensor_jacobian,
    transform_gradient,
)

__all__ = [
    "Mesh",
    "FacetTopology",
    "build_facet_topology",
    "local_facets",
    "QuadratureRule",
    "get_quadrature",
    "cell_quadrature",
    "simplex_quadrature",
    "reference_dimension",
    "apply_affine_map",
    "cell_diameter",
    "cell_measure",
    "compute_jacobian",
    "polygon_area",
    "polyhedron_volume",
    "simplex_jacobian",
    "tensor_jacobian",
    "transform_gradient",
]
