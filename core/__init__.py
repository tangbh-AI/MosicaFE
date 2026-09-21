"""核心模块：网格、面拓扑、求积规则与几何工具。"""

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
