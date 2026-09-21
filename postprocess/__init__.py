"""后处理：误差分析、收敛性研究、可视化。"""

from .convergence import convergence_study, estimate_convergence_rate, rate_table
from .error import (
    compute_error,
    compute_errors,
    h1_error,
    l2_error,
    linf_error,
    mixed_error_norms,
    rate_table as error_rate_table,
)
from .visualization import (
    compare_solutions,
    plot_convergence,
    plot_error,
    plot_flux_2d,
    plot_mesh,
    plot_pressure_2d,
    plot_solution,
)

__all__ = [
    "compute_error",
    "compute_errors",
    "l2_error",
    "h1_error",
    "linf_error",
    "mixed_error_norms",
    "convergence_study",
    "estimate_convergence_rate",
    "rate_table",
    "plot_solution",
    "plot_mesh",
    "plot_error",
    "plot_convergence",
    "plot_pressure_2d",
    "plot_flux_2d",
    "compare_solutions",
]
