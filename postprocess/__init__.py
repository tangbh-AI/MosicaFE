"""后处理：误差分析、收敛性研究、可视化。"""

# 模块职责：公共接口的集中导出
# 把本层常用类和函数导入统一命名空间；用户可从包入口导入接口，无需逐个寻找内部实现文件。
# __all__ 指定公开名称；计算逻辑保存在被导入的模块中，这个文件负责组织调用入口。


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
