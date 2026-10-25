"""线性代数求解器。"""

# 模块职责：公共接口的集中导出
# 把本层常用类和函数导入统一命名空间；用户可从包入口导入接口，无需逐个寻找内部实现文件。
# __all__ 指定公开名称；计算逻辑保存在被导入的模块中，这个文件负责组织调用入口。


from .linear import ConvergenceError, Solver

__all__ = ["Solver", "ConvergenceError"]
