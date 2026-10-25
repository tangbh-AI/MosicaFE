# -*- coding: utf-8 -*-
"""让 pytest 能直接 import MosicaFE（无需先安装）。"""

# 模块职责：测试导入路径设置
# 把项目父目录加入模块搜索路径，使测试可以直接导入 MosicaFE。
# 这里只准备测试环境，不创建网格，也不执行数值求解。


import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
