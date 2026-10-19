# -*- coding: utf-8 -*-
"""让 pytest 能直接 import MosicaFE（无需先安装）。"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _source_import import load_local
load_local()
