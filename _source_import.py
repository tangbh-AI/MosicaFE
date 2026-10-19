"""Load this checkout as MosicaFE even when its folder is MosicaFE-master."""
import importlib.util
from pathlib import Path
import sys


def load_local():
    root = Path(__file__).resolve().parent
    existing = sys.modules.get('MosicaFE')
    if existing is not None:
        if Path(existing.__file__).resolve() != root / '__init__.py':
            raise RuntimeError('Another MosicaFE checkout is already imported')
        return existing
    spec = importlib.util.spec_from_file_location(
        'MosicaFE', root / '__init__.py', submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['MosicaFE'] = module
    spec.loader.exec_module(module)
    return module
