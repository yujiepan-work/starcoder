import importlib
from pathlib import Path
from bios_llm.utils.registry import Registry


def _import_modules(file_path):
    # automatically scan and import modules for registry
    folder = Path(file_path).parent.expanduser().resolve()
    modules = []
    for file in folder.glob('*.py'):
        if file.name.startswith('_'):
            continue
        modules.append(importlib.import_module(f'bios_llm.datasets.{file.stem}'))
    return modules


_import_modules(__file__)
DATASET = Registry('dataset')
