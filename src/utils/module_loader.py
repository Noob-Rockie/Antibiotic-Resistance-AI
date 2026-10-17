"""Load modules that live inside the numbered workflow folders.

The stage folders (`08_RECOMMENDATION_ENGINE`, `10_API_SERVICES`, ...) are named
for human readability, and a leading digit makes them invalid Python package
names. This helper loads a module from such a folder by file path so the rest of
the codebase can still import the logic normally:

    from src.utils.module_loader import load_stage_module
    drug_api = load_stage_module("10_API_SERVICES", "drug_api")
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_CACHE: dict[str, ModuleType] = {}


def load_stage_module(folder: str, module_name: str) -> ModuleType:
    """Import `<PROJECT_ROOT>/<folder>/<module_name>.py` and cache it."""
    cache_key = f"{folder}/{module_name}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    path = PROJECT_ROOT / folder / f"{module_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"No module at {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _CACHE[cache_key] = module
    return module
