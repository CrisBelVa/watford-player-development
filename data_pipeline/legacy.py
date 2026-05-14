from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from .config import ROOT_DIR


LEGACY_DIR = ROOT_DIR / "prueba lucas"


def _load_module(module_name: str, file_name: str) -> ModuleType:
    module_path = LEGACY_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def whoscored_scraper() -> ModuleType:
    return _load_module("legacy_whoscored_scraper", "whoscored_scraping_fun.py")


@lru_cache(maxsize=1)
def whoscored_parser() -> ModuleType:
    return _load_module("legacy_whoscored_parser", "whoscored_json2csv_fun.py")
