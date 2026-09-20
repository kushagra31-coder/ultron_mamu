"""Automatically discover tool modules under the tools package.

A tool module is optional: if one dependency is missing, that module is
reported in LOAD_ERRORS while the rest of the tools remain available.
"""
from __future__ import annotations

import importlib
import pkgutil
import tools

LOAD_ERRORS: dict[str, str] = {}


def load_tools() -> None:
    """Import every tool module exactly once and record optional failures."""
    for module_info in sorted(pkgutil.iter_modules(tools.__path__), key=lambda x: x.name):
        name = module_info.name
        if name.startswith("_") or name in {"registry", "loader"}:
            continue

        try:
            importlib.import_module(f"tools.{name}")
        except Exception as exc:  # keep other tools usable
            LOAD_ERRORS[name] = f"{type(exc).__name__}: {exc}"


def loaded_tool_modules() -> list[str]:
    return sorted(
        name for name in pkgutil.iter_modules(tools.__path__)
        if not name.name.startswith("_") and name.name not in {"registry", "loader"}
    )
