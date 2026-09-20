"""Locate resources in development and inside a frozen macOS application."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

__all__ = [
    "apply_runtime_environment",
    "bundled_binary",
    "is_frozen",
    "resource_root",
]


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    return Path(frozen_root) if frozen_root else Path(__file__).resolve().parents[1]


def bundled_binary(name: str) -> str:
    """Return a packaged executable, a PATH executable, or its bare name."""
    candidate = resource_root() / "bin" / name
    if candidate.is_file():
        return str(candidate)
    return shutil.which(name) or name


def apply_runtime_environment() -> None:
    """Expose bundled command-line helpers to libraries that invoke by name."""
    binary_folder = resource_root() / "bin"
    if not binary_folder.is_dir():
        return
    existing = os.environ.get("PATH", "")
    folders = existing.split(os.pathsep) if existing else []
    if str(binary_folder) not in folders:
        os.environ["PATH"] = os.pathsep.join([str(binary_folder), *folders])
