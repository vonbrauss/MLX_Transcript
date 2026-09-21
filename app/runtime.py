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
    "missing_component_message",
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


def missing_component_message(package: str) -> str:
    """Explain a missing component in terms the reader can act on.

    Someone running from a checkout can install a package. Someone who
    double-clicked the application cannot, and telling them to run ``pip``
    sends them somewhere they cannot go: the component is meant to be inside
    the bundle, so a bundle without it is a broken download.
    """
    if is_frozen():
        return (
            f"This copy of MLX Transcript is missing a component it ships with "
            f"({package}).\n\n"
            "Download the application again and replace this copy. If you "
            "moved it out of a disk image, drag the whole application to your "
            "Applications folder rather than copying part of it.\n\n"
            "Transcription without speaker detection still works."
        )
    return (
        f"{package} is not installed in this environment.\n\n"
        f"Install it with:  pip install {package}\n\n"
        "Transcription without speaker detection still works."
    )


def apply_runtime_environment() -> None:
    """Expose bundled command-line helpers to libraries that invoke by name."""
    binary_folder = resource_root() / "bin"
    if not binary_folder.is_dir():
        return
    existing = os.environ.get("PATH", "")
    folders = existing.split(os.pathsep) if existing else []
    if str(binary_folder) not in folders:
        os.environ["PATH"] = os.pathsep.join([str(binary_folder), *folders])
