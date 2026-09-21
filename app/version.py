"""The one place the application's version number is written down.

The PyInstaller recipe, the build script's archive name, and the About
information all read from here. They used to carry their own copies, which is
how an archive ends up named after a version the bundle does not claim.

``packaging/MLX_Transcript.spec`` and ``scripts/build_macos.sh`` read this
file as text rather than importing it, so neither has to load the application
to find out what it is building.
"""

from __future__ import annotations

__all__ = ["VERSION", "BUILD_NUMBER", "ARCHIVE_NAME"]

#: Marketing version, shown as CFBundleShortVersionString.
VERSION = "0.1.0"

#: Build number, shown as CFBundleVersion. Bumped for rebuilds of one version.
BUILD_NUMBER = "1"

#: Name of the release archive, without its extension.
ARCHIVE_NAME = f"MLX-Transcript-{VERSION}-arm64"
