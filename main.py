#!/usr/bin/env python3
"""Launch MLX Transcript."""

from __future__ import annotations

import sys
import importlib.util
import multiprocessing
import os
import stat
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _route_frozen_helper_processes() -> None:
    """Let PyInstaller divert spawned helpers before the GUI is imported.

    PyInstaller replaces ``multiprocessing.freeze_support`` inside a frozen
    build. Calling it is what stops a worker from continuing through this file
    and opening another copy of the main window.
    """
    multiprocessing.freeze_support()


if __name__ == "__main__":
    _route_frozen_helper_processes()


def _repair_macos_qt_plugin_flags() -> None:
    """Make Qt plugins discoverable after a project sync hides .venv files."""
    if sys.platform != "darwin" or not hasattr(stat, "UF_HIDDEN"):
        return

    pyside_spec = importlib.util.find_spec("PySide6")
    if pyside_spec is None or pyside_spec.origin is None:
        return

    plugins_root = Path(pyside_spec.origin).resolve().parent / "Qt" / "plugins"
    if not plugins_root.is_dir():
        return

    paths = [plugins_root]
    for directory, subdirectories, filenames in os.walk(plugins_root):
        base = Path(directory)
        paths.extend(base / name for name in subdirectories)
        paths.extend(base / name for name in filenames)

    for path in paths:
        try:
            current_flags = path.stat().st_flags
            if current_flags & stat.UF_HIDDEN:
                os.chflags(path, current_flags & ~stat.UF_HIDDEN)
        except OSError:
            # Qt will provide its normal plugin diagnostic if repair is denied.
            pass


_repair_macos_qt_plugin_flags()

from app.runtime import apply_runtime_environment  # noqa: E402

apply_runtime_environment()

# The packaged engine check runs before Qt is imported, so a bundle whose
# transcription stack is broken reports that rather than failing on a display
# it never needed. The Qt plugin repair above still runs first either way.
if __name__ == "__main__" and "--check-engine" in sys.argv:
    from app.selftest import run_engine_selftest  # noqa: E402

    raise SystemExit(run_engine_selftest(sys.argv))

from PySide6.QtCore import QLockFile, QStandardPaths  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.main_window import MainWindow  # noqa: E402
from app.settings import APPLICATION, ORGANIZATION  # noqa: E402
from app.theme import apply_theme  # noqa: E402


def _acquire_instance_lock(
    application: QApplication,
    lock_path: Path | None = None,
) -> QLockFile | None:
    """Keep one GUI instance per macOS user session."""
    if lock_path is None:
        temporary = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.TempLocation
        )
        lock_path = Path(temporary) / "com.vonbrauss.mlxtranscript.lock"
    lock = QLockFile(str(lock_path))
    lock.setStaleLockTime(0)
    if not lock.tryLock(100):
        # A clean shutdown removes the file. After a crash, verify the recorded
        # owner rather than leaving the application permanently unlaunchable.
        try:
            owner_pid, _hostname, _appname = lock.getLockInfo()
        except (OSError, RuntimeError, TypeError):
            owner_pid = 0
        if owner_pid and not _process_is_running(owner_pid):
            try:
                Path(lock.fileName()).unlink(missing_ok=True)
            except OSError:
                return None
            if not lock.tryLock(100):
                return None
        else:
            return None
    # The application owns the lock for its full lifetime.
    application._mlx_transcript_instance_lock = lock  # type: ignore[attr-defined]
    return lock


def _process_is_running(pid: int) -> bool:
    """Return whether a local process still owns a recorded lock PID."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def main() -> int:
    """Start the Qt event loop and show the main window."""
    if "--check-engine" in sys.argv:
        # Reached only when main() is called directly rather than through the
        # module-level shortcut above, which is the path the build script uses.
        from app.selftest import run_engine_selftest

        return run_engine_selftest(sys.argv)

    application = QApplication(sys.argv)
    application.setApplicationName(APPLICATION)
    application.setOrganizationName(ORGANIZATION)
    application.setApplicationDisplayName(APPLICATION)
    if _acquire_instance_lock(application) is None:
        return 0
    apply_theme(application)

    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
