"""Keep the Mac awake while a batch is running.

macOS ships ``caffeinate``, so an assertion is just a child process that lives
as long as the batch. Releasing it is terminating that process, which happens
when the batch finishes, fails, is cancelled, or the window closes.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Callable

__all__ = ["CAFFEINATE_COMMAND", "SleepBlocker"]

#: ``-i`` prevents idle sleep and ``-m`` keeps the disks from spinning down.
CAFFEINATE_COMMAND = ["caffeinate", "-i", "-m"]

PopenFactory = Callable[[list[str]], Any]


class SleepBlocker:
    """Hold a sleep assertion for as long as a batch is active.

    ``popen_factory`` and ``platform`` are injectable so the behavior can be
    tested without spawning anything.
    """

    def __init__(
        self,
        popen_factory: PopenFactory | None = None,
        platform: str | None = None,
    ) -> None:
        self._popen_factory = popen_factory or self._default_popen
        self._platform = platform if platform is not None else sys.platform
        self._process: Any | None = None
        self.last_error: str = ""

    @staticmethod
    def _default_popen(command: list[str]) -> Any:
        return subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @property
    def supported(self) -> bool:
        """True on macOS, where ``caffeinate`` exists."""
        return self._platform == "darwin"

    @property
    def active(self) -> bool:
        """True while the assertion is held."""
        return self._process is not None

    def acquire(self) -> bool:
        """Start the assertion. Returns True when one is now held."""
        if self._process is not None:
            return True
        if not self.supported:
            return False
        try:
            self._process = self._popen_factory(list(CAFFEINATE_COMMAND))
        except (OSError, ValueError) as error:
            self.last_error = str(error)
            self._process = None
            return False
        self.last_error = ""
        return True

    def release(self) -> None:
        """Drop the assertion. Safe to call when none is held."""
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            process.terminate()
        except (OSError, ValueError) as error:  # pragma: no cover - defensive
            self.last_error = str(error)
            return
        wait = getattr(process, "wait", None)
        if callable(wait):
            try:
                wait(timeout=5)
            except TypeError:  # pragma: no cover - stand-in without timeout
                wait()
            except Exception:  # pragma: no cover - defensive
                pass

    def __enter__(self) -> "SleepBlocker":
        self.acquire()
        return self

    def __exit__(self, *_exception: object) -> None:
        self.release()
