"""Packaged-engine smoke test.

The build script runs this against the frozen executable before it publishes a
release archive, which is the only way to prove that a PyInstaller bundle can
actually reach MLX. Importing a module is not enough: the native ``mlx.core``
extension pulls Python helpers in at import time, the word-timestamp path
compiles through numba and llvmlite, and the tokenizer needs its vocabulary
assets. Each of those is checked here separately so a failure names the real
cause instead of a generic one.

Nothing in this module imports Qt, so the check runs without a display and
without the GUI being loadable.
"""

from __future__ import annotations

import subprocess
import traceback
from dataclasses import dataclass, field
from typing import Callable, Sequence

__all__ = [
    "CheckResult",
    "ENGINE_CHECKS",
    "run_checks",
    "format_report",
    "run_engine_selftest",
]


@dataclass
class CheckResult:
    """The outcome of one named check."""

    name: str
    passed: bool
    detail: str = ""
    traceback_text: str = ""

    @property
    def symbol(self) -> str:
        return "ok  " if self.passed else "FAIL"


# --------------------------------------------------------------------- checks


def check_mlx_helpers() -> str:
    """The Python helpers ``mlx.core`` imports the moment it loads.

    These are invisible to static analysis because the import happens inside
    the native extension, so a bundle that misses them fails with a bare
    ImportError that looks like MLX not being installed at all.
    """
    import importlib.util

    found = []
    for name in ("mlx._reprlib_fix", "mlx.__array_api_info"):
        if importlib.util.find_spec(name) is None:
            raise ModuleNotFoundError(f"No module named {name!r}")
        found.append(name)
    # Avoid loading mlx.core indirectly before its dedicated check. Native
    # nanobind modules must be initialized exactly once in a frozen process.
    return ", ".join(found)


def check_mlx_core() -> str:
    """Load the native extension and run one real computation on it."""
    import mlx.core as mx

    total = mx.array([1.0, 2.0, 3.0]).sum()
    mx.eval(total)
    value = float(total.item())
    if value != 6.0:
        raise RuntimeError(f"mlx.core computed {value}, expected 6.0")
    return f"mlx {getattr(mx, '__version__', 'unknown')}, device {mx.default_device()}"


def check_mlx_neural_network() -> str:
    """The model layers and the tree utilities the weights loader needs."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    layer = nn.Linear(4, 2)
    output = layer(mx.zeros((1, 4)))
    mx.eval(output)
    return f"{len(tree_flatten(layer.parameters()))} parameter tensors"


def check_mlx_whisper_import() -> str:
    """Import the package itself, which pulls in the whole decoding stack."""
    import mlx_whisper

    return f"mlx_whisper {getattr(mlx_whisper, '__version__', 'unknown')}"


def check_tokenizer() -> str:
    """Build the multilingual tokenizer from its bundled vocabulary."""
    from mlx_whisper.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(multilingual=True)
    encoded = tokenizer.encoding.encode("rolling on the interview")
    if not encoded:
        raise RuntimeError("the tokenizer produced no tokens")
    return f"{len(encoded)} tokens from the packaged vocabulary"


def check_mel_filters() -> str:
    """The mel filterbank asset has to travel with the bundle."""
    from mlx_whisper import audio

    filters = audio.mel_filters(80)
    shape = getattr(filters, "shape", None)
    if not shape:
        raise RuntimeError("mel_filters returned nothing usable")
    return f"mel filterbank {tuple(shape)}"


def check_word_timestamp_path() -> str:
    """Word timestamps go through numba, so the JIT must work in the bundle.

    The application always asks for word timestamps, which means a bundle
    without a working llvmlite fails on the first real clip rather than at
    import time.
    """
    import numba
    import numpy

    from mlx_whisper import timing  # noqa: F401 - imports numba and scipy

    @numba.jit(nopython=True, cache=False)
    def _total(values):
        result = 0.0
        for value in values:
            result += value
        return result

    total = _total(numpy.array([1.0, 2.0, 3.0]))
    if total != 6.0:
        raise RuntimeError(f"the numba kernel returned {total}, expected 6.0")
    return f"numba {numba.__version__} compiled and ran a kernel"


def check_media_tools() -> str:
    """ffmpeg and ffprobe must be reachable, bundled or on PATH."""
    from app.runtime import bundled_binary

    found = []
    for name in ("ffmpeg", "ffprobe"):
        located = bundled_binary(name)
        result = subprocess.run(
            [located, "-version"], capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"{name} exited {result.returncode}")
        found.append(f"{name} at {located}")
    return "; ".join(found)


def check_diarization_backend() -> str:
    """Speaker detection is optional, so this reports rather than fails."""
    from transcription.diarization import backend_available

    return "sherpa-onnx present" if backend_available() else "sherpa-onnx absent"


@dataclass(frozen=True)
class Check:
    """One named check and whether the build may ship without it."""

    name: str
    run: Callable[[], str]
    required: bool = True


ENGINE_CHECKS: tuple[Check, ...] = (
    Check("mlx dynamic helpers", check_mlx_helpers),
    Check("mlx.core computation", check_mlx_core),
    Check("mlx.nn layers", check_mlx_neural_network),
    Check("mlx_whisper import", check_mlx_whisper_import),
    Check("whisper tokenizer", check_tokenizer),
    Check("mel filter assets", check_mel_filters),
    Check("word timestamp path", check_word_timestamp_path),
    Check("bundled media tools", check_media_tools),
    Check("diarization backend", check_diarization_backend, required=False),
)


# ---------------------------------------------------------------- the runner


@dataclass
class SelfTestReport:
    """Every check's outcome plus the overall verdict."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [result for result in self.results if not result.passed]

    @property
    def passed(self) -> bool:
        return not self.failures


def run_checks(checks: Sequence[Check] = ENGINE_CHECKS) -> SelfTestReport:
    """Run every check, never stopping at the first failure."""
    report = SelfTestReport()
    for check in checks:
        try:
            detail = check.run()
        except Exception as error:
            if check.required:
                report.results.append(
                    CheckResult(
                        name=check.name,
                        passed=False,
                        detail=f"{type(error).__name__}: {error}",
                        traceback_text=traceback.format_exc(),
                    )
                )
            else:
                report.results.append(
                    CheckResult(
                        name=check.name,
                        passed=True,
                        detail=f"skipped ({type(error).__name__}: {error})",
                    )
                )
            # A failed native MLX initialization may register nanobind types
            # before reporting that Metal is unavailable. Importing it again
            # in this same process then aborts instead of raising normally.
            # Preserve the useful first traceback and stop this diagnostic.
            if "No Metal device available" in str(error):
                break
        else:
            report.results.append(
                CheckResult(name=check.name, passed=True, detail=detail)
            )
    return report


def format_report(report: SelfTestReport) -> str:
    """Render the report, with the real traceback for anything that failed."""
    lines = ["MLX Transcript packaged engine check", ""]
    for result in report.results:
        lines.append(f"  [{result.symbol}] {result.name}: {result.detail}")

    if report.passed:
        lines.append("")
        lines.append("All required checks passed.")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"{len(report.failures)} required check(s) failed.")
    for result in report.failures:
        lines.append("")
        lines.append(f"--- {result.name} ---")
        lines.append(result.traceback_text.rstrip())
    return "\n".join(lines)


def run_engine_selftest(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``MLX Transcript --check-engine``.

    Returns a process exit status: 0 when every required check passed.
    ``--report <path>`` also writes the report to a file so a build log can
    keep it.
    """
    from pathlib import Path

    argv = list(argv or [])
    report = run_checks()
    text = format_report(report)

    if "--report" in argv:
        position = argv.index("--report") + 1
        if position < len(argv):
            try:
                Path(argv[position]).write_text(text, encoding="utf-8")
            except OSError as error:
                text += f"\n\nCould not write the report file: {error}"

    print(text, flush=True)
    return 0 if report.passed else 1
