"""The PyInstaller recipe and the packaged-engine smoke test.

None of this builds anything. It pins the parts of the recipe that a bundle
silently fails without, and exercises the smoke test's own logic with stand-in
checks, so a regression shows up here rather than in a 200 MB archive.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from app import selftest
from app.selftest import (
    ENGINE_CHECKS,
    Check,
    CheckResult,
    SelfTestReport,
    format_report,
    run_checks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "packaging" / "MLX_Transcript.spec"
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_macos.sh"
MAIN_PATH = PROJECT_ROOT / "main.py"


@pytest.fixture(scope="module")
def spec_source() -> str:
    return SPEC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def build_source() -> str:
    return BUILD_SCRIPT.read_text(encoding="utf-8")


# ------------------------------------------------------------- the mlx helpers


@pytest.mark.parametrize(
    "module",
    [
        "mlx",
        "mlx.core",
        "mlx._reprlib_fix",
        "mlx.__array_api_info",
        "mlx.nn",
        "mlx.optimizers",
        "mlx.utils",
    ],
)
def test_every_required_mlx_module_is_named_explicitly(spec_source, module):
    """These are imported by the native extension, not by any Python source.

    A build that omits them raises ImportError on "import mlx_whisper", which
    is indistinguishable from mlx-whisper not being installed.
    """
    assert f'"{module}"' in spec_source


def test_the_required_list_is_a_literal_the_spec_actually_uses(spec_source):
    assert "REQUIRED_MLX_HIDDENIMPORTS" in spec_source
    assert "hiddenimports += REQUIRED_MLX_HIDDENIMPORTS" in spec_source


def test_mlx_native_libraries_are_not_bundled_twice(spec_source):
    assert 'if not entry[0].endswith(".dylib")' in spec_source
    assert 'collect_dynamic_libs("mlx")' not in spec_source


def test_the_word_timestamp_dependencies_are_named(spec_source):
    """Word timestamps are always on, so numba and scipy must be bundled."""
    for module in ("numba", "scipy.signal", "mlx_whisper.timing"):
        assert f'"{module}"' in spec_source


def test_the_tokenizer_dependencies_are_named(spec_source):
    for module in ("tiktoken", "mlx_whisper.tokenizer"):
        assert f'"{module}"' in spec_source


def test_data_folders_that_are_not_modules_are_skipped(spec_source):
    """Walking mlx/include produced hidden imports that cannot resolve."""
    assert "MLX_NON_MODULE_FOLDERS" in spec_source
    for folder in ("include", "lib", "share"):
        assert f'"{folder}"' in spec_source


def test_the_spec_parses_as_python(spec_source):
    ast.parse(spec_source)


def test_the_build_stays_apple_silicon_only(spec_source, build_source):
    assert 'target_arch="arm64"' in spec_source
    assert 'uname -m' in build_source and 'arm64' in build_source


def test_frozen_helpers_are_diverted_before_qt_is_imported():
    source = MAIN_PATH.read_text(encoding="utf-8")
    route = source.index("\n    _route_frozen_helper_processes()")
    qt_import = source.index("from PySide6")
    assert route < qt_import
    assert "multiprocessing.freeze_support()" in source


def test_a_second_gui_cannot_acquire_the_same_instance_lock(tmp_path):
    from types import SimpleNamespace

    import main

    application = SimpleNamespace()
    path = tmp_path / "instance.lock"
    first = main._acquire_instance_lock(application, path)
    assert first is not None
    try:
        assert main._acquire_instance_lock(application, path) is None
    finally:
        first.unlock()


def test_a_crashed_instance_lock_is_recovered_immediately(tmp_path):
    from types import SimpleNamespace

    import main

    path = tmp_path / "crashed.lock"
    script = (
        "import os; from PySide6.QtCore import QLockFile; "
        f"lock=QLockFile({str(path)!r}); lock.setStaleLockTime(0); "
        "assert lock.tryLock(0); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    assert path.exists()

    recovered = main._acquire_instance_lock(SimpleNamespace(), path)
    assert recovered is not None
    recovered.unlock()


def test_ffmpeg_bundling_and_ad_hoc_signing_are_preserved(spec_source, build_source):
    assert "MLX_TRANSCRIPT_BUNDLE_FFMPEG" in spec_source
    assert "ffmpeg" in spec_source and "ffprobe" in spec_source
    assert "codesign --force --deep --sign -" in build_source
    # Ad hoc signing means local testing needs no Developer ID.
    assert "Developer ID" not in build_source.split("# ")[0]


# ------------------------------------------------------- the module discovery


def test_discovery_only_returns_importable_modules(tmp_path):
    """Read the helper straight out of the spec and exercise it."""
    namespace: dict = {}
    source = SPEC_PATH.read_text(encoding="utf-8")
    start = source.index("def discover_package_modules")
    end = source.index("mlx_path = mlx_package_path")
    exec("from pathlib import Path\n" + source[start:end], namespace)
    discover = namespace["discover_package_modules"]

    package = tmp_path / "fake"
    (package / "nn").mkdir(parents=True)
    (package / "include" / "deep").mkdir(parents=True)
    (package / "stubs").mkdir()
    (package / "helper.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "nn" / "__init__.py").write_text("", encoding="utf-8")
    (package / "nn" / "layers.py").write_text("", encoding="utf-8")
    (package / "include" / "deep" / "Generator.py").write_text("", encoding="utf-8")
    (package / "stubs" / "thing.py").write_text("", encoding="utf-8")

    found = discover(package, "fake", {"include"})

    assert "fake.helper" in found
    assert "fake.nn" in found
    assert "fake.nn.layers" in found
    # No __init__.py means it is not an importable package.
    assert not any(name.startswith("fake.stubs") for name in found)
    # Skipped folders never contribute, however deep they go.
    assert not any("include" in name for name in found)
    # __init__ is the package itself, never a module name of its own.
    assert "fake.__init__" not in found


# --------------------------------------------------------- the engine checks


def test_the_check_list_covers_the_failure_modes_that_matter():
    names = [check.name for check in ENGINE_CHECKS]
    assert "mlx dynamic helpers" in names
    assert "mlx.core computation" in names
    assert "whisper tokenizer" in names
    assert "word timestamp path" in names
    assert "bundled media tools" in names


def test_importing_the_smoke_test_loads_no_engine():
    """The module must stay importable without MLX, Qt, or a display."""
    import sys

    assert "mlx" not in sys.modules
    assert "PySide6.QtWidgets" not in str(selftest.__dict__.get("__all__", ""))


def test_a_passing_run_reports_every_check():
    checks = (
        Check("first", lambda: "all good"),
        Check("second", lambda: "also good"),
    )
    report = run_checks(checks)

    assert report.passed is True
    assert [result.name for result in report.results] == ["first", "second"]
    text = format_report(report)
    assert "All required checks passed." in text
    assert "all good" in text


def test_a_failing_check_reports_the_real_error_not_a_guess():
    def explode() -> str:
        raise ImportError("No module named 'mlx._reprlib_fix'")

    report = run_checks((Check("mlx dynamic helpers", explode),))

    assert report.passed is False
    text = format_report(report)
    assert "mlx._reprlib_fix" in text
    assert "ImportError" in text
    assert "Traceback" in text
    # The old message blamed a package the user had installed all along.
    assert "pip install mlx-whisper" not in text


def test_every_check_runs_even_after_one_fails():
    calls: list[str] = []

    def record(name: str):
        def run() -> str:
            calls.append(name)
            if name == "second":
                raise RuntimeError("boom")
            return name

        return run

    report = run_checks(
        tuple(Check(name, record(name)) for name in ("first", "second", "third"))
    )

    assert calls == ["first", "second", "third"]
    assert len(report.failures) == 1


def test_metal_unavailable_stops_before_reimporting_native_module():
    calls: list[str] = []

    def no_metal() -> str:
        calls.append("core")
        raise ImportError("No Metal device available")

    def unsafe_retry() -> str:
        calls.append("retry")
        return "should not run"

    report = run_checks((Check("core", no_metal), Check("retry", unsafe_retry)))
    assert calls == ["core"]
    assert len(report.failures) == 1


def test_an_optional_check_never_fails_the_build():
    def explode() -> str:
        raise RuntimeError("sherpa-onnx absent")

    report = run_checks((Check("diarization backend", explode, required=False),))

    assert report.passed is True
    assert "skipped" in report.results[0].detail


def test_the_exit_status_follows_the_verdict(monkeypatch, capsys):
    monkeypatch.setattr(
        selftest, "run_checks", lambda *args, **kwargs: SelfTestReport(
            [CheckResult("thing", passed=True, detail="fine")]
        )
    )
    assert selftest.run_engine_selftest([]) == 0

    monkeypatch.setattr(
        selftest, "run_checks", lambda *args, **kwargs: SelfTestReport(
            [CheckResult("thing", passed=False, detail="broken")]
        )
    )
    assert selftest.run_engine_selftest([]) == 1
    assert "broken" in capsys.readouterr().out


def test_the_report_can_be_written_to_a_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        selftest, "run_checks", lambda *args, **kwargs: SelfTestReport(
            [CheckResult("thing", passed=True, detail="fine")]
        )
    )
    target = tmp_path / "engine-check.txt"
    selftest.run_engine_selftest(["--report", str(target)])
    assert "fine" in target.read_text(encoding="utf-8")


# ------------------------------------------------------------ the entry point


def test_the_check_runs_before_qt_is_imported():
    """A broken engine must not be masked by a display problem."""
    source = MAIN_PATH.read_text(encoding="utf-8")
    check = source.index("--check-engine")
    qt_import = source.index("from PySide6")
    assert check < qt_import


def test_the_qt_plugin_repair_still_runs_first():
    source = MAIN_PATH.read_text(encoding="utf-8")
    repair = source.index("\n_repair_macos_qt_plugin_flags()")
    assert repair < source.index("--check-engine")
    assert repair < source.index("from PySide6")


# -------------------------------------------------------------- the build gate


def test_the_archive_is_gated_on_the_engine_check(build_source):
    check = build_source.index("--check-engine")
    archive = build_source.index("ditto -c -k")
    assert check < archive
    assert "No archive was written." in build_source


def test_a_stale_app_cannot_survive_a_build(build_source):
    """A leftover .app in dist/ is how a fixed bug appears to come back."""
    assert 'rm -rf "$project_root/dist/MLX Transcript.app"' in build_source
    assert 'ditto "$app_path" "$project_root/dist/MLX Transcript.app"' not in build_source
