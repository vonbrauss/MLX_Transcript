"""A destination must never put the output tree inside our own source.

The output tree is created at ``<output parent>/Transcription``. macOS
filesystems are case-insensitive by default, so on a checkout of this project
that path resolves onto the ``transcription/`` Python package instead of to a
new folder, and creating the tree writes ScriptSync and Timecoded folders
straight into the application's source. This happened for real while the
release build was being prepared.

Two kinds of test live here. The ones that force the exact folder name run
everywhere, including case-sensitive CI. The ones that rely on a
case-insensitive filesystem are the true reproduction and run on macOS, where
it matters; elsewhere they skip and say why.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from app.main_window import MainWindow  # noqa: E402
from app.settings import AppSettings  # noqa: E402
from transcription import outputs  # noqa: E402
from transcription.outputs import (  # noqa: E402
    SOURCE_PACKAGE_CONFLICT_MESSAGE,
    OutputFormat,
    output_parent_conflicts_with_source,
    output_roots,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(outputs.__file__).resolve().parent


# --------------------------------------------------------------- filesystem


def filesystem_is_case_insensitive(folder: Path) -> bool:
    """Probe the filesystem holding ``folder`` rather than guessing from sys.platform."""
    probe = folder / "CaseProbe"
    probe.mkdir()
    try:
        return (folder / "caseprobe").is_dir()
    finally:
        probe.rmdir()


def require_case_insensitive(folder: Path) -> None:
    if not filesystem_is_case_insensitive(folder):
        pytest.skip(
            "needs a case-insensitive filesystem; this is the macOS "
            "reproduction and cannot occur on a case-sensitive one"
        )


def make_source_package(folder: Path, name: str) -> Path:
    """Create something that looks exactly like our Python package."""
    package = folder / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""stand-in package."""\n')
    (package / "pipeline.py").write_text("# stand-in module\n")
    return package


def make_output_tree(folder: Path) -> Path:
    """Create something that looks exactly like a real output tree."""
    roots = output_roots(folder)
    roots.create(
        (
            OutputFormat.SCRIPTSYNC,
            OutputFormat.TIMECODED,
            OutputFormat.SRT,
        )
    )
    (roots.scriptsync / "clip.txt").write_text("a transcript\n")
    return roots.transcription


# ------------------------------------------- the exact case, case-insensitively


def test_a_lowercase_source_package_is_detected_on_a_case_insensitive_disk(tmp_path):
    """The real bug: transcription/ resolves from <parent>/Transcription."""
    require_case_insensitive(tmp_path)
    package = make_source_package(tmp_path, "transcription")

    conflict = output_parent_conflicts_with_source(tmp_path)

    assert conflict is not None
    # Compared by folder identity, not by spelling: on a case-insensitive
    # filesystem the resolved path keeps the case it was asked for.
    assert conflict.samefile(package)


def test_the_project_root_itself_is_refused_on_a_case_insensitive_disk():
    """Pointing the destination at this checkout must be refused.

    Only ``transcription/`` exists here, so the checkout's own filesystem is
    case-insensitive exactly when ``Transcription`` resolves to a directory.
    That is a read-only probe, so nothing is written into the repository.
    """
    if not (PROJECT_ROOT / "Transcription").is_dir():
        pytest.skip(
            "this checkout is on a case-sensitive filesystem, where the "
            "collision cannot occur"
        )

    conflict = output_parent_conflicts_with_source(PROJECT_ROOT)

    assert conflict is not None
    assert conflict.samefile(PACKAGE_DIR)


def test_an_odd_case_source_package_is_also_detected(tmp_path):
    require_case_insensitive(tmp_path)
    package = make_source_package(tmp_path, "TrAnScRiPtIoN")

    conflict = output_parent_conflicts_with_source(tmp_path)

    assert conflict is not None
    assert conflict.samefile(package)


# ------------------------------------------ the same code path, any filesystem


def test_a_folder_holding_our_package_is_refused(tmp_path):
    package = make_source_package(tmp_path, "Transcription")

    conflict = output_parent_conflicts_with_source(tmp_path)

    assert conflict is not None
    assert conflict.samefile(package)


def test_the_package_is_recognised_even_without_an_init(tmp_path, monkeypatch):
    """The identity check catches a checkout whose __init__.py is unreadable."""
    candidate = tmp_path / "Transcription"
    candidate.mkdir()
    # Make the running package appear to be exactly this folder.
    monkeypatch.setattr(outputs, "__file__", str(candidate / "outputs.py"))

    conflict = output_parent_conflicts_with_source(tmp_path)

    assert conflict is not None
    assert conflict.samefile(candidate)


# ----------------------------------------------- valid destinations stay valid


def test_an_empty_destination_is_fine(tmp_path):
    assert output_parent_conflicts_with_source(tmp_path) is None


def test_a_destination_with_an_existing_output_tree_is_fine(tmp_path):
    make_output_tree(tmp_path)

    assert output_parent_conflicts_with_source(tmp_path) is None


def test_a_destination_that_does_not_exist_yet_is_fine(tmp_path):
    assert output_parent_conflicts_with_source(tmp_path / "not-created-yet") is None


def test_no_destination_is_fine():
    assert output_parent_conflicts_with_source(None) is None


def test_a_sibling_named_transcription_does_not_trip_the_guard(tmp_path):
    """Only <parent>/Transcription matters, not a package further down."""
    make_source_package(tmp_path / "somewhere" / "deeper", "transcription")

    assert output_parent_conflicts_with_source(tmp_path) is None


def test_a_valid_destination_still_creates_all_three_folders(tmp_path):
    """The normal output structure must be untouched by this guard."""
    assert output_parent_conflicts_with_source(tmp_path) is None

    roots = output_roots(tmp_path)
    roots.create(
        (
            OutputFormat.SCRIPTSYNC,
            OutputFormat.TIMECODED,
            OutputFormat.SRT,
            OutputFormat.VTT,
        )
    )

    assert (tmp_path / "Transcription" / "ScriptSync").is_dir()
    assert (tmp_path / "Transcription" / "Timecoded").is_dir()
    assert (tmp_path / "Transcription" / "Subtitles").is_dir()
    # And the guard still passes afterwards, so a second run is not blocked.
    assert output_parent_conflicts_with_source(tmp_path) is None


# ------------------------------------------------------------- the message


def test_the_message_is_the_agreed_wording():
    assert SOURCE_PACKAGE_CONFLICT_MESSAGE == (
        "Choose another save location. This folder contains MLX Transcript's "
        "application files."
    )


def test_the_message_uses_no_jargon():
    lowered = SOURCE_PACKAGE_CONFLICT_MESSAGE.lower()
    for word in ("package", "python", "case-insensitive", "directory", "path"):
        assert word not in lowered


# ------------------------------------------------------------- the interface


@pytest.fixture(scope="module")
def application():
    existing = QApplication.instance()
    yield existing or QApplication([])


@pytest.fixture
def warnings(monkeypatch):
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.main_window.QMessageBox.warning",
        staticmethod(
            lambda parent, title, text, *args, **kwargs: seen.append((title, text))
        ),
    )
    return seen


@pytest.fixture
def window(application):
    built = MainWindow(AppSettings())
    yield built
    built.close()


def test_browsing_to_an_unusable_folder_is_refused(
    window, tmp_path, monkeypatch, warnings
):
    make_source_package(tmp_path, "Transcription")
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *args, **kwargs: str(tmp_path)),
    )

    window._choose_output_parent()

    assert window.output_field.text() == ""
    assert warnings
    assert SOURCE_PACKAGE_CONFLICT_MESSAGE in warnings[0][1]


def test_browsing_to_a_good_folder_is_accepted(window, tmp_path, monkeypatch, warnings):
    destination = tmp_path / "Deliverables"
    destination.mkdir()
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *args, **kwargs: str(destination)),
    )

    window._choose_output_parent()

    assert window.output_field.text() == str(destination)
    assert warnings == []


def test_typing_an_unusable_path_clears_the_field(window, tmp_path, warnings):
    make_source_package(tmp_path, "Transcription")
    window.output_field.setText(str(tmp_path))

    window._on_output_text_changed()

    assert window.output_field.text() == ""
    assert warnings
    assert SOURCE_PACKAGE_CONFLICT_MESSAGE in warnings[0][1]


def test_typing_a_good_path_is_kept(window, tmp_path, warnings):
    window.output_field.setText(str(tmp_path))

    window._on_output_text_changed()

    assert window.output_field.text() == str(tmp_path)
    assert warnings == []


def test_starting_a_batch_refuses_an_unusable_destination(
    window, tmp_path, monkeypatch, warnings
):
    from fractions import Fraction

    from app.models import QueueItem, QueueStatus
    from transcription.media_probe import MediaInfo

    make_source_package(tmp_path, "Transcription")
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"0")
    item = QueueItem(
        source=clip,
        source_root=tmp_path,
        status=QueueStatus.READY,
        media=MediaInfo(path=clip, duration_seconds=1.0, frame_rate=Fraction(24, 1)),
    )
    window.items = [item]
    window._append_row(item)
    # Set the field directly so the guard on typing does not clear it first.
    window.output_field.setText(str(tmp_path))
    monkeypatch.setattr(window, "_confirm_whisper_download", lambda: True)

    window._start_transcription()

    assert not window.is_transcribing
    assert warnings
    assert SOURCE_PACKAGE_CONFLICT_MESSAGE in warnings[0][1]
    assert window.section_stack.currentWidget() is window.folders_group


def test_an_inferred_destination_never_arms_an_unusable_folder(
    window, tmp_path, monkeypatch, warnings
):
    """Choosing a source folder must not auto-fill a destination we refuse."""
    make_source_package(tmp_path, "Transcription")
    (tmp_path / "clip.mov").write_bytes(b"0")
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *args, **kwargs: str(tmp_path)),
    )
    monkeypatch.setattr(window, "_queue_sources", lambda paths: None)

    window._choose_source_folder()

    assert window.source_field.text() == str(tmp_path)
    assert window.output_field.text() == ""
    # No dialog: the user did not pick this destination, it was inferred.
    assert warnings == []
