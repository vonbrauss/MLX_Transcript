"""Settings that travel through Qt widgets must come back as real enums.

Qt stores a string enum as a plain string, so a combo box hands back ``"skip"``
rather than ``ExistingFilePolicy.SKIP``. Every identity check in the pipeline
depends on getting the member itself, which is what these tests pin down.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QMessageBox  # noqa: E402

from app.main_window import MainWindow  # noqa: E402
from app.models import (  # noqa: E402
    BatchSummary,
    ExistingFilePolicy,
    QueueItem,
    QueueStatus,
    ReviewMode,
)
from app.settings import AppSettings  # noqa: E402
from transcription.diarization import SpeakerCountMode  # noqa: E402
from transcription.model_cache import ModelState  # noqa: E402
from transcription.outputs import FolderLayout, NameStyle, OutputFormat  # noqa: E402


@pytest.fixture(scope="module")
def application():
    existing = QApplication.instance()
    yield existing or QApplication([])


@pytest.fixture
def window(application):
    built = MainWindow(AppSettings())
    yield built
    built.close()


def test_qt_really_does_flatten_a_string_enum(application):
    """The behaviour this whole module exists to defend against."""
    picker = QComboBox()
    for mode in ReviewMode:
        picker.addItem(mode.label, mode)
    assert picker.currentData() is not ReviewMode.EVERY_FILE
    assert picker.currentData() == ReviewMode.EVERY_FILE.value


@pytest.mark.parametrize("policy", list(ExistingFilePolicy))
def test_the_existing_file_policy_round_trips(window, policy):
    index = window.policy_picker.findData(policy)
    assert index >= 0
    window.policy_picker.setCurrentIndex(index)

    collected = window._collect_settings().existing_policy
    assert collected is policy


@pytest.mark.parametrize("mode", list(ReviewMode))
def test_the_review_mode_round_trips(window, mode):
    index = window.review_mode_picker.findData(mode)
    assert index >= 0
    window.review_mode_picker.setCurrentIndex(index)

    assert window._collect_settings().review_mode is mode


@pytest.mark.parametrize("mode", list(SpeakerCountMode))
def test_the_speaker_count_mode_round_trips(window, mode):
    index = window.speaker_count_picker.findData(mode)
    assert index >= 0
    window.speaker_count_picker.setCurrentIndex(index)

    assert window._collect_settings().speaker_count_mode is mode


def test_a_junk_value_falls_back_to_the_default(window):
    window.policy_picker.addItem("Broken", "not-a-policy")
    window.policy_picker.setCurrentIndex(window.policy_picker.count() - 1)
    assert window._collect_settings().existing_policy is ExistingFilePolicy.ASK


def test_the_speaker_preferences_round_trip(window):
    window.detect_speakers.setChecked(True)
    window.speakers_in_scriptsync.setChecked(False)
    window.exact_speakers_spin.setValue(4)

    collected = window._collect_settings()
    assert collected.detect_speakers is True
    assert collected.speakers_in_scriptsync is False
    assert collected.exact_speakers == 4


@pytest.mark.parametrize("style", list(NameStyle))
def test_output_name_style_round_trips(window, style):
    window.name_style_picker.setCurrentIndex(window.name_style_picker.findData(style))
    assert window._collect_settings().name_style is style


@pytest.mark.parametrize("layout", list(FolderLayout))
def test_output_folder_layout_round_trips(window, layout):
    window.folder_layout_picker.setCurrentIndex(
        window.folder_layout_picker.findData(layout)
    )
    assert window._collect_settings().folder_layout is layout


def test_selected_output_formats_build_options(window):
    window.output_scriptsync.setChecked(False)
    window.output_timecoded.setChecked(True)
    window.output_srt.setChecked(True)
    window.output_vtt.setChecked(False)
    assert window._collect_settings().output_options().formats == (
        OutputFormat.TIMECODED,
        OutputFormat.SRT,
    )


def test_the_exact_count_field_follows_the_mode(window):
    window.detect_speakers.setChecked(True)
    exact = window.speaker_count_picker.findData(SpeakerCountMode.EXACT)
    window.speaker_count_picker.setCurrentIndex(exact)
    assert window.exact_speakers_spin.isEnabled() is True

    automatic = window.speaker_count_picker.findData(SpeakerCountMode.AUTOMATIC)
    window.speaker_count_picker.setCurrentIndex(automatic)
    assert window.exact_speakers_spin.isEnabled() is False


def test_speaker_controls_are_disabled_until_detection_is_on(window):
    window.detect_speakers.setChecked(False)
    assert not any(widget.isEnabled() for widget in window.speaker_controls)
    window.detect_speakers.setChecked(True)
    assert window.review_mode_picker.isEnabled() is True


def test_diarization_options_are_built_from_the_settings():
    settings = AppSettings(
        detect_speakers=True,
        speaker_count_mode=SpeakerCountMode.EXACT,
        exact_speakers=3,
    )
    options = settings.diarization_options()
    assert options.enabled is True
    assert options.as_backend_kwargs()["num_clusters"] == 3


def test_overwrite_all_is_not_a_persisted_setting():
    assert not hasattr(AppSettings(), "overwrite_all")


def test_activity_line_moves_from_model_loading_to_transcribing(window, tmp_path):
    window.items = [QueueItem(source=tmp_path / "interview.mov", source_root=tmp_path)]

    window._on_stage_changed(0, QueueStatus.LOADING_MODEL)
    assert "Downloading or loading model" in window.current_file_label.fullText()

    window._on_stage_changed(0, QueueStatus.TRANSCRIBING)
    assert window.current_file_label.fullText().startswith("Transcribing")
    assert "Downloading or loading model" not in window.current_file_label.fullText()


def test_activity_line_keeps_the_batch_result_after_controls_update(
    window, monkeypatch
):
    monkeypatch.setattr(window, "_show_batch_summary", lambda _summary: None)
    summary = BatchSummary(completed=1)

    window._on_batch_finished(summary)

    assert window.current_file_label.fullText() == summary.as_sentence()


def test_mixed_queue_roots_gain_distinct_output_labels(window, tmp_path):
    first_root = tmp_path / "Interviews"
    second_root = tmp_path / "Media Day"
    window.items = [
        QueueItem(first_root / "A.mov", first_root),
        QueueItem(second_root / "B.mov", second_root),
    ]

    labels = window._batch_root_labels([0, 1])

    assert labels[window._source_identity(first_root)] == "Interviews"
    assert labels[window._source_identity(second_root)] == "Media Day"


def test_one_queue_root_does_not_add_an_extra_output_folder(window, tmp_path):
    root = tmp_path / "Interviews"
    window.items = [QueueItem(root / "A.mov", root)]
    assert window._batch_root_labels([0]) == {window._source_identity(root): None}


def test_starting_transcription_opens_the_queue_tab(window, monkeypatch, tmp_path):
    """The batch workspace becomes visible as soon as Start is pressed."""
    from app.models import QueueItem
    from transcription.media_probe import MediaInfo

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    window.source_field.setText(str(tmp_path))
    window.output_field.setText(str(tmp_path / "out"))
    window.detect_speakers.setChecked(True)
    window.items = [
            QueueItem(
                source=source,
                source_root=tmp_path,
                relative_folder=".",
            media=MediaInfo(path=source, duration_seconds=1.0),
        )
    ]
    monkeypatch.setattr(window, "_prepare_speaker_models", lambda: False)
    # The first-download disclosure is a separate consent step now.
    monkeypatch.setattr(window, "_confirm_whisper_download", lambda: True)
    window._select_section("folders")

    window._start_transcription()

    assert window.section_stack.currentWidget() is window.queue_group


def test_model_download_accepts_an_equivalent_ok_value(window, monkeypatch):
    """Qt button results must be compared by value rather than identity."""

    class EquivalentOk:
        def __eq__(self, other):
            return other == QMessageBox.StandardButton.Ok

    monkeypatch.setattr(window, "model_state", lambda: ModelState.NOT_DOWNLOADED)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: EquivalentOk())

    assert window._confirm_model_download() is True
