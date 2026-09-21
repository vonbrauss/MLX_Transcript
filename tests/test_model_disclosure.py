"""What the application tells the user before it downloads several gigabytes.

The speaker models were disclosed in detail while the far larger Whisper
weights arrived silently behind a progress bar stuck at zero. These cover the
disclosure, the indeterminate progress state, the privacy wording, and the
message shown when a bundled component is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from app import runtime  # noqa: E402
from app.main_window import (  # noqa: E402
    PRIVACY_DETAIL,
    WHISPER_DOWNLOAD_TEXT,
    MainWindow,
)
from app.models import QueueStatus  # noqa: E402
from app.runtime import missing_component_message  # noqa: E402
from app.settings import AppSettings  # noqa: E402
from transcription.engine import (  # noqa: E402
    DEFAULT_MODEL,
    TURBO_MODEL,
    model_cache_root,
    model_is_cached,
)


@pytest.fixture(scope="module")
def application():
    existing = QApplication.instance()
    yield existing or QApplication([])


@pytest.fixture
def window(application):
    built = MainWindow(AppSettings())
    yield built
    built.close()


# ------------------------------------------------------- where weights live


def test_the_cache_root_follows_hf_hub_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))

    assert model_cache_root() == tmp_path / "explicit"


def test_the_cache_root_falls_back_to_hf_home(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))

    assert model_cache_root() == tmp_path / "home" / "hub"


def test_the_cache_root_defaults_under_the_user_cache(monkeypatch, tmp_path):
    for name in ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    assert model_cache_root() == tmp_path / ".cache" / "huggingface" / "hub"


def test_an_empty_cache_is_not_cached(tmp_path):
    assert model_is_cached(DEFAULT_MODEL, root=tmp_path) is False


def test_a_repository_with_a_snapshot_counts_as_cached(tmp_path):
    snapshot = (
        tmp_path
        / "models--mlx-community--whisper-large-v3-mlx"
        / "snapshots"
        / "abc123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "weights.safetensors").write_bytes(b"0")

    assert model_is_cached(DEFAULT_MODEL, root=tmp_path) is True


def test_an_empty_snapshot_folder_does_not_count(tmp_path):
    snapshot = (
        tmp_path / "models--mlx-community--whisper-large-v3-turbo" / "snapshots" / "x"
    )
    snapshot.mkdir(parents=True)

    assert model_is_cached(TURBO_MODEL, root=tmp_path) is False


# --------------------------------------------------------- the disclosure


def test_the_disclosure_is_shown_when_the_model_is_not_cached(
    window, monkeypatch, tmp_path
):
    monkeypatch.setattr("app.main_window.model_is_cached", lambda repo: False)
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda parent, title, text, *args, **kwargs: (
                shown.append(text), QMessageBox.StandardButton.Ok
            )[1]
        ),
    )

    assert window._confirm_whisper_download() is True
    assert shown
    assert "several gigabytes" in shown[0]
    assert "Hugging Face" in shown[0]
    assert PRIVACY_DETAIL in shown[0]


def test_the_disclosure_names_the_cache_folder(window, monkeypatch):
    monkeypatch.setattr("app.main_window.model_is_cached", lambda repo: False)
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda parent, title, text, *args, **kwargs: (
                shown.append(text), QMessageBox.StandardButton.Ok
            )[1]
        ),
    )

    window._confirm_whisper_download()

    assert str(model_cache_root()) in shown[0]


def test_declining_the_disclosure_stops_the_batch(window, monkeypatch):
    monkeypatch.setattr("app.main_window.model_is_cached", lambda repo: False)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Cancel),
    )

    assert window._confirm_whisper_download() is False


def test_a_cached_model_is_not_disclosed_again(window, monkeypatch):
    monkeypatch.setattr("app.main_window.model_is_cached", lambda repo: True)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda *args, **kwargs: pytest.fail("the disclosure was shown twice")
        ),
    )

    assert window._confirm_whisper_download() is True


def test_the_disclosure_template_carries_the_model_label():
    text = WHISPER_DOWNLOAD_TEXT.format(label="Whisper Large v3", folder="/somewhere")

    assert "Whisper Large v3" in text
    assert "/somewhere" in text


# ------------------------------------------------- indeterminate progress


def test_loading_the_model_shows_an_indeterminate_bar(window, tmp_path):
    from app.models import QueueItem
    from transcription.media_probe import MediaInfo

    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"0")
    item = QueueItem(
        source=clip,
        source_root=tmp_path,
        media=MediaInfo(path=clip, duration_seconds=1.0),
    )
    window.items = [item]
    window._append_row(item)
    window._batch_total = 1

    window._on_stage_changed(0, QueueStatus.LOADING_MODEL)

    assert window.progress_bar.maximum() == 0


def test_transcribing_restores_the_counted_bar(window, tmp_path):
    from app.models import QueueItem
    from transcription.media_probe import MediaInfo

    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"0")
    item = QueueItem(
        source=clip,
        source_root=tmp_path,
        media=MediaInfo(path=clip, duration_seconds=1.0),
    )
    window.items = [item]
    window._append_row(item)
    window._batch_total = 4
    window._batch_done = 1

    window._on_stage_changed(0, QueueStatus.LOADING_MODEL)
    assert window.progress_bar.maximum() == 0

    window._on_stage_changed(0, QueueStatus.TRANSCRIBING)

    assert window.progress_bar.maximum() == 4
    assert window.progress_bar.value() == 1


# ------------------------------------------------------------- privacy text


def test_the_privacy_sentence_is_the_agreed_wording():
    assert PRIVACY_DETAIL == (
        "Your media never leaves this Mac. Models download once from Hugging "
        "Face and are reused locally."
    )


def test_the_privacy_sentence_is_on_the_status_chip(window):
    assert window.privacy_label.toolTip() == PRIVACY_DETAIL


def test_the_privacy_sentence_is_on_the_help_page(window):
    assert PRIVACY_DETAIL in window.help_text.toHtml().replace("\n", " ")


# ------------------------------------------- missing component, frozen or not


def test_a_development_checkout_still_gets_pip_instructions(monkeypatch):
    monkeypatch.setattr(runtime, "is_frozen", lambda: False)

    message = missing_component_message("sherpa-onnx")

    assert "pip install sherpa-onnx" in message


def test_a_packaged_application_is_told_to_reinstall(monkeypatch):
    monkeypatch.setattr(runtime, "is_frozen", lambda: True)

    message = missing_component_message("sherpa-onnx")

    assert "pip" not in message
    assert "Download the application again" in message


def test_both_messages_say_transcription_still_works(monkeypatch):
    for frozen in (True, False):
        monkeypatch.setattr(runtime, "is_frozen", lambda: frozen)
        assert "still works" in missing_component_message("sherpa-onnx")
