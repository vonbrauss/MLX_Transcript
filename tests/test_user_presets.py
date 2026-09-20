"""Named processing presets persist independently from job folders."""

from __future__ import annotations

from PySide6.QtCore import QSettings

from app.models import ExistingFilePolicy
from app.settings import (
    AppSettings,
    delete_user_preset,
    list_user_presets,
    load_user_preset,
    save_user_preset,
)
from transcription.outputs import FolderLayout


def test_named_preset_round_trip_keeps_current_job_folders(tmp_path, monkeypatch):
    import app.settings as settings_module

    path = tmp_path / "settings.ini"
    monkeypatch.setattr(
        settings_module,
        "_qsettings",
        lambda: QSettings(str(path), QSettings.Format.IniFormat),
    )
    saved = AppSettings(
        source_folder="/old/source",
        output_parent="/old/output",
        language="es",
        existing_policy=ExistingFilePolicy.SKIP,
        folder_layout=FolderLayout.FLAT,
        output_srt=True,
    )
    save_user_preset("Spanish captions", saved)

    current = AppSettings(
        source_folder="/new/source",
        output_parent="/new/output",
    )
    loaded = load_user_preset("Spanish captions", current)

    assert loaded is not None
    assert loaded.source_folder == "/new/source"
    assert loaded.output_parent == "/new/output"
    assert loaded.language == "es"
    assert loaded.existing_policy is ExistingFilePolicy.SKIP
    assert loaded.folder_layout is FolderLayout.FLAT
    assert loaded.output_srt is True


def test_named_presets_are_sorted_replaced_and_deleted(tmp_path, monkeypatch):
    import app.settings as settings_module

    path = tmp_path / "settings.ini"
    monkeypatch.setattr(
        settings_module,
        "_qsettings",
        lambda: QSettings(str(path), QSettings.Format.IniFormat),
    )
    save_user_preset("Zulu", AppSettings(language="en"))
    save_user_preset("alpha", AppSettings(language="fr"))
    save_user_preset("Zulu", AppSettings(language="de"))

    assert list_user_presets() == ("alpha", "Zulu")
    assert load_user_preset("Zulu", AppSettings()).language == "de"
    assert delete_user_preset("Zulu") is True
    assert delete_user_preset("Zulu") is False
    assert list_user_presets() == ("alpha",)
