"""Speaker presets, persistence, interface behavior, and pipeline options."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.main_window import MainWindow  # noqa: E402
from app.settings import AppSettings, load_settings, save_settings  # noqa: E402
from transcription.speaker_presets import (  # noqa: E402
    PRESET_VALUES,
    SpeakerPreset,
    preset_for_values,
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


@pytest.fixture
def stored(tmp_path, monkeypatch, application):
    import app.settings as settings_module

    path = tmp_path / "settings.ini"

    def scratch() -> QSettings:
        return QSettings(str(path), QSettings.Format.IniFormat)

    monkeypatch.setattr(settings_module, "_qsettings", scratch)
    return scratch


def _select(window: MainWindow, preset: SpeakerPreset) -> None:
    window.speaker_preset_picker.setCurrentIndex(
        window.speaker_preset_picker.findData(preset)
    )


def _values(window: MainWindow) -> tuple[float, ...]:
    return tuple(spin.value() for spin in window.speaker_setting_spins)


def test_every_named_preset_has_all_five_values():
    assert set(PRESET_VALUES) == set(SpeakerPreset) - {SpeakerPreset.CUSTOM}
    for values in PRESET_VALUES.values():
        assert set(values) == {
            "clustering_threshold",
            "min_duration_on",
            "min_duration_off",
            "nearest_tolerance",
            "merge_gap",
        }


@pytest.mark.parametrize("preset", list(PRESET_VALUES))
def test_selecting_a_preset_updates_every_control(window, preset):
    _select(window, preset)
    expected = PRESET_VALUES[preset]
    assert _values(window) == pytest.approx(tuple(expected.values()))
    assert window.speaker_preset_description.text() == preset.description


def test_editing_one_value_switches_to_custom_without_changing_the_rest(window):
    before = _values(window)
    window.speaker_threshold_spin.setValue(0.72)
    after = _values(window)
    assert window._current_speaker_preset() is SpeakerPreset.CUSTOM
    assert after[0] == pytest.approx(0.72)
    assert after[1:] == pytest.approx(before[1:])


def test_advanced_speaker_controls_are_hidden_until_requested(window):
    assert window.speaker_advanced_panel.isHidden()
    window.detect_speakers.setChecked(True)
    window.speaker_advanced_toggle.setChecked(True)
    assert not window.speaker_advanced_panel.isHidden()


def test_collected_values_reach_the_diarization_options(window):
    window.detect_speakers.setChecked(True)
    window.speaker_threshold_spin.setValue(0.61)
    window.min_speech_spin.setValue(0.42)
    window.min_pause_spin.setValue(0.31)
    window.speaker_tolerance_spin.setValue(0.57)
    window.speaker_merge_spin.setValue(0.83)
    options = window._collect_settings().diarization_options()
    assert options.clustering_threshold == pytest.approx(0.61)
    assert options.min_duration_on == pytest.approx(0.42)
    assert options.min_duration_off == pytest.approx(0.31)
    assert options.nearest_tolerance == pytest.approx(0.57)
    assert options.merge_gap == pytest.approx(0.83)


def test_custom_values_survive_a_settings_round_trip(stored):
    save_settings(
        AppSettings(
            speaker_preset=SpeakerPreset.CUSTOM,
            speaker_advanced_expanded=True,
            clustering_threshold=0.72,
            min_duration_on=0.44,
            min_duration_off=0.28,
            nearest_tolerance=0.63,
            merge_gap=0.91,
        )
    )
    loaded = load_settings()
    assert loaded.speaker_preset is SpeakerPreset.CUSTOM
    assert loaded.speaker_advanced_expanded is True
    assert loaded.clustering_threshold == pytest.approx(0.72)
    assert loaded.min_duration_on == pytest.approx(0.44)
    assert loaded.min_duration_off == pytest.approx(0.28)
    assert loaded.nearest_tolerance == pytest.approx(0.63)
    assert loaded.merge_gap == pytest.approx(0.91)


def test_old_default_settings_migrate_to_balanced(stored):
    handle = stored()
    handle.setValue("clustering_threshold", 0.5)
    handle.sync()
    assert load_settings().speaker_preset is SpeakerPreset.BALANCED


def test_unmatched_old_settings_migrate_to_custom(stored):
    handle = stored()
    handle.setValue("clustering_threshold", 0.77)
    handle.sync()
    loaded = load_settings()
    assert loaded.speaker_preset is SpeakerPreset.CUSTOM
    assert loaded.clustering_threshold == pytest.approx(0.77)


def test_preset_matching_round_trip():
    for preset, values in PRESET_VALUES.items():
        assert preset_for_values(**values) is preset
