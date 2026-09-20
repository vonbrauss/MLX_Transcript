"""Transcription cleanup presets: values, migration, and the interface."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.main_window import MainWindow  # noqa: E402
from app.settings import AppSettings, load_settings, save_settings  # noqa: E402
from transcription.presets import (  # noqa: E402
    BALANCED,
    PRESERVE_DIFFICULT_SPEECH,
    PRESET_VALUES,
    REDUCE_HALLUCINATIONS,
    THRESHOLD_KEYS,
    CleanupPreset,
    matches_preset,
    preset_for_values,
    preset_values,
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


def spin_values(window: MainWindow) -> tuple[float, float, float]:
    return (
        window.hallucination_spin.value(),
        window.no_speech_spin.value(),
        window.logprob_spin.value(),
    )


def select(window: MainWindow, preset: CleanupPreset) -> None:
    index = window.cleanup_picker.findData(preset)
    assert index >= 0
    window.cleanup_picker.setCurrentIndex(index)


# ------------------------------------------------------------------- values


def test_balanced_has_the_documented_values():
    assert BALANCED == {
        "hallucination_silence_threshold": 1.0,
        "no_speech_threshold": 0.60,
        "logprob_threshold": -1.0,
    }


def test_reduce_hallucinations_has_the_documented_values():
    assert REDUCE_HALLUCINATIONS == {
        "hallucination_silence_threshold": 0.5,
        "no_speech_threshold": 0.50,
        "logprob_threshold": -0.8,
    }


def test_preserve_difficult_speech_has_the_documented_values():
    assert PRESERVE_DIFFICULT_SPEECH == {
        "hallucination_silence_threshold": 1.5,
        "no_speech_threshold": 0.70,
        "logprob_threshold": -1.2,
    }


def test_every_named_preset_sets_all_three_thresholds():
    for values in PRESET_VALUES.values():
        assert set(values) == set(THRESHOLD_KEYS)


def test_custom_has_no_values_of_its_own():
    assert preset_values(CleanupPreset.CUSTOM) is None
    assert CleanupPreset.CUSTOM.is_named is False


def test_preset_values_returns_a_copy():
    values = preset_values(CleanupPreset.BALANCED)
    values["no_speech_threshold"] = 0.99
    assert BALANCED["no_speech_threshold"] == 0.60


# ------------------------------------------------------------------- labels


def test_the_labels_read_as_specified():
    assert CleanupPreset.BALANCED.label == "Balanced — Recommended"
    assert (
        CleanupPreset.REDUCE_HALLUCINATIONS.label
        == "Reduce repeated or invented speech"
    )
    assert (
        CleanupPreset.PRESERVE_DIFFICULT_SPEECH.label
        == "Preserve quiet or difficult speech"
    )
    assert CleanupPreset.CUSTOM.label == "Custom"


def test_the_explanations_read_as_specified():
    assert CleanupPreset.BALANCED.description == (
        "Recommended for most interviews and spoken dialogue."
    )
    assert CleanupPreset.REDUCE_HALLUCINATIONS.description == (
        "Rejects more uncertain text around silence, noise, and music."
    )
    assert CleanupPreset.PRESERVE_DIFFICULT_SPEECH.description == (
        "Keeps more low-confidence dialogue but may also retain mistakes."
    )
    assert CleanupPreset.CUSTOM.description == (
        "Uses the detailed controls under Advanced Settings."
    )


# ---------------------------------------------------------------- migration


@pytest.mark.parametrize("preset", list(PRESET_VALUES))
def test_a_preset_combination_migrates_to_that_preset(preset):
    values = PRESET_VALUES[preset]
    assert (
        preset_for_values(
            values["hallucination_silence_threshold"],
            values["no_speech_threshold"],
            values["logprob_threshold"],
        )
        is preset
    )


@pytest.mark.parametrize(
    "values",
    [
        (1.0, 0.60, -0.9),
        (0.9, 0.60, -1.0),
        (1.0, 0.55, -1.0),
        (2.0, 0.20, -3.0),
        (0.5, 0.70, -1.2),
    ],
)
def test_any_other_combination_migrates_to_custom(values):
    assert preset_for_values(*values) is CleanupPreset.CUSTOM


def test_matching_tolerates_stored_float_rounding():
    assert matches_preset(CleanupPreset.BALANCED, 1.0, 0.6000000001, -1.0) is True
    assert matches_preset(CleanupPreset.BALANCED, 1.0, 0.61, -1.0) is False


def test_custom_never_matches_by_value():
    assert matches_preset(CleanupPreset.CUSTOM, 1.0, 0.60, -1.0) is False


# ----------------------------------------------------------------- settings


def test_balanced_is_the_default_for_new_users():
    settings = AppSettings()
    assert settings.cleanup_preset is CleanupPreset.BALANCED
    assert settings.hallucination_silence_threshold == 1.0
    assert settings.no_speech_threshold == 0.60
    assert settings.logprob_threshold == -1.0


def test_with_preset_replaces_all_three_values():
    settings = AppSettings().with_preset(CleanupPreset.REDUCE_HALLUCINATIONS)
    assert settings.hallucination_silence_threshold == 0.5
    assert settings.no_speech_threshold == 0.50
    assert settings.logprob_threshold == -0.8


def test_with_preset_custom_keeps_the_numbers():
    settings = AppSettings(
        hallucination_silence_threshold=2.0, no_speech_threshold=0.33
    ).with_preset(CleanupPreset.CUSTOM)
    assert settings.cleanup_preset is CleanupPreset.CUSTOM
    assert settings.hallucination_silence_threshold == 2.0
    assert settings.no_speech_threshold == 0.33


def test_the_description_follows_the_preset():
    settings = AppSettings(cleanup_preset=CleanupPreset.PRESERVE_DIFFICULT_SPEECH)
    assert "low-confidence dialogue" in settings.cleanup_description


# --------------------------------------------------------- stored settings


@pytest.fixture
def stored(tmp_path, monkeypatch, application):
    """Point QSettings at a scratch file so nothing real is touched."""
    from PySide6.QtCore import QSettings

    import app.settings as settings_module

    path = tmp_path / "settings.ini"

    def scratch() -> QSettings:
        return QSettings(str(path), QSettings.Format.IniFormat)

    monkeypatch.setattr(settings_module, "_qsettings", scratch)
    return scratch


def test_a_stored_preset_combination_migrates_on_load(stored):
    """Settings written before presets existed carry no preset key."""
    handle = stored()
    for key, value in REDUCE_HALLUCINATIONS.items():
        handle.setValue(key, value)
    handle.sync()

    loaded = load_settings()
    assert loaded.cleanup_preset is CleanupPreset.REDUCE_HALLUCINATIONS
    assert loaded.no_speech_threshold == 0.50


def test_a_stored_nonmatching_combination_becomes_custom(stored):
    handle = stored()
    handle.setValue("hallucination_silence_threshold", 2.25)
    handle.setValue("no_speech_threshold", 0.42)
    handle.setValue("logprob_threshold", -1.75)
    handle.sync()

    loaded = load_settings()
    assert loaded.cleanup_preset is CleanupPreset.CUSTOM
    assert loaded.hallucination_silence_threshold == 2.25
    assert loaded.no_speech_threshold == 0.42
    assert loaded.logprob_threshold == -1.75


def test_an_empty_store_gives_a_new_user_balanced(stored):
    loaded = load_settings()
    assert loaded.cleanup_preset is CleanupPreset.BALANCED


def test_custom_values_persist_across_a_restart(stored):
    save_settings(
        AppSettings(
            cleanup_preset=CleanupPreset.CUSTOM,
            hallucination_silence_threshold=3.0,
            no_speech_threshold=0.15,
            logprob_threshold=-2.5,
        )
    )
    loaded = load_settings()
    assert loaded.cleanup_preset is CleanupPreset.CUSTOM
    assert loaded.hallucination_silence_threshold == 3.0
    assert loaded.no_speech_threshold == 0.15
    assert loaded.logprob_threshold == -2.5


def test_a_named_preset_persists_and_restores_its_values(stored):
    save_settings(AppSettings().with_preset(CleanupPreset.PRESERVE_DIFFICULT_SPEECH))
    loaded = load_settings()
    assert loaded.cleanup_preset is CleanupPreset.PRESERVE_DIFFICULT_SPEECH
    assert loaded.hallucination_silence_threshold == 1.5
    assert loaded.no_speech_threshold == 0.70
    assert loaded.logprob_threshold == -1.2


def test_a_named_preset_repairs_drifted_stored_numbers(stored):
    handle = stored()
    handle.setValue("cleanup_preset", CleanupPreset.BALANCED.value)
    handle.setValue("no_speech_threshold", 0.11)
    handle.sync()

    assert load_settings().no_speech_threshold == 0.60


# ---------------------------------------------------------------- interface


def test_the_picker_offers_every_preset(window):
    labels = [
        window.cleanup_picker.itemText(index)
        for index in range(window.cleanup_picker.count())
    ]
    assert labels == [preset.label for preset in CleanupPreset]


def test_the_window_starts_on_balanced(window):
    assert window._current_preset() is CleanupPreset.BALANCED
    assert spin_values(window) == (1.0, 0.60, -1.0)
    assert window.cleanup_description.text() == CleanupPreset.BALANCED.description


@pytest.mark.parametrize("preset", list(PRESET_VALUES))
def test_selecting_a_preset_updates_all_three_controls(window, preset):
    select(window, preset)
    values = PRESET_VALUES[preset]
    assert spin_values(window) == (
        values["hallucination_silence_threshold"],
        values["no_speech_threshold"],
        values["logprob_threshold"],
    )
    assert window._current_preset() is preset


def test_selecting_a_preset_updates_the_explanation(window):
    select(window, CleanupPreset.REDUCE_HALLUCINATIONS)
    assert window.cleanup_description.text() == (
        "Rejects more uncertain text around silence, noise, and music."
    )


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("hallucination_spin", 2.0),
        ("no_speech_spin", 0.25),
        ("logprob_spin", -3.0),
    ],
)
def test_editing_any_control_selects_custom(window, attribute, value):
    assert window._current_preset() is CleanupPreset.BALANCED
    getattr(window, attribute).setValue(value)

    assert window._current_preset() is CleanupPreset.CUSTOM
    assert window.cleanup_description.text() == CleanupPreset.CUSTOM.description


def test_editing_a_control_keeps_the_other_two(window):
    window.no_speech_spin.setValue(0.25)
    assert spin_values(window) == (1.0, 0.25, -1.0)


def test_switching_back_to_a_preset_restores_its_values(window):
    window.no_speech_spin.setValue(0.25)
    assert window._current_preset() is CleanupPreset.CUSTOM

    select(window, CleanupPreset.BALANCED)
    assert spin_values(window) == (1.0, 0.60, -1.0)


def test_choosing_custom_explicitly_keeps_the_current_numbers(window):
    select(window, CleanupPreset.PRESERVE_DIFFICULT_SPEECH)
    select(window, CleanupPreset.CUSTOM)
    assert spin_values(window) == (1.5, 0.70, -1.2)


def test_collapsing_advanced_settings_does_not_reset_values(window):
    window.no_speech_spin.setValue(0.25)
    window.hallucination_spin.setValue(2.0)

    window.advanced_group.setChecked(True)
    window.advanced_group.setChecked(False)
    window.advanced_group.setChecked(True)

    assert spin_values(window) == (2.0, 0.25, -1.0)
    assert window._current_preset() is CleanupPreset.CUSTOM


def test_the_controls_stay_visible_while_advanced_is_expanded(window):
    # isHidden rather than isVisible: the window itself is never shown here,
    # so isVisible would be False for every child regardless.
    window.advanced_group.setChecked(True)
    assert not any(field.isHidden() for field in window.advanced_fields)
    window.advanced_group.setChecked(False)
    assert all(field.isHidden() for field in window.advanced_fields)


def test_the_advanced_note_names_the_selected_preset(window):
    select(window, CleanupPreset.REDUCE_HALLUCINATIONS)
    assert CleanupPreset.REDUCE_HALLUCINATIONS.label in window.preset_note.text()
    assert "switches Transcription cleanup to Custom" in window.preset_note.text()

    window.no_speech_spin.setValue(0.25)
    assert "your own values" in window.preset_note.text()


def test_each_tooltip_names_the_underlying_setting(window):
    assert "hallucination_silence_threshold" in window.hallucination_spin.toolTip()
    assert "no_speech_threshold" in window.no_speech_spin.toolTip()
    assert "logprob_threshold" in window.logprob_spin.toolTip()


def test_the_preset_round_trips_through_collect(window):
    select(window, CleanupPreset.PRESERVE_DIFFICULT_SPEECH)
    collected = window._collect_settings()
    assert collected.cleanup_preset is CleanupPreset.PRESERVE_DIFFICULT_SPEECH
    assert collected.no_speech_threshold == 0.70

    window.logprob_spin.setValue(-2.0)
    collected = window._collect_settings()
    assert collected.cleanup_preset is CleanupPreset.CUSTOM
    assert collected.logprob_threshold == -2.0


def test_a_window_restored_with_custom_values_keeps_them(application):
    built = MainWindow(
        AppSettings(
            cleanup_preset=CleanupPreset.CUSTOM,
            hallucination_silence_threshold=2.75,
            no_speech_threshold=0.31,
            logprob_threshold=-1.9,
        )
    )
    try:
        assert built._current_preset() is CleanupPreset.CUSTOM
        assert spin_values(built) == (2.75, 0.31, -1.9)
    finally:
        built.close()


# ----------------------------------------------------------- whisper kwargs


@pytest.mark.parametrize("preset", list(PRESET_VALUES))
def test_the_whisper_keyword_arguments_carry_the_preset_values(preset):
    settings = AppSettings().with_preset(preset)
    kwargs = settings.transcription_options().as_whisper_kwargs()
    for key, value in PRESET_VALUES[preset].items():
        assert kwargs[key] == value


def test_custom_values_reach_the_whisper_keyword_arguments():
    settings = AppSettings(
        cleanup_preset=CleanupPreset.CUSTOM,
        hallucination_silence_threshold=2.0,
        no_speech_threshold=0.25,
        logprob_threshold=-3.0,
    )
    kwargs = settings.transcription_options().as_whisper_kwargs()
    assert kwargs["hallucination_silence_threshold"] == 2.0
    assert kwargs["no_speech_threshold"] == 0.25
    assert kwargs["logprob_threshold"] == -3.0


@pytest.mark.parametrize("preset", list(CleanupPreset))
def test_the_fixed_safeguards_are_never_part_of_a_preset(preset):
    kwargs = AppSettings().with_preset(preset).transcription_options().as_whisper_kwargs()
    assert kwargs["temperature"] == 0.0
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["word_timestamps"] is True
    assert kwargs["task"] == "transcribe"
    assert kwargs["compression_ratio_threshold"] == 2.4


def test_a_preset_does_not_touch_speaker_detection():
    settings = AppSettings(detect_speakers=True).with_preset(
        CleanupPreset.REDUCE_HALLUCINATIONS
    )
    assert settings.detect_speakers is True
    assert settings.diarization_options().enabled is True
