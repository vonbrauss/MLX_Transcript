"""Persisted user choices, stored with QSettings under the app's own domain."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
import json

from PySide6.QtCore import QSettings

from app.models import ExistingFilePolicy, ReviewMode
from transcription.diarization import DiarizationOptions, SpeakerCountMode
from transcription.engine import (
    AUTO_DETECT_LANGUAGE,
    DEFAULT_MODEL,
    MODEL_CHOICES,
    TranscriptionOptions,
)
from transcription.presets import CleanupPreset, preset_for_values, preset_values
from transcription.speaker_presets import (
    SpeakerPreset,
    preset_for_values as speaker_preset_for_values,
    preset_values as speaker_preset_values,
)
from transcription.outputs import FolderLayout, NameStyle, OutputFormat, OutputOptions

__all__ = [
    "ORGANIZATION",
    "APPLICATION",
    "LANGUAGE_CHOICES",
    "AppSettings",
    "load_settings",
    "save_settings",
    "list_user_presets",
    "load_user_preset",
    "save_user_preset",
    "delete_user_preset",
]

ORGANIZATION = "MLX Transcript"
APPLICATION = "MLX Transcript"
USER_PRESETS_KEY = "user_presets"

PRESET_EXCLUDED_FIELDS = {
    "source_folder", "output_parent", "folders_expanded",
    "transcription_expanded", "speaker_expanded", "advanced_expanded",
    "queue_expanded", "output_expanded", "selected_section",
}

_ENUM_FIELDS = {
    "existing_policy": ExistingFilePolicy,
    "cleanup_preset": CleanupPreset,
    "name_style": NameStyle,
    "folder_layout": FolderLayout,
    "speaker_preset": SpeakerPreset,
    "speaker_count_mode": SpeakerCountMode,
    "review_mode": ReviewMode,
}

#: Label shown in the language picker mapped to the Whisper language code.
LANGUAGE_CHOICES: dict[str, str] = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Korean": "ko",
    "Mandarin": "zh",
    "Auto Detect": AUTO_DETECT_LANGUAGE,
}


@dataclass
class AppSettings:
    """Every choice the window restores between launches."""

    source_folder: str = ""
    output_parent: str = ""
    model: str = DEFAULT_MODEL
    language: str = "en"
    existing_policy: ExistingFilePolicy = ExistingFilePolicy.ASK
    # The Transcription cleanup preset, and the three thresholds it fills in.
    # The numbers stay the source of truth for what is passed to Whisper: the
    # preset is only a convenient way to set all three at once.
    cleanup_preset: CleanupPreset = CleanupPreset.BALANCED
    hallucination_silence_threshold: float = 1.0
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    folders_expanded: bool = True
    transcription_expanded: bool = True
    speaker_expanded: bool = False
    advanced_expanded: bool = False
    queue_expanded: bool = True
    output_expanded: bool = True
    selected_section: str = "folders"

    # Output naming, organization, and formats.
    name_style: NameStyle = NameStyle.ORIGINAL
    custom_suffix: str = "_transcript"
    folder_layout: FolderLayout = FolderLayout.TREE
    output_scriptsync: bool = True
    output_timecoded: bool = True
    output_srt: bool = False
    output_vtt: bool = False
    # "Overwrite all" is deliberately absent: it is a decision for one batch
    # and resets every time a new batch starts.

    # Speaker detection
    detect_speakers: bool = False
    speaker_preset: SpeakerPreset = SpeakerPreset.BALANCED
    speaker_advanced_expanded: bool = False
    speaker_count_mode: SpeakerCountMode = SpeakerCountMode.AUTOMATIC
    exact_speakers: int = 2
    clustering_threshold: float = 0.5
    min_duration_on: float = 0.3
    min_duration_off: float = 0.5
    nearest_tolerance: float = 0.75
    merge_gap: float = 1.0
    speakers_in_scriptsync: bool = True
    speakers_in_subtitles: bool = True
    review_mode: ReviewMode = ReviewMode.EVERY_FILE

    @property
    def model_label(self) -> str:
        """Return the picker label for the stored model repository."""
        for label, repository in MODEL_CHOICES.items():
            if repository == self.model:
                return label
        return next(iter(MODEL_CHOICES))

    @property
    def language_label(self) -> str:
        """Return the picker label for the stored language code."""
        for label, code in LANGUAGE_CHOICES.items():
            if code == (self.language or AUTO_DETECT_LANGUAGE):
                return label
        return "English"

    @property
    def cleanup_description(self) -> str:
        """The plain-language line shown under the preset picker."""
        return self.cleanup_preset.description

    def with_preset(self, preset: CleanupPreset) -> "AppSettings":
        """Return a copy using a preset's thresholds, or unchanged for Custom."""
        values = preset_values(preset)
        if values is None:
            return replace(self, cleanup_preset=preset)
        return replace(self, cleanup_preset=preset, **values)

    def diarization_options(self) -> DiarizationOptions:
        """Build the diarizer options from the current settings."""
        return DiarizationOptions(
            enabled=self.detect_speakers,
            count_mode=self.speaker_count_mode,
            exact_speakers=self.exact_speakers,
            clustering_threshold=self.clustering_threshold,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
            nearest_tolerance=self.nearest_tolerance,
            merge_gap=self.merge_gap,
        )

    def with_speaker_preset(self, preset: SpeakerPreset) -> "AppSettings":
        values = speaker_preset_values(preset)
        if values is None:
            return replace(self, speaker_preset=preset)
        return replace(self, speaker_preset=preset, **values)

    def transcription_options(self, detect_speakers: bool | None = None) -> TranscriptionOptions:
        """Build the engine options from the current settings."""
        return TranscriptionOptions(
            model=self.model,
            language=self.language,
            hallucination_silence_threshold=self.hallucination_silence_threshold,
            no_speech_threshold=self.no_speech_threshold,
            logprob_threshold=self.logprob_threshold,
            detect_speakers=(
                self.detect_speakers if detect_speakers is None else detect_speakers
            ),
        )

    def output_options(self) -> OutputOptions:
        formats = tuple(
            item
            for enabled, item in (
                (self.output_scriptsync, OutputFormat.SCRIPTSYNC),
                (self.output_timecoded, OutputFormat.TIMECODED),
                (self.output_srt, OutputFormat.SRT),
                (self.output_vtt, OutputFormat.VTT),
            )
            if enabled
        )
        return OutputOptions(
            name_style=self.name_style,
            custom_suffix=self.custom_suffix,
            folder_layout=self.folder_layout,
            formats=formats,
        )


def _qsettings() -> QSettings:
    return QSettings(ORGANIZATION, APPLICATION)


def _as_float(value: object, fallback: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _as_bool(value: object, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return fallback


def _as_int(value: object, fallback: int, minimum: int | None = None) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    if minimum is not None and number < minimum:
        return fallback
    return number


def _as_enum(enum_type, value: object, fallback):
    """Read a stored string back into one of our string enums."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError):
        return fallback


def load_settings() -> AppSettings:
    """Read the stored settings, falling back to the documented defaults."""
    stored = _qsettings()
    defaults = AppSettings()

    policy_value = str(stored.value("existing_policy", defaults.existing_policy.value))
    try:
        policy = ExistingFilePolicy(policy_value)
    except ValueError:
        policy = defaults.existing_policy

    model = str(stored.value("model", defaults.model))
    if model not in MODEL_CHOICES.values():
        model = defaults.model

    # QSettings can hand back None for a stored empty string, which is exactly
    # what Auto Detect is, so it is normalized rather than stringified.
    language_value = stored.value("language", defaults.language)
    language = AUTO_DETECT_LANGUAGE if language_value is None else str(language_value)
    if language not in LANGUAGE_CHOICES.values():
        language = defaults.language

    hallucination = _as_float(
        stored.value("hallucination_silence_threshold"),
        defaults.hallucination_silence_threshold,
    )
    no_speech = _as_float(
        stored.value("no_speech_threshold"), defaults.no_speech_threshold
    )
    logprob = _as_float(
        stored.value("logprob_threshold"), defaults.logprob_threshold
    )

    stored_preset = stored.value("cleanup_preset", None)
    if stored_preset is None:
        # Settings written before presets existed. An exact match becomes that
        # preset; every other combination becomes Custom and is kept as is.
        preset = preset_for_values(hallucination, no_speech, logprob)
    else:
        preset = _as_enum(CleanupPreset, stored_preset, defaults.cleanup_preset)

    # A named preset owns its numbers, so it also repairs a stored combination
    # that drifted. Custom keeps whatever the user set.
    chosen = preset_values(preset)
    if chosen is not None:
        hallucination = chosen["hallucination_silence_threshold"]
        no_speech = chosen["no_speech_threshold"]
        logprob = chosen["logprob_threshold"]

    speaker_values = {
        "clustering_threshold": _as_float(
            stored.value("clustering_threshold"), defaults.clustering_threshold
        ),
        "min_duration_on": _as_float(
            stored.value("min_duration_on"), defaults.min_duration_on
        ),
        "min_duration_off": _as_float(
            stored.value("min_duration_off"), defaults.min_duration_off
        ),
        "nearest_tolerance": _as_float(
            stored.value("nearest_tolerance"), defaults.nearest_tolerance
        ),
        "merge_gap": _as_float(stored.value("merge_gap"), defaults.merge_gap),
    }
    stored_speaker_preset = stored.value("speaker_preset", None)
    if stored_speaker_preset is None:
        speaker_preset = speaker_preset_for_values(**speaker_values)
    else:
        speaker_preset = _as_enum(
            SpeakerPreset, stored_speaker_preset, defaults.speaker_preset
        )
    named_speaker_values = speaker_preset_values(speaker_preset)
    if named_speaker_values is not None:
        speaker_values = named_speaker_values

    return AppSettings(
        source_folder=str(stored.value("source_folder", defaults.source_folder)),
        output_parent=str(stored.value("output_parent", defaults.output_parent)),
        model=model,
        language=language,
        existing_policy=policy,
        cleanup_preset=preset,
        hallucination_silence_threshold=hallucination,
        no_speech_threshold=no_speech,
        logprob_threshold=logprob,
        folders_expanded=_as_bool(
            stored.value("folders_expanded"), defaults.folders_expanded
        ),
        transcription_expanded=_as_bool(
            stored.value("transcription_expanded"),
            defaults.transcription_expanded,
        ),
        speaker_expanded=_as_bool(
            stored.value("speaker_expanded"), defaults.speaker_expanded
        ),
        advanced_expanded=_as_bool(
            stored.value("advanced_expanded"), defaults.advanced_expanded
        ),
        queue_expanded=_as_bool(
            stored.value("queue_expanded"), defaults.queue_expanded
        ),
        output_expanded=_as_bool(
            stored.value("output_expanded"), defaults.output_expanded
        ),
        selected_section=str(
            stored.value("selected_section", defaults.selected_section)
        ),
        name_style=_as_enum(
            NameStyle, stored.value("name_style"), defaults.name_style
        ),
        custom_suffix=str(stored.value("custom_suffix", defaults.custom_suffix)),
        folder_layout=_as_enum(
            FolderLayout, stored.value("folder_layout"), defaults.folder_layout
        ),
        output_scriptsync=_as_bool(
            stored.value("output_scriptsync"), defaults.output_scriptsync
        ),
        output_timecoded=_as_bool(
            stored.value("output_timecoded"), defaults.output_timecoded
        ),
        output_srt=_as_bool(stored.value("output_srt"), defaults.output_srt),
        output_vtt=_as_bool(stored.value("output_vtt"), defaults.output_vtt),
        detect_speakers=_as_bool(
            stored.value("detect_speakers"), defaults.detect_speakers
        ),
        speaker_preset=speaker_preset,
        speaker_advanced_expanded=_as_bool(
            stored.value("speaker_advanced_expanded"),
            defaults.speaker_advanced_expanded,
        ),
        speaker_count_mode=_as_enum(
            SpeakerCountMode,
            stored.value("speaker_count_mode"),
            defaults.speaker_count_mode,
        ),
        exact_speakers=_as_int(
            stored.value("exact_speakers"), defaults.exact_speakers, minimum=1
        ),
        **speaker_values,
        speakers_in_scriptsync=_as_bool(
            stored.value("speakers_in_scriptsync"), defaults.speakers_in_scriptsync
        ),
        speakers_in_subtitles=_as_bool(
            stored.value("speakers_in_subtitles"), defaults.speakers_in_subtitles
        ),
        review_mode=_as_enum(
            ReviewMode, stored.value("review_mode"), defaults.review_mode
        ),
    )


def save_settings(settings: AppSettings) -> None:
    """Write the settings back to disk."""
    stored = _qsettings()
    for key, value in asdict(settings).items():
        if isinstance(value, Enum):
            value = value.value
        stored.setValue(key, value)
    stored.sync()


def _preset_payload(settings: AppSettings) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in asdict(settings).items():
        if key not in PRESET_EXCLUDED_FIELDS:
            payload[key] = value.value if isinstance(value, Enum) else value
    return payload


def _read_user_presets() -> dict[str, dict[str, object]]:
    raw = _qsettings().value(USER_PRESETS_KEY, "{}")
    try:
        decoded = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {
        str(name): value for name, value in decoded.items()
        if str(name).strip() and isinstance(value, dict)
    }


def list_user_presets() -> tuple[str, ...]:
    return tuple(sorted(_read_user_presets(), key=str.casefold))


def save_user_preset(name: str, settings: AppSettings) -> None:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Preset name cannot be empty")
    presets = _read_user_presets()
    presets[cleaned] = _preset_payload(settings)
    stored = _qsettings()
    stored.setValue(USER_PRESETS_KEY, json.dumps(presets, sort_keys=True))
    stored.sync()


def load_user_preset(name: str, current: AppSettings) -> AppSettings | None:
    payload = _read_user_presets().get(name)
    if payload is None:
        return None
    valid = {item.name for item in fields(AppSettings)}
    changes: dict[str, object] = {}
    for key, value in payload.items():
        if key not in valid or key in PRESET_EXCLUDED_FIELDS:
            continue
        enum_type = _ENUM_FIELDS.get(key)
        if enum_type is not None:
            try:
                value = enum_type(value)
            except (TypeError, ValueError):
                continue
        changes[key] = value
    return replace(current, **changes)


def delete_user_preset(name: str) -> bool:
    presets = _read_user_presets()
    if name not in presets:
        return False
    del presets[name]
    stored = _qsettings()
    stored.setValue(USER_PRESETS_KEY, json.dumps(presets, sort_keys=True))
    stored.sync()
    return True
