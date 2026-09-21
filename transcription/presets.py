"""Transcription cleanup presets.

A preset is nothing more than a named set of the three MLX Whisper decoding
thresholds the application already passes. There is no second pass, no text
rewriting, and no effect on speaker detection: picking a preset only fills in
``hallucination_silence_threshold``, ``no_speech_threshold``, and
``logprob_threshold``.

The fixed safeguards are not part of any preset and never change:
``temperature=0.0``, ``condition_on_previous_text=False``, and
``word_timestamps=True``.
"""

from __future__ import annotations

import math
from enum import Enum

__all__ = [
    "CleanupPreset",
    "BALANCED",
    "REDUCE_HALLUCINATIONS",
    "PRESERVE_DIFFICULT_SPEECH",
    "PRESET_VALUES",
    "THRESHOLD_KEYS",
    "preset_values",
    "preset_for_values",
    "matches_preset",
]

#: The three keyword arguments a preset sets, in the order the interface shows.
THRESHOLD_KEYS = (
    "hallucination_silence_threshold",
    "no_speech_threshold",
    "logprob_threshold",
)

BALANCED = {
    "hallucination_silence_threshold": 1.0,
    "no_speech_threshold": 0.60,
    "logprob_threshold": -1.0,
}

REDUCE_HALLUCINATIONS = {
    "hallucination_silence_threshold": 0.5,
    "no_speech_threshold": 0.50,
    "logprob_threshold": -0.8,
}

PRESERVE_DIFFICULT_SPEECH = {
    "hallucination_silence_threshold": 1.5,
    "no_speech_threshold": 0.70,
    "logprob_threshold": -1.2,
}

#: Stored values are read back as text, so comparison allows for rounding.
_TOLERANCE = 1e-6


class CleanupPreset(str, Enum):
    """How aggressively Whisper should discard uncertain text."""

    BALANCED = "balanced"
    REDUCE_HALLUCINATIONS = "reduce_hallucinations"
    PRESERVE_DIFFICULT_SPEECH = "preserve_difficult_speech"
    CUSTOM = "custom"

    @property
    def label(self) -> str:
        return {
            CleanupPreset.BALANCED: "Balanced — Recommended",
            CleanupPreset.REDUCE_HALLUCINATIONS: "Reduce repeated or invented speech",
            CleanupPreset.PRESERVE_DIFFICULT_SPEECH: "Preserve quiet or difficult speech",
            CleanupPreset.CUSTOM: "Custom",
        }[self]

    @property
    def description(self) -> str:
        return {
            CleanupPreset.BALANCED: (
                "Recommended for most interviews and spoken dialogue."
            ),
            CleanupPreset.REDUCE_HALLUCINATIONS: (
                "Rejects more uncertain text around silence, noise, and music."
            ),
            CleanupPreset.PRESERVE_DIFFICULT_SPEECH: (
                "Keeps more low-confidence dialogue but may also retain mistakes."
            ),
            CleanupPreset.CUSTOM: (
                "Uses the detailed controls under Advanced Settings."
            ),
        }[self]

    @property
    def is_named(self) -> bool:
        """True for everything except Custom, which has no fixed values."""
        return self is not CleanupPreset.CUSTOM


#: Every named preset's values, keyed by preset.
PRESET_VALUES: dict[CleanupPreset, dict[str, float]] = {
    CleanupPreset.BALANCED: BALANCED,
    CleanupPreset.REDUCE_HALLUCINATIONS: REDUCE_HALLUCINATIONS,
    CleanupPreset.PRESERVE_DIFFICULT_SPEECH: PRESERVE_DIFFICULT_SPEECH,
}


def preset_values(preset: CleanupPreset) -> dict[str, float] | None:
    """Return a named preset's thresholds, or None for Custom."""
    values = PRESET_VALUES.get(preset)
    return dict(values) if values is not None else None


def matches_preset(
    preset: CleanupPreset,
    hallucination_silence_threshold: float,
    no_speech_threshold: float,
    logprob_threshold: float,
) -> bool:
    """True when these three values are exactly a named preset's values."""
    values = PRESET_VALUES.get(preset)
    if values is None:
        return False
    supplied = {
        "hallucination_silence_threshold": hallucination_silence_threshold,
        "no_speech_threshold": no_speech_threshold,
        "logprob_threshold": logprob_threshold,
    }
    return all(
        math.isclose(float(supplied[key]), values[key], abs_tol=_TOLERANCE)
        for key in THRESHOLD_KEYS
    )


def preset_for_values(
    hallucination_silence_threshold: float,
    no_speech_threshold: float,
    logprob_threshold: float,
) -> CleanupPreset:
    """Return the preset these values represent, or Custom.

    This is what migrates a stored threshold combination from before presets
    existed: an exact match becomes that preset, anything else becomes Custom.
    """
    for preset in PRESET_VALUES:
        if matches_preset(
            preset,
            hallucination_silence_threshold,
            no_speech_threshold,
            logprob_threshold,
        ):
            return preset
    return CleanupPreset.CUSTOM
