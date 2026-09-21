"""Plain-language presets for speaker diarization."""

from __future__ import annotations

import math
from enum import Enum

__all__ = ["SpeakerPreset", "PRESET_VALUES", "preset_values", "preset_for_values"]


class SpeakerPreset(str, Enum):
    BALANCED = "balanced"
    FEWER_SPEAKERS = "fewer_speakers"
    MORE_SEPARATION = "more_separation"
    FAST_CONVERSATION = "fast_conversation"
    CUSTOM = "custom"

    @property
    def label(self) -> str:
        return {
            self.BALANCED: "Balanced — Recommended",
            self.FEWER_SPEAKERS: "Fewer speakers",
            self.MORE_SEPARATION: "More separation",
            self.FAST_CONVERSATION: "Fast conversation",
            self.CUSTOM: "Custom",
        }[self]

    @property
    def description(self) -> str:
        return {
            self.BALANCED: "A good starting point for interviews and general dialogue.",
            self.FEWER_SPEAKERS: "Combines similar voices and ignores very short speaker changes.",
            self.MORE_SEPARATION: "Separates similar voices more readily and may find more speakers.",
            self.FAST_CONVERSATION: "Keeps short replies and rapid changes between speakers.",
            self.CUSTOM: "Uses your values under Advanced speaker settings.",
        }[self]


PRESET_VALUES: dict[SpeakerPreset, dict[str, float]] = {
    SpeakerPreset.BALANCED: {
        "clustering_threshold": 0.50,
        "min_duration_on": 0.30,
        "min_duration_off": 0.50,
        "nearest_tolerance": 0.75,
        "merge_gap": 1.00,
    },
    SpeakerPreset.FEWER_SPEAKERS: {
        "clustering_threshold": 0.65,
        "min_duration_on": 0.50,
        "min_duration_off": 0.70,
        "nearest_tolerance": 0.85,
        "merge_gap": 1.40,
    },
    SpeakerPreset.MORE_SEPARATION: {
        "clustering_threshold": 0.35,
        "min_duration_on": 0.20,
        "min_duration_off": 0.25,
        "nearest_tolerance": 0.50,
        "merge_gap": 0.60,
    },
    SpeakerPreset.FAST_CONVERSATION: {
        "clustering_threshold": 0.45,
        "min_duration_on": 0.15,
        "min_duration_off": 0.20,
        "nearest_tolerance": 0.40,
        "merge_gap": 0.35,
    },
}


def preset_values(preset: SpeakerPreset) -> dict[str, float] | None:
    values = PRESET_VALUES.get(preset)
    return dict(values) if values is not None else None


def preset_for_values(**values: float) -> SpeakerPreset:
    for preset, expected in PRESET_VALUES.items():
        if all(
            math.isclose(float(values.get(key, float("nan"))), value, abs_tol=1e-6)
            for key, value in expected.items()
        ):
            return preset
    return SpeakerPreset.CUSTOM
