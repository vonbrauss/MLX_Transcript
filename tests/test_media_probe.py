"""ffprobe payload parsing and duration formatting."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from transcription.media_probe import (
    DEFAULT_AUDIO_FRAME_RATE,
    format_duration,
    parse_probe_payload,
)

CLIP = Path("/Source/Day 01/A001.mov")


def test_video_payload_reports_rate_timecode_and_duration():
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
                "tags": {"timecode": "01:00:00;00"},
            },
            {"codec_type": "audio", "avg_frame_rate": "0/0"},
        ],
        "format": {"duration": "126.5"},
    }
    info = parse_probe_payload(payload, CLIP)
    assert info.frame_rate == Fraction(30000, 1001)
    assert info.start_timecode == "01:00:00;00"
    assert info.has_embedded_timecode is True
    assert info.has_video is True
    assert info.duration_seconds == pytest.approx(126.5)
    assert info.duration_label == "00:02:06"


def test_format_level_timecode_is_used_when_no_stream_carries_one():
    payload = {
        "streams": [{"codec_type": "video", "avg_frame_rate": "24/1"}],
        "format": {"duration": "10", "tags": {"timecode": "09:59:50:00"}},
    }
    info = parse_probe_payload(payload, CLIP)
    assert info.start_timecode == "09:59:50:00"
    assert info.has_embedded_timecode is True


def test_missing_timecode_falls_back_to_zero():
    payload = {
        "streams": [{"codec_type": "video", "avg_frame_rate": "25/1"}],
        "format": {"duration": "4"},
    }
    info = parse_probe_payload(payload, CLIP)
    assert info.start_timecode == "00:00:00:00"
    assert info.missing_timecode is True


def test_audio_only_payload_uses_the_fallback_frame_rate():
    payload = {
        "streams": [{"codec_type": "audio", "duration": "61.0"}],
        "format": {"duration": "61.0"},
    }
    info = parse_probe_payload(payload, Path("/Source/Interview.wav"))
    assert info.has_video is False
    assert info.frame_rate == DEFAULT_AUDIO_FRAME_RATE
    assert info.duration_label == "00:01:01"


def test_unusable_frame_rate_falls_back_to_r_frame_rate():
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": "0/0",
                "r_frame_rate": "24000/1001",
            }
        ],
        "format": {"duration": "8"},
    }
    info = parse_probe_payload(payload, CLIP)
    assert info.frame_rate == Fraction(24000, 1001)


def test_stream_duration_is_used_when_the_format_has_none():
    payload = {
        "streams": [
            {"codec_type": "video", "avg_frame_rate": "24/1", "duration": "42.0"}
        ],
        "format": {},
    }
    info = parse_probe_payload(payload, CLIP)
    assert info.duration_seconds == pytest.approx(42.0)


def test_unknown_duration_is_reported_clearly():
    payload = {"streams": [], "format": {}}
    info = parse_probe_payload(payload, CLIP)
    assert info.duration_seconds is None
    assert info.duration_label == "--:--:--"


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "00:00:00"),
        (59.4, "00:00:59"),
        (3600, "01:00:00"),
        (3661, "01:01:01"),
        (None, "--:--:--"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected
