"""Non-drop-frame and drop-frame timecode conversion."""

from __future__ import annotations

from fractions import Fraction

import pytest

from transcription.timecode import (
    TimecodeConverter,
    TimecodeError,
    nominal_frame_rate,
    parse_timecode,
)

NTSC_30 = Fraction(30000, 1001)
NTSC_24 = Fraction(24000, 1001)


def test_parse_non_drop_timecode():
    parsed = parse_timecode("01:02:03:04")
    assert (parsed.hours, parsed.minutes, parsed.seconds, parsed.frames) == (1, 2, 3, 4)
    assert parsed.drop_frame is False
    assert parsed.separator == ":"


def test_parse_drop_frame_timecode():
    parsed = parse_timecode("00:01:00;02")
    assert parsed.drop_frame is True
    assert parsed.separator == ";"
    assert parsed.frames == 2


@pytest.mark.parametrize("bad", ["", "01:02:03", "not a timecode", "01-02-03-04"])
def test_parse_rejects_unusable_timecode(bad):
    with pytest.raises(TimecodeError):
        parse_timecode(bad)


@pytest.mark.parametrize(
    "rate, expected",
    [
        (Fraction(24, 1), 24),
        (NTSC_24, 24),
        (Fraction(25, 1), 25),
        (NTSC_30, 30),
        (Fraction(60, 1), 60),
    ],
)
def test_nominal_frame_rate_rounds_to_the_counting_rate(rate, expected):
    assert nominal_frame_rate(rate) == expected


def test_non_drop_start_is_returned_unchanged():
    convert = TimecodeConverter(Fraction(24, 1), "01:00:00:00")
    assert convert(0) == "01:00:00:00"


def test_non_drop_offsets_advance_by_frames():
    convert = TimecodeConverter(Fraction(24, 1), "01:00:00:00")
    assert convert(1.0) == "01:00:01:00"
    assert convert(0.5) == "01:00:00:12"
    assert convert(60.0) == "01:01:00:00"


def test_non_drop_2997_counts_thirty_frames_per_second():
    convert = TimecodeConverter(NTSC_30, "00:00:00:00")
    assert convert(0) == "00:00:00:00"
    assert convert(1.0) == "00:00:01:00"
    assert ";" not in convert(1.0)


def test_drop_frame_start_round_trips():
    convert = TimecodeConverter(NTSC_30, "00:01:00;02")
    assert convert(0) == "00:01:00;02"


def test_drop_frame_skips_two_frames_at_the_minute():
    convert = TimecodeConverter(NTSC_30, "00:00:59;29")
    assert convert(1 / float(NTSC_30)) == "00:01:00;02"


def test_drop_frame_keeps_frames_at_the_tenth_minute():
    convert = TimecodeConverter(NTSC_30, "00:00:00;00")
    # Ten minutes of real time is 17982 frames, and drop-frame counting adds
    # the 18 frames it skipped over the nine preceding minutes back in.
    assert convert.frames_at(600.0) == 17982
    assert convert(600.0) == "00:10:00;00"


def test_drop_frame_and_non_drop_use_different_separators():
    drop = TimecodeConverter(NTSC_30, "00:00:00;00")
    non_drop = TimecodeConverter(NTSC_30, "00:00:00:00")
    assert drop(0).count(";") == 1
    assert non_drop(0).count(";") == 0


def test_timecode_wraps_at_twenty_four_hours():
    convert = TimecodeConverter(Fraction(25, 1), "23:59:59:24")
    assert convert(1 / 25) == "00:00:00:00"


def test_frames_at_matches_the_real_frame_rate():
    convert = TimecodeConverter(NTSC_24, "00:00:00:00")
    assert convert.frames_at(0) == 0
    assert convert.frames_at(1.0) == 24
    assert convert.frames_at(10.0) == 240


def test_empty_timecode_falls_back_to_zero():
    convert = TimecodeConverter(Fraction(25, 1), "")
    assert convert(0) == "00:00:00:00"


def test_unusable_frame_rate_is_rejected():
    with pytest.raises(TimecodeError):
        TimecodeConverter(Fraction(0, 1), "00:00:00:00")
