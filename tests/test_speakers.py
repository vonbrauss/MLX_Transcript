"""Speaker data model: naming, merging, reassignment, and export rows."""

from __future__ import annotations

import pytest

from transcription.speakers import (
    AttributedSegment,
    Speaker,
    SpeakerNameError,
    SpeakerTranscript,
    SpeakerTurn,
    Word,
    clean_speaker_name,
    default_display_name,
    scriptsync_speaker_label,
)


def transcript() -> SpeakerTranscript:
    return SpeakerTranscript.from_segments(
        [
            AttributedSegment(0.0, 4.0, "Welcome, everyone.", "speaker_00"),
            AttributedSegment(4.5, 8.0, "Can you discuss the fourth quarter?", "speaker_01"),
            AttributedSegment(8.5, 14.0, "We defended with more discipline.", "speaker_00"),
        ]
    )


# ------------------------------------------------------------------- naming


def test_speakers_are_numbered_in_order_of_first_appearance():
    built = transcript()
    assert [speaker.display_name for speaker in built.speakers] == [
        "Speaker 1",
        "Speaker 2",
    ]
    assert built.identifiers == ["speaker_00", "speaker_01"]


def test_default_display_name_is_one_based():
    assert default_display_name(0) == "Speaker 1"
    assert default_display_name(2) == "Speaker 3"


def test_names_are_trimmed():
    assert clean_speaker_name("  Coach Smith  ") == "Coach Smith"


def test_line_breaks_and_control_characters_are_removed():
    assert clean_speaker_name("Coach\nSmith") == "Coach Smith"
    assert clean_speaker_name("Coach\r\n\tSmith\x07") == "Coach Smith"
    assert clean_speaker_name("Coach Smith") == "Coach Smith"


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t\r\n", "\x00"])
def test_blank_names_are_rejected(blank):
    with pytest.raises(SpeakerNameError):
        clean_speaker_name(blank)


def test_absurdly_long_names_are_rejected():
    with pytest.raises(SpeakerNameError):
        clean_speaker_name("a" * 500)


def test_unicode_names_are_preserved():
    assert clean_speaker_name("  Renée Müller ") == "Renée Müller"
    assert clean_speaker_name("宮崎 駿") == "宮崎 駿"


def test_renaming_a_speaker_updates_the_export_rows():
    built = transcript()
    built.rename("speaker_00", "Coach Smith")
    built.rename("speaker_01", "Reporter")
    rows = built.as_output_segments()
    assert [row["speaker"] for row in rows] == ["Coach Smith", "Reporter", "Coach Smith"]


def test_renaming_an_unknown_speaker_raises():
    with pytest.raises(SpeakerNameError):
        transcript().rename("speaker_99", "Nobody")


def test_duplicate_names_are_refused_by_default():
    built = transcript()
    built.rename("speaker_00", "Coach Smith")
    with pytest.raises(SpeakerNameError) as error:
        built.rename("speaker_01", "Coach Smith")
    assert "already named" in str(error.value)
    assert built.display_name("speaker_01") == "Speaker 2"


def test_duplicate_names_are_allowed_when_confirmed():
    built = transcript()
    built.rename("speaker_00", "Coach Smith")
    built.rename("speaker_01", "Coach Smith", allow_duplicate=True)
    assert built.display_name("speaker_01") == "Coach Smith"


def test_duplicate_detection_ignores_case_and_spacing():
    built = transcript()
    built.rename("speaker_00", "Coach Smith")
    assert built.duplicates_of("  coach smith ") == ["speaker_00"]


def test_reset_names_restores_the_numbering():
    built = transcript()
    built.rename("speaker_00", "Coach Smith")
    built.rename("speaker_01", "Reporter")
    built.reset_names()
    assert [speaker.display_name for speaker in built.speakers] == [
        "Speaker 1",
        "Speaker 2",
    ]


# ----------------------------------------------------------- scriptsync labels


def test_scriptsync_labels_are_uppercase_ascii():
    assert scriptsync_speaker_label("Coach Smith") == "COACH SMITH"
    assert scriptsync_speaker_label("Renée Müller") == "RENEE MULLER"


def test_scriptsync_labels_drop_hyphens_like_the_body_text():
    assert scriptsync_speaker_label("Anne-Marie") == "ANNE MARIE"
    assert "-" not in scriptsync_speaker_label("co-host")


def test_a_label_with_no_ascii_left_comes_back_empty():
    assert scriptsync_speaker_label("宮崎 駿") == ""


# ------------------------------------------------------------------- merging


def test_merging_moves_every_segment_to_the_target():
    built = transcript()
    built.rename("speaker_00", "Coach Smith")
    built.merge("speaker_01", "speaker_00")

    assert built.identifiers == ["speaker_00"]
    assert all(segment.speaker_id == "speaker_00" for segment in built.segments)
    assert {row["speaker"] for row in built.as_output_segments()} == {"Coach Smith"}


def test_merging_keeps_the_target_name():
    built = transcript()
    built.rename("speaker_01", "Reporter")
    built.merge("speaker_00", "speaker_01")
    assert built.display_name("speaker_01") == "Reporter"


def test_merging_a_speaker_into_itself_does_nothing():
    built = transcript()
    built.merge("speaker_00", "speaker_00")
    assert len(built.speakers) == 2


def test_merging_an_unknown_speaker_raises():
    with pytest.raises(SpeakerNameError):
        transcript().merge("speaker_00", "speaker_99")


def test_merging_rewrites_overlap_identifiers_without_duplicates():
    built = SpeakerTranscript.from_segments(
        [
            AttributedSegment(
                0.0, 2.0, "Crosstalk.", "speaker_00",
                overlap_identifiers=("speaker_00", "speaker_01"),
            ),
            AttributedSegment(2.0, 4.0, "Go ahead.", "speaker_01"),
        ]
    )
    built.merge("speaker_01", "speaker_00")
    assert built.segments[0].overlap_identifiers == ("speaker_00",)


# --------------------------------------------------------------- reassigning


def test_reassigning_moves_one_segment():
    built = transcript()
    built.reassign(1, "speaker_00")
    assert [segment.speaker_id for segment in built.segments] == [
        "speaker_00",
        "speaker_00",
        "speaker_00",
    ]


def test_reassigning_to_nothing_leaves_the_segment_unlabeled():
    built = transcript()
    built.reassign(0, None)
    rows = built.as_output_segments()
    assert "speaker" not in rows[0]


def test_reassigning_to_an_unknown_speaker_raises():
    with pytest.raises(SpeakerNameError):
        transcript().reassign(0, "speaker_99")


def test_reassigning_out_of_range_raises():
    with pytest.raises(IndexError):
        transcript().reassign(99, "speaker_00")


# ------------------------------------------------------------------ review


def test_speaking_duration_and_segment_counts():
    built = transcript()
    assert built.segment_count("speaker_00") == 2
    assert built.speaking_duration("speaker_00") == pytest.approx(9.5)
    assert built.segment_count("speaker_01") == 1


def test_samples_are_short_and_limited():
    built = SpeakerTranscript.from_segments(
        [
            AttributedSegment(0.0, 1.0, "word " * 60, "speaker_00"),
            AttributedSegment(2.0, 3.0, "second", "speaker_00"),
            AttributedSegment(4.0, 5.0, "third", "speaker_00"),
            AttributedSegment(6.0, 7.0, "fourth", "speaker_00"),
        ]
    )
    samples = built.samples("speaker_00", count=3, length=40)
    assert len(samples) == 3
    assert all(len(sample) <= 40 for sample in samples)
    assert samples[0].endswith("…")


def test_unattributed_segments_are_counted():
    built = SpeakerTranscript.from_segments(
        [
            AttributedSegment(0.0, 1.0, "Hello.", "speaker_00"),
            AttributedSegment(2.0, 3.0, "(music)", None),
        ]
    )
    assert built.unattributed_count == 1


def test_export_rows_can_omit_speakers_entirely():
    rows = transcript().as_output_segments(include_speakers=False)
    assert all("speaker" not in row for row in rows)


def test_export_rows_keep_wording_untouched():
    built = transcript()
    assert [row["text"] for row in built.as_output_segments()] == [
        "Welcome, everyone.",
        "Can you discuss the fourth quarter?",
        "We defended with more discipline.",
    ]


# ------------------------------------------------------------- small helpers


def test_speaker_turn_overlap_and_distance():
    turn = SpeakerTurn(2.0, 5.0, "speaker_00")
    assert turn.duration == pytest.approx(3.0)
    assert turn.overlap_with(4.0, 6.0) == pytest.approx(1.0)
    assert turn.overlap_with(6.0, 7.0) == 0.0
    assert turn.distance_to(6.0, 7.0) == pytest.approx(1.0)
    assert turn.distance_to(0.0, 1.0) == pytest.approx(1.0)
    assert turn.distance_to(3.0, 4.0) == 0.0


def test_word_from_payload_handles_both_key_names():
    assert Word.from_payload({"word": " Hello ", "start": 0.0, "end": 1.0}).text == "Hello"
    assert Word.from_payload({"text": "Hi", "start": 0.0, "end": 1.0}).text == "Hi"


@pytest.mark.parametrize(
    "payload",
    [
        {"word": "Hello"},
        {"word": "", "start": 0.0, "end": 1.0},
        {"word": "Hello", "start": None, "end": 1.0},
        {"word": "Hello", "start": "x", "end": 1.0},
    ],
)
def test_unusable_word_payloads_are_dropped(payload):
    assert Word.from_payload(payload) is None


def test_speaker_numbered_helper():
    assert Speaker.numbered("speaker_04", 3).display_name == "Speaker 4"
