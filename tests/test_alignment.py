"""Word-to-speaker alignment: overlap, tolerance, splitting, and merging."""

from __future__ import annotations

import pytest

from transcription.alignment import (
    MERGE_GAP_SECONDS,
    NEAREST_SPEAKER_TOLERANCE,
    assign_word,
    attribute_transcript,
    segment_words,
)
from transcription.speakers import SpeakerTurn, Word


def word(start: float, end: float, text: str = "word") -> Word:
    return Word(start=start, end=end, text=text)


def whisper_segment(start: float, end: float, words: list[tuple[float, float, str]]):
    return {
        "start": start,
        "end": end,
        "text": " ".join(text for _s, _e, text in words),
        "words": [
            {"word": text, "start": begin, "end": finish}
            for begin, finish, text in words
        ],
    }


TWO_SPEAKERS = [
    SpeakerTurn(0.0, 5.0, "speaker_00"),
    SpeakerTurn(5.0, 10.0, "speaker_01"),
]


# -------------------------------------------------------------- word pulling


def test_words_come_from_the_word_list_when_present():
    segment = whisper_segment(0.0, 2.0, [(0.0, 0.5, "Hello"), (0.6, 1.2, "there")])
    assert [item.text for item in segment_words(segment)] == ["Hello", "there"]


def test_a_segment_without_word_timestamps_becomes_one_span():
    segment = {"start": 3.0, "end": 9.0, "text": "  Hello   there  "}
    words = segment_words(segment)
    assert len(words) == 1
    assert words[0].text == "Hello there"
    assert (words[0].start, words[0].end) == (3.0, 9.0)


def test_a_segment_with_unusable_words_falls_back_to_the_whole_span():
    segment = {
        "start": 1.0,
        "end": 2.0,
        "text": "Hello",
        "words": [{"word": "Hello"}],
    }
    assert len(segment_words(segment)) == 1


def test_an_empty_segment_yields_nothing():
    assert segment_words({"start": 0.0, "end": 1.0, "text": "   "}) == []


# ----------------------------------------------------------------- assignment


def test_a_word_goes_to_the_speaker_it_overlaps_most():
    # 4.0 to 6.0 straddles the boundary: 1.0s with speaker_00, 1.0s with 01.
    # 4.0 to 5.5 is mostly speaker_00.
    speaker, _overlaps = assign_word(word(4.0, 5.5), TWO_SPEAKERS)
    assert speaker == "speaker_00"
    speaker, _overlaps = assign_word(word(4.8, 7.0), TWO_SPEAKERS)
    assert speaker == "speaker_01"


def test_an_exact_tie_goes_to_the_earlier_turn():
    speaker, _ = assign_word(word(4.0, 6.0), TWO_SPEAKERS)
    assert speaker == "speaker_00"


def test_every_overlapping_speaker_is_recorded_primary_first():
    _speaker, overlaps = assign_word(word(4.8, 7.0), TWO_SPEAKERS)
    assert overlaps[0] == "speaker_01"
    assert set(overlaps) == {"speaker_00", "speaker_01"}


def test_a_word_with_no_overlap_borrows_the_nearest_speaker():
    turns = [SpeakerTurn(0.0, 2.0, "speaker_00")]
    speaker, _ = assign_word(word(2.3, 2.5), turns)
    assert speaker == "speaker_00"


def test_a_word_beyond_the_tolerance_stays_unattributed():
    turns = [SpeakerTurn(0.0, 2.0, "speaker_00")]
    speaker, overlaps = assign_word(word(20.0, 20.5), turns)
    assert speaker is None
    assert overlaps == ()


def test_the_tolerance_is_configurable():
    turns = [SpeakerTurn(0.0, 2.0, "speaker_00")]
    assert assign_word(word(4.0, 4.2), turns)[0] is None
    assert assign_word(word(4.0, 4.2), turns, tolerance=5.0)[0] == "speaker_00"


def test_the_documented_default_tolerance():
    assert NEAREST_SPEAKER_TOLERANCE == 0.75


def test_no_turns_means_no_speaker():
    assert assign_word(word(0.0, 1.0), []) == (None, ())


# ------------------------------------------------------- splitting and merging


def test_a_segment_is_split_where_its_words_change_speaker():
    segment = whisper_segment(
        4.0,
        6.0,
        [(4.0, 4.4, "Coach"), (4.5, 4.9, "says"), (5.2, 5.6, "Reporter"), (5.7, 6.0, "asks")],
    )
    built = attribute_transcript([segment], TWO_SPEAKERS)

    assert [item.speaker_id for item in built.segments] == ["speaker_00", "speaker_01"]
    assert built.segments[0].text == "Coach says"
    assert built.segments[1].text == "Reporter asks"


def test_a_long_segment_is_not_handed_to_one_speaker():
    segment = whisper_segment(
        0.0,
        10.0,
        [(0.5, 1.0, "one"), (2.0, 2.5, "two"), (6.0, 6.5, "three"), (8.0, 8.5, "four")],
    )
    built = attribute_transcript([segment], TWO_SPEAKERS)
    assert len({item.speaker_id for item in built.segments}) == 2


def test_same_speaker_words_merge_into_one_segment():
    segment = whisper_segment(
        0.0, 3.0, [(0.0, 0.4, "one"), (0.5, 0.9, "two"), (1.0, 1.4, "three")]
    )
    built = attribute_transcript([segment], TWO_SPEAKERS)
    assert len(built.segments) == 1
    assert built.segments[0].text == "one two three"


def test_a_long_silence_splits_even_for_the_same_speaker():
    segment = whisper_segment(0.0, 5.0, [(0.0, 0.4, "one"), (3.0, 3.4, "two")])
    built = attribute_transcript([segment], TWO_SPEAKERS)
    assert len(built.segments) == 2
    assert built.segments[0].text == "one"


def test_the_merge_gap_is_configurable():
    segment = whisper_segment(0.0, 5.0, [(0.0, 0.4, "one"), (1.8, 2.2, "two")])
    assert len(attribute_transcript([segment], TWO_SPEAKERS).segments) == 2
    assert len(
        attribute_transcript([segment], TWO_SPEAKERS, merge_gap=5.0).segments
    ) == 1


def test_the_documented_default_merge_gap():
    assert MERGE_GAP_SECONDS == 1.0


def test_segments_merge_across_whisper_segment_boundaries():
    first = whisper_segment(0.0, 1.0, [(0.0, 0.4, "one")])
    second = whisper_segment(1.0, 2.0, [(0.5, 0.9, "two")])
    built = attribute_transcript([first, second], TWO_SPEAKERS)
    assert len(built.segments) == 1
    assert built.segments[0].text == "one two"


def test_wording_is_never_altered_by_grouping():
    segment = whisper_segment(
        0.0, 3.0, [(0.0, 0.4, "Um,"), (0.5, 0.9, "we"), (1.0, 1.4, "we")]
    )
    built = attribute_transcript([segment], TWO_SPEAKERS)
    assert built.segments[0].text == "Um, we we"


# ----------------------------------------------------------------- edge cases


def test_an_empty_diarization_result_leaves_everything_unattributed():
    segment = whisper_segment(0.0, 2.0, [(0.0, 0.5, "Hello")])
    built = attribute_transcript([segment], [])
    assert built.has_speakers is False
    assert built.segments[0].speaker_id is None
    assert built.segments[0].text == "Hello"


def test_one_detected_speaker_labels_everything():
    turns = [SpeakerTurn(0.0, 30.0, "speaker_00")]
    segment = whisper_segment(0.0, 3.0, [(0.0, 0.5, "Hello"), (1.0, 1.5, "again")])
    built = attribute_transcript([segment], turns)
    assert len(built.speakers) == 1
    assert built.speakers[0].display_name == "Speaker 1"


def test_missing_word_timestamps_still_get_a_speaker():
    segment = {"start": 0.5, "end": 4.0, "text": "No word timings here"}
    built = attribute_transcript([segment], TWO_SPEAKERS)
    assert built.segments[0].speaker_id == "speaker_00"
    assert built.segments[0].text == "No word timings here"


def test_a_short_interjection_between_turns_is_attributed():
    turns = [
        SpeakerTurn(0.0, 3.0, "speaker_00"),
        SpeakerTurn(3.4, 8.0, "speaker_01"),
    ]
    segment = whisper_segment(3.0, 3.4, [(3.05, 3.2, "Right.")])
    built = attribute_transcript([segment], turns)
    assert built.segments[0].speaker_id == "speaker_00"


def test_silence_and_noise_far_from_any_turn_stay_unlabeled():
    turns = [SpeakerTurn(0.0, 2.0, "speaker_00")]
    segment = whisper_segment(40.0, 42.0, [(40.0, 41.0, "(music)")])
    built = attribute_transcript([segment], turns)
    assert built.segments[0].speaker_id is None


def test_overlapped_speech_keeps_a_primary_and_records_the_rest():
    turns = [
        SpeakerTurn(0.0, 6.0, "speaker_00"),
        SpeakerTurn(4.0, 10.0, "speaker_01"),
    ]
    segment = whisper_segment(4.0, 5.0, [(4.0, 4.4, "Both"), (4.5, 4.9, "talking")])
    built = attribute_transcript([segment], turns)

    only = built.segments[0]
    assert only.speaker_id == "speaker_00"
    assert only.has_overlap is True
    assert only.overlap_identifiers[0] == "speaker_00"
    assert "speaker_01" in only.overlap_identifiers


def test_out_of_order_words_start_a_new_segment():
    segment = {
        "start": 0.0,
        "end": 5.0,
        "text": "one two",
        "words": [
            {"word": "one", "start": 2.0, "end": 2.4},
            {"word": "two", "start": 0.5, "end": 0.9},
        ],
    }
    built = attribute_transcript([segment], TWO_SPEAKERS)
    assert len(built.segments) == 2


def test_turns_are_sorted_before_alignment():
    turns = [
        SpeakerTurn(5.0, 10.0, "speaker_01"),
        SpeakerTurn(0.0, 5.0, "speaker_00"),
    ]
    segment = whisper_segment(0.0, 8.0, [(0.5, 1.0, "first"), (6.0, 6.5, "second")])
    built = attribute_transcript([segment], turns)
    assert [item.speaker_id for item in built.segments] == [
        "speaker_00",
        "speaker_01",
    ]


def test_segment_boundaries_come_from_the_words():
    segment = whisper_segment(0.0, 9.0, [(1.0, 1.4, "one"), (1.5, 2.25, "two")])
    built = attribute_transcript([segment], TWO_SPEAKERS)
    assert built.segments[0].start == pytest.approx(1.0)
    assert built.segments[0].end == pytest.approx(2.25)
