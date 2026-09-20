#!/usr/bin/env python3
"""Create recursive Avid ScriptSync and source-timecoded transcript trees."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unicodedata
from fractions import Fraction
from pathlib import Path


MEDIA_EXTENSIONS = {
    ".aac", ".aif", ".aiff", ".flac", ".m4a", ".m4v", ".mkv",
    ".mov", ".mp3", ".mp4", ".mpeg", ".mpg", ".ogg", ".opus",
    ".wav", ".webm", ".wma",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe all media below INPUT_FOLDER into parallel ScriptSync "
            "and Timecoded folder trees."
        )
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        nargs="?",
        help="Source folder; omit to be prompted",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="Parent for the Transcription folder (default: INPUT_FOLDER)",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-mlx",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--hallucination-silence-threshold",
        type=float,
        default=1.0,
        help="Seconds of silence used to reject likely hallucinations (default: 1.0)",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.6,
        help="Probability above which a low-confidence segment is silence (default: 0.6)",
    )
    parser.add_argument(
        "--logprob-threshold",
        type=float,
        default=-1.0,
        help="Confidence threshold used with no-speech detection (default: -1.0)",
    )
    parser.add_argument(
        "--overwrite-all",
        "--overwrite",
        dest="overwrite_all",
        action="store_true",
        help="Replace all existing transcripts without prompting",
    )
    return parser.parse_args()


def prompted_path(prompt: str, default: Path | None = None) -> Path:
    while True:
        try:
            response = input(prompt).strip()
        except EOFError:
            response = ""
        if not response and default is not None:
            return default
        if not response:
            print("Please enter a folder path.")
            continue
        try:
            parts = shlex.split(response)
        except ValueError:
            parts = [response.strip("'\"")]
        if len(parts) != 1:
            print("Please enter one folder path.")
            continue
        return Path(parts[0]).expanduser()


def clean_scriptsync_text(segments: list[dict]) -> str:
    text = " ".join(segment.get("text", "").strip() for segment in segments)
    text = text.replace("\u2013", " ").replace("\u2014", " ").replace("-", " ")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    lines = textwrap.wrap(
        text,
        width=64,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\r\n".join(lines) + "\r\n"


def probe_media(source: Path) -> tuple[Fraction, str, bool]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries",
            "stream=codec_type,avg_frame_rate,r_frame_rate:"
            "stream_tags=timecode:format_tags=timecode",
            "-of", "json", str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    streams = metadata.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if video is None:
        raise RuntimeError("No video stream was found")

    rate_text = video.get("avg_frame_rate") or video.get("r_frame_rate")
    if not rate_text or rate_text in {"0/0", "0:0"}:
        raise RuntimeError("No usable video frame rate was found")
    rate = Fraction(rate_text)

    timecode = next(
        (
            stream.get("tags", {}).get("timecode")
            for stream in streams
            if stream.get("tags", {}).get("timecode")
        ),
        None,
    )
    if not timecode:
        timecode = metadata.get("format", {}).get("tags", {}).get("timecode")

    missing_timecode = not bool(timecode)
    return rate, timecode or "00:00:00:00", missing_timecode


def timecode_converter(rate: Fraction, source_timecode: str):
    nominal_fps = round(float(rate))
    drop_frame = ";" in source_timecode
    separator = ";" if drop_frame else ":"
    parts = source_timecode.replace(";", ":").split(":")
    if len(parts) != 4:
        raise RuntimeError(f"Unrecognized timecode: {source_timecode}")

    hours, minutes, seconds, frames = map(int, parts)
    drop_frames = round(nominal_fps * 0.066666) if drop_frame else 0
    total_minutes = hours * 60 + minutes
    start_frame = (
        (hours * 3600 + minutes * 60 + seconds) * nominal_fps + frames
    )
    if drop_frame:
        start_frame -= drop_frames * (total_minutes - total_minutes // 10)

    def convert(offset_seconds: float) -> str:
        frame_number = start_frame + round(offset_seconds * float(rate))
        if drop_frame:
            frames_per_10_minutes = round(float(rate) * 600)
            frames_per_minute = round(float(rate) * 60)
            blocks = frame_number // frames_per_10_minutes
            remainder = frame_number % frames_per_10_minutes
            frame_number += drop_frames * 9 * blocks
            if remainder > drop_frames:
                frame_number += drop_frames * (
                    (remainder - drop_frames) // frames_per_minute
                )

        frame_number %= nominal_fps * 60 * 60 * 24
        ff = frame_number % nominal_fps
        total_seconds = frame_number // nominal_fps
        ss = total_seconds % 60
        total_minutes_value = total_seconds // 60
        mm = total_minutes_value % 60
        hh = total_minutes_value // 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}{separator}{ff:02d}"

    return convert


def make_timecoded_text(
    source: Path,
    segments: list[dict],
    rate: Fraction,
    source_timecode: str,
    missing_timecode: bool,
) -> str:
    convert = timecode_converter(rate, source_timecode)
    lines = [
        f"FILE: {source.name}",
        f"START TIMECODE: {source_timecode}",
        f"FRAME RATE: {rate.numerator}/{rate.denominator}",
    ]
    if missing_timecode:
        lines.append("NOTE: No embedded timecode found; 00:00:00:00 was used.")
    lines.append("")

    for segment in segments:
        timestamp = convert(float(segment.get("start", 0)))
        spoken = " ".join(segment.get("text", "").split())
        lines.append(f"{timestamp}  {spoken}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = arguments()
    source_input = args.input_folder or prompted_path("Input media folder: ")
    source_root = source_input.expanduser().resolve()
    if args.output_folder:
        output_root = args.output_folder.expanduser().resolve()
    else:
        output_root = prompted_path(
            f"Output parent folder [{source_root}]: ", default=source_root
        ).resolve()
    transcription_root = output_root / "Transcription"
    scriptsync_root = transcription_root / "ScriptSync"
    timecoded_root = transcription_root / "Timecoded"

    if not source_root.is_dir():
        print(f"Input folder does not exist: {source_root}", file=sys.stderr)
        return 2
    for program in ("mlx_whisper", "ffprobe"):
        if shutil.which(program) is None:
            print(f"Required program was not found: {program}", file=sys.stderr)
            return 2

    media_files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in MEDIA_EXTENSIONS
        and transcription_root not in path.parents
    )
    if not media_files:
        print(f"No supported media files found below: {source_root}")
        return 0

    print(f"Found {len(media_files)} media file(s).")
    print(f"ScriptSync tree: {scriptsync_root}")
    print(f"Timecoded tree:  {timecoded_root}")

    completed = skipped = failed = 0
    overwrite_all = args.overwrite_all
    for number, source in enumerate(media_files, start=1):
        relative_parent = source.relative_to(source_root).parent
        enclosing_folder = source.parent.name
        prefix = f"{enclosing_folder} \u2014 {source.stem}"
        scriptsync_path = (
            scriptsync_root / relative_parent / f"{prefix} ScriptSync.txt"
        )
        timecoded_path = (
            timecoded_root / relative_parent / f"{prefix} timecoded.txt"
        )

        existing = [
            path for path in (scriptsync_path, timecoded_path) if path.exists()
        ]
        overwrite_this = overwrite_all
        if existing and not overwrite_all:
            print(f"[{number}/{len(media_files)}] Existing output found for: {source}")
            for path in existing:
                print(f"  {path}")
            while True:
                try:
                    choice = input(
                        "Overwrite [o]nly this clip, overwrite [a]ll remaining, "
                        "[s]kip existing, or [q]uit? "
                    ).strip().lower()
                except EOFError:
                    choice = "s"
                if choice in {"o", "only"}:
                    overwrite_this = True
                    break
                if choice in {"a", "all"}:
                    overwrite_all = True
                    overwrite_this = True
                    break
                if choice in {"s", "skip", ""}:
                    if len(existing) == 2:
                        print(f"Skipping completed: {source}")
                        skipped += 1
                        break
                    print("Keeping the existing file and creating the missing one.")
                    overwrite_this = False
                    break
                if choice in {"q", "quit"}:
                    print(
                        f"Stopped. Completed: {completed}; skipped: {skipped}; "
                        f"failed: {failed}."
                    )
                    return 0
                print("Please enter o, a, s, or q.")
            if len(existing) == 2 and not overwrite_this:
                continue

        print(f"[{number}/{len(media_files)}] Transcribing: {source}")
        try:
            with tempfile.TemporaryDirectory(prefix="avid-transcript-") as temp:
                temp_path = Path(temp)
                subprocess.run(
                    [
                        "mlx_whisper", str(source),
                        "--model", args.model,
                        "--language", args.language,
                        "--condition-on-previous-text", "False",
                        "--word-timestamps", "True",
                        "--hallucination-silence-threshold",
                        str(args.hallucination_silence_threshold),
                        "--no-speech-threshold", str(args.no_speech_threshold),
                        "--logprob-threshold", str(args.logprob_threshold),
                        "--output-format", "json",
                        "--output-dir", str(temp_path),
                        "--output-name", "transcript",
                    ],
                    check=True,
                )
                transcript = json.loads(
                    (temp_path / "transcript.json").read_text(encoding="utf-8")
                )
                segments = transcript.get("segments", [])
                rate, source_timecode, missing = probe_media(source)

                if overwrite_this or not scriptsync_path.exists():
                    scriptsync_path.parent.mkdir(parents=True, exist_ok=True)
                    scriptsync_path.write_text(
                        clean_scriptsync_text(segments), encoding="ascii"
                    )
                if overwrite_this or not timecoded_path.exists():
                    timecoded_path.parent.mkdir(parents=True, exist_ok=True)
                    timecoded_path.write_text(
                        make_timecoded_text(
                            source, segments, rate, source_timecode, missing
                        ),
                        encoding="utf-8",
                    )
            completed += 1
        except Exception as error:
            failed += 1
            print(f"FAILED: {source}\n  {error}", file=sys.stderr)

    print(
        f"Finished. Completed: {completed}; skipped: {skipped}; failed: {failed}."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
