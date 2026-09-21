#!/usr/bin/env python3
"""What MLX Transcript needs from the FFmpeg it bundles, in one place.

The first standalone build shipped an FFmpeg that could not write raw PCM:

    Requested output format 's16le' is not known.

The configure line had asked for the components as comma-separated lists,
``--enable-muxer=f32le,s16le,wav,null``. FFmpeg's configure turns that value
into a shell case pattern and matches it against its component list, whose
entries are named ``s16le_muxer`` and ``f32le_muxer``. Nothing matched, the
warning scrolled past in a long build log, and ``--disable-muxers`` won. The
application then failed on its first real clip.

So this module is the single source of truth. The configure arguments, the
build-time validation, the documentation and the tests all read the same
tuples, and every component is emitted as its own flag.

Run it directly:

    python3 packaging/ffmpeg_requirements.py --configure-args
    python3 packaging/ffmpeg_requirements.py --verify <ffmpeg> <ffprobe>
"""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

__all__ = [
    "REQUIRED_ENCODERS",
    "REQUIRED_MUXERS",
    "REQUIRED_FILTERS",
    "REQUIRED_DEMUXERS",
    "REQUIRED_DECODERS",
    "FORBIDDEN_CONFIGURE_FLAGS",
    "REQUIRED_CONFIGURE_FLAGS",
    "PCM_OUTPUT_FORMATS",
    "MediaToolError",
    "configure_arguments",
    "listed_components",
    "make_probe_media",
    "verify_media_tools",
]


# --------------------------------------------------------------- the contract

#: Raw PCM encoders. ``pcm_s16le`` is what mlx_whisper's ``load_audio`` asks
#: for; ``pcm_f32le`` is what the speaker-detection pipe asks for.
REQUIRED_ENCODERS: tuple[str, ...] = ("pcm_s16le", "pcm_f32le")

#: Output formats. ``s16le`` and ``f32le`` are headerless raw PCM, which is
#: the whole point: both callers read samples straight off a pipe. ``wav`` and
#: ``null`` exist for diagnostics and for the build's own validation.
REQUIRED_MUXERS: tuple[str, ...] = ("s16le", "f32le", "wav", "null")

#: Audio graph. ``-ac 1 -ar 16000`` makes FFmpeg insert ``aresample`` and
#: ``aformat``; the rest keep simple graphs and diagnostics working.
REQUIRED_FILTERS: tuple[str, ...] = (
    "aresample",
    "anull",
    "aformat",
    "atrim",
    "copy",
)

#: Containers the application accepts, by demuxer name. Demuxing is never
#: trimmed, so this is a guard against a future configure change rather than
#: something the recipe has to ask for.
REQUIRED_DEMUXERS: tuple[str, ...] = (
    "mov",
    "matroska",
    "mpegts",
    "avi",
    "asf",
    "mxf",
    "wav",
    "aiff",
    "mp3",
    "flac",
    "ogg",
)

#: Audio codecs those containers carry.
REQUIRED_DECODERS: tuple[str, ...] = (
    "aac",
    "ac3",
    "eac3",
    "alac",
    "flac",
    "mp2",
    "mp3",
    "opus",
    "vorbis",
    "wmav2",
    "pcm_s16le",
    "pcm_s24le",
    "pcm_f32le",
)

#: A build carrying any of these is not LGPL and must never be shipped.
FORBIDDEN_CONFIGURE_FLAGS: tuple[str, ...] = (
    "--enable-gpl",
    "--enable-nonfree",
    "--enable-libx264",
    "--enable-libx265",
    "--enable-libvmaf",
    "--enable-libpostproc",
)

#: A build must say it turned those off.
REQUIRED_CONFIGURE_FLAGS: tuple[str, ...] = (
    "--disable-gpl",
    "--disable-nonfree",
)

#: Output format and sample width for each raw PCM decode the build proves.
PCM_OUTPUT_FORMATS: tuple[tuple[str, str, int], ...] = (
    ("s16le", "pcm_s16le", 2),
    ("f32le", "pcm_f32le", 4),
)

SAMPLE_RATE = 16_000


class MediaToolError(RuntimeError):
    """Raised when a located FFmpeg cannot do what the application needs."""


# ------------------------------------------------------- configure arguments


def configure_arguments(prefix: str = "$HOME/ffmpeg-lgpl") -> list[str]:
    """Return the full configure line, one flag per component.

    Every encoder, muxer and filter gets its own ``--enable-...`` flag. A
    comma-separated list is silently ignored by FFmpeg's configure, which is
    the bug this exists to prevent.
    """
    arguments = [
        f"--prefix={prefix}",
        # Licensing: LGPL only, no GPL or non-free components at all.
        "--disable-gpl",
        "--disable-nonfree",
        "--disable-version3",
        # Nothing this application uses needs these.
        "--disable-doc",
        "--disable-debug",
        "--disable-network",
        "--disable-devices",
        "--disable-ffplay",
        # Decoding, demuxing, parsing and protocols stay fully enabled, which
        # is what keeps MXF, AVI, MTS, M2TS, WMV, MOV, MP4, MKV and every
        # audio container readable.
        "--disable-encoders",
        "--disable-muxers",
        "--disable-filters",
    ]
    arguments += [f"--enable-encoder={name}" for name in REQUIRED_ENCODERS]
    arguments += [f"--enable-muxer={name}" for name in REQUIRED_MUXERS]
    arguments += [f"--enable-filter={name}" for name in REQUIRED_FILTERS]
    arguments += [
        "--enable-static",
        "--disable-shared",
        "--enable-videotoolbox",
        "--enable-audiotoolbox",
    ]
    return arguments


# ----------------------------------------------------------------- inspection

_COMPONENT_LINE = re.compile(r"^\s*[A-Z.]{1,6}\s+(\S+)")


def listed_components(executable: str, switch: str) -> set[str]:
    """Return the component names an FFmpeg reports for ``-encoders`` and friends.

    Each line looks like ``" DE s16le  PCM signed 16-bit little-endian"``, and
    a demuxer line can name several formats at once
    (``"mov,mp4,m4a,3gp,3g2,mj2"``), so the name field is split on commas.
    """
    try:
        result = subprocess.run(
            [executable, "-hide_banner", switch],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise MediaToolError(f"{executable} {switch} could not run: {error}") from error

    found: set[str] = set()
    for line in (result.stdout or "").splitlines():
        match = _COMPONENT_LINE.match(line)
        if match is None:
            continue
        for name in match.group(1).split(","):
            if name:
                found.add(name)
    return found


def configuration_of(executable: str) -> str:
    """Return the configure line an FFmpeg reports, or an empty string."""
    try:
        result = subprocess.run(
            [executable, "-version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise MediaToolError(f"{executable} -version could not run: {error}") from error
    for line in (result.stdout or "").splitlines():
        if line.strip().startswith("configuration:"):
            return line.split(":", 1)[1].strip()
    return ""


# ------------------------------------------------------------- probe media


def make_probe_media(destination: Path, seconds: float = 0.25) -> Path:
    """Write a small real media file to decode during validation.

    A 16-bit PCM WAV is built with the standard library, so validation needs
    no pre-existing sample and no encoder. Decoding it exercises the whole
    chain the application depends on: demux, decode, the resampling filter
    graph, the raw PCM encoder, and the headerless muxer writing to stdout.
    """
    destination = Path(destination)
    frames = int(SAMPLE_RATE * seconds)
    samples = bytearray()
    for index in range(frames):
        # A quiet tone rather than silence, so a decode that drops audio
        # cannot pass by producing zeros.
        value = int(8000 * ((index % 100) / 100.0 - 0.5))
        samples += struct.pack("<h", value)
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(bytes(samples))
    return destination


def decode_command(executable: str, media: Path, output_format: str, encoder: str) -> list[str]:
    """Return the decode-to-stdout command the build validates with.

    Deliberately the same shape the application uses at runtime, so a build
    that passes this cannot fail the way the first standalone build did.
    """
    return [
        executable,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(media),
        "-map",
        "a:0?",
        "-f",
        output_format,
        "-acodec",
        encoder,
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]


# ------------------------------------------------------------- verification


def verify_media_tools(ffmpeg: str, ffprobe: str, allow_gpl: bool = False) -> list[str]:
    """Check a located FFmpeg pair, or raise :class:`MediaToolError`.

    Returns the lines of a short report for the build log. Raises on the first
    group of problems found, naming every one of them, so a failed build says
    what to add to the configure line rather than only that something is off.
    """
    report: list[str] = []

    configuration = configuration_of(ffmpeg)
    report.append(f"ffmpeg: {ffmpeg}")
    report.append(f"configuration: {configuration or '(not reported)'}")

    offending = [flag for flag in FORBIDDEN_CONFIGURE_FLAGS if flag in configuration]
    if offending and not allow_gpl:
        raise MediaToolError(
            f"{ffmpeg} was built with {' '.join(offending)}. That makes the "
            "binary GPL or non-free and it must not be bundled. Rebuild with "
            "--disable-gpl --disable-nonfree."
        )
    if configuration:
        missing_flags = [
            flag for flag in REQUIRED_CONFIGURE_FLAGS if flag not in configuration
        ]
        if missing_flags and not allow_gpl:
            raise MediaToolError(
                f"{ffmpeg} does not report {' '.join(missing_flags)}. Rebuild "
                "it from the recipe in docs/BUILDING.md."
            )

    problems: list[str] = []
    for switch, required, label in (
        ("-encoders", REQUIRED_ENCODERS, "encoder"),
        ("-muxers", REQUIRED_MUXERS, "muxer"),
        ("-filters", REQUIRED_FILTERS, "filter"),
        ("-demuxers", REQUIRED_DEMUXERS, "demuxer"),
        ("-decoders", REQUIRED_DECODERS, "decoder"),
    ):
        available = listed_components(ffmpeg, switch)
        missing = [name for name in required if name not in available]
        if missing:
            problems += [
                f"missing {label} {name}  (add --enable-{label}={name})"
                for name in missing
            ]
        else:
            report.append(f"{label}s: all {len(required)} present")

    if problems:
        raise MediaToolError(
            "The bundled FFmpeg is missing components the application needs:\n  "
            + "\n  ".join(problems)
        )

    with tempfile.TemporaryDirectory(prefix="mlx-ffmpeg-verify-") as staging:
        media = make_probe_media(Path(staging) / "probe.wav")
        for output_format, encoder, width in PCM_OUTPUT_FORMATS:
            command = decode_command(ffmpeg, media, output_format, encoder)
            try:
                result = subprocess.run(command, capture_output=True, timeout=120)
            except (OSError, subprocess.SubprocessError) as error:
                raise MediaToolError(
                    f"decoding to {output_format} could not run: {error}"
                ) from error
            if result.returncode != 0:
                detail = (result.stderr or b"").decode("utf-8", "ignore").strip()
                raise MediaToolError(
                    f"{ffmpeg} cannot decode to {output_format}: "
                    f"{detail or f'exit {result.returncode}'}\n"
                    f"Command: {' '.join(command)}"
                )
            if not result.stdout:
                raise MediaToolError(
                    f"{ffmpeg} produced no {output_format} samples on stdout."
                )
            if len(result.stdout) % width:
                raise MediaToolError(
                    f"{ffmpeg} produced {len(result.stdout)} bytes of "
                    f"{output_format}, which is not a whole number of "
                    f"{width}-byte samples."
                )
            report.append(
                f"decoded to {output_format}: {len(result.stdout)} bytes "
                f"({len(result.stdout) // width} samples)"
            )

    try:
        probe = subprocess.run(
            [ffprobe, "-hide_banner", "-version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise MediaToolError(f"{ffprobe} could not run: {error}") from error
    if probe.returncode != 0:
        raise MediaToolError(f"{ffprobe} exited {probe.returncode}.")
    report.append(f"ffprobe: {ffprobe} runs")

    return report


# ------------------------------------------------------------------ the CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--configure-args",
        action="store_true",
        help="print the configure flags, one per line",
    )
    group.add_argument(
        "--verify",
        nargs=2,
        metavar=("FFMPEG", "FFPROBE"),
        help="check a built FFmpeg pair and exit non-zero if it falls short",
    )
    parser.add_argument(
        "--prefix",
        default="$HOME/ffmpeg-lgpl",
        help="install prefix used by --configure-args",
    )
    parser.add_argument(
        "--allow-gpl",
        action="store_true",
        help="skip the licence check; for local builds that are never shipped",
    )
    arguments = parser.parse_args(argv)

    if arguments.configure_args:
        for flag in configure_arguments(arguments.prefix):
            print(flag)
        return 0

    ffmpeg, ffprobe = arguments.verify
    try:
        for line in verify_media_tools(ffmpeg, ffprobe, allow_gpl=arguments.allow_gpl):
            print(f"  {line}")
    except MediaToolError as error:
        print(f"\nFFmpeg validation failed.\n\n{error}\n", file=sys.stderr)
        return 1
    print("\nThe bundled media tools can do everything the application needs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
