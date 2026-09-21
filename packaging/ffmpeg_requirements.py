#!/usr/bin/env python3
"""What MLX Transcript needs from the FFmpeg it bundles, in one place.

The first standalone build shipped an FFmpeg that could not write raw PCM:

    Requested output format 's16le' is not known.

There were two causes, one behind the other.

First, the configure line asked for the components as comma-separated lists,
``--enable-muxer=f32le,s16le,wav,null``. FFmpeg's configure turns that value
into a shell case pattern and matches it against its component list, so
nothing matched, the warning scrolled past in a long build log, and
``--disable-muxers`` won.

Second, and this survived the one-flag-per-component fix: a raw PCM muxer's
*configure component* is not spelled the same as its *format name*. The
format is ``s16le``, which is what ``-f s16le`` and ``ffmpeg -muxers`` say,
but the component configure knows is ``pcm_s16le``. So
``--enable-muxer=s16le`` is accepted, prints one more

    WARNING: Option --enable-muxer=s16le did not match anything

and enables nothing. ``wav`` and ``null`` are spelled the same either way,
which is why only the two PCM muxers went missing both times.

So this module is the single source of truth, and it keeps the two spellings
apart: :data:`REQUIRED_MUXERS` holds format names, for what a built FFmpeg is
asked to report and to write, and :data:`MUXER_COMPONENT_CANDIDATES` maps each
one to the component names configure might know it by. The build resolves
those against the source tree's own ``configure --list-muxers`` rather than
trusting this file, then checks after configure that they really were enabled.

Run it directly:

    python3 packaging/ffmpeg_requirements.py --configure-args
    python3 packaging/ffmpeg_requirements.py --configure-args \
        --source-tree /path/to/ffmpeg-7.1.1
    python3 packaging/ffmpeg_requirements.py --check-configured <source-tree>
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
    "MUXER_COMPONENT_CANDIDATES",
    "REQUIRED_FILTERS",
    "REQUIRED_DEMUXERS",
    "REQUIRED_DECODERS",
    "FORBIDDEN_CONFIGURE_FLAGS",
    "REQUIRED_CONFIGURE_FLAGS",
    "PCM_OUTPUT_FORMATS",
    "MediaToolError",
    "assert_flags_are_well_formed",
    "available_muxer_components",
    "configure_arguments",
    "configured_components",
    "configured_muxer_components",
    "required_components_by_kind",
    "assert_disables_precede_enables",
    "BROAD_DISABLE_FLAGS",
    "CHECKED_COMPONENT_KINDS",
    "GENERATED_CONFIG_HEADERS",
    "resolve_muxer_components",
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

#: Format name -> the component names ``configure`` might know it by, in the
#: order to prefer. The raw PCM muxers are the reason this mapping exists:
#: their format is ``s16le`` but the component is ``pcm_s16le``, so asking for
#: the format name enables nothing and only warns. The build resolves these
#: against the source tree's ``configure --list-muxers``, so a rename in a
#: future FFmpeg is caught rather than silently dropped.
MUXER_COMPONENT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "s16le": ("pcm_s16le", "s16le"),
    "f32le": ("pcm_f32le", "f32le"),
    "wav": ("wav",),
    "null": ("null",),
}

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


#: A configure flag is exactly two hyphens, then a name, then optionally
#: ``=value``. Anything else -- a stray backslash from a line continuation, a
#: quote, leading whitespace -- means the generation or the transport damaged
#: it, and configure would either reject it or, worse, ignore it.
#: ``$`` is allowed in a value because the documented default prefix is
#: ``$HOME/ffmpeg-lgpl``; quotes and backticks are not, because a flag that
#: needs quoting has already been mangled by whatever produced it.
_WELL_FORMED_FLAG = re.compile(r"^--[A-Za-z0-9][A-Za-z0-9_.-]*(=[^\s\\'\"`]+)?$")


def assert_flags_are_well_formed(flags: list[str]) -> None:
    """Raise unless every flag begins with exactly ``--`` and is otherwise sane.

    The specific failure this catches is a flag arriving as ``\\--enable-...``.
    A backslash-escaped flag is not an error configure reports usefully: it
    warns once, in a build log thousands of lines long, and carries on without
    the component.
    """
    problems = []
    for flag in flags:
        if flag.startswith("\\") or "\\" in flag:
            problems.append(f"{flag!r} contains a backslash")
        elif not flag.startswith("--"):
            problems.append(f"{flag!r} does not begin with '--'")
        elif not _WELL_FORMED_FLAG.match(flag):
            problems.append(f"{flag!r} is not a well-formed configure flag")

    if problems:
        raise MediaToolError(
            "Refusing to run configure with damaged flags:\n  "
            + "\n  ".join(problems)
        )


def available_muxer_components(source_tree: str | Path) -> set[str]:
    """Ask a source tree's own configure which muxer components it has.

    ``./configure --list-muxers`` prints the component names, which is the
    spelling ``--enable-muxer=`` takes. It is the only authority on that, so
    the build asks rather than assumes.
    """
    tree = Path(source_tree)
    configure = tree / "configure"
    if not configure.is_file():
        raise MediaToolError(f"{configure} does not exist; is that an FFmpeg source tree?")

    try:
        result = subprocess.run(
            ["sh", str(configure), "--list-muxers"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(tree),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise MediaToolError(f"configure --list-muxers could not run: {error}") from error

    if result.returncode != 0:
        raise MediaToolError(
            "configure --list-muxers failed:\n" + (result.stderr or result.stdout).strip()
        )
    return {name for name in result.stdout.split() if name}


def resolve_muxer_components(available: set[str] | None = None) -> dict[str, str]:
    """Return format name -> the component name to enable it by.

    With ``available`` from :func:`available_muxer_components`, each format's
    candidates are tried in order and the first one the tree actually has
    wins. Without it, the first candidate is used, which is what the
    documented recipe shows.
    """
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for muxer in REQUIRED_MUXERS:
        candidates = MUXER_COMPONENT_CANDIDATES.get(muxer, (muxer,))
        if available is None:
            resolved[muxer] = candidates[0]
            continue
        for candidate in candidates:
            if candidate in available:
                resolved[muxer] = candidate
                break
        else:
            missing.append(
                f"the {muxer} muxer: tried {', '.join(candidates)}, "
                "none of which this FFmpeg has"
            )

    if missing:
        raise MediaToolError(
            "This FFmpeg source tree cannot provide a muxer the application "
            "needs:\n  " + "\n  ".join(missing)
        )
    return resolved


def _component_for(label: str, name: str) -> str:
    """The configure component name for a reported component, by kind."""
    if label == "muxer":
        return MUXER_COMPONENT_CANDIDATES.get(name, (name,))[0]
    return name


#: Where a finished configure records its per-component ``#define``s. FFmpeg
#: 5.1 moved every ``CONFIG_<NAME>_<KIND>`` out of ``config.h`` and into
#: ``config_components.h``, to stop a component change rebuilding the world.
#: Reading only ``config.h`` on such a tree finds the file, finds none of the
#: defines, and reports every component disabled -- including ``null``, which
#: has no dependencies and cannot plausibly be off. All four candidates are
#: read and merged so the check works on either layout.
GENERATED_CONFIG_HEADERS: tuple[str, ...] = (
    "config_components.h",
    "ffbuild/config_components.h",
    "config.h",
    "ffbuild/config.h",
)

#: The component kinds the gate checks, as the ``#define`` suffix.
CHECKED_COMPONENT_KINDS: tuple[str, ...] = ("ENCODER", "MUXER", "FILTER")


def configured_components(source_tree: str | Path, kind: str) -> set[str]:
    """Read back which components of ``kind`` a finished configure turned on.

    configure writes ``#define CONFIG_PCM_S16LE_MUXER 1`` for each component
    it enabled and ``0`` for each it did not, so this is the state compilation
    is about to use rather than a restatement of what was asked for.
    """
    tree = Path(source_tree)
    headers = [tree / relative for relative in GENERATED_CONFIG_HEADERS]
    present = [header for header in headers if header.is_file()]
    if not present:
        raise MediaToolError(
            f"No generated config header under {tree} (looked for "
            + ", ".join(GENERATED_CONFIG_HEADERS)
            + "); configure has not run yet."
        )

    pattern = re.compile(rf"^#define CONFIG_([A-Z0-9_]+)_{kind.upper()} 1$", re.M)
    enabled: set[str] = set()
    for header in present:
        text = header.read_text(encoding="utf-8", errors="replace")
        enabled |= {match.group(1).lower() for match in pattern.finditer(text)}
    return enabled


def configured_muxer_components(source_tree: str | Path) -> set[str]:
    """The muxers a finished configure turned on. See :func:`configured_components`."""
    return configured_components(source_tree, "MUXER")


def required_components_by_kind(
    available_muxers: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Per kind, the component name to enable keyed by the name it is known as.

    For encoders and filters the two are the same. For muxers they are not:
    the format is ``s16le`` and the component is ``pcm_s16le``, so the key is
    what a built FFmpeg reports and the value is what configure takes.
    """
    return {
        "ENCODER": {name: name for name in REQUIRED_ENCODERS},
        "MUXER": resolve_muxer_components(available_muxers),
        "FILTER": {name: name for name in REQUIRED_FILTERS},
    }


#: ``--disable-<kind>s`` and the selective flag it must precede.
BROAD_DISABLE_FLAGS: tuple[tuple[str, str], ...] = (
    ("--disable-encoders", "--enable-encoder="),
    ("--disable-muxers", "--enable-muxer="),
    ("--disable-filters", "--enable-filter="),
)


def assert_disables_precede_enables(flags: list[str]) -> None:
    """Raise unless each broad disable comes before its selective enables.

    FFmpeg's configure applies options in the order it reads them, so a
    ``--disable-muxers`` after ``--enable-muxer=pcm_s16le`` undoes it without
    a word of complaint. The recipe has always been ordered correctly, and
    this is what keeps an edit from quietly reversing it.
    """
    problems = []
    for broad, selective in BROAD_DISABLE_FLAGS:
        if broad not in flags:
            continue
        broad_at = flags.index(broad)
        for position, flag in enumerate(flags):
            if flag.startswith(selective) and position < broad_at:
                problems.append(
                    f"{broad} is at position {broad_at}, after {flag} at "
                    f"position {position}; it would undo it"
                )

    if problems:
        raise MediaToolError(
            "The configure flags are in an order that cancels itself:\n  "
            + "\n  ".join(problems)
        )


def configure_arguments(
    prefix: str = "$HOME/ffmpeg-lgpl",
    available_muxers: set[str] | None = None,
) -> list[str]:
    """Return the full configure line, one flag per component.

    Every encoder, muxer and filter gets its own ``--enable-...`` flag. A
    comma-separated list is silently ignored by FFmpeg's configure, which is
    half of the bug this exists to prevent; the other half is the muxer
    component spelling, which ``available_muxers`` pins down when the source
    tree is at hand.
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
    components = resolve_muxer_components(available_muxers)
    arguments += [f"--enable-encoder={name}" for name in REQUIRED_ENCODERS]
    arguments += [f"--enable-muxer={components[name]}" for name in REQUIRED_MUXERS]
    arguments += [f"--enable-filter={name}" for name in REQUIRED_FILTERS]
    arguments += [
        "--enable-static",
        "--disable-shared",
        "--enable-videotoolbox",
        "--enable-audiotoolbox",
    ]
    assert_flags_are_well_formed(arguments)
    assert_disables_precede_enables(arguments)
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
            # The advice has to name the *component*, which for a raw PCM
            # muxer is not the format name reported here.
            problems += [
                f"missing {label} {name}  "
                f"(add --enable-{label}={_component_for(label, name)})"
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
        "--list-muxer-components",
        metavar="SOURCE_TREE",
        help="print the muxer component names an FFmpeg source tree has",
    )
    group.add_argument(
        "--check-configured",
        metavar="SOURCE_TREE",
        help=(
            "after configure, confirm the required muxers were really enabled "
            "and exit non-zero before anything is compiled"
        ),
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
        "--source-tree",
        help=(
            "FFmpeg source tree to resolve muxer component names against, "
            "used with --configure-args"
        ),
    )
    parser.add_argument(
        "--allow-gpl",
        action="store_true",
        help="skip the licence check; for local builds that are never shipped",
    )
    arguments = parser.parse_args(argv)

    if arguments.configure_args:
        available = None
        if arguments.source_tree:
            try:
                available = available_muxer_components(arguments.source_tree)
            except MediaToolError as error:
                print(f"\n{error}\n", file=sys.stderr)
                return 1
        try:
            flags = configure_arguments(arguments.prefix, available_muxers=available)
        except MediaToolError as error:
            print(f"\n{error}\n", file=sys.stderr)
            return 1
        for flag in flags:
            print(flag)
        return 0

    if arguments.list_muxer_components:
        try:
            for name in sorted(
                available_muxer_components(arguments.list_muxer_components)
            ):
                print(name)
        except MediaToolError as error:
            print(f"\n{error}\n", file=sys.stderr)
            return 1
        return 0

    if arguments.check_configured:
        tree = arguments.check_configured
        try:
            available = available_muxer_components(tree)
            wanted = required_components_by_kind(available)
        except MediaToolError as error:
            print(f"\n{error}\n", file=sys.stderr)
            return 1

        missing = []
        confirmed = []
        for kind in CHECKED_COMPONENT_KINDS:
            try:
                enabled = configured_components(tree, kind)
            except MediaToolError as error:
                print(f"\n{error}\n", file=sys.stderr)
                return 1
            label = kind.lower()
            for name, component in sorted(wanted[kind].items()):
                if component in enabled:
                    if name == component:
                        confirmed.append(f"  {label} {name} enabled")
                    else:
                        confirmed.append(
                            f"  {label} {name} enabled as {component}"
                        )
                else:
                    missing.append(
                        f"--enable-{label}={component} did not enable the "
                        f"{name} {label} (CONFIG_{component.upper()}_{kind} "
                        "is not 1)"
                    )

        if missing:
            print(
                "\nconfigure finished but required components are not enabled.\n"
                "Compiling now would produce the FFmpeg that fails with\n"
                '  "Requested output format \'s16le\' is not known."\n\n  '
                + "\n  ".join(missing)
                + "\n",
                file=sys.stderr,
            )
            return 1
        print("\n".join(confirmed))
        print(
            f"\nAll {len(confirmed)} required encoders, muxers and filters are "
            "enabled. Safe to compile."
        )
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
