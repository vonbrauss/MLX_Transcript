"""The bundled FFmpeg contract: configure flags and build-time validation.

The first standalone build shipped an FFmpeg with no `s16le` muxer, because
the configure line asked for components as comma-separated lists and FFmpeg's
configure matched that value against entries named ``s16le_muxer``. Nothing
matched, ``--disable-muxers`` won, and the application failed on its first
real clip.

So these tests pin three things: every component is emitted as its own flag,
the validation actually runs a decode rather than only reading a version
string, and the documentation and build script cannot drift from the module
that defines the contract.
"""

from __future__ import annotations

import importlib.util
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGING = PROJECT_ROOT / "packaging"
BUILDING = PROJECT_ROOT / "docs" / "BUILDING.md"
SCRIPT = PROJECT_ROOT / "scripts" / "finish_release_prep.sh"
SPEC = PROJECT_ROOT / "packaging" / "MLX_Transcript.spec"

if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))

import ffmpeg_requirements as req  # noqa: E402


# ------------------------------------------------- one flag per component


def test_no_configure_flag_carries_a_comma():
    """The exact defect: a comma-separated value is silently ignored."""
    for flag in req.configure_arguments():
        assert "," not in flag, f"{flag} would match nothing in FFmpeg's configure"


def test_every_required_encoder_has_its_own_flag():
    flags = req.configure_arguments()
    for name in req.REQUIRED_ENCODERS:
        assert f"--enable-encoder={name}" in flags


def test_every_required_muxer_has_its_own_flag():
    flags = req.configure_arguments()
    components = req.resolve_muxer_components()
    for name in req.REQUIRED_MUXERS:
        assert f"--enable-muxer={components[name]}" in flags


def test_every_required_filter_has_its_own_flag():
    flags = req.configure_arguments()
    for name in req.REQUIRED_FILTERS:
        assert f"--enable-filter={name}" in flags


def test_the_two_pcm_pairs_the_application_needs_are_present():
    """s16le for Whisper's audio loading, f32le for speaker detection."""
    flags = req.configure_arguments()
    for encoder, component in (("pcm_s16le", "pcm_s16le"), ("pcm_f32le", "pcm_f32le")):
        assert f"--enable-encoder={encoder}" in flags
        assert f"--enable-muxer={component}" in flags


def test_wav_and_null_muxers_are_available_for_validation():
    flags = req.configure_arguments()
    assert "--enable-muxer=wav" in flags
    assert "--enable-muxer=null" in flags


def test_the_flag_count_matches_the_declared_components():
    flags = req.configure_arguments()
    assert sum(f.startswith("--enable-encoder=") for f in flags) == len(
        req.REQUIRED_ENCODERS
    )
    assert sum(f.startswith("--enable-muxer=") for f in flags) == len(
        req.REQUIRED_MUXERS
    )
    assert sum(f.startswith("--enable-filter=") for f in flags) == len(
        req.REQUIRED_FILTERS
    )


def test_the_prefix_is_honoured():
    assert "--prefix=/opt/somewhere" in req.configure_arguments("/opt/somewhere")


# ------------------------------------------------------------- the licence


def test_the_recipe_disables_gpl_and_nonfree():
    flags = req.configure_arguments()
    for flag in req.REQUIRED_CONFIGURE_FLAGS:
        assert flag in flags


def test_the_recipe_never_enables_a_gpl_component():
    flags = " ".join(req.configure_arguments())
    for forbidden in req.FORBIDDEN_CONFIGURE_FLAGS:
        assert forbidden not in flags
    for library in ("libx264", "libx265", "libvmaf", "libpostproc"):
        assert f"--enable-{library}" not in flags


def test_decoding_and_demuxing_are_never_trimmed():
    flags = req.configure_arguments()
    for never in ("--disable-decoders", "--disable-demuxers", "--disable-parsers",
                  "--disable-protocols", "--disable-everything"):
        assert never not in flags


# --------------------------------------------------- the validation commands


def test_the_decode_command_matches_what_the_application_runs():
    """A build that passes validation cannot fail the way the first one did."""
    command = req.decode_command("ffmpeg", Path("clip.mov"), "s16le", "pcm_s16le")

    assert command[0] == "ffmpeg"
    assert "-f" in command and command[command.index("-f") + 1] == "s16le"
    assert "-acodec" in command and command[command.index("-acodec") + 1] == "pcm_s16le"
    assert command[-1] == "-", "output has to go to stdout"
    assert "-ac" in command and command[command.index("-ac") + 1] == "1"
    assert "-ar" in command and command[command.index("-ar") + 1] == "16000"
    assert "-map" in command and command[command.index("-map") + 1] == "a:0?"


def test_both_pcm_formats_are_validated_with_their_sample_widths():
    assert req.PCM_OUTPUT_FORMATS == (
        ("s16le", "pcm_s16le", 2),
        ("f32le", "pcm_f32le", 4),
    )


def test_the_probe_media_is_a_real_readable_wav(tmp_path):
    media = req.make_probe_media(tmp_path / "probe.wav")

    with wave.open(str(media), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == req.SAMPLE_RATE
        frames = handle.readframes(handle.getnframes())

    assert len(frames) > 0
    # Not silence: a decode that dropped the audio must not be able to pass.
    assert any(struct.unpack("<h", frames[i : i + 2])[0] for i in range(0, 200, 2))


def test_the_component_parser_reads_an_ffmpeg_listing(monkeypatch):
    listing = (
        "Muxers:\n"
        " E s16le           PCM signed 16-bit little-endian\n"
        " E f32le           PCM 32-bit floating-point little-endian\n"
        " E wav             WAV / WAVE\n"
        " D  mov,mp4,m4a,3gp,3g2,mj2 QuickTime / MOV\n"
    )

    class Result:
        stdout = listing

    monkeypatch.setattr(req.subprocess, "run", lambda *a, **k: Result())

    found = req.listed_components("ffmpeg", "-muxers")

    assert {"s16le", "f32le", "wav"} <= found
    # Comma-separated names are split, which is how demuxer lines are written.
    assert {"mov", "mp4", "m4a"} <= found


def test_a_missing_muxer_is_reported_with_the_flag_that_fixes_it(monkeypatch):
    monkeypatch.setattr(req, "configuration_of", lambda executable: "--disable-gpl --disable-nonfree")
    monkeypatch.setattr(
        req,
        "listed_components",
        lambda executable, switch: set() if switch == "-muxers" else set(
            req.REQUIRED_ENCODERS
            + req.REQUIRED_FILTERS
            + req.REQUIRED_DEMUXERS
            + req.REQUIRED_DECODERS
        ),
    )

    with pytest.raises(req.MediaToolError) as caught:
        req.verify_media_tools("ffmpeg", "ffprobe")

    message = str(caught.value)
    assert "missing muxer s16le" in message
    # The advice has to name the configure component, not the format name.
    assert "--enable-muxer=pcm_s16le" in message
    assert "--enable-muxer=pcm_f32le" in message


def test_a_gpl_build_is_refused_by_the_validation(monkeypatch):
    monkeypatch.setattr(
        req, "configuration_of", lambda executable: "--enable-gpl --enable-libx264"
    )

    with pytest.raises(req.MediaToolError) as caught:
        req.verify_media_tools("ffmpeg", "ffprobe")

    assert "--enable-gpl" in str(caught.value)


def test_a_build_that_does_not_say_it_disabled_gpl_is_refused(monkeypatch):
    monkeypatch.setattr(req, "configuration_of", lambda executable: "--enable-shared")

    with pytest.raises(req.MediaToolError) as caught:
        req.verify_media_tools("ffmpeg", "ffprobe")

    assert "--disable-gpl" in str(caught.value)


def test_a_decode_that_produces_nothing_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(
        req, "configuration_of", lambda e: "--disable-gpl --disable-nonfree"
    )
    everything = set(
        req.REQUIRED_ENCODERS
        + req.REQUIRED_MUXERS
        + req.REQUIRED_FILTERS
        + req.REQUIRED_DEMUXERS
        + req.REQUIRED_DECODERS
    )
    monkeypatch.setattr(req, "listed_components", lambda e, switch: everything)

    class Empty:
        returncode = 0
        stdout = b""
        stderr = b""

    monkeypatch.setattr(req.subprocess, "run", lambda *a, **k: Empty())

    with pytest.raises(req.MediaToolError) as caught:
        req.verify_media_tools("ffmpeg", "ffprobe")

    assert "no s16le samples" in str(caught.value)


def test_a_failed_decode_reports_the_command(monkeypatch):
    monkeypatch.setattr(
        req, "configuration_of", lambda e: "--disable-gpl --disable-nonfree"
    )
    everything = set(
        req.REQUIRED_ENCODERS
        + req.REQUIRED_MUXERS
        + req.REQUIRED_FILTERS
        + req.REQUIRED_DEMUXERS
        + req.REQUIRED_DECODERS
    )
    monkeypatch.setattr(req, "listed_components", lambda e, switch: everything)

    class Failed:
        returncode = 1
        stdout = b""
        stderr = b"Requested output format 's16le' is not known."

    monkeypatch.setattr(req.subprocess, "run", lambda *a, **k: Failed())

    with pytest.raises(req.MediaToolError) as caught:
        req.verify_media_tools("ffmpeg", "ffprobe")

    message = str(caught.value)
    assert "is not known" in message
    assert "Command:" in message


def test_misaligned_pcm_output_is_refused(monkeypatch):
    monkeypatch.setattr(
        req, "configuration_of", lambda e: "--disable-gpl --disable-nonfree"
    )
    everything = set(
        req.REQUIRED_ENCODERS
        + req.REQUIRED_MUXERS
        + req.REQUIRED_FILTERS
        + req.REQUIRED_DEMUXERS
        + req.REQUIRED_DECODERS
    )
    monkeypatch.setattr(req, "listed_components", lambda e, switch: everything)

    class Odd:
        returncode = 0
        stdout = b"abc"  # not a whole number of 2-byte samples
        stderr = b""

    monkeypatch.setattr(req.subprocess, "run", lambda *a, **k: Odd())

    with pytest.raises(req.MediaToolError) as caught:
        req.verify_media_tools("ffmpeg", "ffprobe")

    assert "whole number" in str(caught.value)


# ------------------------------------------------ the CLI the script invokes


def test_the_configure_args_command_prints_one_flag_per_line():
    result = subprocess.run(
        [sys.executable, str(PACKAGING / "ffmpeg_requirements.py"), "--configure-args"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines == req.configure_arguments()
    assert all("," not in line for line in lines)


def test_the_verify_command_fails_on_a_binary_that_does_not_exist(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGING / "ffmpeg_requirements.py"),
            "--verify",
            str(tmp_path / "nope"),
            str(tmp_path / "nope"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "validation failed" in result.stderr


# ----------------------------------------- the script and docs cannot drift


def test_the_build_script_generates_the_flags_rather_than_repeating_them():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "ffmpeg_requirements.py" in text
    assert "--configure-args" in text
    # The guarded expansion, because Bash 3.2 treats "${a[@]}" on an empty
    # array as an unbound variable under set -u.
    assert './configure ${configure_args[@]+"${configure_args[@]}"}' in text


def test_the_build_script_no_longer_carries_a_comma_separated_flag():
    text = SCRIPT.read_text(encoding="utf-8")

    for kind in ("encoder", "muxer", "filter"):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"--enable-{kind}=") and "," in stripped:
                pytest.fail(f"comma-separated flag left in the script: {stripped}")


def test_the_build_script_validates_after_building():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "stage_validate_ffmpeg" in text
    assert "--verify" in text


def test_the_documentation_lists_every_flag_explicitly():
    text = BUILDING.read_text(encoding="utf-8")

    for name in req.REQUIRED_ENCODERS:
        assert f"--enable-encoder={name}" in text
    for name in req.resolve_muxer_components().values():
        assert f"--enable-muxer={name}" in text
    for name in req.REQUIRED_FILTERS:
        assert f"--enable-filter={name}" in text


def test_the_documentation_explains_the_two_pcm_formats():
    text = BUILDING.read_text(encoding="utf-8")

    assert "Why two raw PCM formats" in text
    assert "mlx_whisper.load_audio" in text
    assert "speaker detection" in text
    assert "Requested output format 's16le' is not known." in text
    assert "headerless" in text


def test_the_documentation_warns_against_comma_separated_flags():
    text = BUILDING.read_text(encoding="utf-8")

    assert "matches nothing" in text
    assert "one flag per component" in text


# ------------------------------------------------- the packaging gate itself


def test_the_spec_loads_the_shared_requirements():
    text = SPEC.read_text(encoding="utf-8")

    assert "ffmpeg_requirements.py" in text
    assert "verify_media_tools" in text


def test_the_spec_refuses_to_package_a_failing_ffmpeg():
    text = SPEC.read_text(encoding="utf-8")

    assert "Refusing to package" in text
    assert "MediaToolError" in text
    # The refusal has to come before PyInstaller analyses anything.
    assert text.index("verify_media_tools") < text.index("a = Analysis(")


def test_the_spec_records_the_validation_in_the_provenance_file():
    text = SPEC.read_text(encoding="utf-8")

    assert "ffmpeg-configuration.txt" in text
    assert "record_media_tool_provenance(provenance, validation)" in text


def test_the_spec_still_parses_as_python():
    source = SPEC.read_text(encoding="utf-8")
    compile(source, str(SPEC), "exec")


def test_the_requirements_module_loads_the_way_the_spec_loads_it():
    """The spec loads it by path, so that path has to keep working."""
    location = PACKAGING / "ffmpeg_requirements.py"
    spec = importlib.util.spec_from_file_location("probe_requirements", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.REQUIRED_MUXERS == req.REQUIRED_MUXERS
    assert callable(module.verify_media_tools)
