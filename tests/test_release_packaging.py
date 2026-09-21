"""Packaging and release compliance.

The review found an archive that could not legally be published, an archive
format that smuggles AppleDouble files into the bundle, a version number
written down in three places, and no licence statement anywhere. These pin
down the fixes without needing a Mac to build on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = PROJECT_ROOT / "packaging" / "MLX_Transcript.spec"
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_macos.sh"
README = PROJECT_ROOT / "README.md"
PREP_SCRIPT = PROJECT_ROOT / "scripts" / "finish_release_prep.sh"
NOTICES = PROJECT_ROOT / "THIRD_PARTY_NOTICES.md"


@pytest.fixture(scope="module")
def spec_text() -> str:
    return SPEC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def build_text() -> str:
    return BUILD_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


# ------------------------------------------------------- one version number


def test_the_version_module_defines_the_constants():
    from app.version import ARCHIVE_NAME, BUILD_NUMBER, VERSION

    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION)
    assert BUILD_NUMBER.isdigit()
    assert ARCHIVE_NAME == f"MLX-Transcript-{VERSION}-arm64"


def test_the_spec_reads_the_version_rather_than_repeating_it(spec_text):
    assert 'read_version_constant("VERSION")' in spec_text
    assert '"CFBundleShortVersionString": VERSION' in spec_text
    assert '"CFBundleVersion": BUILD_NUMBER' in spec_text


def test_no_version_literal_is_left_in_the_spec(spec_text):
    from app.version import VERSION

    assert f'"{VERSION}"' not in spec_text


def test_the_build_script_derives_the_archive_name(build_text):
    assert "app/version.py" in build_text
    assert 'release_zip="$project_root/dist/${archive_name}.zip"' in build_text


def test_no_version_literal_is_left_in_the_build_script(build_text):
    from app.version import VERSION

    assert VERSION not in build_text


# ---------------------------------------------------------- the FFmpeg guard


def test_the_spec_refuses_a_gpl_ffmpeg_by_default(spec_text):
    assert "GPL_CONFIGURE_FLAGS" in spec_text
    assert "--enable-gpl" in spec_text
    assert "--enable-nonfree" in spec_text
    assert "Refusing to bundle" in spec_text


def test_the_refusal_can_be_overridden_deliberately(spec_text):
    assert "MLX_TRANSCRIPT_ALLOW_GPL_FFMPEG" in spec_text


def test_the_spec_can_be_pointed_at_a_specific_ffmpeg(spec_text):
    assert "MLX_TRANSCRIPT_FFMPEG" in spec_text
    assert "MLX_TRANSCRIPT_FFPROBE" in spec_text


def test_the_bundled_configuration_is_recorded(spec_text):
    assert "ffmpeg-configuration.txt" in spec_text


def test_the_gpl_detector_flags_a_homebrew_configuration(spec_text):
    """Exercise the helper itself rather than trusting the source text."""
    namespace: dict[str, object] = {}
    body = spec_text.split("GPL_CONFIGURE_FLAGS")[1]
    exec(  # noqa: S102 - running our own build recipe's helper
        "GPL_CONFIGURE_FLAGS" + body.split("def ffmpeg_configuration")[0],
        namespace,
    )
    exec(  # noqa: S102
        "def gpl_flags_in(configuration):\n"
        "    return [f for f in GPL_CONFIGURE_FLAGS if f in configuration]\n",
        namespace,
    )
    homebrew = (
        "--prefix=/opt/homebrew/Cellar/ffmpeg/9.0.2 --enable-gpl "
        "--enable-version3 --enable-libx264 --enable-libx265"
    )
    lgpl = "--prefix=/opt/lgpl --disable-gpl --disable-nonfree --enable-shared"

    assert namespace["gpl_flags_in"](homebrew) == ["--enable-gpl"]
    assert namespace["gpl_flags_in"](lgpl) == []


# ------------------------------------------------------- the archive format


def test_the_archive_sequesters_resource_forks(build_text):
    assert "--sequesterRsrc" in build_text
    assert "ditto -c -k --sequesterRsrc --keepParent" in build_text


def test_the_build_extracts_and_verifies_the_archive(build_text):
    assert "ditto -x -k" in build_text
    assert 'find "$extracted" -name \'._*\'' in build_text
    assert 'codesign --verify --deep --strict "$extracted"' in build_text


def test_the_build_fails_on_appledouble_files(build_text):
    assert "AppleDouble file(s) were found inside the extracted app" in build_text
    # The report has to be followed by a non-zero exit, not just an echo.
    after = build_text.split(
        "AppleDouble file(s) were found inside the extracted app"
    )[1]
    assert "exit 1" in after[:400]


def test_the_build_runs_the_media_tools_from_the_extracted_app(build_text):
    assert 'for tool in ffmpeg ffprobe' in build_text
    assert 'Contents/Frameworks/bin/$tool" -version' in build_text


def test_the_build_runs_the_engine_check_on_the_extracted_app(build_text):
    assert "engine-check-extracted.txt" in build_text


def test_the_build_does_not_claim_gatekeeper_acceptance(build_text):
    lowered = build_text.lower()
    assert "notarized" not in lowered or "not notarized" in lowered
    assert "gatekeeper approved" not in lowered


# ------------------------------------------------------------------- the icon


def test_the_icon_is_checked_in():
    icon = PROJECT_ROOT / "packaging" / "MLX Transcript.icns"

    assert icon.is_file()
    assert icon.stat().st_size > 10_000
    assert icon.read_bytes()[:4] == b"icns"


def test_the_icon_carries_the_sizes_macos_asks_for():
    icon = (PROJECT_ROOT / "packaging" / "MLX Transcript.icns").read_bytes()

    for kind in (b"icp4", b"ic07", b"ic08", b"ic09", b"ic10", b"ic11", b"ic13"):
        assert kind in icon, f"{kind!r} is missing from the icon"


def test_the_spec_uses_the_icon(spec_text):
    assert 'icon_path = project / "packaging" / "MLX Transcript.icns"' in spec_text
    assert spec_text.count("icon=str(icon_path)") == 2


# --------------------------------------------------------- licence and notices


def test_the_project_ships_the_gpl_text():
    licence = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "GNU GENERAL PUBLIC LICENSE" in licence
    assert "Version 3" in licence


def test_the_readme_states_the_licence(readme_text):
    flowed = " ".join(readme_text.split())

    assert "## License" in readme_text
    assert "GNU General Public License, version 3" in flowed
    assert "THIRD_PARTY_NOTICES.md" in readme_text


def test_the_notices_file_exists_and_names_the_bundled_components():
    text = NOTICES.read_text(encoding="utf-8")

    for component in (
        "FFmpeg",
        "PySide6",
        "MLX",
        "mlx_whisper",
        "sherpa-onnx",
        "ONNX Runtime",
        "tiktoken",
        "numba",
        "NumPy",
        "SciPy",
        "CPython",
    ):
        assert component in text, f"{component} is not covered by the notices"


def test_the_notices_cover_the_downloaded_model_assets():
    text = NOTICES.read_text(encoding="utf-8")

    assert "sherpa-onnx-pyannote-segmentation-3-0.onnx" in text
    assert "wespeaker_en_voxceleb_resnet34_LM.onnx" in text
    assert "whisper-large-v3-mlx" in text
    assert "whisper-large-v3-turbo" in text


def test_the_notices_explain_the_lgpl_replacement_right():
    text = NOTICES.read_text(encoding="utf-8")

    assert "LGPL" in text
    assert "replace" in text.lower()


def test_the_notices_warn_about_a_gpl_ffmpeg():
    text = NOTICES.read_text(encoding="utf-8")

    assert "MLX_TRANSCRIPT_ALLOW_GPL_FFMPEG" in text
    assert "source-distribution" in text


# ------------------------------------------------------ Gatekeeper guidance


def test_the_readme_explains_opening_a_preview_build(readme_text):
    assert "Privacy & Security" in readme_text
    assert "Open Anyway" in readme_text
    assert "not notarized" in readme_text


def test_the_readme_says_right_click_open_no_longer_works(readme_text):
    """It was the old advice, so the guide has to say it stopped working."""
    flowed = " ".join(readme_text.split()).lower()

    assert "right-clicking the application and choosing open no longer" in flowed


def test_the_readme_warns_against_third_party_unzip_tools(readme_text):
    flowed = " ".join(readme_text.split()).lower()

    assert "do not unzip it with a third-party tool" in flowed


def test_the_readme_no_longer_hardcodes_a_test_count(readme_text):
    assert "449 tests" not in readme_text
    assert not re.search(r"\b\d{3} tests\b", readme_text)


def test_the_readme_says_the_packaged_app_needs_no_python(readme_text):
    section = readme_text.split("## Requirements")[1].split("##")[0]
    assert "Nothing else" in section


# docs/BUILDING.md carried this checklist until fe49668 removed it. Three of
# its five items are already checked against the artifacts that own them:
# MLX_TRANSCRIPT_FFMPEG by test_the_spec_can_be_pointed_at_a_specific_ffmpeg,
# sequesterRsrc by test_the_archive_sequesters_resource_forks, and the GPL
# flags by test_the_recipe_disables_gpl_and_nonfree. The two that lived only in the guide are
# pinned here, against the notices a user reads and the script that runs.


def test_the_notices_name_the_lgpl_configure_flags():
    """The licence claim has to name what the build actually did."""
    text = NOTICES.read_text(encoding="utf-8")

    assert "--disable-gpl" in text
    assert "--disable-nonfree" in text


def test_the_release_prep_records_the_model_digests():
    text = PREP_SCRIPT.read_text(encoding="utf-8")

    assert "record_model_digests" in text


def test_the_digest_recorder_exists():
    script = PROJECT_ROOT / "scripts" / "record_model_digests.py"

    assert script.is_file()
    assert "sha256" in script.read_text(encoding="utf-8")
