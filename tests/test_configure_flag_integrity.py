"""The flags that reach FFmpeg's configure have to be exactly what was recorded.

Two separate defects produced the same symptom, an FFmpeg that reports its
version happily and then fails on the first real clip with::

    Requested output format 's16le' is not known.

The first was a comma-separated value, which configure ignores. The second
survived that fix: ``--enable-muxer=s16le`` names the *format*, but the
*component* configure knows is ``pcm_s16le``, so the flag matched nothing and
only warned. A third shape of the same failure is a flag arriving as
``\\--enable-muxer=s16le``, which configure also accepts and also ignores.

None of those are loud. configure prints one warning line in a log thousands
of lines long, exits 0, and the build runs to completion. So the checks here
are the ones that have to catch it: the flags are well formed, the muxer
components are resolved against the source tree rather than assumed, and
after configure has run its own generated config.h is read back before a
single object file is compiled.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = PROJECT_ROOT / "packaging" / "ffmpeg_requirements.py"
PREP = PROJECT_ROOT / "scripts" / "finish_release_prep.sh"
BUILDING = PROJECT_ROOT / "docs" / "BUILDING.md"


def load_requirements():
    specification = importlib.util.spec_from_file_location(
        "ffmpeg_requirements_integrity", REQUIREMENTS
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


req = load_requirements()


# ------------------------------------------------------- every flag is clean


def test_no_generated_flag_begins_with_a_backslash():
    """The reported symptom: '\\--enable-muxer=s16le' in the args file."""
    for flag in req.configure_arguments():
        assert not flag.startswith("\\"), f"{flag!r} begins with a backslash"


def test_no_generated_flag_contains_a_backslash_anywhere():
    for flag in req.configure_arguments():
        assert "\\" not in flag, f"{flag!r} contains a backslash"


def test_every_generated_flag_begins_with_exactly_two_hyphens():
    for flag in req.configure_arguments():
        assert flag.startswith("--"), f"{flag!r} does not begin with '--'"
        assert not flag.startswith("---"), f"{flag!r} has too many hyphens"


def test_no_generated_flag_carries_stray_whitespace_or_quotes():
    for flag in req.configure_arguments():
        assert flag == flag.strip(), f"{flag!r} has surrounding whitespace"
        for character in ('"', "'", "`", "\n", "\t"):
            assert character not in flag, f"{flag!r} contains {character!r}"


@pytest.mark.parametrize(
    "damaged",
    [
        "\\--enable-muxer=s16le",
        "\\--disable-gpl",
        "--enable-muxer=s16le\\",
        "--enable-muxer=s16\\le",
        "-enable-muxer=s16le",
        "enable-muxer=s16le",
        " --enable-muxer=s16le",
        '--enable-muxer="s16le"',
    ],
)
def test_a_damaged_flag_is_rejected(damaged):
    with pytest.raises(req.MediaToolError) as caught:
        req.assert_flags_are_well_formed(["--disable-gpl", damaged])

    message = str(caught.value)
    assert "Refusing to run configure" in message
    # repr() escapes a backslash, so compare on the flag's visible name.
    assert damaged.strip().lstrip("\\-").split("=")[0] in message


def test_the_real_recipe_passes_its_own_check():
    req.assert_flags_are_well_formed(req.configure_arguments())


def test_a_backslash_is_named_as_such_in_the_message():
    with pytest.raises(req.MediaToolError) as caught:
        req.assert_flags_are_well_formed(["\\--enable-muxer=s16le"])

    assert "backslash" in str(caught.value)


# ------------------------------------------- the muxer component resolution


def test_the_pcm_muxers_are_enabled_by_their_component_name():
    """The second defect. The format is s16le; the component is pcm_s16le."""
    flags = req.configure_arguments()

    assert "--enable-muxer=pcm_s16le" in flags
    assert "--enable-muxer=pcm_f32le" in flags
    assert "--enable-muxer=s16le" not in flags
    assert "--enable-muxer=f32le" not in flags


def test_the_format_names_are_still_what_a_built_ffmpeg_is_asked_for():
    """-muxers and -f use the format name, so that list must not change."""
    assert req.REQUIRED_MUXERS == ("s16le", "f32le", "wav", "null")
    assert [name for name, _, _ in req.PCM_OUTPUT_FORMATS] == ["s16le", "f32le"]


def test_wav_and_null_are_spelled_the_same_either_way():
    components = req.resolve_muxer_components()

    assert components["wav"] == "wav"
    assert components["null"] == "null"


def test_every_required_muxer_has_a_candidate_list():
    for muxer in req.REQUIRED_MUXERS:
        assert muxer in req.MUXER_COMPONENT_CANDIDATES
        assert req.MUXER_COMPONENT_CANDIDATES[muxer]


def test_a_source_tree_listing_picks_the_name_it_actually_has():
    components = req.resolve_muxer_components(
        {"pcm_s16le", "pcm_f32le", "wav", "null", "mov", "matroska"}
    )

    assert components == {
        "s16le": "pcm_s16le",
        "f32le": "pcm_f32le",
        "wav": "wav",
        "null": "null",
    }


def test_a_tree_using_the_bare_name_is_also_handled():
    """If a future FFmpeg renamed the component, the fallback still works."""
    components = req.resolve_muxer_components({"s16le", "f32le", "wav", "null"})

    assert components["s16le"] == "s16le"
    assert components["f32le"] == "f32le"


def test_a_tree_missing_the_muxer_entirely_is_refused():
    with pytest.raises(req.MediaToolError) as caught:
        req.resolve_muxer_components({"wav", "null"})

    message = str(caught.value)
    assert "s16le" in message
    assert "pcm_s16le" in message


def test_generating_the_flags_against_a_tree_that_lacks_a_muxer_fails():
    with pytest.raises(req.MediaToolError):
        req.configure_arguments(available_muxers={"wav"})


# -------------------------------------------------- the post-configure check


def write_config_header(tree: Path, enabled: dict[str, int]) -> None:
    (tree / "ffbuild").mkdir(parents=True, exist_ok=True)
    lines = ["/* Automatically generated by configure - do not modify! */"]
    for name, value in enabled.items():
        lines.append(f"#define CONFIG_{name.upper()}_MUXER {value}")
    (tree / "ffbuild" / "config.h").write_text("\n".join(lines), encoding="utf-8")


def test_the_enabled_muxers_are_read_out_of_the_generated_header(tmp_path):
    write_config_header(tmp_path, {"pcm_s16le": 1, "pcm_f32le": 1, "wav": 1, "mp4": 0})

    enabled = req.configured_muxer_components(tmp_path)

    assert {"pcm_s16le", "pcm_f32le", "wav"} <= enabled
    assert "mp4" not in enabled


def test_a_header_at_the_old_top_level_path_is_also_read(tmp_path):
    (tmp_path / "config.h").write_text(
        "#define CONFIG_PCM_S16LE_MUXER 1\n", encoding="utf-8"
    )

    assert "pcm_s16le" in req.configured_muxer_components(tmp_path)


def test_no_header_at_all_says_configure_has_not_run(tmp_path):
    with pytest.raises(req.MediaToolError) as caught:
        req.configured_muxer_components(tmp_path)

    assert "configure has not run" in str(caught.value)


def fake_source_tree(tmp_path: Path, listed: str, enabled: dict[str, int]) -> Path:
    """A stand-in tree whose configure answers --list-muxers."""
    tree = tmp_path / "ffmpeg-fake"
    tree.mkdir()
    configure = tree / "configure"
    configure.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--list-muxers" ]; then\n'
        f'  printf "%s\\n" "{listed}"\n'
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    configure.chmod(0o755)
    write_config_header(tree, enabled)
    return tree


def run_cli(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REQUIREMENTS), *arguments],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_check_passes_when_configure_really_enabled_them(tmp_path):
    tree = complete_tree(tmp_path)

    result = run_cli("--check-configured", str(tree))

    assert result.returncode == 0, result.stderr
    assert "Safe to compile" in result.stdout
    assert "muxer s16le enabled as pcm_s16le" in result.stdout


def test_the_check_fails_before_compilation_when_a_muxer_is_off(tmp_path):
    """Exactly the state the last build compiled for forty minutes in."""
    tree = fake_source_tree(
        tmp_path,
        "pcm_s16le pcm_f32le wav null",
        {"pcm_s16le": 0, "pcm_f32le": 0, "wav": 1, "null": 1},
    )

    result = run_cli("--check-configured", str(tree))

    assert result.returncode == 1
    assert "Safe to compile" not in result.stdout
    assert "pcm_s16le" in result.stderr
    assert "'s16le' is not known" in result.stderr


def test_the_check_names_the_config_symbol_it_looked_for(tmp_path):
    tree = fake_source_tree(
        tmp_path, "pcm_s16le pcm_f32le wav null", {"wav": 1, "null": 1}
    )

    result = run_cli("--check-configured", str(tree))

    assert result.returncode == 1
    assert "CONFIG_PCM_S16LE_MUXER" in result.stderr


def test_the_check_refuses_a_directory_that_is_not_a_source_tree(tmp_path):
    result = run_cli("--check-configured", str(tmp_path))

    assert result.returncode == 1
    assert "source tree" in result.stderr


def test_listing_the_components_of_a_tree_works(tmp_path):
    tree = fake_source_tree(tmp_path, "pcm_s16le wav", {"wav": 1})

    result = run_cli("--list-muxer-components", str(tree))

    assert result.returncode == 0
    assert result.stdout.split() == ["pcm_s16le", "wav"]


def test_generating_against_a_tree_uses_that_trees_spelling(tmp_path):
    tree = fake_source_tree(
        tmp_path, "s16le f32le wav null", {"s16le": 1, "f32le": 1, "wav": 1, "null": 1}
    )

    result = run_cli("--configure-args", "--source-tree", str(tree))

    assert result.returncode == 0, result.stderr
    assert "--enable-muxer=s16le" in result.stdout.splitlines()
    assert "--enable-muxer=pcm_s16le" not in result.stdout.splitlines()


def test_generating_against_a_tree_without_the_muxer_fails_loudly(tmp_path):
    tree = fake_source_tree(tmp_path, "wav null", {"wav": 1, "null": 1})

    result = run_cli("--configure-args", "--source-tree", str(tree))

    assert result.returncode == 1
    assert "s16le" in result.stderr


# ----------------------------------------------------- the build script path


def test_the_script_validates_the_flags_before_configure():
    text = PREP.read_text(encoding="utf-8")

    assert "assert_configure_args_are_clean" in text
    body = text.split("assert_configure_args_are_clean() {", 1)[1]
    assert "begins with a backslash" in body
    assert "does not begin with --" in body


def test_the_flag_check_runs_before_configure_is_invoked():
    text = PREP.read_text(encoding="utf-8")

    assert text.index("assert_configure_args_are_clean\n") < text.index("./configure ")


def test_the_script_compares_the_loaded_flags_against_the_recorded_file():
    """Byte-for-byte, so the recorded file is evidence rather than a guess."""
    text = PREP.read_text(encoding="utf-8")

    assert "cmp -s build/ffmpeg-configure-args.txt" in text


def test_the_script_resolves_the_components_against_the_source_tree():
    text = PREP.read_text(encoding="utf-8")

    assert "--source-tree" in text
    assert 'read_configure_args "$tree"' in text


def test_the_script_checks_the_generated_configuration_before_compiling():
    text = PREP.read_text(encoding="utf-8")

    assert "--check-configured" in text
    assert text.index("--check-configured") < text.index("make -j")


def test_the_script_treats_a_did_not_match_warning_as_fatal():
    text = PREP.read_text(encoding="utf-8")

    assert "did not match anything" in text
    assert text.index("did not match anything") < text.index("make -j")


def test_the_script_reads_the_flags_with_read_r():
    """Without -r, read would consume a backslash rather than report it."""
    text = PREP.read_text(encoding="utf-8")

    assert "while IFS= read -r line" in text


def test_the_flag_check_is_bash_3_2_compatible():
    text = PREP.read_text(encoding="utf-8")
    body = text.split("assert_configure_args_are_clean() {", 1)[1].split("\n}", 1)[0]

    for feature in ("mapfile", "readarray", "declare -A", "[-1]", ",,}"):
        assert feature not in body, f"{feature} is not in Bash 3.2"


# ------------------------------------------------------------ documentation


def test_the_documentation_shows_the_component_spelling():
    text = BUILDING.read_text(encoding="utf-8")

    assert "--enable-muxer=pcm_s16le" in text
    assert "--enable-muxer=pcm_f32le" in text


def test_the_documentation_explains_the_two_spellings():
    text = BUILDING.read_text(encoding="utf-8")

    assert "did not match anything" in text
    assert "component" in text


# ------------------------------------------------------------- flag ordering


def test_the_broad_disables_come_before_every_selective_enable():
    """configure applies options in order, so a late disable undoes them."""
    flags = req.configure_arguments()

    for broad, selective in req.BROAD_DISABLE_FLAGS:
        assert broad in flags, f"{broad} is missing from the recipe"
        broad_at = flags.index(broad)
        for position, flag in enumerate(flags):
            if flag.startswith(selective):
                assert position > broad_at, (
                    f"{flag} at {position} comes before {broad} at "
                    f"{broad_at}, which would undo it"
                )


@pytest.mark.parametrize(
    "broad,selective",
    [
        ("--disable-encoders", "--enable-encoder=pcm_s16le"),
        ("--disable-muxers", "--enable-muxer=pcm_s16le"),
        ("--disable-filters", "--enable-filter=aresample"),
    ],
)
def test_a_broad_disable_after_its_enable_is_rejected(broad, selective):
    with pytest.raises(req.MediaToolError) as caught:
        req.assert_disables_precede_enables(["--disable-gpl", selective, broad])

    message = str(caught.value)
    assert broad in message
    assert selective in message
    assert "undo" in message


def test_the_correct_order_passes_the_check():
    req.assert_disables_precede_enables(
        ["--disable-muxers", "--enable-muxer=pcm_s16le", "--enable-muxer=wav"]
    )


def test_a_recipe_without_the_broad_disable_is_not_flagged():
    """Nothing to undo, so nothing to complain about."""
    req.assert_disables_precede_enables(["--enable-muxer=pcm_s16le"])


def test_every_broad_disable_is_paired_with_its_selective_prefix():
    assert req.BROAD_DISABLE_FLAGS == (
        ("--disable-encoders", "--enable-encoder="),
        ("--disable-muxers", "--enable-muxer="),
        ("--disable-filters", "--enable-filter="),
    )


def test_generating_the_flags_runs_the_order_check_itself():
    """configure_arguments must not be able to return a cancelling order."""
    source = REQUIREMENTS.read_text(encoding="utf-8")
    body = source.split("def configure_arguments(", 1)[1].split("\ndef ", 1)[0]

    assert "assert_disables_precede_enables(arguments)" in body


def test_the_script_checks_the_order_of_the_array_not_the_file():
    text = PREP.read_text(encoding="utf-8")
    body = text.split("assert_disables_precede_enables() {", 1)[1].split("\n}", 1)[0]

    # It has to index the array configure is handed, not re-read the file.
    assert "${configure_args[$index]}" in body
    assert "ffmpeg-configure-args.txt" not in body


def test_the_order_check_runs_before_configure():
    text = PREP.read_text(encoding="utf-8")

    assert text.index("assert_disables_precede_enables\n") < text.index("./configure ")


def test_the_order_check_is_bash_3_2_compatible():
    text = PREP.read_text(encoding="utf-8")
    body = text.split("assert_disables_precede_enables() {", 1)[1].split("\n}", 1)[0]

    for feature in ("declare -A", "mapfile", "readarray", "[-1]", ",,}"):
        assert feature not in body, f"{feature} is not in Bash 3.2"


# ------------------------------------- the gate covers all three kinds now


def test_the_gate_checks_encoders_muxers_and_filters():
    assert req.CHECKED_COMPONENT_KINDS == ("ENCODER", "MUXER", "FILTER")


def test_the_required_components_are_grouped_by_kind():
    wanted = req.required_components_by_kind()

    assert set(wanted["ENCODER"]) == set(req.REQUIRED_ENCODERS)
    assert set(wanted["MUXER"]) == set(req.REQUIRED_MUXERS)
    assert set(wanted["FILTER"]) == set(req.REQUIRED_FILTERS)
    # Only the muxers have a second spelling.
    assert wanted["MUXER"]["s16le"] == "pcm_s16le"
    assert wanted["ENCODER"]["pcm_s16le"] == "pcm_s16le"
    assert wanted["FILTER"]["aresample"] == "aresample"


def write_components_header(
    tree: Path,
    muxers: dict[str, int] | None = None,
    encoders: dict[str, int] | None = None,
    filters: dict[str, int] | None = None,
    filename: str = "config_components.h",
) -> None:
    """Write the header layout FFmpeg 5.1 and later actually generate."""
    lines = ["/* Automatically generated by configure - do not modify! */"]
    for kind, values in (
        ("MUXER", muxers or {}),
        ("ENCODER", encoders or {}),
        ("FILTER", filters or {}),
    ):
        for name, value in values.items():
            lines.append(f"#define CONFIG_{name.upper()}_{kind} {value}")
    (tree / filename).write_text("\n".join(lines), encoding="utf-8")


def complete_tree(tmp_path: Path) -> Path:
    tree = tmp_path / "ffmpeg-complete"
    tree.mkdir()
    configure = tree / "configure"
    configure.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--list-muxers" ]; then\n'
        '  printf "%s\\n" "pcm_s16le pcm_f32le wav null"\n'
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    configure.chmod(0o755)
    # config.h carries the global settings only, which is the layout that
    # made the gate report every component disabled.
    (tree / "config.h").write_text(
        "#define CONFIG_STATIC 1\n#define CONFIG_SHARED 0\n", encoding="utf-8"
    )
    write_components_header(
        tree,
        muxers={"pcm_s16le": 1, "pcm_f32le": 1, "wav": 1, "null": 1},
        encoders={"pcm_s16le": 1, "pcm_f32le": 1},
        filters={name: 1 for name in req.REQUIRED_FILTERS},
    )
    return tree


def test_the_components_header_is_where_modern_ffmpeg_records_them():
    """FFmpeg 5.1 moved them out of config.h; reading only that found none."""
    assert "config_components.h" in req.GENERATED_CONFIG_HEADERS
    assert req.GENERATED_CONFIG_HEADERS.index("config_components.h") < (
        req.GENERATED_CONFIG_HEADERS.index("config.h")
    )


def test_a_complete_tree_in_the_modern_layout_passes(tmp_path):
    """The exact false negative: config.h present, components elsewhere."""
    tree = complete_tree(tmp_path)

    result = run_cli("--check-configured", str(tree))

    assert result.returncode == 0, result.stderr
    assert "Safe to compile" in result.stdout
    for line in ("muxer s16le enabled as pcm_s16le", "encoder pcm_s16le enabled",
                 "filter aresample enabled"):
        assert line in result.stdout


def test_a_missing_encoder_fails_the_gate(tmp_path):
    tree = complete_tree(tmp_path)
    write_components_header(
        tree,
        muxers={"pcm_s16le": 1, "pcm_f32le": 1, "wav": 1, "null": 1},
        encoders={"pcm_s16le": 1, "pcm_f32le": 0},
        filters={name: 1 for name in req.REQUIRED_FILTERS},
    )

    result = run_cli("--check-configured", str(tree))

    assert result.returncode == 1
    assert "CONFIG_PCM_F32LE_ENCODER" in result.stderr


def test_a_missing_filter_fails_the_gate(tmp_path):
    tree = complete_tree(tmp_path)
    write_components_header(
        tree,
        muxers={"pcm_s16le": 1, "pcm_f32le": 1, "wav": 1, "null": 1},
        encoders={"pcm_s16le": 1, "pcm_f32le": 1},
        filters={"aresample": 1, "anull": 1, "aformat": 1, "atrim": 1, "copy": 0},
    )

    result = run_cli("--check-configured", str(tree))

    assert result.returncode == 1
    assert "CONFIG_COPY_FILTER" in result.stderr


def test_the_old_single_header_layout_still_works(tmp_path):
    """An older tree puts everything in config.h; both are read and merged."""
    tree = complete_tree(tmp_path)
    (tree / "config_components.h").unlink()
    write_components_header(
        tree,
        muxers={"pcm_s16le": 1, "pcm_f32le": 1, "wav": 1, "null": 1},
        encoders={"pcm_s16le": 1, "pcm_f32le": 1},
        filters={name: 1 for name in req.REQUIRED_FILTERS},
        filename="config.h",
    )

    result = run_cli("--check-configured", str(tree))

    assert result.returncode == 0, result.stderr


def test_a_header_under_ffbuild_is_also_read(tmp_path):
    tree = complete_tree(tmp_path)
    (tree / "config.h").unlink()
    (tree / "config_components.h").unlink()
    (tree / "ffbuild").mkdir()
    write_components_header(
        tree / "ffbuild",
        muxers={"pcm_s16le": 1, "pcm_f32le": 1, "wav": 1, "null": 1},
        encoders={"pcm_s16le": 1, "pcm_f32le": 1},
        filters={name: 1 for name in req.REQUIRED_FILTERS},
    )

    result = run_cli("--check-configured", str(tree))

    assert result.returncode == 0, result.stderr


def test_no_header_anywhere_says_configure_has_not_run(tmp_path):
    tree = complete_tree(tmp_path)
    (tree / "config.h").unlink()
    (tree / "config_components.h").unlink()

    result = run_cli("--check-configured", str(tree))

    assert result.returncode == 1
    assert "configure has not run" in result.stderr


def test_a_kind_can_be_read_on_its_own(tmp_path):
    tree = complete_tree(tmp_path)

    assert "pcm_s16le" in req.configured_components(tree, "ENCODER")
    assert "aresample" in req.configured_components(tree, "FILTER")
    assert "pcm_s16le" in req.configured_muxer_components(tree)


def test_the_script_names_the_broader_log_file():
    text = PREP.read_text(encoding="utf-8")

    assert "ffmpeg-configured-components.txt" in text
    assert "encoders, muxers and filters" in text
