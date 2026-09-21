"""The build scripts have to run on the Bash macOS actually ships.

The FFmpeg stage failed before it did anything:

    bash scripts/finish_release_prep.sh ffmpeg
    scripts/finish_release_prep.sh: line 18: A: unbound variable

Line 18 was ``project_root="${0:A:h:h}"``. That is zsh: ``:A`` makes a path
absolute and ``:h`` takes its directory. The script carried a zsh shebang, so
running it as ``bash script`` bypassed the shebang, and Bash read ``${0:A:h}``
as a substring expansion whose offset is the variable ``A``. Under ``set -u``
an unset variable is fatal, so it died on startup.

Both scripts are now plain Bash targeting 3.2, which is what ``/bin/bash``
still is on current macOS. These tests keep them that way: a static sweep for
zsh expansions and for everything Bash 4 added, plus running the non-build
paths under a real Bash.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((PROJECT_ROOT / "scripts").glob("*.sh"))
PREP = PROJECT_ROOT / "scripts" / "finish_release_prep.sh"
BASH = shutil.which("bash")

#: Expansions and builtins that only zsh understands. Each one is paired with
#: what to use instead, because the message is what a future reader needs.
ZSH_ONLY = (
    (r"\$\{0:A", "${0:A...} is zsh; use $(cd \"$(dirname \"$0\")\" && pwd)"),
    (r":h:h", ":h is a zsh path modifier; use dirname"),
    (r"\$\{\([A-Za-z@]", "${(...)} is zsh parameter flags"),
    (r"\(@f\)", "(@f) is zsh line splitting; use a while read loop"),
    (r"\$\{\(F\)", "${(F)} is zsh array joining; use printf '%s\\n'"),
    (r"^\s*print -r", "print -r is zsh; use printf"),
    (r"\bautoload\b", "autoload is zsh"),
    (r"\bsetopt\b", "setopt is zsh"),
)

#: Features Bash added after 3.2. macOS never shipped a newer Bash, because
#: 4.0 moved to GPLv3.
BASH_4_ONLY = (
    (r"\bmapfile\b", "mapfile is Bash 4; use a while read loop"),
    (r"\breadarray\b", "readarray is Bash 4; use a while read loop"),
    (r"declare\s+-A", "associative arrays are Bash 4"),
    (r"local\s+-A", "associative arrays are Bash 4"),
    (r"typeset\s+-A", "associative arrays are Bash 4"),
    (r"\$\{[A-Za-z_][A-Za-z0-9_]*,,", "${var,,} is Bash 4; use tr"),
    (r"\$\{[A-Za-z_][A-Za-z0-9_]*\^\^", "${var^^} is Bash 4; use tr"),
    (r"\$\{[A-Za-z_][A-Za-z0-9_]*\[-[0-9]", "negative subscripts are Bash 4.3"),
    (r"\bcoproc\b", "coproc is Bash 4"),
    (r";;&", ";;& in a case statement is Bash 4"),
    (r"\bwait\s+-n\b", "wait -n is Bash 4.3"),
    (r"&>>", "&>> is Bash 4"),
)


def script_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def offending_lines(text: str, pattern: str) -> list[tuple[int, str]]:
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        # A line that only documents the construct is fine, and both scripts
        # explain the bug they were fixed for.
        if stripped.startswith("#"):
            continue
        if re.search(pattern, line):
            found.append((number, line.rstrip()))
    return found


# ------------------------------------------------------------ static sweep


def test_there_are_scripts_to_check():
    assert SCRIPTS, "no shell scripts found under scripts/"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_the_shebang_is_bash(script):
    first = script_text(script).splitlines()[0]

    assert first.startswith("#!"), f"{script.name} has no shebang"
    assert "zsh" not in first, (
        f"{script.name} still has a zsh shebang. Running it as "
        "'bash script' bypasses the shebang, which is how the zsh-only "
        "${0:A:h:h} produced 'A: unbound variable'."
    )
    assert "bash" in first, f"{script.name} should be a bash script: {first}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
@pytest.mark.parametrize("pattern,advice", ZSH_ONLY, ids=lambda v: str(v)[:28])
def test_no_zsh_only_construct(script, pattern, advice):
    hits = offending_lines(script_text(script), pattern)

    assert not hits, f"{script.name}: {advice}\n" + "\n".join(
        f"  line {number}: {line}" for number, line in hits
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
@pytest.mark.parametrize("pattern,advice", BASH_4_ONLY, ids=lambda v: str(v)[:28])
def test_no_bash_4_only_feature(script, pattern, advice):
    hits = offending_lines(script_text(script), pattern)

    assert not hits, f"{script.name}: {advice}\n" + "\n".join(
        f"  line {number}: {line}" for number, line in hits
    )


def test_the_array_count_is_written_the_bash_way():
    """In zsh ${#arr} is the element count; in Bash it is len(arr[0])."""
    text = script_text(PREP)

    assert "${#configure_args[@]}" in text
    assert "${#configure_args}" not in text.replace("${#configure_args[@]}", "")


def test_a_possibly_empty_array_is_expanded_defensively():
    """Bash 3.2 through 4.3 treat "${a[@]}" on an empty array as unbound."""
    text = script_text(PREP)

    assert '${configure_args[@]+"${configure_args[@]}"}' in text, (
        "expand a possibly-empty array as ${a[@]+\"${a[@]}\"} so set -u "
        "does not abort on Bash 3.2"
    )


def test_the_array_is_read_from_a_file_not_a_pipe():
    """A pipe into "while read" runs in a subshell, losing the array."""
    text = script_text(PREP)

    assert "< build/ffmpeg-configure-args.txt" in text


def test_every_variable_used_at_startup_has_a_default():
    """set -u is kept, so startup reads must all be guarded."""
    text = script_text(PREP)
    # Everything before the first function definition runs on every call.
    # Comments are dropped first: the header explains the Bash 4 expansions
    # this script avoids, and naming them is not using them.
    startup = "\n".join(
        line
        for line in text.split("log() {")[0].splitlines()
        if not line.strip().startswith("#")
    )

    for expansion in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)([^}]*)\}", startup):
        name, rest = expansion
        if name in {"BASH_VERSION"} or rest.startswith((":-", ":=", "+", ":?")):
            continue
        pytest.fail(
            f"${{{name}{rest}}} runs at startup without a default; "
            "set -u will abort if it is unset"
        )


def test_the_shell_guard_precedes_pipefail():
    """A non-Bash shell rejects "set -o pipefail" before reaching the guard."""
    text = script_text(PREP)

    assert text.index("BASH_VERSION") < text.index("set -euo pipefail")


def test_set_u_is_still_on():
    for script in SCRIPTS:
        assert "set -euo pipefail" in script_text(script), script.name


# --------------------------------------------------------- run it for real


@pytest.mark.skipif(BASH is None, reason="bash is not installed")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_the_script_parses_under_bash(script):
    result = subprocess.run(
        [BASH, "-n", str(script)], capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stderr


def run_prep(*arguments: str) -> subprocess.CompletedProcess:
    """Run the prep script the way the failure was reported: via bash."""
    environment = dict(os.environ)
    # A prefix that cannot exist, so no stage can find a built FFmpeg and
    # decide to do real work.
    environment["MLX_TRANSCRIPT_FFMPEG_PREFIX"] = "/nonexistent/mlx-test-prefix"
    return subprocess.run(
        [BASH, str(PREP), *arguments],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
        env=environment,
    )


@pytest.mark.skipif(BASH is None, reason="bash is not installed")
def test_the_reported_startup_failure_is_gone():
    """The exact regression: 'A: unbound variable' before anything happens."""
    result = run_prep("help")
    combined = result.stdout + result.stderr

    assert "unbound variable" not in combined, combined
    assert result.returncode == 0, combined


@pytest.mark.skipif(BASH is None, reason="bash is not installed")
def test_help_describes_the_stages():
    result = run_prep("help")

    for stage in ("preflight", "tests", "ffmpeg", "build"):
        assert stage in result.stdout


@pytest.mark.skipif(BASH is None, reason="bash is not installed")
def test_an_unknown_stage_says_so_clearly():
    result = run_prep("not-a-stage")
    combined = result.stdout + result.stderr

    assert result.returncode == 1
    assert "Unknown stage: not-a-stage" in combined
    assert "unbound variable" not in combined


@pytest.mark.skipif(BASH is None, reason="bash is not installed")
def test_preflight_never_dies_on_an_unset_variable():
    """It may report problems on this machine; it must not crash."""
    result = run_prep("preflight")
    combined = result.stdout + result.stderr

    assert "unbound variable" not in combined, combined
    assert "Preflight" in combined
    # Whatever it concludes, it has to say why rather than abort.
    assert result.returncode in (0, 1), combined


@pytest.mark.skipif(BASH is None, reason="bash is not installed")
def test_the_ffmpeg_stage_refuses_a_missing_prefix_without_crashing():
    """With no FFmpeg and no nasm it must stop with a readable reason."""
    result = run_prep("ffmpeg")
    combined = result.stdout + result.stderr

    assert "unbound variable" not in combined, combined
    if result.returncode != 0:
        assert "nasm is required" in combined or "configure flags" in combined


@pytest.mark.skipif(BASH is None, reason="bash is not installed")
def test_a_non_bash_shell_is_told_what_to_run():
    """dash and friends reject 'set -o pipefail'; the guard comes first."""
    for candidate in ("dash", "sh"):
        shell = shutil.which(candidate)
        if shell is None:
            continue
        result = subprocess.run(
            [shell, str(PREP), "help"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        combined = result.stdout + result.stderr
        if "Illegal option" in combined or "pipefail" in combined:
            pytest.fail(f"{candidate} reached set -o pipefail before the guard")
        assert "needs Bash" in combined or result.returncode == 0, combined


# ------------------------------------------- the configure arguments survive


@pytest.mark.skipif(BASH is None, reason="bash is not installed")
def fake_source_tree(tmp_path: Path) -> Path:
    """A stand-in whose configure answers --list-muxers, so no download."""
    tree = tmp_path / "ffmpeg-fake"
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
    return tree


@pytest.mark.skipif(BASH is None, reason="bash is not installed")
def test_reading_the_configure_flags_keeps_every_one(tmp_path):
    """The while-read loop must not lose the array to a subshell."""
    harness = tmp_path / "harness.sh"
    source = PREP.read_text(encoding="utf-8")
    bodies = [
        re.search(rf"{name}\(\) \{{.*?\n\}}", source, re.S)
        for name in (
            "read_configure_args",
            "assert_configure_args_are_clean",
            "assert_disables_precede_enables",
        )
    ]
    assert all(bodies), "the configure-argument functions could not be located"
    tree = fake_source_tree(tmp_path)
    harness.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f'cd "{PROJECT_ROOT}"\n'
        'python_bin() { command -v python3; }\n'
        'fail() { printf "FAIL %s\\n" "$*" >&2; exit 1; }\n'
        f'requirements="{PROJECT_ROOT}/packaging/ffmpeg_requirements.py"\n'
        'ffmpeg_prefix=/tmp/mlx-harness\n'
        'mkdir -p build\n'
        + "\n".join(body.group(0) for body in bodies)
        + "\n"
        f'read_configure_args "{tree}"\n'
        'printf "%s\\n" "${#configure_args[@]}"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [BASH, str(harness)], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
    count = int(result.stdout.strip().splitlines()[-1])
    assert count > 20, f"only {count} flags survived the read loop"
