#!/bin/bash
# Finish the release preparation on this Mac.
#
# Everything in this script needs macOS, Apple silicon, or network access to
# Hugging Face, which is why it is not already done. Run the stages in order.
# Each one writes a log under build/ and stops on the first failure.
#
#   scripts/finish_release_prep.sh help       # what each stage does
#   scripts/finish_release_prep.sh preflight  # check prerequisites, build nothing
#   scripts/finish_release_prep.sh tests      # ~10 seconds
#   scripts/finish_release_prep.sh digests    # ~1 minute, needs the network
#   scripts/finish_release_prep.sh ffmpeg     # 20-40 minutes, one time only
#   scripts/finish_release_prep.sh build      # ~5 minutes
#   scripts/finish_release_prep.sh all        # tests, ffmpeg, build
#
# Nothing here pushes, tags, or publishes anything.
#
# PORTABILITY
# -----------
# This targets the Bash that ships with macOS, which is 3.2 from 2007 and is
# what /bin/bash still is on current macOS. It therefore avoids everything
# Bash 4 added: no associative arrays, no mapfile or readarray, no ${var,,}
# case conversion, no negative array subscripts.
#
# It also avoids zsh expansions, which is what broke the first version. That
# one carried a zsh shebang and used ${0:A:h:h} to find its own directory.
# Run it as "bash script" and the shebang is bypassed, Bash reads ${0:A:h} as
# a substring expansion whose offset is the variable A, and set -u reports
# "A: unbound variable" on line 18 before any work starts.
#
# Bash 3.2 also errors on "${array[@]}" when the array is empty and set -u is
# on, so a possibly-empty array is expanded as ${array[@]+"${array[@]}"}.
#
# tests/test_shell_compatibility.py enforces all of this.

# Checked before "set -o pipefail", which a non-Bash shell rejects outright:
# otherwise the reader gets "Illegal option -o p" instead of being told what
# to run.
if [ -z "${BASH_VERSION:-}" ]; then
    printf '%s\n' "This script needs Bash. Run: bash scripts/finish_release_prep.sh" >&2
    exit 1
fi

set -euo pipefail

# Resolve the project root without zsh's :A:h modifiers.
script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_dir/.." && pwd)
cd "$project_root"
mkdir -p build

# Where the LGPL FFmpeg is built and installed. Outside the project, because
# it is a toolchain artifact rather than part of the application source.
ffmpeg_prefix="${MLX_TRANSCRIPT_FFMPEG_PREFIX:-$HOME/ffmpeg-lgpl}"
ffmpeg_version="${MLX_TRANSCRIPT_FFMPEG_VERSION:-7.1.1}"
requirements="$project_root/packaging/ffmpeg_requirements.py"

log() { printf '==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
fail() { printf '!!! %s\n' "$*" >&2; exit 1; }

python_bin() {
    # The stages that only need the standard library can use any Python 3.
    if [ -x "$project_root/.venv/bin/python" ]; then
        printf '%s\n' "$project_root/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        fail "No Python 3 found. See docs/BUILDING.md."
    fi
}

require_venv() {
    [ -x "$project_root/.venv/bin/python" ] \
        || fail "No .venv found at $project_root/.venv. See docs/BUILDING.md."
}

stage_help() {
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    printf '\n'
    log "Paths"
    note "project        $project_root"
    note "ffmpeg prefix  $ffmpeg_prefix"
    note "ffmpeg version $ffmpeg_version"
    note "logs           $project_root/build"
}

stage_preflight() {
    # Everything this script depends on, checked without building anything.
    log "Preflight"
    note "bash           $BASH_VERSION"
    note "project        $project_root"
    note "ffmpeg prefix  $ffmpeg_prefix"

    problems=0

    if [ -x "$project_root/.venv/bin/python" ]; then
        note ".venv          $("$project_root/.venv/bin/python" -V 2>&1)"
    else
        note ".venv          missing (needed by the tests and build stages)"
        problems=$((problems + 1))
    fi

    if [ -f "$requirements" ]; then
        if "$(python_bin)" "$requirements" --configure-args >/dev/null 2>&1; then
            note "configure args $("$(python_bin)" "$requirements" --configure-args | wc -l | tr -d ' ') flags available"
        else
            note "configure args could not be generated"
            problems=$((problems + 1))
        fi
    else
        note "requirements   $requirements is missing"
        problems=$((problems + 1))
    fi

    if [ -x "$ffmpeg_prefix/bin/ffmpeg" ]; then
        note "ffmpeg         already built at $ffmpeg_prefix/bin/ffmpeg"
    else
        note "ffmpeg         not built yet (run the ffmpeg stage)"
    fi

    if command -v nasm >/dev/null 2>&1; then
        note "nasm           $(command -v nasm)"
    else
        note "nasm           missing (brew install nasm pkg-config) for the ffmpeg stage"
    fi

    if [ "$(uname -m)" = "arm64" ]; then
        note "architecture   arm64"
    else
        note "architecture   $(uname -m); the build stage needs Apple silicon"
    fi

    printf '\n'
    if [ "$problems" -gt 0 ]; then
        fail "$problems problem(s) above have to be resolved first."
    fi
    log "Preflight passed"
}

stage_tests() {
    require_venv
    log "Running the full test suite in .venv"
    "$project_root/.venv/bin/python" -m pytest -q 2>&1 | tee build/pytest-macos.log
    log "Test log: build/pytest-macos.log"
}

stage_digests() {
    require_venv
    log "Recording the diarization model digests"
    log "This downloads about 31 MB to a temporary folder, then deletes it."
    "$project_root/.venv/bin/python" scripts/record_model_digests.py --write \
        2>&1 | tee build/model-digests.log
    log "transcription/model_cache.py now pins both assets."
    log "Re-run the tests after this: scripts/finish_release_prep.sh tests"
}

read_configure_args() {
    # $1 is an unpacked FFmpeg source tree, so the muxer component names are
    # resolved against that tree's own "configure --list-muxers" instead of
    # being assumed. A raw PCM muxer's format name is s16le but its component
    # is pcm_s16le, and asking for the format name enables nothing.
    source_tree="${1:-}"
    [ -n "$source_tree" ] || fail "read_configure_args needs a source tree."

    "$(python_bin)" "$requirements" --configure-args \
        --prefix "$ffmpeg_prefix" --source-tree "$source_tree" \
        > build/ffmpeg-configure-args.txt \
        || fail "Could not generate the configure flags from $requirements"

    # Read back with a redirect, not a pipe: a pipe into "while read" runs the
    # loop in a subshell in Bash, so the array would be empty by the time the
    # caller saw it. read -r keeps backslashes literal rather than letting
    # them escape the next character, so a flag cannot be silently reshaped
    # between the file and configure.
    configure_args=()
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        configure_args+=("$line")
    done < build/ffmpeg-configure-args.txt

    [ "${#configure_args[@]}" -gt 0 ] \
        || fail "No configure flags were generated. See build/ffmpeg-configure-args.txt"

    assert_configure_args_are_clean
}

assert_configure_args_are_clean() {
    # Every argument has to begin with exactly two hyphens. A leading
    # backslash is the specific damage this guards against: configure accepts
    # "\--enable-muxer=s16le", warns once in a log thousands of lines long,
    # and builds without the component. Nothing is compiled until this passes.
    bad=0
    index=0
    while [ "$index" -lt "${#configure_args[@]}" ]; do
        argument="${configure_args[$index]}"
        case "$argument" in
            "\\"*)
                printf '!!! flag %s begins with a backslash: %s\n' \
                    "$index" "$argument" >&2
                bad=1
                ;;
            --?*) : ;;
            *)
                printf '!!! flag %s does not begin with --: %s\n' \
                    "$index" "$argument" >&2
                bad=1
                ;;
        esac
        case "$argument" in
            *"\\"*)
                printf '!!! flag %s contains a backslash: %s\n' \
                    "$index" "$argument" >&2
                bad=1
                ;;
        esac
        index=$((index + 1))
    done

    [ "$bad" -eq 0 ] || fail "Refusing to run configure with damaged flags."

    # The arguments about to be passed to configure have to be byte-for-byte
    # the lines in the recorded file, so the recorded file is evidence of what
    # was actually used rather than a parallel guess.
    printf '%s\n' ${configure_args[@]+"${configure_args[@]}"} \
        > build/ffmpeg-configure-args.check.txt
    cmp -s build/ffmpeg-configure-args.txt build/ffmpeg-configure-args.check.txt \
        || fail "The loaded flags differ from build/ffmpeg-configure-args.txt. See build/ffmpeg-configure-args.check.txt"

    assert_disables_precede_enables
}

assert_disables_precede_enables() {
    # Checked on the array that is about to be expanded into ./configure, not
    # on the recorded file, because the array is what configure actually
    # receives. configure applies options in the order it reads them, so a
    # --disable-muxers sitting after --enable-muxer=pcm_s16le would undo it
    # without a word of complaint.
    #
    # Two plain arrays instead of one associative array, which is Bash 4.
    broad_flags="--disable-encoders --disable-muxers --disable-filters"
    out_of_order=0

    for broad in $broad_flags; do
        case "$broad" in
            --disable-encoders) selective="--enable-encoder=" ;;
            --disable-muxers)   selective="--enable-muxer=" ;;
            *)                  selective="--enable-filter=" ;;
        esac

        broad_at=-1
        index=0
        while [ "$index" -lt "${#configure_args[@]}" ]; do
            [ "${configure_args[$index]}" = "$broad" ] && { broad_at="$index"; break; }
            index=$((index + 1))
        done
        [ "$broad_at" -ge 0 ] || continue

        index=0
        while [ "$index" -lt "$broad_at" ]; do
            case "${configure_args[$index]}" in
                "$selective"*)
                    printf '!!! %s at position %s comes after %s at position %s; it would undo it\n' \
                        "$broad" "$broad_at" "${configure_args[$index]}" "$index" >&2
                    out_of_order=1
                    ;;
            esac
            index=$((index + 1))
        done
    done

    if [ "$out_of_order" -ne 0 ]; then
        printf '\n'
        printf '    the order configure would receive:\n' >&2
        index=0
        while [ "$index" -lt "${#configure_args[@]}" ]; do
            printf '      %2s %s\n' "$index" "${configure_args[$index]}" >&2
            index=$((index + 1))
        done
        fail "The configure flags are in an order that cancels itself."
    fi

}

stage_validate_ffmpeg() {
    # Licence, every required component, and two real decodes to raw PCM. A
    # build that passes this cannot fail the way the first one did.
    log "Validating: licence, components, and two real decodes to raw PCM"
    if ! "$(python_bin)" "$requirements" --verify \
        "$ffmpeg_prefix/bin/ffmpeg" "$ffmpeg_prefix/bin/ffprobe" \
        2>&1 | tee build/ffmpeg-validation.txt; then
        fail "This FFmpeg cannot do what MLX Transcript needs. See build/ffmpeg-validation.txt"
    fi
    "$ffmpeg_prefix/bin/ffmpeg" -version | grep '^ *configuration:' \
        > build/ffmpeg-lgpl-configuration.txt || true
}

stage_ffmpeg() {
    if [ -x "$ffmpeg_prefix/bin/ffmpeg" ] && [ -x "$ffmpeg_prefix/bin/ffprobe" ]; then
        log "An LGPL FFmpeg is already installed at $ffmpeg_prefix"
        log "Validating it rather than trusting it"
        stage_validate_ffmpeg
        return 0
    fi

    command -v nasm >/dev/null 2>&1 \
        || fail "nasm is required: brew install nasm pkg-config"

    # Every component is enabled as its own configure flag, generated from
    # packaging/ffmpeg_requirements.py so the recipe, the packaging gate and
    # the tests cannot drift apart.
    #
    # This is what the first standalone build got wrong. FFmpeg's configure
    # turns a comma-separated value into a shell case pattern and matches it
    # against component entries named "s16le_muxer" and "f32le_muxer", so
    # --enable-muxer=f32le,s16le,wav,null matched nothing, the warning
    # scrolled past in the build log, --disable-muxers won, and the app failed
    # on its first real clip with
    # "Requested output format 's16le' is not known."
    work=$(mktemp -d /private/tmp/mlx-ffmpeg-build.XXXXXX) \
        || fail "Could not create a build directory."
    log "Building FFmpeg $ffmpeg_version in $work"

    (
        cd "$work" || exit 1
        curl -LO "https://ffmpeg.org/releases/ffmpeg-${ffmpeg_version}.tar.xz" || exit 1
        tar xf "ffmpeg-${ffmpeg_version}.tar.xz" || exit 1
    ) 2>&1 | tee build/ffmpeg-fetch.log
    tree="$work/ffmpeg-${ffmpeg_version}"
    [ -f "$tree/configure" ] || fail "FFmpeg $ffmpeg_version did not unpack. See build/ffmpeg-fetch.log"

    # Resolved against this exact source tree, then checked for damage. Both
    # happen before configure runs, so a bad flag costs a second rather than a
    # 40-minute build and a broken app.
    read_configure_args "$tree"
    log "${#configure_args[@]} configure flags recorded in build/ffmpeg-configure-args.txt"

    log "This takes 20 to 40 minutes. Output goes to build/ffmpeg-build.log"
    jobs=$(sysctl -n hw.ncpu 2>/dev/null || printf '%s\n' 4)

    (
        cd "$tree" || exit 1
        ./configure ${configure_args[@]+"${configure_args[@]}"} || exit 1
    ) 2>&1 | tee build/ffmpeg-configure.log

    # configure's own record of what it enabled, read out of the generated
    # config.h. This is the gate the last build had no equivalent of: it
    # noticed nothing, compiled for 40 minutes, installed, and only the first
    # real clip revealed the muxers were missing.
    log "Confirming the required encoders, muxers and filters were enabled"
    "$(python_bin)" "$requirements" --check-configured "$tree" \
        2>&1 | tee build/ffmpeg-configured-components.txt
    grep -q 'Safe to compile' build/ffmpeg-configured-components.txt \
        || fail "configure did not enable everything required. See build/ffmpeg-configured-components.txt"

    if grep -q 'did not match anything' build/ffmpeg-configure.log; then
        printf '\n'
        grep 'did not match anything' build/ffmpeg-configure.log >&2
        fail "configure ignored a flag above. See build/ffmpeg-configure.log"
    fi

    (
        cd "$tree" || exit 1
        make -j"$jobs" || exit 1
        make install || exit 1
    ) 2>&1 | tee build/ffmpeg-build.log

    [ -x "$ffmpeg_prefix/bin/ffmpeg" ] || fail "The FFmpeg build did not install."
    stage_validate_ffmpeg
    log "Installed at $ffmpeg_prefix"
}

stage_build() {
    require_venv
    [ -x "$ffmpeg_prefix/bin/ffmpeg" ] \
        || fail "No LGPL FFmpeg at $ffmpeg_prefix. Run the ffmpeg stage first."

    MLX_TRANSCRIPT_FFMPEG="$ffmpeg_prefix/bin/ffmpeg"
    MLX_TRANSCRIPT_FFPROBE="$ffmpeg_prefix/bin/ffprobe"
    export MLX_TRANSCRIPT_FFMPEG MLX_TRANSCRIPT_FFPROBE

    log "Re-validating the media tools before packaging"
    "$(python_bin)" "$requirements" --verify \
        "$MLX_TRANSCRIPT_FFMPEG" "$MLX_TRANSCRIPT_FFPROBE" \
        || fail "The media tools no longer pass validation."

    log "Building the application with the LGPL media tools"
    bash "$project_root/scripts/build_macos.sh" 2>&1 | tee build/app-build.log

    version=$("$project_root/.venv/bin/python" -c 'import re, pathlib; print(re.search(r"^VERSION = \"([^\"]+)\"", pathlib.Path("app/version.py").read_text(), re.M).group(1))')
    archive="dist/MLX-Transcript-${version}-arm64.zip"

    log "Archive: $project_root/$archive"
    shasum -a 256 "$archive" | tee build/archive-sha256.txt

    printf '\n'
    log "What is verified, and what is not"
    note "verified: no AppleDouble files inside the extracted app"
    note "verified: codesign --verify --deep --strict on the extracted app"
    note "verified: bundled ffmpeg and ffprobe run from the bundle"
    note "verified: real mlx.core computation, tokenizer, numba path"
    note "verified: sherpa-onnx reported by the engine check"
    note "NOT verified: Gatekeeper acceptance of a quarantined download."
    note "              To test that, upload the archive somewhere, download"
    note "              it in Safari, unzip in Finder, and follow the"
    note "              Privacy & Security steps in README.md."
}

stage="${1:-all}"
case "$stage" in
    help | --help | -h) stage_help ;;
    preflight) stage_preflight ;;
    tests) stage_tests ;;
    digests) stage_digests ;;
    ffmpeg) stage_ffmpeg ;;
    build) stage_build ;;
    all)
        # digests is deliberately not here: it needs a large download and is
        # opt-in. Run it on its own when you want the assets pinned.
        stage_tests
        stage_ffmpeg
        stage_build
        ;;
    *)
        fail "Unknown stage: $stage. Use help, preflight, tests, digests, ffmpeg, build, or all."
        ;;
esac

log "Done: $stage"
