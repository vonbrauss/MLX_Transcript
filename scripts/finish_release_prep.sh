#!/bin/zsh
# Finish the release preparation on this Mac.
#
# Everything in this script needs macOS, Apple silicon, or network access to
# Hugging Face, which is why it is not already done. Run the stages in order.
# Each one writes a log under build/ and stops on the first failure.
#
#   scripts/finish_release_prep.sh tests     # ~10 seconds
#   scripts/finish_release_prep.sh digests   # ~1 minute, needs the network
#   scripts/finish_release_prep.sh ffmpeg    # 20-40 minutes, one time only
#   scripts/finish_release_prep.sh build     # ~5 minutes
#   scripts/finish_release_prep.sh all       # every stage in that order
#
# Nothing here pushes, tags, or publishes anything.

set -euo pipefail

project_root="${0:A:h:h}"
cd "$project_root"
mkdir -p build

# Where the LGPL FFmpeg is built and installed. Outside the project, because
# it is a toolchain artifact rather than part of the application source.
ffmpeg_prefix="${MLX_TRANSCRIPT_FFMPEG_PREFIX:-$HOME/ffmpeg-lgpl}"
ffmpeg_version="${MLX_TRANSCRIPT_FFMPEG_VERSION:-7.1.1}"

log() { print -r -- "==> $*"; }
fail() { print -r -- "!!! $*" >&2; exit 1; }

require_venv() {
  [[ -x .venv/bin/python ]] || fail "No .venv found. See docs/BUILDING.md."
}

stage_tests() {
  require_venv
  log "Running the full test suite in .venv"
  .venv/bin/python -m pytest -q 2>&1 | tee build/pytest-macos.log
  log "Test log: build/pytest-macos.log"
}

stage_digests() {
  require_venv
  log "Recording the diarization model digests"
  log "This downloads about 31 MB to a temporary folder, then deletes it."
  .venv/bin/python scripts/record_model_digests.py --write \
    2>&1 | tee build/model-digests.log
  log "transcription/model_cache.py now pins both assets."
  log "Re-run the tests after this: scripts/finish_release_prep.sh tests"
}

stage_ffmpeg() {
  if [[ -x "$ffmpeg_prefix/bin/ffmpeg" && -x "$ffmpeg_prefix/bin/ffprobe" ]]; then
    log "An LGPL FFmpeg is already installed at $ffmpeg_prefix"
    log "Validating it rather than trusting it"
    stage_validate_ffmpeg
    return 0
  fi

  command -v nasm >/dev/null || fail "nasm is required: brew install nasm pkg-config"

  # Every component is enabled as its own configure flag, generated from
  # packaging/ffmpeg_requirements.py so the recipe, the packaging gate and the
  # tests cannot drift apart.
  #
  # This is what the first standalone build got wrong. FFmpeg's configure
  # turns a comma-separated value into a shell case pattern and matches it
  # against component entries named "s16le_muxer" and "f32le_muxer", so
  # --enable-muxer=f32le,s16le,wav,null matched nothing, the warning scrolled
  # past in the build log, --disable-muxers won, and the app failed on its
  # first real clip with "Requested output format 's16le' is not known."
  local -a configure_args
  configure_args=("${(@f)$(python3 "$project_root/packaging/ffmpeg_requirements.py" \
    --configure-args --prefix "$ffmpeg_prefix")}")
  print -r -- "${(F)configure_args}" > build/ffmpeg-configure-args.txt
  log "${#configure_args} configure flags recorded in build/ffmpeg-configure-args.txt"

  local work
  work="$(mktemp -d /private/tmp/mlx-ffmpeg-build.XXXXXX)"
  log "Building FFmpeg $ffmpeg_version in $work"
  log "This takes 20 to 40 minutes. Output goes to build/ffmpeg-build.log"

  {
    cd "$work"
    curl -LO "https://ffmpeg.org/releases/ffmpeg-${ffmpeg_version}.tar.xz"
    tar xf "ffmpeg-${ffmpeg_version}.tar.xz"
    cd "ffmpeg-${ffmpeg_version}"

    ./configure "${configure_args[@]}"

    make -j"$(sysctl -n hw.ncpu)"
    make install
  } 2>&1 | tee build/ffmpeg-build.log

  [[ -x "$ffmpeg_prefix/bin/ffmpeg" ]] || fail "The FFmpeg build did not install."
  stage_validate_ffmpeg
  log "Installed at $ffmpeg_prefix"
}

stage_validate_ffmpeg() {
  # Licence, every required component, and two real decodes to raw PCM. A
  # build that passes this cannot fail the way the first one did.
  log "Validating: licence, components, and two real decodes to raw PCM"
  if ! python3 "$project_root/packaging/ffmpeg_requirements.py" --verify \
       "$ffmpeg_prefix/bin/ffmpeg" "$ffmpeg_prefix/bin/ffprobe" \
       2>&1 | tee build/ffmpeg-validation.txt; then
    fail "This FFmpeg cannot do what MLX Transcript needs. See build/ffmpeg-validation.txt"
  fi
  "$ffmpeg_prefix/bin/ffmpeg" -version | grep '^ *configuration:' \
    > build/ffmpeg-lgpl-configuration.txt || true
}

stage_build() {
  require_venv
  [[ -x "$ffmpeg_prefix/bin/ffmpeg" ]] \
    || fail "No LGPL FFmpeg at $ffmpeg_prefix. Run the ffmpeg stage first."

  export MLX_TRANSCRIPT_FFMPEG="$ffmpeg_prefix/bin/ffmpeg"
  export MLX_TRANSCRIPT_FFPROBE="$ffmpeg_prefix/bin/ffprobe"

  log "Re-validating the media tools before packaging"
  python3 "$project_root/packaging/ffmpeg_requirements.py" --verify \
    "$MLX_TRANSCRIPT_FFMPEG" "$MLX_TRANSCRIPT_FFPROBE" \
    || fail "The media tools no longer pass validation."

  log "Building the application with the LGPL media tools"
  scripts/build_macos.sh 2>&1 | tee build/app-build.log

  local version archive
  version="$(.venv/bin/python -c 'import re,pathlib;print(re.search(r"^VERSION = \"([^\"]+)\"", pathlib.Path("app/version.py").read_text(), re.M).group(1))')"
  archive="dist/MLX-Transcript-${version}-arm64.zip"

  log "Archive: $project_root/$archive"
  shasum -a 256 "$archive" | tee build/archive-sha256.txt

  print -r -- ""
  log "What is verified, and what is not"
  print -r -- "  verified: no AppleDouble files inside the extracted app"
  print -r -- "  verified: codesign --verify --deep --strict on the extracted app"
  print -r -- "  verified: bundled ffmpeg and ffprobe run from the bundle"
  print -r -- "  verified: real mlx.core computation, tokenizer, numba path"
  print -r -- "  verified: sherpa-onnx reported by the engine check"
  print -r -- "  NOT verified: Gatekeeper acceptance of a quarantined download."
  print -r -- "               To test that, upload the archive somewhere, download"
  print -r -- "               it in Safari, unzip in Finder, and follow the"
  print -r -- "               Privacy & Security steps in README.md."
}

stage="${1:-all}"
case "$stage" in
  tests)   stage_tests ;;
  digests) stage_digests ;;
  ffmpeg)  stage_ffmpeg ;;
  build)   stage_build ;;
  all)
    # digests is deliberately not here: it needs a large download and is
    # opt-in. Run it on its own when you want the assets pinned.
    stage_tests
    stage_ffmpeg
    stage_build
    ;;
  *) fail "Unknown stage: $stage. Use tests, digests, ffmpeg, build, or all." ;;
esac

log "Done: $stage"
