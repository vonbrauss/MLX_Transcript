#!/bin/zsh
set -euo pipefail

project_root="${0:A:h:h}"
cd "$project_root"
stage="$(mktemp -d /private/tmp/mlx-transcript-build.XXXXXX)"
export PYINSTALLER_CONFIG_DIR="$stage/pyinstaller-config"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "MLX Transcript must be built on an Apple Silicon Mac." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Create .venv and install requirements before building." >&2
  exit 1
fi

# The version lives in app/version.py only. Reading it here is what stops the
# archive name and the bundle's own CFBundleShortVersionString from drifting.
version="$(.venv/bin/python -c 'import re,pathlib;print(re.search(r"^VERSION = \"([^\"]+)\"", pathlib.Path("app/version.py").read_text(), re.M).group(1))')"
archive_name="$(.venv/bin/python -c 'import re,pathlib;print(re.search(r"^ARCHIVE_NAME = f\"([^\"]+)\"", pathlib.Path("app/version.py").read_text(), re.M).group(1).replace("{VERSION}", "'"$version"'"))')"
echo "Building MLX Transcript $version"

.venv/bin/python -m PyInstaller \
  --noconfirm \
  --clean \
  --distpath "$stage/dist" \
  --workpath "$stage/build" \
  packaging/MLX_Transcript.spec

app_path="$stage/dist/MLX Transcript.app"

# Cloud/project synchronization can attach resource-fork or Finder metadata
# that codesign rejects. Strip it, clear hidden flags, then apply an ad hoc
# signature for local and community testing. A Developer ID can replace this
# signature in the release workflow later.
xattr -cr "$app_path"
chflags -R nohidden "$app_path"
codesign --force --deep --sign - "$app_path"
codesign --verify --deep --strict "$app_path"

# Exercise the frozen engine before publishing the archive. This runs the real
# import chain, one mlx.core computation, the tokenizer, the numba word
# timestamp path, and the bundled media tools, so a bundle that would fail on
# the user's first clip never reaches a release archive.
mkdir -p "$project_root/build"
engine_report="$project_root/build/engine-check.txt"
if [[ "${MLX_TRANSCRIPT_SKIP_ENGINE_CHECK:-0}" != "1" ]]; then
  if ! "$app_path/Contents/MacOS/MLX Transcript" --check-engine --report "$engine_report"; then
    echo "" >&2
    echo "The packaged engine check failed. No archive was written." >&2
    echo "Full report: $engine_report" >&2
    exit 1
  fi
fi

mkdir -p "$project_root/dist"
release_zip="$project_root/dist/${archive_name}.zip"

# Replace the previous build's artifacts. A stale "MLX Transcript.app" left in
# dist/ from an earlier run is indistinguishable from a current one in Finder,
# and launching it is how a fixed bug appears to come back. Only this script's
# own outputs are removed, never anything else in the project.
rm -rf "$project_root/dist/MLX Transcript.app"
rm -rf "$project_root/dist/MLX Transcript"
rm -f "$release_zip"

# --sequesterRsrc keeps extended attributes and resource forks out of the
# bundle itself. Without it ditto writes AppleDouble "._" files next to the
# real ones inside the .app, and any unzip other than Finder's leaves them on
# disk, which breaks the code signature and produces "the application is
# damaged" on the user's Mac.
ditto -c -k --sequesterRsrc --keepParent "$app_path" "$release_zip"

# Verify what was actually archived, not what was signed a moment ago. The
# archive is extracted into a clean folder and checked there, because that is
# the copy the user will run.
verify_root="$(mktemp -d /private/tmp/mlx-transcript-verify.XXXXXX)"
ditto -x -k "$release_zip" "$verify_root"
extracted="$verify_root/MLX Transcript.app"

if [[ ! -d "$extracted" ]]; then
  echo "The archive did not contain MLX Transcript.app." >&2
  exit 1
fi

appledouble_count="$(find "$extracted" -name '._*' | wc -l | tr -d ' ')"
if [[ "$appledouble_count" != "0" ]]; then
  echo "" >&2
  echo "$appledouble_count AppleDouble file(s) were found inside the extracted app." >&2
  find "$extracted" -name '._*' | head -20 >&2
  exit 1
fi

if ! codesign --verify --deep --strict "$extracted"; then
  echo "The extracted application failed signature verification." >&2
  exit 1
fi

for tool in ffmpeg ffprobe; do
  if ! "$extracted/Contents/Frameworks/bin/$tool" -version >/dev/null 2>&1; then
    echo "The bundled $tool did not run from the extracted app." >&2
    exit 1
  fi
done

if [[ "${MLX_TRANSCRIPT_SKIP_ENGINE_CHECK:-0}" != "1" ]]; then
  extracted_report="$project_root/build/engine-check-extracted.txt"
  if ! "$extracted/Contents/MacOS/MLX Transcript" --check-engine --report "$extracted_report"; then
    echo "The extracted application failed its engine check." >&2
    echo "Full report: $extracted_report" >&2
    exit 1
  fi
fi

echo ""
echo "Built: $release_zip"
shasum -a 256 "$release_zip"
echo "Engine check report: $engine_report"
echo "Extracted copy verified at: $extracted"
echo ""
echo "This build is ad hoc signed and not notarized. See README.md for the"
echo "steps a downloader has to take in Privacy & Security."
