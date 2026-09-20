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
release_zip="$project_root/dist/MLX-Transcript-0.1.0-arm64.zip"

# Replace the previous build's artifacts. A stale "MLX Transcript.app" left in
# dist/ from an earlier run is indistinguishable from a current one in Finder,
# and launching it is how a fixed bug appears to come back. Only this script's
# own outputs are removed, never anything else in the project.
rm -rf "$project_root/dist/MLX Transcript.app"
rm -rf "$project_root/dist/MLX Transcript"
rm -f "$release_zip"

ditto -c -k --keepParent "$app_path" "$release_zip"

echo "Built: $release_zip"
shasum -a 256 "$release_zip"
echo "Engine check report: $engine_report"
