# Building the macOS application

MLX Transcript is an Apple Silicon application. The current development build
targets macOS 14 or later and produces an arm64-only app bundle.

## Prepare the environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-build.txt
```

FFmpeg and ffprobe must be available while building. The development recipe
locates them on `PATH` and includes them in the application.

## Build

```bash
scripts/build_macos.sh
```

The build runs in a temporary folder so Finder and File Provider metadata do
not invalidate its signature. It creates:

```text
dist/MLX-Transcript-0.1.0-arm64.zip
```

The app has an ad hoc signature for testing. It is not notarized. A downloaded
copy therefore requires the user to approve it in macOS Privacy & Security.

## FFmpeg release requirement

The Homebrew FFmpeg used by the development environment enables GPL
components. Do not publish that binary without completing the corresponding
GPL notice and source-availability obligations. A public release may instead
use a separately produced LGPL-compatible FFmpeg build.

## Future signed releases

The same PyInstaller recipe can be extended with a Developer ID Application
identity, Hardened Runtime options, `notarytool`, and ticket stapling. Signing
credentials belong in the developer keychain or protected CI secrets and must
never be committed.
