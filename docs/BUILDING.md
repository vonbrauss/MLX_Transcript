# Building the macOS application

MLX Transcript is an Apple Silicon application. The current development build
targets macOS 14 or later and produces an arm64-only app bundle.

The version number lives in `app/version.py` and nowhere else. The PyInstaller
recipe and the build script both read it from there, so the archive name and
the bundle's `CFBundleShortVersionString` cannot drift apart.

## Prepare the environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-build.txt
```

## FFmpeg: which build to bundle

MLX Transcript demuxes containers, reads duration, frame rate, and embedded
timecode with `ffprobe`, and decodes audio to 16 kHz mono PCM. It never
encodes anything. It therefore needs **no GPL components at all**: not x264,
not x265, not libvmaf, not libpostproc.

A stock Homebrew FFmpeg is built with `--enable-gpl --enable-version3` and
links x264 and x265. Shipping that binary makes the distributed application
subject to the GPL's source-distribution obligation for FFmpeg and for each
GPL library it links. The build refuses to do it:

```text
Refusing to bundle /opt/homebrew/bin/ffmpeg.
It was built with --enable-gpl, which makes the binary GPL and puts a
source-distribution obligation on every release.
```

### Why two raw PCM formats

MLX Transcript never encodes media, but it does need FFmpeg to hand it raw
audio samples on a pipe, and it needs them in two different shapes:

| Format | Encoder | Muxer | Who asks for it |
| --- | --- | --- | --- |
| 16-bit signed integer | `pcm_s16le` | `s16le` | `mlx_whisper.load_audio`, on every clip |
| 32-bit float | `pcm_f32le` | `f32le` | `transcription/diarization.py`, for speaker detection |

Both are *headerless*: the muxer writes nothing but samples, which is exactly
what the two callers read. `wav` and `null` are enabled as well, because the
build's own validation and any hand debugging want a container with a header
and a sink that discards output.

An FFmpeg missing either pair looks completely healthy. It reports its
version, it probes files, durations and timecode appear in the queue, and then
the first real clip fails with:

```text
Requested output format 's16le' is not known.
```

That is what shipped in the first standalone build, so the components are now
declared in one place, `packaging/ffmpeg_requirements.py`, and the build
refuses to package an FFmpeg that cannot produce both.

### Building an LGPL FFmpeg to bundle

The easy way is to let the script do it:

```bash
scripts/finish_release_prep.sh ffmpeg
```

That generates the configure flags from `packaging/ffmpeg_requirements.py`,
builds, and then validates the result. To do it by hand:

```bash
brew install nasm pkg-config

curl -LO https://ffmpeg.org/releases/ffmpeg-7.1.1.tar.xz
tar xf ffmpeg-7.1.1.tar.xz
cd ffmpeg-7.1.1

./configure \
  --prefix="$HOME/ffmpeg-lgpl" \
  --disable-gpl \
  --disable-nonfree \
  --disable-version3 \
  --disable-doc \
  --disable-debug \
  --disable-network \
  --disable-devices \
  --disable-ffplay \
  --disable-encoders \
  --disable-muxers \
  --disable-filters \
  --enable-encoder=pcm_s16le \
  --enable-encoder=pcm_f32le \
  --enable-muxer=s16le \
  --enable-muxer=f32le \
  --enable-muxer=wav \
  --enable-muxer=null \
  --enable-filter=aresample \
  --enable-filter=anull \
  --enable-filter=aformat \
  --enable-filter=atrim \
  --enable-filter=copy \
  --enable-static \
  --disable-shared \
  --enable-videotoolbox \
  --enable-audiotoolbox

make -j"$(sysctl -n hw.ncpu)"
make install
```

**Every component is its own flag, and that matters.** FFmpeg's `configure`
turns a comma-separated value into a shell `case` pattern and matches it
against its component list, whose entries are named `s16le_muxer` and
`f32le_muxer`. So this:

```bash
--enable-muxer=f32le,s16le,wav,null     # WRONG: matches nothing
```

matches nothing at all, prints a single `did not match anything` warning that
scrolls past in a long build log, and leaves `--disable-muxers` in force. Use
one flag per component instead:

```bash
--enable-muxer=f32le --enable-muxer=s16le --enable-muxer=wav --enable-muxer=null
```

Decoders, demuxers, parsers and protocols are all left enabled, which is what
keeps MXF, AVI, MTS, M2TS, WMV, MOV, MP4, MKV and every audio container
readable. Only encoding, muxing and filtering are trimmed, and only down to
what the two PCM pipes above need.

### Validating the result

```bash
python3 packaging/ffmpeg_requirements.py --verify \
  "$HOME/ffmpeg-lgpl/bin/ffmpeg" "$HOME/ffmpeg-lgpl/bin/ffprobe"
```

This checks three things and exits non-zero on any of them:

1. The reported configuration carries no `--enable-gpl`, `--enable-nonfree`,
   x264, x265, libvmaf or libpostproc, and does report `--disable-gpl` and
   `--disable-nonfree`.
2. Every required encoder, muxer, filter, demuxer and decoder is listed by
   `-encoders`, `-muxers`, `-filters`, `-demuxers` and `-decoders`. A missing
   one is named along with the flag that would add it.
3. A real media file decodes to **both** `s16le` and `f32le` PCM on stdout,
   using the same command shape the application uses at runtime. The file is
   a small WAV generated with the standard library, so no sample media has to
   be checked in.

The same validation runs again inside `packaging/MLX_Transcript.spec` before
PyInstaller analyses anything, so a build cannot be packaged around it.

### Then build with it

```bash
export MLX_TRANSCRIPT_FFMPEG="$HOME/ffmpeg-lgpl/bin/ffmpeg"
export MLX_TRANSCRIPT_FFPROBE="$HOME/ffmpeg-lgpl/bin/ffprobe"
scripts/build_macos.sh
```

Verify afterwards that representative media still reads. At minimum: one MXF,
one MOV with embedded timecode, one AVCHD `.mts`, one `.wav` and one `.mp3`.
Check that durations, frame rates and timecode appear in the queue and that a
transcript is produced.

### The environment variables

| Variable | Effect |
| --- | --- |
| `MLX_TRANSCRIPT_FFMPEG` | Path to the `ffmpeg` to bundle, instead of the one on `PATH`. |
| `MLX_TRANSCRIPT_FFPROBE` | Path to the `ffprobe` to bundle. |
| `MLX_TRANSCRIPT_BUNDLE_FFMPEG=0` | Do not bundle media tools at all; the app falls back to `PATH`. |
| `MLX_TRANSCRIPT_ALLOW_GPL_FFMPEG=1` | Override the refusal. For local builds that will never be published. |
| `MLX_TRANSCRIPT_SKIP_ENGINE_CHECK=1` | Skip the packaged engine self-test. For iterating on packaging only. |

Whatever gets bundled, its full configure line is written to
`build/ffmpeg-configuration.txt`. Copy that into the release notes.

### If you do ship a GPL FFmpeg anyway

You must, for that exact binary and each GPL library it links:

1. State the FFmpeg version and the complete configure line.
2. Offer the corresponding source, either alongside the release archive or
   through a written offer valid for three years.
3. Do the same for x264, x265, and anything else GPL that was linked in.

`THIRD_PARTY_NOTICES.md` has to be updated to match. This is why the default
is to refuse.

## Build

```bash
scripts/build_macos.sh
```

The build runs in a temporary folder so Finder and File Provider metadata do
not invalidate its signature. It creates:

```text
dist/MLX-Transcript-<version>-arm64.zip
```

### What the build verifies before it reports success

The signature is checked on the bundle, and then everything is checked again
on a copy extracted from the archive, because the extracted copy is what a
user actually runs:

- the archive contains `MLX Transcript.app`
- no AppleDouble (`._*`) files are inside the extracted bundle
- `codesign --verify --deep --strict` passes on the extracted bundle
- the bundled `ffmpeg` and `ffprobe` run from the extracted bundle
- the packaged engine check passes on the extracted bundle: a real
  `mlx.core` computation, the Whisper tokenizer, the mel filterbank, the numba
  word-timestamp path, and both media tools

The archive is written with `ditto -c -k --sequesterRsrc --keepParent`. The
`--sequesterRsrc` flag is what keeps extended attributes out of the bundle.
Without it, `ditto` stores AppleDouble `._*` entries inside the `.app`, and any
extraction other than Finder's leaves them on disk, which invalidates the code
signature and produces "the application is damaged" for the user.

## Signing and Gatekeeper

The app has an ad hoc signature for testing. It is not notarized, so a
downloaded copy has to be approved in System Settings → Privacy & Security.
README.md has the step-by-step instructions to put in release notes.

`codesign --deep` is deprecated by Apple and signs nested content
inconsistently. It is adequate for an ad hoc preview signature; a Developer ID
release should sign inner binaries individually, enable the Hardened Runtime,
and then notarize and staple:

```bash
codesign --force --options runtime --timestamp \
  --sign "Developer ID Application: ..." <each nested binary>
codesign --force --options runtime --timestamp \
  --sign "Developer ID Application: ..." "MLX Transcript.app"
xcrun notarytool submit dist/MLX-Transcript-<version>-arm64.zip \
  --keychain-profile "notary" --wait
xcrun stapler staple "MLX Transcript.app"
```

Signing credentials belong in the developer keychain or protected CI secrets
and must never be committed.

## The application icon

`packaging/MLX Transcript.icns` is generated artwork checked into the
repository. The recipe passes it to both `EXE` and `BUNDLE`, so the bundle
carries it rather than PyInstaller's placeholder. Replacing it is a matter of
dropping a new `.icns` at that path.

## Recording the model digests

The application refuses to cache a diarization model whose SHA-256 does not
match the recorded value. Those values are captured on a machine that can
reach the model host:

```bash
python scripts/record_model_digests.py --write
```

This downloads each asset to a temporary folder, prints its size and digest,
and patches `transcription/model_cache.py`. An asset whose `sha256` is empty
is not pinned: the transfer length is still checked, but nothing stronger.
