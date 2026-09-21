# Third-party notices

MLX Transcript is distributed under the GNU General Public License version 3.
See `LICENSE` for the full text.

The macOS application bundle carries the components listed below. Each entry
names what is bundled, under which licence, and where its source can be
obtained. Nothing here is a substitute for the licences themselves; the full
texts travel with each project at the source locations given.

The exact FFmpeg configuration of a given build is recorded at build time in
`build/ffmpeg-configuration.txt` and should be copied into the release notes
for that build.

---

## Media tools

### FFmpeg (`ffmpeg`, `ffprobe`, and the `libav*` / `libsw*` libraries)

- **Licence:** LGPL-2.1-or-later, for a build configured with `--disable-gpl`
  and `--disable-nonfree`.
- **Source:** https://ffmpeg.org/download.html and
  https://git.ffmpeg.org/ffmpeg.git
- **Used for:** demuxing containers, reading duration, frame rate, and
  embedded timecode with `ffprobe`, and decoding audio to 16 kHz mono PCM.
  MLX Transcript never encodes with FFmpeg.

> **Build requirement.** MLX Transcript's PyInstaller recipe refuses to bundle
> an FFmpeg built with `--enable-gpl` or `--enable-nonfree` unless
> `MLX_TRANSCRIPT_ALLOW_GPL_FFMPEG=1` is set. A stock Homebrew FFmpeg is a GPL
> build and must not be shipped without also satisfying the GPL's
> source-distribution obligations for that exact binary and for the GPL
> libraries it links, such as x264 and x265. See `docs/BUILDING.md`.

Because MLX Transcript is itself GPL-3.0, a user who receives this application
may replace the bundled LGPL FFmpeg libraries with their own build. They are
ordinary dynamic libraries inside
`MLX Transcript.app/Contents/Frameworks/`.

---

## Application framework

### Qt 6 and PySide6 / Shiboken6

- **Licence:** LGPL-3.0-only (the open-source Qt for Python wheels; also
  available under GPL-2.0-only or GPL-3.0-only, and commercially from
  The Qt Company).
- **Source:** https://code.qt.io/cgit/pyside/pyside-setup.git/ and
  https://download.qt.io/official_releases/QtForPython/
- **Bundled:** `QtCore`, `QtGui`, `QtWidgets`, `QtNetwork`, `QtSvg`, `QtDBus`,
  `QtOpenGL`, `QtPdf`, `QtQml*`, `QtQuick`, `QtVirtualKeyboard*`,
  `libpyside6`, `libshiboken6`, and the Qt plugins under `PySide6/Qt/plugins`.

Under the LGPL, a recipient is entitled to replace the bundled Qt libraries
with their own versions. They are unmodified dynamic libraries and frameworks
inside `MLX Transcript.app/Contents/Frameworks/`, so a replacement build of
the same major version can be substituted in place.

---

## Transcription engine

| Component | Licence | Source |
| --- | --- | --- |
| MLX (`mlx`, `mlx-metal`, `libmlx.dylib`, `libjaccl.dylib`) | MIT | https://github.com/ml-explore/mlx |
| MLX Whisper (`mlx_whisper`) | MIT | https://github.com/ml-explore/mlx-examples |
| tiktoken | MIT | https://github.com/openai/tiktoken |
| numba | BSD-2-Clause | https://github.com/numba/numba |
| llvmlite | BSD-2-Clause, with Apache-2.0 WITH LLVM-exception components | https://github.com/numba/llvmlite |
| NumPy | BSD-3-Clause, with 0BSD, MIT, Zlib and CC0-1.0 components | https://github.com/numpy/numpy |
| SciPy | BSD-3-Clause | https://github.com/scipy/scipy |
| regex | Apache-2.0 and CNRI-Python | https://github.com/mrabarnett/mrab-regex |
| tqdm | MPL-2.0 and MIT | https://github.com/tqdm/tqdm |
| huggingface_hub, hf-xet | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| certifi | MPL-2.0 | https://github.com/certifi/python-certifi |
| charset-normalizer | MIT | https://github.com/jawah/charset_normalizer |
| click | BSD-3-Clause | https://github.com/pallets/click |
| MarkupSafe | BSD-3-Clause | https://github.com/pallets/markupsafe |
| setuptools | MIT | https://github.com/pypa/setuptools |

---

## Speaker detection

| Component | Licence | Source |
| --- | --- | --- |
| sherpa-onnx (`libsherpa-onnx-*.dylib`, `sherpa_onnx`) | Apache-2.0 | https://github.com/k2-fsa/sherpa-onnx |
| ONNX Runtime (`libonnxruntime.dylib`) | MIT | https://github.com/microsoft/onnxruntime |

### Model assets

These two files are **not** in the application bundle. They are downloaded
once, on the user's explicit confirmation, into
`~/Library/Application Support/MLX Transcript/models/` and reused offline.

| Asset | Licence | Source |
| --- | --- | --- |
| `sherpa-onnx-pyannote-segmentation-3-0.onnx` (pyannote segmentation 3.0, ONNX export by csukuangfj) | MIT | https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0 |
| `wespeaker_en_voxceleb_resnet34_LM.onnx` (WeSpeaker VoxCeleb ResNet34-LM) | Apache-2.0 | https://huggingface.co/csukuangfj/speaker-embedding-models |

No account or access token is required for either, and nothing is uploaded.

---

## Transcription models

Whisper model weights are **not** in the application bundle either. They are
downloaded once from Hugging Face on first use, after the application
discloses the download, and are cached under
`~/.cache/huggingface/hub/`.

| Model | Licence | Source |
| --- | --- | --- |
| `mlx-community/whisper-large-v3-mlx` | MIT | https://huggingface.co/mlx-community/whisper-large-v3-mlx |
| `mlx-community/whisper-large-v3-turbo` | MIT | https://huggingface.co/mlx-community/whisper-large-v3-turbo |

Both derive from OpenAI Whisper, which is MIT licensed:
https://github.com/openai/whisper

---

## Runtime

### CPython

- **Licence:** PSF License Agreement
- **Source:** https://www.python.org/downloads/source/
- **Bundled:** `Python.framework`, `python3.12`, `base_library.zip`, and the
  standard library modules the application imports.

### Supporting libraries linked by CPython and FFmpeg

| Component | Licence | Source |
| --- | --- | --- |
| OpenSSL (`libssl`, `libcrypto`) | Apache-2.0 | https://github.com/openssl/openssl |
| SQLite (`libsqlite3`) | Public domain | https://sqlite.org/src |
| XZ Utils (`liblzma`) | 0BSD | https://github.com/tukaani-project/xz |
| libmpdec | BSD-2-Clause | https://www.bytereef.org/mpdecimal/ |

---

## Build tooling (not distributed)

PyInstaller (GPL-2.0-or-later with a bundling exception), its hooks
contribution package (Apache-2.0), macholib and altgraph (MIT), and pytest
(MIT) are used to build and test MLX Transcript. They are not part of the
application bundle.

---

## Reporting a problem with this file

If a component is bundled that is not listed here, or a licence is recorded
incorrectly, please open an issue at
https://github.com/vonbrauss/MLX_Transcript/issues so it can be corrected.
