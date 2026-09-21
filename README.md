# MLX Transcript

A local-first batch transcription app for editorial media. It runs on Apple
silicon Macs, uses MLX Whisper in process, and never uploads media anywhere.

Choose one media file, several files, or one or more folders, then pick where the output goes and choose the
names, folder layout, and transcript formats you need. The defaults preserve
the source hierarchy and create ScriptSync and timecoded text using the media
file's original name:

```text
<output parent>/
└── Transcription/
    ├── ScriptSync/     plain ASCII text for Avid ScriptSync
    ├── Timecoded/      source-timecode stamped transcripts
    └── Subtitles/      optional SRT and WebVTT captions
```

Files can instead be collected into flat output folders; same-named clips are
numbered `(2)`, `(3)`, and so on. Names can use the original media stem, append
the transcript type, or add a custom suffix. Any combination of ScriptSync,
Timecoded, SRT, and WebVTT can be produced in one pass.

MLX Transcript accepts any local media file FFmpeg can read that contains an
audio stream, whatever its filename. Dropped folders are scanned recursively and each
candidate file is probed with `ffprobe` in the background, so a clip with an
unusual extension, or none at all, is queued as long as it holds audio. A
short list of formats that cannot carry audio, such as PDFs, images and
subtitle files, is skipped without probing as a fast path. Anything skipped is
counted in the queue summary with a reason, so nothing is dropped silently.

Drag media files or folders onto the Media workspace or Queue to append them.
A translucent sheet marks the workspace while a Finder drag is over it.
Duplicate files are ignored. When a queue combines multiple source roots while
using the source-tree layout, each root receives its own output folder so
unrelated folders cannot collide. The queue can also remove selected entries,
clear its contents, and reveal a selected source in Finder.

The Media workspace is empty until media is queued, and then shows the queue
inline in the same area. The Queue workspace is the same queue at full-page
size: both read one list, so order, statuses, selection, and the summary are
always in step. Dragging a row up or down changes the order files are
transcribed in, and an insertion line shows where the row will land.
Reordering is unavailable while a batch is running.

The window uses a single-row workspace menu for Media, Transcription,
Output, Speaker Detection, Advanced Settings, and Queue. Only the selected
workspace is shown, and starting a batch opens Queue automatically while the
progress and action bar stays visible.
The app saves settled changes automatically and restores the last working
setup on the next launch. Named presets store reusable transcription, output,
and speaker settings while keeping each job's source and destination folders.
Each launch begins on Media. The Help workspace provides a quick guide to
selecting folders, choosing output formats, speaker detection, and running a
batch.
The packaged entry point routes background helper processes before Qt starts
and holds a per-user instance lock, preventing transcription helpers or a
second launch from opening another application window.
If the application stops unexpectedly, the next launch verifies the recorded
process and removes its abandoned lock immediately.

## Requirements

Running the packaged application:

- Apple silicon Mac running macOS 14 or later

Nothing else. FFmpeg, MLX, and the speaker detection backend all travel inside
the application bundle.

Running from a checkout:

- Apple silicon Mac running macOS 14 or later
- Python 3.10 or newer (the bundled virtual environment uses 3.12)
- FFmpeg, for `ffmpeg` and `ffprobe`: `brew install ffmpeg`

## Setup

```bash
cd "MLX_Transcript"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Standalone macOS build

An Apple Silicon development app can be built without installing Python on the
destination Mac:

```bash
python -m pip install -r requirements-build.txt
scripts/build_macos.sh
```

The versioned ZIP is written to `dist/`. The build verifies the archive it
just wrote: it extracts it to a clean folder, confirms no AppleDouble files
landed inside the bundle, re-checks the code signature on the extracted copy,
runs the bundled `ffmpeg` and `ffprobe`, and runs the packaged engine check
against a real MLX computation.

Preview builds are ad hoc signed and are not notarized. See "Opening a preview build" below for what a downloader has to do.

## Opening a preview build

A preview build is signed ad hoc rather than with a Developer ID, and it is
not notarized. macOS therefore refuses to open it on the first attempt, with
a message about the application being damaged or from an unidentified
developer. Nothing is wrong with the download; macOS is telling you it cannot
verify who produced it.

1. Unzip the archive in Finder by double-clicking it. Do not unzip it with a
   third-party tool.
2. Drag **MLX Transcript.app** to your Applications folder.
3. Double-click it once. macOS will refuse and offer only **Done**.
4. Open **System Settings → Privacy & Security**, scroll to the Security
   section, and click **Open Anyway** next to the message about MLX
   Transcript.
5. Confirm with **Open**. macOS remembers the decision, so this is a one-time
   step.

On macOS 15 and later, right-clicking the application and choosing Open no
longer bypasses this; the Privacy & Security step is the supported route.

If you would rather not do any of that, run it from a checkout with
`python main.py`, which Gatekeeper does not gate.

From PyCharm, set the project interpreter to `.venv` and add a run
configuration that points at `main.py` with the project root as the working
directory.

## Tests

```bash
python -m pytest
```

The tests cover timecode conversion, selectable output names and layouts,
ScriptSync cleanup, SRT and WebVTT formatting, media discovery, ffprobe payload
parsing, engine option construction, batch processing, conflict policy,
cancellation, atomic writes, the sleep assertion, batch resilience, worker
lifecycle, queue bookkeeping and its scaling, window shutdown during a batch,
drag and drop onto both drop targets, and model download integrity. None of
them need FFmpeg, the model weights, MLX, or a display: the transcription
engine is always a stand-in.

## How it is put together

| Module | Responsibility |
| --- | --- |
| `main.py` | Qt entry point |
| `app/main_window.py` | Window, pickers, queue table, progress |
| `app/models.py` | Queue items, status enum, speaker labels |
| `app/speaker_review.py` | The rename, merge, and reassign dialog |
| `app/settings.py` | Choices persisted with QSettings |
| `app/power.py` | Keeps the Mac awake while a batch runs |
| `app/workers.py` | Scan worker and transcription worker, each on a thread |
| `transcription/discovery.py` | Recursive media discovery, output-tree exclusion |
| `transcription/media_probe.py` | ffprobe duration, frame rate, source timecode |
| `transcription/timecode.py` | Drop-frame and non-drop-frame conversion |
| `transcription/outputs.py` | Path layout, ScriptSync cleanup, atomic writers |
| `transcription/pipeline.py` | Batch policy: conflicts, review, cancellation, summary |
| `transcription/engine.py` | MLX Whisper options and in-process transcription |
| `transcription/diarization.py` | sherpa-onnx speaker detection, loaded lazily |
| `transcription/alignment.py` | Word-to-speaker assignment, splitting, merging |
| `transcription/speakers.py` | Speaker transcript model: names, merges, exports |
| `transcription/model_cache.py` | Where diarization models live and how they arrive |
| `reference/avid_transcribe_folders.py` | The original script, kept as the behavior spec |

`reference/avid_transcribe_folders.py` still runs on its own. The application
does not shell out to it; the reusable parts were ported into the modules
above.

## Privacy

**Your media never leaves this Mac. Models download once from Hugging Face and
are reused locally.**

Transcription and speaker detection both run in this process on this machine.
There is no cloud service, no analytics, no telemetry, and no API key. The
only network traffic the application ever makes is fetching model files:

- Whisper weights, on first use of a model, after the application tells you
  how large the download is and where it will be cached.
- The two speaker detection models, about 31 MB in total, only if you turn
  speaker detection on and confirm the download.

Both are verified before they are stored and reused offline from then on.

## Transcription

MLX Whisper runs in this process on a background thread, so the model stays
resident for a whole batch and the interface stays responsive. The default
model is `mlx-community/whisper-large-v3-mlx`, with Turbo available as the
faster option. Weights download automatically on first use.

Decoding settings are fixed at the values that keep Whisper from inventing or
repeating speech: `temperature=0.0`, `condition_on_previous_text=False`,
`compression_ratio_threshold=2.4`, and `word_timestamps=True`. The no-speech,
log-probability, and hallucination-silence thresholds are exposed under
Advanced Settings. Choosing Auto Detect omits the language argument entirely
rather than passing an empty one.

Transcripts are written atomically: a temporary file in the destination folder
is replaced into place only after the write succeeds. Nothing else is written,
so no JSON, SRT, VTT, TSV, or temporary audio lands in the output tree.

Cancel stops the batch after the clip in flight finishes safely, leaving every
completed transcript intact. A clip that fails is reported in its queue row and
the rest of the queue continues. While a batch runs, the Mac is kept awake with
a `caffeinate` assertion that is released when the batch ends, however it ends.

## Speaker detection

Optional, off by default, and local. With it on, each clip is transcribed,
then diarized with sherpa-onnx running pyannote segmentation 3.0 and WeSpeaker
embeddings through ONNX Runtime. No PyTorch, no CUDA, no account, and no
access token. Audio is decoded to 16 kHz mono through an ffmpeg pipe in memory,
so no extracted audio file is written anywhere.

Whisper word timestamps decide who said what. Each word goes to the speaker
interval it overlaps most, and a Whisper segment is split wherever its words
change speaker. The Balanced, Fewer speakers, More separation, and Fast
conversation presets tune clustering, minimum turn and pause durations, word
assignment tolerance, and nearby-speech merging. Every value is also available
under Advanced speaker settings. When the number of people is known, Exact
number bypasses automatic threshold clustering. Overlapped speech keeps a
primary speaker, chosen by longest overlap, and the other speakers are retained
in the data model for a richer export later. Recognized wording is never
changed.

Before anything is written, the review dialog shows each detected speaker with
their speaking time, segment count, and a few samples, plus a timecoded
preview. Names can be edited, reset, merged, and individual lines reassigned.
Names apply to that clip only: Speaker 1 in one recording is not Speaker 1 in
the next, and there is no reliable way to match speakers across files by label
alone, so no cross-file name reuse is offered.

If diarization fails, the transcript is still exported without labels and the
reason is recorded in the batch summary as a warning, never as a failure.

The models are about 31 MB in total, downloaded once from their public Hugging
Face repositories and kept in
`~/Library/Application Support/MLX Transcript/models/`, never in the project or
the output tree. After that they work offline. Install the backend with
`pip install sherpa-onnx`; without it the section reports Unavailable and
ordinary transcription is unaffected.

## License

MLX Transcript is free software licensed under the **GNU General Public
License, version 3**. See `LICENSE` for the full text.

The application bundle carries third-party components under their own
licences, including LGPL-licensed Qt and FFmpeg. `THIRD_PARTY_NOTICES.md`
lists every bundled component, its licence, and where to obtain its source.

## Status

Milestones 1 to 3 are complete: the shell, discovery, probing, the queue, the
transcription pipeline, conflict handling, cancellation, speaker detection,
speaker review, frozen-process startup, and navigation. Run `python -m pytest`
for the current test count rather than trusting a number written here.
