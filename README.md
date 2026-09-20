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

Drag media files or folders onto the Folders drop area or Queue to append them.
Duplicate files are ignored. When a queue combines multiple source roots while
using the source-tree layout, each root receives its own output folder so
unrelated folders cannot collide. The queue can also remove selected entries,
clear its contents, and reveal a selected source in Finder.

The window uses a single-row workspace menu for Folders, Transcription,
Output, Speaker Detection, Advanced Settings, and Queue. Only the selected
workspace is shown, and starting a batch opens Queue automatically while the
progress and action bar stays visible.
The app saves settled changes automatically and restores the last working
setup on the next launch. Named presets store reusable transcription, output,
and speaker settings while keeping each job's source and destination folders.
Each launch begins on Folders. The Help workspace provides a quick guide to
selecting folders, choosing output formats, speaker detection, and running a
batch.
The packaged entry point routes background helper processes before Qt starts
and holds a per-user instance lock, preventing transcription helpers or a
second launch from opening another application window.
If the application stops unexpectedly, the next launch verifies the recorded
process and removes its abandoned lock immediately.

## Requirements

- Apple silicon Mac running macOS 14 or later
- Python 3.10 or newer (the bundled virtual environment uses 3.12)
- FFmpeg, for `ffprobe`: `brew install ffmpeg`

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

The versioned ZIP is written to `dist/`. Development builds are ad hoc signed
and are not notarized. See `docs/BUILDING.md` for the Gatekeeper, signing, and
FFmpeg distribution details.

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
cancellation, atomic writes, and the sleep assertion. None of them need FFmpeg,
the model weights, MLX, or a display: the
transcription engine is always a stand-in.

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

Transcription runs on this Mac. There is no cloud service, no analytics, no
telemetry, and no API key. Model weights are downloaded once from Hugging Face
the first time a model is selected, and media never leaves the machine.

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

## Status

Milestones 1 to 3 are complete: the shell, discovery, probing, the queue, the
transcription pipeline, conflict handling, cancellation, speaker detection,
speaker review, frozen-process startup, navigation, and 449 tests.
