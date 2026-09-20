# MLX Transcript — First-Pass Build Brief

You are the implementation engineer for a macOS desktop application named **MLX Transcript**. Build the first working version in this PyCharm project.

## Product goal

Create a local-first batch transcription application for editorial media. It runs on Apple silicon Macs and uses MLX Whisper directly. It must never upload media for transcription.

The app lets an editor choose a source folder of media, choose where output should go, process the media recursively, and create two transcript trees:

```text
<chosen output parent>/
└── Transcription/
    ├── ScriptSync/
    └── Timecoded/
```

Each tree preserves the source folder hierarchy. For example:

```text
Source/
└── Media day sound/
    └── Atlanta Dream.mov

Output/
└── Transcription/
    ├── ScriptSync/
    │   └── Media day sound/
    │       └── Media day sound — Atlanta Dream ScriptSync.txt
    └── Timecoded/
        └── Media day sound/
            └── Media day sound — Atlanta Dream timecoded.txt
```

## Existing behavior specification

Before writing the transcription engine, inspect and preserve the behavior in `reference/avid_transcribe_folders.py`. Treat it as the working specification for:

- Media discovery and excluding the output tree
- Embedded MOV timecode and frame-rate extraction through `ffprobe`
- Drop-frame timecode handling
- Avid ScriptSync text cleanup: ASCII, no timestamps, no hyphens, hard wraps at about 64 characters
- Timecoded transcript format
- Filename construction
- Hallucination controls

Refactor the reusable parts into application modules. Do not invoke the existing script as a subprocess for each file.

## Required first-pass application behavior

Build a polished, functional PySide6 desktop app with this layout:

1. Source folder field and **Choose…** button.
2. Output parent field and **Choose…** button. The app creates the `Transcription` folder beneath this location.
3. Model picker with:
   - `Whisper Large v3` as the default
   - `Whisper Large v3 Turbo` as the speed option
4. Language picker, defaulting to English.
5. Existing-files picker:
   - Ask every time
   - Skip existing
   - Overwrite all
6. Collapsed Advanced Settings section for:
   - Hallucination silence threshold
   - No-speech threshold
   - Log-probability threshold
7. A queue table showing file, relative folder, duration, and status.
8. Overall progress bar and current-file label.
9. Buttons for Start Transcription, Cancel, and Reveal Output.
10. An output preview showing `Transcription / ScriptSync` and `Transcription / Timecoded`.
11. A privacy label: `Processing locally on this Mac`.

## Transcription requirements

- Use `mlx_whisper.transcribe(...)` inside the application process so the model remains loaded during a batch.
- Default model: `mlx-community/whisper-large-v3-mlx`.
- Turbo alternative: `mlx-community/whisper-large-v3-turbo`.
- Use English transcription by default.
- Use `condition_on_previous_text=False`.
- Enable `word_timestamps=True`; the hallucination silence threshold only works when word timestamps are enabled.
- Preserve user-tunable silence and confidence controls.
- Run batch work on a background worker. The UI must remain responsive.
- Support cancellation between files and leave completed outputs intact.
- Report individual failures in the queue and continue with remaining files.

## Output conflict behavior

When a transcript already exists and the selected policy is **Ask every time**, present a dialog with:

- Overwrite this clip
- Overwrite all remaining conflicts
- Skip this clip
- Cancel batch

If only one of the two outputs exists, keep the existing file when Skip is selected and create the missing one.

## Speaker detection: design now, build later

Include a disabled-but-visible `Detect Speakers` option and a short tooltip stating that it is planned for the next milestone. Design the transcript and queue data structures so a future diarization stage can add `Speaker 1`, `Speaker 2`, etc. and allow manual renaming. Do not add a pyannote or other diarization dependency in this first pass.

## Technical constraints

- Target Apple silicon Macs running macOS 14 or later.
- Use Python 3.10+ and PySide6.
- Use `ffprobe` from FFmpeg for duration, source timecode, and frame-rate inspection.
- No cloud transcription, analytics, telemetry, or API keys.
- Use type hints, clear module boundaries, and concise docstrings.
- Do not hard-code personal folders or media paths.
- Avoid destructive operations. Existing outputs are overwritten only after the chosen policy permits it.

## Suggested project structure

```text
mlx-transcript/
├── main.py
├── app/
│   ├── main_window.py
│   ├── models.py
│   ├── settings.py
│   └── workers.py
├── transcription/
│   ├── engine.py
│   ├── discovery.py
│   ├── outputs.py
│   ├── timecode.py
│   └── media_probe.py
├── reference/
│   └── avid_transcribe_folders.py
├── tests/
│   ├── test_outputs.py
│   └── test_timecode.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Deliverables for this pass

1. A runnable PySide6 application launched by `python main.py`.
2. A concise README with local setup and run instructions.
3. `requirements.txt`.
4. Unit tests for output path construction, ScriptSync cleanup, and non-drop/drop-frame timecode conversion.
5. A short implementation summary listing completed features and known limitations.

## Acceptance checks

- Selecting a source folder finds supported media recursively and displays the queue before processing.
- Selecting an output parent previews the exact two output trees.
- A processed clip produces one ScriptSync file and one Timecoded file in the correct relative folders.
- The app recognizes existing outputs and follows the selected overwrite policy.
- The interface stays responsive during transcription.
- The project can run from PyCharm using the project virtual environment.

Work incrementally. Start by creating the project structure, application shell, queue scanning, media-duration probing, and tests. Then integrate the transcription engine and output writers. At each stage, run the relevant tests and report what you changed.
