#!/usr/bin/env python3
"""Record the exact size and SHA-256 of every diarization model asset.

The application refuses to promote a downloaded model that does not match its
recorded digest, which is what stops a truncated or substituted file from
being cached and then trusted offline forever. Those digests have to come from
somewhere, and that somewhere is this script, run on a machine that can reach
the model host.

Usage::

    python scripts/record_model_digests.py            # print the values
    python scripts/record_model_digests.py --write    # patch model_cache.py

Nothing is downloaded into the real cache folder: the files go to a temporary
directory and are deleted afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transcription.model_cache import DEFAULT_BUNDLE, ModelAsset  # noqa: E402

MODEL_CACHE_SOURCE = PROJECT_ROOT / "transcription" / "model_cache.py"


def fetch(asset: ModelAsset, destination: Path) -> tuple[int, str]:
    """Download one asset and return its byte count and SHA-256."""
    request = urllib.request.Request(
        asset.url, headers={"User-Agent": "MLX-Transcript"}
    )
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=120) as response:
        declared = int(response.headers.get("Content-Length") or 0)
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if declared:
                    percent = 100.0 * size / declared
                    print(f"\r  {asset.name}: {percent:5.1f}%", end="", flush=True)
    print()
    if declared and size != declared:
        raise SystemExit(
            f"{asset.name}: got {size} bytes, server declared {declared}."
        )
    return size, digest.hexdigest()


def patch_source(results: dict[str, tuple[int, str]]) -> None:
    """Write the recorded values back into the asset definitions."""
    text = MODEL_CACHE_SOURCE.read_text(encoding="utf-8")
    for filename, (size, digest) in results.items():
        block = re.search(
            r"(filename=\"" + re.escape(filename) + r"\",.*?\n\)",
            text,
            re.DOTALL,
        )
        if block is None:
            raise SystemExit(f"Could not find the asset block for {filename}.")
        updated = block.group(0)
        updated = re.sub(
            r"size_bytes=[\d_]+,", f"size_bytes={size:_d},", updated, count=1
        )
        updated = re.sub(
            r'sha256="[^"]*",', f'sha256="{digest}",', updated, count=1
        )
        text = text.replace(block.group(0), updated, 1)
    MODEL_CACHE_SOURCE.write_text(text, encoding="utf-8")
    print(f"Updated {MODEL_CACHE_SOURCE.relative_to(PROJECT_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="patch transcription/model_cache.py with the recorded values",
    )
    arguments = parser.parse_args()

    results: dict[str, tuple[int, str]] = {}
    with tempfile.TemporaryDirectory(prefix="mlx-transcript-digests-") as staging:
        for asset in DEFAULT_BUNDLE.assets:
            destination = Path(staging) / asset.filename
            size, digest = fetch(asset, destination)
            results[asset.filename] = (size, digest)
            note = "" if size == asset.size_bytes else f"  (was {asset.size_bytes})"
            print(f"{asset.filename}\n  size_bytes={size:_d}{note}\n  sha256=\"{digest}\"")

    if arguments.write:
        patch_source(results)
    else:
        print("\nRe-run with --write to patch the asset definitions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
