"""Manifest state machine — the single source of truth lives in data/manifest.jsonl.

One JSON object per line, keyed by clip_id. Every pipeline stage:
  rows = load()
  for r in by_stage(rows, "segmented"): ... process ...
  update(rows, clip_id, stage="music_checked", has_music=False, ...)
  save(rows)

Crash-safe: save() writes to a temp file then atomically replaces the manifest, so an
interrupted run never leaves a half-written manifest. Idempotent: rerunning a stage
skips rows already past it (callers filter with by_stage).

See CLAUDE.md for the no-overwrite rule: machine (asr_*/llm_*) and human (human_*) keys
are separate and never share a key.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, Iterator

MANIFEST_PATH = os.path.join("data", "manifest.jsonl")


def load(path: str = MANIFEST_PATH) -> Dict[str, dict]:
    """Load the manifest into {clip_id: row}. Missing file -> empty dict."""
    rows: Dict[str, dict] = {}
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["clip_id"]] = row
    return rows


def save(rows: Dict[str, dict], path: str = MANIFEST_PATH) -> None:
    """Atomically write {clip_id: row} back to JSONL (UTF-8, ensure_ascii=False)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for clip_id in sorted(rows):
                f.write(json.dumps(rows[clip_id], ensure_ascii=False))
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def update(rows: Dict[str, dict], clip_id: str, **fields) -> dict:
    """Set fields on a row in place (creating it if new) and return it.

    Does not persist — call save(rows) when ready. New rows must include clip_id.
    """
    row = rows.setdefault(clip_id, {"clip_id": clip_id})
    row.update(fields)
    return row


def by_stage(rows: Dict[str, dict], stage: str) -> Iterator[dict]:
    """Yield rows currently at the given stage."""
    for row in rows.values():
        if row.get("stage") == stage:
            yield row
