"""Match artifact management: event log, results log, output directory.

Three classes:
  EventLog   — append-only JSONL of operational events (events.jsonl).
  ResultsLog — append-only JSONL of match results (match_results.jsonl).
  MatchDir   — creates the per-match artifact directory and owns both logs.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger("uvicorn.error")


class EventLog:
    """Append-only JSONL event log. Writes {event, **fields} per line."""

    def __init__(self, path: str) -> None:
        self._path = path

    def emit(self, event: str, **fields) -> None:
        if not self._path:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": event, **fields}) + "\n")
        except Exception:
            logger.exception(f"failed to write event {event!r}")


class ResultsLog:
    """Append-only JSONL match results log. Writes one dict per line."""

    def __init__(self, path: str) -> None:
        self._path = path

    def append(self, record: dict) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class MatchDir:
    """Creates and owns the per-match artifact directory.

    Call once at match start (ensure_created is idempotent). Exposes:
      .path    — absolute string path to the directory
      .events  — EventLog for events.jsonl
      .results — ResultsLog for match_results.jsonl

    Lazily initialised: events and results are None until ensure_created()
    is called, so the server can be instantiated without immediately touching
    the filesystem.
    """

    def __init__(self, track: str, stage: str) -> None:
        self._track = track
        self._stage = stage
        self.path: str | None = None
        self.events: EventLog | None = None
        self.results: ResultsLog | None = None
        self._created = False

    def ensure_created(self) -> None:
        """Idempotent. Creates the directory and both log files on first call."""
        if self._created:
            return
        self._created = True
        try:
            tz = ZoneInfo("Asia/Singapore")
        except Exception:
            tz = timezone(timedelta(hours=8))
        ts = datetime.now(tz).strftime("%Y-%m-%d-%H-%M-%S")
        self.path = f"../artifacts/match_{self._track}__{self._stage}_{ts}"
        os.makedirs(self.path, exist_ok=True)
        self.events = EventLog(f"{self.path}/events.jsonl")
        self.results = ResultsLog(f"{self.path}/match_results.jsonl")

    def noised_dir(self) -> Path | None:
        """Return the audit dump path for noised images, or None if not created."""
        if self.path is None:
            return None
        return Path(self.path) / "noised"

    def events_path(self) -> str | None:
        return f"{self.path}/events.jsonl" if self.path else None

    def video_path(self) -> str | None:
        return f"{self.path}/match.mp4" if self.path else None
