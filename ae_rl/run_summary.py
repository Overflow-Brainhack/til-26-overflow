"""Run-summary writer for ae_rl training scripts.

Writes a JSON file describing what happened in a training run — args, runtime,
checkpoint paths, validation history, action distribution, status (completed/
failed/interrupted). Designed so an autonomous caller (Claude or a CI script)
can read structured results instead of parsing tqdm stdout.

Default location: ``ae_rl/runs/<stage>/latest.json`` (overwritten per run) plus
``ae_rl/runs/<stage>/<timestamp>.json`` for history. The JSON is rewritten
atomically on every ``.write()`` so a long-running reader sees a partial
snapshot mid-run and the final state when training exits.

Usage:
    with RunSummary(stage="stage2_ppo", args=vars(args), path=path) as summary:
        summary.set("device", str(device))
        for update in range(...):
            ...
            summary.increment("updates_completed")
            summary.record("updates", {"update": update, "ret": ...})
            if validation_due:
                summary.record("validations", {"update": update, "score": ...})
                summary.set("best_validation_score", best)
                summary.set("best_checkpoint", str(best_ckpt))
            if update % 10 == 0:
                summary.write()  # checkpoint progress for live polling
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def default_summary_path(stage: str) -> Path:
    """Canonical summary path for a stage: ``ae_rl/runs/<stage>/latest.json``."""
    return Path(__file__).resolve().parent / "runs" / stage / "latest.json"


def _make_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_make_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return str(value)


class RunSummary:
    """Context-managed JSON summary for one training-script invocation."""

    def __init__(self, *, stage: str, args: dict, path: Path | str | None = None):
        self.stage = stage
        self.path = Path(path) if path else default_summary_path(stage)
        # History copy (timestamped) alongside latest.json. Helps when a sequence
        # of runs gets compared without overwrite.
        self._history_path = (
            self.path.parent / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        )
        self.data: dict[str, Any] = {
            "stage": stage,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "args": _make_jsonable(args),
            "updates_completed": 0,
            "validations": [],
        }
        self._t0 = time.time()

    # ── public helpers ──────────────────────────────────────────────────────
    def set(self, key: str, value: Any) -> None:
        self.data[key] = _make_jsonable(value)

    def update_dict(self, mapping: dict) -> None:
        for k, v in mapping.items():
            self.set(k, v)

    def record(self, key: str, value: Any) -> None:
        self.data.setdefault(key, []).append(_make_jsonable(value))

    def increment(self, key: str, by: int = 1) -> None:
        self.data[key] = int(self.data.get(key, 0)) + int(by)

    def write(self) -> None:
        """Atomic write to ``self.path`` (and timestamped copy)."""
        self.data["last_written_at"] = datetime.now(timezone.utc).isoformat()
        self.data["elapsed_seconds"] = round(time.time() - self._t0, 2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.data, indent=2, default=str)
        for target in (self.path, self._history_path):
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(target)

    # ── context manager ─────────────────────────────────────────────────────
    def __enter__(self) -> "RunSummary":
        self.write()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.data["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.data["duration_seconds"] = round(time.time() - self._t0, 2)
        if exc_type is None:
            self.data["status"] = "completed"
        elif issubclass(exc_type, KeyboardInterrupt):
            self.data["status"] = "interrupted"
        else:
            self.data["status"] = "failed"
            self.data["error"] = f"{exc_type.__name__}: {exc}"
        self.write()
        return False  # never suppress the exception
