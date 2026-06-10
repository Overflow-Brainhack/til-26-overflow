"""replay recorder: writes one JSONL line per turn for later playback"""

from __future__ import annotations

import json
from pathlib import Path

from engine.actions import ActionPayload, payload_to_dict
from engine.state import GameState

# Cap how much chat text is PERSISTED to the replay, so a length-DoS player
# (oversized "context bomb" messages) can't bloat the .jsonl or the viewer's
# in-memory load (ReplayLoader reads the whole file). Truncation happens here at
# the data layer, not just at display time. It does NOT touch the live
# observation sent to opponents — the in-flight DoS is unaffected; only the
# saved record shrinks.
#
# Chat text reaches the replay through TWO fields and BOTH must be clipped:
#   1. the per-turn `chat` log (the delivered messages), and
#   2. the `send_chat` actions inside each player's `actions` payload (the raw
#      action that produced the message — a length-DoS bomb lives here in full).
# Clipping only (1) left (2) uncapped, so the bomb was still persisted whole.
_MAX_CHAT_CHARS = 200


def _clip(text: str) -> str:
    return text[: _MAX_CHAT_CHARS - 3] + "..." if len(text) > _MAX_CHAT_CHARS else text


def _truncate_chat(chat: list[dict] | None) -> list[dict]:
    """Return chat records with overlong text clipped to _MAX_CHAT_CHARS + '...'."""
    out: list[dict] = []
    for m in chat or []:
        text = m.get("text", "")
        if isinstance(text, str) and len(text) > _MAX_CHAT_CHARS:
            m = {**m, "text": _clip(text)}
        out.append(m)
    return out


def _truncate_action_chat(actions: dict[str, dict]) -> dict[str, dict]:
    """Clip overlong `send_chat` text inside each serialised action payload, so a
    length-DoS bomb is not stored uncapped in the replay's `actions` field."""
    out: dict[str, dict] = {}
    for pid, payload in actions.items():
        acts = payload.get("actions", [])
        new_acts = []
        for a in acts:
            if (
                isinstance(a, dict)
                and a.get("type") == "send_chat"
                and isinstance(a.get("text"), str)
                and len(a["text"]) > _MAX_CHAT_CHARS
            ):
                a = {**a, "text": _clip(a["text"])}
            new_acts.append(a)
        out[pid] = {**payload, "actions": new_acts}
    return out


class ReplayRecorder:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._file = open(path, "w", encoding="utf-8")

    def record_initial(
        self,
        state: GameState,
        seed: int,
        chat: list[dict] | None = None,
        treaties: list[dict] | None = None,
    ) -> None:
        """write the header record with full initial state and map seed"""
        record = {
            "turn": 0,
            "seed": seed,
            "map_width": state.grid.width,
            "map_height": state.grid.height,
            "state_snapshot": state.to_dict(),
            "chat": _truncate_chat(chat),
            "treaties": treaties or [],
            "actions": {},
        }
        self._write(record)

    def record_turn(
        self,
        turn_number: int,
        actions: dict[str, ActionPayload],
        state: GameState,
        chat: list[dict] | None = None,
        treaties: list[dict] | None = None,
    ) -> None:
        """write one record after a turn has been processed.

        `treaties` is the list of treaty pairs in effect AFTER this turn resolved
        (from DiplomacyManager.active_treaty_pairs) — drives the viewer's treaty
        graph. Empty from the treaty-cutoff turn onward.
        """
        record = {
            "turn": turn_number,
            "state_snapshot": state.to_dict(),
            "chat": _truncate_chat(chat),
            "treaties": treaties or [],
            "actions": _truncate_action_chat(
                {pid: payload_to_dict(p) for pid, p in actions.items()}
            ),
        }
        self._write(record)

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __del__(self) -> None:
        try:
            if not self._file.closed:
                self._file.close()
        except Exception:
            pass


class ReplayLoader:
    """load and iterate records from a replay file"""

    def __init__(self, path: str) -> None:
        self._path = path
        self._records: list[dict] = []
        self._load()

    def _load(self) -> None:
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    @property
    def header(self) -> dict:
        return self._records[0] if self._records else {}

    @property
    def turns(self) -> list[dict]:
        return self._records[1:] if len(self._records) > 1 else []

    @property
    def total_turns(self) -> int:
        return len(self._records)

    def get_turn(self, index: int) -> dict:
        return self._records[index]

    def treaties_for_turn(self, index: int) -> list[dict]:
        """Treaty pairs in effect at the given record index (empty if absent —
        older replays predating treaty recording, or the post-cutoff turns)."""
        if 0 <= index < len(self._records):
            return self._records[index].get("treaties", [])
        return []

    def seed(self) -> int:
        return self.header.get("seed", 0)

    def map_dims(self) -> tuple[int, int]:
        return self.header.get("map_width", 35), self.header.get("map_height", 30)
