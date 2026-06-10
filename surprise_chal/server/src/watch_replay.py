"""Watch a replay (the interactive viewer).

    python watch_replay.py                 # opens the newest replay in replays/
    python watch_replay.py path/to.jsonl   # opens a specific replay

Needs the viewer extras (arcade):  pip install -r ../requirements-viewer.txt
Controls: Space play/pause · ←/→ step · +/- speed · scroll zoom · drag pan ·
F fog perspective · Tab chat channel · C chat · F11 fullscreen.
"""

from __future__ import annotations

import glob
import os
import sys


def _newest_replay() -> str | None:
    # An in-process run writes server/src/replays/; docker compose writes the
    # repo-root replays/ (two levels up from here). Search the likely spots.
    files = (
        glob.glob("replays/*.jsonl")
        + glob.glob("../replays/*.jsonl")
        + glob.glob("../../replays/*.jsonl")
    )
    return max(files, key=os.path.getmtime) if files else None


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else _newest_replay()
    if not path or not os.path.exists(path):
        sys.exit(
            "no replay found. Run a game first:  docker compose up --build\n"
            "or pass a path:  python watch_replay.py replays/<file>.jsonl"
        )
    try:
        from replay.viewer import launch_viewer
    except ImportError as exc:
        sys.exit(
            f"viewer dependencies missing ({exc}).\n"
            "Install them:  pip install -r ../requirements-viewer.txt"
        )
    print(f"opening {path} …")
    launch_viewer(path)


if __name__ == "__main__":
    main()
