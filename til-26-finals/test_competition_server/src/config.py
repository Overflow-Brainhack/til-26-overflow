"""Match configuration: load from CONFIG env var + JSON file.

A single MatchConfig frozen dataclass carries every fact a server process
needs.  Callers import load_config(); everything else is derived from the
JSON on disk.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatchConfig:
    teams: list[str]
    track: str
    stage: str
    match: int
    stage_dir: Path
    nlp_eval_model_path: Path


def load_config(data_root: Path = Path("../data")) -> MatchConfig:
    """Read CONFIG env var, load the JSON, return a MatchConfig."""
    filename = os.environ["CONFIG"]
    config_path = Path(f"../configs/{filename}.json")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Seat the real participant's team. The HQ accepts WebSocket connections only from
    # teams in this list, so if TEAM_NAME isn't already a configured seat we inject it at
    # slot 0 (preserving the seat count, so dataset-splitting math is unchanged). This lets
    # a team test under their own name without editing config JSONs. No-op when TEAM_NAME is
    # unset (production) or already present. The stubs apply the same rule, so the two agree.
    teams = list(raw["teams"])
    real_team = os.environ.get("TEAM_NAME", "").strip()
    if real_team and teams and real_team not in teams:
        print(f"[config] seating real team {real_team!r} at slot 0 (was {teams[0]!r})", flush=True)
        teams = [real_team, *teams[1:]]

    return MatchConfig(
        teams=teams,
        track=raw["track"],
        stage=raw["stage"],
        match=raw["match"],
        stage_dir=data_root,
        nlp_eval_model_path=data_root / "nlp" / "models" / "nlp_eval_512",
    )
