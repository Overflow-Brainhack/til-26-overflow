"""Match setup: pre-match dataset splitting — task pools, noise partition,
per-team CV pools.

MatchSetup is the single entry point. Construct once at match start and read
the resulting attributes.

Invariants:
  - Per-team CV slice: each team noises cv_count // n_teams images drawn from
    the full CV dataset. Only the batch-aligned portion is usable:
    usable_per_team = (cv_count//n_teams//batch_size)*batch_size.
    cv_for_cv = usable_per_team * (n_teams - 1) so target_batches never
    exceeds the actual interleaved pool size.
  - Each team's per_team_cv_pool is truncated to exactly n items
    (= n//batch_size batches), matching the ASR and NLP pool sizes so all
    three task types exhaust simultaneously.
  - noise_phase_skipped: True when every team's noise assignment is empty
    (n_teams <= 1, or cv_count // n_teams < batch_size).
  - Determinism: same seed → identical task_pools, noise_partition,
    per_team_cv_pools across restarts.
"""

import json
import logging
import random
from pathlib import Path

import constants
from domain import TaskType

logger = logging.getLogger("uvicorn.error")


def _build_task_pools(
    data_dir: Path,
    nlp_questions: list[dict],
    batch_size: int,
    n_teams: int,
    seed: int | None = None,
) -> dict[TaskType, list[list[int]]]:
    rng = random.Random(seed)

    with open(data_dir / "asr" / "asr.jsonl", encoding="utf-8") as f:
        asr_count = sum(1 for line in f if line.strip())
    with open(data_dir / "cv" / "annotations.json", encoding="utf-8") as f:
        cv_count = len(json.load(f)["images"])
    nlp_count = len(nlp_questions)

    if n_teams > 1:
        usable_per_team = (cv_count // n_teams // batch_size) * batch_size
        cv_for_cv = usable_per_team * (n_teams - 1)
    else:
        cv_for_cv = cv_count

    n_max = min(asr_count, cv_for_cv, nlp_count)
    n = (n_max // batch_size) * batch_size

    def pick_and_batch(total: int) -> list[list[int]]:
        if n <= 0 or total < n:
            return []
        indices = rng.sample(range(total), n)
        return [indices[i : i + batch_size] for i in range(0, n, batch_size)]

    return {
        TaskType.ASR: pick_and_batch(asr_count),
        TaskType.CV: pick_and_batch(cv_for_cv),
        TaskType.NLP: pick_and_batch(nlp_count),
    }


def _build_noise_partition(
    data_dir: Path,
    team_names: list[str],
    batch_size: int,
    seed: int | None = None,
) -> dict[str, list[list[int]]]:
    with open(data_dir / "cv" / "annotations.json", encoding="utf-8") as f:
        cv_count = len(json.load(f)["images"])

    n_teams = len(team_names)
    noise_per_team = min(cv_count // n_teams, 20 * 4)

    if n_teams <= 1 or noise_per_team < batch_size:
        logger.warning(
            f"Cannot build noise partition: n_teams={n_teams}, "
            f"noise_per_team={noise_per_team} < batch_size={batch_size}; "
            f"noise phase will be skipped"
        )
        return {team: [] for team in team_names}

    unused = cv_count - n_teams * noise_per_team
    if unused > 0:
        logger.info(
            f"noise partition: {unused} CV image(s) unused (cv_count % n_teams)"
        )

    rng = random.Random((seed or 0) ^ constants.NOISE_PARTITION_SEED_SALT)
    all_indices = list(range(cv_count))
    rng.shuffle(all_indices)
    all_indices = all_indices[: n_teams * noise_per_team]

    n_batches = noise_per_team // batch_size
    return {
        team: [
            all_indices[
                i * noise_per_team
                + j * batch_size : i * noise_per_team
                + (j + 1) * batch_size
            ]
            for j in range(n_batches)
        ]
        for i, team in enumerate(team_names)
    }


def _build_per_team_cv_pools(
    noise_partition: dict[str, list[list[int]]],
    team_names: list[str],
    n: int,
    batch_size: int,
) -> dict[str, list[list[int]] | None]:
    if all(len(v) == 0 for v in noise_partition.values()):
        return {team: None for team in team_names}
    target_batches = n // batch_size
    result = {}
    for team in team_names:
        others: list[list[list[int]]] = [
            batches
            for other_team, batches in noise_partition.items()
            if other_team != team
        ]
        interleaved = [
            batch for round_batches in zip(*others) for batch in round_batches
        ]
        result[team] = interleaved[:target_batches]
    return result


class MatchSetup:
    """Encapsulates the three pre-match dataset-splitting steps.

    Construct once at match start; read the resulting attributes.

    Args:
        data_dir:      Root data directory for the current stage.
        team_names:    Ordered list of team names for this match.
        nlp_questions: Pre-loaded NLP question rows (caller owns loading).
        batch_size:    Items per batch. Defaults to constants.MISSION_BATCH_SIZE.
        seed:          Match seed for deterministic shuffles. None → random.

    Attributes:
        task_pools:          {TaskType: [[idx, ...], ...]} — shared batch deck.
        noise_partition:     {team_name: [[cv_idx, ...], ...]} — per-team CV slice.
        per_team_cv_pools:   {team_name: [[cv_idx, ...], ...] | None}.
        pool_sizes:          {TaskType: batch_count} — convenience for logging.
        noise_phase_skipped: True when every team's noise assignment is empty.
    """

    def __init__(
        self,
        data_dir: Path,
        team_names: list[str],
        nlp_questions: list[dict],
        *,
        batch_size: int = constants.MISSION_BATCH_SIZE,
        seed: int | None = None,
    ) -> None:
        self.task_pools = _build_task_pools(
            data_dir, nlp_questions, batch_size, len(team_names), seed
        )
        n = len(self.task_pools[TaskType.ASR]) * batch_size
        self.noise_partition = _build_noise_partition(
            data_dir, team_names, batch_size, seed
        )
        self.per_team_cv_pools = _build_per_team_cv_pools(
            self.noise_partition, team_names, n, batch_size
        )
        self.pool_sizes: dict[TaskType, int] = {
            t: len(v) for t, v in self.task_pools.items()
        }
        self.noise_phase_skipped: bool = all(
            len(v) == 0 for v in self.noise_partition.values()
        )
