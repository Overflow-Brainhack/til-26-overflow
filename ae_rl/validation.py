"""Validation helpers for gated RL checkpointing."""

from __future__ import annotations

from benchmark import benchmark


def validate_model(
    model,
    *,
    rounds: int,
    learners: int,
    novice: bool = True,
    seed: int = 0,
    advanced_rounds: int = 0,
) -> dict:
    """Run quiet fixed-seed benchmark(s) and return a scalar promotion score.

    The score is mean reward delta over the heuristic reference, averaged across
    enabled map modes. Positive means the learned agents beat the 6x heuristic
    reference under this validation suite.
    """
    results = []
    if rounds > 0:
        results.append(benchmark(
            None, rounds, learners, novice, seed,
            model=model, quiet=True, deterministic=True, rotate_slots=True,
        ))
    if advanced_rounds > 0:
        results.append(benchmark(
            None, advanced_rounds, learners, False, seed + 10_000,
            model=model, quiet=True, deterministic=True, rotate_slots=True,
        ))

    deltas = [r["delta"] for r in results if r.get("delta") is not None]
    rl_means = [r["rl_mean"] for r in results if r.get("rl_mean") is not None]
    baselines = [r["heur_baseline"] for r in results if r.get("heur_baseline") is not None]
    return {
        "score": float(sum(deltas) / len(deltas)) if deltas else float("-inf"),
        "rl_mean": float(sum(rl_means) / len(rl_means)) if rl_means else 0.0,
        "heur_baseline": float(sum(baselines) / len(baselines)) if baselines else 0.0,
        "num_suites": len(results),
    }
