"""NoisePhase: orchestrate all team drivers and build the noised_lookup.

NoisePhase runs six TeamNoiseDriver coroutines concurrently, collects their
TeamNoiseResult returns, builds the noised_lookup dict, emits fairness events,
and writes the audit dump.
"""

import asyncio
import logging
from pathlib import Path

import numpy as np
from artifacts import EventLog, MatchDir
from domain import TeamNoiseResult
from missions import TaskHandler
from transport import WebSocketManager

from .driver import TeamNoiseDriver

logger = logging.getLogger("uvicorn.error")


class NoisePhase:
    """Pre-match noise phase: distribute, collect, fairness-check, build lookup.

    run() drives all teams concurrently and returns the noised_lookup dict
    {filename: bytes} ready to be passed to TaskHandler constructors.
    """

    def __init__(
        self,
        team_names: list[str],
        noise_partition: dict[str, list[list[int]]],
        noise_phase_skipped: bool,
        ref_handler: TaskHandler,
        ws: WebSocketManager,
        pending_batches: dict[str, tuple[str, asyncio.Future]],
        events: EventLog,
        match_dir: MatchDir,
    ) -> None:
        self._team_names = team_names
        self._noise_partition = noise_partition
        self._noise_phase_skipped = noise_phase_skipped
        self._ref_handler = ref_handler
        self._ws = ws
        self._pending_batches = pending_batches
        self._events = events
        self._match_dir = match_dir
        self._distributed = False

    async def run(self) -> dict[str, bytes]:
        """Return noised_lookup {filename: bytes}. Idempotent."""
        if self._distributed:
            return {}
        self._distributed = True

        if self._noise_phase_skipped:
            logger.info("Noise partition is empty — skipping noise phase")
            self._events.emit("noise_phase_skipped", reason="empty_partition")
            return {}

        self._events.emit(
            "noise_phase_start",
            items_per_team={
                t: sum(len(b) for b in batches)
                for t, batches in self._noise_partition.items()
            },
        )

        # Run all team drivers concurrently
        drivers = [
            TeamNoiseDriver(
                team=team,
                assignment=self._noise_partition.get(team, []),
                ref_handler=self._ref_handler,
                ws=self._ws,
                pending_batches=self._pending_batches,
                events=self._events,
            )
            for team in self._team_names
        ]
        raw: list = await asyncio.gather(
            *[d.run() for d in drivers],
            return_exceptions=True,
        )

        noised_lookup: dict[str, bytes] = {}
        for team, outcome in zip(self._team_names, raw):
            if isinstance(outcome, BaseException):
                logger.error(
                    f"[{team}] noise driver failed: {outcome!r}; originals used"
                )
                self._events.emit("noise_driver_failed", team=team, error=repr(outcome))
                continue
            result: TeamNoiseResult = outcome
            noised_lookup.update(result.noised)
            self._emit_fairness_events(result)
            if result.items:
                all_recs = list(result.fairness.values())
                passed = sum(1 for r in all_recs if r.passed)
                fallback = sum(1 for r in all_recs if r.fallback_used)
                timeout = len(result.items) - len(all_recs)
                logger.info(
                    f"[{result.team}] noise complete"
                    f"  images={len(result.items)}"
                    f"  passed={passed}"
                    f"  fallback={fallback}"
                    f"  timeout={timeout}"
                )

        # Audit dump
        if noised_lookup and self._match_dir.path:
            snap = dict(noised_lookup)
            audit_dir = self._match_dir.noised_dir()
            if audit_dir is not None:
                await asyncio.to_thread(self._write_audit, snap, audit_dir)

        self._events.emit("noise_phase_end", noised_count=len(noised_lookup))
        logger.info(f"Noise phase complete: {len(noised_lookup)} images stored")
        return noised_lookup

    def _emit_fairness_events(self, result: TeamNoiseResult) -> None:
        team = result.team
        if not result.items:
            return

        all_records = list(result.fairness.values())
        total = len(result.items)
        passed_count = sum(1 for r in all_records if r.passed)
        failed_count = sum(1 for r in all_records if not r.passed)
        timeout_count = total - len(all_records)

        for fn, rec in result.fairness.items():
            if rec.passed:
                continue
            self._events.emit(
                "noise_fairness_image_failed",
                team=team,
                filename=fn,
                metrics={k: round(float(v), 4) for k, v in rec.metrics.items()},
                failed_checks=rec.failed_checks,
                noised_missing=rec.noised_missing,
                fallback_used=rec.fallback_used,
            )

        def _agg(name: str) -> dict:
            vals = [
                r.metrics[name]
                for r in all_records
                if name in r.metrics and r.metrics[name] == r.metrics[name]
            ]
            if not vals:
                return {}
            return {
                "mean": round(float(np.mean(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

        self._events.emit(
            "noise_fairness_result",
            team=team,
            total=total,
            passed=passed_count,
            failed=failed_count,
            skipped=timeout_count,
            n_noised_missing=sum(1 for r in all_records if r.noised_missing),
            metric_stats={
                "L2 (RMSE)": _agg("L2 (RMSE)"),
                "L2 inside": _agg("L2 inside"),
                "SSIM inside": _agg("SSIM inside"),
            },
        )
        if failed_count:
            first_fn = next(fn for fn, r in result.fairness.items() if not r.passed)
            first_rec = result.fairness[first_fn]
            logger.warning(
                f"[{team}] {failed_count}/{total} fairness fails; "
                f"first={first_fn} metrics={first_rec.metrics} "
                f"failed_checks={first_rec.failed_checks}"
            )
        logger.info(
            f"[{team}] noise fairness: {passed_count}/{total} passed"
            f" ({failed_count} failed, {timeout_count} skipped)"
        )

    @staticmethod
    def _write_audit(noised_lookup: dict[str, bytes], audit_dir: Path) -> None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        for fname, img_bytes in noised_lookup.items():
            try:
                (audit_dir / fname).write_bytes(img_bytes)
            except Exception:
                pass
