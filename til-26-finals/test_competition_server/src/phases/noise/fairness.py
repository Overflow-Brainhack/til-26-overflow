"""Noise fairness check: validate noised images, fall back to originals.

apply_fairness_fallback() runs synchronously (call it via asyncio.to_thread)
and returns one FairnessRecord per image alongside the bytes to actually use
(noised if it passed every check, original otherwise).
"""

import base64
import io
import logging
import threading
from dataclasses import dataclass

import constants
import numpy as np
from PIL import Image

from noise_eval.fairness_checker import FairnessChecker
from noise_eval.pipeline import EvalPipeline

logger = logging.getLogger("uvicorn.error")


@dataclass
class FairnessRecord:
    """Per-image result from the fairness check."""

    passed: bool
    metrics: dict[str, float]
    failed_checks: list[dict]
    noised_missing: bool
    fallback_used: bool


# Lazy-init noise eval singletons (thread-safe double-checked locking)
_pipeline: EvalPipeline | None = None
_checker: FairnessChecker | None = None
_eval_lock = threading.Lock()


def _get_pipeline() -> EvalPipeline:
    global _pipeline
    if _pipeline is None:
        with _eval_lock:
            if _pipeline is None:
                _pipeline = EvalPipeline()
    return _pipeline


def _get_checker() -> FairnessChecker:
    global _checker
    if _checker is None:
        with _eval_lock:
            if _checker is None:
                _checker = FairnessChecker(constants.NOISE_FAIRNESS_CONFIG_PATH)
    return _checker


def apply_fairness_fallback(
    original_bytes_list: list[bytes],
    noised_b64_list: list[str | None],
    boxes_list: list[np.ndarray],
) -> tuple[list[bytes], list[FairnessRecord]]:
    """Per-image fairness check with fallback to originals on failure.

    Runs synchronously (call via asyncio.to_thread from an async context).
    Returns (result_bytes, records) in the same order as the input lists.

    Each image is evaluated INDEPENDENTLY: a malformed noised image (garbage
    base64, wrong dimensions, smaller than the SSIM window, undecodable) only
    falls back to its own original — it can never raise out of this function
    and discard the rest of the team's images.
    """
    pipeline = _get_pipeline()
    checker = _get_checker()

    result_bytes: list[bytes] = []
    records: list[FairnessRecord] = []

    for orig_bytes, noised_b64, boxes in zip(
        original_bytes_list, noised_b64_list, boxes_list
    ):
        # No noised image supplied → fall back to the original.
        if noised_b64 is None:
            result_bytes.append(orig_bytes)
            records.append(
                FairnessRecord(
                    passed=False,
                    metrics={},
                    failed_checks=[],
                    noised_missing=True,
                    fallback_used=True,
                )
            )
            continue

        try:
            noised_raw = base64.b64decode(noised_b64)
            orig_arr = np.array(
                Image.open(io.BytesIO(orig_bytes)).convert("RGB"), dtype=np.uint8
            )
            noised_arr = np.array(
                Image.open(io.BytesIO(noised_raw)).convert("RGB"), dtype=np.uint8
            )
            summary = pipeline.evaluate_batched_with_boxes(
                [orig_arr], [noised_arr], [boxes]
            )
            metrics_dict = summary.per_image[0].to_dict()
            check = checker.evaluate(metrics_dict)
            use_noised = check.passed
            failed_checks = [
                {
                    "metric": c.metric,
                    "value": c.value,
                    "threshold": c.threshold,
                    "op": c.op,
                    "missing": c.missing,
                }
                for c in check.checks
                if not c.passed
            ]
            result_bytes.append(noised_raw if use_noised else orig_bytes)
            records.append(
                FairnessRecord(
                    passed=bool(use_noised),
                    metrics=metrics_dict,
                    failed_checks=failed_checks,
                    noised_missing=False,
                    fallback_used=not use_noised,
                )
            )
        except Exception:
            # Undecodable / wrong-sized / too-small image, etc. Contain it to
            # this one image and fall back to the original.
            logger.warning(
                "fairness check failed for one image; using original",
                exc_info=True,
            )
            result_bytes.append(orig_bytes)
            records.append(
                FairnessRecord(
                    passed=False,
                    metrics={},
                    failed_checks=[],
                    noised_missing=False,
                    fallback_used=True,
                )
            )

    return result_bytes, records
