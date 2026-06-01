"""Shared frozen dataclasses and enums for competition_server_v2.

All cross-module data shapes live here: task types, per-item scoring results,
per-batch scoring results, noise phase results, and batch accumulation. Using
frozen dataclasses makes the data flow explicit — no mutation-as-return-value.
"""

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any


class TaskType(StrEnum):
    ASR = auto()
    CV = auto()
    NLP = auto()


@dataclass(frozen=True)
class BatchItem:
    """One item within a batch — index into the dataset + wire task_id."""

    index: int
    task_id: Any  # audio filename | image filename | jsonl row index


@dataclass(frozen=True)
class Batch:
    """One mission batch: a task type plus its items, with correlation IDs.

    Replaces the untyped batch dict that previously flowed through
    MissionQueue → wire_payload_for_batch → score_batch →
    accumulate_batch_result.
    """

    batch_id: str
    mission_id: str | None
    mission_num: int | None
    task: TaskType
    items: tuple[BatchItem, ...]


@dataclass(frozen=True)
class ScoredItem:
    """Unified per-item result: accuracy score plus all display fields.

    to_minimal_dict() gives the compact form saved to
    match_results.jsonl.
    """

    task_id: Any
    accuracy_score: float
    # ASR + NLP: hypothesis text
    hypothesis: str | None = None
    # ASR: word/char diff display
    ref: str | None = None
    ref_tokens: list | None = None
    hyp_tokens: list | None = None
    is_cjk: bool | None = None
    # CV: bounding-box display
    image_b64: str | None = None
    gt_boxes: list | None = None
    pred_boxes: list | None = None
    greens: list | None = None
    reds: list | None = None
    yellows: list | None = None
    # NLP: question + retrieval display
    question: str | None = None
    difficulty: str | None = None
    source_docs: list | None = None
    pred_documents: list | None = None
    is_equivalent: bool | None = None
    gt_doc_chips: list | None = None
    pred_doc_chips: list | None = None
    retrieval_correct: bool | None = None

    def to_minimal_dict(self) -> dict:
        """Compact form: task_id + accuracy_score + hypothesis (ASR/NLP)."""
        d: dict = {
            "task_id": self.task_id,
            "accuracy_score": round(float(self.accuracy_score), 4),
        }
        if self.hypothesis is not None:
            d["hypothesis"] = self.hypothesis
        return d


@dataclass(frozen=True)
class ScoredBatch:
    """Return value of score_batch(). Carries both minimal and display forms."""

    task: TaskType
    batch_accuracy: float
    time_score: float
    batch_score: float
    items: tuple[ScoredItem, ...]

    def to_dict(self) -> dict:
        """Dict shape expected by accumulate_batch_result."""
        return {
            "task": self.task.value,
            "batch_accuracy": round(float(self.batch_accuracy), 4),
            "time_score": round(float(self.time_score), 4),
            "batch_score": round(float(self.batch_score), 4),
            "per_item": [it.to_minimal_dict() for it in self.items],
        }


@dataclass(frozen=True)
class BatchAccumulation:
    """Return value of accumulate_batch_result(). Pure — no I/O."""

    new_team_scores: tuple[float, ...]
    mission_multiplier: float
    record: dict


@dataclass(frozen=True)
class TeamNoiseResult:
    """Return value of TeamNoiseDriver.run(). Replaces four mutated lists."""

    team: str
    # (filename, original_bytes) for every image in the noise assignment
    items: tuple[tuple[str, bytes], ...]
    # filename -> FairnessRecord (only for images that went through fairness check)
    fairness: dict[str, Any]
    # filename -> final bytes (noised if passed, original otherwise)
    noised: dict[str, bytes]
