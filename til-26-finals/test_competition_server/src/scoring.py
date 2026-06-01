"""Pure batch scoring: score_batch() → ScoredBatch, accumulate_batch_result() → BatchAccumulation.

All display data is folded into ScoredItem so there is one shape per item.

Module-level helpers (score_asr, score_cv_single, score_nlp_rows) are
importable directly for unit tests — no filesystem access required.
"""

import contextlib
import io
import logging
import re
import traceback
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import constants
import jiwer
import nlp_eval
from domain import (
    Batch,
    BatchAccumulation,
    BatchItem,
    ScoredBatch,
    ScoredItem,
    TaskType,
)
from imaging import thumbnail_b64

if TYPE_CHECKING:
    from missions import TaskHandler
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = logging.getLogger("uvicorn.error")

# ── display helpers ───────────────────────────────────────────────────────────

_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")


def _tokenize_display(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    if _CJK_RE.search(s):
        return [c for c in s if c.strip()]
    return list(s.lower().split())


def _word_diff_display(ref: str, hyp: str) -> tuple[list[dict], list[dict], bool]:
    """Levenshtein word/char diff for ASR display."""
    rw = _tokenize_display(ref)
    hw = _tokenize_display(hyp)
    is_cjk = bool(_CJK_RE.search((ref or "") + (hyp or "")))
    n, m = len(rw), len(hw)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = (
                dp[i - 1][j - 1]
                if rw[i - 1] == hw[j - 1]
                else 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
            )
    r_err = [False] * n
    h_err = [False] * m
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and rw[i - 1] == hw[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            i -= 1
            j -= 1
            r_err[i] = True
            h_err[j] = True
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j - 1] + 1):
            j -= 1
            h_err[j] = True
        else:
            i -= 1
            r_err[i] = True
    return (
        [{"t": w, "e": r_err[k]} for k, w in enumerate(rw)],
        [{"t": w, "e": h_err[k]} for k, w in enumerate(hw)],
        is_cjk,
    )


def _iou_display(a: list, b: list) -> float:
    ax2 = a[0] + a[2]
    ay2 = a[1] + a[3]
    bx2 = b[0] + b[2]
    by2 = b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    return 0.0 if inter == 0 else inter / (a[2] * a[3] + b[2] * b[3] - inter)


def _match_boxes_display(gt_boxes: list, pred_boxes: list, thr: float = 0.5) -> dict:
    """Classify boxes: greens=TP preds, reds=FN GTs, yellows=FP preds."""
    gt_m = [False] * len(gt_boxes)
    p_m = [False] * len(pred_boxes)
    for pi, pred in enumerate(pred_boxes):
        best, gi = 0.0, -1
        for i, gt in enumerate(gt_boxes):
            if not gt_m[i]:
                iou = _iou_display(gt, pred)
                if iou > best:
                    best, gi = iou, i
        if best >= thr:
            gt_m[gi] = True
            p_m[pi] = True
    return {
        "greens": [b for i, b in enumerate(pred_boxes) if p_m[i]],
        "reds": [b for i, b in enumerate(gt_boxes) if not gt_m[i]],
        "yellows": [b for i, b in enumerate(pred_boxes) if not p_m[i]],
    }


def _doc_chips_display(source_docs: list, pred_docs: list) -> dict:
    """Classify NLP doc chips for display."""
    src = [str(d) for d in (source_docs or [])]
    pred = [str(d) for d in (pred_docs or [])]
    matched = {d for d in pred if d in set(src)}
    any_match = bool(matched)

    def cls(d: str) -> str:
        return "correct" if d in matched else ("neutral" if any_match else "wrong")

    return {
        "gt_doc_chips": [{"id": d, "cls": cls(d)} for d in src],
        "pred_doc_chips": [{"id": d, "cls": cls(d)} for d in pred],
        "retrieval_correct": any_match,
    }


def _cv_thumbnail_b64(handler: "TaskHandler", img_id: int) -> str | None:
    try:
        file_name = handler.cv_img_info[img_id]["file_name"]
        with Image.open(handler.cv_dir / "images" / file_name) as pil_img:
            return thumbnail_b64(pil_img, quality=50)
    except Exception:
        return None


# ── ASR transforms ────────────────────────────────────────────────────────────

wer_transforms = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"-": " ", "—": " ", "–": " "}),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

cer_transforms = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"-": "", "—": "", "–": ""}),
        jiwer.RemoveWhiteSpace(replace_by_space=False),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)


def _asr_scorer_for(language: str):
    if language == "chinese":
        return (
            partial(
                jiwer.process_characters,
                reference_transform=cer_transforms,
                hypothesis_transform=cer_transforms,
            ),
            False,
        )
    return (
        partial(
            jiwer.process_words,
            reference_transform=wer_transforms,
            hypothesis_transform=wer_transforms,
        ),
        True,
    )


# ── public module-level scorers ───────────────────────────────────────────────


def score_asr(reference: str, hypothesis: str, language: str = "english") -> float:
    """Per-item ASR accuracy: max(1 - WER/CER, 0.0). Empty hypothesis → 0."""
    scorer, is_word_level = _asr_scorer_for(language)
    output = scorer(reference or "", hypothesis or "")
    error_rate = output.wer if is_word_level else output.cer
    return max(1.0 - error_rate, 0.0)


class COCOPatched(COCO):
    def __init__(self, annotations):
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        assert type(annotations) is dict, (
            f"Annotation format {type(annotations)} not supported"
        )
        self.dataset = annotations
        self.createIndex()


def score_cv_single(
    gt_annotations: list[dict],
    detections: list[dict],
    img_info: dict,
    categories: list[dict],
) -> float:
    """mAP@.5:.05:.95 for one image. Returns 0.0 on empty/invalid preds."""
    if not detections:
        return 0.0
    img_id = img_info["id"]
    preds = [
        {
            "image_id": img_id,
            "score": 1.0,
            "bbox": det["bbox"],
            "category_id": det["category_id"],
        }
        for det in detections
        if "bbox" in det and "category_id" in det
    ]
    if not preds:
        return 0.0
    try:
        anns = {
            "images": [img_info],
            "annotations": gt_annotations,
            "categories": categories,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            gt_coco = COCOPatched(anns)
            res = gt_coco.loadRes(preds)
            coco_eval = COCOeval(gt_coco, res, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        return float(coco_eval.stats[0].item())
    except Exception:
        traceback.print_exc()
        return 0.0


def score_nlp_rows(rows: list[tuple], evaluator) -> list[float]:
    """Score NLP rows using an already-constructed answer-equivalence evaluator."""
    return [r.score for r in evaluator.batch_evaluate(rows)]


# ── per-item scorers returning ScoredItem ─────────────────────────────────────


def _valid_bbox(b: Any) -> bool:
    """A usable COCO bbox: a list/tuple of at least 4 real numbers."""
    return (
        isinstance(b, (list, tuple))
        and len(b) >= 4
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in b[:4])
    )


def _score_asr_item(
    handler: "TaskHandler", item: BatchItem, pred: dict | None
) -> ScoredItem:
    inst = handler.asr_instances[item.index]
    ref = inst.get("transcript", "") or ""
    # The hypothesis is untrusted. Only a string is a valid transcript; anything
    # else (dict/list/number) is coerced to "" so it is never handed to jiwer,
    # which scores the item 0 for a non-empty reference rather than crashing.
    raw_hyp = (pred or {}).get("answer", "")
    if raw_hyp not in (None, "") and not isinstance(raw_hyp, str):
        logger.warning(
            "ASR item task_id=%r: non-string answer (%s); scoring 0.0",
            item.task_id,
            type(raw_hyp).__name__,
        )
    hyp = raw_hyp if isinstance(raw_hyp, str) else ""
    accuracy = (
        score_asr(ref, hyp, inst.get("language", "english"))
        if pred is not None
        else 0.0
    )
    ref_tokens, hyp_tokens, is_cjk = _word_diff_display(ref, hyp)
    return ScoredItem(
        task_id=item.task_id,
        accuracy_score=accuracy,
        hypothesis=hyp,
        ref=ref,
        ref_tokens=ref_tokens,
        hyp_tokens=hyp_tokens,
        is_cjk=is_cjk,
    )


def _score_cv_item(
    handler: "TaskHandler", item: BatchItem, pred: dict | None
) -> ScoredItem:
    img_id = handler.cv_image_ids[item.index]
    img_info = handler.cv_img_info[img_id]
    raw_dets = (pred or {}).get("detections", []) if pred is not None else []
    # Detections are untrusted: keep only well-formed dicts with a usable bbox so
    # neither COCOeval nor the display path below is ever handed garbage.
    detections = [
        d
        for d in (raw_dets if isinstance(raw_dets, (list, tuple)) else [])
        if isinstance(d, dict) and _valid_bbox(d.get("bbox"))
    ]
    accuracy = (
        score_cv_single(
            handler.cv_ann_info.get(img_id, []),
            detections,
            img_info,
            handler.cv_categories,
        )
        if pred is not None
        else 0.0
    )

    # Build display data
    orig_w = img_info.get("width") or 1
    orig_h = img_info.get("height") or 1
    thumb_scale = min(1.0, 200.0 / max(orig_w, orig_h, 1))

    def scale_box(b: list) -> list:
        return [
            round(b[0] * thumb_scale, 2),
            round(b[1] * thumb_scale, 2),
            round(b[2] * thumb_scale, 2),
            round(b[3] * thumb_scale, 2),
        ]

    gt_boxes = [scale_box(ann["bbox"]) for ann in handler.cv_ann_info.get(img_id, [])]
    pred_boxes = [scale_box(det["bbox"]) for det in detections]  # already validated
    box_result = _match_boxes_display(gt_boxes, pred_boxes)

    return ScoredItem(
        task_id=item.task_id,
        accuracy_score=accuracy,
        image_b64=_cv_thumbnail_b64(handler, img_id),
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        greens=box_result["greens"],
        reds=box_result["reds"],
        yellows=box_result["yellows"],
    )


def _normalize_nlp_pred(pred: Any) -> tuple[list, str] | None:
    """Validate one untrusted NLP prediction into (documents, answer).

    Returns None when the prediction is malformed and must be scored 0 WITHOUT
    being passed to the model. ``pred is None`` (no reply / timeout) is not
    malformed — it maps to an empty answer, scored by the evaluator's normal
    empty-string branch.
    """
    if pred is None:
        return [], ""
    if not isinstance(pred, dict):
        return None
    answer = pred.get("answer", "")
    if answer is None:
        answer = ""
    if not isinstance(answer, str):
        return None
    documents = pred.get("documents", [])
    if documents is None:
        documents = []
    if not isinstance(documents, (list, tuple)):
        return None
    return [str(d) for d in documents][:3], answer


def _score_nlp_items(
    handler: "TaskHandler",
    items: Sequence[BatchItem],
    pred_by_id: dict[Any, dict],
    has_predictions: bool,
) -> list[ScoredItem]:
    """Score all NLP items in one model call, returning a ScoredItem per item.

    Untrusted prediction fields are validated up front; a malformed prediction
    is scored 0.0 and excluded from the model call so it is never passed to the
    evaluator. The batched model call is itself guarded so that even an
    unexpected failure scores the affected items 0.0 rather than dropping the
    whole batch.
    """
    # task_id -> (documents, answer); malformed predictions are recorded so they
    # are scored 0 and kept out of the model batch.
    normalized: dict[Any, tuple[list, str]] = {}
    malformed_ids: set = set()
    if has_predictions:
        for item in items:
            norm = _normalize_nlp_pred(pred_by_id.get(item.task_id))
            if norm is None:
                malformed_ids.add(item.task_id)
                logger.warning(
                    "NLP item task_id=%r: malformed prediction; scoring 0.0",
                    item.task_id,
                )
            else:
                normalized[item.task_id] = norm

    # Build HF rows only for well-formed items
    rows: list[tuple] = []
    row_ids: list[Any] = []
    if has_predictions:
        for item in items:
            if item.task_id in malformed_ids:
                continue
            docs, answer = normalized.get(item.task_id, ([], ""))
            gt = handler.nlp_questions[item.index]
            rows.append(
                (
                    list(gt.get("source_docs", [])),
                    docs,
                    gt.get("question", ""),
                    gt.get("answer") or "",
                    answer,
                )
            )
            row_ids.append(item.task_id)

    nlp_accuracy_by_id: dict[Any, float] = {}
    if rows:
        try:
            assert handler.nlp_eval_model_path is not None, (
                "nlp_eval_model_path is required for NLP scoring"
            )
            evaluator = nlp_eval.get_evaluator(
                model_path=handler.nlp_eval_model_path,
                threshold=constants.NLP_EVAL_THRESHOLD,
                max_length=constants.NLP_EVAL_MAX_LENGTH,
            )
            for task_id, acc in zip(row_ids, score_nlp_rows(rows, evaluator)):
                nlp_accuracy_by_id[task_id] = acc
        except Exception:
            logger.exception(
                "NLP batch evaluation failed; scoring its %d items 0.0", len(rows)
            )
            # leave nlp_accuracy_by_id empty → these items default to 0.0 below

    result: list[ScoredItem] = []
    for item in items:
        accuracy = nlp_accuracy_by_id.get(item.task_id, 0.0)
        gt = handler.nlp_questions[item.index]
        source_docs = gt.get("source_docs", [])
        # Display from the validated values (empty for missing/malformed preds).
        pred_documents, hyp = normalized.get(item.task_id, ([], ""))
        chips = _doc_chips_display(source_docs, pred_documents)
        result.append(
            ScoredItem(
                task_id=item.task_id,
                accuracy_score=accuracy,
                hypothesis=hyp,
                question=gt.get("question", ""),
                ref=gt.get("answer") or None,
                difficulty=gt.get("difficulty", ""),
                source_docs=source_docs,
                pred_documents=pred_documents,
                is_equivalent=accuracy >= constants.NLP_EVAL_THRESHOLD,
                gt_doc_chips=chips["gt_doc_chips"],
                pred_doc_chips=chips["pred_doc_chips"],
                retrieval_correct=chips["retrieval_correct"],
            )
        )
    return result


# ── batch-level entry point ───────────────────────────────────────────────────


def _safe_score_item(
    scorer, handler: "TaskHandler", item: BatchItem, pred: dict | None, task: TaskType
) -> ScoredItem:
    """Score one item, never raising.

    Validation in the per-item scorers should keep malformed input away from
    jiwer/COCOeval, but this is the final safety net: any unexpected failure is
    logged and the item is scored 0.0 so the rest of the batch still scores.
    """
    try:
        return scorer(handler, item, pred)
    except Exception:
        logger.warning(
            "scoring %s item task_id=%r failed (malformed prediction?); scoring 0.0",
            task.value,
            getattr(item, "task_id", "?"),
            exc_info=True,
        )
        return ScoredItem(task_id=item.task_id, accuracy_score=0.0)


def score_batch(
    handler: "TaskHandler",
    batch: Batch,
    predictions: list[dict] | None,
    elapsed: float,
) -> ScoredBatch:
    """Score one batch, returning a frozen ScoredBatch.

    Args:
        handler:     TaskHandler for data access (asr_instances, cv_img_info, …).
        batch:       The Batch to score (task type + items + correlation IDs).
        predictions: Raw results array from team's WS reply, or None on timeout.
        elapsed:     Wall-clock seconds from send to reply.
    """
    task = batch.task
    items = batch.items
    # `predictions` is the untrusted team reply. Keep only well-formed dict
    # entries that carry a task_id; anything else is ignored (its batch item
    # then has no prediction and is scored 0) so a non-dict element can never
    # reach a scorer.
    pred_by_id: dict[Any, dict] = {
        p["task_id"]: p
        for p in (predictions or [])
        if isinstance(p, dict) and "task_id" in p
    }
    has_predictions = predictions is not None

    if not has_predictions:
        time_score = 0.0
    else:
        t_max = constants.MAX_TIME_PER_TEST_CASE
        time_score = 1.0 - min(elapsed, t_max) / t_max

    match task:
        case TaskType.ASR:
            scored_items = [
                _safe_score_item(
                    _score_asr_item,
                    handler,
                    item,
                    pred_by_id.get(item.task_id) if has_predictions else None,
                    task,
                )
                for item in items
            ]
        case TaskType.CV:
            scored_items = [
                _safe_score_item(
                    _score_cv_item,
                    handler,
                    item,
                    pred_by_id.get(item.task_id) if has_predictions else None,
                    task,
                )
                for item in items
            ]
        case TaskType.NLP:
            scored_items = _score_nlp_items(handler, items, pred_by_id, has_predictions)
        case _:
            scored_items = [
                ScoredItem(task_id=item.task_id, accuracy_score=0.0) for item in items
            ]

    batch_accuracy = (
        sum(it.accuracy_score for it in scored_items) / len(scored_items)
        if scored_items
        else 0.0
    )
    batch_score = (
        constants.PERFORMANCE_WEIGHT * batch_accuracy
        + constants.SPEED_WEIGHT * time_score
    )
    return ScoredBatch(
        task=task,
        batch_accuracy=batch_accuracy,
        time_score=time_score,
        batch_score=batch_score,
        items=tuple(scored_items),
    )


def accumulate_batch_result(
    prev_team_scores: list[float],
    scored_dict: dict,
    batch: Batch,
    elapsed: float,
    timed_out: bool,
    team_name: str,
) -> BatchAccumulation:
    """Pure accumulation step for one completed batch.

    Returns a BatchAccumulation with:
      new_team_scores: prev_team_scores + [batch_score] (new tuple, not in-place).
      mission_multiplier: mean of new_team_scores.
      record: dict to append to match_results.jsonl.

    scored_dict must be the result of ScoredBatch.to_dict() — callers compute
    it once and pass it here so the base64 CV thumbnail encoding isn't repeated.
    """
    batch_score = float(scored_dict["batch_score"])
    new_scores = (*prev_team_scores, batch_score)
    mission_multiplier = sum(new_scores) / len(new_scores)
    record = {
        "team": team_name,
        "batch_id": batch.batch_id,
        "mission_id": batch.mission_id,
        "mission_num": batch.mission_num,
        "task": scored_dict["task"],
        "batch_accuracy": scored_dict["batch_accuracy"],
        "time_score": scored_dict["time_score"],
        "batch_score": scored_dict["batch_score"],
        "per_item": scored_dict["per_item"],
        "elapsed_s": round(elapsed, 3),
        "timed_out": timed_out,
        "mission_avg_so_far": round(mission_multiplier, 4),
    }
    return BatchAccumulation(
        new_team_scores=new_scores,
        mission_multiplier=mission_multiplier,
        record=record,
    )
