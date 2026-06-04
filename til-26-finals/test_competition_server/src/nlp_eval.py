"""Answer-Equivalence Evaluator used to score per-mission RAG QA results.

Adapted from `test/test_nlp.py`. Lazy-loaded by the competition server so the
weights only hit GPU memory on the first NLP eval, not at boot.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from string import printable
from typing import Sequence, cast

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)

logger = logging.getLogger("uvicorn.error")

RETRIEVAL_ONLY_SCORE = 0.4
MAX_CANDIDATE_TOKEN_LENGTH = 64


@dataclass
class AEResult:
    index: int
    score: float
    equivalent: bool
    prob_equivalent: float


class AnswerEquivalenceEvaluator:
    """
    Wraps a fine-tuned encoder for answer-equivalence inference.

    Each input row is (source_docs, pred_documents, question, reference_answer,
    candidate_answer). Returns one ``AEResult`` per row, in input order.

    Scoring rules (mirrors `test/test_nlp.py`):
    * all-empty (no docs on either side, both answers blank) -> 1.0
    * one answer blank but at least one retrieved doc overlaps -> 1.0 if
      reference is also blank, else RETRIEVAL_ONLY_SCORE (0.4)
    * no doc overlap -> 0.0 (retrieval failure)
    * otherwise -> model probability of equivalence; score = 1.0 above
      threshold, RETRIEVAL_ONLY_SCORE below.
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.5,
        device: str | None = None,
        max_length: int = 128,
    ):
        self.threshold = threshold
        self.max_length = max_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading NLP eval model from {model_path} on {self.device}")
        self.tokenizer: PreTrainedTokenizer = cast(
            "PreTrainedTokenizer",
            AutoTokenizer.from_pretrained(str(model_path)),
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        ).to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"NLP eval model loaded: {n_params:,} parameters")

    def _format_input(self, question: str, reference: str, candidate: str) -> str:
        _printable = "".join(filter(lambda x: x in printable, candidate))
        tokens = self.tokenizer.tokenize(
            _printable,
            max_length=MAX_CANDIDATE_TOKEN_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )
        reconstructed_candidate = self.tokenizer.convert_tokens_to_string(tokens)
        return (
            f"Question: {question} "
            f"Reference: {reference} "
            f"Candidate: {reconstructed_candidate}"
        )

    @torch.no_grad()
    def batch_evaluate(
        self,
        data: list[tuple[Sequence[str], Sequence[str], str, str, str]],
        batch_size: int = 64,
    ) -> list[AEResult]:
        empty_str_results: list[AEResult] = []
        non_empty_indexed_triples: list[tuple[int, str, str, str]] = []

        for i, (docs, pred_docs, q, r, c) in enumerate(data):
            overlap_docs = len(set(docs).intersection(set(pred_docs))) >= 1
            if len(docs) == 0 and len(pred_docs) == 0 and r == "" and c == "":
                empty_str_results.append(
                    AEResult(index=i, score=1.0, equivalent=True, prob_equivalent=1.0)
                )
            elif (r == "" or c == "") and overlap_docs:
                _equivalent = r == c
                empty_str_results.append(
                    AEResult(
                        index=i,
                        score=1.0 if _equivalent else RETRIEVAL_ONLY_SCORE,
                        equivalent=_equivalent,
                        prob_equivalent=1.0 if _equivalent else 0.0,
                    )
                )
            elif overlap_docs:
                non_empty_indexed_triples.append((i, q, r, c))
            else:
                empty_str_results.append(
                    AEResult(index=i, score=0.0, equivalent=False, prob_equivalent=0.0)
                )

        texts = [
            (i, self._format_input(q, r, c)) for i, q, r, c in non_empty_indexed_triples
        ]
        all_results: list[AEResult] = []

        for i in range(0, len(texts), batch_size):
            batch_indices, batch_texts = zip(*texts[i : i + batch_size])
            encoding = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)

            logits = self.model(**encoding).logits
            probs = F.softmax(logits, dim=-1)

            for prob_idx, prob in enumerate(probs):
                prob_eq = prob[1].item()
                _equivalent = prob_eq >= self.threshold
                all_results.append(
                    AEResult(
                        index=batch_indices[prob_idx],
                        score=1.0 if _equivalent else RETRIEVAL_ONLY_SCORE,
                        equivalent=_equivalent,
                        prob_equivalent=prob_eq,
                    )
                )

        all_results.extend(empty_str_results)
        all_results.sort(key=lambda r: r.index)
        return all_results


_evaluator: AnswerEquivalenceEvaluator | None = None
_evaluator_lock = threading.Lock()


def get_evaluator(
    model_path: str | Path, threshold: float, max_length: int
) -> AnswerEquivalenceEvaluator:
    """Lazy-loaded module-level singleton. Thread-safe: concurrent callers block
    on the lock while the first caller loads the model."""
    global _evaluator
    if _evaluator is not None:
        return _evaluator
    with _evaluator_lock:
        if _evaluator is None:
            _evaluator = AnswerEquivalenceEvaluator(
                model_path=model_path, threshold=threshold, max_length=max_length
            )
    return _evaluator


def ensure_loaded(model_path: str | Path, threshold: float, max_length: int) -> None:
    """Force-load the AE evaluator if not already loaded. Idempotent.

    Callers wanting to pre-warm from an async context should wrap this in
    ``asyncio.to_thread`` so the synchronous model load doesn't block the
    event loop.
    """
    get_evaluator(model_path=model_path, threshold=threshold, max_length=max_length)
