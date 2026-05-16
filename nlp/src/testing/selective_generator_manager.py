from __future__ import annotations

import os
import re
from typing import Any

from nlp_manager import NLPManager as BaseNLPManager
from nlp_manager import RetrievedChunk


class NLPManager(BaseNLPManager):
    """Current pipeline, but only use the generator for hard-looking cases.

    This is meant to test whether most L1 questions can be answered by the
    existing extractive fallback while saving SmolLM3 for questions that are
    more likely to need synthesis/arithmetic.
    """

    _HARD_QUESTION_RE = re.compile(
        r"\b("
        r"add|after|average|before|between|calculate|compare|combined|difference|"
        r"earlier|except|greater|higher|how did|how does|infer|later|less|lower|"
        r"minus|more|not|ratio|same|similar|sum|total|unstated|why"
        r")\b",
        re.I,
    )

    def __init__(self) -> None:
        self.selective_generator_mode = os.getenv("NLP_SELECTIVE_GENERATOR_MODE", "hard_only").strip().lower()
        self.selective_max_groups = max(1, int(os.getenv("NLP_SELECTIVE_MAX_GROUPS", "1")))
        self.selective_easy_answer_words = int(os.getenv("NLP_SELECTIVE_EASY_ANSWER_WORDS", "42"))
        super().__init__()

    def qa_result_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        predictions = [self._empty_prediction() for _question in questions]
        if not self.loaded:
            return predictions

        pending: list[tuple[int, str]] = []
        for index, question in enumerate(questions):
            question = question.strip()
            if not question:
                continue
            if question in self.prediction_cache:
                predictions[index] = self._copy_prediction(self.prediction_cache[question])
            else:
                pending.append((index, question))

        expanded_queries = [self._expanded_dense_query(question) for _index, question in pending]
        query_embeddings = self._encode_queries(expanded_queries)

        work_items: list[dict[str, Any]] = []
        generation_inputs: list[tuple[str, list[RetrievedChunk]]] = []
        generation_owners: list[tuple[int, int]] = []

        for pending_index, (answer_index, question) in enumerate(pending):
            query_embedding = self._embedding_at(query_embeddings, pending_index)
            retrieval_k = max(self.top_k, self.rerank_top_k, self.output_doc_count * 8)
            retrieved = self._retrieve(question, retrieval_k, query_embedding)
            doc_ids = self._top_document_ids(retrieved)
            if not retrieved:
                prediction = self._prediction(doc_ids, "")
                self.prediction_cache[question] = prediction
                self.answer_cache[question] = ""
                continue

            candidate_groups = self._candidate_document_groups(retrieved)
            fallback_answers = [
                self._extractive_answer(question, candidate_group)
                for candidate_group in candidate_groups
            ]
            work_index = len(work_items)
            work_items.append(
                {
                    "answer_index": answer_index,
                    "question": question,
                    "doc_ids": doc_ids,
                    "candidate_groups": candidate_groups,
                    "fallback_answers": fallback_answers,
                    "generated_answers": [],
                }
            )

            if self._should_generate(question, retrieved, fallback_answers):
                for candidate_index, candidate_group in enumerate(candidate_groups[: self.selective_max_groups]):
                    generation_owners.append((work_index, candidate_index))
                    generation_inputs.append((question, candidate_group))

        generated_answers = self._generate_answers(generation_inputs, allow_no_answer=True)
        for (work_index, _candidate_index), generated_answer in zip(generation_owners, generated_answers):
            work_items[work_index]["generated_answers"].append(generated_answer)

        for item in work_items:
            answer = self._select_candidate_answer(item["generated_answers"], item["fallback_answers"])
            prediction = self._prediction(item["doc_ids"], answer)
            question = item["question"]
            self.prediction_cache[question] = prediction
            self.answer_cache[question] = answer
            predictions[item["answer_index"]] = self._copy_prediction(prediction)

        return predictions

    def _should_generate(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        fallback_answers: list[str],
    ) -> bool:
        if self.generator_model is None or self.generator_tokenizer is None:
            return False
        if self.selective_generator_mode in {"0", "false", "off", "never", "extractive"}:
            return False
        if self.selective_generator_mode in {"1", "true", "on", "always"}:
            return True

        fallback = next((answer for answer in fallback_answers if answer), "")
        if not fallback:
            return True

        if self._HARD_QUESTION_RE.search(question):
            return True

        # Direct L1-looking questions are usually fine with the extracted sentence.
        lower = question.lower()
        direct_question = re.search(r"\b(who|what|when|where|which)\b", lower) is not None
        short_enough = len(fallback.split()) <= self.selective_easy_answer_words
        has_query_overlap = bool(set(self._tokens(question)) & set(self._tokens(fallback)))
        if direct_question and short_enough and has_query_overlap:
            return False

        # If the top chunks are weak or answer text is long, spend the generator.
        top_score = retrieved[0].score if retrieved else 0.0
        if top_score < float(os.getenv("NLP_SELECTIVE_LOW_SCORE", "0.35")):
            return True
        if len(fallback.split()) > self.selective_easy_answer_words:
            return True

        return False

