from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from nlp_manager import NLPManager as BaseNLPManager
from nlp_manager import RetrievedChunk


class NLPManager(BaseNLPManager):
    """Use current retrieval but a small seq2seq answer model instead of SmolLM3.

    Train/export a small FLAN-T5 model to ``nlp/src/models/flan_t5_answer_model``
    or let this try ``google/flan-t5-small``. For Docker/offline evaluation, the
    model still has to be present in the image/cache.
    """

    def __init__(self) -> None:
        self.seq2seq_max_input_tokens = int(os.getenv("NLP_SEQ2SEQ_MAX_INPUT_TOKENS", "1024"))
        super().__init__()

    def _default_generator_model(self) -> str:
        local_model = Path(__file__).resolve().parents[1] / "models" / "flan_t5_answer_model"
        if local_model.exists():
            return str(local_model)
        return os.getenv("NLP_SEQ2SEQ_MODEL", "google/flan-t5-small")

    def _load_generator_model(self) -> None:
        if not self.generator_enabled:
            return

        torch = self._ensure_torch()
        if torch is None:
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.generator_tokenizer = AutoTokenizer.from_pretrained(
                self.generator_model_name,
                local_files_only=self.embedding_local_files_only,
            )
            self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.generator_model_name,
                local_files_only=self.embedding_local_files_only,
                **self._model_kwargs_for_device(self.device),
            )
            self.generator_model.to(self.device)
            self.generator_model.eval()
        except Exception:
            self.generator_tokenizer = None
            self.generator_model = None

    def _generator_prompt(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        allow_no_answer: bool = False,
    ) -> str:
        context = self._build_generator_context(retrieved)
        if not context:
            return ""
        no_answer_line = (
            f"If the evidence does not answer the question, output {self.generator_no_answer_token}.\n"
            if allow_no_answer
            else ""
        )
        return (
            "Answer the question using only the evidence. "
            "Return the shortest correct answer, with no explanation.\n"
            f"{no_answer_line}"
            f"Question: {question}\n"
            f"Evidence:\n{context}\n"
            "Answer:"
        )

    def _generate_answers(
        self,
        items: list[tuple[str, list[RetrievedChunk]]],
        allow_no_answer: bool = False,
    ) -> list[str]:
        if not items:
            return []
        if self.generator_model is None or self.generator_tokenizer is None:
            return [""] * len(items)

        answers = [""] * len(items)
        prompts = [
            (index, self._generator_prompt(question, retrieved, allow_no_answer=allow_no_answer))
            for index, (question, retrieved) in enumerate(items)
        ]
        prompts = [(index, prompt) for index, prompt in prompts if prompt]
        if not prompts:
            return answers

        try:
            for start in range(0, len(prompts), self.generator_batch_size):
                batch = prompts[start : start + self.generator_batch_size]
                batch_indices, batch_prompts = zip(*batch)
                inputs = self.generator_tokenizer(
                    list(batch_prompts),
                    padding=True,
                    truncation=True,
                    max_length=self.seq2seq_max_input_tokens,
                    return_tensors="pt",
                )
                inputs = self._to_device(inputs)
                inference_mode = getattr(self.torch, "inference_mode", self.torch.no_grad)
                with inference_mode():
                    output_ids = self.generator_model.generate(
                        **inputs,
                        max_new_tokens=self.generator_max_new_tokens,
                        do_sample=False,
                    )
                decoded = self.generator_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                for answer_index, answer in zip(batch_indices, decoded):
                    answers[answer_index] = self._clean_answer(answer)
            return answers
        except Exception:
            return [""] * len(items)

    def _generate_answer(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        allow_no_answer: bool = False,
    ) -> str:
        return self._generate_answers([(question, retrieved)], allow_no_answer=allow_no_answer)[0]

