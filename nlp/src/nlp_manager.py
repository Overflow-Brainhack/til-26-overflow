"""Manages the NLP RAG QA model."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextChunk:
    doc_id: int
    chunk_id: int
    text: str
    title: str


class NLPManager:
    _TOKEN_RE = re.compile(r"\d+(?:\.\d+)?%?|[a-z]+(?:[-'][a-z0-9]+)*", re.I)
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}|^[-*]\s+", re.M)

    def __init__(self):
        self.loaded = False
        self.chunks: list[TextChunk] = []
        self.chunk_terms: list[Counter[str]] = []
        self.chunk_lengths: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 1.0
        self.answer_cache: dict[str, str] = {}

        self.max_chunk_words = int(os.getenv("NLP_MAX_CHUNK_WORDS", "190"))
        self.chunk_overlap_words = int(os.getenv("NLP_CHUNK_OVERLAP_WORDS", "45"))
        self.top_k = int(os.getenv("NLP_TOP_K", "6"))
        self.max_context_chars = int(os.getenv("NLP_MAX_CONTEXT_CHARS", "5600"))

        default_model_dir = Path(__file__).resolve().parent / "models" / "nlp_answer_model"
        self.model_dir = Path(os.getenv("NLP_MODEL_DIR", str(default_model_dir)))
        self.tokenizer = None
        self.model = None
        self.torch = None
        self.device = "cpu"
        self._load_answer_model()

    def load_corpus(self, documents: list[str]) -> None:
        self.loaded = False
        self.answer_cache.clear()
        self.chunks = []
        self.chunk_terms = []
        self.chunk_lengths = []
        self.idf = {}

        for doc_id, document in enumerate(documents):
            self.chunks.extend(self._chunk_document(document, doc_id))

        self._build_bm25()
        self.loaded = bool(self.chunks)

    def qa(self, question: str) -> str:
        if not self.loaded:
            return ""

        question = question.strip()
        if not question:
            return ""
        if question in self.answer_cache:
            return self.answer_cache[question]

        retrieved = self._retrieve(question, self.top_k)
        if not retrieved:
            self.answer_cache[question] = ""
            return ""

        answer = ""
        if self.model is not None and self.tokenizer is not None:
            answer = self._generate_answer(question, retrieved)

        if not answer:
            answer = self._extractive_answer(question, retrieved)

        answer = self._clean_answer(answer)
        self.answer_cache[question] = answer
        return answer

    def _load_answer_model(self) -> None:
        if not self.model_dir.exists():
            return

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception:
            return

        try:
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir))
            self.model.to(self.device)
            self.model.eval()
        except Exception:
            self.tokenizer = None
            self.model = None
            self.torch = None
            self.device = "cpu"

    def _chunk_document(self, document: str, doc_id: int) -> list[TextChunk]:
        document = document.replace("\r\n", "\n").replace("\r", "\n")
        title = self._document_title(document, doc_id)

        paragraphs = [
            self._normalize_space(part)
            for part in re.split(r"\n\s*\n", document)
            if self._normalize_space(part)
        ]

        chunks: list[TextChunk] = []
        current_words: list[str] = []
        chunk_id = 0

        def flush() -> None:
            nonlocal chunk_id, current_words
            if not current_words:
                return
            body = " ".join(current_words)
            text = f"{title}\n{body}" if title and title not in body[:120] else body
            chunks.append(TextChunk(doc_id=doc_id, chunk_id=chunk_id, text=text, title=title))
            chunk_id += 1
            if self.chunk_overlap_words > 0:
                current_words = current_words[-self.chunk_overlap_words :]
            else:
                current_words = []

        for paragraph in paragraphs:
            words = paragraph.split()
            if len(words) > self.max_chunk_words:
                for start in range(0, len(words), self.max_chunk_words - self.chunk_overlap_words):
                    window = words[start : start + self.max_chunk_words]
                    if window:
                        text = f"{title}\n{' '.join(window)}"
                        chunks.append(
                            TextChunk(doc_id=doc_id, chunk_id=chunk_id, text=text, title=title)
                        )
                        chunk_id += 1
                current_words = []
                continue

            if current_words and len(current_words) + len(words) > self.max_chunk_words:
                flush()
            current_words.extend(words)

        flush()
        return chunks

    def _build_bm25(self) -> None:
        document_frequency: Counter[str] = Counter()

        for chunk in self.chunks:
            tokens = self._tokens(chunk.text)
            terms = Counter(tokens)
            self.chunk_terms.append(terms)
            self.chunk_lengths.append(len(tokens))
            document_frequency.update(terms.keys())

        if not self.chunks:
            self.avgdl = 1.0
            return

        n_chunks = len(self.chunks)
        self.avgdl = sum(self.chunk_lengths) / max(1, n_chunks)
        self.idf = {
            term: math.log(1.0 + (n_chunks - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def _retrieve(self, question: str, k: int) -> list[tuple[TextChunk, float]]:
        query_tokens = self._tokens(question)
        if not query_tokens:
            return []

        query_terms = Counter(query_tokens)
        k1 = 1.45
        b = 0.72
        scores: list[tuple[int, float]] = []

        for index, terms in enumerate(self.chunk_terms):
            score = 0.0
            length = self.chunk_lengths[index] or 1
            length_norm = k1 * (1.0 - b + b * length / self.avgdl)

            for term, query_count in query_terms.items():
                frequency = terms.get(term, 0)
                if not frequency:
                    continue
                tf = (frequency * (k1 + 1.0)) / (frequency + length_norm)
                score += self.idf.get(term, 0.0) * tf * (1.0 + 0.15 * (query_count - 1))

            if score > 0:
                scores.append((index, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [(self.chunks[index], score) for index, score in scores[:k]]

    def _generate_answer(self, question: str, retrieved: list[tuple[TextChunk, float]]) -> str:
        prompt = self._build_prompt(question, retrieved)
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with self.torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=96,
                    num_beams=4,
                    length_penalty=0.8,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return self._clean_answer(answer)
        except Exception:
            return ""

    def _build_prompt(self, question: str, retrieved: list[tuple[TextChunk, float]]) -> str:
        context_blocks: list[str] = []
        used_chars = 0

        for chunk, _score in retrieved:
            block = f"[DOC {chunk.doc_id + 1} / CHUNK {chunk.chunk_id}]\n{chunk.text.strip()}"
            if used_chars + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - used_chars
                if remaining <= 400:
                    break
                block = block[:remaining]
            context_blocks.append(block)
            used_chars += len(block)

        context = "\n\n".join(context_blocks)
        return (
            "Answer the question using only the context. "
            "Return a concise answer, not a full explanation. "
            "Every question is answerable in the corpus, so provide the best supported answer.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

    def _extractive_answer(self, question: str, retrieved: list[tuple[TextChunk, float]]) -> str:
        query_terms = set(self._tokens(question))
        candidates: list[tuple[float, str]] = []

        for chunk, chunk_score in retrieved[:4]:
            for sentence in self._SENTENCE_SPLIT_RE.split(chunk.text):
                sentence = self._normalize_space(sentence)
                if len(sentence) < 8 or sentence.startswith("#"):
                    continue
                sentence_terms = set(self._tokens(sentence))
                overlap = query_terms & sentence_terms
                if not overlap:
                    continue
                score = chunk_score + sum(self.idf.get(term, 0.0) for term in overlap)
                score -= min(len(sentence), 500) / 900.0
                candidates.append((score, sentence))

        if not candidates:
            return ""

        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][1]
        best = re.sub(r"^\*\*([^*]+)\*\*:\s*", "", best)
        best = re.sub(r"^[A-Z][A-Z0-9 ._/-]{2,}:\s*", "", best)
        if len(best) > 360:
            best = best[:360].rsplit(" ", 1)[0]
        return best

    def _clean_answer(self, answer: str) -> str:
        answer = self._normalize_space(answer)
        answer = re.sub(r"^(answer|candidate answer)\s*:\s*", "", answer, flags=re.I)
        if answer.lower() in {
            "not answerable",
            "unanswerable",
            "unknown",
            "not enough information",
            "no answer",
            "none",
            '""',
        }:
            return ""
        answer = answer.strip("\"' ")
        if len(answer) > 420:
            answer = answer[:420].rsplit(" ", 1)[0]
        return answer

    def _document_title(self, document: str, doc_id: int) -> str:
        for line in document.splitlines():
            line = self._normalize_space(line).strip("#* ")
            if line:
                return line[:180]
        return f"Document {doc_id + 1}"

    def _tokens(self, text: str) -> list[str]:
        return [match.group(0).lower() for match in self._TOKEN_RE.finditer(text)]

    def _normalize_space(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
