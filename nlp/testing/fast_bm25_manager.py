from __future__ import annotations

import heapq
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextChunk:
    doc_id: int
    chunk_id: int
    text: str
    title: str
    section: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: TextChunk
    score: float
    source: str


class NLPManager:
    """Speed-first drop-in NLP manager.

    This is deliberately simple: runtime corpus chunking, inverted-index BM25,
    top-3 document output, and an extractive sentence/window answer.
    """

    _TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?%?|[a-z0-9]+(?:[-'][a-z0-9]+)*", re.I)
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}|^[-*]\s+", re.M)
    _HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
    _NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
    _YEAR_RE = re.compile(r"\b(?:19|20|21)\d{2}\b")
    _NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|[A-Z]\.)){1,4}\b")

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "their",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "with",
    }

    _EXPANSION_GROUPS = {
        "amount": {
            "amount",
            "budget",
            "cost",
            "funding",
            "much",
            "paid",
            "payment",
            "price",
            "revenue",
            "spend",
            "spent",
            "value",
            "worth",
        },
        "count": {"count", "many", "number", "quantity", "total"},
        "date": {"date", "day", "deadline", "month", "schedule", "time", "timeline", "when", "year"},
        "person": {
            "accountable",
            "ceo",
            "chair",
            "chief",
            "commander",
            "director",
            "founder",
            "head",
            "leader",
            "led",
            "manager",
            "officer",
            "person",
            "who",
        },
        "place": {"city", "country", "district", "facility", "location", "place", "region", "site", "where", "zone"},
        "negation": {"absent", "missing", "not", "noted", "omitted", "stated", "undisclosed", "unspecified"},
    }

    _QUESTION_WORDS_TO_GROUPS = {
        "how many": {"count"},
        "how much": {"amount"},
        "when": {"date"},
        "where": {"place"},
        "who": {"person"},
        "whom": {"person"},
        "whose": {"person"},
    }

    def __init__(self) -> None:
        self.loaded = False
        self.chunks: list[TextChunk] = []
        self.chunk_terms: list[Counter[str]] = []
        self.chunk_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.avgdl = 1.0
        self.doc_ids: list[str] = []
        self.prediction_cache: dict[str, dict[str, Any]] = {}

        self.max_chunk_words = int(os.getenv("NLP_FAST_MAX_CHUNK_WORDS", "150"))
        self.chunk_overlap_words = int(os.getenv("NLP_FAST_CHUNK_OVERLAP_WORDS", "25"))
        self.output_doc_count = int(os.getenv("NLP_OUTPUT_DOC_COUNT", "3"))
        self.retrieval_top_k = int(os.getenv("NLP_FAST_RETRIEVAL_TOP_K", "24"))
        self.answer_chunk_limit = int(os.getenv("NLP_FAST_ANSWER_CHUNK_LIMIT", "6"))
        self.answer_style = os.getenv("NLP_FAST_ANSWER_STYLE", "sentence").strip().lower()
        self.k1 = float(os.getenv("NLP_FAST_BM25_K1", "1.45"))
        self.b = float(os.getenv("NLP_FAST_BM25_B", "0.72"))

    def load_corpus(self, documents: list[Any]) -> None:
        self.loaded = False
        self.chunks = []
        self.chunk_terms = []
        self.chunk_lengths = []
        self.postings = defaultdict(list)
        self.idf = {}
        self.doc_ids = []
        self.prediction_cache.clear()

        for doc_index, document in enumerate(documents):
            doc_id, text = self._coerce_document(document, doc_index)
            self.doc_ids.append(doc_id)
            self.chunks.extend(self._chunk_document(text, doc_index))

        self._build_index()
        self.loaded = bool(self.chunks)

    def qa(self, question: str) -> dict[str, Any]:
        return self.qa_batch([question])[0]

    def qa_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        predictions = [self._empty_prediction() for _question in questions]
        if not self.loaded:
            return predictions

        for index, raw_question in enumerate(questions):
            question = str(raw_question).strip()
            if not question:
                continue
            cached = self.prediction_cache.get(question)
            if cached is not None:
                predictions[index] = self._copy_prediction(cached)
                continue

            retrieved = self._retrieve(question, self.retrieval_top_k)
            doc_ids = self._top_document_ids(retrieved)
            answer = self._extractive_answer(question, retrieved)
            prediction = self._prediction(doc_ids, answer)
            self.prediction_cache[question] = prediction
            predictions[index] = self._copy_prediction(prediction)

        return predictions

    def qa_result_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        return self.qa_batch(questions)

    def _coerce_document(self, document: Any, doc_index: int) -> tuple[str, str]:
        if isinstance(document, dict):
            doc_id = (
                document.get("id")
                or document.get("doc_id")
                or document.get("document_id")
                or f"DOC-{doc_index + 1:04d}"
            )
            text = (
                document.get("document")
                or document.get("text")
                or document.get("content")
                or ""
            )
            return str(doc_id), str(text)
        return f"DOC-{doc_index + 1:04d}", str(document)

    def _chunk_document(self, document: str, doc_id: int) -> list[TextChunk]:
        document = document.replace("\r\n", "\n").replace("\r", "\n")
        title = self._document_title(document, doc_id)
        chunks: list[TextChunk] = []
        chunk_id = 0

        for section_title, section_text in self._document_sections(document, title):
            paragraphs = [
                self._normalize_space(part)
                for part in re.split(r"\n\s*\n", section_text)
                if self._normalize_space(part)
            ]
            current_words: list[str] = []

            def flush() -> None:
                nonlocal chunk_id, current_words
                if not current_words:
                    return
                body = " ".join(current_words)
                chunks.append(
                    TextChunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=self._chunk_text_with_heading(title, section_title, body),
                        title=title,
                        section=section_title,
                    )
                )
                chunk_id += 1
                current_words = current_words[-self.chunk_overlap_words :] if self.chunk_overlap_words > 0 else []

            stride = max(1, self.max_chunk_words - self.chunk_overlap_words)
            for paragraph in paragraphs:
                words = paragraph.split()
                if len(words) > self.max_chunk_words:
                    flush()
                    for start in range(0, len(words), stride):
                        window = words[start : start + self.max_chunk_words]
                        if not window:
                            continue
                        chunks.append(
                            TextChunk(
                                doc_id=doc_id,
                                chunk_id=chunk_id,
                                text=self._chunk_text_with_heading(title, section_title, " ".join(window)),
                                title=title,
                                section=section_title,
                            )
                        )
                        chunk_id += 1
                    current_words = []
                    continue
                if current_words and len(current_words) + len(words) > self.max_chunk_words:
                    flush()
                current_words.extend(words)

            flush()

        return chunks

    def _document_sections(self, document: str, title: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, list[str]]] = []
        current_title = title
        current_lines: list[str] = []
        for line in document.splitlines():
            match = self._HEADER_RE.match(line)
            if match:
                if current_lines:
                    sections.append((current_title, current_lines))
                    current_lines = []
                current_title = self._normalize_space(match.group(1).strip("#* "))
                continue
            current_lines.append(line)
        if current_lines:
            sections.append((current_title, current_lines))
        return [
            (section_title, "\n".join(lines))
            for section_title, lines in sections
            if self._normalize_space("\n".join(lines))
        ] or [(title, document)]

    def _build_index(self) -> None:
        document_frequency: Counter[str] = Counter()
        for index, chunk in enumerate(self.chunks):
            terms = Counter(self._tokens(chunk.text))
            self.chunk_terms.append(terms)
            self.chunk_lengths.append(sum(terms.values()))
            document_frequency.update(terms.keys())
            for term, frequency in terms.items():
                self.postings[term].append((index, frequency))

        n_chunks = len(self.chunks)
        if not n_chunks:
            self.avgdl = 1.0
            return
        self.avgdl = sum(self.chunk_lengths) / max(1, n_chunks)
        self.idf = {
            term: math.log(1.0 + (n_chunks - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def _retrieve(self, question: str, k: int) -> list[RetrievedChunk]:
        query_weights = self._query_term_weights(question)
        if not query_weights:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term, weight in query_weights.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, frequency in self.postings.get(term, []):
                length = self.chunk_lengths[index] or 1
                length_norm = self.k1 * (1.0 - self.b + self.b * length / self.avgdl)
                tf = (frequency * (self.k1 + 1.0)) / (frequency + length_norm)
                scores[index] += idf * tf * weight

        if not scores:
            return []

        # Tiny phrase/title boosts help synthetic direct questions without model calls.
        content_terms = [term for term in self._tokens(question) if term not in self._STOPWORDS]
        phrase = " ".join(content_terms[:5])
        for index in list(scores.keys()):
            chunk = self.chunks[index]
            lower_text = chunk.text.lower()
            if phrase and phrase in lower_text:
                scores[index] += 2.0
            if any(term in chunk.title.lower() for term in content_terms[:4]):
                scores[index] += 0.5

        top = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
        return [RetrievedChunk(self.chunks[index], score, "bm25-inverted") for index, score in top if score > 0]

    def _top_document_ids(self, retrieved: list[RetrievedChunk]) -> list[str]:
        doc_ids: list[str] = []
        seen: set[str] = set()
        for item in retrieved:
            doc_id = self._doc_id_for_index(item.chunk.doc_id)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc_ids.append(doc_id)
            if len(doc_ids) >= self.output_doc_count:
                break
        return doc_ids

    def _extractive_answer(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        query_terms = set(self._tokens(question))
        content_terms = query_terms - self._STOPWORDS
        cue_groups = self._question_cue_groups(question)
        cue_terms = self._cue_terms(cue_groups)
        candidates: list[tuple[float, str]] = []

        for retrieved_item in retrieved[: self.answer_chunk_limit]:
            sentences = [
                self._normalize_space(sentence)
                for sentence in self._SENTENCE_SPLIT_RE.split(retrieved_item.chunk.text)
            ]
            sentences = [sentence for sentence in sentences if len(sentence) >= 8 and not sentence.startswith("#")]
            for sentence_index, sentence in enumerate(sentences):
                windows = [sentence]
                if sentence_index > 0:
                    windows.append(f"{sentences[sentence_index - 1]} {sentence}")
                if sentence_index + 1 < len(sentences):
                    windows.append(f"{sentence} {sentences[sentence_index + 1]}")
                for candidate in windows:
                    self._score_answer_candidate(
                        candidate,
                        retrieved_item.score,
                        query_terms,
                        content_terms,
                        cue_terms,
                        cue_groups,
                        candidates,
                    )

        if not candidates:
            return self._first_reasonable_sentence(retrieved)

        candidates.sort(key=lambda item: item[0], reverse=True)
        answer = candidates[0][1]
        if self.answer_style in {"short", "phrase"}:
            answer = self._shorten_answer(question, answer) or answer
        return self._clean_answer(answer)

    def _score_answer_candidate(
        self,
        candidate: str,
        chunk_score: float,
        query_terms: set[str],
        content_terms: set[str],
        cue_terms: set[str],
        cue_groups: set[str],
        candidates: list[tuple[float, str]],
    ) -> None:
        sentence = self._normalize_space(candidate)
        if len(sentence) < 8:
            return
        sentence_terms = set(self._tokens(sentence))
        overlap = query_terms & sentence_terms
        content_overlap = content_terms & sentence_terms
        cue_overlap = cue_terms & sentence_terms
        if not overlap and not cue_overlap:
            return

        score = chunk_score
        score += sum(self.idf.get(term, 0.0) for term in overlap)
        score += 0.9 * sum(self.idf.get(term, 0.0) for term in content_overlap)
        score += 1.8 * len(cue_overlap)
        if cue_groups & {"amount", "count"} and self._NUMBER_RE.search(sentence):
            score += 2.2
        if "date" in cue_groups and self._YEAR_RE.search(sentence):
            score += 2.0
        if "person" in cue_groups and self._NAME_RE.search(sentence):
            score += 1.8
        if "place" in cue_groups and re.search(r"\b[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3}\b", sentence):
            score += 0.8
        score -= min(len(sentence), 700) / 1400.0
        candidates.append((score, sentence))

    def _shorten_answer(self, question: str, answer: str) -> str:
        lower_question = question.lower()
        if "how many" in lower_question or "how much" in lower_question:
            match = self._NUMBER_RE.search(answer)
            if match:
                return match.group(0)
        if "when" in lower_question:
            match = re.search(
                r"\b(?:on|in|by|before|after|during)\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|[A-Z][a-z]+\s+\d{4}|\d{4})\b",
                answer,
            )
            if match:
                return match.group(1)
            match = self._YEAR_RE.search(answer)
            if match:
                return match.group(0)
        if re.search(r"\bwho|whom|whose\b", lower_question):
            match = self._NAME_RE.search(answer)
            if match:
                return match.group(0)
        return ""

    def _prediction(self, documents: list[str], answer: str) -> dict[str, Any]:
        return {"documents": documents[: self.output_doc_count], "answer": self._clean_answer(answer)}

    def _empty_prediction(self) -> dict[str, Any]:
        return {"documents": [], "answer": ""}

    def _copy_prediction(self, prediction: dict[str, Any]) -> dict[str, Any]:
        return {
            "documents": [str(doc_id) for doc_id in prediction.get("documents", [])],
            "answer": str(prediction.get("answer", "")),
        }

    def _query_term_weights(self, question: str) -> dict[str, float]:
        original_terms = Counter(self._tokens(question))
        expanded_terms = Counter(self._tokens(self._expanded_keyword_query(question)))
        weights: dict[str, float] = {}
        for term, count in original_terms.items():
            weights[term] = (0.35 if term in self._STOPWORDS else 1.0) * count
        for term, count in expanded_terms.items():
            if term not in original_terms:
                weights[term] = max(weights.get(term, 0.0), 0.45 * count)
        return weights

    def _expanded_keyword_query(self, question: str) -> str:
        terms = [question]
        for group in self._question_cue_groups(question):
            terms.extend(sorted(self._EXPANSION_GROUPS[group]))
        for token in self._tokens(question):
            if "-" in token or "'" in token:
                terms.extend(re.split(r"[-']", token))
                terms.append(token.replace("-", " ").replace("'", ""))
        content_terms = [token for token in self._tokens(question) if token not in self._STOPWORDS]
        if content_terms:
            terms.append(" ".join(content_terms))
        return " ".join(term for term in terms if term)

    def _question_cue_groups(self, question: str) -> set[str]:
        lower_question = question.lower()
        query_terms = set(self._tokens(question))
        groups: set[str] = set()
        for phrase, phrase_groups in self._QUESTION_WORDS_TO_GROUPS.items():
            if phrase in lower_question:
                groups.update(phrase_groups)
        for group_name, group_terms in self._EXPANSION_GROUPS.items():
            if query_terms & group_terms:
                groups.add(group_name)
        if re.search(r"\b(no|not|never|without|missing|unstated|unmentioned)\b", lower_question):
            groups.add("negation")
        return groups

    def _cue_terms(self, cue_groups: set[str]) -> set[str]:
        terms: set[str] = set()
        for group in cue_groups:
            terms.update(self._EXPANSION_GROUPS.get(group, set()))
            terms.add(group)
        return terms

    def _clean_answer(self, answer: str) -> str:
        answer = self._normalize_space(answer)
        answer = "".join(character for character in answer if character.isprintable())
        answer = re.sub(r"^(answer|candidate answer|final answer)\s*:\s*", "", answer, flags=re.I)
        answer = re.sub(r"^(the answer is|it is)\s+", "", answer, flags=re.I)
        answer = re.sub(r"^\*\*([^*]+)\*\*:\s*", "", answer)
        answer = re.sub(r"^[A-Z][A-Z0-9 ._/-]{2,}:\s*", "", answer)
        answer = re.sub(r"^[-*\s]+", "", answer).strip("\"' ")
        tokens = answer.split()
        if len(tokens) > 64:
            answer = " ".join(tokens[:64])
        if len(answer) > 420:
            answer = answer[:420].rsplit(" ", 1)[0]
        return answer

    def _first_reasonable_sentence(self, retrieved: list[RetrievedChunk]) -> str:
        for item in retrieved:
            for sentence in self._SENTENCE_SPLIT_RE.split(item.chunk.text):
                sentence = self._normalize_space(sentence)
                if len(sentence) >= 8 and not sentence.startswith("#"):
                    return self._clean_answer(sentence)
        return ""

    def _document_title(self, document: str, doc_id: int) -> str:
        for line in document.splitlines():
            line = self._normalize_space(line).strip("#* ")
            if line:
                return line[:180]
        return f"Document {doc_id + 1}"

    def _chunk_text_with_heading(self, title: str, section: str, body: str) -> str:
        heading_parts = []
        if title:
            heading_parts.append(title)
        if section and section != title:
            heading_parts.append(section)
        heading = " - ".join(heading_parts)
        if heading and heading not in body[:160]:
            return f"{heading}\n{body}"
        return body

    def _doc_id_for_index(self, doc_index: int) -> str:
        if 0 <= doc_index < len(self.doc_ids):
            return self.doc_ids[doc_index]
        return f"DOC-{doc_index + 1:04d}"

    def _tokens(self, text: str) -> list[str]:
        return [match.group(0).lower() for match in self._TOKEN_RE.finditer(text)]

    def _normalize_space(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text)).strip()

