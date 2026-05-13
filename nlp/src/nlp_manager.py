#uses harrier model + bm25

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    doc_id: int
    chunk_id: int
    text: str
    title: str


class NLPManager:
    _TOKEN_RE = re.compile(r"\d+(?:\.\d+)?%?|[a-z]+(?:[-'][a-z0-9]+)*", re.I)
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}|^[-*]\s+", re.M)
    _NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
    _YEAR_RE = re.compile(r"\b(?:19|20|21)\d{2}\b")
    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
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
    _QUESTION_CUE_GROUPS = {
        "amount": {
            "amount",
            "budget",
            "cost",
            "costs",
            "funding",
            "much",
            "paid",
            "price",
            "revenue",
            "spend",
            "spent",
            "value",
            "valued",
            "worth",
        },
        "count": {"count", "many", "number", "total"},
        "date": {"date", "day", "month", "time", "when", "year"},
        "person": {"ceo", "chair", "chief", "director", "founder", "head", "leader", "led", "who"},
        "place": {"city", "country", "district", "facility", "location", "site", "where"},
    }

    def __init__(self):
        self.loaded = False
        self.chunks: list[TextChunk] = []
        self.chunk_terms: list[Counter[str]] = []
        self.chunk_lengths: list[int] = []
        self.chunk_embeddings = None
        self.idf: dict[str, float] = {}
        self.avgdl = 1.0
        self.answer_cache: dict[str, str] = {}

        self.max_chunk_words = int(os.getenv("NLP_MAX_CHUNK_WORDS", "190"))
        self.chunk_overlap_words = int(os.getenv("NLP_CHUNK_OVERLAP_WORDS", "45"))
        self.top_k = int(os.getenv("NLP_TOP_K", "10"))
        self.bm25_top_k = int(os.getenv("NLP_BM25_TOP_K", "12"))
        self.dense_top_k = int(os.getenv("NLP_DENSE_TOP_K", "12"))
        self.sentence_chunk_limit = int(os.getenv("NLP_SENTENCE_CHUNK_LIMIT", "8"))
        self.rrf_k = int(os.getenv("NLP_RRF_K", "60"))
        self.embedding_batch_size = int(os.getenv("NLP_EMBEDDING_BATCH_SIZE", "16"))
        self.embedding_max_length = int(os.getenv("NLP_EMBEDDING_MAX_LENGTH", "512"))
        self.embedding_local_files_only = os.getenv("NLP_EMBEDDING_LOCAL_FILES_ONLY", "1") != "0"
        self.embedding_model_name = os.getenv(
            "NLP_EMBEDDING_MODEL",
            "microsoft/harrier-oss-v1-270m",
        )
        self.embedding_query_prompt = os.getenv(
            "NLP_EMBEDDING_QUERY_PROMPT",
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        )

        self.embedding_tokenizer = None
        self.embedding_model = None
        self.torch = None
        self.device = "cpu"
        self._load_embedding_model()

    def load_corpus(self, documents: list[str]) -> None:
        """Loads and indexes the corpus documents for RAG QA."""
        self.loaded = False
        self.answer_cache.clear()
        self.chunks = []
        self.chunk_terms = []
        self.chunk_lengths = []
        self.chunk_embeddings = None
        self.idf = {}

        for doc_id, document in enumerate(documents):
            self.chunks.extend(self._chunk_document(document, doc_id))

        self._build_bm25()
        self._build_embeddings()
        self.loaded = bool(self.chunks)

    def qa(self, question: str) -> str:
        return self.qa_batch([question])[0]

    def qa_batch(self, questions: list[str]) -> list[str]:
        answers = [""] * len(questions)
        if not self.loaded:
            return answers

        pending: list[tuple[int, str]] = []
        for index, question in enumerate(questions):
            question = question.strip()
            if not question:
                continue
            if question in self.answer_cache:
                answers[index] = self.answer_cache[question]
            else:
                pending.append((index, question))

        query_embeddings = self._encode_queries([question for _index, question in pending])

        for pending_index, (answer_index, question) in enumerate(pending):
            query_embedding = self._embedding_at(query_embeddings, pending_index)
            retrieved = self._retrieve(question, self.top_k, query_embedding)
            if not retrieved:
                self.answer_cache[question] = ""
                continue

            answer = self._extractive_answer(question, retrieved)
            answer = self._clean_answer(answer)
            self.answer_cache[question] = answer
            answers[answer_index] = answer

        return answers

    def _load_embedding_model(self) -> None:
        #harrier embedding model
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception:
            return

        try:
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {}
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )

            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name,
                local_files_only=self.embedding_local_files_only,
            )
            if self.embedding_tokenizer.pad_token is None:
                self.embedding_tokenizer.pad_token = (
                    self.embedding_tokenizer.eos_token or self.embedding_tokenizer.unk_token
                )

            self.embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name,
                local_files_only=self.embedding_local_files_only,
                **model_kwargs,
            )
            self.embedding_model.to(self.device)
            self.embedding_model.eval()
        except Exception:
            self.embedding_tokenizer = None
            self.embedding_model = None
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

    def _build_embeddings(self) -> None:
        if self.embedding_model is None or not self.chunks:
            return

        embeddings = self._encode_texts([chunk.text for chunk in self.chunks], is_query=False)
        if embeddings is None:
            self.chunk_embeddings = None
            return

        self.chunk_embeddings = embeddings

    def _retrieve(
        self,
        question: str,
        k: int,
        query_embedding=None,
    ) -> list[tuple[TextChunk, float]]:
        bm25_results = self._retrieve_bm25_scores(question, self.bm25_top_k)
        dense_results = self._retrieve_dense_scores(question, self.dense_top_k, query_embedding)

        if not dense_results:
            return [(self.chunks[index], score) for index, score in bm25_results[:k]]

        fused_scores: dict[int, float] = {}
        for rank, (index, _score) in enumerate(bm25_results, start=1):
            fused_scores[index] = fused_scores.get(index, 0.0) + 1.15 / (self.rrf_k + rank)
        for rank, (index, _score) in enumerate(dense_results, start=1):
            fused_scores[index] = fused_scores.get(index, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [(self.chunks[index], score * 100.0) for index, score in ranked[:k]]

    def _retrieve_dense_scores(
        self,
        question: str,
        k: int,
        query_embedding=None,
    ) -> list[tuple[int, float]]:
        if self.embedding_model is None or self.chunk_embeddings is None:
            return []

        if query_embedding is None:
            query_embeddings = self._encode_queries([question])
            query_embedding = self._embedding_at(query_embeddings, 0)
            if query_embedding is None:
                return []

        scores = self._dense_scores(query_embedding)
        scores.sort(key=lambda item: item[1], reverse=True)
        return [(index, score) for index, score in scores[:k] if score > 0]

    def _retrieve_bm25_scores(self, question: str, k: int) -> list[tuple[int, float]]:
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
        return scores[:k]

    def _extractive_answer(self, question: str, retrieved: list[tuple[TextChunk, float]]) -> str:
        query_terms = set(self._tokens(question))
        content_terms = query_terms - self._STOPWORDS
        cue_terms = self._question_cue_terms(query_terms)
        candidates: list[tuple[float, str]] = []

        for chunk, chunk_score in retrieved[: self.sentence_chunk_limit]:
            sentences = [
                self._normalize_space(sentence)
                for sentence in self._SENTENCE_SPLIT_RE.split(chunk.text)
            ]
            sentences = [
                sentence for sentence in sentences if len(sentence) >= 8 and not sentence.startswith("#")
            ]

            for index, sentence in enumerate(sentences):
                windows = [sentence]
                if index > 0:
                    windows.append(f"{sentences[index - 1]} {sentence}")
                if index + 1 < len(sentences):
                    windows.append(f"{sentence} {sentences[index + 1]}")

                for candidate in windows:
                    self._score_candidate(
                        candidate,
                        chunk_score,
                        query_terms,
                        content_terms,
                        cue_terms,
                        candidates,
                    )

        if not candidates:
            return self._first_reasonable_sentence(retrieved)

        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][1]
        best = re.sub(r"^\*\*([^*]+)\*\*:\s*", "", best)
        best = re.sub(r"^[A-Z][A-Z0-9 ._/-]{2,}:\s*", "", best)
        if len(best) > 360:
            best = best[:360].rsplit(" ", 1)[0]
        return best

    def _score_candidate(
        self,
        candidate: str,
        chunk_score: float,
        query_terms: set[str],
        content_terms: set[str],
        cue_terms: set[str],
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

        score = chunk_score + sum(self.idf.get(term, 0.0) for term in overlap)
        score += 0.8 * sum(self.idf.get(term, 0.0) for term in content_overlap)
        score += 2.5 * len(cue_overlap)

        if cue_terms and self._NUMBER_RE.search(sentence):
            score += 1.25
        if "date" in cue_terms and self._YEAR_RE.search(sentence):
            score += 1.5
        if "person" in cue_terms and re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", sentence):
            score += 1.0

        score -= min(len(sentence), 650) / 1200.0
        candidates.append((score, sentence))

    def _question_cue_terms(self, query_terms: set[str]) -> set[str]:
        cues: set[str] = set()
        for group_name, group_terms in self._QUESTION_CUE_GROUPS.items():
            if query_terms & group_terms:
                cues.update(group_terms)
                cues.add(group_name)
        return cues

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

    def _first_reasonable_sentence(self, retrieved: list[tuple[TextChunk, float]]) -> str:
        for chunk, _score in retrieved:
            for sentence in self._SENTENCE_SPLIT_RE.split(chunk.text):
                sentence = self._normalize_space(sentence)
                if len(sentence) >= 8 and not sentence.startswith("#"):
                    return sentence
        return ""

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

    def _encode_queries(self, questions: list[str]):
        if self.embedding_model is None or self.chunk_embeddings is None or not questions:
            return None

        return self._encode_texts(questions, is_query=True)

    def _encode_texts(self, texts: list[str], is_query: bool):
        if self.embedding_model is None or self.embedding_tokenizer is None or not texts:
            return None

        try:
            prepared_texts = [
                f"{self.embedding_query_prompt}{text}" if is_query else text for text in texts
            ]
            embeddings = []
            for start in range(0, len(prepared_texts), self.embedding_batch_size):
                batch_texts = prepared_texts[start : start + self.embedding_batch_size]
                inputs = self.embedding_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.embedding_max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                with self.torch.no_grad():
                    output = self.embedding_model(**inputs)

                attention_mask = inputs["attention_mask"]
                if attention_mask[:, -1].sum() == attention_mask.shape[0]:
                    pooled = output.last_hidden_state[:, -1]
                else:
                    last_token_indices = attention_mask.sum(dim=1) - 1
                    batch_indices = self.torch.arange(
                        output.last_hidden_state.size(0),
                        device=output.last_hidden_state.device,
                    )
                    pooled = output.last_hidden_state[batch_indices, last_token_indices]
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.detach())

            if not embeddings:
                return None
            return self.torch.cat(embeddings, dim=0)
        except Exception:
            return None

    def _embedding_at(self, embeddings, index: int):
        if embeddings is None:
            return None
        try:
            return embeddings[index]
        except Exception:
            return None

    def _dense_scores(self, query_embedding) -> list[tuple[int, float]]:
        try:
            scores = self.chunk_embeddings @ query_embedding
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            return [(index, float(score)) for index, score in enumerate(scores)]
        except Exception:
            query_vector = self._as_unit_vector(
                query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
            )
            return [
                (index, self._dot(query_vector, chunk_embedding))
                for index, chunk_embedding in enumerate(self.chunk_embeddings)
            ]

    def _as_unit_vector(self, vector) -> list[float]:
        values = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]

    def _dot(self, left: list[float], right: list[float]) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right))
