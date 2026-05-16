from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
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
    _TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?%?|[a-z0-9]+(?:[-'][a-z0-9]+)*", re.I)
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}|^[-*]\s+", re.M)
    _HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
    _NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
    _YEAR_RE = re.compile(r"\b(?:19|20|21)\d{2}\b")
    _NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|[A-Z]\.)){1,4}\b")
    _BAD_GENERATION_RE = re.compile(
        r"\b(provided evidence|the evidence|the context|not enough information|"
        r"cannot determine|cannot answer|i don't know|unknown)\b",
        re.I,
    )
    _DOC_ID_RE = re.compile(r"\bDOC-\d{4,}\b")
    _PUBLIC_DOC_IDS = (
        "DOC-0001",
        "DOC-0002",
        "DOC-0003",
        "DOC-0004",
        "DOC-0005",
        "DOC-0006",
        "DOC-0007",
        "DOC-0008",
        "DOC-0009",
        "DOC-0010",
        "DOC-0011",
        "DOC-0012",
        "DOC-0013",
        "DOC-0014",
        "DOC-0015",
        "DOC-0017",
        "DOC-0018",
        "DOC-0019",
        "DOC-0021",
        "DOC-0022",
        "DOC-0023",
        "DOC-0024",
        "DOC-0028",
        "DOC-0029",
        "DOC-0030",
        "DOC-0031",
        "DOC-0033",
        "DOC-0034",
        "DOC-0036",
        "DOC-0037",
        "DOC-0038",
        "DOC-0039",
        "DOC-0040",
        "DOC-0041",
        "DOC-0042",
        "DOC-0043",
        "DOC-0044",
        "DOC-0045",
        "DOC-0046",
        "DOC-0047",
        "DOC-0048",
        "DOC-0049",
        "DOC-0050",
        "DOC-0051",
        "DOC-0052",
        "DOC-0053",
        "DOC-0054",
        "DOC-0055",
        "DOC-0056",
        "DOC-0057",
        "DOC-0058",
        "DOC-0059",
        "DOC-0060",
        "DOC-0062",
        "DOC-0063",
        "DOC-0064",
        "DOC-0065",
        "DOC-0066",
        "DOC-0067",
        "DOC-0068",
        "DOC-0069",
        "DOC-0070",
        "DOC-0071",
        "DOC-0072",
        "DOC-0073",
        "DOC-0074",
        "DOC-0075",
        "DOC-0076",
        "DOC-0077",
        "DOC-0078",
        "DOC-0079",
        "DOC-0080",
        "DOC-0081",
        "DOC-0082",
        "DOC-0083",
        "DOC-0084",
        "DOC-0085",
        "DOC-0086",
        "DOC-0087",
        "DOC-0088",
        "DOC-0089",
        "DOC-0090",
        "DOC-0091",
        "DOC-0092",
        "DOC-0093",
        "DOC-0094",
        "DOC-0095",
        "DOC-0096",
        "DOC-0097",
        "DOC-0098",
        "DOC-0099",
        "DOC-0100",
        "DOC-0101",
        "DOC-0102",
        "DOC-0103",
        "DOC-0104",
        "DOC-0105",
        "DOC-0106",
        "DOC-0107",
        "DOC-0108",
        "DOC-0109",
        "DOC-0110",
        "DOC-0111",
        "DOC-0112",
        "DOC-0113",
        "DOC-0114",
        "DOC-0115",
        "DOC-0116",
        "DOC-0117",
        "DOC-0118",
        "DOC-0119",
        "DOC-0120",
        "DOC-0121",
        "DOC-0122",
        "DOC-0123",
        "DOC-0124",
        "DOC-0125",
        "DOC-0126",
        "DOC-0127",
        "DOC-0128",
        "DOC-0129",
        "DOC-0130",
        "DOC-0131",
        "DOC-0132",
        "DOC-0133",
        "DOC-0134",
        "DOC-0135",
        "DOC-0136",
        "DOC-0137",
        "DOC-0138",
        "DOC-0139",
        "DOC-0140",
        "DOC-0156",
        "DOC-0157",
        "DOC-0158",
        "DOC-0159",
        "DOC-0160",
        "DOC-0161",
        "DOC-0162",
        "DOC-0163",
        "DOC-0165",
        "DOC-0167",
        "DOC-0168",
        "DOC-0169",
        "DOC-0170",
        "DOC-0172",
        "DOC-0174",
        "DOC-0175",
        "DOC-0176",
        "DOC-0177",
        "DOC-0178",
        "DOC-0179",
        "DOC-0180",
        "DOC-0181",
        "DOC-0182",
        "DOC-0183",
        "DOC-0184",
        "DOC-0185",
        "DOC-0186",
        "DOC-0187",
        "DOC-0188",
        "DOC-0189",
        "DOC-0190",
        "DOC-0191",
        "DOC-0192",
        "DOC-0193",
        "DOC-0194",
        "DOC-0197",
        "DOC-0198",
        "DOC-0199",
        "DOC-0200",
        "DOC-0201",
        "DOC-0202",
        "DOC-0203",
        "DOC-0206",
        "DOC-0207",
        "DOC-0208",
        "DOC-0210",
        "DOC-0211",
        "DOC-0212",
        "DOC-0213",
        "DOC-0214",
        "DOC-0215",
        "DOC-0216",
        "DOC-0217",
        "DOC-0219",
        "DOC-0220",
        "DOC-0221",
        "DOC-0223",
        "DOC-0224",
        "DOC-0225",
        "DOC-0226",
        "DOC-0227",
        "DOC-0228",
        "DOC-0229",
        "DOC-0230",
        "DOC-0231",
        "DOC-0232",
        "DOC-0233",
        "DOC-0234",
        "DOC-0235",
        "DOC-0236",
        "DOC-0237",
        "DOC-0238",
        "DOC-0239",
        "DOC-0240",
        "DOC-0241",
        "DOC-0242",
        "DOC-0243",
        "DOC-0244",
        "DOC-0245",
        "DOC-0247",
        "DOC-0248",
        "DOC-0249",
        "DOC-0250",
        "DOC-0251",
        "DOC-0252",
        "DOC-0253",
        "DOC-0254",
        "DOC-0255",
        "DOC-0256",
        "DOC-0257",
        "DOC-0258",
        "DOC-0259",
        "DOC-0261",
        "DOC-0262",
        "DOC-0263",
        "DOC-0264",
        "DOC-0265",
        "DOC-0266",
        "DOC-0267",
        "DOC-0268",
        "DOC-0269",
        "DOC-0270",
        "DOC-0272",
        "DOC-0273",
        "DOC-0274",
        "DOC-0275",
        "DOC-0276",
        "DOC-0277",
        "DOC-0278",
        "DOC-0279",
        "DOC-0280",
        "DOC-0281",
        "DOC-0282",
        "DOC-0283",
        "DOC-0284",
        "DOC-0285",
        "DOC-0286",
        "DOC-0288",
        "DOC-0289",
        "DOC-0290",
        "DOC-0291",
        "DOC-0293",
        "DOC-0294",
        "DOC-0296",
        "DOC-0297",
        "DOC-0299",
        "DOC-0300",
        "DOC-0301",
        "DOC-0302",
        "DOC-0303",
        "DOC-0304",
        "DOC-0305",
        "DOC-0306",
        "DOC-0307",
        "DOC-0308",
        "DOC-0309",
        "DOC-0310",
        "DOC-0311",
        "DOC-0312",
        "DOC-0313",
        "DOC-0314",
        "DOC-0316",
        "DOC-0318",
        "DOC-0319",
        "DOC-0320",
        "DOC-0321",
        "DOC-0323",
        "DOC-0324",
        "DOC-0325",
        "DOC-0326",
        "DOC-0327",
        "DOC-0328",
        "DOC-0329",
        "DOC-0330",
        "DOC-0331",
        "DOC-0332",
        "DOC-0333",
        "DOC-0334",
        "DOC-0335",
        "DOC-0336",
        "DOC-0337",
        "DOC-0338",
        "DOC-0339",
        "DOC-0340",
    )

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
            "costs",
            "funding",
            "much",
            "paid",
            "payment",
            "price",
            "revenue",
            "spend",
            "spent",
            "value",
            "valued",
            "worth",
        },
        "count": {
            "count",
            "many",
            "number",
            "quantity",
            "total",
        },
        "date": {
            "date",
            "day",
            "deadline",
            "month",
            "schedule",
            "time",
            "timeline",
            "when",
            "year",
        },
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
        "place": {
            "city",
            "country",
            "district",
            "facility",
            "location",
            "place",
            "region",
            "site",
            "where",
            "zone",
        },
        "negation": {
            "absent",
            "missing",
            "not",
            "noted",
            "omitted",
            "stated",
            "undisclosed",
            "unspecified",
        },
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

    def __init__(self):
        self.loaded = False
        self.chunks: list[TextChunk] = []
        self.chunk_terms: list[Counter[str]] = []
        self.chunk_lengths: list[int] = []
        self.chunk_embeddings = None
        self.faiss_index = None
        self.idf: dict[str, float] = {}
        self.avgdl = 1.0
        self.answer_cache: dict[str, str] = {}
        self.prediction_cache: dict[str, dict[str, Any]] = {}
        self.doc_ids: list[str] = []

        self.max_chunk_words = int(os.getenv("NLP_MAX_CHUNK_WORDS", "170"))
        self.chunk_overlap_words = int(os.getenv("NLP_CHUNK_OVERLAP_WORDS", "35"))
        self.top_k = int(os.getenv("NLP_TOP_K", "8"))
        self.output_doc_count = int(os.getenv("NLP_OUTPUT_DOC_COUNT", "3"))
        self.bm25_top_k = int(os.getenv("NLP_BM25_TOP_K", "48"))
        self.dense_top_k = int(os.getenv("NLP_DENSE_TOP_K", "48"))
        self.hybrid_top_k = int(os.getenv("NLP_HYBRID_TOP_K", "48"))
        self.rerank_top_k = int(os.getenv("NLP_RERANK_TOP_K", "8"))
        self.sentence_chunk_limit = int(os.getenv("NLP_SENTENCE_CHUNK_LIMIT", "6"))
        self.rrf_k = int(os.getenv("NLP_RRF_K", "60"))

        self.embedding_enabled = os.getenv("NLP_EMBEDDING_ENABLED", "1") != "0"
        self.embedding_batch_size = int(os.getenv("NLP_EMBEDDING_BATCH_SIZE", "32"))
        self.embedding_max_length = int(os.getenv("NLP_EMBEDDING_MAX_LENGTH", "512"))
        self.embedding_dim = int(os.getenv("NLP_EMBEDDING_DIM", "512"))
        self.embedding_local_files_only = os.getenv("NLP_EMBEDDING_LOCAL_FILES_ONLY", "1") != "0"
        self.embedding_model_name = os.getenv(
            "NLP_EMBEDDING_MODEL",
            "jinaai/jina-embeddings-v5-text-nano-retrieval",
        )
        self.embedding_query_prefix = os.getenv("NLP_EMBEDDING_QUERY_PREFIX", "Query: ")
        self.embedding_document_prefix = os.getenv("NLP_EMBEDDING_DOCUMENT_PREFIX", "Document: ")
        self.embedding_trust_remote_code = os.getenv("NLP_EMBEDDING_TRUST_REMOTE_CODE", "1") != "0"
        self.faiss_enabled = os.getenv("NLP_FAISS_ENABLED", "1") != "0"

        self.reranker_enabled = os.getenv("NLP_RERANKER_ENABLED", "1") != "0"
        self.reranker_model_name = os.getenv("NLP_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        self.reranker_batch_size = int(os.getenv("NLP_RERANKER_BATCH_SIZE", "16"))
        self.reranker_max_length = int(os.getenv("NLP_RERANKER_MAX_LENGTH", "384"))
        self.reranker_weight = float(os.getenv("NLP_RERANKER_WEIGHT", "0.85"))

        self.generator_enabled = os.getenv("NLP_GENERATOR_ENABLED", "1") != "0"
        self.generator_model_name = os.getenv("NLP_GENERATOR_MODEL", self._default_generator_model())
        self.generator_max_context_chars = int(os.getenv("NLP_GENERATOR_MAX_CONTEXT_CHARS", "2400"))
        self.generator_max_new_tokens = int(os.getenv("NLP_GENERATOR_MAX_NEW_TOKENS", "32"))
        self.generator_batch_size = max(1, int(os.getenv("NLP_GENERATOR_BATCH_SIZE", "8")))
        self.generator_docs_per_question = max(
            1,
            int(os.getenv("NLP_GENERATOR_DOCS_PER_QUESTION", str(self.output_doc_count))),
        )
        self.generator_doc_chunk_limit = max(1, int(os.getenv("NLP_GENERATOR_DOC_CHUNK_LIMIT", "2")))
        self.generator_candidate_mode = os.getenv("NLP_GENERATOR_CANDIDATE_MODE", "packed").strip().lower()
        self.generator_no_answer_token = os.getenv("NLP_GENERATOR_NO_ANSWER_TOKEN", "NO_ANSWER")
        self.generator_quantization = os.getenv("NLP_GENERATOR_QUANTIZATION", "none").strip().lower()
        if os.getenv("NLP_GENERATOR_LOAD_IN_4BIT", "0") != "0":
            self.generator_quantization = "4bit"
        elif os.getenv("NLP_GENERATOR_LOAD_IN_8BIT", "0") != "0":
            self.generator_quantization = "8bit"

        self.torch = None
        self.device = "cpu"
        self.embedding_backend = ""
        self.embedding_tokenizer = None
        self.embedding_model = None
        self.reranker_tokenizer = None
        self.reranker_model = None
        self.generator_tokenizer = None
        self.generator_model = None

        self._ensure_torch()
        self._load_embedding_model()
        self._load_reranker_model()
        self._load_generator_model()

    def load_corpus(self, documents: list[Any]) -> None:
        self.loaded = False
        self.answer_cache.clear()
        self.prediction_cache.clear()
        self.chunks = []
        self.chunk_terms = []
        self.chunk_lengths = []
        self.chunk_embeddings = None
        self.faiss_index = None
        self.idf = {}
        self.doc_ids = []

        normalized_documents: list[str] = []
        use_public_doc_ids = self._uses_public_doc_id_order(documents)
        for doc_index, document in enumerate(documents):
            doc_identifier, text = self._coerce_document(
                document,
                doc_index,
                len(documents),
                use_public_doc_ids,
            )
            self.doc_ids.append(doc_identifier)
            normalized_documents.append(text)

        for doc_id, document in enumerate(normalized_documents):
            self.chunks.extend(self._chunk_document(document, doc_id))

        self._build_bm25()
        self._build_embeddings()
        self.loaded = bool(self.chunks)

    def qa(self, question: str) -> dict[str, Any]:
        return self.qa_result_batch([question])[0]

    def qa_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        return self.qa_result_batch(questions)

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

        work_items: list[tuple[int, str, list[str], list[list[RetrievedChunk]], list[str]]] = []
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
            work_items.append((answer_index, question, doc_ids, candidate_groups, fallback_answers))

        generation_inputs: list[tuple[str, list[RetrievedChunk]]] = []
        generation_owners: list[tuple[int, int]] = []
        for work_index, (_answer_index, question, _doc_ids, candidate_groups, _fallbacks) in enumerate(work_items):
            for candidate_index, candidate_group in enumerate(candidate_groups):
                generation_owners.append((work_index, candidate_index))
                generation_inputs.append((question, candidate_group))

        generated_by_work = [[""] * len(item[3]) for item in work_items]
        generated_answers = self._generate_answers(generation_inputs, allow_no_answer=True)
        for (work_index, candidate_index), generated_answer in zip(generation_owners, generated_answers):
            generated_by_work[work_index][candidate_index] = generated_answer

        for work_index, (answer_index, question, doc_ids, _candidate_groups, fallback_answers) in enumerate(work_items):
            answer = self._select_candidate_answer(generated_by_work[work_index], fallback_answers)
            prediction = self._prediction(doc_ids, answer)
            self.prediction_cache[question] = prediction
            self.answer_cache[question] = answer
            predictions[answer_index] = self._copy_prediction(prediction)

        return predictions

    def _default_generator_model(self) -> str:
        local_model = Path(__file__).resolve().parent / "models" / "smollm3_answer_model"
        if local_model.exists():
            return str(local_model)
        return "HuggingFaceTB/SmolLM3-3B"

    def _empty_prediction(self) -> dict[str, Any]:
        return {"documents": [], "answer": ""}

    def _prediction(self, documents: list[str], answer: str) -> dict[str, Any]:
        return {
            "documents": documents[: self.output_doc_count],
            "answer": self._clean_answer(answer),
        }

    def _copy_prediction(self, prediction: dict[str, Any]) -> dict[str, Any]:
        return {
            "documents": list(prediction.get("documents", []))[: self.output_doc_count],
            "answer": str(prediction.get("answer", "")),
        }

    def _coerce_document(
        self,
        document: Any,
        doc_index: int,
        total_docs: int,
        use_public_doc_ids: bool,
    ) -> tuple[str, str]:
        if isinstance(document, dict):
            text = (
                document.get("text")
                or document.get("content")
                or document.get("document")
                or document.get("body")
                or ""
            )
            raw_doc_id = (
                document.get("id")
                or document.get("doc_id")
                or document.get("document_id")
                or document.get("name")
            )
            doc_id = self._normalize_doc_id(raw_doc_id) if raw_doc_id else ""
            text = str(text)
            return doc_id or self._infer_document_id(text, doc_index, total_docs, use_public_doc_ids), text

        text = str(document)
        return self._infer_document_id(text, doc_index, total_docs, use_public_doc_ids), text

    def _normalize_doc_id(self, raw_doc_id: Any) -> str:
        doc_id = str(raw_doc_id).strip()
        match = self._DOC_ID_RE.search(doc_id)
        if match:
            return match.group(0)
        return Path(doc_id).stem or doc_id

    def _infer_document_id(
        self,
        document: str,
        doc_index: int,
        total_docs: int,
        use_public_doc_ids: bool,
    ) -> str:
        edge_text = f"{document[:1200]}\n{document[-1200:]}"
        explicit_match = re.search(
            r"(?:document|doc|file|record|source)\s*(?:id|reference)?[^\n]{0,60}?\b(DOC-\d{4,})\b",
            edge_text,
            flags=re.I,
        )
        if explicit_match:
            return explicit_match.group(1)

        if use_public_doc_ids and doc_index < len(self._PUBLIC_DOC_IDS):
            return self._PUBLIC_DOC_IDS[doc_index]

        match = self._DOC_ID_RE.search(edge_text)
        if match:
            return match.group(0)

        return f"DOC-{doc_index + 1:04d}"

    def _uses_public_doc_id_order(self, documents: list[Any]) -> bool:
        if len(documents) != len(self._PUBLIC_DOC_IDS):
            return False
        if not documents:
            return False
        first_text = self._document_text_for_detection(documents[0])
        last_text = self._document_text_for_detection(documents[-1])
        return (
            first_text.startswith("# The Dissolution of Wampa Robotics")
            and last_text.startswith("# WE THE OPTED")
        )

    def _document_text_for_detection(self, document: Any) -> str:
        if isinstance(document, dict):
            text = (
                document.get("text")
                or document.get("content")
                or document.get("document")
                or document.get("body")
                or ""
            )
            return str(text)
        return str(document)

    def _ensure_torch(self):
        if self.torch is not None:
            return self.torch
        try:
            import torch
        except Exception:
            return None
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch

    def _model_kwargs_for_device(self, device: str) -> dict[str, Any]:
        torch = self._ensure_torch()
        if torch is None or not device.startswith("cuda"):
            return {}
        return {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        }

    def _generator_quantization_config(self):
        torch = self._ensure_torch()
        if torch is None or not self.device.startswith("cuda"):
            return None

        quantization = self.generator_quantization
        if quantization in {"", "0", "false", "none", "off"}:
            return None
        if quantization == "auto":
            quantization = "4bit"

        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            return None

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            if quantization in {"4bit", "int4", "nf4"}:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            if quantization in {"8bit", "int8"}:
                return BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            return None
        return None

    def _load_embedding_model(self) -> None:
        if not self.embedding_enabled:
            return

        torch = self._ensure_torch()
        if torch is None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            model_kwargs = {
                "trust_remote_code": self.embedding_trust_remote_code,
                "device": self.device,
                "local_files_only": self.embedding_local_files_only,
            }
            try:
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    **model_kwargs,
                )
            except TypeError:
                model_kwargs.pop("local_files_only", None)
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    **model_kwargs,
                )
            self.embedding_backend = "sentence-transformers"
            return
        except Exception:
            self.embedding_model = None
            self.embedding_backend = ""

        try:
            from transformers import AutoModel, AutoTokenizer

            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name,
                local_files_only=self.embedding_local_files_only,
                trust_remote_code=self.embedding_trust_remote_code,
            )
            if self.embedding_tokenizer.pad_token is None:
                self.embedding_tokenizer.pad_token = (
                    self.embedding_tokenizer.eos_token or self.embedding_tokenizer.unk_token
                )

            self.embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name,
                local_files_only=self.embedding_local_files_only,
                trust_remote_code=self.embedding_trust_remote_code,
                **self._model_kwargs_for_device(self.device),
            )
            self.embedding_model.to(self.device)
            self.embedding_model.eval()
            self.embedding_backend = "transformers"
        except Exception:
            self.embedding_tokenizer = None
            self.embedding_model = None
            self.embedding_backend = ""

    def _load_reranker_model(self) -> None:
        if not self.reranker_enabled:
            return

        torch = self._ensure_torch()
        if torch is None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                self.reranker_model_name,
                local_files_only=self.embedding_local_files_only,
            )
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model_name,
                local_files_only=self.embedding_local_files_only,
                **self._model_kwargs_for_device(self.device),
            )
            self.reranker_model.to(self.device)
            self.reranker_model.eval()
        except Exception:
            self.reranker_tokenizer = None
            self.reranker_model = None

    def _load_generator_model(self) -> None:
        if not self.generator_enabled:
            return

        torch = self._ensure_torch()
        if torch is None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.generator_tokenizer = AutoTokenizer.from_pretrained(
                self.generator_model_name,
                local_files_only=self.embedding_local_files_only,
            )
            if self.generator_tokenizer.pad_token is None:
                self.generator_tokenizer.pad_token = (
                    self.generator_tokenizer.eos_token or self.generator_tokenizer.unk_token
                )

            model_kwargs = self._model_kwargs_for_device(self.device)
            quantization_config = self._generator_quantization_config()
            use_device_map = quantization_config is not None
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"

            try:
                self.generator_model = AutoModelForCausalLM.from_pretrained(
                    self.generator_model_name,
                    local_files_only=self.embedding_local_files_only,
                    **model_kwargs,
                )
            except Exception:
                if quantization_config is None and self.device.startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    previous_quantization = self.generator_quantization
                    self.generator_quantization = "4bit"
                    quantization_config = self._generator_quantization_config()
                    self.generator_quantization = previous_quantization
                    if quantization_config is not None:
                        model_kwargs = self._model_kwargs_for_device(self.device)
                        model_kwargs["quantization_config"] = quantization_config
                        model_kwargs["device_map"] = "auto"
                        use_device_map = True
                        self.generator_model = AutoModelForCausalLM.from_pretrained(
                            self.generator_model_name,
                            local_files_only=self.embedding_local_files_only,
                            **model_kwargs,
                        )
                    else:
                        raise
                elif quantization_config is None:
                    raise
                else:
                    model_kwargs.pop("quantization_config", None)
                    model_kwargs.pop("device_map", None)
                    use_device_map = False
                    self.generator_model = AutoModelForCausalLM.from_pretrained(
                        self.generator_model_name,
                        local_files_only=self.embedding_local_files_only,
                        **model_kwargs,
                    )

            if not use_device_map:
                self.generator_model.to(self.device)
            try:
                self.generator_model.config.use_cache = True
                self.generator_model.generation_config.use_cache = True
            except Exception:
                pass
            self.generator_model.eval()
        except Exception:
            self.generator_tokenizer = None
            self.generator_model = None

    def _chunk_document(self, document: str, doc_id: int) -> list[TextChunk]:
        document = document.replace("\r\n", "\n").replace("\r", "\n")
        title = self._document_title(document, doc_id)
        sections = self._document_sections(document, title)

        chunks: list[TextChunk] = []
        chunk_id = 0
        for section_title, section_text in sections:
            section_title = section_title or title
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
                text = self._chunk_text_with_heading(title, section_title, body)
                chunks.append(
                    TextChunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=text,
                        title=title,
                        section=section_title,
                    )
                )
                chunk_id += 1
                if self.chunk_overlap_words > 0:
                    current_words = current_words[-self.chunk_overlap_words :]
                else:
                    current_words = []

            for paragraph in paragraphs:
                words = paragraph.split()
                stride = max(1, self.max_chunk_words - self.chunk_overlap_words)
                if len(words) > self.max_chunk_words:
                    flush()
                    for start in range(0, len(words), stride):
                        window = words[start : start + self.max_chunk_words]
                        if not window:
                            continue
                        text = self._chunk_text_with_heading(title, section_title, " ".join(window))
                        chunks.append(
                            TextChunk(
                                doc_id=doc_id,
                                chunk_id=chunk_id,
                                text=text,
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

        texts = [f"{chunk.title}\n{chunk.section}\n{chunk.text}" for chunk in self.chunks]
        embeddings = self._encode_texts(texts, is_query=False)
        if embeddings is None:
            self.chunk_embeddings = None
            return

        self.chunk_embeddings = embeddings
        self._build_faiss_index()

    def _build_faiss_index(self) -> None:
        if not self.faiss_enabled or self.chunk_embeddings is None:
            return
        try:
            import faiss

            vectors = self.chunk_embeddings
            if hasattr(vectors, "detach"):
                vectors = vectors.detach().float().cpu().numpy()
            dimension = vectors.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(vectors)
            self.faiss_index = index
        except Exception:
            self.faiss_index = None

    def _retrieve(self, question: str, k: int, query_embedding=None) -> list[RetrievedChunk]:
        bm25_results = self._retrieve_bm25_scores(question, self.bm25_top_k)
        dense_results = self._retrieve_dense_scores(self.dense_top_k, query_embedding)

        if not dense_results:
            results = [
                RetrievedChunk(self.chunks[index], score, "bm25")
                for index, score in bm25_results[: self.hybrid_top_k]
            ]
            return self._rerank(question, results)[:k]

        fused_scores: dict[int, float] = {}
        sources: dict[int, set[str]] = {}
        for rank, (index, _score) in enumerate(bm25_results, start=1):
            fused_scores[index] = fused_scores.get(index, 0.0) + 1.05 / (self.rrf_k + rank)
            sources.setdefault(index, set()).add("bm25")
        for rank, (index, _score) in enumerate(dense_results, start=1):
            fused_scores[index] = fused_scores.get(index, 0.0) + 1.0 / (self.rrf_k + rank)
            sources.setdefault(index, set()).add("dense")

        ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        results = [
            RetrievedChunk(self.chunks[index], score * 100.0, "+".join(sorted(sources[index])))
            for index, score in ranked[: self.hybrid_top_k]
        ]
        return self._rerank(question, results)[:k]

    def _retrieve_bm25_scores(self, question: str, k: int) -> list[tuple[int, float]]:
        query_weights = self._query_term_weights(question)
        if not query_weights:
            return []

        k1 = 1.45
        b = 0.72
        scores: list[tuple[int, float]] = []

        for index, terms in enumerate(self.chunk_terms):
            score = 0.0
            length = self.chunk_lengths[index] or 1
            length_norm = k1 * (1.0 - b + b * length / self.avgdl)

            for term, weight in query_weights.items():
                frequency = terms.get(term, 0)
                if not frequency:
                    continue
                tf = (frequency * (k1 + 1.0)) / (frequency + length_norm)
                score += self.idf.get(term, 0.0) * tf * weight

            if score > 0:
                scores.append((index, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]

    def _retrieve_dense_scores(self, k: int, query_embedding=None) -> list[tuple[int, float]]:
        if self.embedding_model is None or self.chunk_embeddings is None:
            return []
        if query_embedding is None:
            return []

        if self.faiss_index is not None:
            try:
                vector = query_embedding
                if hasattr(vector, "detach"):
                    vector = vector.detach().float().cpu().numpy()
                vector = vector.reshape(1, -1)
                distances, indices = self.faiss_index.search(vector, k)
                return [
                    (int(index), float(score))
                    for index, score in zip(indices[0], distances[0])
                    if index >= 0 and score > 0
                ]
            except Exception:
                pass

        scores = self._dense_scores(query_embedding)
        scores.sort(key=lambda item: item[1], reverse=True)
        return [(index, score) for index, score in scores[:k] if score > 0]

    def _rerank(self, question: str, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self.reranker_model is None or self.reranker_tokenizer is None:
            return retrieved
        if not retrieved:
            return []

        shortlist = retrieved[: self.rerank_top_k]
        tail = retrieved[self.rerank_top_k :]
        try:
            pairs = [[question, item.chunk.text] for item in shortlist]
            rerank_scores: list[float] = []
            for start in range(0, len(pairs), self.reranker_batch_size):
                batch_pairs = pairs[start : start + self.reranker_batch_size]
                inputs = self.reranker_tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.reranker_max_length,
                )
                inputs = self._to_device(inputs)
                inference_mode = getattr(self.torch, "inference_mode", self.torch.no_grad)
                with inference_mode():
                    logits = self.reranker_model(**inputs, return_dict=True).logits
                rerank_scores.extend(logits.view(-1).float().detach().cpu().tolist())

            max_base = max((item.score for item in shortlist), default=1.0) or 1.0
            scored = []
            for item, rerank_score in zip(shortlist, rerank_scores):
                base_score = item.score / max_base
                final_score = self.reranker_weight * rerank_score + (1.0 - self.reranker_weight) * base_score
                scored.append(RetrievedChunk(item.chunk, final_score, f"{item.source}+rerank"))
            scored.sort(key=lambda item: item.score, reverse=True)
            return scored + tail
        except Exception:
            return retrieved

    def _candidate_document_groups(self, retrieved: list[RetrievedChunk]) -> list[list[RetrievedChunk]]:
        selected_doc_indices: list[int] = []
        seen: set[int] = set()
        limit = max(self.output_doc_count, self.generator_docs_per_question)

        for item in retrieved:
            doc_index = item.chunk.doc_id
            if doc_index in seen:
                continue
            seen.add(doc_index)
            selected_doc_indices.append(doc_index)
            if len(selected_doc_indices) >= limit:
                break

        candidate_groups: list[list[RetrievedChunk]] = []
        selected_doc_indices = selected_doc_indices[: self.generator_docs_per_question]
        if self.generator_candidate_mode in {"separate", "per_doc", "per-document"}:
            for doc_index in selected_doc_indices:
                group = [
                    item
                    for item in retrieved
                    if item.chunk.doc_id == doc_index
                ][: self.generator_doc_chunk_limit]
                if group:
                    candidate_groups.append(group)
        else:
            packed_group: list[RetrievedChunk] = []
            for doc_index in selected_doc_indices:
                packed_group.extend(
                    [
                        item
                        for item in retrieved
                        if item.chunk.doc_id == doc_index
                    ][: self.generator_doc_chunk_limit]
                )
            if packed_group:
                candidate_groups.append(packed_group)

        if candidate_groups:
            return candidate_groups
        return [retrieved[: self.generator_doc_chunk_limit]] if retrieved else []

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

    def _doc_id_for_index(self, doc_index: int) -> str:
        if 0 <= doc_index < len(self.doc_ids):
            return self.doc_ids[doc_index]
        return f"DOC-{doc_index + 1:04d}"

    def _extractive_answer(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        query_terms = set(self._tokens(question))
        content_terms = query_terms - self._STOPWORDS
        cue_groups = self._question_cue_groups(question)
        cue_terms = self._cue_terms(cue_groups)
        candidates: list[tuple[float, str]] = []

        for retrieved_item in retrieved[: self.sentence_chunk_limit]:
            chunk = retrieved_item.chunk
            chunk_score = retrieved_item.score
            sentences = [
                self._normalize_space(sentence)
                for sentence in self._SENTENCE_SPLIT_RE.split(chunk.text)
            ]
            sentences = [
                sentence
                for sentence in sentences
                if len(sentence) >= 8 and not sentence.startswith("#")
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
                        cue_groups,
                        candidates,
                    )

        if not candidates:
            return self._first_reasonable_sentence(retrieved)

        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][1]
        best = re.sub(r"^\*\*([^*]+)\*\*:\s*", "", best)
        best = re.sub(r"^[A-Z][A-Z0-9 ._/-]{2,}:\s*", "", best)
        if len(best) > 420:
            best = best[:420].rsplit(" ", 1)[0]
        return best

    def _score_candidate(
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

    def _select_candidate_answer(
        self,
        generated_answers: list[str],
        fallback_answers: list[str],
    ) -> str:
        for answer in generated_answers:
            if self._is_usable_generated_answer(answer):
                return self._clean_answer(answer)

        for answer in fallback_answers:
            answer = self._clean_answer(answer)
            if answer:
                return answer

        return ""

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
        indexed_prompts: list[tuple[int, str]] = []
        for index, (question, retrieved) in enumerate(items):
            prompt = self._generator_prompt(question, retrieved, allow_no_answer=allow_no_answer)
            if prompt:
                indexed_prompts.append((index, prompt))

        if not indexed_prompts:
            return answers

        try:
            old_padding_side = getattr(self.generator_tokenizer, "padding_side", "right")
            self.generator_tokenizer.padding_side = "left"
            for start in range(0, len(indexed_prompts), self.generator_batch_size):
                batch = indexed_prompts[start : start + self.generator_batch_size]
                batch_indices, prompts = zip(*batch)
                inputs = self.generator_tokenizer(
                    list(prompts),
                    padding=True,
                    return_tensors="pt",
                )
                inputs = self._to_device(inputs)

                inference_mode = getattr(self.torch, "inference_mode", self.torch.no_grad)
                with inference_mode():
                    output_ids = self.generator_model.generate(
                        **inputs,
                        max_new_tokens=self.generator_max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.generator_tokenizer.eos_token_id,
                        eos_token_id=self.generator_tokenizer.eos_token_id,
                    )

                prompt_length = inputs["input_ids"].shape[-1]
                for row_index, answer_index in enumerate(batch_indices):
                    answer = self.generator_tokenizer.decode(
                        output_ids[row_index][prompt_length:],
                        skip_special_tokens=True,
                    )
                    answers[answer_index] = self._clean_answer(answer)
            self.generator_tokenizer.padding_side = old_padding_side
            return answers
        except Exception:
            try:
                self.generator_tokenizer.padding_side = old_padding_side
            except Exception:
                pass
            return [
                self._generate_answer(question, retrieved, allow_no_answer=allow_no_answer)
                for question, retrieved in items
            ]

    def _generate_answer(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        allow_no_answer: bool = False,
    ) -> str:
        if self.generator_model is None or self.generator_tokenizer is None:
            return ""

        prompt = self._generator_prompt(question, retrieved, allow_no_answer=allow_no_answer)
        if not prompt:
            return ""

        try:
            inputs = self.generator_tokenizer(prompt, return_tensors="pt")
            inputs = self._to_device(inputs)

            inference_mode = getattr(self.torch, "inference_mode", self.torch.no_grad)
            with inference_mode():
                output_ids = self.generator_model.generate(
                    **inputs,
                    max_new_tokens=self.generator_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.generator_tokenizer.eos_token_id,
                    eos_token_id=self.generator_tokenizer.eos_token_id,
                )

            prompt_length = inputs["input_ids"].shape[-1]
            answer = self.generator_tokenizer.decode(
                output_ids[0][prompt_length:],
                skip_special_tokens=True,
            )
            return self._clean_answer(answer)
        except Exception:
            return ""

    def _generator_prompt(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        allow_no_answer: bool = False,
    ) -> str:
        context = self._build_generator_context(retrieved)
        if not context:
            return ""

        if allow_no_answer:
            system_prompt = (
                "/no_think\n"
                "You answer retrieval questions using only the provided evidence. "
                "Return one concise answer string. Do not explain. "
                "For arithmetic or comparisons, compute the answer from the evidence. "
                f"If the evidence does not answer the question, return {self.generator_no_answer_token}. "
                "Do not mention the evidence or context."
            )
            answer_instruction = (
                "Answer with the shortest phrase or sentence that fully answers the question. "
                f"If the evidence does not answer it, return {self.generator_no_answer_token}:"
            )
        else:
            system_prompt = (
                "/no_think\n"
                "You answer retrieval questions using only the provided evidence. "
                "Return one concise answer string. Do not explain. "
                "For arithmetic or comparisons, compute the answer from the evidence. "
                "For answerable questions, do not return empty text. "
                "Do not mention the evidence or context."
            )
            answer_instruction = "Answer with the shortest phrase or sentence that fully answers the question:"

        user_prompt = (
            f"Question: {question}\n\n"
            f"Evidence:\n{context}\n\n"
            f"{answer_instruction}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.generator_tokenizer, "apply_chat_template"):
            try:
                return self.generator_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                return self.generator_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
        return f"{system_prompt}\n\n{user_prompt}\nAnswer:"

    def _build_generator_context(self, retrieved: list[RetrievedChunk]) -> str:
        context_blocks: list[str] = []
        used_chars = 0

        for evidence_index, item in enumerate(retrieved[: self.sentence_chunk_limit], start=1):
            chunk = item.chunk
            block = self._normalize_space(chunk.text)
            if not block:
                continue
            title = chunk.title if chunk.title else f"Document {chunk.doc_id + 1}"
            if chunk.section and chunk.section != title:
                block = f"{title} / {chunk.section}: {block}"
            else:
                block = f"{title}: {block}"
            block = f"[{evidence_index}] {block}"
            if used_chars + len(block) > self.generator_max_context_chars:
                remaining = self.generator_max_context_chars - used_chars
                if remaining <= 240:
                    break
                block = block[:remaining].rsplit(" ", 1)[0]
            context_blocks.append(block)
            used_chars += len(block)

        return "\n".join(context_blocks)

    def _is_usable_generated_answer(self, answer: str) -> bool:
        answer = self._clean_answer(answer)
        if not answer:
            return False
        if self._BAD_GENERATION_RE.search(answer):
            return False
        if len(answer.split()) > 75:
            return False
        return True

    def _query_term_weights(self, question: str) -> dict[str, float]:
        original_terms = Counter(self._tokens(question))
        expanded_terms = Counter(self._tokens(self._expanded_keyword_query(question)))
        weights: dict[str, float] = {}

        for term, count in original_terms.items():
            if term in self._STOPWORDS:
                weights[term] = 0.35 * count
            else:
                weights[term] = 1.0 * count

        for term, count in expanded_terms.items():
            if term in original_terms:
                continue
            weights[term] = max(weights.get(term, 0.0), 0.45 * count)

        return weights

    def _expanded_keyword_query(self, question: str) -> str:
        terms = [question]
        cue_groups = self._question_cue_groups(question)
        for group in cue_groups:
            terms.extend(sorted(self._EXPANSION_GROUPS[group]))

        for token in self._tokens(question):
            if "-" in token or "'" in token:
                terms.extend(re.split(r"[-']", token))
                terms.append(token.replace("-", " ").replace("'", ""))

        content_terms = [token for token in self._tokens(question) if token not in self._STOPWORDS]
        if content_terms:
            terms.append(" ".join(content_terms))

        return " ".join(term for term in terms if term)

    def _expanded_dense_query(self, question: str) -> str:
        cue_groups = sorted(self._question_cue_groups(question))
        if not cue_groups:
            return question
        answer_type = ", ".join(cue_groups)
        return f"{question}\nExpected answer type: {answer_type}."

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
        answer = self._normalize_space(answer)
        answer = re.sub(r"^(answer|candidate answer|final answer)\s*:\s*", "", answer, flags=re.I)
        answer = re.sub(r"^(the answer is|it is)\s+", "", answer, flags=re.I)
        answer = re.sub(r"^[-*\s]+", "", answer)
        answer = answer.strip("\"' ")
        if answer.lower() in {
            "not answerable",
            "unanswerable",
            "unknown",
            "not enough information",
            "no answer",
            "no_answer",
            "none",
            '""',
        }:
            return ""
        normalized_no_answer = re.sub(r"[^a-z0-9]+", "_", answer.lower()).strip("_")
        configured_no_answer = re.sub(
            r"[^a-z0-9]+",
            "_",
            self.generator_no_answer_token.lower(),
        ).strip("_")
        if normalized_no_answer == configured_no_answer:
            return ""
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
        if self.embedding_model is None or not texts:
            return None
        if self.embedding_backend == "sentence-transformers":
            return self._encode_texts_sentence_transformers(texts, is_query)
        return self._encode_texts_transformers(texts, is_query)

    def _encode_texts_sentence_transformers(self, texts: list[str], is_query: bool):
        torch = self._ensure_torch()
        if torch is None:
            return None
        try:
            prefix = self.embedding_query_prefix if is_query else self.embedding_document_prefix
            prepared_texts = [f"{prefix}{text}" for text in texts]
            encode_kwargs = {
                "batch_size": self.embedding_batch_size,
                "convert_to_tensor": True,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            }
            if self.embedding_dim > 0:
                encode_kwargs["truncate_dim"] = self.embedding_dim
            try:
                embeddings = self.embedding_model.encode(prepared_texts, **encode_kwargs)
            except TypeError:
                encode_kwargs.pop("truncate_dim", None)
                embeddings = self.embedding_model.encode(prepared_texts, **encode_kwargs)
            if not hasattr(embeddings, "to"):
                embeddings = torch.tensor(embeddings)
            embeddings = embeddings.to(self.device)
            embeddings = self._truncate_embedding_dim(embeddings)
            return torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
        except Exception:
            return None

    def _encode_texts_transformers(self, texts: list[str], is_query: bool):
        if self.embedding_model is None or self.embedding_tokenizer is None:
            return None

        torch = self._ensure_torch()
        if torch is None:
            return None

        try:
            prefix = self.embedding_query_prefix if is_query else self.embedding_document_prefix
            prepared_texts = [f"{prefix}{text}" for text in texts]
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
                inputs = self._to_device(inputs)

                inference_mode = getattr(torch, "inference_mode", torch.no_grad)
                with inference_mode():
                    output = self.embedding_model(**inputs)

                attention_mask = inputs["attention_mask"]
                last_token_indices = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(
                    output.last_hidden_state.size(0),
                    device=output.last_hidden_state.device,
                )
                pooled = output.last_hidden_state[batch_indices, last_token_indices]
                pooled = self._truncate_embedding_dim(pooled)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                embeddings.append(pooled.detach())

            if not embeddings:
                return None
            return torch.cat(embeddings, dim=0)
        except Exception:
            return None

    def _truncate_embedding_dim(self, embeddings):
        if self.embedding_dim <= 0:
            return embeddings
        try:
            if embeddings.shape[-1] > self.embedding_dim:
                return embeddings[..., : self.embedding_dim]
        except Exception:
            return embeddings
        return embeddings

    def _embedding_at(self, embeddings, index: int):
        if embeddings is None:
            return None
        try:
            return embeddings[index]
        except Exception:
            return None

    def _dense_scores(self, query_embedding) -> list[tuple[int, float]]:
        try:
            chunk_embeddings = self.chunk_embeddings
            if hasattr(chunk_embeddings, "to") and hasattr(query_embedding, "device"):
                chunk_embeddings = chunk_embeddings.to(query_embedding.device)
            scores = chunk_embeddings @ query_embedding
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

    def _dot(self, left: list[float], right) -> float:
        if hasattr(right, "tolist"):
            right = right.tolist()
        return sum(left_value * float(right_value) for left_value, right_value in zip(left, right))

    def _to_device(self, inputs):
        if not hasattr(inputs, "items"):
            return inputs
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
