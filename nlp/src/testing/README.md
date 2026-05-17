# NLP Speed Testing Sandbox

This folder holds experimental manager variants for speed-first NLP submissions.
They do not change the active service until you explicitly wire one in.

## Quickest Tests

From `nlp/src/nlp_server.py`, temporarily swap:

```python
from nlp_manager import NLPManager
```

to one of:

```python
from testing.fast_bm25_manager import NLPManager
from testing.selective_generator_manager import NLPManager
from testing.tiny_t5_answer_manager import NLPManager
```

Then rebuild/test NLP as usual.

## Variants

`fast_bm25_manager.py`

- Pure Python BM25 with an inverted index.
- No embeddings, reranker, or generator.
- Designed to chase the `0.94+` speed style scores.
- Expected accuracy target: retrieval-heavy partial credit, maybe strong on L1.

`selective_generator_manager.py`

- Reuses the current `nlp_manager.py` retrieval/models.
- Skips SmolLM3 for easy direct questions and only generates for hard-looking questions.
- Designed as a compromise if pure extractive loses too much accuracy.

`tiny_t5_answer_manager.py`

- Reuses the current retrieval stack but swaps CausalLM generation for `AutoModelForSeq2SeqLM`.
- Default model is `google/flan-t5-small`, or local `nlp/src/models/flan_t5_answer_model` if exported.
- If you use this in Docker, add the chosen model to the Dockerfile snapshot download or copy a local model folder into `src/models`.

`benchmark_variants.py`

- Loads one manager file directly and times corpus loading plus QA.
- It also reports top-3 retrieval hit rate if your `nlp.jsonl` has `source_docs`.

Example:

```bash
python nlp/src/testing/benchmark_variants.py ^
  --manager nlp/src/testing/fast_bm25_manager.py ^
  --data /home/jupyter/novice/nlp ^
  --limit 256 ^
  --batch-size 32
```

## Useful Current-Manager Speed Profiles

These are for the existing `nlp_manager.py`, without swapping files:

```bash
# Max speed baseline: no model calls at inference.
set NLP_GENERATOR_ENABLED=0
set NLP_RERANKER_ENABLED=0
set NLP_EMBEDDING_ENABLED=0

# Middle ground: keep dense retrieval, no reranker/generator.
set NLP_GENERATOR_ENABLED=0
set NLP_RERANKER_ENABLED=0
set NLP_EMBEDDING_ENABLED=1

# Keep SmolLM3 but make it much cheaper.
set NLP_GENERATOR_DOCS_PER_QUESTION=1
set NLP_GENERATOR_DOC_CHUNK_LIMIT=1
set NLP_GENERATOR_MAX_CONTEXT_CHARS=900
set NLP_GENERATOR_MAX_NEW_TOKENS=16
set NLP_GENERATOR_BATCH_SIZE=16
set NLP_RERANK_TOP_K=4
```

On Workbench/Linux, use `export NAME=value` instead of `set NAME=value`.

