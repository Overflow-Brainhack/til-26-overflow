# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a competitor repo for DSTA BrainHack TIL-AI 2026. Five independent ML challenges — `asr/`, `cv/`, `nlp/`, `ae/`, `noise/` — are each built into their own Docker image and submitted separately. There is no single application: the repo is five sibling microservices that share a project skeleton plus a `test/` harness that probes each service over HTTP.

## Per-challenge architecture (the shared pattern)

Every challenge directory has the same shape; understanding one means understanding all five:

- `<challenge>/src/<challenge>_server.py` — thin FastAPI wrapper. **Do not modify** unless the server contract genuinely needs to change. It exposes `POST /<challenge>`, `GET /health`, and (AE only) `POST /reset`.
- `<challenge>/src/<challenge>_manager.py` — the `*Manager` class is where model loading and inference go. This is the file you edit to implement a challenge.
- `<challenge>/Dockerfile` — builds the image. Most use `nvcr.io/nvidia/pytorch:25.11-py3`; **AE uses `python:3.11-slim` (CPU only)** because AE agents are not expected to need GPU and a smaller image speeds up evaluation.
- `<challenge>/requirements.txt` — deps bundled **into the Docker image**. Distinct from the repo-root `requirements.txt` / `requirements-dev.txt`, which are local-dev only. **Adding a runtime dep means editing the per-challenge file**, otherwise it won't be present at evaluation time.

Server-side I/O contract (consistent across all challenges):
- Request body: `{"instances": [...]}`. Most challenges base64-encode bytes in `b64` keys; AE sends a structured `observation` dict; NLP sends `question` strings.
- Response body: `{"predictions": [...]}`. Order must match `instances` index-for-index.
- Fixed ports: ASR `5001`, CV `5002`, noise `5003`, NLP `5004`, AE `5005`. The local test scripts hardcode `http://localhost:<port>`.

Stateful behaviors to be careful about:
- **NLP** — the *first* request carries `documents` (the RAG corpus). The server checks `manager.loaded` and dispatches a one-shot `load_corpus` call before any QA traffic. Don't break that branch in [nlp/src/nlp_server.py](nlp/src/nlp_server.py).
- **AE** — the Docker container is **not** restarted between rounds during Qualifiers. Per-round reset happens two ways: via `POST /reset`, or implicitly when an inbound observation has `step == 0` (the server then re-instantiates `AEManager`). Don't store round-persistent state outside the manager instance, and if you must, clear it in `/reset` too.

## Build / test / submit

The competition GCP Workbench instance ships a `til` CLI that wraps Docker:

```bash
til build <challenge> [tag]    # docker build -t TEAM_ID-<challenge>:<tag>
til test  <challenge> [tag]    # runs container on offline net + test/test_<challenge>.py
til submit <challenge> [tag]   # uploads for evaluation
```

For local dev outside the Workbench (e.g. this devcontainer), do the same manually: `cd <challenge> && docker build -t <name> . && docker run -p <port>:<port> <name>`, then run the matching `test/test_<challenge>.py` against it.

## Local testing harness (`test/`)

Every `test/test_<challenge>.py` is a runnable script that:
1. Reads `TEAM_NAME` and `TEAM_TRACK` from `.env` via `python-dotenv`.
2. Loads ground-truth data from `/home/jupyter/{TEAM_TRACK}/<challenge>` (track is `novice` or `advanced`).
3. POSTs batched requests to the *already-running* server on `localhost:<port>`.
4. Scores predictions and writes results to `/home/jupyter/{TEAM_NAME}/`.

Run a single test: `python test/test_asr.py` (and equivalents). The container/server must already be up on the right port — the scripts do not start it. AE's test additionally requires the `til_environment` package from the `til-26-ae` submodule.

The noise challenge has its own scoring sub-package at [test/noise_eval/](test/noise_eval/) (`pipeline.py`, `metrics.py`, `fairness_checker.py`, `eval_thresholds_v2.yaml`) — image-level fairness is gated by those thresholds.

## Dependency layout — read this before adding a package

Three separate dependency surfaces:

- **`pyproject.toml` + `uv.lock`** — declares the local-dev environment, including a `pytorch-cu118` index pin. Use `uv sync` if you use uv.
- **`requirements-dev.txt`** — local-only training/testing deps; includes `-e ./til-26-ae` (the editable `til_environment` install).
- **`<challenge>/requirements.txt`** — what ends up in each Docker image. Today these only pin `fastapi`, `uvicorn`, `gunicorn`. **Anything your manager imports (torch, transformers, etc.) must be added here** or the image build will succeed but inference will crash at runtime.

## Submodules

- `til-26-ae` — provides `til_environment` (the AE simulator). Initialize with `git submodule update --init`. Don't modify it or `.gitmodules`.
- `til-26-finals` — referenced in the README but currently commented out in `.gitmodules`; will be activated for Semifinals/Finals.

## Python version

`pyproject.toml` requires `>=3.12`; `.python-version` pins `3.13`. The competition Workbench supports `3.10+`. Match the local-dev version to whatever the active venv/conda env was created with.

## Things to avoid

- Editing `til-26-finals/`, `til-26-ae/`, or `.gitmodules` (per repo rules).
- Treating the five challenges as a single service — they ship and version independently.
- Adding deps only to root `requirements.txt`; the Docker image won't see them.
- Forgetting that AE's `/reset` and `step==0` paths must keep the manager fully clean for the next round.
