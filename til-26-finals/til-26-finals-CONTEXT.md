# `til-26-finals/` — Context

The Finals submodule for BrainHack TIL-AI 2026. Where Qualifiers shipped five
*independent* task containers (`asr`, `cv`, `nlp`, `ae`, `noise`), Finals wires
them into **one orchestrated stack** driven by a live Bomberman match. This
folder contains (a) the **participant orchestration server** you submit, and
(b) a **local copy of the competition server** so you can test end-to-end.

> This directory is its own git repo (a submodule of `til-26-overflow`). Per the
> root `CLAUDE.md`, don't edit the submodule's tracked contents or `.gitmodules`.
> This context file lives in the parent repo, not inside the submodule.

---

## The two servers (don't confuse them)

- **Finals Server** (a.k.a. participant / team server) — **what YOU build and
  submit.** Lives in [`finals/`](til-26-finals/finals/). Receives match traffic
  from the competition server over a WebSocket and fans it out to your five
  model containers over HTTP. This is the only image that's genuinely *yours* to
  implement here.
- **Competition Server** — **runs the Bomberman environment.** At MBS the real
  one runs it; locally, the copy in
  [`test_competition_server/`](til-26-finals/test_competition_server/) lets you
  drive a full match offline. You generally don't edit this — it's a faithful
  stand-in for grading.

At the real finals (MBS, June 10–11 2026) all six containers run on one 6-way
Desktop in a single Docker Compose stack — **note the hardware differs from GCP**
(Blackwell 5070 Ti locally vs. Turing T4 on GCP; CUDA differences can be
catastrophic if untested).

---

## Finals Server — the part you implement

Two files, both in [`finals/src/`](til-26-finals/finals/src/):

- **`participant_server.py`** — connects to
  `ws://$COMPETITION_SERVER_IP:$COMPETITION_SERVER_PORT/ws/$TEAM_NAME`, also
  serves `GET /health` on **port 5000** (returns the bool `true` only once the WS
  is live — Compose healthchecks gate on this). Message types it handles:
  - `type=task, task=ae` — per-step Bomberman observation → reply one action.
  - `type=mission_batch` (and `type=noise`) — a batch of N items of **one** task
    type (asr/cv/nlp/noise) → run the model, reply all N results.
  - `type=corpus` — one-shot RAG corpus broadcast → ingest into NLP, then ack.
  - `type=done` — match over; drain in-flight tasks and close.
  - `type=health` — reply `health_ack`.
  Each message is dispatched to its own `asyncio` task so batches/AE run
  concurrently. Reconnects automatically via `websockets.connect` async-for.
- **`models_manager.py`** — `ModelsManager`, the HTTP client to the five
  containers. **Port map: ASR `5001`, CV `5002`, noise `5003`, NLP `5004`, AE
  `5005`** (same fixed ports as Qualifiers). Each `run_*_batch` re-shapes items
  into `{"instances":[...]}`, POSTs, and pairs `predictions` back to `task_id`s.
  `ingest_corpus` POSTs the corpus then polls `:5004/nlp` with `{"poll":"true"}`
  until status is `loaded`/`error`.

**Wire ↔ container contract** (what the orchestrator translates between):
| Task  | Container request (per item)        | Container response (per item)                  |
|-------|-------------------------------------|------------------------------------------------|
| ASR   | `{"b64": ...}`                      | answer string                                  |
| CV    | `{"b64": ...}`                      | `[{bbox, category_id}, ...]`                   |
| noise | `{"key": task_id, "b64": ...}`      | noised `b64` (nulls dropped)                   |
| NLP   | `{"question": str}`                 | `{"answer": str, "documents": [str,...]}`      |
| AE    | `{"observation": {...}}`            | `{"action": int}`                              |

Build: [`finals/Dockerfile`](til-26-finals/finals/Dockerfile) — `python:3.13-slim`,
CPU-only (it's just an HTTP/WS router; the models live in the other containers).
Deps in [`finals/requirements.txt`](til-26-finals/finals/requirements.txt):
`websockets, asyncio, httpx, fastapi, uvicorn`.

---

## Competition Server (test copy) — how a match actually scores

Lives in [`test_competition_server/src/`](til-26-finals/test_competition_server/src/).
FastAPI thin transport + a `MatchCoordinator` delegating to phase objects.

**Match lifecycle** (`server.py` `POST /start` → `match.py`):
1. **Phase 1 — corpus** (`phases/corpus.py`): broadcast RAG docs, wait for acks
   up to `CORPUS_INGEST_DEADLINE_SEC` (60s). Unconnected teams pre-acked.
2. **Phase 2 — noise** (`phases/noise/`): distribute CV images to be de-noised,
   collect, **fairness-check** (image-level thresholds), build `noised_lookup`.
3. **Phase 3 — NLP pre-warm**: load the HF answer-equivalence eval model.
4. **AE loop** (`phases/ae_loop.py`): PettingZoo Bomberman, stepped once/sec.
   Each step sends observations, waits `AE_TIME_CUTOFF` (2s) for actions
   (slow/invalid → `Action.STAY`), advances env, and on `add_mission` info flags
   **enqueues a mission** for that team.

**Missions & scoring** (`missions.py` + `scoring.py`):
- One mission tile = **3 batches queued: ASR → CV → NLP**, each
  `MISSION_BATCH_SIZE` (4) items. `MissionQueue` drains **per-team FIFO, one
  batch in flight per team**; teams are supervised independently (one team's
  crash never affects others).
- Batch timeout `MISSION_BATCH_TIMEOUT_SEC` (10s) → batch abandoned, scores 0.
- `batch_score = 0.75·accuracy + 0.25·time_score`, where
  `time_score = 1 − min(elapsed, 5s)/5s` (`MAX_TIME_PER_TEST_CASE`). Accuracy:
  ASR = `max(1−WER/CER,0)` (jiwer; char-level for Chinese), CV = COCO
  mAP@.5:.05:.95, NLP = HF answer-equivalence (threshold 0.9).
- **Final team score = AE reward × mission_multiplier**, where
  `mission_multiplier` = mean of that team's batch scores. See
  `MatchCoordinator.get_scores()`.
- All untrusted team replies are validated defensively before scoring (malformed
  → item scored 0, never crashes the batch).

Tunables live in [`constants.py`](til-26-finals/test_competition_server/src/constants.py).
Match config (teams/track/stage/seed) in
[`configs/config_test.json`](til-26-finals/test_competition_server/configs/config_test.json);
loaded by `config.py`, which **auto-seats `$TEAM_NAME` at slot 0** if absent so
you can test under your own name without editing configs.

Other notable modules: `env_state.py` (PettingZoo env wrapper), `transport.py`
(`WebSocketManager`), `artifacts.py` (event/results JSONL + match dir),
`render_match_video.py` (writes `match.mp4`), `nlp_eval.py` (HF evaluator),
`noise_eval/` (fairness pipeline + `eval_thresholds_v2.yaml`).

---

## Local stubs (filling the other 5 seats)

[`test/stub_participants.py`](til-26-finals/test/stub_participants.py) — one
process opening a WS per non-real team. Returns random valid AE actions
(respects `action_mask`), dummy empty ASR/CV/NLP results, passes images through
(or all-black for `BLACK_IMAGE_TEAMS`) for noise, and acks corpus. Mirrors the
HQ's slot-0 seating so it never spawns the displaced team. `AEManager` in
[`test/ae_manager.py`](til-26-finals/test/ae_manager.py).

---

## Build / test / submit (`finals.sh`)

```bash
bash finals.sh build  finals [--build_all]   # build {TEAM_NAME}-server:finals (+ all task imgs)
bash finals.sh submit finals [--submit_all]  # push server to Artifact Registry (+ 5 task imgs)
bash finals.sh test                          # full local end-to-end match
```
- The tag **must be `finals`** — only `finals`-tagged images are pulled.
- `submit finals` builds `./finals/` → `{TEAM_NAME}-server:finals`, pushes to
  `repo-til-26-{TEAM_NAME}`. `--submit_all` also builds/pushes the five task
  images from your `til-26` repo (`$TIL_FOLDER/<task>/Dockerfile`).
- Per-task qualifier commands (`finals.sh submit asr`, etc.) still work.
- **Custom compose**: drop a `custom-compose.yml` at the root of your GCS bucket
  (`/home/jupyter/{TEAM_NAME}`) to run a non-standard stack (e.g. 4 containers
  sharing a model). Keep the `.env` line. Template:
  [`custom-compose.yml`](til-26-finals/custom-compose.yml).

**`finals.sh test`** drives
[`docker-compose-test.yml`](til-26-finals/docker-compose-test.yml):
`til-competition-test` (HQ + data + configs mounted) ← `til-stubs` (other seats)
+ `til-finals` (yours, `network_mode: host`) → `til-{asr,cv,noise,nlp,ae}` (GPU,
ports 5001–5005) + `til-starter` (one-shot: waits for health, `POST /start`,
polls `/match_status` until `ended:true`). Needs locally-built
`{TEAM_NAME}-{task}:finals` images first.

---

## Setup checklist

1. `git submodule update --init` (pulls `til-26-ae` → `til_environment`, inside
   `test_competition_server/`).
2. `cp .env.example .env`; set `TEAM_NAME`, `CONFIG`, `HOST_DATA_DIR`,
   `COMPETITION_SERVER_PORT`. `mkdir -p artifacts`.
3. On GCP, ensure `$HOME/$TRACK` (novice|advanced) data is mounted. Expected data
   layout: `asr/<track>/`, `cv/<track>/`, `nlp/<track>.jsonl`, `nlp/documents/`,
   `nlp/models/nlp_eval_512/`.

Key env vars: `COMPETITION_SERVER_IP`/`LOCAL_IP` default to
`host.docker.internal` (til-finals runs on host network, reaches models via
exposed ports). See [`.env.example`](til-26-finals/.env.example).

---

## Gotchas

- The orchestrator is **CPU-only and stateless** — all heavy lifting is in the
  five model containers; don't put model code here.
- Dependencies that the model containers import still belong in **each task's own
  `requirements.txt`** (in the `til-26` repo), not here.
- `corpus` is one-shot and arrives **before** QA traffic — the orchestrator must
  ingest + ack it, then poll NLP until loaded. Don't break that path.
- AE is **per-step synchronous-ish**: a missing/slow/invalid action defaults to
  `STAY`, so latency directly costs moves.
- Reply to **every** batch even on error (the orchestrator sends empty results on
  exception) so the HQ's 10s `wait_for` resolves cleanly.
- Wire format ≠ container format — `ModelsManager` is the only translation layer;
  keep it in sync with the contract table above.
