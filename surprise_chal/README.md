# TIL-26 Surprise

Everything you need to build, run, and submit your player for the competition.

## Overview

Each turn the competition server sends your server `POST /observe` with a JSON
**observation** and expects a JSON **action payload** back **within ~10 seconds**. Miss the deadline and that
turn is a no-op.

Your job: implement `decide(observation) -> ActionPayload`.

- The full observation and action formats are in [**`participant/RULES.md`**](https://github.com/til-ai/til-26-surprise/blob/main/participant/RULES.md) (see "Observation Payload" and "Action Reference"). Refer to the rules for full details of the game.
- **The `server/src/engine/` code is the source of truth** for every rule and number (constants in `server/src/engine/constants.py`). The canonical copy is the one the eval runs — **`server/src/engine/`**; `participant/src/engine/` is an identical read-only mirror, bundled so your submitted image is self-contained.

## How your agent gets run — three stages

Your `decide()` runs in three places. They share the same engine and rules; they differ in opponents, length, map, and limits. **Build for stage 1 — it is the conservative floor, and the real competition enforces the same limits.**

| Stage | How it runs | Opponents | Turns / map | Resource + egress limits |
|---|---|---|---|---|
| **1. Local self-test** | `docker compose up` (this kit) → PASS/FAIL + replay | 19 `RandomAgent` | 300 turns, seed `67` (**local stand-in**) | 1 CPU / 1 GiB · egress → `openrouter.ai:443` only |
| **2. Discord eval** | `./submit_surprise.sh` uploads your image; organisers run it on Vertex AI; result posts to Discord | 19 stronger algo bots | 50 turns, **hidden** seed | Local self-test is at least as restrictive as this |
| **3. Real competition** | organisers **pull your latest `TEAM_NAME-surprise:latest`** and run a full free-for-all | **the other teams' agents** | predetermined, **undisclosed** map | same as stage 1: 1 CPU / 1 GiB · `openrouter.ai:443` only |

- A **stage-1 PASS is the conservative signal**: stage 1 is at least as strict as the remote stages, so what passes locally will run remotely. (Stage 2 is *more* permissive on resources/network; **stage 3 matches stage 1**.)
- **Don't overfit to seed `67`** — it is a local stand-in only; stages 2 and 3 use undisclosed maps.
- Your `AGENT` + `OPENROUTER_API_KEY` must be **baked into the image** (the submit script does this) — no env is injected at stage 2 or 3.

## Layout

The kit ships in two self-contained, separately-built folders:

```
til-26-surprise/
├── docker-compose.yml          wires participant + harness for the local test
├── .env.example                copy to .env for the LLM template / compose
│
├── participant/                ◀── YOUR kit: edit, build, and SUBMIT this folder
│   ├── Dockerfile              builds your single-team server image
│   ├── requirements.txt        fastapi + uvicorn + httpx
│   ├── RULES.md                the rules (start here)
│   └── src/
│       ├── engine/             the game engine — READ ONLY (mirror of server's canonical copy)
│       ├── agent_base.py       PlayerAgent class shared by both templates
│       ├── server.py           your player server — Docker runs this (you can edit it but it works out of the box)
│       ├── llm.py              tiny OpenRouter helper (for the LLM template)
│       ├── algo_agent.py       TEMPLATE 1 — deterministic, no API key   ← edit one of
│       └── llm_agent.py        TEMPLATE 2 — bare-minimal LLM agent       ← these two
│
└── server/                     the eval harness + replay viewer (for local test)
    ├── Dockerfile              builds the harness image
    ├── requirements.txt        httpx only
    ├── requirements-viewer.txt EXTRA: arcade etc., only for watching replays
    └── src/
        ├── engine/             the game engine — the canonical source of truth
        ├── schemas/            observation/action contract
        ├── replay/             replay recorder + the interactive viewer
        ├── renderer/  sprites/ viewer rendering (arcade)
        ├── agent_base.py
        ├── baseline_random.py  the 19 opponents the harness fields (player-1..19)
        ├── game_runner.py      the competition turn loop (engine; don't edit)
        ├── eval_harness.py     engine + 19 random opponents, HTTP → you
        └── watch_replay.py     open a replay in the interactive viewer
```

You only ever edit files under `participant/`. The `server/` folder is the local
twin of the official eval (and the replay viewer) — it's there so you can test
and watch games, not for you to change.

## Write your own agent

Pick the template you want to build on (both live in `participant/src/`):

- **`algo_agent.py`** — deterministic, no API key. Edit this for a pure-code agent.
- **`llm_agent.py`** — LLM-driven. Edit this for an agent that calls a model.

Fill in the `decide(observation) -> ActionPayload` method in that file, building your
`ActionPayload` from the typed actions in `engine.actions`. Then run the test with Docker (see **Run the test** below). Your edits are picked up automatically; there's nothing else to wire up.

Two things to keep in mind:

- Invalid actions are **silently dropped** by the engine (out of range, unaffordable,
  wrong tile, …) — so you can be liberal; a bad action never crashes your turn.
- The **~10s deadline is hard** — exceeding it will result in no-op.

## Sandbox: resource & network limits (read this)

Your container runs in a locked-down local sandbox — `docker compose up` applies
it (the caps on the `participant` service + `egress/squid.conf`). It is
deliberately **at least as strict** as the remote stages (the **real competition
uses these exact limits**), so anything that passes here will run there. Build to these limits:

- **Network egress is allow-listed to `openrouter.ai:443` only.** Every other host
  and port is dropped. The `algo` agent needs no network; the `llm` agent may reach
  **only** OpenRouter. Don't depend on any other API, model host, or package index at
  runtime — the connection will just fail. (`httpx` honors the proxy automatically,
  so `llm_agent.py` needs no change to work behind it.)
- **CPU / RAM is capped:** ~**1 CPU / 1 GiB**. Exceed the RAM and your container is **OOM-killed** — you forfeit that turn (and likely the game). Keep your memory footprint small.
- **~10s per turn** is a hard deadline (above) — and since CPU is capped, heavy
  per-turn local compute can blow it. The model-generation latency for an LLM agent
  lives on OpenRouter, not your container.

## Run the test

Edited your agent? Run the whole match with **one command** from this directory.
The game engine and all 19 random opponents run in the `harness` container and call your
`participant` container over HTTP. You get
**PASS/FAIL** (did your Base survive to the turn limit?) plus a replay in `./replays/`.

> **The match config is FIXED — do not change it:** **`MAX_TURNS` `300`**, map
> **35×30**, and **19 random opponents**. **The map seed here is a stand-in (`67`),
> NOT the competition seed** — the real competition runs on a **predetermined map that
> is not disclosed**, so build an agent that *generalises* rather than one tuned to this
> exact map. A local PASS means your agent runs and survives on a representative map; it
> is not a guarantee on the hidden one. (`AGENT` just selects which of *your* agents
> runs; it isn't part of the match config.)

```bash
docker compose up --build
#   → prints PASS/FAIL; replay lands in ./replays/

# run the LLM agent instead of algo:
AGENT=llm OPENROUTER_API_KEY=sk-... docker compose up --build
```

### Watch the replay (optional)

To view a finished game in the graphical viewer (note that this will need to run on a compute with screen, e.g your own laptop, unless some hacky stuff is done to wire a screen tru ssh):

```bash
python -m venv env && . env/bin/activate && pip install -r server/requirements-viewer.txt   # one-time: arcade, pillow, …
python server/src/watch_replay.py                             # newest replay (searches ../replays/)
python server/src/watch_replay.py replays/<file>.jsonl  # or a specific one
```

Viewer controls: `Space` play/pause · `←`/`→` step · `+`/`-` speed · scroll zoom ·
drag pan · `F` fog perspective · `Tab` chat channel · `C` chat · `F11` fullscreen.

## Submission

Submit with the bundled **`submit_surprise.sh`** script. **Run it from your
Vertex AI Workbench notebook**, in this directory — the notebook is already
authenticated as your team's service account (the identity the evaluator
requires) and already has `TEAM_NAME` set, so you pass nothing:

```bash
chmod +777 submit_surprise.sh
./submit_surprise.sh
```

The script builds the `participant/` image, pushes it to your Artifact Registry
repo, and uploads it as a Vertex AI model — that upload is what triggers
evaluation. Your queue position and the PASS/FAIL result are posted to the
competition Discord. 

To submit the **LLM agent** instead of the default `algo` agent, give it your
key:

```bash
AGENT=llm OPENROUTER_API_KEY=sk-... ./submit_surprise.sh
```

**Notes**
- The discord evaluation runs for 50 steps, against better players. It will only run with **`MAX_TURNS` `50`** and let you know if you managed to survive till the end (returning errors incurred in the process). You can try to exfiltrate the discord evaluation seed, but rest assure that we will not reuse it for the real competition.
- Image used for real competition will be TEAM_NAME-surprise:latest, so please make sure you tag correctly. Also, the $AGENT and $OPENROUTER_API_KEY env should be bundled into your container submitted (the submit script does it for you already), since we will not be providing additional env var during the real competition.
- Your container must listen on **6700** and answer `GET /health` + `POST /observe` — the shipped templates already do, so don't edit `server.py`.
- The surprise eval allows outbound internet so the LLM agent can reach OpenRouter; every other task runs fully offline.
- The competition coordinates (region `asia-southeast1`, project `til-ai-2026`, your `TEAM_NAME`) are fixed and baked into the script — don't override them unless the organisers tell you to.
