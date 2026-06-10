"""Local eval harness — mirrors how the real competition scores you.

This is the participant-facing twin of the official SURPRISE eval job. ONE
process runs the whole thing: the real competition turn loop
(`game_runner.GameRunner`) plus the **19 opponents in-process** (they're
deterministic, so an in-process agent returns the byte-identical payload a
containerised one would over HTTP). The ONLY network call each turn is to YOUR
agent — over real HTTP `POST /observe`, exactly like the live event.

    you (player-0)  ◀──HTTP /observe──  harness {engine + player-1..19 in-process}

So you test the actual thing you submit: your HTTP server. Run your server
first (or use docker compose, which wires both together), then:

    PARTICIPANT_URL=http://localhost:6700 python eval_harness.py

Result is **pass/fail**: did your Base survive to the turn limit? It also writes
a replay you can watch.

The scenario is FIXED so local runs are reproducible. MAX_TURNS, map size, and
the 19-opponent format match the real competition — but the map SEED here is a
local STAND-IN (67), NOT the competition seed. The real competition runs on a
predetermined map that is NOT disclosed, so a local PASS shows your agent runs
and survives on a representative map; it is not the exact competition map. Build
an agent that generalises — do not overfit to seed 67.
    GAME_SEED = 67 (stand-in)   MAX_TURNS = 300   map = 35 x 30
The 19 opponents are always the deterministic RandomAgent baseline.
The only knob is:
    PARTICIPANT_URL   where your server is   (default http://localhost:6700)
"""

from __future__ import annotations

import asyncio
import os

import httpx

from engine.actions import ActionPayload, payload_from_dict
from game_runner import GameConfig, GameRunner, PlayerRegistration
from schemas.observation import build_observation

PARTICIPANT_ID = "player-0"
NUM_OPPONENTS = 19

# ── Fixed local scenario. MAX_TURNS / map / opponent count match the real eval;
#    the seed is a STAND-IN (the real competition map is predetermined & undisclosed).
GAME_SEED = 67  # local stand-in only — NOT the competition seed
MAX_TURNS = 300
MAP_WIDTH, MAP_HEIGHT = 35, 30
RESPONSE_TIMEOUT = 10.0  # per-turn budget for your server, same as the live event


def _make_opponent(pid: str):
    from baseline_random import RandomAgent

    return RandomAgent()


class HttpActor:
    """Drives the participant slot by calling its HTTP server. Returns None on
    any failure (timeout / transport / malformed) so the turn becomes a no-op —
    never raising into the loop. Mirrors the eval job's EndpointActor."""

    def __init__(self, url: str, timeout: float) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.calls = 0
        self.errors = 0

    async def decide(self, observation: dict) -> ActionPayload | None:
        self.calls += 1
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{self.url}/observe", json=observation, timeout=self.timeout
                )
                r.raise_for_status()
                return payload_from_dict(r.json())
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            print(f"participant endpoint error (call {self.calls}): {exc}")
            return None


class HarnessRunner(GameRunner):
    """Reuses the entire engine; overrides only the action-collection seam:
    player-0 → HTTP to the participant, players 1..19 → in-process opponents."""

    def __init__(self, registrations, config, actors: dict) -> None:
        super().__init__(registrations, config)
        self.actors = actors

    async def _collect_actions(self, player_urls):  # type: ignore[override]
        state = self.state
        alive = [pid for pid in player_urls if state.players[pid].alive]

        async def one(pid: str):
            obs = build_observation(
                state, pid, self.diplomacy, self.chat_log, self.config.max_turns
            )
            payload = await self.actors[pid].decide(obs)
            if payload is None:
                payload = ActionPayload(
                    player_id=pid, turn_number=state.turn_number, actions=[]
                )
            return pid, payload

        return dict(await asyncio.gather(*[one(pid) for pid in alive]))


async def _wait_healthy(url: str, timeout: float = 120.0) -> None:
    import time

    url = url.rstrip("/")
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                if (await client.get(f"{url}/health", timeout=3.0)).status_code == 200:
                    print(f"participant server healthy at {url}")
                    return
            except Exception:
                pass
            await asyncio.sleep(2.0)
    raise SystemExit(
        f"participant server at {url} never became healthy — is it running?\n"
        "Start it first:  (in src/) python server.py   — or use docker compose up."
    )


def main() -> None:
    # Only the participant URL is configurable; the scenario (seed / turns /
    # map / timeout) and the random opponents are fixed to match the official eval.
    url = os.environ.get("PARTICIPANT_URL", "http://localhost:6700")

    opp_ids = [f"player-{i}" for i in range(1, NUM_OPPONENTS + 1)]
    regs = [PlayerRegistration(PARTICIPANT_ID, PARTICIPANT_ID, "http://participant")]
    regs += [PlayerRegistration(pid, pid, "local://opponent") for pid in opp_ids]

    actor = HttpActor(url, RESPONSE_TIMEOUT)
    actors: dict = {PARTICIPANT_ID: actor}
    actors.update({pid: _make_opponent(pid) for pid in opp_ids})

    config = GameConfig(
        seed=GAME_SEED,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        max_turns=MAX_TURNS,
        response_timeout=RESPONSE_TIMEOUT,
    )

    print(
        f"harness: you=player-0 (HTTP {url}) vs {NUM_OPPONENTS} random opponents | "
        f"FIXED seed={GAME_SEED} {MAP_WIDTH}x{MAP_HEIGHT} turns={MAX_TURNS}"
    )
    asyncio.run(_wait_healthy(url))

    runner = HarnessRunner(regs, config, actors)
    runner.initialise()
    asyncio.run(runner.run())

    alive = bool(runner.state.players[PARTICIPANT_ID].alive)
    turns = runner.state.turn_number
    print("\n" + "=" * 56)
    print(f"  RESULT: {'PASS — you SURVIVED' if alive else 'FAIL — you were eliminated'}")
    print(f"  player-0 alive at end: {alive}  (reached turn {turns}/{config.max_turns})")
    print(f"  your endpoint: {actor.calls} calls, {actor.errors} errors/timeouts")
    print("=" * 56)


if __name__ == "__main__":
    main()
