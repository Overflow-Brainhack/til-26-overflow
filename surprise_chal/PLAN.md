# Surprise Challenge — Strategy & Implementation Plan

A 20-player free-for-all hex-grid economic wargame. We implement
`decide(observation) -> ActionPayload` in `participant/src/{algo_agent,llm_agent}.py`.
Server is `participant/src/server.py` (port 6700, `POST /observe`, **~10s hard turn
deadline** → miss = no-op). The engine under `participant/src/engine/` is READ-ONLY
truth. Build/run via `docker compose up --build`.

---

## The insight that changes everything: scoring is binary survival

The win condition is **not** domination. From `game_runner.py` and `RULES.md`: you win
by **either** destroying all enemy Bases **or** simply being alive at the turn limit —
and **every survivor co-wins equally, with no tiebreaker** on gold, kills, units, or
buildings. The local harness prints exactly `PASS — you SURVIVED` / `FAIL — you were
eliminated`.

So **kills are worthless** except to remove a direct threat to our own Bases. In a
20-player FFA the other 19 grind each other down; the optimal play is a **defensive
turtle that refuses to die**. This drives both agents.

### Survival levers (priority order)

1. **Base redundancy.** Eliminated only when you have **zero _complete_ Bases** (a Base
   that finishes construction *after* you die does **not** revive you). Holding 2–3
   complete Bases spread far apart means no single army can eliminate you. A 2nd Base
   (300g, 5 turns, foundable on any tile you can see) is the first major purchase after
   the opening economy.
2. **Universal peace.** An active peace treaty blocks all direct attacks (only artillery
   **splash** leaks through — it ignores ownership/treaty). Accept every incoming proposal.
   *Propose* selectively, not reflexively: a delivered proposal adds **you** to the target's
   `known_player_ids` (`turn_processor.py:411`), leaking your id/existence (not position) to a
   player you've merely spotted — a small tension with staying hidden. Accepting is free
   (they already know you); cold-proposing to everyone you glimpse is not. A treaty **break**
   triggers a 5-turn ACTIVE countdown (`breaking_in_turns`) — a free early-warning siren to
   reinforce before war resumes.
3. **Per-Base denial rings (the real Bomber defense).** Keep all 6 neighbour tiles of every
   complete Base **permanently occupied by our own units** — air units are blocked from
   occupied tiles, so a full ring denies a Bomber the adjacency it needs to attack (kill math
   below). Fill with cheap units (Infantry eat 50/hit vs 200 for a building), replace losses
   promptly, and keep local ranged punishment (Tanks 200hp/60atk, Artillery range 3 + splash,
   Fighters for anti-air) nearby to one-turn-kill any Bomber that takes a gap. Treat Fighters
   as **control/punishment, not a shield** — and teleport as a tool to **rebuild/pre-position**
   rings, never to intercept the current turn's attack.
4. **Economy** to fund the above — Mines, especially on rich-resource tiles (50g/turn).
5. **Stay hidden.** Fog is strict with no memory; "met by sight" is one-directional.
   Don't over-extend and paint a target on yourself.

### The base-kill math (why tile-denial, not a teleport reserve, is the real defense)

From `constants.py` / `bomber.py` and the turn order in `turn_processor.py`, the numbers
and timing that decide the game:

- **Base = 300 HP. Bomber = 50 atk ×4 vs buildings = 200/hit** → **2 hits kill a Base; 2
  Bombers kill it in one turn (400).** Bombers fly (ignore elevation + concealment) and with
  the teleport quirk relocate anywhere free in a turn.
- **Combat is simultaneous and resolves *before* movement.** `_phase1_units` computes all
  damage, applies it, and removes the dead — *then* moves execute (`turn_processor.py:74-213`).
  Attacks also fire from the **pre-move** tile. Two consequences, both load-bearing:
  - **Killing a Bomber the turn it strikes does not stop its strike** — the 200 lands anyway.
  - **A reactively teleported reserve cannot intercept this turn's attack** — it was elsewhere
    pre-move, so it can't fire this turn. **Teleport is a pre-positioning / rebuild tool, not
    an interception tool.**
- **So the only things that actually stop a Bomber are:** (a) **denying it an adjacent
  tile** — Bombers attack at range 1 (the 6 neighbours) and **air units are still blocked
  from occupied tiles** (`is_ground_blocked` counts every entity; `state.py:100`), so a full
  6-unit ring means it physically cannot reach the Base; (b) **base redundancy** so losing
  one Base isn't fatal; (c) **staying hidden** so it never finds the Base to target. A unit
  on a ring tile only eats 50/hit (vs 200 for a building), so **fill rings with units, not
  buildings.**
- **Fighters are punishment/control, not a shield.** 250 HP, atk 50, **range 2** (out-ranges
  Bombers), flies. One-turn-killing a 150-HP Bomber needs **≥150 damage**: 3 Fighters, or
  2 Fighters + Tank, or Fighter + 2 Tanks, or Fighter + Tank + Artillery. A pre-positioned
  garrison that kills an intruding Bomber the next turn **caps damage to a single volley**
  (one 200 hit, not two) — the ring's job is to ensure **≤1 Bomber is ever adjacent at once**.
- **Vision for early warning:** only **complete Bases** self-spot (`vision_bonus=3`) — a
  3-tile ring incl. enemy HP. **An incomplete Base, and every Mine/Barracks/Factory/Airbase,
  gives zero vision** — blind spots until a Base or unit covers them. Warning is short
  (~1 turn from a teleport-in), which is *why* the ring must be **standing**, not scrambled.

### Engine caveat: treaty cutoff at turn 200

`TREATY_CUTOFF_TURN = 200` voids **all** treaties and forbids new ones — forced open war
for the rest of the game. Only bites if the real game runs >200 turns (local = 300,
Discord eval = 50, real length undisclosed), so the defensive army must be real, not
just diplomatic.

### Stage map (what we're actually graded on)

| Stage | Opponents | Turns / map | Notes |
|---|---|---|---|
| 1. Local self-test | 19 `RandomAgent` (weak) | 300, seed 67 (stand-in) | PASS = Base alive at end. Don't overfit to seed 67. |
| 2. Discord eval | 19 stronger algo bots | 50, hidden seed | Treaty cutoff never hits; survive-early is the whole game. |
| 3. Real competition | other teams' agents | undisclosed map/length | The real test — genuine defense + diplomacy needed. |

---

## Engine quirks that beat the rulebook (verified against the real engine)

`README.md` is explicit: **the engine code is the source of truth; discrepancies are doc
bugs to be fixed against the code, never the reverse.** So these code behaviours are
authoritative (and unlikely to be patched). Confirmed empirically with `_verify_mechanics.py`
against `participant/src/engine` (byte-identical to the eval's `server/src/engine`). Note
that `ActionValidator` exists but is **never called** — `turn_processor` is the sole
authority and is more permissive.

1. **Units can teleport (no path-adjacency check).** `turn_processor` validates a move by
   only: `path[0]` == current tile, `len(path)-1 ≤ movement_range`, and Σ(entry-cost of the
   *listed* tiles) ≤ `movement_range`. Consecutive waypoints need **not** be adjacent, so a
   single hop `[(0,0)→(15,7)]` legally moves an Infantry 20 tiles. **Every unit (move ≥ 1)
   can relocate to almost any free tile in one turn** (landing *on* a Difficult tile still
   needs move ≥ 2; you teleport *over* walls/units/terrain freely). Attacks still fire from
   the **pre-move** tile, so no teleport-then-snipe in one turn — it's pure repositioning.
   - *Use it:* one mobile reserve defends every base; flash-scout to found hidden remote
     Bases; instant-wall a base with 6 infantry; yank fragile units out of danger.
   - *Defend it:* assume strong stage-3 opponents teleport too (a Bomber can appear next to
     our Base) → redundant/hidden Bases + a teleport-capable anti-air reserve.
   - *Risk:* clearly an unintended bug. Design to **benefit** from it without **depending**
     on it, in case it's ever closed.

2. **Breaking a treaty ends protection immediately.** `is_peace` is `True` only for `ACTIVE`
   status; `break_treaty` flips to `BREAKING`, so a partner can be attacked the **next turn**
   while the victim's observation still shows the treaty with `breaking_in_turns` > 0 and the
   system DM claims "5 turns until war" (the count even ticks 5→4 the same turn).
   - *Defend it:* treat any incoming break as **war this turn**, not in N turns.
   - *Use it (later):* break, then strike next turn — beats any rules-faithful opponent.

3. **No same-turn follow/swap moves.** Collision is checked against current occupancy before
   any move executes, so a unit can't enter a tile a teammate is vacating that turn (nor can
   two units swap). Minor; teleport sidesteps it.

4. **Artillery splash bypasses peace** (documented, but load-bearing). The treaty/ownership
   check is only on the *primary* tile; splash hits everyone in the ring. Firing at an empty
   tile next to a peace partner's Base splashes it for `int(60×0.5)=30` with **no** treaty
   break. Peace does not fully protect our buildings from an artillery-splasher — defensive
   spacing matters, and it's an offensive lever that doesn't cost us the treaty.

---

## Persistent world model (AE-style) — memory is mandatory

The observation is **stateless with zero fog memory**: a tile that leaves vision vanishes
next turn, and the obs carries **no production queues** (not even our own). So, exactly like
the AE agent's `agent/world/planner` split, we carry our own state across turns. The
asymmetry that makes this cheap: **terrain never changes mid-game**, so every tile we ever
see is permanently valid.

- **`WorldModel`** (persists across turns):
  - **Map memory** — accumulate terrain as we explore; once seen, it's forever-valid.
  - **Last-known enemies** — `{id → (q,r,type,hp,last_seen_turn)}` with staleness; drives
    threat assessment and the Hunter target list even when a base drops back into fog.
  - **Own production orders** — *we* must remember what we queued at each Barracks/Factory/
    Airbase, because the obs won't tell us. **Track conservatively:** only record an order
    after our own affordability/validity check says the engine should accept it (cost
    affordable, building complete & able to produce, target within range), so silent no-ops
    don't desync the model. Reconcile against produced-unit sightings when possible.
  - **Inferred diplomacy graph** — who's allied with whom, deduced from chat + observed
    non-aggression (we only ever see our *own* treaties). **Keep this weak:** third-party
    non-aggression is noisy under fog, so it should inform *suspicion*, never drive hard
    commitments.
- **`Planner`** — pure functions over the WorldModel → a stance (turtle / reinforce / hunt).
- **`Actuator`** — turns stance into a validated `ActionPayload`. **All** hex/range math
  lives here, deterministic; the LLM never touches it.

### Offense as defense — the Hunter mode

We field **one agent in one seat**, so this is a *behavior mode*, not a second bot. Binary
survival says a rampage is a bad *primary* plan (sole-victor == co-victor, but dying
mid-rampage == loss — asymmetric downside). **But** the Bomber+teleport kit makes a kill
cheap, and **permanently eliminating a player removes a base-sniping threat**, so in a
hostile meta offense *is* defense. The planner flips to **Hunter** only on triggers:

- a player is attacking us and won't take peace → decapitate them (kill their Bases);
- a spotted enemy Base is undefended and we have spare Bomber capacity → snipe to thin the field;
- **post-turn-200 forced war** → pre-empt the nearest strong neighbour.

Separately, an **always-on roaming Scout finder** teleport-explores to fill the WorldModel
with enemy Base locations — pure intel upside, no commitment.

## Archetype roster (selectable modes *and* sparring opponents)

The same behaviour set serves two jobs: a stance our planner can switch into, and an
opponent model to harden against. Each archetype is deliberately built around one threat so
the evaluator can stress a specific defence. Implemented as lightweight bots in
`seed_eval.py` (sparring partners — simple but functional, **not** the full Phase-1 agent):

| Archetype | Plays | Exercises in us | Engine lever it leans on |
|---|---|---|---|
| **Turtle** | economy → denial rings → 2nd Base → universal peace | our baseline-to-beat | redundancy, peace, ring denial |
| **Aggressor** | rushes Airbase → Bomber, teleports adjacent to the nearest enemy Base and bombs it | **denial rings + redundancy** | teleport + Bomber ×4 vs buildings |
| **Treaty Ambusher** | proposes peace, lulls N turns, then **breaks + strikes the same turn** | "treat any incoming break as **war this turn**" | breaking-treaty ends protection immediately |
| **Splash Raider** | makes peace, then artillery-splashes a peace partner's Base from an *empty* adjacent tile — chipping it **without** breaking peace | defensive **spacing** around Bases | artillery splash bypasses the peace check |
| **Economist** | pure economy + Base-spam, minimal military | a passive survival/redundancy baseline | none (control group) |

(`RandomAgent`, the official stage-1 baseline, is the weakest filler.) The first three map
directly onto modes our own agent can adopt: Turtle is the default, **Aggressor == Hunter**,
and Ambusher/Splash-Raider are post-cutoff or "they-struck-first" escalations. Validate every
defensive change by re-running the evaluator with the matching aggressor in the field.

## Architecture: algo is the engine, LLM is a thin overlay

The **LLM agent contains the algo agent** rather than being an alternative to it:

- **Every turn the algo runs first** (synchronous, ~ms) and produces a complete, valid
  `ActionPayload`: all unit moves, attacks, builds, production, and a sensible *default*
  diplomacy (accept-all / propose-to-met). This is the safe baseline **and** the fallback.
- **The LLM is consulted only for the soft layer** it's good at — chat, treaty
  decisions, and maybe a single defensive/expand mode toggle — with a tight timeout. We
  **never** ask it to do hex-range math (LLMs are bad at it; the template warns so), and
  we only accept `send_chat` / treaty action types back from it.
- **Graceful merge or fallback:** LLM replies in time → splice its diplomacy/chat over
  the algo defaults. LLM times out or errors → ship the algo payload untouched.

**Why this beats letting the LLM drive:** the 10s deadline is hard and CPU is capped;
LLMs are unreliable at the grid math that keeps you alive; tokens cost money over
hundreds of turns; the **uncapped** chat history is a deliberate context-bomb vector.
Putting survival on deterministic rails and using the LLM for negotiation captures the
upside (smart diplomacy, reading opponents) with none of the downside (a timeout/garbage
turn losing the game). It also makes model latency **non-critical** — a slow turn just
falls back — which de-risks the ≤7s requirement.

---

## Implementation phases

### Phase 1 — the strong algo turtle (can PASS on its own)

Structured as `world.py` / `planner.py` / `actuator.py` (AE-style), with the `WorldModel`
persisting across turns:

- **`world.py`** — parse the obs into the WorldModel and update it: own units/buildings,
  visible enemies (refresh last-known + staleness), terrain memory, our own production
  orders, treaties + incoming breaks, inferred diplomacy graph. Torus-correct via the
  engine's `HexGrid`.
- **`planner.py`** — pick a stance per the levers. Economy/build follows the opening script
  below, then: ramp Mines (prefer rich tiles) → found a **2nd Base early** for redundancy →
  scale Base count with the clock. **Air timing is trigger-based, not fixed:** rush an Airbase
  only when an air threat is observed/likely (enemy Airbase or Fighters/Bombers seen, or a
  strong-bot stage); otherwise take a **Factory for Tanks** for cheaper ground stability and
  ring-fill muscle. Decide when to flip **Hunter** (triggers above) vs stay turtle. Never box
  in a producing building (lost unit + gold).
- **`actuator.py`** — emit the validated `ActionPayload`. All hex/range math here:
  - **Standing per-Base denial rings:** keep all 6 neighbours of every complete Base occupied
    by our own units (cheapest available, Infantry first); on a death, refill the gap next turn.
    This is the primary Bomber defense — it physically blocks adjacency. Buildings adjacent to
    the Base also deny their tile but soak 200/hit, so prefer units.
  - Local punishment garrison near each Base sized to **one-turn-kill a Bomber (≥150 dmg:
    3 Fighters / 2F+Tank / F+2Tank / F+Tank+Artillery)** so any Bomber that takes a gap eats
    only one volley. Focus-fire threats in range (**Bombers first**, then Artillery); artillery
    splash with friendly-fire avoidance (and as a peace-proof chip lever); retreat fragile
    units (Scout/Artillery/Medic); medic healing.
  - Mobility via teleport — **pre-positioning/rebuild only, never interception** (attacks
    resolve before moves): shuffle reserves to refill ring gaps, pre-stack a Base on the 5-turn
    treaty-break warning, flash a Scout to reveal a remote tile and found a hidden redundancy
    Base. Keep an adjacent-only fallback in case the quirk is patched.
  - Default diplomacy: accept all incoming peace; propose **selectively** (mind the
    known-players leak); on any incoming treaty-break, treat it as **war this turn**
    (protection is already gone) and pre-position now.
- Stay within 1 CPU / 1 GiB and well under the deadline (bounded per-turn compute, no
  whole-map pathfinding every turn).

#### Opening script (first ~20 turns, budgeted)

Start state (verified `server/src/game_runner.py:88-109`): **500 gold, exactly one Base,
zero units.** The Base **cannot produce units** (no `producible_unit_types`) — it only yields
+10/turn (+50 on a rich tile) and gives vision 3. Non-Base buildings must be placed on an
**empty tile adjacent (≤1) to a *completed* own building** (`turn_processor.py:307-327`), so
the opening necessarily clusters around the Base. Multiple builds/produces per turn are legal
if each is individually affordable.

- **T0 (500g):** Construct **Barracks (100)** + **Mine (200)** on empty tiles adjacent to the
  Base (Mine on a rich neighbour if one is visible → 50/turn). → ~200g left, both complete ~T2.
- **T1–T2:** Income only (~+10–60/turn); save. When the Barracks completes (~T2), start
  **Infantry (50g, 1 turn each)** to seed the denial ring — first unit lands ~T3.
- **T3–T8:** Pump Infantry to fill the Base's 6-neighbour ring (≈300g spread over several
  turns as income allows). Build a **Scout (100)** for intel, first contact (accept-peace),
  and to reveal a defensible/hidden **2nd-Base site**. Add a 2nd Mine if an adjacent slot +
  gold exist.
- **T8–T15:** Once the ring is seeded and income flows, bank for and found a **2nd Base
  (300g, 5 turns)** on the Scout-revealed tile — redundancy is the top survival lever.
- **T15–T20:** Begin its denial ring; take **Factory (300) for Tanks** as default ground
  punishment, or **rush Airbase (500) for Fighters** *only* on the air-threat trigger.

Numbers are a default cadence, not a fixed track — the planner re-derives affordability each
turn and skips/reorders on income, terrain, and observed threats.

### Phase 2 — the LLM overlay

- Hybrid wrapper around the algo (compute-then-consult-then-merge/fallback).
- Bounded prompt builder: hard-truncate the uncapped global/private chat and obs to a
  fixed token budget (defuses context-bombs, controls latency/cost).
- Strict-JSON soft-layer parser; accept only chat + treaty actions from the model.
- Live latency benchmark of 2–3 OpenRouter models (Fable 5 + a fast flash-class model)
  using the provided key, to confirm the ≤7s budget empirically. Key lives in a
  gitignored `.env`; baked into the image at submit time.

### Phase 3 — test & iterate

- `docker compose up --build` for both agents; verify survival on seed 67.
- Harden against the *stronger* dynamics expected in stages 2–3 (not overfitting to the
  random opponents).

### Deferred — opponent-LLM exploitation

The "exploit opponents' LLMs via prompt-injection / feints / chat manipulation" layer is
**parked for Opus 4.8**, to be done after the core agents are solid. It slots cleanly
into the chat layer Phase 2 already builds.

---

## Build order

Start with **Phase 1 (the algo turtle)** — it's the robust floor everything builds on and
the fallback for the LLM agent. Optionally run the model-latency benchmark first if we
want the LLM model choice settled before coding.
