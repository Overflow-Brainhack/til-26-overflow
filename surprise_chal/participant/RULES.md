# TIL-26 Surprise — Game Rules

A multiplayer free-for-all economic strategy wargame. AI agents compete on a hex grid to be the last player standing.

> **The engine code is the source of truth.** This document is a faithful, best-effort abstraction of the engine for human and LLM readers — not the specification itself. Where this document and the implementation ever disagree, **the code wins** (`src/engine/`, with all balance numbers in `src/engine/constants.py`). Treat any such discrepancy as a documentation bug to be fixed against the code, never the other way round.

---

## Core Invariants

These are the load-bearing rules the whole engine guarantees. Everything later in this document **defers to them** rather than restating them — when a detailed rule and an invariant seem to disagree, the invariant wins (and where the document and the **engine code** disagree, the code wins — see the note above). Each is enforced in code and covered by tests.

1. **One entity per tile.** At most **one** entity — a single unit *or* a single building — occupies any tile at any moment. Nothing stacks: not two units, not air-over-ground, not a unit on a building. Any "can two things share a tile?" question answers **no**. (This is why, e.g., an attack's target tile always holds either nothing or exactly one entity.)
2. **Simultaneous, deterministic resolution.** All players submit their actions at once; the server then resolves them in a fixed phase order — **units → buildings → coordination** — identically every time. There is no first-mover or player-index advantage; symmetric conflicts resolve symmetrically.
3. **Invalid actions are silent no-ops.** Anything illegal — out of range, unaffordable, wrong owner, blocked/occupied tile, unmet recipient, malformed input — is **dropped without error or penalty**. One bad action never aborts your other actions, and no single player's input can crash a turn.
4. **Gold is spent up front; no debt.** A cost is deducted the moment an action is accepted; balances can never go negative; anything you cannot afford is simply dropped (per invariant 3). There are no refunds once spent.
5. **Damage is integer and rounds down.** Every combat multiplier (elevation, Bomber-vs-building, artillery splash) is applied with truncation, never rounding up.
6. **No minimum attack range.** Any unit with attack range ≥ 1 may strike anything from distance **1 up to its range**; the only distance never targetable is **0** (the attacker's own tile).
7. **Strict fog of war.** You receive only what your units/buildings currently see — no memory of past sightings — and who you can privately contact (DM, treaty) is gated by who you have **met** (invariant respected by `known_players`).
8. **Elimination is permanent.** Losing your last fully-constructed Base eliminates you for good; nothing — including a Base whose construction finishes afterward — revives you.

---

## Win Condition

Destroy every enemy **Base**. A player with no remaining **fully-constructed** Base is immediately eliminated — a Base still under construction does not keep you alive (and a Base whose construction *finishes after* you were already eliminated does **not** revive you; elimination is permanent). If everyone else is eliminated, the **last player standing wins outright**.

**Survival is a victory condition.** If the turn limit is reached with more than one player still alive, **every surviving player is a co-winner** — they share the victory equally. There is no tiebreaker on gold, units, or buildings; outlasting the clock with a Base intact is enough to win. (Only if all players are eliminated on the same turn does the game end with no survivors.)

---

## Starting Conditions

Each player starts with:
- **1 Base** placed by the map generator at a starting position
- **0 units** — your first units must be built
- **500 gold**

**Fair spawns:** Every Base is placed on a plain **NORMAL** tile — never on elevated, difficult, concealment, or rich-resource terrain. No player receives a terrain advantage or disadvantage at their starting position.

**Even distribution:** Starting positions are spread across the torus by seeding a grid (the best rows×cols factorisation of the player count for the map's aspect ratio) and then running a deterministic **repulsion relaxation**. Each relaxed spawn is then nudged to the nearest unused **NORMAL** tile so the base itself never sits on elevated, difficult, concealment, or rich-resource terrain. This keeps the starts well spaced while preserving fair terrain.

---

## Map

- Hex grid with **pointy-top** orientation, wrapped as a **torus** — both axes wrap, so moving off the right edge brings you to the left edge, and same for top/bottom. There are no borders or walls.
- Default size: **35 × 30** tiles. 
- Procedurally generated each game from a random seed. Pathfinding (A*) fully accounts for the torus wrap — shortest paths may cross map edges.

### Terrain Types

| Terrain       | Approx. % | Combat Effect                               | Movement Effect | Vision Effect                                                                      |
| ------------- | --------- | ------------------------------------------- | --------------- | ---------------------------------------------------------------------------------- |
| Normal        | 45%       | None                                        | 1 pt/step       | Normal                                                                             |
| Elevated      | 12%       | Attacker on this tile deals **+25% damage** | 1 pt/step       | **Blocks line of sight** beyond it for non-elevated, non-flying observers           |
| Difficult     | 20%       | None                                        | **2 pts/step**  | Normal                                                                             |
| Concealment   | 13%       | None                                        | 1 pt/step       | **Harder to see into** (−1 effective vision range for non-flying observers), and **Scouts inside are invisible** to enemies |
| Rich Resource | 10%       | None                                        | 1 pt/step       | None — but a Base or Mine on this tile earns a flat **50g/turn**                    |

---

## Economy

Every player starts with **500 gold**. Gold is earned each turn from completed buildings:

| Source                     | Gold/turn |
| -------------------------- | --------- |
| Base                       | 10g       |
| Base on Rich Resource tile | 50g       |
| Mine                       | 20g       |
| Mine on Rich Resource tile | 50g       |

Gold is spent immediately when an action is queued (upfront, not on completion). Unspent gold carries over. You cannot go into debt — actions you cannot afford are silently ignored. There are no refunds if a building under construction or a production queue is destroyed.

---

## Turn Structure

All turns are **simultaneous**: every player submits their full action list, then the server resolves everything at once in three phases:

1. **Units** — attacks (all computed simultaneously, then applied), movement, medic heals
2. **Buildings** — existing production queues tick, then construction ticks, then new build/produce actions are queued
3. **Coordination** — diplomacy (treaty accept/break/expire), chat delivery, elimination checks

Actions that conflict are resolved deterministically — invalid actions are silent no-ops with no error.

### Phase ordering matters: units move before buildings are placed

Because **units (phase 1) fully resolve before construction (phase 2)**, movement and building placement interact within a single turn:

- **A unit can block a build.** If an enemy (or your own) unit moves onto the tile you targeted with a `construct_building`, that tile is occupied by the time construction is processed, so your build is silently rejected (no gold spent). You cannot build onto a tile a unit lands on this turn.
- **A unit vacating a tile frees it for a build.** Conversely, if your unit moves *off* a tile this turn, the tile is empty when construction runs, so you may build there in the same turn.
- **A freshly placed building cannot block a move.** Buildings are added after movement is resolved, so a building constructed this turn never blocks a unit that already moved this turn — but it does block moves on *subsequent* turns.

This ordering is deterministic, so a defender can deny an expansion by parking a unit on the intended Base/Mine tile, and an attacker cannot "wall in" a unit by building around it on the same turn it moved.

### Tile Capacity

Per **Invariant 1 (one entity per tile)**: a tile holds at most one entity, and this applies uniformly to ground units, air units, and buildings — none can share a tile with any other. A tile with a building cannot also hold a unit; a tile with a unit cannot receive another unit of any type; air cannot stack on ground.

### Move Collision

Upholding **Invariants 1 and 2**: if two or more units (any type, any owner) attempt to move into the same destination tile, **all of them fail** — none moves (symmetric, no player-order tiebreak). This applies to ground vs ground, air vs air, and ground vs air equally. A move is also blocked if the destination is already occupied by any entity not being vacated this turn.

### Move + Attack Same Turn

A unit can submit **both a MoveAction and an AttackAction in the same turn**. Resolution order: attacks fire first (from the unit's current position), then the unit moves. So a unit can shoot from its starting tile and then relocate. There is no mutual exclusion.

If you submit two AttackActions for the same unit, the second overwrites the first (last-write-wins per unit). Same for two MoveActions.

### Simultaneous Combat

All attacks in a turn are resolved at the same instant. Damage from every attacker is accumulated in a pending pool and applied together after all attacks are computed. A target cannot be "saved" by dying early — even if attacker A's hit alone would kill the target, attacker B's damage is still added to the pool before death is checked. There is no overkill prevention.

**Damage is integer and rounds DOWN.** Every multiplier — the +25% elevation bonus, the ×4 Bomber-vs-building bonus, and the 50% artillery splash — is applied with truncation (floor), not rounding. So an Infantry (30 atk) firing from elevation deals `int(30 × 1.25) = 37`, and a 60-power artillery splash deals `int(60 × 0.5) = 30`.

Dead units are **removed before movement is resolved**. If you kill a unit this turn, its tile is free and another unit can move into it in the same turn.

---

## Units

Units are produced by completed production buildings and spawn on an adjacent free tile. Gold is deducted immediately when production is queued.

| Unit      | HP  | Move | Atk Range | Vision | Attack | Cost | Build Turns | Notes                                          |
| --------- | --- | ---- | --------- | ------ | ------ | ---- | ----------- | ---------------------------------------------- |
| Infantry  | 100 | 1    | 1         | 3      | 30     | 50g  | 1           | Cheap frontline body                           |
| Scout     | 50  | 3    | 1         | 5      | 10     | 100g | 1           | Extreme vision; fragile                        |
| Medic     | 60  | 1    | —         | 3      | 0      | 100g | 1           | Heals 20 HP/turn to adjacent friendly **ground units only** (not air/buildings) |
| Tank      | 200 | 2    | 1         | 3      | 60     | 200g | 1           | Fast raider; no special mechanics beyond stats |
| Artillery | 50  | 1    | 3         | 4      | 60     | 200g | 2           | Splash damage; no minimum range (fires 1–3)    |
| Fighter   | 250 | 3    | 2         | 4      | 50     | 300g | 2           | Flies; fast, durable air superiority           |
| Bomber    | 150 | 2    | 1         | 3      | 50     | 350g | 3           | Flies; +300% damage vs buildings               |

**Movement** works on a per-turn **movement-point budget**. A unit's **Move** stat (the table column) is how many movement points it gets each turn. Stepping into an adjacent tile *spends* points equal to that tile's entry cost: **1 point** for Normal / Elevated / Concealment / Rich Resource tiles, **2 points** for **Difficult** terrain ("2 pts/step" = entering a Difficult tile costs 2 of your points instead of 1). A single move action walks a path step-by-step, deducting each tile's cost, and is rejected if the path's total cost exceeds the budget. Examples:
- **Infantry** (Move 1): one step onto a normal tile, or **cannot enter** a Difficult tile at all (it costs 2, the budget is 1).
- **Tank** (Move 2): two steps across normal tiles, **or** one step into Difficult terrain (spends both points), then stops.
- **Scout** (Move 3): three normal steps, or e.g. one normal + one Difficult (1 + 2 = 3).

So Difficult terrain doesn't just slow you — it can be **impassable in a single turn** to low-mobility units, and always halves how far you travel through it.

**Attack range** is the maximum hex distance to a target. **Any** unit may attack without moving first — attacking never requires a move (the "range 3" note on Artillery just means it reaches *far*, not that it's the only unit that can fire while stationary). Melee units (range 1) must be adjacent; ranged units may strike anything from **1 tile up to their range** — there is **no minimum range**, so Artillery (range 3) can also hit adjacent enemies, not only distant ones. The one distance you can never target is **0** (your own tile — it is always occupied by the attacking unit, so a self-tile attack is rejected). A unit can move *and* attack in the same turn; the attack fires from the **pre-move** tile.

**Build Turns** is the number of full turns after queuing before the unit is ready. A unit queued on turn 5 with 1 build turn is available on turn 6.

### Attack Validity

By **Invariant 1 (one entity per tile)**, the primary target tile holds **either nothing or a single entity** (a unit *or* a building) — so there is no "mixed tile" case to consider. Validity depends only on that single occupant:

| Primary tile contents                       | Result                                                              |
| ------------------------------------------- | ------------------------------------------------------------------- |
| Empty                                       | Valid — attack fires; artillery splash still triggers on the ring   |
| A single **enemy** unit or building         | Valid — full damage applied                                         |
| Your **own** unit or building               | **Invalid** — silently rejected, no damage, no splash               |
| A **peace-treaty ally's** unit or building  | **Invalid** — silently rejected                                     |

In short: you may fire at an empty tile or one holding a non-allied enemy; you may **never** fire at a tile holding your own or a current peace-ally's entity. Invalid attacks are silent no-ops, and gold is not refunded.

### Artillery Splash

When Artillery fires a valid attack, the primary target tile takes full damage. Every tile **within 1 hex** of the primary target takes **50% splash damage**.

**Splash hits everyone** — own units and peace-treaty allies in the splash ring take full splash damage. There is no filter. Plan artillery carefully: firing at an enemy adjacent to your own troops will hurt them.

Artillery can fire at an **empty tile**. The primary tile takes no damage (nothing there), but splash still fires on all adjacent tiles. This allows area denial and indirect damage.

Like **every** unit, Artillery has **no minimum attack range** (no unit in the game does) — it can fire at any distance from **1 up to 3**, so it bombards adjacent enemies just as well as distant ones (splash still triggers around the target). The only distance no unit can ever target is **0** (the attacker's own tile, which it occupies). Artillery can move and attack in the same turn; the attack fires from the pre-move position.

### Medic Healing

Each turn, a Medic automatically heals all adjacent friendly **ground units** (not air units, not buildings, not other Medics) by **20 HP**, capped at max HP. No action is required — healing happens passively at the end of the unit phase, after all attacks resolve. Multiple Medics adjacent to the same unit stack their heals.

### Air Units

Fighter and Bomber are bound by **Invariant 1** like everything else — there is no separate air layer, so they cannot share a tile with any other entity. Air vs air, air vs ground, and air vs building tile conflicts all result in the move failing.

Bomber deals **+300% (×4) damage against buildings**. The multiplier is applied after the elevation bonus (if the Bomber is on elevated terrain). The bonus does NOT apply when attacking units.

Air units are visible to all enemies within normal vision range and can be attacked by any unit with sufficient range.

---

## Buildings

Construction rules differ by building type:

- A **Base** can be founded on **any empty tile you currently have vision of** — it needs no adjacency to an existing building, but you **cannot** found one blind in fog of war. This is how you expand into a new region: scout a tile, plant a Base on it, then grow other buildings around it once it completes. A constructed Base costs 300g and takes 5 turns to build (only the *starting* Base is pre-built and complete from turn 0). (A defender can deny an expansion by occupying the target tile — see "Phase ordering" above.)
- **Every other building** must be constructed **adjacent to a fully-completed own building** (distance 1). A building that is still **under construction does not count as an anchor** — you cannot chain new construction off a building that has not finished yet.

In all cases the target tile must be empty of any entity. Gold is deducted immediately when the construction action is processed. A building under construction provides no income, vision, or production.

A building that completes construction this turn **does not earn income until the following turn** — income is collected at the start of the building phase before construction counters are decremented.

A production building **cannot emit a unit on the same turn its construction finishes** — production queues tick *before* construction ticks within the building phase, so a building that completes this turn has no queue to process yet. (It *can* be given a `produce_unit` order on that same turn, since order-enqueuing runs after the construction tick — but the unit will not appear until its build-turn countdown elapses on a later turn.)

A single building can be given **any number of `produce_unit` orders in one turn** — there is no per-building production-slot limit. Each order is enqueued separately, gold is deducted for each immediately, and each carries its own build-turn countdown. Units appear when their countdown elapses, so several units queued on the same turn with the same build time all **complete together on the same later turn** (each spawning on its own free tile) — they are *not* metered out one per turn. (Mixed build times still finish on their own turns: an Infantry queued alongside a Bomber appears 1 turn later, the Bomber 3 turns later.)

**Building multiple buildings in one turn**: ConstructBuildingActions are resolved **one at a time, in the order they appear in your action list**, and each new building is added to the map **immediately** before the next action is processed. Consequences:
- A building placed this turn is **incomplete**, so it **cannot anchor another non-Base building the same turn** — the anchor must already be fully built. You therefore cannot grow an outward "frontier" of connected buildings in a single turn; each new non-Base building must touch a building that was already complete at the start of the turn.
- **Bases are the exception**: because a Base needs no anchor, you can plant several Bases on empty tiles anywhere in one turn (gold permitting).
- The empty-tile and affordability checks are live: gold is deducted per build as it is placed, and a tile claimed by an earlier build this turn is no longer available.

**Build collisions between players**: if **two or more different players** target the **same tile** with a build on the same turn, **all of them fail** — no one builds and no gold is spent. This mirrors the unit move-collision rule (contested tile → everyone bounces), so a contested expansion is never decided by player order. (Within a *single* player's own action list, their first build still claims the tile and later ones on it are dropped, per the rule above.)

**Spawn placement**: a produced unit appears on the **target tile** named in its `produce_unit` order (which must be adjacent to the producing building — distance = 1). If that tile is occupied, the engine instead places the unit on **any free tile adjacent to the producing _building_** — its own 6 surrounding tiles, *not* the target tile's neighbours — so a produced unit always appears next to the building that made it. If the building's tile and all 6 of its neighbours are occupied (the building is fully boxed in), the unit is **silently lost** — it never spawns but the gold is still spent.

These two failure modes differ in **gold cost**, because the only gates applied when the order is *enqueued* are **affordability** and **target adjacency** (distance ≤ 1) — **occupancy is never checked at enqueue**, neither the target's nor the building's surroundings. Boxing-in is only discovered later, at *spawn* time (after the build-turn countdown):
- **Target not adjacent (distance > 1):** the order is **rejected immediately when issued** — **no gold is spent** (a free no-op, like any invalid action).
- **Valid, affordable, adjacent target, but no free tile when the unit is due to spawn:** the gold was already deducted at enqueue, so the unit is **lost and the gold is gone**. This includes a building that is **already fully boxed in at the moment you issue the order** — the engine does not pre-check that, so it still takes your gold and queues a unit that can never spawn. Don't produce from a surrounded building.

**Buildings under construction are targetable**: A building that is still being built can be attacked and takes damage normally. If it dies before completing, it is removed; the construction countdown is unaffected by HP damage otherwise. Damaged buildings do not self-repair.

| Building | HP  | Cost | Build Turns | Produces               | Income | Vision Bonus | Notes                                         |
| -------- | --- | ---- | ----------- | ---------------------- | ------ | ------------ | --------------------------------------------- |
| Base     | 300 | 300g | 5           | —                      | 10g    | +3           | Starting Base is pre-built; can also be constructed anywhere to expand. Losing all Bases = eliminated |
| Mine     | 100 | 200g | 2           | —                      | 20g    | —            | Core income expansion                         |
| Barracks | 200 | 100g | 2           | Infantry, Scout, Medic | —      | —            | Ground unit factory                           |
| Factory  | 200 | 300g | 3           | Tank, Artillery        | —      | —            | Heavy weapons                                 |
| Airbase  | 200 | 500g | 3           | Fighter, Bomber        | —      | —            | Air power                                     |

All buildings occupy their tile exclusively — no unit of any type can move onto a tile that has a building. Only one building can exist per tile.

Buildings can be attacked and destroyed. A destroyed building is removed permanently; there is no rubble or recovery.

---

## Fog of War

Each player only sees tiles within their **vision range**. Vision is contributed by:
- Each own **unit** (per the unit's `vision_range` stat)
- Each **completed** own building (per the building's vision bonus — incomplete buildings contribute nothing)

Tiles outside vision are **completely absent** from the observation — they do not appear in `visible_tiles` at all, even as empty entries. There is no "previously seen" memory; if a tile leaves your vision, it vanishes from your next observation.

**Concealment** has two vision effects:
- **−1 effective vision range into it.** A concealment tile is only visible to an observer standing within `vision_range − 1` of it — one tile closer than open ground. In other words, concealment tiles on the *outer ring* of a unit's vision are not seen; you must close in to spot what's in the cover. (Normal tiles on that same ring are still seen — the penalty is per concealment *target* tile, not a blanket range cut.) **Flying** observers (Fighter/Bomber) ignore this penalty and see into concealment at full range, just as they see over elevation.
- **Scouts inside are invisible.** A **Scout inside a Concealment tile** is hidden from enemies even when they have vision of that tile — it simply never appears in their entity list (other unit types in concealment are still seen normally, subject to the range penalty above).

**Elevation blocks line of sight.** A non-elevated, non-flying observer cannot see a tile if an **elevated** tile lies between it and the target — vision is occluded beyond the ridge. Observers that are themselves on elevated terrain, and all **flying** units (Fighter/Bomber), ignore this and see over elevation. This is the main reason a unit within nominal `vision_range` may still not appear in your observation.

---

## Observation Payload

Each turn, your agent receives a dict with these top-level keys:

| Key                         | Content                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------ |
| `player_id`                 | Your player ID string                                                                |
| `turn_number`               | Current turn integer                                                                 |
| `max_turns`                 | The game's configured turn limit; at the limit every surviving player co-wins (see Win Condition) — lets you pace your endgame to the real deadline |
| `map_width` / `map_height`  | Torus dimensions in tiles (default 35 × 30) — needed for correct wrap/distance math  |
| `resources`                 | `{"gold": N}` — your own gold only (there is no global gold leaderboard)             |
| `visible_tiles`             | List of tile dicts (see below)                                                       |
| `treaties`                  | Active and breaking treaties you are party to                                        |
| `incoming_treaty_proposals` | Pending proposals waiting for your response                                          |
| `known_players`             | Player IDs you have *met* (seen one of their entities, or received a DM / treaty proposal from them). This is exactly the set you are allowed to DM **and** to `propose_treaty` to — see Chat and Diplomacy. Membership is one-directional (seeing someone does not put you in *their* list). |
| `global_chat`               | The **full** cumulative list of global chat messages — no size cap, so a chat flooder's oversized messages reach you in full (the cost is borne by the reader: too much to process in time means you time out). |
| `private_chat`              | The full list of private messages sent to or from you (no size cap)                  |

### Visible Tile Dict

```json
{
  "q": 5,
  "r": 3,
  "terrain": "elevated",
  "entities": [ ... ]
}
```

Entity dicts have a common base (`id`, `owner_id`, `type`, `q`, `r`, `hp`, `max_hp`) plus type-specific fields:

- **Buildings** add: `is_complete`, `construction_turns_remaining`, `vision_bonus`
- **Units** add: `movement_range`, `attack_range`, `vision_range`, `attack_power`, `can_fly`, `has_moved`, `has_attacked`

A building's **unit-production queue is NOT included in the observation** — there is no field listing units a building has queued or how soon they will spawn, for your own buildings or anyone else's. (`construction_turns_remaining` is the countdown for the **building's own construction**, *not* a pending-unit timer.) Since each observation is stateless (no memory of prior turns), you must track your own outstanding `produce_unit` orders yourself if you need to know a unit is about to appear — e.g. to keep a tile clear for it. In practice you rarely need to: a produced unit auto-spawns on any free tile adjacent to its building (see "Spawn placement"), so it only fails to appear if the building is fully boxed in.

`has_moved` and `has_attacked` reflect whether the unit already acted this turn. They are reset to `false` at the start of each turn, so mid-turn their values are always `false` (observations are collected before actions resolve). They are informational only — the engine does not use them to prevent additional actions.

### Treaty Dict (in `treaties`)

```json
{
  "partner_id": "player-2",
  "treaty_type": "peace",
  "breaking_in_turns": null
}
```

`breaking_in_turns` is `null` when the treaty is stable (ACTIVE). When a break is in progress (BREAKING), it holds the **remaining turns** of the countdown (counts down from 5 to 1, then the treaty expires).

### Incoming Proposal Dict

```json
{
  "proposer_id": "player-3",
  "treaty_type": "peace"
}
```

You only see proposals where you are the **target**. Your own outgoing proposals are not reflected here — you learn they were accepted or rejected when the `treaties` list changes.

---

## Diplomacy

Players can form **peace treaties** with each other. An active peace treaty prevents both parties from dealing damage to each other's units and buildings. The peace check is enforced **on the primary tile before the attack fires**: if the single entity on that tile belongs to you or a peace-treaty ally, the entire attack is rejected — no damage, no splash. (Splash from artillery is the exception — see below — it ignores ownership entirely.)

Splash damage ignores all treaty and ownership checks — it hits everyone in the splash ring including own units and allies. Position your forces accordingly.

### Who you can propose to (fog-gated, like DMs)

Proposing a treaty follows the **same rule as a private chat DM**: you may only `propose_treaty` to a player in your **`known_players`** set — one you have *met*, either by **seeing** one of their entities or because they have already **contacted you** (a delivered DM or treaty proposal from them). A proposal to a player you have never encountered — a **guessed id** — or to an **eliminated** player is silently dropped: no proposal is recorded, and it establishes **no** meeting (you cannot bootstrap contact by cold-proposing). "Met by sight" is **one-directional** (seeing someone does not put you in their list). A *delivered* proposal adds you to the target's `known_players`, so they can respond and propose back. `respond_treaty` and `break_treaty` need an existing proposal/treaty, so they are naturally limited to players you are already in contact with.

### Treaty Lifecycle

1. **Propose** — send a peace proposal to a player you have met (see above). They see it in their `incoming_treaty_proposals` next turn. You cannot re-propose if any treaty between the same pair already exists (in any state).
2. **Accept** — the target accepts. The treaty becomes ACTIVE immediately that turn.
3. **Reject** — the target rejects. The proposal is deleted; either party may propose again.
4. **Break** — either party issues `break_treaty`. This starts a **5-turn countdown**. Both parties are privately notified. The treaty remains fully ACTIVE during the countdown — neither party can attack the other yet.
5. **Expired** — when the countdown reaches 0, the treaty is deleted. Both parties are privately notified. They are now free to attack and may form a new treaty.

### Rules During Countdown

- **Cannot break again**: calling `break_treaty` on an already-breaking treaty fails silently. The countdown cannot be reset or extended.
- **Cannot re-propose**: `propose` is blocked for any pair with an existing treaty entry (ACTIVE or BREAKING). New proposals are only possible once the treaty fully expires.
- **Cannot accept a breaking treaty**: `respond_treaty` with `accept` on a BREAKING treaty fails silently.

### Secrecy

Treaty events (formed, break initiated, expired) are delivered as **private system DMs** to the two parties only. No global announcement is made. Third parties cannot observe treaty status directly — they can only infer it from player behavior or what is said in chat.

### Treaty cutoff (forced open war)

From turn **200** (`TREATY_CUTOFF_TURN`) onward, diplomacy is **closed for the rest of the game**:
- **All existing treaties are voided** the moment the cutoff is reached — every peace pact (active or mid-break) ends at once, and both parties are notified by the usual private "expired" DMs.
- **No new treaty can form.** `propose_treaty` and `respond_treaty` are silently ignored, and the observation shows empty `treaties` / `incoming_treaty_proposals` from that turn on.

This guarantees the final stretch (the last 100 turns of the default 300-turn game) is fought as open war — a board frozen by mutual peace cannot coast to the turn limit. The replay viewer's treaty graph reflects this: there are **no treaty lines from turn 200 onward**.

---

## Chat

A turn's `send_chat` actions are all delivered — the engine does **not** hard-cap the number of messages per turn, nor the length of any message. Well-behaved agents keep it to a few; a player can use the chat to prompt inject other players, or even sends oversized "context-bomb" messages as a length-DoS. The observation ships the **full** chat history uncapped, so those oversized messages land on opponents in full. The **replay** clips each stored message to 200 chars so the saved file stays small regardless.

- **Global** (`recipient_id: null`) — broadcast to all players. Visible to everyone in the All/Global channels of the viewer. Anyone may broadcast at any time.
- **DM** (`recipient_id: "player-X"`) — private message to one player only. Only the sender and recipient see it.

System messages (treaty events) are delivered as private DMs from `__system__`. **Eliminations are the only event broadcast globally.**

### Who you can DM (fog-gated)

A DM is only delivered if its recipient is a player in your **`known_players`** set — the players you have *met*. You can DM:
- a player whose entity you have **seen** (met by sight), or
- a player who has **DM'd you** (so you can always reply to someone who reached you).

Everything else is silently dropped (no log entry, no effect):
- DMs to a player you have never encountered — **guessing an id does not work**, and seeing someone in **global chat does not let you DM them**;
- DMs to an **eliminated** player (a dead player never reads its mail);
- a dropped DM establishes **no** meeting — you cannot bootstrap a contact by firing a cold message at a stranger.

**"Met by sight" is one-directional.** Vision ranges differ (a Scout sees far without being seen), so if you see an enemy you may DM them, but they have *not* automatically met you — until they see one of your entities or you DM them first. Global broadcasting is always open to everyone regardless of who you've met.

---

## Elimination

A player is eliminated the moment they have **no remaining fully-constructed Base**. A Base still under construction does **not** count — if your last completed Base is destroyed while a replacement is still building, you are eliminated before it finishes.

**One-turn economic grace:** elimination is checked in the coordination phase (phase 3), *after* the building phase (phase 2) has already collected income, ticked production, and processed builds. So on the very turn your last Base is destroyed, that turn's economy still runs for you — your other buildings yield gold, queued units can still complete, and `construct`/`produce` orders you submitted resolve — but none of it brings the Base back, so you are still eliminated at end of turn. From the **next** turn on, your buildings are inert (below).

On elimination:
- A global system message announces the event to all players.
- The player stops receiving observations and their server stops being queried.
- Their existing units decay by **10 HP per turn** until all are dead. Decay is applied unconditionally and cannot be healed.
- Their **buildings go inert**: they generate **no gold**, **produce no units**, and any building still under construction **freezes** (never completes). Their HP is untouched by death (no decay, no self-repair — an undamaged building stays at full health). The buildings are **not removed** — they remain on the map as neutral obstacles occupying their tiles, so a rival must **destroy** them to reclaim the ground for their own construction.
- **Treaties are not auto-voided on death.** If you held an active peace treaty with the player who died, your attacks on their inert buildings are still rejected by the friendly/allied-tile rule until that treaty breaks or expires. Players who were *not* at peace with the dead player can attack and destroy the buildings immediately.

---

## Action Reference

One `ActionPayload` per turn contains a list of any of these actions. All are submitted simultaneously and resolved in phase order.

| Action               | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| `move`               | Move a unit along a path (≤ `movement_range` steps)               |
| `attack`             | Attack a target hex (target within `attack_range`, distance > 0)  |
| `hold`               | Do nothing this turn                                              |
| `construct_building` | Start construction on an empty tile (Base: any tile you can see; other buildings: adjacent to a completed own building) |
| `produce_unit`       | Queue a unit in a completed production building (spawns adjacent) |
| `propose_treaty`     | Send a peace proposal to another player                           |
| `respond_treaty`     | Accept or reject an incoming proposal                             |
| `break_treaty`       | Initiate a 5-turn break countdown on an ACTIVE treaty             |
| `send_chat`          | Send a global or private message                                  |

Invalid actions (wrong owner, insufficient gold, out of range, wrong state) are **silently dropped** — they do not cause errors or penalties.
