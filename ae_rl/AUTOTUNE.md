# ae_rl Autonomous Tuning Playbook

This file is the reference for an autonomous Claude session driving the AE RL tuning loop. Read this first, then `ae_rl/tuning/state.json` (resumability), then the tail of `ae_rl/tuning/log.jsonl` (history). Everything else flows from those three.

## Goal

Maximise the evaluator's response to `rl_autorun.py --submit ae <tag>` submissions. The evaluator returns `score ∈ [0, 1]` and `speed ∈ [0, 1]` — both 1.0 is perfect. Score is the primary signal; speed matters less but is a tiebreaker.

The evaluator is **noisy**. A 0.02-point eval bump from a single run is not proof of improvement — see "Noise handling" below before declaring a win.

## Allowed / forbidden actions

**Allowed**
- Edit any file in the repo, including training scripts, model code, configs, hyperparameters, and `ae/src/`.
- `git checkout -b`, `git add`, `git commit`, `git tag`, `git branch`, `git log`, `git status`, `git diff`.
- Long-running training via `Bash(run_in_background=True)` (see "Training" below).
- Spawning the Discord watcher in the background if it isn't already running.

**Forbidden — never run these**
- `git push origin main` (or `master`) — only push to non-main branches if pushing at all.
- `git push --force` / `git push -f` to any branch.
- `git reset --hard`, `git checkout -- <file>`, `git clean -f`, `git branch -D` — anything that drops uncommitted work or rewrites shared history.
- `git rebase -i`, `git rebase` of pushed commits.
- Deleting checkpoints with eval results recorded in `log.jsonl` (the corresponding `meta.validation_score` is the only proof they exist).
- Modifying `til-26-ae/`, `til-26-finals/`, or `.gitmodules`.

If in doubt: commit, don't delete. Disk is cheap; lost work is expensive.

## State files (canonical truth)

| File | Purpose | Writer |
|---|---|---|
| `ae_rl/tuning/state.json` | Current iteration, best result, in-flight work. Read at session start. | This loop |
| `ae_rl/tuning/log.jsonl` | Append-only history of every iteration: hypothesis, change, training summary, eval result, verdict. | This loop |
| `logs/eval_results.jsonl` | Eval results ingested from Discord. | The watcher (`rl_autorun.py` or `discord_watcher.py`) |
| `ae_rl/runs/<stage>/latest.json` | Live training-run progress + final summary. | The training scripts via `RunSummary` |
| `ae_rl/checkpoints/` | Trained checkpoints. `stage{1,2,3}*.pt` are auto-discovered by stage loaders. | The training scripts |
| `ae/models/ppo.pt` | Checkpoint that ships in the Docker image. Staged by `rl_autorun.py` from a source set by `RL_AUTORUN_CHECKPOINT` + `RL_AUTORUN_STAGE`. | `rl_autorun.py --submit` |

## Resumability protocol (run at every session start)

1. **Read `ae_rl/tuning/state.json`.** If it doesn't exist, initialise (see "Initialising" below) and continue.
2. **If `state.in_flight.kind == "submission"`:** the previous session submitted a tag and was waiting for the eval. Resume:
   ```bash
   python rl_autorun.py --await-eval ae "$TAG" --since-iso "$SUBMITTED_AT" --timeout 1800 > /tmp/eval.json
   ```
   On success, parse the result, append to `log.jsonl`, clear `in_flight`, update `best_*` if improved. On timeout, leave `in_flight` set and pause for human input (the watcher may be down).
3. **If `state.in_flight.kind == "training"`:** read `state.in_flight.summary_path`. If `status == "running"`, the previous training is still alive — either wait with `Bash(run_in_background)` monitoring, or check whether the bash shell ID is still tracked. If `status == "completed"`, proceed to staging + submission. If `status == "failed"` or `"interrupted"`, log the failure, clear `in_flight`, and pick a different hypothesis.
4. **If `state.in_flight == null`:** read the tail of `log.jsonl` to understand recent verdicts, then pick the next hypothesis.

## Initialising (only on the very first session)

State doesn't exist yet. Bootstrap:

```bash
# Confirm starting branch + write iteration-0 record.
git checkout -b tuning/auto || git checkout tuning/auto
mkdir -p ae_rl/tuning
# Write state.json with iteration=0, best fields all null, no in_flight.
```

The current best known checkpoint is `ae_rl/checkpoints/stage3_league_best.pt` (annotation suffix `_heu+294` indicates internal validation score, *not* an eval score). The first useful iteration is to submit this checkpoint as a known-good baseline so we have a real eval datapoint to beat.

## The loop

```
WHILE not bored AND not blocked:
  state = read(state.json)
  IF state.in_flight is not None:
    handle resume (see protocol above)
    continue
  
  hypothesis = pick_next_hypothesis(log.jsonl tail, hypothesis menu)
  
  IF hypothesis requires training:
    set state.in_flight = {kind: "training", stage, summary_path, hypothesis, started_at}
    launch training in background, polling summary.json status
    on completion: stage checkpoint to ae/models/ppo.pt
  
  tag = generate_tag(iteration, descriptor)  # e.g. "tune-007-lower-lr-5e5"
  set state.in_flight = {kind: "submission", tag, submitted_at: now, hypothesis}
  
  preflight_checks()  # watcher up? docker? gcloud? checkpoint present?
  
  python rl_autorun.py --submit ae <tag>
  python rl_autorun.py --await-eval ae <tag> --since-iso <submitted_at> --timeout 1800
  
  parse eval result
  append entry to log.jsonl
  clear state.in_flight
  
  IF score > state.best_eval_score:
    update best_* fields in state.json
    git commit -am "tune: <hypothesis> — eval=<score> speed=<speed> (Δ=+<delta>)"
    git tag best-<score>  # e.g. best-0.78
  ELSE:
    git commit -am "tune: <hypothesis> — eval=<score> speed=<speed> (no improvement)"
  
  IF last N iterations had no improvement: change strategy (see "Stuck detection")
END
```

## Pre-flight checks (before every submission)

1. **Watcher alive?** `logs/eval_results.jsonl` should have entries within the last few hours. If absent or stale, launch:
   ```bash
   python rl_autorun.py &  # Bash(run_in_background=True)
   ```
   Needs `DISCORD_TOKEN` + `DISCORD_CHANNEL_ID` in `.env`. Without these, do not submit — the eval will land in Discord but never reach `eval_results.jsonl`, and `--await-eval` will hang.

   **Dependency note:** `rl_autorun.py` and `discord_watcher.py` use `discord.py-self` (selfbot, user-token auth), NOT `discord.py`. The package is `discord.py-self>=2.1.0` — listed in `requirements-dev.txt` and `pyproject.toml`. If you see `ModuleNotFoundError: No module named 'discord'`, install via `pip install discord.py-self` (do NOT install plain `discord.py` — it is API-incompatible).
2. **Docker reachable?** `docker info` exits 0.
3. **gcloud authenticated?** `gcloud auth print-access-token` exits 0.
4. **Checkpoint staged.** `rl_autorun.py --submit` calls `stage_ae_checkpoint` itself, which respects `RL_AUTORUN_STAGE` (default `2`) and `RL_AUTORUN_CHECKPOINT` (default `best`). For Stage 3 best, set `RL_AUTORUN_STAGE=3 RL_AUTORUN_CHECKPOINT=best`.

## Hypothesis menu (cheap → expensive)

Always exhaust cheap experiments first. Each row = one iteration of the loop.

**Cheap (no retrain, ≤ 5 min per iteration):**
- Toggle `LayeredRLPolicy` guards in `ae/src/ae_manager.py` — `dodge_override`, `oscillation_break`, `heuristic_fallback`.
- Tune Layered thresholds: `value_threshold` (-1.0 to 0.5) and `entropy_threshold_frac` (0.5 to 0.95).
- Switch deployed policy: `LayeredRLPolicy` ↔ `RLPolicy` ↔ `HeuristicPolicy(**DEFAULT_POLICY_KWARGS)` (one-line swap in `ae_manager.py:85`).
- Swap which checkpoint deploys: `RL_AUTORUN_STAGE=3 RL_AUTORUN_CHECKPOINT=best` vs `=current` vs `=2`.
- Tweak heuristic kwargs in `DEFAULT_POLICY_KWARGS` (only if shipping heuristic).

**Medium (training-time, 30 min – several hours):**
- Continue Stage 3 from current best with different opponent mix (`--berserker-prob`, `--tactical-prob`, etc.).
- Polish phase: `--no-shaping --lr 3e-5 --updates 100` from current best.
- Lower LR + more updates on Stage 2.
- Stage 2 from BC with different `--validation-baseline` (`vanilla` vs `strong`).

**Expensive (multi-day):**
- Stage 1 BC from a different teacher policy (e.g. modified `EditedHeuristicPolicyV2` kwargs).
- Architecture changes in `ae_rl/model.py` (and the mirror in `ae/src/policies/rl_policy.py`).
- Add new shaping signals (novelty bonus, count-based exploration — see `review-the-reinforcement-learning-bright-lemur.md` §6 group 4).

Start with cheap. Only escalate when cheap is exhausted *and* internal benchmarks (`ae_rl/benchmark.py --baseline vanilla`) confirm the current checkpoint genuinely can't go further with deployment-side tweaks.

## Noise handling

Single-eval deltas are unreliable. Decision rule:

- **|Δ| < 0.05:** within noise. Don't commit this as an improvement. Re-submit the same checkpoint with a different tag to get a second datapoint before drawing conclusions.
- **|Δ| ≥ 0.05 and confirmed by `ae_rl/benchmark.py` showing the same direction on the internal baseline:** treat as real.
- **|Δ| ≥ 0.15:** real regardless of internal benchmark.

The internal benchmark (`python ae_rl/benchmark.py --ckpt <path> --rounds 50 --baseline vanilla`) is deterministic at fixed seed and runs in ~5 min. Use it as a cheap pre-flight before every submission to avoid burning eval submissions on changes that don't move the in-house score.

## Stuck detection

If the last 5 logged iterations all have `verdict == "no improvement"`:
1. Stop adding more shallow tweaks.
2. Re-examine `log.jsonl` for patterns — is the eval score plateauing at a specific value? Is one event dominating (run `python ae_rl/diagnose.py --ckpt ae_rl/checkpoints/stage3_league_best.pt --rounds 30`)?
3. Escalate to the next tier of the hypothesis menu.
4. If even expensive experiments don't move the score after 3 attempts, write a checkpoint note in `state.json` (`paused_for_human: "<reason>"`) and stop the loop.

## Git milestone protocol

- **Branch.** Stay on the existing tuning branch (`git branch --show-current`). If you must switch (e.g. comparing two strategies), create child branches off the tuning branch: `tuning/auto/<short-desc>`. Never branch off main directly for tuning work.
- **Commit per iteration.** Even no-improvement iterations get a commit so `git log` is the audit trail.
  ```
  tune: <one-line hypothesis> — eval=<score>/<speed> (Δ=<+0.04|none>)
  
  Hypothesis: <one paragraph>
  Change: <list of edited files>
  Eval tag: <submitted tag>
  Training summary: <path>
  ```
- **Tag improvements.** When `score > previous_best`, `git tag best-<score>` (e.g. `best-0.78`). Tags are cheap and let you `git checkout best-0.78` later.
- **Push policy.** Pushing to a personal branch is fine if the user has configured one; never push to main; never force-push anywhere.

## Clutter management

To prevent the repo (and context window) from bloating:

- **`ae_rl/runs/<stage>/<timestamp>.json` history copies.** Keep the most recent 5 per stage; delete older ones. `latest.json` always stays.
- **`ae_rl/checkpoints/league/`** can grow without bound during Stage 3 training. Cap at 30 snapshots — delete oldest first. The best-validated snapshots are saved separately via `--gated-snapshots`.
- **`ae_rl/checkpoints/stage{2,3}_snapshots/`** per-run snapshot dirs. Delete entire dirs older than 7 days.
- **`logs/eval_results.jsonl`.** Never trim — it's the source of truth for `--await-eval` and `--since-iso` filtering. Append-only.
- **`ae_rl/tuning/log.jsonl`.** Never trim during a run. If it grows past ~1000 lines, rotate to `log.<date>.jsonl` and start fresh.
- **`logs/<watcher>.log`.** Trim to last 10000 lines if it grows past 100MB.

Don't delete checkpoints referenced in `log.jsonl` even if they appear unused — they're the only way to reproduce a logged eval result.

## State file schema

`ae_rl/tuning/state.json`:

```json
{
  "iteration": 0,
  "branch": "tuning/auto",
  "best_eval_score": null,
  "best_eval_speed": null,
  "best_eval_tag": null,
  "best_checkpoint": "ae_rl/checkpoints/stage3_league_best.pt",
  "best_git_commit": null,
  "in_flight": null,
  "paused_for_human": null,
  "last_updated_at": "2026-05-24T00:00:00Z"
}
```

`in_flight` is either `null` or one of:

```json
{ "kind": "training",
  "stage": "stage3_league",
  "summary_path": "ae_rl/runs/stage3_league/latest.json",
  "hypothesis": "lower LR + longer run",
  "started_at": "2026-05-24T12:00:00Z" }
```

```json
{ "kind": "submission",
  "tag": "tune-007-lower-lr",
  "submitted_at": "2026-05-24T13:00:00Z",
  "hypothesis": "lower LR + longer run",
  "from_training_summary": "ae_rl/runs/stage3_league/latest.json" }
```

`ae_rl/tuning/log.jsonl` — one JSON object per line:

```json
{ "iteration": 7,
  "started_at": "...", "finished_at": "...",
  "hypothesis": "...",
  "change_summary": "edited train_stage3_league.py:178 lr 2e-4 -> 5e-5",
  "git_commit": "abc1234",
  "training_summary_path": "ae_rl/runs/stage3_league/2026...json",
  "checkpoint_used": "ae_rl/checkpoints/stage3_league_best.pt",
  "checkpoint_validation_score": 294.5,
  "submitted_tag": "tune-007-lower-lr",
  "eval_result": { "score": 0.74, "speed": 0.91, "errors": 0, "timestamp": "..." },
  "verdict": "no improvement (prev best 0.78)",
  "notes": "internal benchmark agreed (-3 vs +294)" }
```

## Standard commands

```bash
# Internal benchmark (5 min, deterministic — use before every submission)
python ae_rl/benchmark.py --ckpt ae_rl/checkpoints/stage3_league_best.pt \
    --rounds 50 --baseline vanilla

# Per-event diagnostics (when score plateaus)
python ae_rl/diagnose.py --ckpt ae_rl/checkpoints/stage3_league_best.pt \
    --rounds 30 --focus-slot agent_0 --sample-actions

# Stage 3 continuation from current best (medium-cost iteration)
python ae_rl/train_stage3_league.py --updates 50 --validate-every 5 \
    --rollback-on-regress \
    --summary-json ae_rl/runs/stage3_league/latest.json &  # background

# Poll training status
jq -r .status ae_rl/runs/stage3_league/latest.json

# Submit + await (the canonical autonomous round-trip)
TAG="tune-$(date +%Y%m%dT%H%M%S)-<descriptor>"
SUBMITTED_AT="$(python -c 'from datetime import datetime,timezone;print(datetime.now(timezone.utc).isoformat())')"
RL_AUTORUN_STAGE=3 RL_AUTORUN_CHECKPOINT=best \
    python rl_autorun.py --submit ae "$TAG"
python rl_autorun.py --await-eval ae "$TAG" \
    --since-iso "$SUBMITTED_AT" --timeout 1800 > /tmp/eval-$TAG.json
jq . /tmp/eval-$TAG.json  # {challenge,tag,errors,score,speed,timestamp}
# Verify returned .tag == $TAG before treating as our result — other teams share
# the same Discord channel and their evals will appear too.
# If --await-eval times out (30 min, no result): another submission may have
# overwritten ours in the queue. Resubmit and await again before giving up.
```

## Session termination (when user asks to stop / push / end session)

When the user requests to terminate the session:

1. **Kill all background processes** — check for running `python.exe` instances from this session (Discord watcher, `--await-eval`, any training jobs) and stop them:
   ```powershell
   Get-Process python | Where-Object {$_.StartTime -gt (Get-Date).AddHours(-6)} | ForEach-Object {
       $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine
       "$($_.Id)  $cmd"
   }
   Stop-Process -Id <pids> -Force
   ```
2. **Update `state.json`** — set `paused_for_human` to a one-line note explaining what's in flight (eval tag, submitted_at, next step), so the next session can resume without guessing.
3. **Commit all changes** — stage every modified file (excluding `.gitignore`d paths like `ae/models/`) and commit with a summary message. Never leave a session with uncommitted tuning state.
4. **Push to the tuning branch** — `git push origin tuning/auto` (or current branch). Never push to `main`.
5. **Confirm** — report to the user which PIDs were killed and what was pushed.

## When to stop and ask the user

- Watcher won't start (missing `DISCORD_TOKEN`, network issue).
- gcloud auth expired.
- Stuck detection fires twice (10 no-improvement iterations).
- Disk filling up (checkpoints dir > 5 GB) — ask before bulk-deletion.
- Any forbidden git operation feels necessary — there's always a safer alternative; if you can't see it, ask.

Set `state.paused_for_human = "<reason>"` and stop.

## Cross-references

- [HANDOFF.md](HANDOFF.md) — what's on this branch + the autonomous-caller pipeline section.
- [README.md](README.md) — original architecture overview.
- [review-the-reinforcement-learning-bright-lemur.md](review-the-reinforcement-learning-bright-lemur.md) — earlier audit with hypothesis ideas (especially reward shaping menu §6).
- [run_summary.py](run_summary.py) — the helper that writes `ae_rl/runs/<stage>/latest.json`.
- Memory: `claude_pipeline.md` and `project_track.md` describe pipeline + novice-track constraints.
