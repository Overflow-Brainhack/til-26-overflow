# Handover

Date: 2026-06-10

This repo is mid-competition work. Do not blindly reset the tree: there are local
changes and untracked evaluation traces that may be useful.

## Current Git State

- `submit.sh` has a small but important fix: challenge predict routes now submit
  as `/ae`, `/cv`, etc. instead of `//ae` on Windows.
- `surprise_chal/evaluate_agent.py` has local changes that make the hard
  evaluator stronger: extra opponent personalities, seed-rotated opponents,
  teleport-raid behavior, and artillery splash pressure.
- `surprise_chal/participant/src/bastion_agent.py` has local tuning changes:
  it spends surplus gold more aggressively on bases, mines, production, and
  standing defense.
- There are several untracked Surprise trace/eval files in `surprise_chal/`.
  Treat them as useful logs unless confirmed otherwise.

## AE Docker / Finals Submission

The local `overflow-ae:finals` image is not inherently crashing. It was tested
locally and:

- starts successfully
- returns `200` on `/health`
- returns a valid prediction on `/ae`
- returns `404` on `//ae`

The likely failure was the registered Vertex predict route. `submit.sh` used to
emit `//ae` on Windows, which makes the evaluator call a route the FastAPI app
does not serve. That is now patched.

Use Git Bash, not the WSL `bash.exe` launcher on this machine:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' submit.sh ae finals
```

Dry-run after the patch showed:

```text
--container-predict-route=/ae
```

The `overflow-ae:finals` tag points at the same image ID as
`overflow-ae:azbasev3`, so the rename itself was not the problem.

Small Dockerfile caveat: `ae/Dockerfile` has a bad conditional `COPY models/`
inside a `RUN` block. It only matters if building with `AE_INSTALL_TORCH=1`.

## Surprise Challenge

Primary agent to look at: `surprise_chal/participant/src/bastion_agent.py`.

The local evaluator is:

```powershell
python surprise_chal/evaluate_agent.py --agents bastion,shadow --seeds 67,68,69,70,71 --turns 300 --players 20 --opponents hard
```

`surprise_chal/eval_bastion_final.txt` shows `bastion` at 80% survival and
90.3/100 on seeds 67-71, but that output appears to be from before the latest
extra hard-opponent changes in `evaluate_agent.py`. Rerun before trusting it.

Important evaluator interpretation:

- Printed `gold` is final/banked gold in the player state, not total gold gained.
- `avg_gold` exists internally as sampled banked gold over time, but it is not
  the printed `gold` column.
- The evaluator is local and survival-first; it is not the official scorer.

Current strategic direction for `bastion`:

- Survive first. The project notes say every survivor co-wins equally, with no
  tiebreaker on gold, kills, units, or buildings.
- Avoid dying rich. Recent tuning deliberately converts surplus gold into more
  bases, mines, production, and units.
- Base redundancy matters more than pretty economy. Hidden maps and late treaty
  cutoff punish one-base or low-base lines.
- Watch for artillery splash and teleport-like movement quirks in the local
  evaluator; recent hard bots were strengthened around those.

## Useful Files

- `submit.sh` - submission wrapper for all tasks.
- `surprise_chal/evaluate_agent.py` - local multi-seed evaluator.
- `surprise_chal/participant/src/bastion_agent.py` - current main Surprise bot.
- `surprise_chal/participant/src/algo_agent.py` - alternate Surprise bot.
- `surprise_chal/debug_run.py` - per-turn trace tool for one seed.
- `surprise_chal/PLAN.md` and `surprise_chal/strat.txt` - strategy notes.
- `ae/NEXT_SESSION_HANDOVER.md` - older AE-specific handover.

## Quick Checks

AE image smoke test:

```powershell
docker run --rm -d --name ae-smoke -p 15005:5005 overflow-ae:finals
curl.exe -sS http://localhost:15005/health
docker stop ae-smoke
```

AE `/ae` route should work, while `//ae` should not. If Vertex says the image is
crashing, first check the registered predict route.

Surprise quick rerun:

```powershell
python surprise_chal/evaluate_agent.py --agents bastion --seeds 67,68,69,70,71 --turns 300 --players 20 --opponents hard
```

If one seed fails, use `surprise_chal/debug_run.py` or the trace files to inspect
gold, bases, unit mix, and threats around the death turn.
