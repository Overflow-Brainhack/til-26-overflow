"""Stage 1 — Behaviour cloning (jump-start).

Trains the recurrent maskable actor-critic to imitate the production heuristic
(EditedHeuristicPolicyV2). This gives PPO a competent starting policy instead of
forcing it to discover good play from scratch against the sparse base-destruction
reward. Only the policy (CNN + GRU + actor) is trained here; the critic is left
for PPO to fit in Stage 2.

Output: ae_rl/checkpoints/stage1_bc.pt

Usage:
    python ae_rl/train_stage1_bc.py
    python ae_rl/train_stage1_bc.py --episodes 64 --epochs 8 --novice
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm.auto import tqdm, trange

import common  # noqa: F401  (path bootstrap)
from common import STAGE1_CKPT, get_device, seed_everything
from controllers import azbasev1_spec, azbasev4_spec, heuristic_spec
from model import RecurrentMaskableActorCritic, save_checkpoint
from rollout import collect_teacher_dataset, default_workers
from run_summary import RunSummary, default_summary_path


# Teacher name → picklable controller spec builder. Pass one or more via
# --teacher; episodes are split evenly across the chosen teachers and the
# per-teacher datasets are concatenated, so the BC target becomes a blend.
_TEACHER_SPECS = {
    "heuristic": heuristic_spec,
    "azbasev1": azbasev1_spec,
    "azbasev4": azbasev4_spec,
}


def _concat_teacher_data(blocks: list[dict]) -> dict:
    """Concatenate per-teacher datasets along the sequence (axis-1) dimension.

    Each block is (T_i, B_i, …); different teachers can yield different episode
    lengths, so we truncate every block to the global-minimum T before stacking
    so the time axis lines up. Single-block input is returned unchanged.
    """
    if len(blocks) == 1:
        return blocks[0]
    t = min(b["actions"].shape[0] for b in blocks)
    out = {}
    for key in blocks[0]:
        out[key] = np.concatenate([b[key][:t] for b in blocks], axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=48, help="teacher games to record (×6 agents = sequences)")
    ap.add_argument("--teacher", type=str, default="heuristic",
                    help="comma-separated teacher(s) to clone from: any of "
                         f"{sorted(_TEACHER_SPECS)}. Episodes are split evenly "
                         "across them and the datasets concatenated. Default "
                         "'heuristic'. Example: --teacher azbasev1,azbasev4 to "
                         "clone the strongest scripted policies, or "
                         "--teacher heuristic,azbasev1,azbasev4 for a blend.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seq-minibatch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--novice", dest="novice", action="store_true", default=True,
                    help="record on the fixed novice map (default)")
    ap.add_argument("--advanced", dest="novice", action="store_false",
                    help="record on randomised advanced maps")
    ap.add_argument("-j", "--num-workers", type=int, default=default_workers(),
                    help="parallel processes for teacher collection (default: cpus-1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-ckpt", type=str, default="",
                    help="path for the BC checkpoint (default: ae_rl/checkpoints/stage1_bc.pt). "
                         "Set a distinct path when cloning a non-default teacher so you "
                         "don't clobber the heuristic-BC seed, e.g. "
                         "--output-ckpt ae_rl/checkpoints/stage1_bc_azbase.pt")
    ap.add_argument("--summary-json", type=str, default="",
                    help="path for the run-summary JSON (default: ae_rl/runs/stage1_bc/latest.json). "
                         "Read this from an autonomous caller instead of parsing stdout.")
    args = ap.parse_args()

    summary_path = Path(args.summary_json) if args.summary_json else default_summary_path("stage1_bc")
    with RunSummary(stage="stage1_bc", args=vars(args), path=summary_path) as summary:
        seed_everything(args.seed)
        device = get_device()
        summary.set("device", str(device))
        summary.set("summary_path", str(summary_path))
        print(f"Device: {device}")
        print(f"Run summary: {summary_path}")

        # Resolve the teacher list. Unknown names fail loudly here rather than
        # silently falling back to the heuristic.
        teacher_names = [t.strip() for t in args.teacher.split(",") if t.strip()]
        unknown = [t for t in teacher_names if t not in _TEACHER_SPECS]
        if unknown:
            raise SystemExit(
                f"--teacher: unknown teacher(s) {unknown}; "
                f"choose from {sorted(_TEACHER_SPECS)}"
            )
        if not teacher_names:
            teacher_names = ["heuristic"]
        summary.set("teachers", teacher_names)

        # Split episodes evenly across teachers (remainder goes to the first).
        per_teacher = max(1, args.episodes // len(teacher_names))
        episode_alloc = {t: per_teacher for t in teacher_names}
        episode_alloc[teacher_names[0]] += args.episodes - per_teacher * len(teacher_names)

        print(f"Collecting teacher demonstrations: {args.episodes} games across "
              f"{teacher_names} (6 agents each), {args.num_workers} worker(s) …")
        t0 = time.time()
        per_data = []
        for tname in teacher_names:
            n_ep = episode_alloc[tname]
            if n_ep <= 0:
                continue
            print(f"  teacher={tname}: {n_ep} games")
            per_data.append(collect_teacher_dataset(
                n_episodes=n_ep, novice=args.novice, progress=True,
                num_workers=args.num_workers,
                teacher_spec=_TEACHER_SPECS[tname](),
            ))
        data = _concat_teacher_data(per_data)
        n_seq = data["actions"].shape[1]
        t_len = data["actions"].shape[0]
        collect_dt = time.time() - t0
        print(f"  recorded {n_seq} sequences × {t_len} steps in {collect_dt:.1f}s")
        summary.set("teacher_collection_seconds", round(collect_dt, 2))
        summary.set("teacher_sequences", int(n_seq))
        summary.set("teacher_seq_len", int(t_len))

        # Report teacher action distribution (sanity check the dataset isn't all STAY).
        acts, counts = np.unique(data["actions"], return_counts=True)
        dist = {int(a): int(c) for a, c in zip(acts, counts)}
        print(f"  teacher action counts: {dist}")
        summary.set("teacher_action_counts", dist)

        # To CPU tensors (kept off-GPU; minibatches moved per step).
        vc = torch.as_tensor(data["viewcone"])
        bv = torch.as_tensor(data["baseview"])
        sc = torch.as_tensor(data["scalars"])
        mk = torch.as_tensor(data["mask"])
        sm = torch.as_tensor(data["staticmap"])
        act = torch.as_tensor(data["actions"])

        model = RecurrentMaskableActorCritic().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        model.train()
        epoch_bar = trange(args.epochs, desc="BC epochs", unit="epoch")
        for epoch in epoch_bar:
            perm = np.random.permutation(n_seq)
            losses, accs = [], []
            for start in range(0, n_seq, args.seq_minibatch):
                cols = torch.as_tensor(perm[start : start + args.seq_minibatch])
                logits, _, _ = model.forward_sequence(
                    vc[:, cols].to(device), bv[:, cols].to(device),
                    sc[:, cols].to(device), mk[:, cols].to(device),
                    sm[:, cols].to(device),
                )
                target = act[:, cols].to(device)
                # NLL of the teacher action under the masked policy.
                dist = Categorical(logits=logits)
                loss = -dist.log_prob(target).mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    accs.append((pred == target).float().mean().item())
                losses.append(loss.item())
            epoch_loss = float(np.mean(losses))
            epoch_acc = float(np.mean(accs))
            epoch_bar.set_postfix(bc_loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc:.3f}")
            tqdm.write(
                f"  epoch {epoch+1:2d}/{args.epochs}  bc_loss={epoch_loss:.4f}  "
                f"action_acc={epoch_acc:.3f}"
            )
            summary.increment("updates_completed")
            summary.record(
                "epochs",
                {"epoch": epoch + 1, "bc_loss": epoch_loss, "action_acc": epoch_acc},
            )
            summary.write()

        final_acc = float(np.mean(accs))
        out_ckpt = Path(args.output_ckpt) if args.output_ckpt else STAGE1_CKPT
        save_checkpoint(
            out_ckpt, model,
            meta={"stage": "bc", "episodes": args.episodes, "epochs": args.epochs,
                  "action_acc": final_acc, "teachers": teacher_names},
        )
        summary.set("latest_checkpoint", str(out_ckpt))
        summary.set("final_action_acc", final_acc)
        print(f"\nSaved Stage-1 BC checkpoint → {out_ckpt}")
        print("Next: python ae_rl/train_stage2_ppo.py")


if __name__ == "__main__":
    main()
