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
from model import RecurrentMaskableActorCritic, save_checkpoint
from rollout import collect_teacher_dataset, default_workers
from run_summary import RunSummary, default_summary_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=48, help="teacher games to record (×6 agents = sequences)")
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

        print(f"Collecting teacher demonstrations: {args.episodes} games (6×heuristic), "
              f"{args.num_workers} worker(s) …")
        t0 = time.time()
        data = collect_teacher_dataset(n_episodes=args.episodes, novice=args.novice,
                                       progress=True, num_workers=args.num_workers)
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
        save_checkpoint(
            STAGE1_CKPT, model,
            meta={"stage": "bc", "episodes": args.episodes, "epochs": args.epochs,
                  "action_acc": final_acc},
        )
        summary.set("latest_checkpoint", str(STAGE1_CKPT))
        summary.set("final_action_acc", final_acc)
        print(f"\nSaved Stage-1 BC checkpoint → {STAGE1_CKPT}")
        print("Next: python ae_rl/train_stage2_ppo.py")


if __name__ == "__main__":
    main()
