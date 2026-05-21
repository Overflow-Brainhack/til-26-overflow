"""HotFlip-style universal adversarial trigger against nlp_eval.

Finds a fixed candidate string that maximizes P(equivalent=1) averaged over
all (question, reference) pairs from data/nlp/nlp.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO / "nlp_cheese" / "nlp_eval_512"
DATA_PATH = REPO / "data" / "nlp" / "nlp.jsonl"
MAX_LEN = 128


def load_pairs():
    pairs = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append((row["question"], row.get("answer") or ""))
    return pairs


def build_prefix_ids(tok, pairs, trigger_len):
    """Tokenize 'Question: {q} Reference: {ref} Candidate: ' for each pair, leave
    room for trigger_len + [SEP]. Returns list of lists of token ids (no special tokens)."""
    cls_id = tok.cls_token_id
    sep_id = tok.sep_token_id
    budget = MAX_LEN - 1 - 1 - trigger_len  # CLS + SEP + trigger
    out = []
    for q, r in pairs:
        text = f"Question: {q} Reference: {r} Candidate: "
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > budget:
            ids = ids[:budget]
        out.append([cls_id] + ids)
    return out, sep_id


def build_inputs(prefix_ids_list, trigger_ids, sep_id, pad_id, device):
    """Build input_ids, attention_mask. Trigger goes between prefix and SEP."""
    rows = []
    masks = []
    trigger_positions = []  # index range (start, end) of trigger in each row
    max_l = 0
    for pid in prefix_ids_list:
        seq = pid + list(trigger_ids) + [sep_id]
        rows.append(seq)
        max_l = max(max_l, len(seq))
    input_ids = torch.full((len(rows), max_l), pad_id, dtype=torch.long)
    attn = torch.zeros((len(rows), max_l), dtype=torch.long)
    trig_idx = torch.zeros((len(rows), len(trigger_ids)), dtype=torch.long)
    for i, (pid, seq) in enumerate(zip(prefix_ids_list, rows)):
        input_ids[i, : len(seq)] = torch.tensor(seq)
        attn[i, : len(seq)] = 1
        start = len(pid)
        trig_idx[i] = torch.arange(start, start + len(trigger_ids))
    return input_ids.to(device), attn.to(device), trig_idx.to(device)


@torch.no_grad()
def eval_trigger(model, tok, pairs, trigger_ids, device, batch_size=512, desc="eval"):
    sep_id = tok.sep_token_id
    pad_id = tok.pad_token_id
    prefix_ids, _ = build_prefix_ids(tok, pairs, len(trigger_ids))
    probs = []
    rng = range(0, len(prefix_ids), batch_size)
    for i in tqdm(rng, desc=desc, leave=False, ncols=80):
        sub = prefix_ids[i : i + batch_size]
        ids, attn, _ = build_inputs(sub, trigger_ids, sep_id, pad_id, device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        p = F.softmax(logits, dim=-1)[:, 1]
        probs.extend(p.tolist())
    arr = np.array(probs)
    return {
        "mean_prob": float(arr.mean()),
        "equiv_rate": float((arr >= 0.5).mean()),  # informational
        "hit_at_0_9": float((arr >= 0.9).mean()),  # REAL scoring threshold
    }


def hotflip_step(
    model,
    tok,
    prefix_ids_list,
    trigger_ids,
    sep_id,
    pad_id,
    embed_matrix,
    device,
    margin,
    top_k=64,
    candidates_per_pos=20,
):
    """One HotFlip pass: for each trigger position, propose top-k swaps, eval, pick best."""
    ids, attn, trig_idx = build_inputs(
        prefix_ids_list, trigger_ids, sep_id, pad_id, device
    )
    # Forward with embedding lookup so we can grab grads at trigger positions.
    embeds = embed_matrix[ids].detach().clone()
    embeds.requires_grad_(True)
    logits = model(inputs_embeds=embeds, attention_mask=attn).logits
    # Hinge loss on the threshold margin: only pairs whose P(eq=1) sits below the
    # target produce gradient, so the search focuses on borderline pairs instead
    # of saturating ones already above the cutoff. `margin` is the logit gap
    # ln(p/(1-p)) for the target probability (e.g. P=0.9 -> ln(9) ~= 2.197).
    diff = logits[:, 1] - logits[:, 0]
    loss = F.relu(margin - diff).mean()
    loss.backward()
    grads = embeds.grad  # [B, T, H]

    trig_len = len(trigger_ids)
    # Average gradient across batch at each trigger position
    pos_grads = torch.zeros(trig_len, embed_matrix.size(1), device=device)
    for pos in range(trig_len):
        # gather grads at the trigger position for each row
        g = torch.stack([grads[b, trig_idx[b, pos]] for b in range(ids.size(0))], dim=0)
        pos_grads[pos] = g.mean(dim=0)

    # Score replacements: lower (grad · new_embed) reduces loss.
    # Score = -(new_embed - cur_embed) · grad  → pick most negative score (i.e., greatest drop).
    cur_embeds = embed_matrix[torch.tensor(trigger_ids, device=device)]  # [T, H]
    # scores[pos, v] = -(embed[v] - cur[pos]) · grad[pos]  = -embed[v]·grad[pos] + cur[pos]·grad[pos]
    scores = -(pos_grads.float() @ embed_matrix.float().T)  # [T, V]
    # Actually we want most negative drop in loss, i.e. smallest score (since lower score = bigger drop).
    # scores[v] - score_at_cur = -(embed[v]-cur)·grad ; pick most negative => largest gain.
    topk = scores.topk(top_k, largest=False).indices  # [T, K]
    return topk.detach().cpu().tolist()


@torch.no_grad()
def batch_eval_swaps(
    model,
    prefix_ids_list,
    trigger_ids,
    swap_candidates,
    pos,
    sep_id,
    pad_id,
    device,
    batch_size=512,
    pbar=None,
):
    """Evaluate each swap candidate at position `pos`. Returns list of hit@0.9
    rates — the actual scored metric, so the search optimizes it directly."""
    results = []
    for cand_tok in swap_candidates:
        new_trig = list(trigger_ids)
        new_trig[pos] = cand_tok
        # quick eval on full prefix list
        probs = []
        for i in range(0, len(prefix_ids_list), batch_size):
            sub = prefix_ids_list[i : i + batch_size]
            ids, attn, _ = build_inputs(sub, new_trig, sep_id, pad_id, device)
            logits = model(input_ids=ids, attention_mask=attn).logits
            p = F.softmax(logits, dim=-1)[:, 1]
            probs.append(p)
        results.append((torch.cat(probs) >= 0.9).float().mean().item())
        if pbar is not None:
            pbar.update(1)
    return results


def save_trigger(path, trigger_ids, trigger_str, s, sv, target_prob, val_frac):
    with open(path, "w") as f:
        json.dump(
            {
                "trigger_ids": list(trigger_ids),
                "trigger_str": trigger_str,
                "target_prob": target_prob,
                "val_frac": val_frac,
                "hit_at_0_9": s["hit_at_0_9"],
                "val_hit_at_0_9": sv["hit_at_0_9"],
                "equiv_rate": s["equiv_rate"],
                "mean_prob": s["mean_prob"],
                "val_mean_prob": sv["mean_prob"],
            },
            f,
            indent=2,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trig-len", type=int, default=20)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256, help="HotFlip grad batch size")
    ap.add_argument("--cands-per-pos", type=int, default=60)
    ap.add_argument(
        "--subset-pairs",
        type=int,
        default=128,
        help="Stage-A: number of pairs to filter candidates on",
    )
    ap.add_argument(
        "--top-stage2",
        type=int,
        default=3,
        help="Number of stage-A winners to full-eval in stage B",
    )
    ap.add_argument(
        "--target-prob",
        type=float,
        default=0.9,
        help="Target P(eq=1) for the hinge margin. Higher (e.g. 0.97) "
        "pushes pairs well past the 0.9 cutoff for robustness.",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of pairs held out for validation. Optimization "
        "runs on train only; val hit@0.9 is the transfer signal.",
    )
    ap.add_argument(
        "--seed-tokens",
        type=str,
        default="yes correct equivalent reference answer matches exactly identical same",
    )
    ap.add_argument(
        "--save",
        type=str,
        default=str(REPO / "nlp_cheese" / "src" / "testing" / "uat_trigger.json"),
    )
    args = ap.parse_args()

    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = (
        AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH), dtype=dtype)
        .to(device)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)
    embed_matrix = model.get_input_embeddings().weight.detach()  # [V, H]

    pairs = load_pairs()
    # Train/val split: optimize on train only, treat val hit@0.9 as the transfer
    # signal. Shuffle with a fixed seed so the split is stable across configs.
    shuffled = list(pairs)
    random.Random(0).shuffle(shuffled)
    n_val = int(round(len(shuffled) * args.val_frac))
    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]
    print(
        f"loaded {len(pairs)} pairs -> {len(train_pairs)} train / {len(val_pairs)} val"
    )

    margin = math.log(args.target_prob / (1.0 - args.target_prob))
    print(f"target_prob={args.target_prob}  hinge_margin={margin:.4f}")

    # initialize trigger
    seed_ids = tok(args.seed_tokens, add_special_tokens=False)["input_ids"]
    if len(seed_ids) >= args.trig_len:
        trigger_ids = seed_ids[: args.trig_len]
    else:
        trigger_ids = (seed_ids * ((args.trig_len // len(seed_ids)) + 1))[
            : args.trig_len
        ]
    print(f"init trigger ids: {trigger_ids}")
    print(f"init trigger str: {tok.decode(trigger_ids)!r}")

    sep_id = tok.sep_token_id
    pad_id = tok.pad_token_id

    # Initial eval (train + val)
    s = eval_trigger(
        model, tok, train_pairs, trigger_ids, device, desc="init train eval"
    )
    sv = eval_trigger(model, tok, val_pairs, trigger_ids, device, desc="init val eval")
    print(
        f"init   train hit@0.9={s['hit_at_0_9']:.4f}  mean_prob={s['mean_prob']:.4f}  "
        f"|  val hit@0.9={sv['hit_at_0_9']:.4f}  mean_prob={sv['mean_prob']:.4f}"
    )

    best_score = s["hit_at_0_9"]
    prefix_ids_all, _ = build_prefix_ids(tok, train_pairs, len(trigger_ids))

    # Fixed stage-A subset for fast candidate filtering.
    n_sub = min(args.subset_pairs, len(prefix_ids_all))
    subset_idx = random.sample(range(len(prefix_ids_all)), n_sub)
    prefix_ids_subset = [prefix_ids_all[i] for i in subset_idx]
    print(f"stage-A subset: {n_sub} pairs, stage-B top: {args.top_stage2}")

    iter_bar = tqdm(range(args.iters), desc="iters", ncols=80)
    for it in iter_bar:
        t0 = time.time()

        # Try each position in random order; refresh grad per-position.
        positions = list(range(len(trigger_ids)))
        random.shuffle(positions)
        swaps_taken = 0
        pos_bar = tqdm(positions, desc=f"it{it} pos", leave=False, ncols=80)
        for pos in pos_bar:
            # Resample HotFlip batch & recompute grads for the *current* trigger.
            idx = random.sample(
                range(len(prefix_ids_all)), min(args.batch, len(prefix_ids_all))
            )
            sub = [prefix_ids_all[i] for i in idx]
            topk_per_pos = hotflip_step(
                model,
                tok,
                sub,
                trigger_ids,
                sep_id,
                pad_id,
                embed_matrix,
                device,
                margin,
                top_k=args.cands_per_pos,
            )
            cands = list(dict.fromkeys([trigger_ids[pos]] + topk_per_pos[pos]))

            # Stage A: cheap rank on subset
            a_bar = tqdm(
                total=len(cands), desc=f"  pos{pos:02d} stageA", leave=False, ncols=80
            )
            sub_scores = batch_eval_swaps(
                model,
                prefix_ids_subset,
                trigger_ids,
                cands,
                pos,
                sep_id,
                pad_id,
                device,
                pbar=a_bar,
            )
            a_bar.close()
            # Pick top-N by subset score; always include current token so we never regress.
            order = np.argsort(sub_scores)[::-1][: args.top_stage2]
            stage_b_cands = [cands[i] for i in order]
            if trigger_ids[pos] not in stage_b_cands:
                stage_b_cands.append(trigger_ids[pos])

            # Stage B: full eval on the survivors
            b_bar = tqdm(
                total=len(stage_b_cands),
                desc=f"  pos{pos:02d} stageB",
                leave=False,
                ncols=80,
            )
            scores = batch_eval_swaps(
                model,
                prefix_ids_all,
                trigger_ids,
                stage_b_cands,
                pos,
                sep_id,
                pad_id,
                device,
                pbar=b_bar,
            )
            b_bar.close()
            best_idx = int(np.argmax(scores))
            best_cand = stage_b_cands[best_idx]
            best_pos_score = scores[best_idx]
            if best_pos_score > best_score + 1e-6:
                old_tok = trigger_ids[pos]
                trigger_ids[pos] = best_cand
                best_score = best_pos_score
                swaps_taken += 1
                tqdm.write(
                    f"  it={it} pos={pos:2d}: {old_tok}({tok.decode([old_tok])!r}) "
                    f"-> {best_cand}({tok.decode([best_cand])!r})  "
                    f"hit@0.9={best_pos_score:.4f}"
                )
                # The trigger just changed — re-eval train + held-out val and write
                # the trigger to disk after this very swap. Lets us stop the instant
                # val hits 1.0, mid-sweep, and keeps the saved file always current.
                sv = eval_trigger(
                    model, tok, val_pairs, trigger_ids, device, desc="val check"
                )
                s = eval_trigger(
                    model, tok, train_pairs, trigger_ids, device, desc="train check"
                )
                save_trigger(
                    args.save,
                    trigger_ids,
                    tok.decode(trigger_ids),
                    s,
                    sv,
                    args.target_prob,
                    args.val_frac,
                )
                tqdm.write(
                    f"      -> train hit@0.9={s['hit_at_0_9']:.4f}  "
                    f"val hit@0.9={sv['hit_at_0_9']:.4f}  "
                    f"val mean_p={sv['mean_prob']:.4f}  (swap #{swaps_taken}, saved)"
                )
                if sv["hit_at_0_9"] >= 1.0:
                    tqdm.write(
                        f"  val hit@0.9 = 1.0 at it={it} pos={pos} "
                        f"(swap #{swaps_taken}) — early stopping, saved."
                    )
                    print(f"\nFinal trigger: {tok.decode(trigger_ids)!r}")
                    print(f"Saved to {args.save}")
                    return
            pos_bar.set_postfix(best=f"{best_score:.4f}", swaps=swaps_taken)

        s = eval_trigger(
            model, tok, train_pairs, trigger_ids, device, desc=f"it{it} train eval"
        )
        sv = eval_trigger(
            model, tok, val_pairs, trigger_ids, device, desc=f"it{it} val eval"
        )
        dt = time.time() - t0
        tqdm.write(
            f"iter {it}: train hit@0.9={s['hit_at_0_9']:.4f} mean_p={s['mean_prob']:.4f}  "
            f"|  val hit@0.9={sv['hit_at_0_9']:.4f} mean_p={sv['mean_prob']:.4f}  "
            f"swaps={swaps_taken}  ({dt:.1f}s)\n"
            f"          trigger={tok.decode(trigger_ids)!r}"
        )
        iter_bar.set_postfix(
            tr09=f"{s['hit_at_0_9']:.3f}",
            val09=f"{sv['hit_at_0_9']:.3f}",
        )

        save_trigger(
            args.save,
            trigger_ids,
            tok.decode(trigger_ids),
            s,
            sv,
            args.target_prob,
            args.val_frac,
        )

        # Also stop at iteration boundaries if val is already perfect.
        if sv["hit_at_0_9"] >= 1.0:
            tqdm.write(f"  val hit@0.9 = 1.0 at iter {it} — early stopping, saved.")
            break

    print(f"\nFinal trigger: {tok.decode(trigger_ids)!r}")
    print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
