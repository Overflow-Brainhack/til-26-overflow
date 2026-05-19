"""Score a single trigger (string or token-id list) against nlp_eval.

Reports mean_prob, hit@0.5, hit@0.9 over all (question, gold) pairs in nlp.jsonl.
The real scorer uses threshold 0.9.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO / "nlp_cheese" / "nlp_eval_512"
DATA_PATH = REPO / "data" / "nlp" / "nlp.jsonl"

# Iter-3 trigger from the length-20 HotFlip run (best result).
ITER3_TRIGGER_IDS = [
    9820, 3451, 6425, 3806, 8650, 38241, 4555, 1416, 41970, 9820,
    3451, 6425, 281, 3662, 4764, 4555, 265, 5218, 6300, 3451,
]


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


@torch.no_grad()
def score(model, tok, pairs, candidate_str, device, max_length=128, batch_size=512):
    probs = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="eval", ncols=80):
        batch = pairs[i : i + batch_size]
        texts = [
            f"Question: {q} Reference: {r} Candidate: {candidate_str}" for q, r in batch
        ]
        enc = tok(
            texts, max_length=max_length, padding="longest",
            truncation=True, return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        probs.extend(F.softmax(logits, dim=-1)[:, 1].tolist())
    arr = np.array(probs)
    # Emulate the official empty-string short-circuit:
    # if reference is empty and candidate is non-empty -> not equivalent (prob_eq treated as 0)
    if candidate_str != "":
        for i, (_, r) in enumerate(pairs):
            if r == "":
                arr[i] = 0.0
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger-str", type=str, default=None,
                    help="Use this literal candidate string")
    ap.add_argument("--trigger-ids", type=str, default=None,
                    help="Comma-separated token ids (decoded via eval tokenizer)")
    ap.add_argument("--from-json", type=str, default=None,
                    help="Path to uat_trigger.json saved by uat_hotflip")
    ap.add_argument("--use-iter3", action="store_true",
                    help="Use the hardcoded iter-3 trigger ids")
    ap.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH),
                    help="Eval model directory (default: nlp_cheese/nlp_eval)")
    ap.add_argument("--max-length", type=int, default=128,
                    help="Tokenizer max_length (test_nlp uses 128; new model may use 512)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.model_path
    print(f"loading model from {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = (
        AutoModelForSequenceClassification.from_pretrained(model_path, dtype=dtype)
        .to(device).eval()
    )

    if args.use_iter3:
        trigger_ids = ITER3_TRIGGER_IDS
        candidate_str = tok.decode(trigger_ids)
    elif args.trigger_ids:
        trigger_ids = [int(x) for x in args.trigger_ids.split(",")]
        candidate_str = tok.decode(trigger_ids)
    elif args.from_json:
        with open(args.from_json) as f:
            j = json.load(f)
        trigger_ids = j["trigger_ids"]
        candidate_str = j.get("trigger_str") or tok.decode(trigger_ids)
    elif args.trigger_str is not None:
        candidate_str = args.trigger_str
        trigger_ids = None
    else:
        ap.error("provide one of --trigger-str / --trigger-ids / --from-json / --use-iter3")

    pairs = load_pairs()
    print(f"loaded {len(pairs)} pairs")
    print(f"candidate string: {candidate_str!r}")
    if trigger_ids is not None:
        print(f"trigger ids ({len(trigger_ids)}): {trigger_ids}")

    arr = score(model, tok, pairs, candidate_str, device, max_length=args.max_length)
    print()
    print(f"mean_prob       : {arr.mean():.4f}")
    print(f"hit@0.5         : {(arr >= 0.5).mean():.4f}")
    print(f"hit@0.9 (REAL)  : {(arr >= 0.9).mean():.4f}")
    print(f"hit@0.95        : {(arr >= 0.95).mean():.4f}")
    print(f"hit@0.99        : {(arr >= 0.99).mean():.4f}")
    # Distribution
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.01]
    hist, _ = np.histogram(arr, bins=bins)
    print("prob_eq distribution:")
    for lo, hi, c in zip(bins[:-1], bins[1:], hist):
        print(f"  [{lo:.2f},{hi:.2f}): {c:4d}")


if __name__ == "__main__":
    main()
