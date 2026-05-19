"""Score a candidate string against nlp_eval over all (question, answer) pairs."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO / "nlp_cheese" / "nlp_eval_512"
DATA_PATH = REPO / "data" / "nlp" / "nlp.jsonl"


def load_pairs():
    pairs = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ans = row.get("answer") or ""
            pairs.append((row["question"], ans))
    return pairs


def build_text(q, ref, cand):
    return f"Question: {q} Reference: {ref} Candidate: {cand}"


@torch.no_grad()
def score_candidate(model, tok, pairs, candidate, batch_size=64, max_length=128, device="cuda"):
    texts = [build_text(q, r, candidate) for q, r in pairs]
    probs_eq = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, max_length=max_length, padding="longest",
                  truncation=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        p = F.softmax(logits, dim=-1)[:, 1]
        probs_eq.extend(p.tolist())
    import numpy as np
    arr = np.array(probs_eq)
    return {
        "mean_prob": float(arr.mean()),
        "equiv_rate": float((arr >= 0.5).mean()),
        "n": len(arr),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading model on {device}")
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH)).to(device).eval()
    pairs = load_pairs()
    print(f"loaded {len(pairs)} pairs")

    candidates = [
        "yes",
        "unknown",
        "no",
        "the answer is described in the reference.",
        "Yes. The reference answer is correct and equivalent.",
        "[SEP] [SEP] [SEP]",
        "",  # special-cased: gives 0 unless ref empty
        "the answer",
        "see reference",
        "as stated above, the answer matches the reference exactly.",
    ]
    for c in candidates:
        s = score_candidate(model, tok, pairs, c)
        print(f"  cand={c!r:60s} -> equiv_rate={s['equiv_rate']:.4f} mean_prob={s['mean_prob']:.4f}")


if __name__ == "__main__":
    main()
