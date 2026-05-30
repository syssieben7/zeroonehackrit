"""
infer.py — Load a saved checkpoint and complete a partial sequence.

Usage:
    pixi run python infer.py                                      # built-in demo
    pixi run python infer.py "RECEIVE WAFER LOT|LOT IDENTIFICATION|..."  # custom prefix
    pixi run python infer.py --csv IC_variants.csv --seq seq_0001 --cut 0.6
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from hierarchical_tokenizer import (
    generate_labeled_sequence,
    to_augmented_tokens,
    validate_sequence,
)

MODEL_DIR = Path("model_out")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


# ── Model loading ─────────────────────────────────────────────────────────────

def load(model_dir: Path = MODEL_DIR):
    vocab  = json.loads((model_dir / "vocab.json").read_text())
    id2tok = {v: k for k, v in vocab.items()}
    model  = GPT2LMHeadModel.from_pretrained(model_dir).to(DEVICE).eval()
    return model, vocab, id2tok


# ── Greedy completion ─────────────────────────────────────────────────────────

@torch.no_grad()
def complete(model, vocab, id2tok, prefix: list[str], max_new: int = 200) -> list[str]:
    eos_id = vocab[EOS]
    ids = torch.tensor(
        [[vocab[BOS]] + [vocab.get(t, vocab[PAD]) for t in prefix]],
        dtype=torch.long, device=DEVICE,
    )
    generated = []
    for _ in range(max_new):
        next_id = model(ids).logits[0, -1].argmax().item()
        if next_id == eos_id:
            break
        generated.append(id2tok[next_id])
        ids = torch.cat([ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
    return generated


# ── Edit distance + alignment ─────────────────────────────────────────────────

def edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two token lists."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[n]


def align(pred: list[str], gt: list[str]) -> list[tuple[str, str]]:
    """Needleman-Wunsch alignment — returns (pred_tok, gt_tok) pairs with gaps as '—'."""
    m, n = len(pred), len(gt)
    GAP = -1
    MATCH, MISMATCH, INDEL = 2, -1, -1

    score = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): score[i][0] = i * INDEL
    for j in range(n + 1): score[0][j] = j * INDEL

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = MATCH if pred[i-1] == gt[j-1] else MISMATCH
            score[i][j] = max(score[i-1][j-1] + s, score[i-1][j] + INDEL, score[i][j-1] + INDEL)

    aligned = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = MATCH if pred[i-1] == gt[j-1] else MISMATCH
            if score[i][j] == score[i-1][j-1] + s:
                aligned.append((pred[i-1], gt[j-1])); i -= 1; j -= 1; continue
        if i > 0 and (j == 0 or score[i][j] == score[i-1][j] + INDEL):
            aligned.append((pred[i-1], "—")); i -= 1
        else:
            aligned.append(("—", gt[j-1])); j -= 1

    return list(reversed(aligned))


# ── Comparison printer ────────────────────────────────────────────────────────

def compare(prefix: list[str], predicted: list[str], ground_truth: list[str], label: str = ""):
    all_steps  = [t for t in prefix + predicted if not t.startswith("[BLK:")]
    violations = validate_sequence(all_steps)

    pred_steps = [t for t in predicted    if not t.startswith("[BLK:")]
    gt_steps   = [t for t in ground_truth if not t.startswith("[BLK:")]

    ed       = edit_distance(pred_steps, gt_steps)
    max_len  = max(len(pred_steps), len(gt_steps))
    norm_ed  = ed / max_len if max_len else 0.0
    aligned  = align(pred_steps, gt_steps)
    matches  = sum(1 for p, g in aligned if p == g and p != "—")

    sep = "─" * 70
    print(f"\n{sep}")
    if label:
        print(f"  {label}")
    print(f"  prefix     : {len(prefix)} tokens  |  to predict: {len(gt_steps)} steps")
    print(f"  validator  : {'✓ OK' if not violations else '✗ ' + violations[0].rule}")
    print(f"  edit dist  : {ed}  (normalised: {norm_ed:.3f}, lower=better)")
    print(f"  aligned acc: {matches}/{len(gt_steps)} = {matches/len(gt_steps):.1%}")
    print(sep)

    print(f"  {'#':<4}  {'PREDICTED':<45}  GROUND TRUTH")
    print(f"  {'─'*4}  {'─'*45}  {'─'*45}")
    for i, (p, g) in enumerate(aligned, 1):
        mark = "  " if p == g else "≠ "
        print(f"  {mark}{i:<3}  {p:<45}  {g}")
    print()


# ── CSV reader ────────────────────────────────────────────────────────────────

def read_sequence_from_csv(csv_path: Path, seq_id: str) -> list[str]:
    steps = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["SEQUENCE_ID"].strip() == seq_id:
                steps.append(row["STEP"].strip())
    if not steps:
        raise ValueError(f"Sequence '{seq_id}' not found in {csv_path}")
    return steps


# ── Modes ─────────────────────────────────────────────────────────────────────

def mode_csv(args):
    model, vocab, id2tok = load()
    steps = read_sequence_from_csv(Path(args.csv), args.seq)
    cut   = int(len(steps) * args.cut)

    prefix_steps = steps[:cut]
    gt_steps     = steps[cut:]

    # augmented tokens don't exist for external CSV — pass steps directly
    # (model handles missing BLK tokens gracefully; they just aren't conditioned on)
    predicted_toks = complete(model, vocab, id2tok, prefix_steps)
    pred_steps     = [t for t in predicted_toks if not t.startswith("[BLK:")]

    print(f"\nCSV    : {args.csv}  |  seq: {args.seq}  |  cut: {args.cut:.0%}")
    print(f"Prefix : steps 1–{cut}  →  last 3: … {' | '.join(prefix_steps[-3:])}")
    compare(prefix_steps, pred_steps, gt_steps, label=f"{args.seq} @ {args.cut:.0%} cut")


def mode_prefix(prefix_str: str):
    model, vocab, id2tok = load()
    prefix = [s.strip() for s in prefix_str.split("|")]
    predicted = complete(model, vocab, id2tok, prefix)
    pred_steps = [t for t in predicted if not t.startswith("[BLK:")]
    all_steps  = prefix + pred_steps
    violations = validate_sequence(all_steps)
    print(f"\nPREFIX  : … {' | '.join(prefix[-5:])}")
    print(f"PREDICTED ({len(pred_steps)} steps):")
    for i, s in enumerate(pred_steps, 1):
        print(f"  {i:>3}. {s}")
    print(f"\nVALIDATOR: {'✓ OK' if not violations else '✗ ' + violations[0].rule}")


def mode_demo():
    model, vocab, id2tok = load()
    print(f"Model loaded from {MODEL_DIR}  ({DEVICE})\n")
    rng = random.Random(777)
    for family in ("mosfet", "igbt", "ic"):
        labeled     = generate_labeled_sequence(family, rng)
        full_toks   = to_augmented_tokens(labeled)
        cut         = len(full_toks) // 2
        prefix      = full_toks[:cut]
        ground_truth = full_toks[cut:]
        predicted   = complete(model, vocab, id2tok, prefix)
        compare(prefix, predicted, ground_truth, label=family.upper())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix",   nargs="?",  help="Pipe-separated steps for completion")
    parser.add_argument("--csv",    default=None, help="Path to variants CSV")
    parser.add_argument("--seq",    default="seq_0001", help="Sequence ID (default: seq_0001)")
    parser.add_argument("--cut",    type=float, default=0.6, help="Prefix fraction 0–1 (default: 0.6)")
    args = parser.parse_args()

    if args.csv:
        mode_csv(args)
    elif args.prefix:
        mode_prefix(args.prefix)
    else:
        mode_demo()
