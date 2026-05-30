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


# ── Comparison printer ────────────────────────────────────────────────────────

def compare(prefix: list[str], predicted: list[str], ground_truth: list[str], label: str = ""):
    # strip BLK tokens for validator
    all_steps = [t for t in prefix + predicted if not t.startswith("[BLK:")]
    violations = validate_sequence(all_steps)

    # token accuracy (step tokens only, ignoring BLK boundaries)
    pred_steps = [t for t in predicted   if not t.startswith("[BLK:")]
    gt_steps   = [t for t in ground_truth if not t.startswith("[BLK:")]
    n = min(len(pred_steps), len(gt_steps))
    matches = sum(p == g for p, g in zip(pred_steps[:n], gt_steps[:n]))

    sep = "─" * 70
    print(f"\n{sep}")
    if label:
        print(f"  {label}")
    print(f"  prefix   : {len(prefix)} tokens  |  to predict: {len(ground_truth)} tokens")
    print(f"  validator: {'✓ OK' if not violations else '✗ ' + violations[0].rule}")
    print(f"  token acc: {matches}/{n} = {matches/n:.1%}  (step tokens only)")
    print(sep)

    # side-by-side table
    print(f"  {'#':<4}  {'PREDICTED':<45}  GROUND TRUTH")
    print(f"  {'─'*4}  {'─'*45}  {'─'*45}")
    for i in range(max(len(pred_steps), len(gt_steps))):
        p = pred_steps[i] if i < len(pred_steps) else "—"
        g = gt_steps[i]   if i < len(gt_steps)   else "—"
        mark = "  " if p == g else "≠ "
        print(f"  {mark}{i+1:<3}  {p:<45}  {g}")
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
