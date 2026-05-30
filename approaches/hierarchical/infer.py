"""
infer.py — Load a saved checkpoint and complete a partial sequence.

Usage:
    pixi run python infer.py                        # runs built-in demo
    pixi run python infer.py "RECEIVE WAFER LOT|LOT IDENTIFICATION|INITIAL WAFER INSPECTION"
"""

import json
import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from hierarchical_tokenizer import to_augmented_tokens, generate_labeled_sequence, validate_sequence
import random

MODEL_DIR = Path("model_out")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


def load(model_dir: Path):
    vocab = json.loads((model_dir / "vocab.json").read_text())
    id2tok = {v: k for k, v in vocab.items()}
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(DEVICE).eval()
    return model, vocab, id2tok


@torch.no_grad()
def complete(model, vocab, id2tok, prefix: list[str], max_new: int = 120) -> list[str]:
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


def run(prefix_steps: list[str]):
    model, vocab, id2tok = load(MODEL_DIR)

    # Convert plain steps to augmented tokens (adds [BLK:X] boundaries)
    # If the input already contains [BLK:X] tokens, use as-is
    if any(t.startswith("[BLK:") for t in prefix_steps):
        prefix = prefix_steps
    else:
        # Wrap in augmented format: just pass steps, BLK tokens will be missing
        # so we generate without them (model handles it gracefully)
        prefix = prefix_steps

    predicted = complete(model, vocab, id2tok, prefix)

    all_steps = [t for t in prefix + predicted if not t.startswith("[BLK:")]
    violations = validate_sequence(all_steps)

    print(f"PREFIX  ({len(prefix)} tokens): … {' | '.join(prefix[-5:])}")
    print(f"PREDICTED ({len(predicted)} tokens):")
    print("  " + " | ".join(predicted))
    print(f"VALIDATOR: {'OK' if not violations else violations[0].rule}")
    return predicted


def demo():
    model, vocab, id2tok = load(MODEL_DIR)
    print(f"Loaded model from {MODEL_DIR}  (device: {DEVICE})\n")

    rng = random.Random(777)
    for family in ("mosfet", "igbt", "ic"):
        labeled = generate_labeled_sequence(family, rng)
        full_toks = to_augmented_tokens(labeled)
        cut = len(full_toks) // 2   # 50% prefix
        prefix = full_toks[:cut]
        ground_truth = full_toks[cut:]

        predicted = complete(model, vocab, id2tok, prefix)
        all_steps = [t for t in prefix + predicted if not t.startswith("[BLK:")]
        violations = validate_sequence(all_steps)

        # token accuracy over first N predicted tokens
        n = min(len(predicted), len(ground_truth))
        matches = sum(p == g for p, g in zip(predicted[:n], ground_truth[:n]))

        print(f"{'─'*60}")
        print(f"{family.upper()}  (prefix: {cut} tok → predicting {len(ground_truth)} tok)")
        print(f"  prefix ends   : … {' | '.join(prefix[-3:])}")
        print(f"  predicted     : {' | '.join(predicted[:10])} …")
        print(f"  ground truth  : {' | '.join(ground_truth[:10])} …")
        print(f"  token acc     : {matches}/{n} = {matches/n:.1%}")
        print(f"  validator     : {'OK' if not violations else violations[0].rule}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # pipe-separated steps from command line
        steps = sys.argv[1].split("|")
        run([s.strip() for s in steps])
    else:
        demo()
