"""
train.py — Hierarchical GPT POC for semiconductor fab sequences.

Activate venv, then run from this directory:
    python3.12 train.py

Trains a small GPT on augmented token streams where [BLK:X] boundary tokens
mark block transitions. Saves checkpoint + vocab to model_out/.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from hierarchical_tokenizer import (
    BLOCKS,
    generate_labeled_sequence,
    labeled_to_steps,
    to_augmented_tokens,
    validate_sequence,
)
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel

# ── Config ────────────────────────────────────────────────────────────────────

FAMILIES = ["mosfet", "igbt", "ic"]
N_PER_FAMILY = 10000
EPOCHS = 50
BATCH_SIZE = 64
LR = 3e-4
OUT_DIR = Path("model_out")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"

# ── Vocab ─────────────────────────────────────────────────────────────────────


def build_vocab(seed: int = 0) -> dict[str, int]:
    """Collect every unique token that appears across all families."""
    vocab = {PAD: 0, BOS: 1, EOS: 2}
    rng = random.Random(seed)
    for family in FAMILIES:
        for _ in range(300):
            for tok in to_augmented_tokens(generate_labeled_sequence(family, rng)):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab


# ── Dataset ───────────────────────────────────────────────────────────────────


class FabDataset(Dataset):
    def __init__(self, vocab: dict[str, int], n_per_family: int, seed: int):
        self.seqs: list[list[int]] = []
        rng = random.Random(seed)
        bos, eos = vocab[BOS], vocab[EOS]
        for family in FAMILIES:
            for _ in range(n_per_family):
                toks = to_augmented_tokens(generate_labeled_sequence(family, rng))
                ids = [bos] + [vocab.get(t, vocab[PAD]) for t in toks] + [eos]
                self.seqs.append(ids)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.long)


def collate(batch):
    L = max(x.size(0) for x in batch)
    out = torch.zeros(len(batch), L, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, : x.size(0)] = x
    return out


# ── Training ──────────────────────────────────────────────────────────────────


def _save_sequences(dataset: "FabDataset", vocab: dict[str, int], path: Path) -> None:
    id2tok = {v: k for k, v in vocab.items()}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SEQUENCE_ID", "STEP"])
        for i, ids in enumerate(dataset.seqs, start=1):
            seq_id = f"seq_{i:05d}"
            for tok_id in ids:
                tok = id2tok[tok_id]
                if tok in ("<bos>", "<eos>", "<pad>"):
                    continue
                writer.writerow([seq_id, tok])


def train(save_data: bool = False):
    OUT_DIR.mkdir(exist_ok=True)

    print("Building vocab …")
    vocab = build_vocab()
    id2tok = {v: k for k, v in vocab.items()}
    print(f"  vocab size: {len(vocab)}")
    with open(OUT_DIR / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"Generating {N_PER_FAMILY * len(FAMILIES)} training sequences …")
    dataset = FabDataset(vocab, n_per_family=N_PER_FAMILY, seed=42)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
    )
    print(f"  {len(dataset)} sequences, {len(loader)} batches/epoch")

    if save_data:
        _save_sequences(dataset, vocab, OUT_DIR / "training_sequences.csv")
        print(f"  training sequences saved to {OUT_DIR}/training_sequences.csv")

    cfg = GPT2Config(
        vocab_size=len(vocab),
        n_positions=256,
        n_embd=512,
        n_layer=6,
        n_head=8,
        n_inner=2048,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        pad_token_id=vocab[PAD],
        bos_token_id=vocab[BOS],
        eos_token_id=vocab[EOS],
    )
    model = GPT2LMHeadModel(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} params  device: {DEVICE}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits = model(inp).logits
            loss = F.cross_entropy(
                logits.reshape(-1, len(vocab)),
                tgt.reshape(-1),
                ignore_index=vocab[PAD],
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        print(f"  epoch {epoch:>2}/{EPOCHS}  loss={total_loss / len(loader):.4f}")

    model.save_pretrained(OUT_DIR)
    cfg.save_pretrained(OUT_DIR)
    print(f"\nCheckpoint saved to {OUT_DIR}/")
    return model, vocab, id2tok


# ── Greedy completion ─────────────────────────────────────────────────────────


@torch.no_grad()
def complete(
    model: GPT2LMHeadModel,
    vocab: dict[str, int],
    id2tok: dict[int, str],
    prefix: list[str],
    max_new: int = 120,
) -> list[str]:
    model.eval()
    eos_id = vocab[EOS]
    ids = torch.tensor(
        [[vocab[BOS]] + [vocab.get(t, vocab[PAD]) for t in prefix]],
        dtype=torch.long,
        device=DEVICE,
    )
    generated = []
    for _ in range(max_new):
        next_id = model(ids).logits[0, -1].argmax().item()
        if next_id == eos_id:
            break
        generated.append(id2tok[next_id])
        ids = torch.cat([ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
    return generated


# ── Demo ──────────────────────────────────────────────────────────────────────


def demo(model, vocab, id2tok):
    print("\n" + "─" * 60)
    print("COMPLETION DEMO")
    print("─" * 60)
    rng = random.Random(999)
    for family in FAMILIES:
        labeled = generate_labeled_sequence(family, rng)
        full_toks = to_augmented_tokens(labeled)
        cut = len(full_toks) * 4 // 10
        prefix = full_toks[:cut]
        ground_truth = full_toks[cut:]
        predicted = complete(model, vocab, id2tok, prefix)

        all_steps = [t for t in prefix + predicted if not t.startswith("[BLK:")]
        violations = validate_sequence(all_steps)

        print(f"\n{family.upper()}")
        print(f"  prefix ends : … {' | '.join(prefix[-3:])}")
        print(f"  predicted   : {' | '.join(predicted[:8])} …")
        print(f"  ground truth: {' | '.join(ground_truth[:8])} …")
        print(f"  validator   : {'OK' if not violations else violations[0].rule}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-data", action="store_true",
                        help="Save generated training sequences to model_out/training_sequences.csv")
    args = parser.parse_args()

    model, vocab, id2tok = train(save_data=args.save_data)
    demo(model, vocab, id2tok)
