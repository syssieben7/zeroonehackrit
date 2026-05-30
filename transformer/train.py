#!/usr/bin/env python3
"""
train.py — Fine-tune T5-small on semiconductor process sequences.

Tasks (all handled by one model via task prefixes):
  - next step <family>: <prefix>  →  <next step>
  - complete <family>: <prefix>   →  <remaining steps>
  - validate <family>: <sequence> →  "valid" | "invalid <RULE>"

Usage (local prototype):
    python train.py --model t5-small --data_dir ./data --out_dir ./checkpoints

Usage (larger model on cluster):
    python train.py --model google/flan-t5-base --data_dir ./data --out_dir ./checkpoints \
        --epochs 5 --batch_size 32 --max_input_len 512
"""

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_input_len: int, max_output_len: int, max_samples: int = 0):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        # store raw strings only — tokenize lazily in __getitem__
        self.examples = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append((ex["input"], ex["output"], ex.get("task", "unknown")))
                if max_samples and len(self.examples) >= max_samples:
                    break
        print(f"  Loaded {len(self.examples)} examples from {path.name}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        inp, out, task = self.examples[idx]
        enc = self.tokenizer(
            inp,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dec = self.tokenizer(
            out,
            max_length=self.max_output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = dec.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze(),
            "labels":         labels,
            "task":           task,
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(model, loader, device, tokenizer, max_output_len):
    model.eval()
    counts = {"next_step": [0, 0], "completion": [0, 0], "validate": [0, 0], "unknown": [0, 0]}

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            tasks          = batch["task"]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_output_len,
            )
            preds   = tokenizer.batch_decode(generated, skip_special_tokens=True)
            targets = labels.clone()
            targets[targets == -100] = tokenizer.pad_token_id
            refs    = tokenizer.batch_decode(targets, skip_special_tokens=True)

            for pred, ref, task in zip(preds, refs, tasks):
                bucket = counts.get(task, counts["unknown"])
                bucket[0] += 1
                if pred.strip() == ref.strip():
                    bucket[1] += 1

    model.train()

    per_task = {t: (v[1] / v[0] if v[0] else 0.0) for t, v in counts.items() if v[0] > 0}
    total    = sum(v[0] for v in counts.values())
    correct  = sum(v[1] for v in counts.values())
    overall  = correct / total if total else 0.0

    lines = "  |  ".join(f"{t}: {a:.1%}" for t, a in per_task.items())
    print(f"  Per-task: {lines}")

    return overall


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args):
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Model:  {args.model}")

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)

    data_dir = Path(args.data_dir)
    train_ds = SequenceDataset(
        data_dir / "train.jsonl", tokenizer, args.max_input_len, args.max_output_len, args.max_samples
    )
    val_ds = SequenceDataset(
        data_dir / "val.jsonl", tokenizer, args.max_input_len, args.max_output_len
    )

    def collate_fn(batch):
        return {
            "input_ids":      torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels":         torch.stack([b["labels"] for b in batch]),
            "task":           [b["task"] for b in batch],
        }

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    if args.cosine_lr:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            ).loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if step % args.log_every == 0:
                avg = total_loss / step
                print(
                    f"  Epoch {epoch} | step {step}/{len(train_loader)} | loss {avg:.4f}"
                )

        val_acc = evaluate(model, val_loader, device, tokenizer, args.max_output_len)
        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch} done — loss {avg_loss:.4f} | val exact-match {val_acc:.3%}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(out_dir / "best")
            tokenizer.save_pretrained(out_dir / "best")
            print(f"  Saved best checkpoint (val acc {best_acc:.3%})")

    print(f"\nTraining complete. Best val exact-match: {best_acc:.3%}")
    print(f"Checkpoints in: {out_dir}/best")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="t5-small", help="HuggingFace model name or local path"
    )
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--out_dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=256,
        help="Max tokens for input sequence prefix",
    )
    parser.add_argument(
        "--max_output_len",
        type=int,
        default=512,
        help="Max tokens for output (next step or completion)",
    )
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit training to first N samples (0 = use all)")
    parser.add_argument("--cosine_lr", action="store_true",
                        help="Use cosine LR schedule instead of linear warmup")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
