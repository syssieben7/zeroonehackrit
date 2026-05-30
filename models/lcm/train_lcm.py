"""
Training script for Process-LCM (Large Concept Model for process sequences).

Uses the same data pipeline as the Seq2Seq model but trains a causal Transformer
with MSE + CE loss in embedding space.

Usage:
    python train_lcm.py -epochs 50 -batch_size 64 -data_dir ../../data/raw/infineon
"""

import os
import math
import argparse
import random

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

from lcm_model import ProcessLCM
from utils import (PAD, SOS, EOS, IGBT_TOK, IC_TOK, Vocab,
                   read_sequences, build_vocab)
from validator import validate_sequence


# ---------------------------------------------------------------------------
# Dataset: full sequences (no prefix/suffix split needed - causal LM style)
# ---------------------------------------------------------------------------

class LCMSequenceDataset(Dataset):
    """
    Dataset for Process-LCM: produces full sequences with family token prefix.
    The model learns to predict each next step autoregressively via causal mask.

    Also supports prefix-cut augmentation for training robustness.
    """

    def __init__(self, sequences, family_token_idx, vocab, augment_factor=3,
                 max_seq_len=300):
        self.vocab = vocab
        self.family_token_idx = family_token_idx
        self.sequences = []

        for steps in sequences:
            encoded = vocab.encode(steps)
            if len(encoded) < 5:
                continue

            # Full sequence (primary training signal)
            full = [family_token_idx] + encoded + [EOS]
            if len(full) <= max_seq_len:
                self.sequences.append(full)

            # Augmented partial sequences (random start/end cuts)
            for _ in range(augment_factor - 1):
                start = random.randint(0, max(0, len(encoded) // 5))
                end_frac = random.uniform(0.5, 1.0)
                end = max(start + 3, int(len(encoded) * end_frac))
                partial = [family_token_idx] + encoded[start:end] + [EOS]
                if len(partial) <= max_seq_len:
                    self.sequences.append(partial)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def collate_fn(batch):
    """Pad sequences to same length, return (seqs, padding_mask)."""
    padded = pad_sequence(batch, batch_first=True, padding_value=PAD)
    padding_mask = (padded == PAD)
    return padded, padding_mask


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir, families=('igbt', 'ic'), batch_size=64,
              val_ratio=0.1, test_ratio=0.1, augment_factor=3, seed=42):
    """Load data and return train/val/test loaders + vocab."""
    rng = random.Random(seed)
    vocab = build_vocab(data_dir, families)

    family_map = {
        'igbt': ('IGBT/IGBT_variants.csv', IGBT_TOK),
        'ic': ('IC/IC_variants.csv', IC_TOK),
    }

    all_train, all_val, all_test = [], [], []

    for fam in families:
        csv_name, fam_tok = family_map[fam]
        path = Path(data_dir) / csv_name
        seqs = read_sequences(path)
        seq_list = list(seqs.values())
        rng.shuffle(seq_list)

        n_test = max(1, int(len(seq_list) * test_ratio))
        n_val = max(1, int(len(seq_list) * val_ratio))

        test_seqs = seq_list[:n_test]
        val_seqs = seq_list[n_test:n_test + n_val]
        train_seqs = seq_list[n_test + n_val:]

        all_train.append(LCMSequenceDataset(train_seqs, fam_tok, vocab, augment_factor))
        all_val.append(LCMSequenceDataset(val_seqs, fam_tok, vocab, augment_factor=1))
        all_test.append(LCMSequenceDataset(test_seqs, fam_tok, vocab, augment_factor=1))

    train_ds = torch.utils.data.ConcatDataset(all_train)
    val_ds = torch.utils.data.ConcatDataset(all_val)
    test_ds = torch.utils.data.ConcatDataset(all_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, vocab


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, data_loader, device):
    """Compute average loss on a data loader."""
    model.eval()
    total_loss = 0
    total_mse = 0
    total_ce = 0
    n_batches = 0

    with torch.no_grad():
        for seqs, padding_mask in data_loader:
            seqs = seqs.to(device)
            padding_mask = padding_mask.to(device)
            loss, mse, ce = model.compute_loss(seqs, padding_mask)
            total_loss += loss.item()
            total_mse += mse
            total_ce += ce
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_mse = total_mse / max(n_batches, 1)
    avg_ce = total_ce / max(n_batches, 1)
    return avg_loss, avg_mse, avg_ce


def compute_accuracy(model, data_loader, device, vocab):
    """Compute top-1/3/5 next-step accuracy."""
    model.eval()
    top1 = top3 = top5 = total = 0

    with torch.no_grad():
        for seqs, padding_mask in data_loader:
            seqs = seqs.to(device)
            _, logits = model.forward(seqs, padding_mask.to(device))

            # Predict position t → target is position t+1
            pred = logits[:, :-1]  # (B, S-1, V)
            targets = seqs[:, 1:]  # (B, S-1)

            # Mask padded positions
            mask = (targets != PAD) & (targets >= 6)  # only count real steps
            if mask.sum() == 0:
                continue

            pred_flat = pred[mask]  # (N, V)
            tgt_flat = targets[mask]  # (N,)

            top5_indices = pred_flat.topk(5, dim=1).indices
            top1 += (top5_indices[:, 0] == tgt_flat).sum().item()
            top3 += (top5_indices[:, :3] == tgt_flat.unsqueeze(1)).any(1).sum().item()
            top5 += (top5_indices == tgt_flat.unsqueeze(1)).any(1).sum().item()
            total += tgt_flat.size(0)

    if total == 0:
        return 0, 0, 0
    return top1 / total, top3 / total, top5 / total


def compute_completion_metrics(model, data_loader, device, vocab, n_samples=100):
    """Evaluate sequence completion on a subset."""
    model.eval()
    total_edit = 0
    total_valid = 0
    total_seqs = 0

    with torch.no_grad():
        for seqs, padding_mask in data_loader:
            seqs = seqs.to(device)
            B = seqs.size(0)

            for b in range(B):
                if total_seqs >= n_samples:
                    break

                # Get non-padded sequence
                seq = seqs[b]
                length = (seq != PAD).sum().item()
                if length < 10:
                    continue

                # Cut at 60% for completion test
                cut = int(length * 0.6)
                prefix = seq[:cut].unsqueeze(0)
                suffix_ids = seq[cut:length].tolist()
                suffix_ids = [t for t in suffix_ids if t >= 6]  # real steps only

                if not suffix_ids:
                    continue

                # Complete
                predicted_ids = model.complete_sequence(prefix, max_len=200)

                # Edit distance
                ed = _edit_distance(predicted_ids, suffix_ids)
                norm_ed = ed / max(len(predicted_ids), len(suffix_ids), 1)
                total_edit += norm_ed

                # Validation
                prefix_steps = [vocab.itos[t] for t in seq[1:cut].tolist() if t >= 6]
                pred_steps = [vocab.itos[t] for t in predicted_ids if t < len(vocab)]
                full_seq = prefix_steps + pred_steps
                if full_seq:
                    violations = validate_sequence(full_seq)
                    if not violations:
                        total_valid += 1

                total_seqs += 1

            if total_seqs >= n_samples:
                break

    if total_seqs == 0:
        return 0, 0
    return total_edit / total_seqs, total_valid / total_seqs


def _edit_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def parse_arguments():
    p = argparse.ArgumentParser(description='Train Process-LCM')
    p.add_argument('-epochs', type=int, default=50)
    p.add_argument('-batch_size', type=int, default=64)
    p.add_argument('-lr', type=float, default=3e-4)
    p.add_argument('-grad_clip', type=float, default=1.0)
    p.add_argument('-embed_dim', type=int, default=256)
    p.add_argument('-n_heads', type=int, default=8)
    p.add_argument('-n_layers', type=int, default=6)
    p.add_argument('-dim_feedforward', type=int, default=1024)
    p.add_argument('-dropout', type=float, default=0.1)
    p.add_argument('-mse_weight', type=float, default=1.0)
    p.add_argument('-ce_weight', type=float, default=0.5)
    p.add_argument('-patience', type=int, default=10)
    p.add_argument('-data_dir', type=str, default='../../data/raw/infineon')
    p.add_argument('-families', type=str, nargs='+', default=['igbt', 'ic'])
    p.add_argument('-augment_factor', type=int, default=3)
    p.add_argument('-warmup_steps', type=int, default=500)
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_lr(step, warmup_steps, d_model, base_lr):
    """Transformer learning rate schedule with warmup."""
    if step == 0:
        step = 1
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr * (warmup_steps / step) ** 0.5


def main():
    args = parse_arguments()
    device = get_device()
    print(f"[!] device: {device}")

    print("[!] loading data...")
    train_loader, val_loader, test_loader, vocab = load_data(
        data_dir=args.data_dir,
        families=args.families,
        batch_size=args.batch_size,
        augment_factor=args.augment_factor,
    )
    vocab_size = len(vocab)
    print(f"[VOCAB] {vocab_size} tokens")
    print(f"[TRAIN] {len(train_loader)} batches  "
          f"[VAL] {len(val_loader)} batches  "
          f"[TEST] {len(test_loader)} batches")

    print("[!] building Process-LCM...")
    model = ProcessLCM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[!] {n_params:,} trainable parameters")
    print(f"[!] embed_dim={args.embed_dim} n_heads={args.n_heads} "
          f"n_layers={args.n_layers} ff={args.dim_feedforward}")
    print(f"[!] mse_weight={args.mse_weight} ce_weight={args.ce_weight}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.98), weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6)

    best_val_loss = None
    no_improve = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_mse = 0
        total_ce = 0
        n_batches = 0

        for batch_idx, (seqs, padding_mask) in enumerate(train_loader):
            seqs = seqs.to(device)
            padding_mask = padding_mask.to(device)

            # Warmup LR
            global_step += 1
            if global_step <= args.warmup_steps:
                lr = args.lr * global_step / args.warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

            optimizer.zero_grad()
            loss, mse, ce = model.compute_loss(
                seqs, padding_mask,
                mse_weight=args.mse_weight,
                ce_weight=args.ce_weight,
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if global_step > args.warmup_steps:
                scheduler.step()

            total_loss += loss.item()
            total_mse += mse
            total_ce += ce
            n_batches += 1

            if batch_idx % 50 == 0 and batch_idx > 0:
                avg = total_loss / n_batches
                print(f"  [batch {batch_idx}] loss:{avg:.4f} "
                      f"mse:{total_mse/n_batches:.4f} ce:{total_ce/n_batches:.4f}")

        train_loss = total_loss / max(n_batches, 1)
        val_loss, val_mse, val_ce = evaluate(model, val_loader, device)
        top1, top3, top5 = compute_accuracy(model, val_loader, device, vocab)

        # Completion eval every 5 epochs
        comp_str = ""
        if epoch % 5 == 0 or epoch == 1:
            norm_ed, valid_rate = compute_completion_metrics(
                model, val_loader, device, vocab, n_samples=50)
            comp_str = f" edit:{norm_ed:.3f} valid:{valid_rate:.3f}"

        print(f"[Epoch {epoch}] train:{train_loss:.4f} val:{val_loss:.4f} "
              f"mse:{val_mse:.4f} ce:{val_ce:.4f} "
              f"top1:{top1:.3f} top3:{top3:.3f} top5:{top5:.3f}{comp_str}")

        # Save best model
        if best_val_loss is None or val_loss < best_val_loss:
            print("  [!] saving best model...")
            os.makedirs(".save", exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'vocab_itos': vocab.itos,
                'vocab_stoi': vocab.stoi,
                'args': vars(args),
                'epoch': epoch,
                'val_loss': val_loss,
            }, '.save/best_lcm.pt')
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  [!] early stopping after {args.patience} epochs without improvement")
                break

    # Final test evaluation
    print("\n[!] Final test evaluation...")
    test_loss, test_mse, test_ce = evaluate(model, test_loader, device)
    top1, top3, top5 = compute_accuracy(model, test_loader, device, vocab)
    norm_ed, valid_rate = compute_completion_metrics(
        model, test_loader, device, vocab, n_samples=200)
    print(f"[TEST] loss:{test_loss:.4f} mse:{test_mse:.4f} ce:{test_ce:.4f}")
    print(f"[TEST] top1:{top1:.3f} top3:{top3:.3f} top5:{top5:.3f}")
    print(f"[TEST] norm_edit_dist:{norm_ed:.3f} valid_rate:{valid_rate:.3f}")


if __name__ == '__main__':
    main()
