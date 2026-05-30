"""
Data pipeline for semiconductor process sequence prediction.

Reads IGBT/IC variant CSVs (long-format: SEQUENCE_ID, STEP), builds a shared
vocabulary, and produces prefix→suffix training pairs for the Seq2Seq model.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# Special token indices
PAD, UNK, SOS, EOS = 0, 1, 2, 3
FAMILY_TOKENS = {'igbt': '<igbt>', 'ic': '<ic>'}
SPECIALS = ['<pad>', '<unk>', '<sos>', '<eos>', '<igbt>', '<ic>']
# Named indices for family tokens
IGBT_TOK, IC_TOK = 4, 5


class Vocab:
    """Shared vocabulary for process steps + special tokens."""

    def __init__(self, step_tokens):
        self.itos = list(SPECIALS) + sorted(set(step_tokens) - set(SPECIALS))
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK) for t in tokens]

    def decode(self, indices):
        return [self.itos[i] for i in indices if i not in (PAD, SOS, EOS)]


def read_sequences(csv_path):
    """Read a long-format CSV and return dict[seq_id] -> list[step_str]."""
    sequences = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                seq_id, step = row[0], row[1]
                sequences[seq_id].append(step)
    return dict(sequences)


def build_vocab(data_dir, families=('igbt', 'ic')):
    """Build vocabulary from all step tokens in the specified families."""
    all_steps = set()
    family_map = {
        'igbt': 'IGBT/IGBT_variants.csv',
        'ic': 'IC/IC_variants.csv',
        'mosfet': 'MOSFET/MOSFET_variants.csv',
    }
    for fam in families:
        path = Path(data_dir) / family_map[fam]
        seqs = read_sequences(path)
        for steps in seqs.values():
            all_steps.update(steps)
    return Vocab(list(all_steps))


class ProcessSequenceDataset(Dataset):
    """
    Dataset that produces (src, trg) pairs from process sequences.

    src = [family_token] + prefix_steps
    trg = [SOS] + suffix_steps + [EOS]

    Cut point strategy: 40% at 60%, 40% at 80%, 20% uniform [40%-90%].
    """

    def __init__(self, sequences, family_token_idx, vocab, augment_factor=3):
        self.vocab = vocab
        self.family_token_idx = family_token_idx
        self.pairs = []
        for steps in sequences:
            encoded = vocab.encode(steps)
            seq_len = len(encoded)
            if seq_len < 5:
                continue
            for _ in range(augment_factor):
                cut = self._sample_cut(seq_len)
                src = [family_token_idx] + encoded[:cut]
                trg = [SOS] + encoded[cut:] + [EOS]
                self.pairs.append((src, trg))

    def _sample_cut(self, seq_len):
        r = random.random()
        if r < 0.4:
            # 60% cut
            cut = int(seq_len * 0.6)
        elif r < 0.8:
            # 80% cut
            cut = int(seq_len * 0.8)
        else:
            # uniform [40%-90%]
            frac = random.uniform(0.4, 0.9)
            cut = int(seq_len * frac)
        return max(2, min(cut, seq_len - 1))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, trg = self.pairs[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(trg, dtype=torch.long)


def collate_fn(batch):
    """Pad and stack into (seq_len, batch) time-first tensors."""
    srcs, trgs = zip(*batch)
    src_padded = pad_sequence(srcs, padding_value=PAD)
    trg_padded = pad_sequence(trgs, padding_value=PAD)
    return src_padded, trg_padded


def load_data(data_dir, families=('igbt', 'ic'), batch_size=32,
              max_sequences=None, val_ratio=0.1, augment_factor=3, seed=42):
    """
    Load process sequence data and return train/val DataLoaders + vocab.

    Args:
        data_dir: Path to directory containing family subdirs (IGBT/, IC/)
        families: Which families to include in training
        batch_size: Batch size for DataLoaders
        max_sequences: Limit sequences per family (None = use all)
        val_ratio: Fraction of sequences for validation
        augment_factor: Number of random cuts per sequence
        seed: Random seed for reproducibility

    Returns:
        (train_loader, val_loader, vocab)
    """
    rng = random.Random(seed)
    vocab = build_vocab(data_dir, families)

    family_map = {
        'igbt': ('IGBT/IGBT_variants.csv', IGBT_TOK),
        'ic': ('IC/IC_variants.csv', IC_TOK),
    }

    all_train_seqs = []
    all_val_seqs = []

    for fam in families:
        csv_name, fam_tok = family_map[fam]
        path = Path(data_dir) / csv_name
        seqs = read_sequences(path)
        seq_list = list(seqs.values())
        rng.shuffle(seq_list)

        if max_sequences:
            seq_list = seq_list[:max_sequences]

        n_val = max(1, int(len(seq_list) * val_ratio))
        val_seqs = seq_list[:n_val]
        train_seqs = seq_list[n_val:]

        all_train_seqs.append((train_seqs, fam_tok))
        all_val_seqs.append((val_seqs, fam_tok))

    # Build datasets
    train_ds_parts = []
    val_ds_parts = []
    for seqs, fam_tok in all_train_seqs:
        train_ds_parts.append(
            ProcessSequenceDataset(seqs, fam_tok, vocab, augment_factor))
    for seqs, fam_tok in all_val_seqs:
        val_ds_parts.append(
            ProcessSequenceDataset(seqs, fam_tok, vocab, augment_factor=1))

    train_ds = torch.utils.data.ConcatDataset(train_ds_parts)
    val_ds = torch.utils.data.ConcatDataset(val_ds_parts)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, vocab
