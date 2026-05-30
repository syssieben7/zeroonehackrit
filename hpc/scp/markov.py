"""
First-order Markov Chain baseline for process sequence prediction.

Counts bigram transitions from training data and predicts next step
as the most probable successor given the current step.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path


class MarkovChain:
    """First-order Markov chain over process step tokens."""

    def __init__(self):
        # counts[prev_step][next_step] = count
        self.counts = defaultdict(lambda: defaultdict(int))
        # probs[prev_step] = sorted list of (prob, step)
        self.probs = {}
        self.total_transitions = 0

    def fit(self, sequences):
        """Train from a list of sequences (each is a list of step strings)."""
        for seq in sequences:
            for i in range(len(seq) - 1):
                self.counts[seq[i]][seq[i + 1]] += 1
                self.total_transitions += 1
        self._compute_probs()

    def _compute_probs(self):
        self.probs = {}
        for prev, nexts in self.counts.items():
            total = sum(nexts.values())
            ranked = sorted(
                [(count / total, step) for step, count in nexts.items()],
                reverse=True
            )
            self.probs[prev] = ranked

    def predict_next(self, current_step, top_k=10):
        """Return top-k (prob, step) predictions given current step."""
        if current_step not in self.probs:
            return []
        return self.probs[current_step][:top_k]

    def predict_completion(self, prefix, max_steps=200, end_token="SHIP LOT"):
        """Greedily complete a sequence from the last step of prefix."""
        if not prefix:
            return []
        completion = []
        current = prefix[-1]
        for _ in range(max_steps):
            preds = self.predict_next(current, top_k=1)
            if not preds:
                break
            prob, next_step = preds[0]
            if next_step == end_token:
                break
            completion.append(next_step)
            current = next_step
        return completion

    def save(self, path):
        """Save model to JSON."""
        data = {step: dict(nexts) for step, nexts in self.counts.items()}
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        """Load model from JSON."""
        mc = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        for prev, nexts in data.items():
            for nxt, count in nexts.items():
                mc.counts[prev][nxt] = count
                mc.total_transitions += count
        mc._compute_probs()
        return mc


def train_from_csvs(data_dir, families=('igbt', 'ic'), max_sequences=None,
                    test_ratio=0.1, val_ratio=0.1, seed=42):
    """
    Train a Markov chain from the same data splits used by the Seq2Seq model.

    Returns (markov_chain, train_seqs, val_seqs, test_seqs) where seqs are
    flat lists of sequence lists.
    """
    rng = random.Random(seed)
    family_map = {
        'igbt': 'IGBT/IGBT_variants.csv',
        'ic': 'IC/IC_variants.csv',
    }

    all_train = []
    all_val = []
    all_test = []

    for fam in families:
        csv_path = Path(data_dir) / family_map[fam]
        sequences = _read_sequences(csv_path)
        seq_list = list(sequences.values())
        rng.shuffle(seq_list)

        if max_sequences:
            seq_list = seq_list[:max_sequences]

        n_test = max(1, int(len(seq_list) * test_ratio))
        n_val = max(1, int(len(seq_list) * val_ratio))
        test_seqs = seq_list[:n_test]
        val_seqs = seq_list[n_test:n_test + n_val]
        train_seqs = seq_list[n_test + n_val:]

        all_train.extend(train_seqs)
        all_val.extend(val_seqs)
        all_test.extend(test_seqs)

    mc = MarkovChain()
    mc.fit(all_train)
    return mc, all_train, all_val, all_test


def _read_sequences(csv_path):
    """Read long-format CSV → dict[seq_id] → list[step_str]."""
    sequences = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                sequences[row[0]].append(row[1])
    return dict(sequences)


if __name__ == "__main__":
    import sys
    import time

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../../data/raw/infineon"
    print(f"[!] Training Markov chain from: {data_dir}")

    t0 = time.time()
    mc, train, val, test = train_from_csvs(data_dir)
    dt = time.time() - t0
    print(f"[!] Trained in {dt:.1f}s on {len(train)} sequences "
          f"({mc.total_transitions:,} transitions)")
    print(f"[!] Unique states: {len(mc.probs)}")

    # Save
    out_path = Path(__file__).parent / '.save' / 'markov.json'
    out_path.parent.mkdir(exist_ok=True)
    mc.save(out_path)
    print(f"[!] Saved to {out_path}")

    # Quick eval: top-1/3/5 accuracy on test set
    correct = {1: 0, 3: 0, 5: 0}
    total = 0
    for seq in test:
        for i in range(len(seq) - 1):
            preds = mc.predict_next(seq[i], top_k=5)
            pred_steps = [s for _, s in preds]
            actual = seq[i + 1]
            total += 1
            if actual in pred_steps[:1]:
                correct[1] += 1
            if actual in pred_steps[:3]:
                correct[3] += 1
            if actual in pred_steps[:5]:
                correct[5] += 1

    print(f"\n[TEST] {total:,} transitions")
    print(f"  Top-1: {correct[1]/total:.4f}")
    print(f"  Top-3: {correct[3]/total:.4f}")
    print(f"  Top-5: {correct[5]/total:.4f}")
