"""
Evaluate a checkpoint on the test split at specific prefix cutoffs (60%, 80%).
Reports next-step accuracy and completion metrics (edit distance, rule violations).
"""
import sys
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from model import Encoder, Decoder, Seq2Seq
from utils import (PAD, SOS, EOS, IGBT_TOK, IC_TOK, Vocab,
                   read_sequences, build_vocab, collate_fn)
from validator import validate_sequence


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt['args']
    vocab = Vocab.__new__(Vocab)
    vocab.itos = ckpt['vocab_itos']
    vocab.stoi = ckpt['vocab_stoi']

    vocab_size = len(vocab)
    encoder = Encoder(vocab_size, args['embed_size'], args['hidden_size'],
                      n_layers=2, dropout=0.0)
    decoder = Decoder(args['embed_size'], args['hidden_size'], vocab_size,
                      n_layers=args.get('dec_layers', 1), dropout=0.0)
    model = Seq2Seq(encoder, decoder, tie_embeddings=True).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, vocab


def get_test_sequences(data_dir, families=('igbt', 'ic'), seed=42, test_ratio=0.1):
    """Reproduce the same test split as training."""
    rng = random.Random(seed)
    family_map = {
        'igbt': ('IGBT/IGBT_variants.csv', IGBT_TOK),
        'ic': ('IC/IC_variants.csv', IC_TOK),
    }
    test_data = []  # list of (steps, fam_tok)
    for fam in families:
        csv_name, fam_tok = family_map[fam]
        path = Path(data_dir) / csv_name
        seqs = read_sequences(path)
        seq_list = list(seqs.values())
        rng.shuffle(seq_list)
        n_test = max(1, int(len(seq_list) * test_ratio))
        test_seqs = seq_list[:n_test]
        for steps in test_seqs:
            test_data.append((steps, fam_tok))
    return test_data


def edit_distance(seq1, seq2):
    """Levenshtein edit distance between two lists."""
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


def evaluate_at_cutoff(model, vocab, test_data, cutoff_frac, device, max_len=200):
    """
    Evaluate model at a specific prefix cutoff fraction.
    Returns dict with all metrics.
    """
    # Next-step metrics
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    next_step_total = 0

    # Completion metrics
    total_edit_dist = 0.0
    total_norm_edit = 0.0
    total_token_acc = 0.0
    exact_matches = 0
    eos_terminations = 0
    valid_completions = 0
    invalid_completions_details = []
    completion_total = 0

    for steps, fam_tok in test_data:
        seq_len = len(steps)
        if seq_len < 5:
            continue

        cut = max(2, int(seq_len * cutoff_frac))
        cut = min(cut, seq_len - 1)

        prefix = steps[:cut]
        suffix = steps[cut:]  # ground truth remainder

        # Encode prefix
        encoded_prefix = [fam_tok] + vocab.encode(prefix)
        src = torch.tensor(encoded_prefix, dtype=torch.long).unsqueeze(1).to(device)

        with torch.no_grad():
            encoder_output, encoder_hidden = model.encoder(src)
            hidden = model.decoder.init_hidden(encoder_hidden)

            # --- Next-step prediction ---
            sos_input = torch.tensor([SOS], device=device)
            output, _, _ = model.decoder(sos_input, hidden, encoder_output)
            probs = output.exp().squeeze(0)
            top5_indices = probs.topk(5).indices.tolist()

            # Ground truth next step
            gt_next = vocab.stoi.get(suffix[0], -1)
            if gt_next >= 0:
                next_step_total += 1
                if gt_next == top5_indices[0]:
                    top1_correct += 1
                if gt_next in top5_indices[:3]:
                    top3_correct += 1
                if gt_next in top5_indices:
                    top5_correct += 1

            # --- Completion (greedy) ---
            # Re-run encoder for fresh hidden state
            hidden = model.decoder.init_hidden(encoder_hidden)
            input_tok = torch.tensor([SOS], device=device)
            predicted_steps = []
            hit_eos = False
            for _ in range(max_len):
                output, hidden, _ = model.decoder(input_tok, hidden, encoder_output)
                top1 = output.argmax(1)
                tok_idx = top1.item()
                if tok_idx == EOS:
                    hit_eos = True
                    break
                if tok_idx >= 6:
                    predicted_steps.append(vocab.itos[tok_idx])
                input_tok = top1

        completion_total += 1
        if hit_eos:
            eos_terminations += 1

        # Edit distance
        ed = edit_distance(predicted_steps, suffix)
        total_edit_dist += ed
        norm_ed = ed / max(len(predicted_steps), len(suffix), 1)
        total_norm_edit += norm_ed

        # Token accuracy
        compare_len = min(len(predicted_steps), len(suffix))
        if compare_len > 0:
            matches = sum(1 for i in range(compare_len)
                         if predicted_steps[i] == suffix[i])
            total_token_acc += matches / len(suffix)
        else:
            total_token_acc += 0.0

        # Exact match
        if predicted_steps == suffix:
            exact_matches += 1

        # Rule validation on full sequence
        full_seq = prefix + predicted_steps
        violations = validate_sequence(full_seq)
        if not violations:
            valid_completions += 1
        else:
            if len(invalid_completions_details) < 5:  # Keep first 5 examples
                invalid_completions_details.append({
                    'prefix_last3': prefix[-3:],
                    'pred_first3': predicted_steps[:3],
                    'violations': violations,
                })

    return {
        'cutoff': cutoff_frac,
        'n_sequences': completion_total,
        'next_step': {
            'total': next_step_total,
            'top1_acc': top1_correct / max(next_step_total, 1),
            'top3_acc': top3_correct / max(next_step_total, 1),
            'top5_acc': top5_correct / max(next_step_total, 1),
        },
        'completion': {
            'avg_edit_dist': total_edit_dist / max(completion_total, 1),
            'avg_norm_edit': total_norm_edit / max(completion_total, 1),
            'avg_token_acc': total_token_acc / max(completion_total, 1),
            'exact_match_rate': exact_matches / max(completion_total, 1),
            'eos_rate': eos_terminations / max(completion_total, 1),
            'valid_rate': valid_completions / max(completion_total, 1),
            'invalid_rate': 1 - valid_completions / max(completion_total, 1),
        },
        'violation_examples': invalid_completions_details,
    }


def print_results(results):
    c = results['cutoff']
    n = results['n_sequences']
    ns = results['next_step']
    comp = results['completion']

    print(f"\n{'='*70}")
    print(f"  CUTOFF: {int(c*100)}% prefix ({n} test sequences)")
    print(f"{'='*70}")

    print(f"\n  ── Next-Step Prediction ({ns['total']} samples) ──")
    print(f"     Top-1 Accuracy: {ns['top1_acc']:.4f} ({ns['top1_acc']*100:.1f}%)")
    print(f"     Top-3 Accuracy: {ns['top3_acc']:.4f} ({ns['top3_acc']*100:.1f}%)")
    print(f"     Top-5 Accuracy: {ns['top5_acc']:.4f} ({ns['top5_acc']*100:.1f}%)")

    print(f"\n  ── Completion (greedy decode) ──")
    print(f"     Token Accuracy:      {comp['avg_token_acc']:.4f} ({comp['avg_token_acc']*100:.1f}%)")
    print(f"     Exact Match Rate:    {comp['exact_match_rate']:.4f} ({comp['exact_match_rate']*100:.1f}%)")
    print(f"     EOS Termination:     {comp['eos_rate']:.4f} ({comp['eos_rate']*100:.1f}%)")
    print(f"     Avg Edit Distance:   {comp['avg_edit_dist']:.2f}")
    print(f"     Avg Norm Edit Dist:  {comp['avg_norm_edit']:.4f}")
    print(f"     Rule Valid Rate:     {comp['valid_rate']:.4f} ({comp['valid_rate']*100:.1f}%)")
    print(f"     Rule Violation Rate: {comp['invalid_rate']:.4f} ({comp['invalid_rate']*100:.1f}%)")

    if results['violation_examples']:
        print(f"\n  ── Sample Violations (first {len(results['violation_examples'])}) ──")
        for i, ex in enumerate(results['violation_examples'], 1):
            print(f"     [{i}] Prefix ends: ...{' -> '.join(ex['prefix_last3'])}")
            print(f"         Pred starts: {' -> '.join(ex['pred_first3'])}")
            for v in ex['violations'][:3]:
                print(f"         ⚠️  {v}")
            print()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='.save/best_100000_unfinished.pt')
    p.add_argument('--data_dir', default='../../data/raw/infineon')
    p.add_argument('--cutoffs', nargs='+', type=float, default=[0.6, 0.8])
    p.add_argument('--max_test', type=int, default=500,
                   help='Max test sequences to evaluate (for speed on CPU)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[!] Device: {device}")
    print(f"[!] Loading checkpoint: {args.checkpoint}")
    model, vocab = load_model(args.checkpoint, device)
    print(f"[!] Vocab size: {len(vocab)}")

    print(f"[!] Loading test data from: {args.data_dir}")
    test_data = get_test_sequences(args.data_dir)
    print(f"[!] Total test sequences: {len(test_data)}")

    if args.max_test and len(test_data) > args.max_test:
        rng = random.Random(123)
        test_data = rng.sample(test_data, args.max_test)
        print(f"[!] Sampled {args.max_test} sequences for evaluation")

    for cutoff in args.cutoffs:
        print(f"\n[!] Evaluating at {int(cutoff*100)}% cutoff...")
        results = evaluate_at_cutoff(model, vocab, test_data, cutoff, device)
        print_results(results)


if __name__ == '__main__':
    main()
