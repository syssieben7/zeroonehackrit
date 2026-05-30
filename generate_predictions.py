"""
Generate predictions for the official eval using a trained model checkpoint.

Reads eval_input_valid.csv and produces:
  - predictions_nextstep.csv  (EXAMPLE_ID, RANK_1..RANK_5)
  - predictions_completion.csv (EXAMPLE_ID, PREDICTED_SEQUENCE)

Reads eval_input_anomaly.csv and produces:
  - predictions_anomaly.csv (EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE)

Usage:
    python generate_predictions.py --checkpoint .save/best_lcm_10000.pt
"""

import sys
import os
import csv
import argparse

# Add model code to path BEFORE importing model modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hpc', 'scp'))
sys.path.insert(0, os.path.dirname(__file__))  # for HPC where files are co-located

import torch
import torch.nn.functional as F
from lcm_model import ProcessLCM
from model import Encoder, Decoder, Seq2Seq
from utils import PAD, SOS, EOS, IGBT_TOK, IC_TOK, Vocab


def load_checkpoint(path, device):
    """Load model (auto-detect LCM vs Seq2Seq)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt['args']

    vocab = Vocab.__new__(Vocab)
    vocab.itos = ckpt['vocab_itos']
    vocab.stoi = ckpt['vocab_stoi']
    vocab_size = len(vocab)

    if 'embed_dim' in args:
        model = ProcessLCM(
            vocab_size=vocab_size,
            embed_dim=args['embed_dim'],
            n_heads=args['n_heads'],
            n_layers=args['n_layers'],
            dim_feedforward=args['dim_feedforward'],
            dropout=0.0,
        ).to(device)
    else:
        encoder = Encoder(vocab_size, args['embed_size'], args['hidden_size'],
                          n_layers=2, dropout=0.0)
        decoder = Decoder(args['embed_size'], args['hidden_size'], vocab_size,
                          n_layers=args.get('dec_layers', 1), dropout=0.0)
        model = Seq2Seq(encoder, decoder, tie_embeddings=True).to(device)

    state = ckpt['model_state']
    # PE is a deterministic sinusoidal buffer - skip if size mismatches
    state = {k: v for k, v in state.items() if 'pos_encoding.pe' not in k}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, vocab


def detect_family(steps):
    joined = ' '.join(steps).upper()
    if 'P BODY' in joined or 'N BUFFER' in joined or 'CHANNEL STOP' in joined:
        return 'igbt'
    if 'EPITAXIAL WAFER CHECK' in joined or 'FIELD OXIDE' in joined:
        return 'igbt'
    if 'SUBSTRATE CHECK' in joined or 'EPITAXIAL DEPOSITION' in joined:
        return 'mosfet'
    return 'ic'


def encode_prefix(steps, family, vocab):
    fam_map = {'igbt': IGBT_TOK, 'ic': IC_TOK, 'mosfet': IGBT_TOK}
    fam_tok = fam_map.get(family, IC_TOK)
    encoded = [fam_tok] + vocab.encode(steps)
    return torch.tensor(encoded, dtype=torch.long)


def predict_next_step_lcm(model, src, vocab, device, top_k=5):
    src_batch = src.unsqueeze(0).to(device)
    top_indices, top_scores = model.predict_next_step(src_batch, top_k=top_k)
    results = []
    for idx in top_indices[0].tolist():
        if idx >= 6:
            results.append(vocab.itos[idx])
    return results[:top_k]


def predict_next_step_seq2seq(model, src, vocab, device, top_k=5):
    src_t = src.unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_output, encoder_hidden = model.encoder(src_t)
        hidden = model.decoder.init_hidden(encoder_hidden)
        sos_input = torch.tensor([SOS], device=device)
        output, _, _ = model.decoder(sos_input, hidden, encoder_output)
        probs = output.exp().squeeze(0)
        _, top_indices = probs.topk(top_k + 6)  # extra to skip specials

    results = []
    for idx in top_indices.tolist():
        if idx >= 6:
            results.append(vocab.itos[idx])
        if len(results) >= top_k:
            break
    return results


def complete_lcm(model, src, vocab, device, max_len=80):
    src_batch = src.unsqueeze(0).to(device)
    predicted_ids = model.complete_sequence(src_batch, max_len=max_len, eos_idx=EOS)
    return [vocab.itos[idx] for idx in predicted_ids if idx < len(vocab.itos) and idx >= 6]


def complete_seq2seq(model, src, vocab, device, max_len=80):
    src_t = src.unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_output, encoder_hidden = model.encoder(src_t)
        hidden = model.decoder.init_hidden(encoder_hidden)
        input_tok = torch.tensor([SOS], device=device)
        completion = []
        for _ in range(max_len):
            output, hidden, _ = model.decoder(input_tok, hidden, encoder_output)
            top1 = output.argmax(1)
            tok_idx = top1.item()
            if tok_idx == EOS:
                break
            if tok_idx >= 6:
                completion.append(vocab.itos[tok_idx])
            input_tok = top1
    return completion


def anomaly_score_lcm(model, step_ids, device):
    """Compute anomaly score using LCM prediction error."""
    src_batch = step_ids.unsqueeze(0).to(device)
    errors, max_error, mean_error = model.anomaly_scores(src_batch)
    return max_error.item(), mean_error.item(), errors.squeeze(0)


def anomaly_score_seq2seq(model, step_ids, vocab, device):
    """Compute anomaly score using Seq2Seq prediction confidence."""
    src_t = step_ids.unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_output, encoder_hidden = model.encoder(src_t)
        hidden = model.decoder.init_hidden(encoder_hidden)
        input_tok = torch.tensor([SOS], device=device)

        total_surprise = 0.0
        max_surprise = 0.0
        n_steps = 0

        for t in range(1, len(step_ids)):
            output, hidden, _ = model.decoder(input_tok, hidden, encoder_output)
            probs = output.exp().squeeze(0)
            actual_idx = step_ids[t].item()
            if actual_idx >= 6:
                p = probs[actual_idx].item()
                surprise = -torch.log(torch.tensor(max(p, 1e-10))).item()
                total_surprise += surprise
                max_surprise = max(max_surprise, surprise)
                n_steps += 1
            input_tok = torch.tensor([actual_idx], device=device)

    mean_surprise = total_surprise / max(n_steps, 1)
    return max_surprise, mean_surprise


def read_eval_valid(path):
    """Read eval_input_valid.csv."""
    examples = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append({
                'id': row['EXAMPLE_ID'].strip(),
                'family': row['FAMILY'].strip(),
                'fraction': row['COMPLETION_FRACTION'].strip(),
                'steps': [s.strip() for s in row['PARTIAL_SEQUENCE'].split('|') if s.strip()],
            })
    return examples


def read_eval_anomaly(path):
    """Read eval_input_anomaly.csv."""
    examples = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append({
                'id': row['EXAMPLE_ID'].strip(),
                'family': row['FAMILY'].strip(),
                'steps': [s.strip() for s in row['SEQUENCE'].split('|') if s.strip()],
            })
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='hpc/scp/.save/best_lcm_10000.pt')
    p.add_argument('--eval-valid', default='data/participant_files/eval_input_valid.csv')
    p.add_argument('--eval-anomaly', default='data/participant_files/eval_input_anomaly.csv')
    p.add_argument('--output-dir', default='data/participant_files/predictions')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[!] Device: {device}")
    print(f"[!] Loading checkpoint: {args.checkpoint}")
    model, vocab = load_checkpoint(args.checkpoint, device)
    is_lcm = isinstance(model, ProcessLCM)
    print(f"[!] Model type: {'ProcessLCM' if is_lcm else 'Seq2Seq'}")
    print(f"[!] Vocab size: {len(vocab)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Task 1: Next-step prediction ---
    print("\n[!] Generating next-step predictions...")
    valid_examples = read_eval_valid(args.eval_valid)
    print(f"    {len(valid_examples)} examples")

    nextstep_rows = []
    total = len(valid_examples)
    for i, ex in enumerate(valid_examples, 1):
        if i % 50 == 0 or i == total:
            print(f"    [{i}/{total}]", end='\r', flush=True)
        steps = ex['steps']
        family = detect_family(steps)
        src = encode_prefix(steps, family, vocab)

        if is_lcm:
            preds = predict_next_step_lcm(model, src, vocab, device, top_k=5)
        else:
            preds = predict_next_step_seq2seq(model, src, vocab, device, top_k=5)

        # Pad to 5 if needed
        while len(preds) < 5:
            preds.append("")

        nextstep_rows.append({
            'EXAMPLE_ID': ex['id'],
            'RANK_1': preds[0],
            'RANK_2': preds[1],
            'RANK_3': preds[2],
            'RANK_4': preds[3],
            'RANK_5': preds[4],
        })

    ns_path = os.path.join(args.output_dir, 'predictions_nextstep.csv')
    print()  # newline after progress
    with open(ns_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['EXAMPLE_ID', 'RANK_1', 'RANK_2',
                                               'RANK_3', 'RANK_4', 'RANK_5'])
        writer.writeheader()
        writer.writerows(nextstep_rows)
    print(f"    Saved: {ns_path}")

    # --- Task 2: Sequence completion ---
    print("\n[!] Generating completion predictions...")
    completion_rows = []
    total = len(valid_examples)
    for i, ex in enumerate(valid_examples, 1):
        if i % 10 == 0 or i == total:
            print(f"    [{i}/{total}]", end='\r', flush=True)
        steps = ex['steps']
        family = detect_family(steps)
        src = encode_prefix(steps, family, vocab)

        # Estimate remaining steps based on fraction
        frac = float(ex['fraction'])
        prefix_len = len(steps)
        estimated_total = int(prefix_len / frac)
        expected_remaining = estimated_total - prefix_len + 10  # small buffer

        if is_lcm:
            completed = complete_lcm(model, src, vocab, device, max_len=min(expected_remaining, 80))
        else:
            completed = complete_seq2seq(model, src, vocab, device, max_len=min(expected_remaining, 80))

        completion_rows.append({
            'EXAMPLE_ID': ex['id'],
            'PREDICTED_SEQUENCE': '|'.join(completed),
        })

    comp_path = os.path.join(args.output_dir, 'predictions_completion.csv')
    print()  # newline after progress
    with open(comp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['EXAMPLE_ID', 'PREDICTED_SEQUENCE'])
        writer.writeheader()
        writer.writerows(completion_rows)
    print(f"    Saved: {comp_path}")

    # --- Task 3: Anomaly detection ---
    print("\n[!] Generating anomaly predictions...")
    anomaly_examples = read_eval_anomaly(args.eval_anomaly)
    print(f"    {len(anomaly_examples)} examples")

    anomaly_rows = []
    total = len(anomaly_examples)
    for i, ex in enumerate(anomaly_examples, 1):
        if i % 50 == 0 or i == total:
            print(f"    [{i}/{total}]", end='\r', flush=True)
        steps = ex['steps']
        family = detect_family(steps)
        src = encode_prefix(steps, family, vocab)

        if is_lcm:
            max_err, mean_err, per_pos = anomaly_score_lcm(model, src, device)
            # Normalize score to [0,1] where higher = more likely valid
            score = 1.0 / (1.0 + max_err)  # sigmoid-like
            # Store score, decide threshold after collecting all
            is_valid = -1  # placeholder
        else:
            max_surp, mean_surp = anomaly_score_seq2seq(model, src, vocab, device)
            score = 1.0 / (1.0 + max_surp)
            is_valid = 1 if max_surp < 3.0 else 0

        anomaly_rows.append({
            'EXAMPLE_ID': ex['id'],
            'IS_VALID': is_valid,
            'SCORE': f"{score:.6f}",
            'PREDICTED_RULE': '',
        })

    # Adaptive thresholding for LCM: top 50% by score are valid
    if is_lcm:
        scores = sorted([float(r['SCORE']) for r in anomaly_rows], reverse=True)
        # Assume roughly half are valid — use median as threshold
        median_score = scores[len(scores) // 2]
        print(f"\n    Anomaly score stats: min={scores[-1]:.4f} median={median_score:.4f} max={scores[0]:.4f}")
        for r in anomaly_rows:
            r['IS_VALID'] = 1 if float(r['SCORE']) >= median_score else 0
        n_valid = sum(1 for r in anomaly_rows if r['IS_VALID'] == 1)
        print(f"    Predicted valid: {n_valid}, invalid: {len(anomaly_rows) - n_valid}")

    anom_path = os.path.join(args.output_dir, 'predictions_anomaly.csv')
    print()  # newline after progress
    with open(anom_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['EXAMPLE_ID', 'IS_VALID', 'SCORE',
                                               'PREDICTED_RULE'])
        writer.writeheader()
        writer.writerows(anomaly_rows)
    print(f"    Saved: {anom_path}")

    print("\n[!] Done! All predictions saved to:", args.output_dir)


if __name__ == '__main__':
    main()
