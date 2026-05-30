"""
Inference script for process sequence prediction.

Input:  CSV in long format (SEQUENCE_ID, STEP) — same as training data.
Output: Predictions printed to stdout or saved to CSV.

Usage:
    # Next-step prediction (top-5)
    python infer.py --mode next_step --input sequences.csv --checkpoint .save/best.pt

    # Full completion until EOS
    python infer.py --mode completion --input sequences.csv --checkpoint .save/best.pt

    # Save output to CSV
    python infer.py --mode completion --input sequences.csv --output predictions.csv

    # Specify family (auto-detects if not set)
    python infer.py --mode next_step --input sequences.csv --family igbt
"""

import argparse
import csv
import sys
from collections import defaultdict

import torch

from model import Encoder, Decoder, Seq2Seq
from utils import PAD, SOS, EOS, IGBT_TOK, IC_TOK, Vocab
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


def read_input_csv(path):
    """Read long-format CSV (SEQUENCE_ID, STEP). Returns ordered dict."""
    sequences = defaultdict(list)
    order = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                seq_id, step = row[0].strip(), row[1].strip()
                if seq_id not in sequences:
                    order.append(seq_id)
                sequences[seq_id].append(step)
    return [(sid, sequences[sid]) for sid in order]


def detect_family(steps):
    """Auto-detect family from step names."""
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


def predict_next_step(model, src_tensor, vocab, device, top_k=5):
    """Return list of (step_name, probability) for next step."""
    src = src_tensor.unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_output, encoder_hidden = model.encoder(src)
        hidden = model.decoder.init_hidden(encoder_hidden)
        sos_input = torch.tensor([SOS], device=device)
        output, _, _ = model.decoder(sos_input, hidden, encoder_output)
        probs = output.exp().squeeze(0)

    top_probs, top_indices = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token = vocab.itos[idx]
        if idx >= 6:  # skip special tokens
            results.append((token, prob))
    return results


def predict_completion(model, src_tensor, vocab, device, max_len=200):
    """Greedy decode until EOS. Return list of step strings."""
    src = src_tensor.unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_output, encoder_hidden = model.encoder(src)
        hidden = model.decoder.init_hidden(encoder_hidden)
        input_tok = torch.tensor([SOS], device=device)
        predicted = []
        for _ in range(max_len):
            output, hidden, _ = model.decoder(input_tok, hidden, encoder_output)
            top1 = output.argmax(1)
            tok_idx = top1.item()
            if tok_idx == EOS:
                break
            if tok_idx >= 6:  # skip special tokens
                predicted.append(vocab.itos[tok_idx])
            input_tok = top1
    return predicted


def main():
    p = argparse.ArgumentParser(description='Inference for process sequences')
    p.add_argument('--mode', choices=['next_step', 'completion'], required=True,
                   help='1=next_step (top-k), 2=completion (until EOS)')
    p.add_argument('--input', required=True, help='Input CSV (SEQUENCE_ID, STEP)')
    p.add_argument('--output', default=None, help='Output CSV path (optional)')
    p.add_argument('--checkpoint', default='.save/best.pt', help='Model checkpoint')
    p.add_argument('--family', default=None, help='Force family (igbt/ic/mosfet)')
    p.add_argument('--top_k', type=int, default=5, help='Top-K for next_step mode')
    p.add_argument('--max_len', type=int, default=200, help='Max decode length')
    p.add_argument('--validate', action='store_true',
                   help='Run process-logic validation on completions')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab = load_model(args.checkpoint, device)
    sequences = read_input_csv(args.input)
    print(f"[!] Loaded {len(sequences)} sequences, vocab={len(vocab)}, device={device}",
          file=sys.stderr)

    output_rows = []

    for seq_id, steps in sequences:
        family = args.family or detect_family(steps)
        src = encode_prefix(steps, family, vocab)

        if args.mode == 'next_step':
            preds = predict_next_step(model, src, vocab, device, args.top_k)
            print(f"\n{'='*60}")
            print(f"SEQ: {seq_id} | family={family} | prefix_len={len(steps)}")
            print(f"Last 3 steps: {' -> '.join(steps[-3:])}")
            print(f"{'─'*60}")
            for rank, (step, prob) in enumerate(preds, 1):
                print(f"  {rank}. {step:45s} p={prob:.4f}")
                output_rows.append({
                    'SEQUENCE_ID': seq_id, 'RANK': rank,
                    'PREDICTED_STEP': step, 'PROBABILITY': f'{prob:.6f}'
                })

        else:  # completion
            completion = predict_completion(model, src, vocab, device, args.max_len)
            print(f"\n{'='*60}")
            print(f"SEQ: {seq_id} | family={family} | prefix_len={len(steps)}")
            print(f"Last 3 prefix steps: {' -> '.join(steps[-3:])}")
            print(f"Completion ({len(completion)} steps):")
            for i, s in enumerate(completion):
                print(f"  {i+1:3d}. {s}")

            if args.validate:
                full = steps + completion
                violations = validate_sequence(full)
                if violations:
                    print(f"  [INVALID] {len(violations)} rule(s) broken: "
                          f"{', '.join(violations)}")
                else:
                    print(f"  [VALID] Passes all 10 process-logic rules")

            for step in completion:
                output_rows.append({'SEQUENCE_ID': seq_id, 'STEP': step})

    # Write output CSV
    if args.output and output_rows:
        if args.mode == 'next_step':
            fields = ['SEQUENCE_ID', 'RANK', 'PREDICTED_STEP', 'PROBABILITY']
        else:
            fields = ['SEQUENCE_ID', 'STEP']
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"\n[!] Saved {len(output_rows)} rows to {args.output}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
