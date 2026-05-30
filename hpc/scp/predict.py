"""
Inference script for process sequence prediction (Tasks 1 & 2).

Task 1: Next-step prediction — return top-k next steps + probabilities.
Task 2: Sequence completion — greedy decode until EOS or max_len.

Usage:
    # Next-step prediction on eval input
    python predict.py --task next_step --input eval_input_valid.csv --output predictions_task1.csv

    # Sequence completion
    python predict.py --task completion --input eval_input_valid.csv --output predictions_task2.csv

    # Interactive mode
    python predict.py --interactive --checkpoint .save/best.pt
"""

import argparse
import csv
import torch
from pathlib import Path
from collections import defaultdict

from model import Encoder, Decoder, Seq2Seq
from utils import PAD, SOS, EOS, IGBT_TOK, IC_TOK, Vocab


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt['args']

    vocab = Vocab.__new__(Vocab)
    vocab.itos = ckpt['vocab_itos']
    vocab.stoi = ckpt['vocab_stoi']

    vocab_size = len(vocab)
    encoder = Encoder(vocab_size, args['embed_size'], args['hidden_size'],
                      n_layers=2, dropout=0.0)
    decoder = Decoder(args['embed_size'], args['hidden_size'], vocab_size,
                      n_layers=1, dropout=0.0)
    model = Seq2Seq(encoder, decoder, tie_embeddings=True).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, vocab


def encode_prefix(steps, family, vocab):
    """Encode a prefix sequence with family token."""
    fam_tok = IGBT_TOK if family.lower() == 'igbt' else IC_TOK
    encoded = [fam_tok] + vocab.encode(steps)
    return torch.tensor(encoded, dtype=torch.long)


def predict_next_step(model, src_tensor, vocab, device, top_k=5):
    """
    Predict the next step given a prefix.
    Returns list of (step_name, probability) tuples, sorted by probability.
    """
    src = src_tensor.unsqueeze(1).to(device)  # (seq_len, 1)
    with torch.no_grad():
        encoder_output, encoder_hidden = model.encoder(src)
        hidden = model.decoder.init_hidden(encoder_hidden)
        sos_input = torch.tensor([SOS], device=device)
        output, _, _ = model.decoder(sos_input, hidden, encoder_output)
        # output is log_softmax: (1, vocab_size)
        probs = output.exp().squeeze(0)  # (vocab_size,)

    top_probs, top_indices = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token = vocab.itos[idx]
        if token not in ('<pad>', '<unk>', '<sos>', '<eos>', '<igbt>', '<ic>'):
            results.append((token, prob))
    return results


def complete_sequence(model, src_tensor, vocab, device, max_len=80):
    """
    Complete a sequence given a prefix using greedy decoding.
    Returns list of predicted step strings.
    """
    src = src_tensor.unsqueeze(1).to(device)  # (seq_len, 1)
    with torch.no_grad():
        encoder_output, encoder_hidden = model.encoder(src)
        hidden = model.decoder.init_hidden(encoder_hidden)
        input_tok = torch.tensor([SOS], device=device)
        predicted = []
        for _ in range(max_len):
            output, hidden, _ = model.decoder(input_tok, hidden, encoder_output)
            top1 = output.argmax(1)
            token_idx = top1.item()
            if token_idx == EOS:
                break
            if token_idx not in (PAD, SOS):
                predicted.append(vocab.itos[token_idx])
            input_tok = top1
    return predicted


def read_eval_input(csv_path):
    """
    Read eval input CSV. Expected format: SEQUENCE_ID, STEP (long format).
    Returns dict[seq_id] -> list[step_str].
    """
    sequences = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                seq_id, step = row[0], row[1]
                sequences[seq_id].append(step)
    return dict(sequences)


def detect_family(steps):
    """Heuristic to detect family from step content. Default to IGBT."""
    step_str = ' '.join(steps).upper()
    if 'EPITAXIAL' in step_str or 'EPITAXY' in step_str:
        return 'igbt'
    return 'ic'


def main():
    p = argparse.ArgumentParser(description='Inference for process sequences')
    p.add_argument('--task', choices=['next_step', 'completion'],
                   default='next_step')
    p.add_argument('--input', type=str, help='input CSV (eval format)')
    p.add_argument('--output', type=str, help='output CSV for predictions')
    p.add_argument('--checkpoint', type=str, default='.save/best.pt')
    p.add_argument('--family', type=str, default=None,
                   help='force family (igbt/ic). Auto-detect if not set.')
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--max_len', type=int, default=80)
    p.add_argument('--interactive', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[!] device: {device}")
    print(f"[!] loading model from {args.checkpoint}")
    model, vocab = load_model(args.checkpoint, device)
    print(f"[!] vocab size: {len(vocab)}")

    if args.interactive:
        interactive_mode(model, vocab, device, args)
        return

    if not args.input or not args.output:
        print("Error: --input and --output required for batch mode")
        return

    sequences = read_eval_input(args.input)
    print(f"[!] loaded {len(sequences)} sequences from {args.input}")

    if args.task == 'next_step':
        run_next_step(model, vocab, device, sequences, args)
    else:
        run_completion(model, vocab, device, sequences, args)


def run_next_step(model, vocab, device, sequences, args):
    """Task 1: next-step prediction."""
    results = []
    for seq_id, steps in sequences.items():
        family = args.family or detect_family(steps)
        src = encode_prefix(steps, family, vocab)
        preds = predict_next_step(model, src, vocab, device, args.top_k)
        for rank, (step, prob) in enumerate(preds, 1):
            results.append({
                'SEQUENCE_ID': seq_id,
                'RANK': rank,
                'PREDICTED_STEP': step,
                'PROBABILITY': f"{prob:.6f}",
            })

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['SEQUENCE_ID', 'RANK',
                                               'PREDICTED_STEP', 'PROBABILITY'])
        writer.writeheader()
        writer.writerows(results)
    print(f"[!] wrote {len(results)} predictions to {args.output}")


def run_completion(model, vocab, device, sequences, args):
    """Task 2: sequence completion."""
    results = []
    for seq_id, steps in sequences.items():
        family = args.family or detect_family(steps)
        src = encode_prefix(steps, family, vocab)
        completed = complete_sequence(model, src, vocab, device, args.max_len)
        for step_idx, step in enumerate(completed):
            results.append({
                'SEQUENCE_ID': seq_id,
                'STEP': step,
            })

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['SEQUENCE_ID', 'STEP'])
        writer.writeheader()
        writer.writerows(results)
    print(f"[!] wrote {len(results)} completion steps to {args.output}")


def interactive_mode(model, vocab, device, args):
    """Interactive testing mode."""
    print("\n[Interactive Mode] Enter process steps separated by ' -> '")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            line = input("Steps: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ('quit', 'exit', 'q'):
            break
        steps = [s.strip() for s in line.split('->')]
        family = args.family or detect_family(steps)
        src = encode_prefix(steps, family, vocab)

        print(f"\n  Family: {family}")
        print(f"  Prefix length: {len(steps)} steps")

        # Next step
        preds = predict_next_step(model, src, vocab, device, args.top_k)
        print(f"\n  Next-step predictions:")
        for rank, (step, prob) in enumerate(preds, 1):
            print(f"    {rank}. {step} ({prob:.4f})")

        # Completion
        completed = complete_sequence(model, src, vocab, device, args.max_len)
        print(f"\n  Full completion ({len(completed)} steps):")
        for i, step in enumerate(completed[:10]):
            print(f"    {i+1}. {step}")
        if len(completed) > 10:
            print(f"    ... ({len(completed)-10} more steps)")
        print()


if __name__ == "__main__":
    main()
