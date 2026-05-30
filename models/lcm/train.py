import os
import math
import argparse
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import load_data, PAD, SOS, EOS
from validator import validate_sequence


def parse_arguments():
    p = argparse.ArgumentParser(description='Train Seq2Seq on process sequences')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='batch size for training')
    p.add_argument('-lr', type=float, default=3e-4,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='gradient clip norm')
    p.add_argument('-hidden_size', type=int, default=256,
                   help='RNN hidden size')
    p.add_argument('-embed_size', type=int, default=128,
                   help='token embedding size')
    p.add_argument('-patience', type=int, default=10,
                   help='early-stop after N epochs without val-loss improvement')
    p.add_argument('-data_dir', type=str, default='../../data/raw/infineon',
                   help='path to data directory containing family subdirs')
    p.add_argument('-families', type=str, nargs='+', default=['igbt', 'ic'],
                   help='families to train on')
    p.add_argument('-max_sequences', type=int, default=None,
                   help='limit sequences per family (None=all)')
    p.add_argument('-augment_factor', type=int, default=5,
                   help='number of random cuts per sequence')
    p.add_argument('-teacher_forcing_start', type=float, default=0.5,
                   help='initial teacher forcing ratio')
    p.add_argument('-teacher_forcing_end', type=float, default=0.1,
                   help='final teacher forcing ratio')
    p.add_argument('-label_smoothing', type=float, default=0.1,
                   help='label smoothing factor')
    p.add_argument('-dec_layers', type=int, default=2,
                   help='number of decoder GRU layers')
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def step_loss(model, src, trg, vocab_size, teacher_forcing_ratio,
              label_smoothing=0.0):
    output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
    # output is log_softmax: (seq_len, batch, vocab)
    logits = output[1:]  # skip position 0
    targets = trg[1:]    # skip SOS

    if label_smoothing > 0:
        # Manual label-smoothed NLL loss
        log_probs = logits.reshape(-1, vocab_size)
        tgt_flat = targets.reshape(-1)
        mask = (tgt_flat != PAD).float()
        nll = F.nll_loss(log_probs, tgt_flat, reduction='none')
        # Smooth: (1-eps)*nll + eps*uniform_entropy
        smooth_loss = -log_probs.mean(dim=1)  # uniform prior
        loss_per_token = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss
        return (loss_per_token * mask).sum() / mask.sum()
    else:
        return F.nll_loss(logits.reshape(-1, vocab_size),
                          targets.reshape(-1), ignore_index=PAD)


def compute_accuracy(model, data_iter, vocab_size, device):
    """Compute top-1/3/5 next-step accuracy on validation set."""
    model.eval()
    correct_1 = correct_3 = correct_5 = total = 0
    with torch.no_grad():
        for src, trg in data_iter:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)
            # Only look at position 1 (first predicted token after SOS)
            preds = output[1]  # (batch, vocab)
            targets = trg[1]  # (batch,)
            mask = targets != PAD
            if mask.sum() == 0:
                continue
            preds = preds[mask]
            targets = targets[mask]
            top5 = preds.topk(min(5, vocab_size), dim=1).indices
            correct_1 += (top5[:, 0] == targets).sum().item()
            correct_3 += (top5[:, :3] == targets.unsqueeze(1)).any(1).sum().item()
            correct_5 += (top5 == targets.unsqueeze(1)).any(1).sum().item()
            total += targets.size(0)
    if total == 0:
        return 0, 0, 0
    return correct_1 / total, correct_3 / total, correct_5 / total


def compute_completion_accuracy(model, data_iter, device, vocab, max_len=200):
    """
    Test full greedy sequence completion with FREE generation (no target length hint).
    Decodes until EOS or max_len. Reports:
      - tok_acc: token-level accuracy (clipped to target length)
      - exact_rate: fraction of sequences with perfect match
      - eos_rate: fraction that correctly terminated with EOS
      - norm_edit_dist: average normalized edit distance
      - valid_rate: fraction of completions that pass all 10 process-logic rules
    """
    model.eval()
    total_tokens = 0
    correct_tokens = 0
    exact_matches = 0
    eos_terminations = 0
    valid_completions = 0
    total_seqs = 0
    total_edit_dist = 0.0

    with torch.no_grad():
        for src, trg in data_iter:
            src, trg = src.to(device), trg.to(device)
            batch_size = src.size(1)
            # Free generation: decode up to max_len, no target provided
            output = model(src, trg=None, max_len=max_len)
            # output: (max_len, batch, vocab)
            pred_tokens = output[1:].argmax(-1)  # (max_len-1, batch)
            target_tokens = trg[1:]  # (trg_len-1, batch)

            for b in range(batch_size):
                pred_seq = pred_tokens[:, b].cpu().tolist()
                tgt_seq = target_tokens[:, b].cpu().tolist()
                src_seq = src[:, b].cpu().tolist()
                # Target length: up to (but not including) PAD
                tgt_len = next((i for i, t in enumerate(tgt_seq) if t == PAD),
                               len(tgt_seq))
                if tgt_len == 0:
                    continue
                total_seqs += 1
                tgt_clipped = tgt_seq[:tgt_len]  # includes EOS at end

                # Check if model produced EOS
                try:
                    pred_eos_pos = pred_seq.index(EOS)
                    pred_clipped = pred_seq[:pred_eos_pos + 1]  # include EOS
                    eos_terminations += 1
                except ValueError:
                    pred_clipped = pred_seq[:tgt_len]  # no EOS found, clip

                # Token accuracy (compare up to min of both lengths)
                compare_len = min(len(pred_clipped), tgt_len)
                matches = sum(1 for i in range(compare_len)
                              if pred_clipped[i] == tgt_clipped[i])
                correct_tokens += matches
                total_tokens += tgt_len

                # Exact match
                if pred_clipped == tgt_clipped:
                    exact_matches += 1

                # Normalized edit distance (Levenshtein / max_len)
                ed = _edit_distance(pred_clipped, tgt_clipped)
                total_edit_dist += ed / max(len(pred_clipped), tgt_len, 1)

                # Process-logic validation:
                # Reconstruct full sequence = prefix (src minus special tokens)
                #                           + generated suffix (minus EOS)
                # Special tokens are indices 0-5: PAD,UNK,SOS,EOS,<igbt>,<ic>
                prefix_ids = [t for t in src_seq if t >= 6]
                suffix_ids = [t for t in pred_clipped if t >= 6]
                full_ids = prefix_ids + suffix_ids
                full_steps = [vocab.itos[t] for t in full_ids
                              if t < len(vocab.itos)]
                if full_steps:
                    rule_violations = validate_sequence(full_steps)
                    if not rule_violations:
                        valid_completions += 1

    tok_acc = correct_tokens / max(total_tokens, 1)
    exact_rate = exact_matches / max(total_seqs, 1)
    eos_rate = eos_terminations / max(total_seqs, 1)
    norm_ed = total_edit_dist / max(total_seqs, 1)
    valid_rate = valid_completions / max(total_seqs, 1)
    return tok_acc, exact_rate, eos_rate, norm_ed, valid_rate


def _edit_distance(seq1, seq2):
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


def evaluate(model, val_iter, vocab_size, device, label_smoothing=0.0):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in val_iter:
            src, trg = src.to(device), trg.to(device)
            total_loss += step_loss(model, src, trg, vocab_size, 0.0,
                                    label_smoothing).item()
    return total_loss / max(len(val_iter), 1)


def train(model, optimizer, train_iter, vocab_size, grad_clip, device,
          teacher_forcing_ratio, label_smoothing=0.0):
    model.train()
    total_loss = 0
    n_batches = 0
    for b, (src, trg) in enumerate(train_iter):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        loss = step_loss(model, src, trg, vocab_size, teacher_forcing_ratio,
                         label_smoothing)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        if b % 50 == 0 and b != 0:
            avg = total_loss / n_batches
            print(f"  [batch {b}] loss:{avg:.3f} pp:{math.exp(avg):.1f}")
    return total_loss / max(n_batches, 1)


def main():
    args = parse_arguments()
    device = get_device()
    print(f"[!] device: {device}")

    print("[!] loading data...")
    train_iter, val_iter, test_iter, vocab = load_data(
        data_dir=args.data_dir,
        families=args.families,
        batch_size=args.batch_size,
        max_sequences=args.max_sequences,
        augment_factor=args.augment_factor,
    )
    vocab_size = len(vocab)
    print(f"[VOCAB] {vocab_size} tokens")
    print(f"[TRAIN] {len(train_iter)} batches  [VAL] {len(val_iter)} batches  [TEST] {len(test_iter)} batches")

    print("[!] building model...")
    encoder = Encoder(vocab_size, args.embed_size, args.hidden_size,
                      n_layers=2, dropout=0.3)
    decoder = Decoder(args.embed_size, args.hidden_size, vocab_size,
                      n_layers=args.dec_layers, dropout=0.3)
    seq2seq = Seq2Seq(encoder, decoder, tie_embeddings=True).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3)

    n_params = sum(p.numel() for p in seq2seq.parameters() if p.requires_grad)
    print(f"[!] {n_params:,} trainable parameters")
    print(f"[!] label_smoothing={args.label_smoothing} "
          f"tf_schedule={args.teacher_forcing_start}→{args.teacher_forcing_end} "
          f"dec_layers={args.dec_layers}")
    print(seq2seq)

    best_val_loss, no_improve = None, 0
    for e in range(1, args.epochs + 1):
        # Scheduled teacher forcing: linear decay
        tf_ratio = args.teacher_forcing_start - (
            (args.teacher_forcing_start - args.teacher_forcing_end)
            * (e - 1) / max(args.epochs - 1, 1))

        train_loss = train(seq2seq, optimizer, train_iter, vocab_size,
                           args.grad_clip, device, tf_ratio,
                           args.label_smoothing)
        val_loss = evaluate(seq2seq, val_iter, vocab_size, device,
                            args.label_smoothing)
        scheduler.step(val_loss)

        top1, top3, top5 = compute_accuracy(seq2seq, val_iter, vocab_size, device)

        # Full completion eval every 5 epochs
        comp_str = ""
        if e % 5 == 0 or e == 1:
            tok_acc, exact_rate, eos_rate, norm_ed, valid_rate = compute_completion_accuracy(
                seq2seq, val_iter, device, vocab)
            comp_str = (f" comp_tok:{tok_acc:.3f} exact:{exact_rate:.3f}"
                        f" eos_rate:{eos_rate:.3f} edit_dist:{norm_ed:.3f}"
                        f" valid:{valid_rate:.3f}")

        print(f"[Epoch {e}] train:{train_loss:.3f} val:{val_loss:.3f} "
              f"pp:{math.exp(val_loss):.1f} tf:{tf_ratio:.2f} "
              f"top1:{top1:.3f} top3:{top3:.3f} top5:{top5:.3f}{comp_str}")

        if best_val_loss is None or val_loss < best_val_loss:
            print("  [!] saving best model...")
            os.makedirs(".save", exist_ok=True)
            torch.save({
                'model_state': seq2seq.state_dict(),
                'vocab_itos': vocab.itos,
                'vocab_stoi': vocab.stoi,
                'args': vars(args),
            }, './.save/best.pt')
            best_val_loss, no_improve = val_loss, 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  [!] early stop after {e} epochs")
                break

    # Final eval on validation set
    tok_acc, exact_rate, eos_rate, norm_ed, valid_rate = compute_completion_accuracy(
        seq2seq, val_iter, device, vocab)
    print(f"\n[DONE] best val_loss: {best_val_loss:.3f}")
    print(f"[VAL]  completion: tok_acc={tok_acc:.3f} exact={exact_rate:.3f} "
          f"eos_rate={eos_rate:.3f} edit_dist={norm_ed:.3f} valid={valid_rate:.3f}")

    # Final eval on held-out test set
    print("\n[!] Evaluating on TEST set...")
    test_loss = evaluate(seq2seq, test_iter, vocab_size, device, args.label_smoothing)
    top1, top3, top5 = compute_accuracy(seq2seq, test_iter, vocab_size, device)
    tok_acc, exact_rate, eos_rate, norm_ed, valid_rate = compute_completion_accuracy(
        seq2seq, test_iter, device, vocab)
    print(f"[TEST] loss={test_loss:.3f} pp={math.exp(test_loss):.1f} "
          f"top1={top1:.3f} top3={top3:.3f} top5={top5:.3f}")
    print(f"[TEST] completion: tok_acc={tok_acc:.3f} exact={exact_rate:.3f} "
          f"eos_rate={eos_rate:.3f} edit_dist={norm_ed:.3f} valid={valid_rate:.3f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
