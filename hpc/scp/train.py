import os
import math
import argparse
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import load_data, PAD, SOS


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
    p.add_argument('-augment_factor', type=int, default=3,
                   help='number of random cuts per sequence')
    p.add_argument('-teacher_forcing', type=float, default=0.5,
                   help='teacher forcing ratio')
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def step_loss(model, src, trg, vocab_size, teacher_forcing_ratio):
    output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
    return F.nll_loss(output[1:].reshape(-1, vocab_size),
                      trg[1:].reshape(-1), ignore_index=PAD)


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


def evaluate(model, val_iter, vocab_size, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in val_iter:
            src, trg = src.to(device), trg.to(device)
            total_loss += step_loss(model, src, trg, vocab_size, 0.0).item()
    return total_loss / max(len(val_iter), 1)


def train(model, optimizer, train_iter, vocab_size, grad_clip, device,
          teacher_forcing_ratio):
    model.train()
    total_loss = 0
    n_batches = 0
    for b, (src, trg) in enumerate(train_iter):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        loss = step_loss(model, src, trg, vocab_size, teacher_forcing_ratio)
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
    train_iter, val_iter, vocab = load_data(
        data_dir=args.data_dir,
        families=args.families,
        batch_size=args.batch_size,
        max_sequences=args.max_sequences,
        augment_factor=args.augment_factor,
    )
    vocab_size = len(vocab)
    print(f"[VOCAB] {vocab_size} tokens")
    print(f"[TRAIN] {len(train_iter)} batches  [VAL] {len(val_iter)} batches")

    print("[!] building model...")
    encoder = Encoder(vocab_size, args.embed_size, args.hidden_size,
                      n_layers=2, dropout=0.3)
    decoder = Decoder(args.embed_size, args.hidden_size, vocab_size,
                      n_layers=1, dropout=0.3)
    seq2seq = Seq2Seq(encoder, decoder, tie_embeddings=True).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3)

    n_params = sum(p.numel() for p in seq2seq.parameters() if p.requires_grad)
    print(f"[!] {n_params:,} trainable parameters")
    print(seq2seq)

    best_val_loss, no_improve = None, 0
    for e in range(1, args.epochs + 1):
        train_loss = train(seq2seq, optimizer, train_iter, vocab_size,
                           args.grad_clip, device, args.teacher_forcing)
        val_loss = evaluate(seq2seq, val_iter, vocab_size, device)
        scheduler.step(val_loss)

        top1, top3, top5 = compute_accuracy(seq2seq, val_iter, vocab_size, device)
        print(f"[Epoch {e}] train_loss:{train_loss:.3f} val_loss:{val_loss:.3f} "
              f"val_pp:{math.exp(val_loss):.1f} "
              f"top1:{top1:.3f} top3:{top3:.3f} top5:{top5:.3f}")

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

    print(f"\n[DONE] best val_loss: {best_val_loss:.3f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
