"""
demo.py — smoke-test / showcase for the trained GPT checkpoint.

Loads gpt_ckpt.pt and runs three inline examples — one per task.

Task 3 scoring is copied CHARACTER-FOR-CHARACTER from predict.py's
task3_anomaly() so the numbers are identical to the submission.
The demo sequences are real full sequences from eval_input_anomaly.csv
(same file scored in the submission), so valid≈0.71 / invalid≈0.45.

Usage:
    python3 demo.py --data <path/to/training_data>   # contains generate_sequences.py
    python3 demo.py --data <path> --gpu              # force CUDA
    python3 demo.py --data <path> --ckpt other.pt
"""
import argparse
import csv
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── CLI ───────────────────────────────────────────────────────────────────────

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="gpt_ckpt.pt")
p.add_argument("--gpu", action="store_true")
p.add_argument(
    "--data",
    default="/leonardo_scratch/large/usertrain/a08trc0z"
            "/zero_one_hack_01/tracks/industrial-infineon/training_data",
    help="directory containing generate_sequences.py and (optionally) "
         "eval_input_anomaly.csv",
)
args = p.parse_args()

if args.gpu:
    if not torch.cuda.is_available():
        print("ERROR: --gpu specified but no CUDA device found", flush=True)
        sys.exit(1)
    dev = "cuda"
else:
    dev = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device : {dev}", flush=True)
if dev == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}", flush=True)

# ── import validate_sequence (same sys.path trick as predict.py) ──────────────

sys.path.insert(0, args.data)
from generate_sequences import validate_sequence

# ── model (identical to predict.py) ──────────────────────────────────────────

T = 160

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3*d); s.proj = nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        B, L, D = x.shape
        qkv = s.qkv(s.ln1(x)).reshape(B, L, 3, s.h, D//s.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(a.transpose(1, 2).reshape(B, L, D))
        return x + s.mlp(s.ln2(x))

class GPT(nn.Module):
    def __init__(s, V, d=256, h=8, L=6):
        super().__init__()
        s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(T, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, V)
    def forward(s, idx):
        B, L = idx.shape
        x = s.tok(idx) + s.pos(torch.arange(L, device=idx.device))
        for b in s.blocks: x = b(x)
        return s.head(s.lnf(x))

# ── load checkpoint ───────────────────────────────────────────────────────────

print(f"\nloading {args.ckpt} …", flush=True)
ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
stoi, itos = ckpt["stoi"], ckpt["itos"]
V = len(itos)
PAD_ID  = stoi["<PAD>"]
BOS_ID  = stoi["<BOS>"]
SHIP_ID = stoi.get("SHIP LOT")

model = GPT(V).to(dev)
model.load_state_dict(ckpt["model"])
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"vocab   : {V} tokens", flush=True)
print(f"params  : {n_params:,} ({n_params/1e6:.1f}M)", flush=True)
print(f"keys    : {list(ckpt.keys())}", flush=True)

# ── encode (identical to predict.py) ─────────────────────────────────────────

def encode(family, steps):
    fam_id = stoi.get(f"<FAM_{family}>")
    if fam_id is None:
        raise ValueError(f"unknown family: {family}")
    ids = [BOS_ID, fam_id]
    for s in steps:
        if s in stoi:
            ids.append(stoi[s])
        else:
            print(f"  WARN unknown step (skipped): {s!r}", flush=True)
    return ids[:T]

# ── task inference helpers ────────────────────────────────────────────────────

@torch.no_grad()
def top5_next(family, steps):
    ids = encode(family, steps)
    x   = torch.tensor([ids], dtype=torch.long, device=dev)
    lg  = model(x)[0, len(ids) - 1]
    top = torch.topk(lg, 5)
    probs = torch.softmax(lg, 0)
    return [(itos[i], round(float(probs[i]), 4)) for i in top.indices.tolist()]

@torch.no_grad()
def task2_complete(family, steps, max_new=250):
    ids = encode(family, steps)
    generated = []
    for _ in range(max_new):
        if len(ids) >= T:
            break
        x       = torch.tensor([ids], dtype=torch.long, device=dev)
        next_id = int(torch.argmax(model(x)[0, len(ids) - 1]))
        if next_id == PAD_ID:
            break
        ids.append(next_id)
        generated.append(itos[next_id])
        if next_id == SHIP_ID:
            break
    return generated

# ── task3_anomaly — COPIED VERBATIM FROM predict.py ──────────────────────────
#
# Any change here must be mirrored back to predict.py (or vice-versa).
# The score formula: exp(-mean_NLL_per_step) over step tokens only
# (positions 2+ in ids, skipping BOS and FAM conditioning tokens).

def task3_anomaly(family, steps):
    # Rule checker
    violations   = validate_sequence(steps)
    is_valid     = 0 if violations else 1
    pred_rule    = violations[0].rule if violations else ""

    # GPT surprisal: mean NLL over step tokens (skip BOS + FAM prefix positions)
    ids = encode(family, steps)
    if len(ids) >= 3:
        x       = torch.tensor([ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits  = model(x[:, :-1])[0]          # (L-1, V)
        targets = torch.tensor(ids[1:], dtype=torch.long, device=dev)
        # score only step tokens (positions 2+ in ids, i.e. index 1+ in targets after shift)
        nll     = F.cross_entropy(logits[2:], targets[2:], reduction="mean").item()
    else:
        nll = 10.0

    # exp(-nll) ∈ (0,1]: geometric mean per-step probability; higher = more valid
    score = math.exp(-nll)

    # If rule checker already flagged invalid, pull score below 0.5 if it isn't already
    if not is_valid:
        score = min(score, 0.45)

    return is_valid, round(score, 4), pred_rule

# ── DEMO ──────────────────────────────────────────────────────────────────────

SEP = " | "

print("\n" + "="*70, flush=True)
print("TASK 1 — Next-step prediction (Top-5)", flush=True)
print("="*70, flush=True)

examples_t1 = [
    ("MOSFET", ["RECEIVE WAFER LOT", "PRE CLEAN INSPECTION",
                "PRE CLEAN WAFER", "HF DIP", "DEPOSIT FIELD OXIDE"]),
    ("IGBT",   ["RECEIVE WAFER LOT", "PRE CLEAN INSPECTION",
                "PRE CLEAN WAFER", "HF DIP", "DRY WAFER", "EPITAXIAL WAFER CHECK"]),
    ("IC",     ["RECEIVE WAFER LOT", "PRE CLEAN INSPECTION",
                "PRE CLEAN WAFER", "HF DIP", "DEPOSIT FIELD OXIDE",
                "ALIGN MASK LEVEL 1", "EXPOSE LITHO LEVEL 1"]),
]

for family, prefix in examples_t1:
    preds = top5_next(family, prefix)
    print(f"\n  [{family}] prefix ({len(prefix)} steps): … {SEP.join(prefix[-2:])}", flush=True)
    for rank, (step, prob) in enumerate(preds, 1):
        print(f"    Rank {rank}: {step!r:<45} p={prob:.4f}", flush=True)

print("\n" + "="*70, flush=True)
print("TASK 2 — Sequence completion (greedy autoregressive)", flush=True)
print("="*70, flush=True)

family_t2 = "MOSFET"
prefix_t2 = [
    "RECEIVE WAFER LOT", "PRE CLEAN INSPECTION", "PRE CLEAN WAFER", "HF DIP",
    "DEPOSIT FIELD OXIDE", "FIELD PATTERN INSPECTION",
    "ALIGN MASK LEVEL 1", "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST",
    "PATTERN INSPECTION LEVEL 1", "OXIDE ETCH", "STRIP PHOTORESIST",
    "CLEAN AFTER FIELD ETCH",
]

completion = task2_complete(family_t2, prefix_t2)
print(f"\n  [{family_t2}] prefix: {len(prefix_t2)} steps", flush=True)
print(f"  generated: {len(completion)} steps", flush=True)
terminated = bool(completion) and completion[-1] == "SHIP LOT"
print(f"  terminated at SHIP LOT: {terminated}", flush=True)
print(f"\n  Completion sequence:", flush=True)
for i, step in enumerate(completion, 1):
    marker = " ←" if step == "SHIP LOT" else ""
    print(f"    {i:>3}. {step}{marker}", flush=True)

print("\n" + "="*70, flush=True)
print("TASK 3 — Anomaly detection", flush=True)
print("="*70, flush=True)

# Pull two REAL full sequences from eval_input_anomaly.csv:
# one that validate_sequence() marks valid, one it marks invalid.
# These are the same sequences scored in the submission → scores match exactly.
anomaly_csv = "eval_input_anomaly.csv"
demo_valid   = None   # (ex_id, family, steps)
demo_invalid = None

print(f"\n  Scanning {anomaly_csv} for one valid + one invalid sequence …", flush=True)
with open(anomaly_csv, newline="") as f:
    for row in csv.DictReader(f):
        ex_id  = row["EXAMPLE_ID"].strip()
        family = row["FAMILY"].strip()
        steps  = [s.strip() for s in row["SEQUENCE"].split("|") if s.strip()]
        violations = validate_sequence(steps)
        if demo_valid is None and not violations:
            demo_valid = (ex_id, family, steps)
        if demo_invalid is None and violations:
            demo_invalid = (ex_id, family, steps)
        if demo_valid and demo_invalid:
            break

for label, item in [("Valid sequence  ", demo_valid),
                    ("Violated sequence", demo_invalid)]:
    if item is None:
        print(f"  {label}: NOT FOUND in {anomaly_csv}", flush=True)
        continue
    ex_id, family, steps = item
    is_valid, score, pred_rule = task3_anomaly(family, steps)
    print(f"\n  {label}  [{ex_id}]  family={family}  len={len(steps)} steps", flush=True)
    print(f"    IS_VALID       : {is_valid}", flush=True)
    print(f"    PREDICTED_RULE : {pred_rule if pred_rule else '(none)'}", flush=True)
    print(f"    SCORE          : {score}", flush=True)

print("\n" + "="*70, flush=True)
print("DEMO COMPLETE", flush=True)
print("="*70, flush=True)
