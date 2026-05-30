"""
predict.py  DATA_DIR [CKPT]

Produces three submission CSVs from the GPT checkpoint:
  task1_predictions.csv  — next-step top-5 (Task 1)
  task2_predictions.csv  — autoregressive completion (Task 2)
  task3_predictions.csv  — anomaly detection (Task 3)

Requires eval_input_valid.csv and eval_input_anomaly.csv in DATA_DIR.
Task 3 rule attribution uses validate_sequence() from generate_sequences.py.
"""
import sys, csv, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

DATA = Path(sys.argv[1])
CKPT = sys.argv[2] if len(sys.argv) > 2 else "gpt_ckpt.pt"
sys.path.insert(0, str(DATA))
from generate_sequences import validate_sequence

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, flush=True)

# ── model ────────────────────────────────────────────────────────────────────

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

ckpt = torch.load(CKPT, map_location=dev)
stoi, itos = ckpt["stoi"], ckpt["itos"]
V = len(itos)
PAD_ID   = stoi["<PAD>"]
BOS_ID   = stoi["<BOS>"]
SHIP_ID  = stoi.get("SHIP LOT")

model = GPT(V).to(dev)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"loaded {CKPT}  vocab={V}", flush=True)

# ── encoding helpers ─────────────────────────────────────────────────────────

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

@torch.no_grad()
def logits_at(ids):
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    return model(x)[0, len(ids) - 1]   # (V,)

# ── Task 1: next-step top-5 ──────────────────────────────────────────────────

def task1_top5(family, partial_steps):
    ids = encode(family, partial_steps)
    lg  = logits_at(ids)
    return [itos[i] for i in torch.topk(lg, 5).indices.tolist()]

# ── Task 2: autoregressive completion ────────────────────────────────────────

def task2_complete(family, partial_steps, max_new=250):
    ids = encode(family, partial_steps)
    generated = []
    with torch.no_grad():
        for _ in range(max_new):
            if len(ids) >= T:
                break
            lg      = model(torch.tensor([ids], dtype=torch.long, device=dev))[0, len(ids) - 1]
            next_id = int(torch.argmax(lg))
            if next_id == PAD_ID:
                break
            ids.append(next_id)
            generated.append(itos[next_id])
            if next_id == SHIP_ID:
                break
    return generated

# ── Task 3: anomaly detection ─────────────────────────────────────────────────
#
# Strategy:
#   IS_VALID   — rule checker (validate_sequence) is authoritative for the 10 patterns
#   SCORE      — GPT mean per-step log-prob converted to [0,1] via exp(); used for AUC
#   PREDICTED_RULE — first violation from rule checker (empty if valid)
#
# Combining: rule checker catches explicit pattern violations; GPT surprisal
# provides a graded score that may also catch soft/structural anomalies.

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

# ── read eval files ───────────────────────────────────────────────────────────

valid_path   = DATA / "eval_input_valid.csv"
anomaly_path = DATA / "eval_input_anomaly.csv"

missing = [p for p in (valid_path, anomaly_path) if not p.exists()]
if missing:
    print(f"ERROR: eval file(s) not found: {[str(p) for p in missing]}", flush=True)
    print("Wait for organizers to release eval files, then re-run.", flush=True)
    sys.exit(1)

# ── Tasks 1 & 2 — from eval_input_valid.csv ──────────────────────────────────

print("Reading eval_input_valid.csv …", flush=True)
with open(valid_path, newline="") as f:
    rows = list(csv.DictReader(f))

print(f"  {len(rows)} rows", flush=True)

out1 = open("task1_predictions.csv", "w", newline="")
out2 = open("task2_predictions.csv", "w", newline="")
w1   = csv.writer(out1)
w2   = csv.writer(out2)
w1.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
w2.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])

for i, row in enumerate(rows):
    ex_id  = row["EXAMPLE_ID"].strip()
    family = row["FAMILY"].strip()
    steps  = [s.strip() for s in row["PARTIAL_SEQUENCE"].split("|") if s.strip()]

    top5       = task1_top5(family, steps)
    completion = task2_complete(family, steps)

    w1.writerow([ex_id] + top5)
    w2.writerow([ex_id, "|".join(completion)])

    if (i + 1) % 100 == 0:
        print(f"  T1/T2: {i+1}/{len(rows)}", flush=True)

out1.close(); out2.close()
print("wrote task1_predictions.csv  task2_predictions.csv", flush=True)

# ── Task 3 — from eval_input_anomaly.csv ─────────────────────────────────────

print("Reading eval_input_anomaly.csv …", flush=True)
with open(anomaly_path, newline="") as f:
    rows = list(csv.DictReader(f))

print(f"  {len(rows)} rows", flush=True)

out3 = open("task3_predictions.csv", "w", newline="")
w3   = csv.writer(out3)
w3.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])

rule_hits = 0
for i, row in enumerate(rows):
    ex_id  = row["EXAMPLE_ID"].strip()
    family = row["FAMILY"].strip()
    steps  = [s.strip() for s in row["SEQUENCE"].split("|") if s.strip()]

    is_valid, score, pred_rule = task3_anomaly(family, steps)
    if pred_rule:
        rule_hits += 1
    w3.writerow([ex_id, is_valid, score, pred_rule])

    if (i + 1) % 200 == 0:
        print(f"  T3: {i+1}/{len(rows)}  rule_hits_so_far={rule_hits}", flush=True)

out3.close()
print(f"wrote task3_predictions.csv  (rule hits: {rule_hits}/{len(rows)})", flush=True)
print("done.", flush=True)
