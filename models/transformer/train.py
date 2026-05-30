import sys, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

DATA = sys.argv[1]
OOD = "--ood" in sys.argv
sys.path.insert(0, DATA)
from generate_sequences import read_csv_sequences

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, flush=True)

families = ["MOSFET", "IGBT", "IC"]
raw = {f: read_csv_sequences(Path(f"{DATA}/{f}_variants.csv")) for f in families}

random.seed(0)
split = {}  # family → (train_list, test_list) of (fam, seq)
for f in families:
    ids = list(raw[f]); random.shuffle(ids)
    t, v = [], []
    for i, sid in enumerate(ids):
        (v if i < 100 else t).append((f, raw[f][sid]))
    split[f] = (t, v)

if OOD:
    train_families = ["MOSFET", "IGBT"]
    id_families    = ["MOSFET", "IGBT"]
    ood_families   = ["IC"]
    print("=== GPT OOD mode: train on MOSFET+IGBT, test on IC ===", flush=True)
else:
    train_families = families
    id_families    = families
    ood_families   = []
    print("=== GPT full in-dist mode ===", flush=True)

train = [ex for f in train_families for ex in split[f][0]]

# vocab from ALL families so IC steps are representable even in OOD mode
steps = sorted({s for f in families for q in raw[f].values() for s in q})
PAD, BOS = "<PAD>", "<BOS>"
fam_tok = {f: f"<FAM_{f}>" for f in families}
itos = [PAD, BOS] + [fam_tok[f] for f in families] + steps
stoi = {s: i for i, s in enumerate(itos)}
V = len(itos); PAD_ID = stoi[PAD]
print("vocab:", V, flush=True)
T = 160

def encode(f, seq): return ([stoi[BOS], stoi[fam_tok[f]]] + [stoi[s] for s in seq])[:T]

def batchify(exs):
    enc = [encode(f, s) for f, s in exs]
    m = max(len(e) for e in enc)
    X = torch.full((len(enc), m), PAD_ID, dtype=torch.long)
    for i, e in enumerate(enc): X[i, :len(e)] = torch.tensor(e)
    return X

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3*d); s.proj = nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        B, T, D = x.shape
        qkv = s.qkv(s.ln1(x)).reshape(B, T, 3, s.h, D//s.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(a.transpose(1, 2).reshape(B, T, D))
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

model = GPT(V).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
print("params:", sum(p.numel() for p in model.parameters())/1e6, "M", flush=True)

BS = 64
for ep in range(8):
    random.shuffle(train); model.train(); tot = 0; nb = 0
    for i in range(0, len(train), BS):
        X = batchify(train[i:i+BS]).to(dev)
        logits = model(X[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, V), X[:, 1:].reshape(-1), ignore_index=PAD_ID)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
    print(f"epoch {ep} loss {tot/nb:.4f}", flush=True)

def eval_split(seqs, label):
    t1 = t3 = t5 = n = 0
    with torch.no_grad():
        for f, seq in seqs:
            ids = encode(f, seq)
            logits = model(torch.tensor([ids], device=dev))[0]
            for j in range(2, len(ids)-1):
                gold = ids[j+1]; tk = torch.topk(logits[j], 5).indices.tolist(); n += 1
                t1 += gold in tk[:1]; t3 += gold in tk[:3]; t5 += gold in tk[:5]
    print(f"GPT next-step {label}:  Top1={t1/n:.3f}  Top3={t3/n:.3f}  Top5={t5/n:.3f}  (n={n})", flush=True)

model.eval()
id_seqs = [ex for f in id_families for ex in split[f][1]]
eval_split(id_seqs, "in-dist")

if ood_families:
    ood_seqs = [ex for f in ood_families for ex in split[f][1]]
    eval_split(ood_seqs, "OOD (IC)")

ckpt_name = "gpt_ckpt_ood.pt" if OOD else "gpt_ckpt.pt"
torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos, "ood": OOD}, ckpt_name)
print(f"saved {ckpt_name}", flush=True)
