import sys, random
from pathlib import Path
from collections import defaultdict, Counter

DATA = sys.argv[1]
OOD = "--ood" in sys.argv
sys.path.insert(0, DATA)
from generate_sequences import read_csv_sequences

families = ["MOSFET", "IGBT", "IC"]
data = {f: read_csv_sequences(Path(f"{DATA}/{f}_variants.csv")) for f in families}

random.seed(0)
# split: first 100 per family → test, rest → train
split = {}  # family → (train_seqs, test_seqs)
for f in families:
    ids = list(data[f]); random.shuffle(ids)
    t, v = [], []
    for i, sid in enumerate(ids):
        (v if i < 100 else t).append(data[f][sid])
    split[f] = (t, v)

if OOD:
    train_families = ["MOSFET", "IGBT"]
    id_families    = ["MOSFET", "IGBT"]
    ood_families   = ["IC"]
    print("=== n-gram OOD mode: train on MOSFET+IGBT, test on IC ===", flush=True)
else:
    train_families = families
    id_families    = families
    ood_families   = []
    print("=== n-gram full in-dist mode ===", flush=True)

train_seqs = [seq for f in train_families for seq in split[f][0]]

K = 3
model = defaultdict(Counter)
glob = Counter()
for seq in train_seqs:
    for s in seq: glob[s] += 1
    for i in range(len(seq) - 1):
        for back in range(1, K + 1):
            if i - back + 1 < 0: break
            model[tuple(seq[i - back + 1:i + 1])][seq[i + 1]] += 1

def predict_topk(hist, k=5):
    for back in range(min(K, len(hist)), 0, -1):
        ctx = tuple(hist[-back:])
        if ctx in model:
            return [s for s, _ in model[ctx].most_common(k)]
    return [s for s, _ in glob.most_common(k)]

def eval_seqs(seqs, label):
    t1 = t3 = t5 = tot = 0
    for seq in seqs:
        for i in range(len(seq) - 1):
            preds = predict_topk(seq[:i + 1], 5)
            gold = seq[i + 1]; tot += 1
            t1 += gold in preds[:1]; t3 += gold in preds[:3]; t5 += gold in preds[:5]
    print(f"n-gram(k<={K}) {label}:  Top1={t1/tot:.3f}  Top3={t3/tot:.3f}  Top5={t5/tot:.3f}  (n={tot})", flush=True)

id_seqs  = [seq for f in id_families  for seq in split[f][1]]
eval_seqs(id_seqs, "in-dist")

if ood_families:
    ood_seqs = [seq for f in ood_families for seq in split[f][1]]
    eval_seqs(ood_seqs, "OOD (IC)")
