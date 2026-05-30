import sys, random
from pathlib import Path
from collections import defaultdict, Counter

DATA = sys.argv[1]
sys.path.insert(0, DATA)
from generate_sequences import read_csv_sequences

families = ["MOSFET", "IGBT", "IC"]
data = {f: read_csv_sequences(Path(f"{DATA}/{f}_variants.csv")) for f in families}

random.seed(0)
train_seqs, test_seqs = [], []
for f in families:
    ids = list(data[f]); random.shuffle(ids)
    for i, sid in enumerate(ids):
        (test_seqs if i < 100 else train_seqs).append(data[f][sid])

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

t1 = t3 = t5 = tot = 0
for seq in test_seqs:
    for i in range(len(seq) - 1):
        preds = predict_topk(seq[:i + 1], 5)
        gold = seq[i + 1]; tot += 1
        t1 += gold in preds[:1]; t3 += gold in preds[:3]; t5 += gold in preds[:5]
print(f"n-gram(k<={K}) next-step baseline:  Top1={t1/tot:.3f}  Top3={t3/tot:.3f}  Top5={t5/tot:.3f}  (n={tot})")
