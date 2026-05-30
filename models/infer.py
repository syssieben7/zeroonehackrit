"""
models/infer.py — Unified inference interface for all four models.

Each adapter implements:
    predict_next(prefix, top_k)    → list[str]   (Task 1)
    predict_completion(prefix)     → list[str]   (Task 2)
    detect_anomaly(sequence)       → (is_valid: bool, score: float, rule: str)  (Task 3)

Batch eval helpers produce submission CSVs and return scored results.
"""

import csv
import io
import math
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

# ── Path helpers ──────────────────────────────────────────────────────────────

_ROOT       = Path(__file__).parent.parent
_SEQ2SEQ    = Path(__file__).parent / "seq2seq"
_MARKOV_DIR = Path(__file__).parent / "markov"
_HIER_DIR   = Path(__file__).parent / "hierarchical"
_TRANS_DIR  = Path(__file__).parent / "transformer"
_DATA_GEN   = _ROOT / "data" / "gen"

for _p in (_SEQ2SEQ, _MARKOV_DIR, _HIER_DIR, _TRANS_DIR, _DATA_GEN):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from validator import validate_sequence  # from seq2seq/


# ── Base class ────────────────────────────────────────────────────────────────

class ModelInterface(ABC):
    name: str

    @abstractmethod
    def predict_next(self, prefix: list[str], top_k: int = 5) -> list[str]:
        """Return top-k predicted next step strings."""

    @abstractmethod
    def predict_completion(self, prefix: list[str], max_steps: int = 200) -> list[str]:
        """Return predicted remaining steps from prefix."""

    def detect_anomaly(self, sequence: list[str]) -> tuple[bool, float, str]:
        """
        Returns (is_valid, score, rule_id).
        Default implementation: rule-based validator only.
        Override in subclasses that have a learned anomaly signal.
        """
        violations = validate_sequence(sequence)
        is_valid = len(violations) == 0
        score = 1.0 if is_valid else 0.0
        rule = violations[0] if violations else ""
        return is_valid, score, rule


# ── Markov adapter ────────────────────────────────────────────────────────────

class MarkovAdapter(ModelInterface):
    name = "markov"

    def __init__(self, markov_json: Path):
        from markov import MarkovChain
        self._mc = MarkovChain.load(str(markov_json))

    def predict_next(self, prefix: list[str], top_k: int = 5) -> list[str]:
        if not prefix:
            return []
        return [step for _, step in self._mc.predict_next(prefix[-1], top_k=top_k)]

    def predict_completion(self, prefix: list[str], max_steps: int = 200) -> list[str]:
        return self._mc.predict_completion(prefix, max_steps=max_steps)


# ── Seq2Seq adapter ───────────────────────────────────────────────────────────

class Seq2SeqAdapter(ModelInterface):
    name = "seq2seq"

    def __init__(self, checkpoint: Path):
        import torch
        from model import Encoder, Decoder, Seq2Seq
        from utils import PAD, SOS, EOS, IGBT_TOK, IC_TOK, Vocab

        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(checkpoint), map_location=self._dev, weights_only=False)
        args = ckpt["args"]

        vocab = Vocab.__new__(Vocab)
        vocab.itos = ckpt["vocab_itos"]
        vocab.stoi = ckpt["vocab_stoi"]
        self._vocab = vocab

        V = len(vocab)
        enc = Encoder(V, args["embed_size"], args["hidden_size"], n_layers=2, dropout=0.0)
        dec = Decoder(args["embed_size"], args["hidden_size"], V,
                      n_layers=args.get("dec_layers", 1), dropout=0.0)
        self._model = Seq2Seq(enc, dec, tie_embeddings=True).to(self._dev)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

        self._SOS = SOS
        self._EOS = EOS
        self._IGBT = IGBT_TOK
        self._IC   = IC_TOK

    def _family_tok(self, steps: list[str]) -> int:
        joined = " ".join(steps).upper()
        if any(k in joined for k in ("P BODY", "N BUFFER", "CHANNEL STOP",
                                     "EPITAXIAL WAFER CHECK", "FIELD OXIDE")):
            return self._IGBT
        if any(k in joined for k in ("SUBSTRATE CHECK", "EPITAXIAL DEPOSITION")):
            return self._IGBT  # mosfet shares IGBT tok in this vocab
        return self._IC

    def _encode(self, steps: list[str]):
        import torch
        ids = [self._family_tok(steps)] + self._vocab.encode(steps)
        return torch.tensor(ids, dtype=torch.long)

    def predict_next(self, prefix: list[str], top_k: int = 5) -> list[str]:
        import torch
        src = self._encode(prefix).unsqueeze(1).to(self._dev)
        with torch.no_grad():
            enc_out, enc_hid = self._model.encoder(src)
            hid = self._model.decoder.init_hidden(enc_hid)
            out, _, _ = self._model.decoder(
                torch.tensor([self._SOS], device=self._dev), hid, enc_out)
            probs = out.exp().squeeze(0)
        results = []
        for idx in probs.topk(top_k * 3).indices.tolist():
            if idx >= 6 and len(results) < top_k:
                results.append(self._vocab.itos[idx])
        return results

    def predict_completion(self, prefix: list[str], max_steps: int = 200) -> list[str]:
        import torch
        src = self._encode(prefix).unsqueeze(1).to(self._dev)
        with torch.no_grad():
            enc_out, enc_hid = self._model.encoder(src)
            hid = self._model.decoder.init_hidden(enc_hid)
            tok = torch.tensor([self._SOS], device=self._dev)
            result = []
            for _ in range(max_steps):
                out, hid, _ = self._model.decoder(tok, hid, enc_out)
                top1 = out.argmax(1)
                idx = top1.item()
                if idx == self._EOS:
                    break
                if idx >= 6:
                    result.append(self._vocab.itos[idx])
                tok = top1
        return result


# ── Hierarchical (GPT-2 + block tokens) adapter ───────────────────────────────

class HierarchicalAdapter(ModelInterface):
    name = "hierarchical"

    def __init__(self, model_dir: Path):
        import json
        import torch
        from transformers import GPT2LMHeadModel

        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._vocab  = json.loads((model_dir / "vocab.json").read_text())
        self._id2tok = {v: k for k, v in self._vocab.items()}
        self._model  = GPT2LMHeadModel.from_pretrained(str(model_dir)).to(self._dev).eval()
        self._BOS = "<bos>"
        self._EOS = "<eos>"
        self._PAD = "<pad>"
        # read max context from the saved config (n_positions=256 for this checkpoint)
        cfg = json.loads((model_dir / "config.json").read_text())
        self._max_len = cfg.get("n_positions", 256)

    def _ids(self, prefix: list[str]) -> list[int]:
        ids = ([self._vocab[self._BOS]] +
               [self._vocab.get(t, self._vocab[self._PAD]) for t in prefix])
        # keep the most recent context when prefix exceeds the trained context window
        return ids[-self._max_len:]

    def predict_next(self, prefix: list[str], top_k: int = 5) -> list[str]:
        import torch
        x = torch.tensor([self._ids(prefix)], dtype=torch.long, device=self._dev)
        logits = self._model(x).logits[0, -1]
        for tok, idx in self._vocab.items():
            if tok.startswith("[BLK:") or tok in (self._PAD, self._BOS, self._EOS):
                logits[idx] = float("-inf")
        return [self._id2tok[i] for i in logits.topk(top_k).indices.tolist()]

    def predict_completion(self, prefix: list[str], max_steps: int = 200) -> list[str]:
        import torch
        eos_id = self._vocab[self._EOS]
        ids = self._ids(prefix)
        result = []
        with torch.no_grad():
            for _ in range(max_steps):
                # slide the window if we hit the context limit
                x = torch.tensor([ids[-self._max_len:]], dtype=torch.long, device=self._dev)
                nxt = int(self._model(x).logits[0, -1].argmax())
                if nxt == eos_id:
                    break
                tok = self._id2tok[nxt]
                if not tok.startswith("[BLK:"):
                    result.append(tok)
                ids.append(nxt)
        return result


# ── Transformer (custom GPT from scratch) adapter ────────────────────────────

class TransformerAdapter(ModelInterface):
    name = "transformer"

    _T = 160

    def __init__(self, checkpoint: Path):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

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
            def __init__(s, V, T, d=256, h=8, L=6):
                super().__init__()
                s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(T, d)
                s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
                s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, V)
            def forward(s, idx):
                B, L = idx.shape
                x = s.tok(idx) + s.pos(torch.arange(L, device=idx.device))
                for b in s.blocks: x = b(x)
                return s.head(s.lnf(x))

        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._F = F
        ckpt = torch.load(str(checkpoint), map_location=self._dev, weights_only=False)
        self._stoi = ckpt["stoi"]
        self._itos = ckpt["itos"]
        V = len(self._itos)
        self._model = GPT(V, self._T).to(self._dev)
        self._model.load_state_dict(ckpt["model"])
        self._model.eval()
        self._PAD_ID  = self._stoi["<PAD>"]
        self._BOS_ID  = self._stoi["<BOS>"]
        self._SHIP_ID = self._stoi.get("SHIP LOT")
        self._torch = torch

    def _encode(self, family: str, steps: list[str]) -> list[int]:
        fam_id = self._stoi.get(f"<FAM_{family.upper()}>")
        ids = [self._BOS_ID, fam_id] if fam_id is not None else [self._BOS_ID]
        for s in steps:
            if s in self._stoi:
                ids.append(self._stoi[s])
        return ids[:self._T]

    @staticmethod
    def _detect_family(steps: list[str]) -> str:
        joined = " ".join(steps).upper()
        if any(k in joined for k in ("P BODY", "N BUFFER", "CHANNEL STOP",
                                     "EPITAXIAL WAFER CHECK", "FIELD OXIDE")):
            return "IGBT"
        if any(k in joined for k in ("SUBSTRATE CHECK", "EPITAXIAL DEPOSITION")):
            return "MOSFET"
        return "IC"

    def predict_next(self, prefix: list[str], top_k: int = 5) -> list[str]:
        fam = self._detect_family(prefix)
        ids = self._encode(fam, prefix)
        x = self._torch.tensor([ids], dtype=self._torch.long, device=self._dev)
        with self._torch.no_grad():
            lg = self._model(x)[0, len(ids) - 1]
        return [self._itos[i] for i in self._torch.topk(lg, top_k).indices.tolist()]

    def predict_completion(self, prefix: list[str], max_steps: int = 200) -> list[str]:
        fam = self._detect_family(prefix)
        ids = self._encode(fam, prefix)
        result = []
        with self._torch.no_grad():
            for _ in range(max_steps):
                if len(ids) >= self._T:
                    break
                x = self._torch.tensor([ids], dtype=self._torch.long, device=self._dev)
                nxt = int(self._torch.argmax(self._model(x)[0, len(ids) - 1]))
                if nxt == self._PAD_ID:
                    break
                ids.append(nxt)
                tok = self._itos[nxt]
                result.append(tok)
                if nxt == self._SHIP_ID:
                    break
        return result

    def detect_anomaly(self, sequence: list[str]) -> tuple[bool, float, str]:
        """GPT surprisal score combined with rule checker."""
        violations = validate_sequence(sequence)
        is_valid = len(violations) == 0
        rule = violations[0] if violations else ""

        fam = self._detect_family(sequence)
        ids = self._encode(fam, sequence)
        if len(ids) >= 3:
            x = self._torch.tensor([ids], dtype=self._torch.long, device=self._dev)
            with self._torch.no_grad():
                logits = self._model(x[:, :-1])[0]
            targets = self._torch.tensor(ids[1:], dtype=self._torch.long, device=self._dev)
            nll = self._F.cross_entropy(logits[2:], targets[2:], reduction="mean").item()
        else:
            nll = 10.0

        score = math.exp(-nll)
        if not is_valid:
            score = min(score, 0.45)
        return is_valid, round(score, 4), rule


# ── Factory ───────────────────────────────────────────────────────────────────

_DEFAULT_PATHS = {
    "markov":       _MARKOV_DIR / "markov.json",
    "seq2seq":      _SEQ2SEQ / ".save" / "best.pt",
    "hierarchical": _HIER_DIR / "model_out",
    "transformer":  _TRANS_DIR / "gpt_ckpt.pt",
}

_ADAPTER_CLS = {
    "markov":      MarkovAdapter,
    "seq2seq":     Seq2SeqAdapter,
    "hierarchical": HierarchicalAdapter,
    "transformer": TransformerAdapter,
}


def load_model(name: str, path: Path = None) -> ModelInterface:
    p = path or _DEFAULT_PATHS[name]
    return _ADAPTER_CLS[name](p)


def available_models() -> list[str]:
    """Return names of models whose checkpoint/directory exists."""
    return [name for name, path in _DEFAULT_PATHS.items() if path.exists()]


def missing_models() -> dict[str, str]:
    """Return {name: expected_path} for models whose checkpoint is absent."""
    return {name: str(path) for name, path in _DEFAULT_PATHS.items()
            if not path.exists()}


# ── Batch eval helpers ────────────────────────────────────────────────────────

def _read_valid_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_anomaly_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def run_task1(model: ModelInterface, eval_valid_csv: Path,
              progress_cb=None) -> tuple[str, str]:
    """
    Task 1 — Next-step prediction.
    Returns (predictions_csv_str, scores_str).
    """
    rows = _read_valid_csv(eval_valid_csv)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])

    for i, row in enumerate(rows):
        steps = [s.strip() for s in row["PARTIAL_SEQUENCE"].split("|") if s.strip()]
        top5 = model.predict_next(steps, top_k=5)
        # pad to 5 if fewer returned
        top5 = (top5 + [""] * 5)[:5]
        w.writerow([row["EXAMPLE_ID"]] + top5)
        if progress_cb and (i + 1) % 50 == 0:
            progress_cb(i + 1, len(rows))

    pred_csv = buf.getvalue()
    msg = (
        f"Predictions generated for {len(rows)} examples.\n"
        "eval_input_valid.csv contains no ground-truth labels — submit the CSV "
        "to the organizers for official scoring.\n"
        "Use 'Self-Benchmark' to score against your own held-out split."
    )
    return pred_csv, msg


def run_task2(model: ModelInterface, eval_valid_csv: Path,
              progress_cb=None) -> tuple[str, str]:
    """
    Task 2 — Sequence completion.
    Returns (predictions_csv_str, scores_str).
    """
    rows = _read_valid_csv(eval_valid_csv)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])

    for i, row in enumerate(rows):
        steps = [s.strip() for s in row["PARTIAL_SEQUENCE"].split("|") if s.strip()]
        completion = model.predict_completion(steps)
        w.writerow([row["EXAMPLE_ID"], "|".join(completion)])
        if progress_cb and (i + 1) % 50 == 0:
            progress_cb(i + 1, len(rows))

    pred_csv = buf.getvalue()
    msg = (
        f"Predictions generated for {len(rows)} examples.\n"
        "eval_input_valid.csv contains no ground-truth labels — submit the CSV "
        "to the organizers for official scoring.\n"
        "Use 'Self-Benchmark' to score against your own held-out split."
    )
    return pred_csv, msg


def run_task3(model: ModelInterface, eval_anomaly_csv: Path,
              progress_cb=None) -> tuple[str, str]:
    """
    Task 3 — Anomaly detection.
    Returns (predictions_csv_str, scores_str).
    """
    rows = _read_anomaly_csv(eval_anomaly_csv)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])

    for i, row in enumerate(rows):
        steps = [s.strip() for s in row["SEQUENCE"].split("|") if s.strip()]
        is_valid, score, rule = model.detect_anomaly(steps)
        w.writerow([row["EXAMPLE_ID"], int(is_valid), score, rule])
        if progress_cb and (i + 1) % 100 == 0:
            progress_cb(i + 1, len(rows))

    pred_csv = buf.getvalue()
    scores = _score_task3(rows, pred_csv)
    return pred_csv, scores


# ── Inline scoring (mirrors eval_metrics.py without file I/O) ─────────────────

def _safe_div(n, d):
    return n / d if d else float("nan")


def _score_task1(gt_rows: list[dict], pred_csv: str) -> str:
    pred = {}
    for row in csv.DictReader(io.StringIO(pred_csv)):
        eid = row["EXAMPLE_ID"]
        pred[eid] = [row.get(f"RANK_{i}", "").strip() for i in range(1, 6)]

    top1 = top3 = top5 = total = 0
    rr = 0.0
    by_fam = {}
    by_frac = {}

    for row in gt_rows:
        eid    = row["EXAMPLE_ID"]
        truth  = row.get("NEXT_STEP", "").strip()
        fam    = row.get("FAMILY", "ALL")
        frac   = row.get("COMPLETION_FRACTION", "ALL")
        ranks  = pred.get(eid, [])

        # For eval_input_valid.csv the ground truth next step isn't included —
        # we can only score if we also have the full sequence. Skip MRR if absent.
        if not truth:
            continue

        h1 = bool(ranks) and ranks[0] == truth
        h3 = truth in ranks[:3]
        h5 = truth in ranks[:5]
        r  = 1.0 / (ranks.index(truth) + 1) if truth in ranks else 0.0

        top1 += h1; top3 += h3; top5 += h5; rr += r; total += 1
        for key, d in [(fam, by_fam), (frac, by_frac)]:
            d.setdefault(key, [0, 0, 0, 0.0, 0])
            d[key][0] += h1; d[key][1] += h3; d[key][2] += h5
            d[key][3] += r;  d[key][4] += 1

    if total == 0:
        return "Task 1 scored: 0 examples with NEXT_STEP ground truth in eval file.\n(eval_input_valid.csv does not include ground-truth next steps — use a labelled ground-truth file to get scores.)"

    lines = [
        "=" * 55,
        "TASK 1 — NEXT-STEP PREDICTION",
        "=" * 55,
        f"Evaluated: {total} examples",
        f"Top-1 Accuracy : {_safe_div(top1,total):.4f}  ({top1}/{total})",
        f"Top-3 Accuracy : {_safe_div(top3,total):.4f}  ({top3}/{total})",
        f"Top-5 Accuracy : {_safe_div(top5,total):.4f}  ({top5}/{total})",
        f"MRR            : {_safe_div(rr,total):.4f}",
    ]
    if by_fam:
        lines.append("\nBy family:")
        for fam, (t1, t3, t5, r, n) in sorted(by_fam.items()):
            lines.append(f"  {fam:<8}  Top-1:{_safe_div(t1,n):.3f}  Top-3:{_safe_div(t3,n):.3f}  Top-5:{_safe_div(t5,n):.3f}  MRR:{_safe_div(r,n):.3f}")
    return "\n".join(lines)


def _levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], (prev if a[i-1] == b[j-1]
                                   else 1 + min(prev, dp[j], dp[j-1]))
    return dp[n]


def _major_block(step: str) -> str:
    s = step.upper()
    if "LITHO" in s or s.startswith("SPIN COAT PHOTORESIST") or "MASK LEVEL" in s: return "LITHO"
    if "ETCH" in s or s.startswith("OPEN PAD WINDOW"): return "ETCH"
    if "IMPLANT" in s or "ANNEAL" in s or "DIFFUSION" in s: return "DOPING_THERMAL"
    if s.startswith("DEPOSIT") or "OXIDATION" in s or "GROWTH" in s: return "DEPOSITION"
    if s.startswith("CMP") or "PLANAR" in s: return "PLANARIZATION"
    if "VIA" in s: return "VIA"
    if "PASSIVATION" in s: return "PASSIVATION"
    if "BACKSIDE" in s or "GRIND" in s: return "BACKSIDE"
    if "TEST" in s or "MEASURE" in s or "INSPECT" in s: return "METROLOGY_TEST"
    if "LOT" in s or "RELEASE" in s or "SHIP" in s: return "LOGISTICS"
    return "OTHER"


def _block_sig(seq):
    sig, prev = [], None
    for s in seq:
        b = _major_block(s)
        if b != prev:
            sig.append(b); prev = b
    return sig


def _score_task2(gt_rows: list[dict], pred_csv: str) -> str:
    pred = {}
    for row in csv.DictReader(io.StringIO(pred_csv)):
        eid = row["EXAMPLE_ID"]
        seq_str = row.get("PREDICTED_SEQUENCE", "")
        pred[eid] = [s for s in seq_str.split("|") if s]

    ned_all = []; exact_all = []; tacc_all = []; bacc_all = []
    by_fam = {}; by_frac = {}; total = 0

    for row in gt_rows:
        eid      = row["EXAMPLE_ID"]
        partial  = [s.strip() for s in row["PARTIAL_SEQUENCE"].split("|") if s.strip()]
        full_str = row.get("FULL_SEQUENCE", "")
        if not full_str:
            continue
        full     = [s.strip() for s in full_str.split("|") if s.strip()]
        ref      = full[len(partial):]
        predicted = pred.get(eid, [])
        fam      = row.get("FAMILY", "ALL")
        frac     = row.get("COMPLETION_FRACTION", "ALL")

        ned   = (_levenshtein(predicted, ref) / max(len(predicted), len(ref), 1))
        exact = predicted == ref
        n     = min(len(predicted), len(ref))
        tacc  = (sum(p == r for p, r in zip(predicted, ref)) / n) if n else 0.0
        bacc  = (sum(p == r for p, r in zip(_block_sig(predicted), _block_sig(ref)))
                 / max(len(_block_sig(ref)), 1))

        ned_all.append(ned); exact_all.append(exact)
        tacc_all.append(tacc); bacc_all.append(bacc); total += 1

        for key, d in [(fam, by_fam), (frac, by_frac)]:
            d.setdefault(key, {"ned": [], "exact": []})
            d[key]["ned"].append(ned); d[key]["exact"].append(exact)

    if total == 0:
        return "Task 2 scored: 0 examples — FULL_SEQUENCE column not present in eval file.\n(Use a labelled ground-truth file to get completion scores.)"

    def mean(lst): return sum(lst) / len(lst) if lst else float("nan")

    lines = [
        "=" * 55,
        "TASK 2 — SEQUENCE COMPLETION",
        "=" * 55,
        f"Evaluated: {total} examples",
        f"Mean NED (lower=better) : {mean(ned_all):.4f}",
        f"Exact Match Rate        : {mean(exact_all):.4f}  ({sum(exact_all)}/{total})",
        f"Mean Token Accuracy     : {mean(tacc_all):.4f}",
        f"Mean Block Accuracy     : {mean(bacc_all):.4f}",
    ]
    if by_fam:
        lines.append("\nBy family:")
        for fam, d in sorted(by_fam.items()):
            lines.append(f"  {fam:<8}  NED:{mean(d['ned']):.3f}  Exact:{mean(d['exact']):.3f}")
    if by_frac:
        lines.append("\nBy completion fraction:")
        for frac, d in sorted(by_frac.items()):
            lines.append(f"  {frac:<6}  NED:{mean(d['ned']):.3f}  Exact:{mean(d['exact']):.3f}")
    return "\n".join(lines)


def _score_task3(gt_rows: list[dict], pred_csv: str) -> str:
    """
    gt_rows from eval_input_anomaly.csv have no IS_VALID label (it's blind).
    We score using rule-checker ground truth on the sequences themselves.
    """
    pred = {}
    for row in csv.DictReader(io.StringIO(pred_csv)):
        eid = row["EXAMPLE_ID"]
        try:
            iv = int(float(row.get("IS_VALID", "1")))
        except (ValueError, TypeError):
            iv = 1
        try:
            sc = float(row.get("SCORE", "1.0"))
        except (ValueError, TypeError):
            sc = float(iv)
        pred[eid] = {"is_valid": iv, "score": sc,
                     "rule": row.get("PREDICTED_RULE", "").strip()}

    # derive ground truth from the sequences themselves via rule checker
    labels = []; scores = []; preds_bin = []
    rule_gt_list = []; rule_pred_list = []

    for row in gt_rows:
        eid   = row["EXAMPLE_ID"]
        steps = [s.strip() for s in row["SEQUENCE"].split("|") if s.strip()]
        viols = validate_sequence(steps)
        gt_valid = 1 if not viols else 0
        gt_rule  = viols[0] if viols else ""

        p = pred.get(eid, {"is_valid": 1, "score": 1.0, "rule": ""})
        labels.append(gt_valid)
        scores.append(p["score"])
        preds_bin.append(p["is_valid"])
        if gt_rule:
            rule_gt_list.append(gt_rule)
            rule_pred_list.append(p["rule"])

    n = len(labels)
    if n == 0:
        return "Task 3: no examples to score."

    accuracy = sum(p == l for p, l in zip(preds_bin, labels)) / n
    tp = sum((l == 0) and (p == 0) for l, p in zip(labels, preds_bin))
    tn = sum((l == 1) and (p == 1) for l, p in zip(labels, preds_bin))
    fp = sum((l == 1) and (p == 0) for l, p in zip(labels, preds_bin))
    fn = sum((l == 0) and (p != 0) for l, p in zip(labels, preds_bin))
    prec = _safe_div(tp, tp + fp)
    rec  = _safe_div(tp, tp + fn)
    f1   = _safe_div(2 * prec * rec, prec + rec)

    # ROC-AUC
    pos_sc = [s for s, l in zip(scores, labels) if l == 1]
    neg_sc = [s for s, l in zip(scores, labels) if l == 0]
    if pos_sc and neg_sc:
        total_pairs = len(pos_sc) * len(neg_sc)
        conc = sum(p > n for p in pos_sc for n in neg_sc)
        tied = sum(p == n for p in pos_sc for n in neg_sc)
        auc  = (conc + 0.5 * tied) / total_pairs
        auc_str = f"{auc:.4f}"
    else:
        auc_str = "n/a (single class)"

    rule_attr_str = "n/a"
    if rule_pred_list:
        correct_det = [(rg, rp) for rg, rp, lb, pb
                       in zip(rule_gt_list, rule_pred_list, labels[:len(rule_gt_list)], preds_bin[:len(rule_gt_list)])
                       if lb == 0 and pb == 0]
        if correct_det:
            rule_attr = sum(rg == rp for rg, rp in correct_det) / len(correct_det)
            rule_attr_str = f"{rule_attr:.4f}  ({sum(rg==rp for rg,rp in correct_det)}/{len(correct_det)})"

    n_invalid = sum(l == 0 for l in labels)
    n_valid   = sum(l == 1 for l in labels)

    lines = [
        "=" * 55,
        "TASK 3 — ANOMALY DETECTION",
        "=" * 55,
        f"Evaluated       : {n} examples  ({n_invalid} invalid / {n_valid} valid)",
        f"Binary Accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)",
        f"Precision       : {prec:.4f}",
        f"Recall          : {rec:.4f}",
        f"F1 Score        : {f1:.4f}",
        f"ROC-AUC         : {auc_str}",
        f"Rule Attribution: {rule_attr_str}",
        f"Confusion Matrix:",
        f"  TP={tp}  FP={fp}",
        f"  FN={fn}  TN={tn}",
    ]
    return "\n".join(lines)


# ── Self-benchmark (scored against variant CSVs) ──────────────────────────────

_EXTENDED_CANDIDATES = [
    # preferred: extended files are independent of variant-trained models
    _ROOT / "data" / "raw" / "MOSFET" / "MOSFET_extended.csv",
    _ROOT / "data" / "raw" / "IGBT"   / "IGBT_extended.csv",
    _ROOT / "data" / "raw" / "IC"     / "IC_extended.csv",
    # old layout
    _ROOT / "data" / "raw" / "infineon" / "MOSFET" / "MOSFET_extended.csv",
    _ROOT / "data" / "raw" / "infineon" / "IGBT"   / "IGBT_extended.csv",
    _ROOT / "data" / "raw" / "infineon" / "IC"     / "IC_extended.csv",
]

_VARIANT_FALLBACK = [
    _ROOT / "data" / "raw" / "MOSFET" / "MOSFET_variants.csv",
    _ROOT / "data" / "raw" / "IGBT"   / "IGBT_variants.csv",
    _ROOT / "data" / "raw" / "IC"     / "IC_variants.csv",
    _ROOT / "data" / "raw" / "infineon" / "MOSFET" / "MOSFET_variants.csv",
    _ROOT / "data" / "raw" / "infineon" / "IGBT"   / "IGBT_variants.csv",
    _ROOT / "data" / "raw" / "infineon" / "IC"     / "IC_variants.csv",
]


def _load_benchmark_seqs(max_per_family: int = 150,
                         seed: int = 42) -> tuple[list[tuple[str, list[str]]], str]:
    """
    Load sequences for self-benchmarking.
    Prefers *_extended.csv (unseen by variant-trained models) and falls back
    to *_variants.csv. Returns (sequences, source_label).
    """
    import random
    rng = random.Random(seed)
    seqs_out = []
    seen_families: set[str] = set()
    source_label = "extended"

    candidates = _EXTENDED_CANDIDATES
    if not any(p.exists() for p in candidates):
        candidates = _VARIANT_FALLBACK
        source_label = "variants (extended not found)"

    seen_paths: set[str] = set()
    for path in candidates:
        if not path.exists() or str(path) in seen_paths:
            continue
        seen_paths.add(str(path))
        stem   = path.stem  # e.g. "MOSFET_extended"
        family = stem.split("_")[0].upper()
        if family in seen_families:
            continue
        seen_families.add(family)

        seqs: dict[str, list[str]] = {}
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                sid  = row.get("SEQUENCE_ID", "").strip()
                step = row.get("STEP", "").strip()
                if sid and step:
                    seqs.setdefault(sid, []).append(step)

        ids = list(seqs.keys())
        rng.shuffle(ids)
        for sid in ids[:max_per_family]:
            seqs_out.append((family, seqs[sid]))

    return seqs_out, source_label


def self_benchmark_task1(model: ModelInterface,
                         progress_cb=None,
                         max_per_family: int = 50) -> str:
    """
    Evaluate next-step accuracy on extended sequences (unseen by variant-trained models).
    Samples every 5th position between 10% and 90% of each sequence.
    """
    test_seqs, source = _load_benchmark_seqs(max_per_family=max_per_family)
    if not test_seqs:
        return "No extended or variant CSVs found. Expected at data/raw/MOSFET|IGBT|IC/*_extended.csv"

    top1 = top3 = top5 = total = 0
    rr = 0.0
    by_fam: dict[str, list] = {}

    for i, (family, steps) in enumerate(test_seqs):
        n = len(steps)
        positions = range(max(1, n // 10), int(n * 0.9), max(1, n // 20))
        for pos in positions:
            prefix    = steps[:pos]
            true_next = steps[pos]
            preds     = model.predict_next(prefix, top_k=5)

            h1 = bool(preds) and preds[0] == true_next
            h3 = true_next in preds[:3]
            h5 = true_next in preds[:5]
            r  = 1.0 / (preds.index(true_next) + 1) if true_next in preds else 0.0

            top1 += h1; top3 += h3; top5 += h5; rr += r; total += 1
            by_fam.setdefault(family, [0, 0, 0, 0.0, 0])
            d = by_fam[family]
            d[0] += h1; d[1] += h3; d[2] += h5; d[3] += r; d[4] += 1

        if progress_cb and (i + 1) % 20 == 0:
            progress_cb(i + 1, len(test_seqs))

    if total == 0:
        return "No evaluation positions found."

    lines = [
        "=" * 55,
        "SELF-BENCHMARK — TASK 1: NEXT-STEP PREDICTION",
        f"(source: {source}, {len(test_seqs)} sequences, {total} positions)",
        "=" * 55,
        f"Top-1 Accuracy : {_safe_div(top1,total):.4f}  ({top1}/{total})",
        f"Top-3 Accuracy : {_safe_div(top3,total):.4f}",
        f"Top-5 Accuracy : {_safe_div(top5,total):.4f}",
        f"MRR            : {_safe_div(rr,total):.4f}",
        "\nBy family:",
    ]
    for fam, (t1, t3, t5, r, n) in sorted(by_fam.items()):
        lines.append(
            f"  {fam:<8}  Top-1:{_safe_div(t1,n):.3f}  "
            f"Top-3:{_safe_div(t3,n):.3f}  Top-5:{_safe_div(t5,n):.3f}  "
            f"MRR:{_safe_div(r,n):.3f}  (n={n})"
        )
    return "\n".join(lines)


def self_benchmark_task2(model: ModelInterface,
                         progress_cb=None,
                         max_per_family: int = 15) -> str:
    """
    Evaluate sequence completion on extended sequences at 60% and 80% cut points.
    """
    test_seqs, source = _load_benchmark_seqs(max_per_family=max_per_family)
    if not test_seqs:
        return "No extended or variant CSVs found. Expected at data/raw/MOSFET|IGBT|IC/*_extended.csv"

    ned_all = []; exact_all = []; tacc_all = []; bacc_all = []
    by_frac: dict[str, dict] = {}

    for i, (family, steps) in enumerate(test_seqs):
        n = len(steps)
        for frac in (0.6, 0.8):
            cut       = int(n * frac)
            prefix    = steps[:cut]
            ref       = steps[cut:]
            if not ref:
                continue
            predicted = model.predict_completion(prefix)

            ned   = _levenshtein(predicted, ref) / max(len(predicted), len(ref), 1)
            exact = predicted == ref
            m     = min(len(predicted), len(ref))
            tacc  = sum(p == r for p, r in zip(predicted, ref)) / m if m else 0.0
            bacc  = (sum(p == r for p, r in zip(_block_sig(predicted), _block_sig(ref)))
                     / max(len(_block_sig(ref)), 1))

            ned_all.append(ned); exact_all.append(exact)
            tacc_all.append(tacc); bacc_all.append(bacc)
            key = f"{int(frac*100)}%"
            by_frac.setdefault(key, {"ned": [], "exact": [], "tacc": [], "bacc": []})
            by_frac[key]["ned"].append(ned);   by_frac[key]["exact"].append(exact)
            by_frac[key]["tacc"].append(tacc); by_frac[key]["bacc"].append(bacc)

        if progress_cb and (i + 1) % 10 == 0:
            progress_cb(i + 1, len(test_seqs))

    if not ned_all:
        return "No evaluation examples found."

    def mean(lst): return sum(lst) / len(lst) if lst else float("nan")

    lines = [
        "=" * 55,
        "SELF-BENCHMARK — TASK 2: SEQUENCE COMPLETION",
        f"(source: {source}, {len(test_seqs)} sequences)",
        "=" * 55,
        f"Mean NED (lower=better) : {mean(ned_all):.4f}",
        f"Exact Match Rate        : {mean(exact_all):.4f}  ({sum(exact_all)}/{len(exact_all)})",
        f"Mean Token Accuracy     : {mean(tacc_all):.4f}",
        f"Mean Block Accuracy     : {mean(bacc_all):.4f}",
        "\nBy cut point:",
    ]
    for frac, d in sorted(by_frac.items()):
        lines.append(
            f"  {frac}  NED:{mean(d['ned']):.3f}  Exact:{mean(d['exact']):.3f}  "
            f"TokAcc:{mean(d['tacc']):.3f}  BlkAcc:{mean(d['bacc']):.3f}"
        )
    return "\n".join(lines)
