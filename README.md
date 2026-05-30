# Semiconductor Process Sequence Predictor

Four models for learning and benchmarking semiconductor fab process sequences (next-step prediction, sequence completion, anomaly detection).

**Models:** `markov` · `seq2seq` · `hierarchical` · `transformer`

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate      # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
python app.py
# → open http://127.0.0.1:7860
```

## Interface

The app has three tabs:

**Self-Benchmark** — score any model against held-out `*_extended.csv` sequences (no training-data leakage). Pick Task 1 (next-step) or Task 2 (completion), set sequences per family, hit Run. Results include Top-1/3/5, MRR, NED, Exact Match etc. Download the per-row CSV for further analysis.

**Official** — generate submission CSVs for the three hackathon tasks using `eval_input_valid.csv` and `eval_input_anomaly.csv`. Task 3 (anomaly) scores immediately via the rule checker; Tasks 1 & 2 need the organiser's ground-truth file to score.

**Custom** — interactive inference. Paste or upload a partial sequence, pick a model, and either predict the next step, complete the full sequence, or detect rule violations.

---

## I don't like GUIs...

`models/infer.py` is fully usable as a library. Run from the repo root:

```python
from models.infer import load_model, run_task1, run_task2, run_task3
from pathlib import Path

model = load_model("seq2seq")  # markov | hierarchical | transformer

VALID   = Path("data/participant_files/eval_input_valid.csv")
ANOMALY = Path("data/participant_files/eval_input_anomaly.csv")

pred1, _ = run_task1(model, VALID)
pred2, _ = run_task2(model, VALID)
pred3, _ = run_task3(model, ANOMALY)

Path("task1_predictions.csv").write_text(pred1)
Path("task2_predictions.csv").write_text(pred2)
Path("task3_predictions.csv").write_text(pred3)
```

Score Task 3 locally (Tasks 1 & 2 need the organiser's ground-truth file):

```bash
python data/participant_files/eval_metrics.py \
    --task anomaly \
    --ground-truth data/participant_files/eval_input_anomaly.csv \
    --predictions task3_predictions.csv
```

---

## Model checkpoints

| Model | Checkpoint location | Notes |
|---|---|---|
| `markov` | `models/markov/markov.json` | No training needed — transition counts from training data |
| `seq2seq` | `models/seq2seq/.save/best_100000_unfinished.pt` | GRU encoder-decoder with Bahdanau attention |
| `hierarchical` | `models/hierarchical/model_out/` | GPT-2 fine-tuned with block-boundary tokens |
| `transformer` | `models/transformer/gpt_ckpt.pt` | Decoder-only GPT trained from scratch |
