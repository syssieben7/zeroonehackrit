# Semiconductor Process Sequence Predictor

Four models for learning and benchmarking semiconductor fab process sequences (next-step prediction, sequence completion, anomaly detection).

**Models:** `markov` · `lcm` · `hierarchical` · `transformer`

## Model checkpoints

| Model | Checkpoint location | Notes |
|---|---|---|
| `markov` | `models/markov/markov.json` | We used this as a baseline — transition counts from training data |
| `lcm` | `models/lcm/.save/best_100000_unfinished.pt` | GRU encoder-decoder with Bahdanau attention |
| `hierarchical` | `models/hierarchical/model_out/` | GPT-2 fine-tuned with block-boundary tokens |
| `transformer` | `models/transformer/gpt_ckpt.pt` | Decoder-only GPT trained from scratch |

Each model has its strengths and weaknesses
---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
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

Everything works from the terminal. Source into your virtual environment and run the following:

**Generate submission CSVs:**
```bash
python -m models.infer --model lcm --task 1 --output task1_predictions.csv
python -m models.infer --model lcm --task 2 --output task2_predictions.csv
python -m models.infer --model lcm --task 3 --output task3_predictions.csv
```

**Self-benchmark (scored locally against extended CSVs):**
```bash
python -m models.infer --model hierarchical --task self1
python -m models.infer --model hierarchical --task self2 --n-seqs 15
```

**Single sequence:**
```bash
python -m models.infer --model markov --next "RECEIVE WAFER LOT|LOT IDENTIFICATION|INITIAL WAFER INSPECTION"
python -m models.infer --model lcm --complete "RECEIVE WAFER LOT|LOT IDENTIFICATION"
```

**Score Task 3 locally** (Tasks 1 & 2 need the organiser's ground-truth file):
```bash
python data/participant_files/eval_metrics.py \
    --task anomaly \
    --ground-truth data/participant_files/eval_input_anomaly.csv \
    --predictions task3_predictions.csv
```
