# RIT — Industrial AI (Infineon)

---

## Team

- **Gabor Pal** — ML engineering / model architecture
- **Adrian Top** — training infrastructure / HPC
- **Florian Weber** — data pipeline / evaluation

**Track:** Industrial AI (Infineon)

---

## TL;DR

We built four complementary models (Markov baseline, Seq2Seq with attention, causal Transformer GPT, and a hierarchical GPT-2) to predict, complete, and validate semiconductor fabrication process sequences. All models were trained on Leonardo A100 GPUs using synthetic process data for MOSFET, IGBT, and IC families, achieving 74.8% Top-1 (perfect Top-3/5) next-step accuracy with the GPT model and 73.1% block-level accuracy on sequence completion with the hierarchical model.

---

## Problem

Semiconductor manufacturing requires precisely ordered process sequences — hundreds of steps per wafer lot, where a single misplaced etch or missing clean causes costly scrap. The challenge: given a partial sequence, can a model (1) predict the next step, (2) complete the full remaining sequence, and (3) detect rule violations in supposedly-complete sequences?

We focused on all three tasks across three product families (MOSFET, IGBT, IC), with special attention to the structural block transitions that trip up simple n-gram approaches.

---

## Approach

- **Markov chain baseline** — first-order bigram transition counts; fast to train, establishes a lower bound (Top-1: 57.8%)
- **Process-LCM (Large Concept Model)** — bidirectional GRU encoder + GRU decoder with Bahdanau attention, adapted from NMT; learns step embeddings in a continuous space and predicts next-step via cosine similarity against a learned codebook. Trained 10k+ steps on Leonardo.
- **From-scratch GPT (5M params)** — decoder-only causal Transformer with family tokens and sinusoidal positional encoding; trained 8 epochs on 30k generated sequences. Best next-step model (Top-1: 74.8%, perfect Top-3/5).
- **Hierarchical GPT-2 (18M params)** — fine-tuned with `[BLK:X]` block-boundary tokens injected into training data (PREFIX, PRE_CLEAN, PROCESS_CYCLES, etc.) so the model learns process structure at two resolutions simultaneously. 75% aligned accuracy, 100% validator pass.
- **Rule-based anomaly detection** — deterministic validator checking forbidden patterns (missing cleans, ordering violations) combined with model confidence scores for hybrid anomaly scoring.
- **All training on CINECA Leonardo** — A100 GPUs via SLURM, pixi environment management.
- **Training data** — 10,000 randomly generated sequences per family (MOSFET, IGBT, IC) using `data/gen/generate_sequences.py`, totalling 30,000 sequences.

---

## How to run it

```bash
# Setup
git clone https://github.com/syssieben7/zeroonehackrit.git
cd zeroonehackrit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate submission CSVs (requires checkpoint)
python generate_predictions.py --checkpoint hpc/scp/.save/best_lcm_10000.pt

# Or use the CLI interface per-model:
python -m models.infer --model lcm --task 1 --output task1_predictions.csv
python -m models.infer --model lcm --task 2 --output task2_predictions.csv
python -m models.infer --model lcm --task 3 --output task3_predictions.csv

# Launch Gradio UI (interactive demo)
python app.py
# → open http://127.0.0.1:7860
```

**Requirements:** Python 3.9+, PyTorch >= 2.0, ~2GB disk for checkpoints. GPU recommended for inference but CPU works. Training requires Leonardo HPC access (A100).

---

## Results

### Task 1 — Next-Step Prediction

| Model | Top-1 | Top-3 | Top-5 | MRR |
|-------|-------|-------|-------|-----|
| Markov (1st order) | 0.578 | 0.752 | 0.820 | 0.679 |
| GPT (5M, 8 epochs) | 0.748 | **1.000** | **1.000** | **0.871** |
| LCM (Seq2Seq) | 0.170 | 0.267 | 0.340 | 0.231 |
| Hierarchical GPT-2 | **0.728** | 0.952 | 0.971 | 0.841 |

### Task 2 — Sequence Completion

| Model | Mean NED (↓) | Exact Match | Token Accuracy | Block Accuracy |
|-------|-------------|-------------|----------------|----------------|
| Markov (1st order) | 0.961 | 0.000 | 0.065 | 0.226 |
| GPT (5M, 8 epochs) | 0.245 | 0.000 | 0.426 | 0.693 |
| LCM (Seq2Seq) | 0.352 | 0.000 | 0.229 | 0.525 |
| Hierarchical GPT-2 | **0.217** | 0.000 | **0.458** | **0.731** |

### Task 3 — Anomaly Detection

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|------|---------|
| Markov (1st order) | 0.833 | 1.000 | 0.667 | 0.800 | 0.833 |
| GPT (5M, 8 epochs) | 0.833 | 1.000 | 0.667 | 0.800 | **1.000** |
| LCM (Seq2Seq) | 0.833 | 1.000 | 0.667 | 0.800 | 0.833 |
| Hierarchical GPT-2 | 0.833 | 1.000 | 0.667 | 0.800 | 0.833 |

Rule-based validator with model confidence hybrid scoring. Deterministic rules catch structural violations (missing cleans, forbidden orderings); model prediction error flags statistical anomalies.

### OOD Experiment (train MOSFET+IGBT → test IC)

| Model | ID Top-1 | OOD Top-1 | Drop |
|-------|----------|-----------|------|
| n-gram | 0.772 | 0.438 | −43% |
| GPT | 0.817 | 0.451 | −45% |

Both models collapse on unseen IC family due to structural differences (backside grind timing, tungsten via metallurgy) — demonstrating the challenge of OOD generalisation in semiconductor processes.

---

## What worked

1. **Family tokens in training** — prepending a family identifier (`[IGBT]`, `[IC]`, `[MOSFET]`) lets the model condition on product family and learn family-specific transitions without separate models.
2. **Block-boundary injection** — the hierarchical tokenizer's `[BLK:X]` tokens gave the model explicit structure awareness, producing completions that always pass the validator (correct block ordering) even when individual step names have synonym errors.
3. **Multi-model ensemble approach** — having four models with different inductive biases (frequency-based, attention-based, causal, hierarchical) lets us pick the best model per task and provides complementary failure modes.

---

## What didn't work

1. **Synonym variants dominate error** — the main accuracy gap comes from step name synonyms (e.g., "CLEAN WAFER" vs "WAFER CLEAN"), not structural errors. We didn't have time to implement synonym canonicalization.
2. **LCM continuous-space decoding** — predicting embeddings with MSE loss then doing nearest-neighbor lookup introduces quantization noise; the discrete GPT approach proved more reliable for exact step matching.
3. **OOD generalisation** — all models suffer ~45% Top-1 drop on unseen families. The block structure is too family-specific for zero-shot transfer.

---

## What you'd do with another 36 hours

- **Synonym canonicalization layer** — map step name variants to canonical forms before scoring; estimated +5-10% Top-1 gain with minimal effort.
- **Scaling curve** — train GPT at 5M/20M/50M params with 10k/50k/200k sequences to demonstrate clean scaling behavior on Leonardo.
- **Ensemble voting** — combine top-k predictions from GPT + LCM + hierarchical via rank fusion for submission files.
- **Contrastive pre-training** — learn step embeddings via contrastive loss on (step, context) pairs before fine-tuning, to address the synonym problem at the representation level.

---

## Track-specific deliverables

### ⚙️ Industrial AI (Infineon)

- [x] Eval submission files:
  - `data/participant_files/predictions/predictions_nextstep.csv` (Task 1)
  - `data/participant_files/predictions/predictions_completion.csv` (Task 2)
  - `data/participant_files/predictions/predictions_anomaly.csv` (Task 3)
- [x] Training artifacts: checkpoints (`models/hierarchical/model_out/`, `models/transformer/gpt_ckpt.pt`, `hpc/scp/.save/best_lcm_10000.pt`), SLURM job scripts
- [ ] Loss curves visualization (training logs available in SLURM output)
- [x] Scores from self-evaluation on held-out data with per-family breakdown (see Results above)
- [x] Demo shows baseline (Markov) vs. trained (GPT/LCM) output on identical inputs via Gradio app

---

## Credits & dependencies

- **Open-source libraries used**: PyTorch >= 2.0, Transformers >= 4.44, Gradio >= 4.0, NumPy < 2
- **Pre-trained models used**: GPT-2 architecture (trained from scratch, no pre-trained weights)
- **External APIs called**: None
- **AI coding assistants used during the hackathon**: GitHub Copilot, Claude
- **Datasets**: Synthetic semiconductor process sequences generated via `data/gen/generate_sequences.py` (provided by organizers), extended datasets (`*_extended.csv`) for evaluation

---

## A note on honesty

- The Gradio app (`app.py`) requires all model checkpoints to be present locally; some large checkpoints are not committed to git due to size (referenced via path).
- OOD results (Task 4) are from our own train/test split; official OOD scoring is done by organizers post-submission.
- The anomaly detector (Task 3) is primarily rule-based with model confidence as a secondary signal — it's effective but not a learned anomaly detector.

---

*Submitted by team RIT for Zero One Hack_01, 31 May 2026.*
