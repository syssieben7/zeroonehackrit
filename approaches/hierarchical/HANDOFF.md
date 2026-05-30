# Hierarchical GPT — Handoff

## What This Is

A small GPT trained from scratch to complete semiconductor fabrication process sequences. Given the first 60% or 80% of a fab process (e.g. "RECEIVE WAFER LOT → ... → VIA ETCH"), it predicts the remaining steps.

Trained on the Infineon track of the ZeroHack hackathon. Three product families: MOSFET, IGBT, IC.

---

## The Core Idea

### Why Not Just Use the Generator?

`generate_sequences.py` can produce valid sequences, but at inference time you only have a partial sequence — you don't know which family it's from, where in the process you are, or which random choices were made. The model infers all of that from context.

### What Makes It "Hierarchical"

Semiconductor fab sequences have a natural block structure defined in `generation_rules.md`:

```
PREFIX → INIT_MEAS → PRE_CLEAN → FAMILY_PREP → FIRST_OX →
PROCESS_CYCLES (×3–6) → ILD → VIA → METAL → PASSIVATION →
BACKSIDE → FINAL_INSP → TEST_SUITE → SUFFIX
```

`hierarchical_tokenizer.py` injects `[BLK:X]` boundary tokens at each block transition:

```
[BLK:PREFIX] RECEIVE WAFER LOT | LOT IDENTIFICATION |
[BLK:PRE_CLEAN] PRE CLEAN WAFER | HF DIP |
[BLK:FIRST_OX] THERMAL OXIDATION | ...
```

The model learns both coarse structure (which block comes next) and fine detail (which step within a block) in a single forward pass. After seeing `[BLK:PASSIVATION]`, it knows only passivation steps can follow — which is why the validator almost always passes.

### The Model

A GPT-2 architecture trained **from scratch** (no pretrained weights). Each full step name is one token (e.g. "SPIN COAT PHOTORESIST" = token 83). Vocab is ~215 tokens.

Current config (in `train.py`):
- `n_embd=512`, `n_layer=6`, `n_head=8` → ~18M parameters
- Trained on 30,000 generated sequences (10,000 per family)
- 50 epochs, batch size 64, AdamW lr=3e-4

### Training Data

**Generated entirely on the fly** — `FabDataset` calls `generate_sequences.py` 30,000 times at the start of training. The CSV files in `data/raw/infineon/` are NOT used. Same grammar, same distribution, but no disk I/O.

---

## Results

Tested on `IC_extended.csv` seq_0042 at 30% cut (70% to predict):

```
edit distance  : 26 / 86 tokens  (normalised 0.306)
aligned acc    : 60/80 = 75.0%
validator      : ✓ OK
```

Almost all mismatches are **synonyms** (e.g. `STRIP RESIST` vs `STRIP PHOTORESIST`) — the grammar randomly picks between equivalent step names and the model picks a different valid synonym. Structurally, the completions are correct.

---

## Key Files

| File | What it does |
|---|---|
| `hierarchical_tokenizer.py` | Wraps `generate_sequences.py` to tag steps with block labels and inject `[BLK:X]` tokens |
| `train.py` | Generates data, builds vocab, trains GPT, saves checkpoint to `model_out/` |
| `infer.py` | Loads checkpoint, runs completion + eval modes |
| `pixi.toml` | Environment — pytorch from pytorch channel (CUDA 12.1 on Linux, CPU on Mac) |
| `slurm.sh` | Leonardo job submission |
| `model_out/` | Checkpoint: `model.safetensors`, `config.json`, `vocab.json` |
| `USAGE.md` | Command reference |

---

## Known Issues / Limitations

**1. Synonyms inflate edit distance**
The generator randomly picks between equivalent step names (`STRIP RESIST` / `STRIP PHOTORESIST`, `CMP METAL` / `CMP VIA FILL`, etc). The model picks a different valid synonym → counted as error. Fix: canonicalize all synonyms to one form before training. Not yet implemented.

**2. Model trains on same distribution as eval**
Training data comes from `generate_sequences.py`, which is the same grammar that generated `IC_variants.csv`, `IC_extended.csv`, and likely `eval_input_valid.csv`. Good performance on Tasks 1–3 partly reflects this shared distribution. Task 4 (hidden OOD family) will be the real test.

**3. Greedy decoding**
Inference uses argmax at each step. Optional steps (present 30–75% of training sequences) are always or never predicted depending on whether their probability is above 50%. Beam search or temperature sampling would produce more varied completions.

**4. CUDA version mismatch on Leonardo**
Leonardo's CUDA driver = 12.2 but default PyPI torch builds for 12.4+. Fixed by using `pytorch` from the `pytorch` conda channel with `pytorch-cuda = "12.1.*"` in `pixi.toml`. Always run `rm -rf .pixi && pixi install` on the login node after any `pixi.toml` change — never on the compute node (no internet).

---

## Eval Format (from `eval_metrics.py`)

**Task 1 — Next-step prediction:**
```
EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5
valid_0001, DEPOSIT METAL SEED, FILL VIA METAL, CMP VIA FILL, ...
```

**Task 2 — Sequence completion:**
```
EXAMPLE_ID, PREDICTED_SEQUENCE
valid_0001, DEPOSIT METAL SEED|FILL VIA METAL|CMP VIA FILL|...
```

Input file (`eval_input_valid.csv`) has: `EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE`

The `PARTIAL_SEQUENCE` IS the prefix — pass it directly to `infer.py`. The organizers hold the ground truth (`FULL_SEQUENCE`) and score with `eval_metrics.py`.

---

## Workflow

### Training on Leonardo
```bash
git pull
rm -rf .pixi        # always do this after pixi.toml changes
pixi install        # on login node (has internet)
sbatch slurm.sh     # submit job
tail -f <jobid>.out # watch training loss
```

### Inference on Leonardo
```bash
pixi run python infer.py                                        # demo
pixi run python infer.py --csv path/to/IC_extended.csv --seq seq_0042 --cut 0.5
pixi run python infer.py --csv path/to/IC_extended.csv --eval-next-step --n 1000
```

### Copy model checkpoint locally
```bash
scp -r a08trd01@login05-ext.leonardo.cineca.it:/leonardo/home/usertrain/a08trd01/sd/zeroonehackrit/approaches/hierarchical/model_out/ approaches/hierarchical/model_out/
```

---

## What to Do Next

1. **Synonym canonicalization** — collapse synonyms to one canonical form before training. Reduces vocab ~215→175 and eliminates the main source of edit distance error.
2. **Submit predictions** — add `--submit` mode to `infer.py` that reads `eval_input_valid.csv` and writes `predictions_completion.csv` + `predictions_nextstep.csv` in submission format.
3. **Anomaly detection** — use per-token log-probability of a full sequence as anomaly score. Low avg log-prob → likely rule violation.
4. **Scale up** — the loss converged around 0.30 with 18M params. Try larger model or more data.
5. **Beam search** — replace greedy argmax with beam search (k=5) for better optional-step handling.
