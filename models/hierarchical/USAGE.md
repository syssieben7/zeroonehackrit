# Hierarchical GPT — Usage on Leonardo

All commands run from `approaches/hierarchical/` with the pixi env active.

---

## Training

```bash
# standard run
pixi run train

# save generated training sequences to model_out/training_sequences.csv
pixi run python train.py --save-data

# submit to SLURM
sbatch slurm.sh
```

**Config** (edit top of `train.py`):

| Variable | Default | Effect |
|---|---|---|
| `N_PER_FAMILY` | 10000 | Sequences generated per family (mosfet/igbt/ic) |
| `EPOCHS` | 50 | Training epochs |
| `BATCH_SIZE` | 64 | Sequences per gradient step |
| `LR` | 3e-4 | Learning rate |
| `OUT_DIR` | `model_out/` | Where checkpoint + vocab are saved |
| `n_embd` | 512 | Token embedding size |
| `n_layer` | 6 | Transformer layers |
| `n_head` | 8 | Attention heads |

---

## Inference

### Built-in demo (3 families, 50% cut, generated sequences)
```bash
pixi run python infer.py
```

### Complete a custom prefix (pipe-separated steps)
```bash
pixi run python infer.py "RECEIVE WAFER LOT|LOT IDENTIFICATION|PRE CLEAN WAFER|HF DIP"
```

### Complete a sequence from a CSV
```bash
pixi run python infer.py --csv <path> --seq <seq_id> --cut <fraction>
```

| Param | Default | Effect |
|---|---|---|
| `--csv` | — | Path to variants CSV (e.g. `../../data/raw/infineon/IC/IC_extended.csv`) |
| `--seq` | `seq_0001` | Sequence ID to load |
| `--cut` | `0.6` | Fraction of sequence used as prefix (0.4 = harder, 0.8 = easier) |

**Example:**
```bash
pixi run python infer.py \
  --csv ../../data/raw/infineon/IC/IC_extended.csv \
  --seq seq_0042 \
  --cut 0.5
```

Output: side-by-side aligned table with edit distance, aligned accuracy, validator result.

---

## Next-Step Accuracy Eval

Evaluates Top-1 / Top-3 / Top-5 accuracy and MRR across many sequences.

```bash
pixi run python infer.py --eval-next-step --csv <path> [--cut <fraction>] [--n <count>]
```

| Param | Default | Effect |
|---|---|---|
| `--csv` | — | Path to variants CSV |
| `--cut` | `0.6` | Evaluate at all positions up to this fraction of each sequence |
| `--n` | `200` | Max number of sequences to evaluate |

**Example:**
```bash
pixi run python infer.py \
  --csv ../../data/raw/infineon/IC/IC_extended.csv \
  --eval-next-step \
  --cut 0.8 \
  --n 5000
```

---

## Data paths on Leonardo

```
../../data/raw/infineon/IC/IC_variants.csv       # 1000 sequences
../../data/raw/infineon/IC/IC_extended.csv        # 5000 sequences
../../data/raw/infineon/MOSFET/MOSFET_variants.csv
../../data/raw/infineon/MOSFET/MOSFET_extended.csv
../../data/raw/infineon/IGBT/IGBT_variants.csv
../../data/raw/infineon/IGBT/IGBT_extended.csv
```

---

## Monitoring jobs

```bash
squeue --me                  # check running jobs
tail -f <jobid>.out          # training loss (stdout)
tail -f <jobid>.err          # pixi/error output (stderr)
cat <jobid>.out              # full output after completion
```
