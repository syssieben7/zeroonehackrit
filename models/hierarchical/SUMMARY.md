# Weekend Summary — ZeroHack Infineon Track

## What We Built

A small GPT (~18M params) trained from scratch to complete semiconductor fab process sequences for the Infineon track (Tasks 1 & 2).

**Key innovation:** Hierarchical block tokenization — injecting `[BLK:X]` boundary tokens (PREFIX, PRE_CLEAN, PROCESS_CYCLES, etc.) into the training stream, inspired by stable diffusion's coarse-to-fine generation. The model learns process structure at two resolutions simultaneously.

## What We Did

- Designed and implemented `hierarchical_tokenizer.py` — wraps the provided `generate_sequences.py` to tag each step with its grammatical block and inject boundary tokens
- Built `train.py` — GPT-2 from scratch, 30k generated sequences (10k/family), 50 epochs on Leonardo A100
- Built `infer.py` — completion, CSV testing, next-step top-k eval, aligned accuracy with edit distance
- Got everything running on Leonardo HPC via pixi (fought through libstdc++, CUDA driver version, NFS cache issues)
- Tested on `IC_extended.csv` at 30% cut → **75% aligned accuracy, validator always OK**

## Results So Far

| Metric | Value |
|---|---|
| Aligned accuracy (30% prefix) | 75% |
| Normalised edit distance | 0.306 |
| Validator pass rate | 100% |
| Main error source | Synonym variants (not structural errors) |

## What's Left

- Synonym canonicalization (main quick win on accuracy)
- `--submit` mode → generate submission CSVs for Tasks 1 & 2
- Anomaly detection (Task 3) — use sequence log-probability as anomaly score
- OOD evaluation (Task 4, hidden)
