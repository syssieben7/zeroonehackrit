# [Approach name] — results summary

> Copy this file to your approach directory as RESULTS.md and fill in the
> bracketed fields. Delete sections that don't apply. Commit it with a number
> or don't put it in the comparison table — code without a committed number
> can't go in the table.

---

## Approach

| Field       | Value |
|-------------|-------|
| Model       | [e.g. T5-base fine-tuned / Seq2Seq+attention / Hierarchical GPT2] |
| Families    | [MOSFET + IGBT + IC / MOSFET + IGBT only / …] |
| Train data  | [variants.csv / extended.csv / synthetic / …] |
| Checkpoint  | [path or HF repo] |

---

## Shared eval protocol

All approaches use the same split so numbers are directly comparable:

- **Hold-out:** 100 sequences / family, `seed=0`, drawn before training
- **Vocab:** built from all 3 families (MOSFET + IGBT + IC) in all cases —
  this tests process-logic generalisation, not out-of-vocabulary handling
- **Metric:** Top-K accuracy at the last step of a partial prefix
  (same definition as `eval_metrics.py` from the organizers)

Reference numbers to beat:

| Model        | Top1      | Top3  | Top5  | n      |
|--------------|-----------|-------|-------|--------|
| n-gram (k≤3) | 0.756     | 0.991 | 1.000 | 38524  |
| GPT (5M, 8ep)| **0.806** | 0.995 | 1.000 | 38524  |

---

## Task 1 — Next-step prediction

Held-out 100 sequences/family, seed=0. All 3 families in train+test.

| Model         | Top1  | Top3  | Top5  | n      |
|---------------|-------|-------|-------|--------|
| n-gram (k≤3)  | 0.756 | 0.991 | 1.000 | 38524  |
| GPT (5M, 8ep) | 0.806 | 0.995 | 1.000 | 38524  |
| [Your model]  |       |       |       |        |

**One honest finding:** [What does your model do better or worse than GPT,
and why? One paragraph. If you don't know yet, leave blank and add it before
the presentation.]

---

## Task 2 — Sequence completion *(fill in if attempted)*

| Model        | Metric              | Value |
|--------------|---------------------|-------|
| GPT (greedy) | % ending at SHIP LOT| 100%  |
| GPT (greedy) | Mean steps generated| 41    |
| [Your model] | % ending at SHIP LOT|       |
| [Your model] | Mean steps generated|       |

---

## Task 3 — Anomaly detection *(fill in if attempted)*

| Model        | IS_VALID F1 | Mean score (valid) | Mean score (invalid) |
|--------------|-------------|--------------------|----------------------|
| GPT + rules  | TBD         | 0.707              | 0.450                |
| [Your model] |             |                    |                      |

---

## Repro

```bash
# one command that produces a number from a clean checkout
[e.g. cd approaches/parallel-gpt && python3 baseline.py $DATA]
```

---

## Status

- [ ] Task 1 number committed
- [ ] Task 2 submission generated
- [ ] Task 3 submission generated
- [ ] Official scores received
