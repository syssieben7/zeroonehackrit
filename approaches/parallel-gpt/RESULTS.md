# Parallel track: n-gram baseline + autoregressive GPT

## In-distribution results (train+test on all 3 families, 100 held-out/family)

| Model         | Top1  | Top3  | Top5  | n      |
|---------------|-------|-------|-------|--------|
| n-gram (k≤3)  | 0.756 | 0.991 | 1.000 | 38524  |
| GPT (5M, 8ep) | 0.806 | 0.995 | 1.000 | 38524  |

GPT beats n-gram by +5.0 pts Top1 in-distribution.

## OOD experiment: train on MOSFET+IGBT, test on held-out IC

| Model         | Test set              | Top1  | Top3  | Top5  | n      |
|---------------|-----------------------|-------|-------|-------|--------|
| n-gram (k≤3)  | MOS+IGBT held-out (ID)| 0.772 | 0.996 | 1.000 | 27098  |
| n-gram (k≤3)  | IC held-out (OOD)     | 0.438 | 0.630 | 0.640 | 11426  |
| GPT (5M, 8ep) | MOS+IGBT held-out (ID)| 0.817 | 0.996 | 1.000 | 27098  |
| GPT (5M, 8ep) | IC held-out (OOD)     | 0.451 | 0.635 | 0.658 | 11426  |

ID→OOD drop: n-gram −0.334 (43%), GPT −0.366 (45%) absolute Top1.

**Interpretation:** Both models collapse roughly equally on the held-out IC family.
IC has a fundamentally different family-specific prep block (backside grind early,
tungsten vias, no epitaxy) — family-specific steps are unseen at training time for both.
GPT shows a marginal Top5 edge (0.658 vs 0.640), suggesting slightly better
distribution coverage, but the "GPT understands process logic; n-gram memorizes"
hypothesis is not cleanly supported by OOD numbers alone.

The shared backbone (PREFIX → CLEAN → LITHO cycles → TEST → SUFFIX) is where both
models succeed OOD; IC-specific transition sequences are where both fail.

## Environment
- Compute: Leonardo HPC, A100 GPU, SLURM
- Env: pixi at $SCRATCH/mlenv (python3.12, torch)
- Data: 1000 seqs/family; 100 held-out/family, seed=0
- Run: `cd approaches/parallel-gpt && ./sync_run.sh`

## Status
- [x] In-dist baseline (n-gram + GPT)
- [x] OOD experiment (train MOS+IGBT, test IC)
- [ ] Submission outputs (predict.py) — waiting for eval_input_*.csv from organizers
- [ ] T3 anomaly detection (rule-checker + GPT surprisal)
