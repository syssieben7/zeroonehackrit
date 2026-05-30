# Parallel track: n-gram baseline + autoregressive GPT (Task 1, next-step)
n-gram (k<=3): Top1=0.756  Top3=0.991  Top5=1.000
GPT (5M, 8ep): Top1=0.806  Top3=0.995  Top5=1.000   (+5.0 pts Top1)
Held-out 100 seq/family. GPT beats n-gram in-distribution.
NEXT: OOD (train MOSFET+IGBT, test IC) — generalization vs memorization.
Env lives on Leonardo at $SCRATCH/mlenv. Run via sync_run.sh.
