#!/usr/bin/env python3
"""
validate_submission.py — validate prediction CSVs against generation_rules.md §5.

Checks per task
---------------
T1  EXAMPLE_ID,RANK_1,RANK_2,RANK_3,RANK_4,RANK_5
    601 lines (header + 600 data rows)
    every EXAMPLE_ID from eval_input_valid.csv present, no extras
    5 non-empty rank columns per row

T2  EXAMPLE_ID,PREDICTED_SEQUENCE
    601 lines
    every EXAMPLE_ID from eval_input_valid.csv present, no extras
    PREDICTED_SEQUENCE non-empty; each pipe-separated token non-empty

T3  EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE
    988 lines (header + 987 data rows)
    every EXAMPLE_ID from eval_input_anomaly.csv present, no extras
    IS_VALID in {0, 1}
    SCORE float in (0.0, 1.0] (required; used for AUC)
    PREDICTED_RULE present when IS_VALID=0, empty when IS_VALID=1

Exit codes
----------
0  all checks pass
1  one or more checks fail

Usage
-----
# validate my files (default paths):
    python validate_submission.py

# validate the transformer team set:
    python validate_submission.py \\
        --t1 models/transformer/predictions/task1_predictions.csv \\
        --t2 models/transformer/predictions/task2_predictions.csv \\
        --t3 models/transformer/predictions/task3_predictions.csv

# override eval inputs (if run from outside repo root):
    python validate_submission.py --eval-valid /path/to/eval_input_valid.csv ...
"""

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent

# ── §5 spec constants ─────────────────────────────────────────────────────────

T1_COLS      = ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"]
T2_COLS      = ["EXAMPLE_ID", "PREDICTED_SEQUENCE"]
T3_COLS      = ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"]

T1_TOTAL     = 601   # header + 600 data rows
T2_TOTAL     = 601
T3_TOTAL     = 988   # header + 987 data rows

# ── helpers ───────────────────────────────────────────────────────────────────

def load_ids(path: Path) -> set:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["EXAMPLE_ID"].strip() for row in csv.DictReader(f)}


def read_csv(path: Path):
    """Return (header_list, list_of_row_lists)."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], rows[1:]


class Results:
    def __init__(self, label: str):
        self.label   = label
        self.checks  = []   # (name, passed, detail)
        self.any_fail = False

    def record(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append((name, passed, detail))
        if not passed:
            self.any_fail = True
        return passed

    def print(self):
        width = max(len(c[0]) for c in self.checks) + 2
        print(f"\n── {self.label} ──")
        for name, passed, detail in self.checks:
            tag  = "PASS" if passed else "FAIL"
            line = f"  [{tag}]  {name}"
            if detail and not passed:
                line += f"\n          → {detail}"
            print(line)


# ── per-task validators ───────────────────────────────────────────────────────

def validate_task1(path: Path, eval_ids: set) -> Results:
    r = Results(f"Task 1 — next-step prediction  ({path})")

    if not r.record("file exists", path.exists(), str(path)):
        for name in ("exact columns", "row count", "EXAMPLE_ID coverage", "rank values"):
            r.record(name, False, "skipped — file missing")
        return r

    header, data = read_csv(path)

    r.record("exact columns", header == T1_COLS,
             f"got {header}" if header != T1_COLS else "")

    r.record(f"row count == {T1_TOTAL} (incl. header)", len(data) + 1 == T1_TOTAL,
             f"got {len(data) + 1}")

    found_ids = {row[0].strip() for row in data if row}
    missing   = eval_ids - found_ids
    extra     = found_ids - eval_ids
    id_detail = ""
    if missing: id_detail += f"{len(missing)} missing e.g. {sorted(missing)[:3]}  "
    if extra:   id_detail += f"{len(extra)} unexpected e.g. {sorted(extra)[:3]}"
    r.record("EXAMPLE_ID coverage (no missing, no extras)", not missing and not extra, id_detail)

    errs = []
    for i, row in enumerate(data, start=2):
        if len(row) != len(T1_COLS):
            errs.append(f"row {i}: {len(row)} cols")
            continue
        for col, val in zip(T1_COLS[1:], row[1:]):   # RANK_1..5
            if not val.strip():
                errs.append(f"row {i}: {col} is empty")
        if len(errs) >= 5:
            errs.append("…(truncated)")
            break
    r.record("all rank columns non-empty", not errs, "; ".join(errs))

    return r


def validate_task2(path: Path, eval_ids: set) -> Results:
    r = Results(f"Task 2 — sequence completion  ({path})")

    if not r.record("file exists", path.exists(), str(path)):
        for name in ("exact columns", "row count", "EXAMPLE_ID coverage",
                     "PREDICTED_SEQUENCE non-empty", "pipe tokens non-empty"):
            r.record(name, False, "skipped — file missing")
        return r

    header, data = read_csv(path)

    r.record("exact columns", header == T2_COLS,
             f"got {header}" if header != T2_COLS else "")

    r.record(f"row count == {T2_TOTAL} (incl. header)", len(data) + 1 == T2_TOTAL,
             f"got {len(data) + 1}")

    found_ids = {row[0].strip() for row in data if row}
    missing   = eval_ids - found_ids
    extra     = found_ids - eval_ids
    id_detail = ""
    if missing: id_detail += f"{len(missing)} missing e.g. {sorted(missing)[:3]}  "
    if extra:   id_detail += f"{len(extra)} unexpected e.g. {sorted(extra)[:3]}"
    r.record("EXAMPLE_ID coverage (no missing, no extras)", not missing and not extra, id_detail)

    empty_seq, bad_tokens = [], []
    for i, row in enumerate(data, start=2):
        if len(row) < 2:
            empty_seq.append(i)
            continue
        seq = row[1]
        if not seq.strip():
            empty_seq.append(i)
            continue
        tokens = [t.strip() for t in seq.split("|")]
        if any(t == "" for t in tokens):
            bad_tokens.append(f"row {i}: empty token in pipe-separated sequence")
        if len(empty_seq) + len(bad_tokens) >= 5:
            empty_seq.append(-1)   # sentinel for truncation
            break

    empty_detail = f"rows {empty_seq[:5]}" if empty_seq else ""
    r.record("PREDICTED_SEQUENCE non-empty", not empty_seq, empty_detail)
    r.record("pipe-separated tokens all non-empty", not bad_tokens,
             "; ".join(bad_tokens[:3]))

    return r


def validate_task3(path: Path, eval_ids: set) -> Results:
    r = Results(f"Task 3 — anomaly detection  ({path})")

    if not r.record("file exists", path.exists(), str(path)):
        for name in ("exact columns", "row count", "EXAMPLE_ID coverage",
                     "IS_VALID in {0,1}", "SCORE in (0.0,1.0]",
                     "PREDICTED_RULE consistency"):
            r.record(name, False, "skipped — file missing")
        return r

    header, data = read_csv(path)

    r.record("exact columns", header == T3_COLS,
             f"got {header}" if header != T3_COLS else "")

    r.record(f"row count == {T3_TOTAL} (incl. header)", len(data) + 1 == T3_TOTAL,
             f"got {len(data) + 1}")

    found_ids = {row[0].strip() for row in data if row}
    missing   = eval_ids - found_ids
    extra     = found_ids - eval_ids
    id_detail = ""
    if missing: id_detail += f"{len(missing)} missing e.g. {sorted(missing)[:3]}  "
    if extra:   id_detail += f"{len(extra)} unexpected e.g. {sorted(extra)[:3]}"
    r.record("EXAMPLE_ID coverage (no missing, no extras)", not missing and not extra, id_detail)

    bad_valid, bad_score, bad_rule = [], [], []
    for i, row in enumerate(data, start=2):
        if len(row) != 4:
            bad_valid.append(f"row {i}: {len(row)} cols")
            continue

        _, is_valid_s, score_s, pred_rule = row
        is_valid_s = is_valid_s.strip()
        score_s    = score_s.strip()
        pred_rule  = pred_rule.strip()

        # IS_VALID
        if is_valid_s not in ("0", "1"):
            bad_valid.append(f"row {i}: IS_VALID={is_valid_s!r}")

        # SCORE
        try:
            score = float(score_s)
            if not (0.0 < score <= 1.0):
                bad_score.append(f"row {i}: SCORE={score} not in (0,1]")
        except ValueError:
            bad_score.append(f"row {i}: SCORE={score_s!r} not a float")

        # PREDICTED_RULE consistency: present iff IS_VALID==0
        if is_valid_s == "0" and not pred_rule:
            bad_rule.append(f"row {i}: IS_VALID=0 but PREDICTED_RULE empty")
        elif is_valid_s == "1" and pred_rule:
            bad_rule.append(f"row {i}: IS_VALID=1 but PREDICTED_RULE={pred_rule!r}")

        if max(len(bad_valid), len(bad_score), len(bad_rule)) >= 5:
            for lst in (bad_valid, bad_score, bad_rule):
                if len(lst) >= 5:
                    lst.append("…(truncated)")
            break

    r.record("IS_VALID in {0, 1}", not bad_valid, "; ".join(bad_valid[:4]))
    r.record("SCORE in (0.0, 1.0]", not bad_score, "; ".join(bad_score[:4]))
    r.record("PREDICTED_RULE present iff IS_VALID=0", not bad_rule,
             "; ".join(bad_rule[:4]))

    return r


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Validate prediction CSVs against generation_rules.md §5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--t1", default=str(SCRIPT_DIR / "predictions/task1_predictions.csv"),
                   help="Task 1 CSV (default: predictions/task1_predictions.csv)")
    p.add_argument("--t2", default=str(SCRIPT_DIR / "predictions/task2_predictions.csv"),
                   help="Task 2 CSV (default: predictions/task2_predictions.csv)")
    p.add_argument("--t3", default=str(SCRIPT_DIR / "predictions/task3_predictions.csv"),
                   help="Task 3 CSV (default: predictions/task3_predictions.csv)")
    p.add_argument("--eval-valid",
                   default=str(REPO_ROOT / "data/participant_files/eval_input_valid.csv"),
                   help="eval_input_valid.csv from organizers")
    p.add_argument("--eval-anomaly",
                   default=str(REPO_ROOT / "data/participant_files/eval_input_anomaly.csv"),
                   help="eval_input_anomaly.csv from organizers")
    args = p.parse_args()

    # load eval IDs
    try:
        valid_ids   = load_ids(Path(args.eval_valid))
        anomaly_ids = load_ids(Path(args.eval_anomaly))
    except FileNotFoundError as exc:
        print(f"ERROR: eval input file not found: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Eval IDs loaded: {len(valid_ids)} valid, {len(anomaly_ids)} anomaly")

    results = [
        validate_task1(Path(args.t1), valid_ids),
        validate_task2(Path(args.t2), valid_ids),
        validate_task3(Path(args.t3), anomaly_ids),
    ]

    any_fail = False
    for res in results:
        res.print()
        if res.any_fail:
            any_fail = True

    print()
    if any_fail:
        print("══════════════════════════════════════")
        print("  OVERALL RESULT: FAIL")
        print("══════════════════════════════════════")
        sys.exit(1)
    else:
        print("══════════════════════════════════════")
        print("  OVERALL RESULT: PASS")
        print("══════════════════════════════════════")
        sys.exit(0)


if __name__ == "__main__":
    main()
