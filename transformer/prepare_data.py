#!/usr/bin/env python3
"""
prepare_data.py — Convert raw sequence CSVs into T5 training pairs.

Generates three task types from each sequence:
  - next_step:  prefix -> single next step
  - completion: prefix (40-85% cut) -> remaining steps
  - validate:   full sequence (valid or injected violation) -> "valid" / "invalid <RULE>"

Usage:
    python prepare_data.py --data_dir ../data/raw/infineon --out_dir ./data
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

FAMILIES_EXTENDED = {
    "MOSFET": "MOSFET_extended.csv",
    "IGBT":   "IGBT_extended.csv",
    "IC":     "IC_extended.csv",
}
FAMILIES_VARIANTS = {
    "MOSFET": "MOSFET_variants.csv",
    "IGBT":   "IGBT_variants.csv",
    "IC":     "IC_variants.csv",
}

# ---------------------------------------------------------------------------
# Load sequences
# ---------------------------------------------------------------------------


def load_sequences(
    data_dir: Path,
    families: list[str] | None = None,
    variants_only: bool = False,
) -> dict[str, list[list[str]]]:
    family_map = FAMILIES_VARIANTS if variants_only else FAMILIES_EXTENDED
    selected = {f: family_map[f] for f in (families or family_map)}
    result = {}
    for family, filename in selected.items():
        path = data_dir / family / filename
        if not path.exists():
            fallback = data_dir / family / FAMILIES_VARIANTS[family]
            if fallback.exists():
                path = fallback
            else:
                raise FileNotFoundError(f"No CSV found for {family} at {path}")
        seqs: dict[str, list[str]] = {}
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                sid = row["SEQUENCE_ID"].strip()
                seqs.setdefault(sid, []).append(row["STEP"].strip())
        result[family] = list(seqs.values())
        print(f"  Loaded {len(result[family])} {family} sequences from {path.name}")
    return result


# ---------------------------------------------------------------------------
# Violation injection (for anomaly detection training pairs)
# ---------------------------------------------------------------------------


def inject_violation(steps: list[str], rng: random.Random) -> tuple[list[str], str]:
    """Return a corrupted copy of steps and the rule that was broken."""
    s = steps[:]

    violations = [
        _inject_dep_no_clean,
        _inject_etch_no_mask,
        _inject_ship_before_test,
        _inject_implant_no_mask,
    ]
    return rng.choice(violations)(s, rng)


def _inject_dep_no_clean(s, rng):
    dep_steps = [
        "DEPOSIT BARRIER METAL",
        "DEPOSIT METAL 1",
        "DEPOSIT INTERLAYER DIELECTRIC",
        "DEPOSIT POLYSILICON",
        "DEPOSIT PASSIVATION",
    ]
    for i, step in enumerate(s):
        if step in dep_steps and i >= 12:
            # remove clean steps in the 12-step window before it
            for j in range(max(0, i - 12), i):
                if any(c in s[j] for c in ["CLEAN", "HF DIP", "ANNEAL", "DRY WAFER"]):
                    s[j] = "MEASURE FILM THICKNESS"  # replace with a neutral step
                    return s, "RULE_DEP_NO_CLEAN"
    return s, "RULE_DEP_NO_CLEAN"


def _inject_etch_no_mask(s, rng):
    etch_steps = {
        "OXIDE ETCH",
        "POLYSILICON ETCH",
        "VIA ETCH",
        "METAL ETCH",
        "METAL ETCH DRY",
    }
    for i, step in enumerate(s):
        if step in etch_steps and i >= 5:
            for j in range(max(0, i - 12), i):
                if "DEVELOP" in s[j]:
                    s[j] = "INSPECT WAFER"
                    return s, "RULE_ETCH_NO_MASK"
    return s, "RULE_ETCH_NO_MASK"


def _inject_ship_before_test(s, rng):
    try:
        ship_i = next(i for i, x in enumerate(s) if x == "SHIP LOT")
        sort_i = next(i for i, x in enumerate(s) if x == "WAFER SORT TEST")
        if ship_i > sort_i:
            s[ship_i], s[sort_i] = s[sort_i], s[ship_i]
    except StopIteration:
        pass
    return s, "RULE_SHIP_BEFORE_TEST"


def _inject_implant_no_mask(s, rng):
    implant_steps = {
        "IMPLANT WELL",
        "IMPLANT SOURCE DRAIN",
        "IMPLANT P BODY",
        "IMPLANT N BUFFER",
        "IMPLANT N-TYPE",
    }
    for i, step in enumerate(s):
        if step in implant_steps and i >= 10:
            for j in range(max(0, i - 15), i):
                if "DEVELOP" in s[j] or "ETCH" in s[j]:
                    s[j] = "MEASURE JUNCTION DEPTH"
                    return s, "RULE_IMPLANT_NO_MASK"
    return s, "RULE_IMPLANT_NO_MASK"


# ---------------------------------------------------------------------------
# Pair builders
# ---------------------------------------------------------------------------

SEP = " | "


def build_next_step_pairs(family: str, seq: list[str]) -> list[dict]:
    pairs = []
    for i in range(1, len(seq)):
        inp = f"next step {family.lower()}: {SEP.join(seq[:i])}"
        out = seq[i]
        pairs.append(
            {"input": inp, "output": out, "task": "next_step", "family": family}
        )
    return pairs


def build_completion_pairs(
    family: str, seq: list[str], rng: random.Random, n_cuts: int = 3
) -> list[dict]:
    pairs = []
    n = len(seq)
    for _ in range(n_cuts):
        frac = rng.uniform(0.3, 0.85)
        cut = max(1, int(n * frac))
        inp = f"complete {family.lower()}: {SEP.join(seq[:cut])}"
        out = SEP.join(seq[cut:])
        pairs.append(
            {"input": inp, "output": out, "task": "completion", "family": family}
        )
    return pairs


def build_validate_pairs(family: str, seq: list[str], rng: random.Random) -> list[dict]:
    pairs = []
    # valid example
    pairs.append(
        {
            "input": f"validate {family.lower()}: {SEP.join(seq)}",
            "output": "valid",
            "task": "validate",
            "family": family,
        }
    )
    # invalid example
    corrupted, rule = inject_violation(seq[:], rng)
    pairs.append(
        {
            "input": f"validate {family.lower()}: {SEP.join(corrupted)}",
            "output": f"invalid {rule}",
            "task": "validate",
            "family": family,
        }
    )
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_split(
    all_seqs: dict[str, list[list[str]]],
    rng: random.Random,
    next_step: bool = True,
    completion: bool = True,
    validate: bool = True,
) -> list[dict]:
    pairs = []
    for family, seqs in all_seqs.items():
        for seq in seqs:
            if len(seq) < 5:
                continue
            if next_step:
                pairs.extend(build_next_step_pairs(family, seq))
            if completion:
                pairs.extend(build_completion_pairs(family, seq, rng))
            if validate:
                pairs.extend(build_validate_pairs(family, seq, rng))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/raw/infineon")
    parser.add_argument("--out_dir", default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.05,
        help="Fraction of sequences held out for validation",
    )
    parser.add_argument(
        "--family",
        nargs="+",
        choices=["MOSFET", "IGBT", "IC"],
        default=None,
        help="Families to include (default: all). E.g. --family IC MOSFET",
    )
    parser.add_argument(
        "--variants_only",
        action="store_true",
        help="Use *_variants.csv instead of *_extended.csv",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading sequences...")
    all_seqs = load_sequences(
        Path(args.data_dir),
        families=args.family,
        variants_only=args.variants_only,
    )

    # Split sequences (not pairs) into train/val to avoid leakage
    train_seqs, val_seqs = {}, {}
    for family, seqs in all_seqs.items():
        shuffled = seqs[:]
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * args.val_frac))
        val_seqs[family] = shuffled[:cut]
        train_seqs[family] = shuffled[cut:]

    print("Building training pairs...")
    train_pairs = build_split(train_seqs, rng)
    rng.shuffle(train_pairs)

    print("Building validation pairs...")
    val_pairs = build_split(val_seqs, rng)

    for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

        by_task = {}
        for p in pairs:
            by_task.setdefault(p["task"], 0)
            by_task[p["task"]] += 1
        print(f"  {split}: {len(pairs):,} pairs — {by_task}")

    print(f"\nData written to {out}/")


if __name__ == "__main__":
    main()
