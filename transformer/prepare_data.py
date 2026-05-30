#!/usr/bin/env python3
"""
prepare_data.py — Convert raw sequence CSVs into T5 training pairs.

Generates three task types from each sequence:
  - next_step:  prefix -> single next step
  - completion: prefix (30-85% cut) -> remaining steps
  - validate:   full sequence (valid or injected violation) -> "valid" / "invalid <RULE>"

All 10 process-logic rules have violation injectors.
Step descriptions from *_longdescription_parameters.csv are optionally
inlined after each step name: "DEPOSIT BARRIER METAL (thin diffusion barrier before via fill)"

Usage:
    python prepare_data.py --data_dir ../data/raw/infineon --out_dir ../data/processed/transformer
    python prepare_data.py --data_dir ../data/raw/infineon --out_dir ../data/processed/transformer --inline_descriptions
"""

import argparse
import csv
import json
import random
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
DESCRIPTION_FILES = {
    "MOSFET": "MOSFET_longdescription_parameters.csv",
    "IGBT":   "IGBT_longdescription_parameters.csv",
    "IC":     "IC_longdescription_parameters.csv",
}

SEP = " | "


# ---------------------------------------------------------------------------
# Load step descriptions
# ---------------------------------------------------------------------------

def load_descriptions(data_dir: Path) -> dict[str, dict[str, str]]:
    """Return {family: {step_name: short_description}}."""
    result = {}
    for family, filename in DESCRIPTION_FILES.items():
        path = data_dir / family / filename
        descs = {}
        if path.exists():
            with path.open(newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    step = row.get("STEP", "").strip().strip('"')
                    desc = row.get("DESCRIPTION", "").strip().strip('"')
                    if step and desc:
                        descs[step] = desc
        result[family] = descs
        print(f"  Loaded {len(descs)} descriptions for {family}")
    return result


def annotate(step: str, descs: dict[str, str]) -> str:
    """Return 'STEP NAME (short description)' if description exists."""
    d = descs.get(step)
    return f"{step} ({d})" if d else step


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
# Violation injectors — all 10 rules
# ---------------------------------------------------------------------------

def inject_violation(steps: list[str], rng: random.Random) -> tuple[list[str], str]:
    s = steps[:]
    injectors = [
        _inject_dep_no_clean,
        _inject_metal_etch_no_litho,
        _inject_etch_no_mask,
        _inject_litho_level_skip,
        _inject_implant_no_mask,
        _inject_cmp_no_dep,
        _inject_pad_open_before_dep,
        _inject_test_before_passivation,
        _inject_ship_before_test,
        _inject_backside_before_passivation,
    ]
    rng.shuffle(injectors)
    for injector in injectors:
        result, rule = injector(s[:], rng)
        if result is not None:
            return result, rule
    # fallback — should not happen
    return s, "RULE_DEP_NO_CLEAN"


def _inject_dep_no_clean(s, rng):
    dep_steps = {
        "DEPOSIT BARRIER METAL", "DEPOSIT METAL 1", "DEPOSIT TOP METAL",
        "DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC",
        "DEPOSIT POLYSILICON", "DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER",
        "DEPOSIT FIELD OXIDE", "DEPOSIT GATE OXIDE OR DIELECTRIC",
    }
    for i, step in enumerate(s):
        if step in dep_steps and i >= 12:
            for j in range(max(0, i - 12), i):
                if any(c in s[j] for c in ["CLEAN", "HF DIP", "ANNEAL", "DRY WAFER", "SURFACE PREP"]):
                    s[j] = "MEASURE FILM THICKNESS"
                    return s, "RULE_DEP_NO_CLEAN"
    return None, ""


def _inject_metal_etch_no_litho(s, rng):
    metal_etch = {"METAL ETCH", "METAL ETCH DRY"}
    for i, step in enumerate(s):
        if step in metal_etch and i >= 15:
            w = s[max(0, i - 15):i]
            expose_idx = next((j for j, x in enumerate(w) if x.startswith("EXPOSE LITHO LEVEL")), None)
            if expose_idx is not None:
                s[max(0, i - 15) + expose_idx] = "MEASURE METAL THICKNESS"
                return s, "RULE_METAL_ETCH_NO_LITHO"
    return None, ""


def _inject_etch_no_mask(s, rng):
    etch_steps = {
        "OXIDE ETCH", "OXIDE ETCH DRY", "POLYSILICON ETCH", "POLYSILICON ETCH DRY",
        "VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA",
        "FIELD OXIDE ETCH", "ETCH SILICON OR OXIDE WINDOW",
    }
    for i, step in enumerate(s):
        if step in etch_steps and i >= 5:
            for j in range(max(0, i - 12), i):
                if "DEVELOP" in s[j]:
                    s[j] = "INSPECT WAFER"
                    return s, "RULE_ETCH_NO_MASK"
    return None, ""


def _inject_litho_level_skip(s, rng):
    align_idxs = [(i, int(x.split("ALIGN MASK LEVEL ")[1]))
                  for i, x in enumerate(s)
                  if x.startswith("ALIGN MASK LEVEL ") and x.split("ALIGN MASK LEVEL ")[1].isdigit()]
    if len(align_idxs) >= 2:
        # swap two non-adjacent levels to create a skip
        i1, l1 = align_idxs[0]
        i2, l2 = align_idxs[-1]
        if l2 - l1 >= 2:
            s[i1] = f"ALIGN MASK LEVEL {l2}"
            s[i2] = f"ALIGN MASK LEVEL {l1}"
            return s, "RULE_LITHO_LEVEL_SKIP"
    return None, ""


def _inject_implant_no_mask(s, rng):
    implant_steps = {
        "IMPLANT WELL", "IMPLANT SOURCE DRAIN", "IMPLANT SOURCE REGION",
        "IMPLANT P BODY", "IMPLANT N BUFFER", "IMPLANT N-TYPE",
        "IMPLANT CHANNEL STOP", "IMPLANT DRAIN / CATHODE REGION", "IMPLANT LDD",
    }
    for i, step in enumerate(s):
        if step in implant_steps and i >= 10:
            for j in range(max(0, i - 15), i):
                if "DEVELOP" in s[j] or "OXIDE ETCH" in s[j] or "ETCH SILICON" in s[j]:
                    s[j] = "MEASURE JUNCTION DEPTH"
                    return s, "RULE_IMPLANT_NO_MASK"
    return None, ""


def _inject_cmp_no_dep(s, rng):
    cmp_steps = {"CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC", "CMP METAL", "CMP VIA FILL"}
    fill_steps = {"FILL VIA METAL", "FILL VIA TUNGSTEN", "DEPOSIT INTERLAYER DIELECTRIC",
                  "DEPOSIT INTERLEVEL DIELECTRIC", "DEPOSIT BARRIER METAL"}
    for i, step in enumerate(s):
        if step in cmp_steps and i >= 6:
            for j in range(max(0, i - 6), i):
                if s[j] in fill_steps:
                    s[j] = "MEASURE PLANARITY"
                    return s, "RULE_CMP_NO_DEP"
    return None, ""


def _inject_pad_open_before_dep(s, rng):
    pad_steps = {"OPEN PAD WINDOW", "OPEN BOND PAD WINDOW", "PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO"}
    try:
        dep_i = next(i for i, x in enumerate(s) if x in ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"))
        pad_i = next(i for i, x in enumerate(s) if x in pad_steps)
        if pad_i > dep_i:
            # move pad window step before passivation deposition
            step = s.pop(pad_i)
            s.insert(max(0, dep_i - 2), step)
            return s, "RULE_PAD_OPEN_BEFORE_DEP"
    except StopIteration:
        pass
    return None, ""


def _inject_test_before_passivation(s, rng):
    test_steps = {"PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST",
                  "THRESHOLD VOLTAGE TEST", "BREAKDOWN VOLTAGE TEST",
                  "LEAKAGE TEST", "SWITCHING TEST"}
    try:
        cure_i = next(i for i, x in enumerate(s) if x == "CURE PASSIVATION")
        test_i = next(i for i, x in enumerate(s) if x in test_steps)
        if test_i > cure_i:
            step = s.pop(test_i)
            s.insert(max(0, cure_i - 1), step)
            return s, "RULE_TEST_BEFORE_PASSIVATION"
    except StopIteration:
        pass
    return None, ""


def _inject_ship_before_test(s, rng):
    try:
        ship_i = next(i for i, x in enumerate(s) if x == "SHIP LOT")
        sort_i = next(i for i, x in enumerate(s) if x == "WAFER SORT TEST")
        if ship_i > sort_i:
            s[ship_i], s[sort_i] = s[sort_i], s[ship_i]
            return s, "RULE_SHIP_BEFORE_TEST"
    except StopIteration:
        pass
    return None, ""


def _inject_backside_before_passivation(s, rng):
    try:
        cure_i  = next(i for i, x in enumerate(s) if x == "CURE PASSIVATION")
        back_i  = next(i for i, x in enumerate(s) if x == "DEPOSIT BACKSIDE METAL")
        if back_i > cure_i:
            step = s.pop(back_i)
            s.insert(max(0, cure_i - 1), step)
            return s, "RULE_BACKSIDE_BEFORE_PASSIVATION"
    except StopIteration:
        pass
    return None, ""


# ---------------------------------------------------------------------------
# Pair builders
# ---------------------------------------------------------------------------

def _fmt(steps: list[str], descs: dict[str, str], inline: bool) -> str:
    if inline:
        return SEP.join(annotate(s, descs) for s in steps)
    return SEP.join(steps)


def build_next_step_pairs(
    family: str, seq: list[str], descs: dict[str, str], inline: bool
) -> list[dict]:
    pairs = []
    for i in range(1, len(seq)):
        inp = f"next step {family.lower()}: {_fmt(seq[:i], descs, inline)}"
        out = seq[i]
        pairs.append({"input": inp, "output": out, "task": "next_step", "family": family})
    return pairs


def build_completion_pairs(
    family: str, seq: list[str], descs: dict[str, str], inline: bool,
    rng: random.Random, n_cuts: int = 3,
) -> list[dict]:
    pairs = []
    n = len(seq)
    for _ in range(n_cuts):
        frac = rng.uniform(0.3, 0.85)
        cut  = max(1, int(n * frac))
        inp  = f"complete {family.lower()}: {_fmt(seq[:cut], descs, inline)}"
        out  = SEP.join(seq[cut:])   # output is always plain step names
        pairs.append({"input": inp, "output": out, "task": "completion", "family": family})
    return pairs


def build_validate_pairs(
    family: str, seq: list[str], descs: dict[str, str], inline: bool,
    rng: random.Random,
) -> list[dict]:
    pairs = []
    # valid
    pairs.append({
        "input":  f"validate {family.lower()}: {_fmt(seq, descs, inline)}",
        "output": "valid",
        "task":   "validate",
        "family": family,
    })
    # invalid
    corrupted, rule = inject_violation(seq[:], rng)
    pairs.append({
        "input":  f"validate {family.lower()}: {_fmt(corrupted, descs, inline)}",
        "output": f"invalid {rule}",
        "task":   "validate",
        "family": family,
    })
    return pairs


# ---------------------------------------------------------------------------
# Build split
# ---------------------------------------------------------------------------

def build_split(
    all_seqs: dict[str, list[list[str]]],
    all_descs: dict[str, dict[str, str]],
    rng: random.Random,
    inline: bool = False,
) -> list[dict]:
    pairs = []
    for family, seqs in all_seqs.items():
        descs = all_descs.get(family, {})
        for seq in seqs:
            if len(seq) < 5:
                continue
            pairs.extend(build_next_step_pairs(family, seq, descs, inline))
            pairs.extend(build_completion_pairs(family, seq, descs, inline, rng))
            pairs.extend(build_validate_pairs(family, seq, descs, inline, rng))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default="../data/raw/infineon")
    parser.add_argument("--out_dir",   default="../data/processed/transformer")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--val_frac",  type=float, default=0.05)
    parser.add_argument(
        "--family", nargs="+", choices=["MOSFET", "IGBT", "IC"], default=None,
        help="Families to include (default: all)",
    )
    parser.add_argument(
        "--variants_only", action="store_true",
        help="Use *_variants.csv instead of *_extended.csv",
    )
    parser.add_argument(
        "--inline_descriptions", action="store_true",
        help="Append short description after each step name in inputs",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading descriptions...")
    all_descs = load_descriptions(Path(args.data_dir))

    print("Loading sequences...")
    all_seqs = load_sequences(Path(args.data_dir), args.family, args.variants_only)

    train_seqs, val_seqs = {}, {}
    for family, seqs in all_seqs.items():
        shuffled = seqs[:]
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * args.val_frac))
        val_seqs[family]   = shuffled[:cut]
        train_seqs[family] = shuffled[cut:]

    print("Building training pairs...")
    train_pairs = build_split(train_seqs, all_descs, rng, args.inline_descriptions)
    rng.shuffle(train_pairs)

    print("Building validation pairs...")
    val_pairs = build_split(val_seqs, all_descs, rng, args.inline_descriptions)

    for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        by_task: dict[str, int] = {}
        for p in pairs:
            by_task[p["task"]] = by_task.get(p["task"], 0) + 1
        print(f"  {split}: {len(pairs):,} pairs — {by_task}")

    print(f"\nDone. Data written to {out}/")


if __name__ == "__main__":
    main()
