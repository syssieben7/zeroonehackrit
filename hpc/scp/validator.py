"""
Process-logic validator for semiconductor sequences.
Checks all 10 rules from generation_rules.md.
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Vocabulary sets used by the validator
# ---------------------------------------------------------------------------

DEPOSITION_STEPS = frozenset({
    "THERMAL OXIDATION", "GATE OXIDE GROWTH", "DEPOSIT PAD OXIDE",
    "EPITAXIAL DEPOSITION", "DEPOSIT POLYSILICON", "DEPOSIT SPACER DIELECTRIC",
    "DEPOSIT FIELD OXIDE", "DEPOSIT GATE OXIDE OR DIELECTRIC",
    "DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC",
    "DEPOSIT BARRIER METAL", "DEPOSIT METAL SEED", "DEPOSIT METAL 1",
    "DEPOSIT TOP METAL", "DEPOSIT BACKSIDE METAL", "DEPOSIT TUNGSTEN SEED",
    "DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER",
    "DEPOSIT BACKSIDE PROTECTION",
})

CLEAN_STEPS = frozenset({
    "PRE CLEAN WAFER", "WAFER CLEAN PRE PROCESS", "WAFER SURFACE CLEAN",
    "RCA CLEAN 1", "RCA CLEAN 2", "WET CLEAN RCA1", "WET CLEAN RCA2",
    "HF DIP", "OXIDE STRIP", "SURFACE PREP FOR DEPOSITION",
    "FRONTSIDE CLEAN", "BACKSIDE CLEAN", "FRONTSIDE CLEAN FINAL",
    "BACKSIDE CLEAN FINAL", "WAFER CLEAN PRE-GRIND",
    "DRY WAFER", "DRY WAFER BACKSIDE",
    "CLEAN AFTER ETCH", "CLEAN AFTER OXIDE ETCH", "CLEAN AFTER POLY ETCH",
    "CLEAN AFTER VIA ETCH", "CLEAN AFTER METAL ETCH",
    "CLEAN AFTER WINDOW ETCH", "CLEAN AFTER FIELD ETCH",
    "CLEAN PAD OPENING", "BACKSIDE ETCH CLEAN", "BACKSIDE RINSE",
    "THERMAL OXIDATION", "GATE OXIDE PREP", "RAPID THERMAL ANNEAL",
    "EPITAXY ANNEAL", "ANNEAL OXIDE",
})

ETCH_STEPS = frozenset({
    "OXIDE ETCH", "OXIDE ETCH DRY", "POLYSILICON ETCH", "POLYSILICON ETCH DRY",
    "ETCH SILICON OR OXIDE WINDOW", "FIELD OXIDE ETCH",
    "VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA",
    "METAL ETCH", "METAL ETCH DRY",
    "PASSIVATION ETCH PAD OPENING", "PASSIVATION ETCH",
})

METAL_ETCH_STEPS = frozenset({"METAL ETCH", "METAL ETCH DRY"})

IMPLANT_STEPS = frozenset({
    "IMPLANT WELL", "IMPLANT SOURCE DRAIN", "IMPLANT SOURCE REGION",
    "IMPLANT LDD", "IMPLANT P BODY", "IMPLANT N BUFFER",
    "IMPLANT CHANNEL STOP", "IMPLANT DRAIN / CATHODE REGION", "IMPLANT N-TYPE",
})

IMPLANT_OPENER_STEPS = frozenset({
    "OXIDE ETCH", "OXIDE ETCH DRY", "ETCH SILICON OR OXIDE WINDOW",
    "DEVELOP PHOTORESIST",
})

CMP_STEPS = frozenset({
    "CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC",
    "CMP METAL", "CMP VIA FILL",
})

FILL_STEPS = frozenset({"FILL VIA METAL", "FILL VIA TUNGSTEN"}) | DEPOSITION_STEPS

PAD_WINDOW_STEPS = frozenset({
    "OPEN PAD WINDOW", "OPEN BOND PAD WINDOW",
    "PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO",
})

ELECTRICAL_TEST_STEPS = frozenset({
    "PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST",
    "THRESHOLD VOLTAGE TEST", "BREAKDOWN VOLTAGE TEST",
    "LEAKAGE TEST", "SWITCHING TEST",
})

BACKSIDE_METAL_STEPS = frozenset({"DEPOSIT BACKSIDE METAL"})


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate_sequence(steps: list[str]) -> list[str]:
    """
    Check a sequence against all 10 process-logic rules.
    Returns a list of rule names violated; empty list means valid.
    """
    violations = []

    def window(i: int, size: int) -> list[str]:
        return steps[max(0, i - size):i]

    def any_in_window(i: int, size: int, targets: frozenset) -> bool:
        return any(s in targets for s in window(i, size))

    # RULE_DEP_NO_CLEAN
    for i, step in enumerate(steps):
        if step in DEPOSITION_STEPS:
            if not any_in_window(i, 12, CLEAN_STEPS):
                violations.append("RULE_DEP_NO_CLEAN")
                break

    # RULE_METAL_ETCH_NO_LITHO
    for i, step in enumerate(steps):
        if step in METAL_ETCH_STEPS:
            w = window(i, 15)
            has_expose = any(s.startswith("EXPOSE LITHO LEVEL") for s in w)
            has_develop = ("DEVELOP PHOTORESIST" in w) or ("DEVELOP PAD WINDOW" in w)
            if not (has_expose and has_develop):
                violations.append("RULE_METAL_ETCH_NO_LITHO")
                break

    # RULE_ETCH_NO_MASK
    for i, step in enumerate(steps):
        if step in ETCH_STEPS:
            w = window(i, 12)
            has_develop = ("DEVELOP PHOTORESIST" in w) or ("DEVELOP PAD WINDOW" in w)
            if not has_develop:
                violations.append("RULE_ETCH_NO_MASK")
                break

    # RULE_LITHO_LEVEL_SKIP
    align_steps = [
        (i, int(step.split("ALIGN MASK LEVEL ")[1]))
        for i, step in enumerate(steps)
        if step.startswith("ALIGN MASK LEVEL ")
        and step.split("ALIGN MASK LEVEL ")[1].isdigit()
    ]
    for idx in range(1, len(align_steps)):
        _, prev_lvl = align_steps[idx - 1]
        _, curr_lvl = align_steps[idx]
        if curr_lvl > prev_lvl + 1 or curr_lvl < prev_lvl:
            violations.append("RULE_LITHO_LEVEL_SKIP")
            break

    # RULE_IMPLANT_NO_MASK
    for i, step in enumerate(steps):
        if step in IMPLANT_STEPS:
            if not any_in_window(i, 15, IMPLANT_OPENER_STEPS):
                violations.append("RULE_IMPLANT_NO_MASK")
                break

    # RULE_CMP_NO_DEP
    for i, step in enumerate(steps):
        if step in CMP_STEPS:
            if not any_in_window(i, 6, FILL_STEPS):
                violations.append("RULE_CMP_NO_DEP")
                break

    # RULE_PAD_OPEN_BEFORE_DEP
    passivation_dep_idx: Optional[int] = None
    cure_passivation_idx: Optional[int] = None
    for i, step in enumerate(steps):
        if step in ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"):
            passivation_dep_idx = i
        if step == "CURE PASSIVATION":
            cure_passivation_idx = i
        if step in PAD_WINDOW_STEPS:
            if passivation_dep_idx is None or i < passivation_dep_idx:
                violations.append("RULE_PAD_OPEN_BEFORE_DEP")
                break
            elif cure_passivation_idx is None or i < cure_passivation_idx:
                violations.append("RULE_PAD_OPEN_BEFORE_DEP")
                break

    # RULE_TEST_BEFORE_PASSIVATION
    cure_idx: Optional[int] = next(
        (i for i, s in enumerate(steps) if s == "CURE PASSIVATION"), None
    )
    for i, step in enumerate(steps):
        if step in ELECTRICAL_TEST_STEPS:
            if cure_idx is None or i < cure_idx:
                violations.append("RULE_TEST_BEFORE_PASSIVATION")
                break

    # RULE_SHIP_BEFORE_TEST
    ship_idx: Optional[int] = next(
        (i for i, s in enumerate(steps) if s == "SHIP LOT"), None
    )
    sort_idx: Optional[int] = next(
        (i for i, s in enumerate(steps) if s == "WAFER SORT TEST"), None
    )
    if ship_idx is not None and (sort_idx is None or ship_idx < sort_idx):
        violations.append("RULE_SHIP_BEFORE_TEST")

    # RULE_BACKSIDE_BEFORE_PASSIVATION
    for i, step in enumerate(steps):
        if step in BACKSIDE_METAL_STEPS:
            if cure_idx is None or i < cure_idx:
                violations.append("RULE_BACKSIDE_BEFORE_PASSIVATION")
                break

    return violations
