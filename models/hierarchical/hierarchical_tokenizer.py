"""
hierarchical_tokenizer.py — Coarse-to-fine sequence augmentation.

Wraps generate_sequences.py to produce block-annotated sequences.
Each step is tagged with its grammatical block (e.g. PREFIX, PROCESS_CYCLES).
Block boundary tokens [BLK:X] are injected into the flat token stream.

Example output token stream:
  [BLK:PREFIX] RECEIVE WAFER LOT | LOT IDENTIFICATION | ...
  [BLK:PRE_CLEAN] PRE CLEAN WAFER | RCA CLEAN 1 | HF DIP | ...
  [BLK:PROCESS_CYCLES] SPIN COAT PHOTORESIST | ...
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".." / ".." / "data" / "gen"))

from generate_sequences import (
    _gen_prefix,
    _gen_initial_measurements,
    _gen_pre_process_clean,
    _gen_family_prep_mosfet,
    _gen_family_prep_igbt,
    _gen_family_prep_ic,
    _gen_first_oxidation,
    _gen_cycles_mosfet,
    _gen_cycles_igbt,
    _gen_cycles_ic,
    _gen_ild_block,
    _gen_via_block,
    _gen_metal_block,
    _gen_passivation_block,
    _gen_backside_block,
    _gen_final_inspection,
    _gen_test_suite,
    _gen_suffix,
    validate_sequence,
)

_FAMILY_PREP = {
    "mosfet": _gen_family_prep_mosfet,
    "igbt":   _gen_family_prep_igbt,
    "ic":     _gen_family_prep_ic,
}
_CYCLE_GEN = {
    "mosfet": _gen_cycles_mosfet,
    "igbt":   _gen_cycles_igbt,
    "ic":     _gen_cycles_ic,
}
_VIA_LEVEL   = {"mosfet": 3, "igbt": 5, "ic": 3}
_METAL_LEVEL = {"mosfet": 4, "igbt": 6, "ic": 4}

BLOCKS = [
    "PREFIX", "INIT_MEAS", "PRE_CLEAN", "FAMILY_PREP",
    "FIRST_OX", "PROCESS_CYCLES", "ILD", "VIA", "METAL",
    "PASSIVATION", "BACKSIDE", "FINAL_INSP", "TEST_SUITE", "SUFFIX",
]


def generate_labeled_sequence(family: str, rng: random.Random) -> list[tuple[str, str]]:
    """
    Generate one sequence and return (step, block_label) pairs.
    Block labels match the 14 named blocks in generation_rules.md §2.
    """
    family = family.lower()

    def tag(steps: list[str], block: str) -> list[tuple[str, str]]:
        return [(s, block) for s in steps]

    labeled: list[tuple[str, str]] = []
    labeled += tag(_gen_prefix(rng),                                      "PREFIX")
    labeled += tag(_gen_initial_measurements(rng, family),                "INIT_MEAS")
    labeled += tag(_gen_pre_process_clean(rng, family),                   "PRE_CLEAN")
    labeled += tag(_FAMILY_PREP[family](rng),                             "FAMILY_PREP")
    labeled += tag(_gen_first_oxidation(rng, family),                     "FIRST_OX")
    labeled += tag(_CYCLE_GEN[family](rng),                               "PROCESS_CYCLES")
    labeled += tag(_gen_ild_block(rng),                                   "ILD")
    labeled += tag(_gen_via_block(rng, _VIA_LEVEL[family], family),       "VIA")
    labeled += tag(_gen_metal_block(rng, _METAL_LEVEL[family], family),   "METAL")
    labeled += tag(_gen_passivation_block(rng),                           "PASSIVATION")
    labeled += tag(_gen_backside_block(rng, family),                      "BACKSIDE")
    labeled += tag(_gen_final_inspection(rng, family),                    "FINAL_INSP")
    labeled += tag(_gen_test_suite(rng, family),                          "TEST_SUITE")
    labeled += tag(_gen_suffix(rng, family),                              "SUFFIX")
    return labeled


def to_augmented_tokens(labeled: list[tuple[str, str]]) -> list[str]:
    """
    Inject [BLK:X] boundary tokens into the step sequence.
    A boundary token is inserted once at the start of each new block.
    """
    tokens: list[str] = []
    prev_block = None
    for step, block in labeled:
        if block != prev_block:
            tokens.append(f"[BLK:{block}]")
            prev_block = block
        tokens.append(step)
    return tokens


def labeled_to_steps(labeled: list[tuple[str, str]]) -> list[str]:
    return [step for step, _ in labeled]


def labeled_to_block_seq(labeled: list[tuple[str, str]]) -> list[str]:
    """Parallel block label list — one entry per step."""
    return [block for _, block in labeled]


if __name__ == "__main__":
    import collections

    rng = random.Random(42)
    for family in ("mosfet", "igbt", "ic"):
        labeled = generate_labeled_sequence(family, rng)
        tokens  = to_augmented_tokens(labeled)
        block_counts = collections.Counter(b for _, b in labeled)

        print(f"\n=== {family.upper()} ({len(labeled)} steps, {len(tokens)} tokens total) ===")
        for block in BLOCKS:
            if block in block_counts:
                print(f"  {block:<18} {block_counts[block]:>3} steps")

        print("\nFirst 30 tokens:")
        print(" | ".join(tokens[:30]))

        violations = validate_sequence(labeled_to_steps(labeled))
        print(f"Validator: {'OK' if not violations else f'{len(violations)} VIOLATIONS'}")
