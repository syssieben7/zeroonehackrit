#!/usr/bin/env python3
"""
infer.py — Run inference with a fine-tuned T5 checkpoint.

Three modes:
  1. Interactive  — type a partial sequence, get predictions live
  2. Eval         — read hackathon eval CSVs, write submission CSVs
  3. Single       — pass a sequence via CLI flags for quick testing

Usage examples
--------------
Interactive mode:
    python infer.py --checkpoint ../checkpoints/best --interactive

Eval mode (generates all 3 submission files):
    python infer.py --checkpoint ../checkpoints/best \
        --eval_valid  ../data/eval/eval_input_valid.csv \
        --eval_anomaly ../data/eval/eval_input_anomaly.csv \
        --out_dir ../submissions

Single next-step prediction:
    python infer.py --checkpoint ../checkpoints/best \
        --family IC --task next_step \
        --prefix "RECEIVE WAFER LOT | LOT IDENTIFICATION | PRE CLEAN INSPECTION"
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

SEP = " | "
MAX_INPUT  = 512   # max tokens fed to encoder
MAX_OUTPUT = 256   # max tokens the decoder can generate


# ---------------------------------------------------------------------------
# Step 1 — Load the model
# Reads the tokenizer and weights from the checkpoint directory saved by
# train.py. The model is moved to GPU if available, otherwise CPU.
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {checkpoint_dir} on {device}...")
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_dir)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir).to(device)
    model.eval()  # disable dropout — we're predicting, not training
    print("Model loaded.\n")
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Step 2 — Build input strings
# The task prefix is how the model knows what kind of output to produce.
# The family tells it which product line (ic / mosfet / igbt).
# ---------------------------------------------------------------------------

def _input_next_step(family: str, steps: list[str]) -> str:
    return f"next step {family.lower()}: {SEP.join(steps)}"

def _input_complete(family: str, steps: list[str]) -> str:
    return f"complete {family.lower()}: {SEP.join(steps)}"

def _input_validate(family: str, steps: list[str]) -> str:
    return f"validate {family.lower()}: {SEP.join(steps)}"


# ---------------------------------------------------------------------------
# Step 3 — Task 1: Next-step prediction
# Uses beam search to generate 5 candidate next steps ranked by probability.
# num_beams=10 explores 10 paths; num_return_sequences=5 returns the best 5.
# ---------------------------------------------------------------------------

def predict_next_step(
    model, tokenizer, device,
    family: str, steps: list[str],
    top_k: int = 5,
) -> list[str]:
    inp = _input_next_step(family, steps)
    enc = tokenizer(
        inp,
        max_length=MAX_INPUT,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=20,        # a step name is short
            num_beams=max(10, top_k),
            num_return_sequences=top_k,
            early_stopping=True,
        )

    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # strip whitespace and deduplicate while preserving order
    seen = set()
    cleaned = []
    for p in predictions:
        p = p.strip().upper()
        if p not in seen:
            seen.add(p)
            cleaned.append(p)
    return cleaned[:top_k]


# ---------------------------------------------------------------------------
# Step 4 — Task 2: Sequence completion
# Generates the remaining steps after the given prefix.
# Uses greedy decoding (num_beams=1) since we want one long coherent sequence.
# The output is pipe-separated, matching the training format.
# ---------------------------------------------------------------------------

def predict_completion(
    model, tokenizer, device,
    family: str, steps: list[str],
) -> list[str]:
    inp = _input_complete(family, steps)
    enc = tokenizer(
        inp,
        max_length=MAX_INPUT,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **enc,
            max_new_tokens=400,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.5,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # split the pipe-separated output back into a list of steps
    completed_steps = [s.strip().upper() for s in decoded.split("|") if s.strip()]
    return completed_steps


# ---------------------------------------------------------------------------
# Step 5 — Task 3: Anomaly detection
# The model outputs "valid" or "invalid RULE_X".
# We also extract a confidence score by comparing the log-probability of
# generating "valid" vs "invalid" as the first token — this gives the SCORE
# column required in the submission (1.0 = very confident it's valid).
# ---------------------------------------------------------------------------

def predict_validity(
    model, tokenizer, device,
    family: str, steps: list[str],
) -> tuple[int, float, str]:
    inp = _input_validate(family, steps)
    enc = tokenizer(
        inp,
        max_length=MAX_INPUT,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # generate the full output text
        output = model.generate(
            **enc,
            max_new_tokens=20,
            num_beams=4,
            early_stopping=True,
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()

        # get log-probabilities of the first generated token to build a score
        logits = model(
            **enc,
            decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id]], device=device),
        ).logits[0, 0]
        probs = torch.softmax(logits, dim=-1)

    # token IDs for "valid" and "invalid"
    valid_id   = tokenizer.encode("valid",   add_special_tokens=False)[0]
    invalid_id = tokenizer.encode("invalid", add_special_tokens=False)[0]
    p_valid   = probs[valid_id].item()
    p_invalid = probs[invalid_id].item()
    total = p_valid + p_invalid + 1e-9
    score = p_valid / total   # probability that sequence is valid, range [0, 1]

    is_valid = 1 if decoded.startswith("valid") else 0
    rule = ""
    if not is_valid and " " in decoded:
        rule = decoded.split(" ", 1)[1].upper()  # e.g. "RULE_DEP_NO_CLEAN"

    return is_valid, round(score, 4), rule


# ---------------------------------------------------------------------------
# Step 6 — Eval mode
# Reads the two hackathon eval CSVs and writes the three submission CSVs.
# eval_input_valid.csv   → task1_submission.csv + task2_submission.csv
# eval_input_anomaly.csv → task3_submission.csv
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, device, args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Tasks 1 & 2 ---
    if args.eval_valid:
        path = Path(args.eval_valid)
        task1_rows, task2_rows = [], []

        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                eid    = row["EXAMPLE_ID"].strip()
                family = row["FAMILY"].strip()
                steps  = [s.strip() for s in row["PARTIAL_SEQUENCE"].split("|") if s.strip()]

                print(f"  [{eid}] next_step + completion ...")

                top5 = predict_next_step(model, tokenizer, device, family, steps)
                while len(top5) < 5:
                    top5.append("")
                task1_rows.append([eid] + top5[:5])

                completion = predict_completion(model, tokenizer, device, family, steps)
                task2_rows.append([eid, SEP.join(completion)])

        with (out_dir / "task1_submission.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
            w.writerows(task1_rows)

        with (out_dir / "task2_submission.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
            w.writerows(task2_rows)

        print(f"Wrote task1_submission.csv and task2_submission.csv to {out_dir}/")

    # --- Task 3 ---
    if args.eval_anomaly:
        path = Path(args.eval_anomaly)
        task3_rows = []

        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                eid    = row["EXAMPLE_ID"].strip()
                family = row["FAMILY"].strip()
                steps  = [s.strip() for s in row["SEQUENCE"].split("|") if s.strip()]

                print(f"  [{eid}] validate ...")
                is_valid, score, rule = predict_validity(model, tokenizer, device, family, steps)
                task3_rows.append([eid, is_valid, score, rule])

        with (out_dir / "task3_submission.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
            w.writerows(task3_rows)

        print(f"Wrote task3_submission.csv to {out_dir}/")


# ---------------------------------------------------------------------------
# Step 7 — Interactive mode
# Loop where you type a partial sequence and the model responds live.
# Useful for quickly testing what the model has learned.
# ---------------------------------------------------------------------------

def run_interactive(model, tokenizer, device):
    print("Interactive mode. Type 'quit' to exit.")
    print("Format: FAMILY | step1 | step2 | ...\n")

    while True:
        try:
            raw = input("Sequence> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw.lower() in ("quit", "exit", "q"):
            break
        if not raw:
            continue

        parts = [p.strip() for p in raw.split("|")]
        family = parts[0].upper()
        steps  = parts[1:]

        if not steps:
            print("  Please include at least one step after the family.\n")
            continue

        print(f"\n  Next step (top 5):")
        for i, s in enumerate(predict_next_step(model, tokenizer, device, family, steps), 1):
            print(f"    {i}. {s}")

        print(f"\n  Completion:")
        comp = predict_completion(model, tokenizer, device, family, steps)
        print(f"    {SEP.join(comp)}")

        is_valid, score, rule = predict_validity(model, tokenizer, device, family, steps + comp)
        verdict = "VALID" if is_valid else f"INVALID ({rule})"
        print(f"\n  Validity: {verdict} (score={score})\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint directory (e.g. ../checkpoints/best)")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive prompt")
    parser.add_argument("--eval_valid",   default=None,
                        help="Path to eval_input_valid.csv (tasks 1 & 2)")
    parser.add_argument("--eval_anomaly", default=None,
                        help="Path to eval_input_anomaly.csv (task 3)")
    parser.add_argument("--out_dir", default="../submissions",
                        help="Directory to write submission CSVs")
    parser.add_argument("--family", default="IC",
                        help="Family for single prediction (IC/MOSFET/IGBT)")
    parser.add_argument("--task",   default="next_step",
                        choices=["next_step", "completion", "validate"],
                        help="Task for single prediction mode")
    parser.add_argument("--prefix", default=None,
                        help="Pipe-separated steps for single prediction")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.checkpoint)

    if args.interactive:
        run_interactive(model, tokenizer, device)

    elif args.eval_valid or args.eval_anomaly:
        run_eval(model, tokenizer, device, args)

    elif args.prefix:
        steps = [s.strip() for s in args.prefix.split("|")]
        if args.task == "next_step":
            print("Top 5 next steps:")
            for i, s in enumerate(predict_next_step(model, tokenizer, device, args.family, steps), 1):
                print(f"  {i}. {s}")
        elif args.task == "completion":
            comp = predict_completion(model, tokenizer, device, args.family, steps)
            print("Completion:")
            print(f"  {SEP.join(comp)}")
        elif args.task == "validate":
            is_valid, score, rule = predict_validity(model, tokenizer, device, args.family, steps)
            verdict = "VALID" if is_valid else f"INVALID ({rule})"
            print(f"Validity: {verdict} (score={score})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
