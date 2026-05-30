"""
Gradio frontend for process sequence prediction.

Run:  python app.py
Then: open http://localhost:7860
"""

import csv
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "models"))

import gradio as gr
from models.infer import (
    available_models,
    missing_models,
    load_model,
    run_task1,
    run_task2,
    run_task3,
    self_benchmark_task1,
    self_benchmark_task2,
)
from models.seq2seq.validator import validate_sequence

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT          = Path(__file__).parent
_EVAL_VALID    = _ROOT / "data" / "participant_files" / "eval_input_valid.csv"
_EVAL_ANOMALY  = _ROOT / "data" / "participant_files" / "eval_input_anomaly.csv"

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def _get_model(name: str):
    if name not in _model_cache:
        _model_cache[name] = load_model(name)
    return _model_cache[name]


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def _parse_input(text: str) -> list[str]:
    lines = text.strip().splitlines()
    if not lines:
        return []
    if "," in lines[0] and any(k in lines[0].upper() for k in ("SEQUENCE_ID", "STEP")):
        steps = []
        for line in lines[1:]:
            parts = line.split(",", 1)
            if len(parts) >= 2:
                steps.append(parts[1].strip())
        return steps
    if " -> " in text or " → " in text:
        return [s.strip() for s in text.replace("→", "->").split("->") if s.strip()]
    return [line.strip() for line in lines if line.strip()]


def _fmt_violations(violations: list[str]) -> str:
    if not violations:
        return "No rule violations detected"
    lines = [f"{len(violations)} rule(s) violated:"]
    for v in violations:
        lines.append(f"  - {v}")
    return "\n".join(lines)


def _handle_upload(file) -> str:
    if file is None:
        return ""
    with open(file, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Interactive tab handlers
# ---------------------------------------------------------------------------

def predict_next(input_text: str, model_name: str) -> tuple[str, str]:
    if not input_text.strip():
        return "No input provided.", ""
    if not model_name:
        return "No model selected.", ""
    steps = _parse_input(input_text)
    if not steps:
        return "Could not parse any steps.", ""

    model = _get_model(model_name)
    top5  = model.predict_next(steps, top_k=10)

    lines = [
        f"Model   : {model_name}",
        f"Prefix  : {len(steps)} steps  (last: {steps[-1]})",
        "",
        "Top-10 Next Step Predictions",
        "─" * 52,
    ]
    for rank, step in enumerate(top5, 1):
        lines.append(f"  {rank:2d}.  {step}")

    viols = validate_sequence(steps)
    return "\n".join(lines), _fmt_violations(viols)


def predict_completion(input_text: str, model_name: str) -> tuple[str, str, str]:
    if not input_text.strip():
        return "No input provided.", "", ""
    if not model_name:
        return "No model selected.", "", ""
    steps = _parse_input(input_text)
    if not steps:
        return "Could not parse any steps.", "", ""

    model      = _get_model(model_name)
    completion = model.predict_completion(steps)

    lines = [
        f"Model     : {model_name}",
        f"Prefix    : {len(steps)} steps",
        f"Generated : {len(completion)} steps",
        "",
        "Completion",
        "─" * 52,
    ]
    for i, s in enumerate(completion, 1):
        lines.append(f"  {i:3d}.  {s}")
    if not completion:
        lines.append("  (no completion produced)")

    in_viols  = _fmt_violations(validate_sequence(steps))
    out_viols = _fmt_violations(validate_sequence(steps + completion))
    return "\n".join(lines), in_viols, out_viols


def detect_anomaly_interactive(input_text: str, model_name: str) -> tuple[str, str]:
    if not input_text.strip():
        return "No input provided.", ""
    if not model_name:
        return "No model selected.", ""
    steps = _parse_input(input_text)
    if not steps:
        return "Could not parse any steps.", ""

    model    = _get_model(model_name)
    is_valid, score, rule = model.detect_anomaly(steps)

    verdict = "VALID" if is_valid else "INVALID"
    lines   = [
        f"Model   : {model_name}",
        f"Steps   : {len(steps)}",
        f"Verdict : {verdict}",
        f"Score   : {score:.4f}  (higher = more likely valid)",
    ]
    if rule:
        lines.append(f"Rule    : {rule}")

    viols = validate_sequence(steps)
    return "\n".join(lines), _fmt_violations(viols)


# ---------------------------------------------------------------------------
# Benchmark tab handlers
# ---------------------------------------------------------------------------

def _check_eval_files() -> tuple[bool, bool]:
    return _EVAL_VALID.exists(), _EVAL_ANOMALY.exists()


def run_benchmark(model_name: str, task: str) -> tuple[str, str]:
    """Returns (scores_text, download_csv_str)."""
    if not model_name:
        return "No model selected.", ""

    valid_ok, anomaly_ok = _check_eval_files()

    model = _get_model(model_name)
    log_lines = []

    def progress(done, total):
        log_lines.append(f"  {done}/{total} sequences processed...")

    if task == "Task 1 — Next-Step Prediction":
        if not valid_ok:
            return f"Eval file not found:\n  {_EVAL_VALID}", ""
        pred_csv, scores = run_task1(model, _EVAL_VALID, progress_cb=progress)
        return scores + "\n\n" + "\n".join(log_lines), pred_csv

    elif task == "Task 2 — Sequence Completion":
        if not valid_ok:
            return f"Eval file not found:\n  {_EVAL_VALID}", ""
        pred_csv, scores = run_task2(model, _EVAL_VALID, progress_cb=progress)
        return scores + "\n\n" + "\n".join(log_lines), pred_csv

    elif task == "Task 3 — Anomaly Detection":
        if not anomaly_ok:
            return f"Eval file not found:\n  {_EVAL_ANOMALY}", ""
        pred_csv, scores = run_task3(model, _EVAL_ANOMALY, progress_cb=progress)
        return scores + "\n\n" + "\n".join(log_lines), pred_csv

    return "Unknown task.", ""


def run_self_benchmark(model_name: str, task: str, n_seqs: int) -> tuple[str, str]:
    if not model_name:
        return "No model selected.", ""
    model = _get_model(model_name)
    log = []

    def progress(done, total):
        log.append(f"  {done}/{total} sequences processed...")

    if "Task 1" in task:
        scores, pred_csv = self_benchmark_task1(model, progress_cb=progress,
                                                max_per_family=n_seqs)
    elif "Task 2" in task:
        scores, pred_csv = self_benchmark_task2(model, progress_cb=progress,
                                                max_per_family=n_seqs)
    else:
        return "Self-benchmark not available for Task 3 (uses rule-checker by definition).", ""

    return scores + "\n\n" + "\n".join(log), pred_csv



def save_predictions(pred_csv: str, task: str) -> str:
    """Write pred_csv to a temp file and return the path for gr.File download."""
    if not pred_csv.strip():
        return None
    slug = task.split("—")[0].strip().lower().replace(" ", "_")
    out_path = _ROOT / f"predictions_{slug}.csv"
    out_path.write_text(pred_csv)
    return str(out_path)


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_app():
    models = available_models()
    default = models[0] if models else None

    if not models:
        print("[WARNING] No trained model checkpoints found. "
              "Train models first or place checkpoints in the expected paths.")

    valid_ok, anomaly_ok = _check_eval_files()
    eval_status = []
    if valid_ok:
        eval_status.append(f"eval_input_valid.csv found ({_EVAL_VALID})")
    else:
        eval_status.append(f"eval_input_valid.csv NOT FOUND (expected at {_EVAL_VALID})")
    if anomaly_ok:
        eval_status.append(f"eval_input_anomaly.csv found ({_EVAL_ANOMALY})")
    else:
        eval_status.append(f"eval_input_anomaly.csv NOT FOUND (expected at {_EVAL_ANOMALY})")

    with gr.Blocks(title="Process Sequence Predictor") as app:
        gr.Markdown("# Semiconductor Process Sequence Predictor")
        missing = missing_models()
        missing_note = ""
        if missing:
            missing_note = "\n\n**Not yet trained:** " + "  |  ".join(
                f"`{n}` (expected `{p}`)" for n, p in missing.items()
            )
        gr.Markdown(
            "**Models available:** " + (", ".join(f"`{m}`" for m in models) if models else "_none — train first_") +
            missing_note +
            "\n\n**Eval files:** " + " | ".join(eval_status)
        )

        # ── Tab 1: Self-Benchmark ────────────────────────────────────────────
        with gr.Tab("Self-Benchmark"):
            gr.Markdown(
                "Scores against `*_extended.csv` sequences — independent from "
                "`*_variants.csv` training data, so no leakage for variant-trained models. "
                "Falls back to variants if extended not found.\n\n"
                "**Task 1:** samples every ~5th position between 10%–90% of each sequence, "
                "predicts top-5 next steps, checks if the true next step appears at rank 1, 3, or 5. "
                "MRR = mean 1/rank of the correct answer.\n\n"
                "**Task 2:** cuts each sequence at 60% and 80%, runs greedy completion to EOS, "
                "compares predicted vs ground-truth suffix. "
                "NED = Levenshtein / max(|pred|,|ref|) (lower is better). "
                "Block Accuracy maps steps to 10 coarse phases before comparing.\n\n"
                "Sequences per family controls sample size — lower is faster."
            )
            with gr.Row():
                self_task_dd = gr.Dropdown(
                    choices=["Task 1 — Next-Step Prediction",
                             "Task 2 — Sequence Completion"],
                    value="Task 1 — Next-Step Prediction",
                    label="Task", scale=2,
                )
                self_model_dd = gr.Dropdown(
                    choices=models, value=default, label="Model", scale=1,
                )
            with gr.Row():
                n_seqs_slider = gr.Slider(
                    minimum=5, maximum=100, value=50, step=5,
                    label="Sequences per family  (Task 1: 50 · Task 2: 15 recommended)",
                    scale=3,
                )
                self_run_btn = gr.Button("Run", variant="primary", scale=1)
            self_scores_box = gr.Textbox(
                label="Results", lines=20, max_lines=40, interactive=False,
            )
            with gr.Row():
                self_pred_state = gr.State("")
                self_dl_btn = gr.Button("Download Predictions CSV", scale=1)
                self_dl_file = gr.File(label="Predictions", scale=3)
            self_run_btn.click(
                fn=run_self_benchmark,
                inputs=[self_model_dd, self_task_dd, n_seqs_slider],
                outputs=[self_scores_box, self_pred_state],
            )
            self_dl_btn.click(
                fn=save_predictions,
                inputs=[self_pred_state, self_task_dd],
                outputs=[self_dl_file],
            )

        # ── Tab 2: Official ──────────────────────────────────────────────────
        with gr.Tab("Official"):
            gr.Markdown(
                "Generates submission CSVs from the organiser-provided eval files. "
                "Ground-truth labels are not included — download the CSV and submit to the organizers.\n\n"
                "| Task | Input | Model does | Metrics (scored by organizers) |\n"
                "|---|---|---|---|\n"
                "| **1 — Next-Step** | `eval_input_valid.csv` (600 partial seqs) | Predict top-5 next steps | Top-1/3/5 Accuracy, MRR |\n"
                "| **2 — Completion** | `eval_input_valid.csv` | Greedy decode to EOS | NED, Exact Match, Token Acc, Block Acc |\n"
                "| **3 — Anomaly** | `eval_input_anomaly.csv` (987 seqs) | Classify valid/invalid + confidence score | Binary Acc, F1, ROC-AUC, Rule Attribution |\n"
            )
            with gr.Row():
                task_dd = gr.Dropdown(
                    choices=[
                        "Task 1 — Next-Step Prediction",
                        "Task 2 — Sequence Completion",
                        "Task 3 — Anomaly Detection",
                    ],
                    value="Task 1 — Next-Step Prediction",
                    label="Task", scale=2,
                )
                bench_model_dd = gr.Dropdown(
                    choices=models, value=default, label="Model", scale=1,
                )
                run_btn = gr.Button("Run", variant="primary", scale=1)
            scores_box = gr.Textbox(
                label="Results", lines=20, max_lines=40, interactive=False,
            )
            with gr.Row():
                pred_state = gr.State("")
                dl_btn = gr.Button("Download Predictions CSV", scale=1)
                dl_file = gr.File(label="Predictions", scale=3)
            run_btn.click(
                fn=run_benchmark,
                inputs=[bench_model_dd, task_dd],
                outputs=[scores_box, pred_state],
            )
            dl_btn.click(
                fn=save_predictions,
                inputs=[pred_state, task_dd],
                outputs=[dl_file],
            )

        # ── Tab 3: Custom (interactive predict) ──────────────────────────────
        with gr.Tab("Custom"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    input_box = gr.Textbox(
                        label="Input Sequence",
                        placeholder=(
                            "One step per line, or arrow-separated:\n"
                            "RECEIVE WAFER LOT\nLOT IDENTIFICATION\n...\n\n"
                            "Or paste CSV (SEQUENCE_ID,STEP)"
                        ),
                        lines=18, max_lines=40,
                    )
                    file_up = gr.File(label="Upload CSV", file_types=[".csv", ".txt"],
                                      type="filepath")
                    input_viols = gr.Textbox(label="Input Validation", lines=3,
                                             interactive=False)

                with gr.Column(scale=1, min_width=180):
                    gr.Markdown("### Actions")
                    custom_model_dd = gr.Dropdown(
                        choices=models, value=default, label="Model",
                    )
                    btn_next  = gr.Button("Predict Next Step", variant="primary",  size="lg")
                    btn_end   = gr.Button("Predict Until End", variant="primary",  size="lg")
                    btn_anom  = gr.Button("Detect Anomaly",    variant="secondary", size="lg")

                with gr.Column(scale=4):
                    output_box   = gr.Textbox(label="Output", lines=18, max_lines=40,
                                              interactive=False)
                    output_viols = gr.Textbox(label="Output Validation", lines=3,
                                              interactive=False)

            file_up.change(fn=_handle_upload, inputs=[file_up], outputs=[input_box])
            btn_next.click(fn=predict_next,
                           inputs=[input_box, custom_model_dd],
                           outputs=[output_box, input_viols])
            btn_end.click(fn=predict_completion,
                          inputs=[input_box, custom_model_dd],
                          outputs=[output_box, input_viols, output_viols])
            btn_anom.click(fn=detect_anomaly_interactive,
                           inputs=[input_box, custom_model_dd],
                           outputs=[output_box, input_viols])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
