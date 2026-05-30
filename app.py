"""
Gradio frontend for process sequence prediction.

Run:  python app.py
Then: open http://localhost:7860
"""

import sys
import os
import glob
import random
import csv
from collections import defaultdict

# Add model code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hpc', 'scp'))

import gradio as gr
import torch
from model import Encoder, Decoder, Seq2Seq
from utils import PAD, SOS, EOS, IGBT_TOK, IC_TOK, Vocab
from validator import validate_sequence
from markov import MarkovChain

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'hpc', 'scp', '.save')
DEVICE = torch.device('cpu')

_loaded_model = None
_loaded_vocab = None
_loaded_ckpt_name = None
_loaded_markov = None

MARKOV_LABEL = '⚡ Markov Chain (baseline)'


def list_checkpoints():
    """Find all .pt/.json model files in the checkpoint directory."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    pts = glob.glob(os.path.join(CHECKPOINT_DIR, '*.pt'))
    choices = sorted([os.path.basename(p) for p in pts])
    # Add Markov baseline if available
    if os.path.isfile(os.path.join(CHECKPOINT_DIR, 'markov.json')):
        choices.insert(0, MARKOV_LABEL)
    return choices


def load_markov():
    global _loaded_markov
    if _loaded_markov is None:
        path = os.path.join(CHECKPOINT_DIR, 'markov.json')
        _loaded_markov = MarkovChain.load(path)
    return _loaded_markov


def load_model(ckpt_name):
    global _loaded_model, _loaded_vocab, _loaded_ckpt_name
    if ckpt_name == _loaded_ckpt_name and _loaded_model is not None:
        return _loaded_model, _loaded_vocab

    path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    args = ckpt['args']

    vocab = Vocab.__new__(Vocab)
    vocab.itos = ckpt['vocab_itos']
    vocab.stoi = ckpt['vocab_stoi']

    vocab_size = len(vocab)
    encoder = Encoder(vocab_size, args['embed_size'], args['hidden_size'],
                      n_layers=2, dropout=0.0)
    decoder = Decoder(args['embed_size'], args['hidden_size'], vocab_size,
                      n_layers=args.get('dec_layers', 1), dropout=0.0)
    model = Seq2Seq(encoder, decoder, tie_embeddings=True).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    _loaded_model = model
    _loaded_vocab = vocab
    _loaded_ckpt_name = ckpt_name
    return model, vocab


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_input(text):
    """Parse input text into a list of steps.
    Accepts:
      - CSV format (SEQUENCE_ID,STEP with header)
      - One step per line
      - Arrow-separated: STEP1 -> STEP2 -> ...
    """
    lines = text.strip().splitlines()
    if not lines:
        return []

    # Detect CSV format (has comma and header-like first line)
    if ',' in lines[0] and ('SEQUENCE_ID' in lines[0].upper() or 'STEP' in lines[0].upper()):
        steps = []
        for line in lines[1:]:
            parts = line.split(',', 1)
            if len(parts) >= 2:
                steps.append(parts[1].strip())
        return steps

    # Detect arrow-separated
    if ' -> ' in text or ' → ' in text:
        parts = text.replace('→', '->').split('->')
        return [s.strip() for s in parts if s.strip()]

    # Default: one step per line
    return [line.strip() for line in lines if line.strip()]


def detect_family(steps):
    joined = ' '.join(steps).upper()
    if 'P BODY' in joined or 'N BUFFER' in joined or 'CHANNEL STOP' in joined:
        return 'igbt'
    if 'EPITAXIAL WAFER CHECK' in joined or 'FIELD OXIDE' in joined:
        return 'igbt'
    if 'SUBSTRATE CHECK' in joined or 'EPITAXIAL DEPOSITION' in joined:
        return 'igbt'
    return 'ic'


def encode_prefix(steps, family, vocab):
    fam_map = {'igbt': IGBT_TOK, 'ic': IC_TOK, 'mosfet': IGBT_TOK}
    fam_tok = fam_map.get(family, IC_TOK)
    encoded = [fam_tok] + vocab.encode(steps)
    return torch.tensor(encoded, dtype=torch.long)


def format_violations(violations):
    if not violations:
        return "✅ No rule violations detected"
    lines = [f"⚠️ {len(violations)} rule(s) violated:"]
    for v in violations:
        lines.append(f"  • {v}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------

def predict_next_step(input_text, checkpoint):
    if not input_text.strip():
        return "No input provided.", ""
    if not checkpoint:
        return "No checkpoint selected.", ""

    steps = parse_input(input_text)
    if not steps:
        return "Could not parse any steps from input.", ""

    # --- Markov chain path ---
    if checkpoint == MARKOV_LABEL:
        mc = load_markov()
        preds = mc.predict_next(steps[-1], top_k=10)
        lines = [f"Model: Markov Chain (baseline)",
                 f"Prefix length: {len(steps)} steps",
                 f"Last step: {steps[-1]}",
                 "", "─── Top-10 Next Step Predictions ───", ""]
        for rank, (prob, step) in enumerate(preds, 1):
            bar = '█' * int(prob * 40)
            lines.append(f"  {rank:2d}. {step:45s} {prob:.4f} {bar}")
        if not preds:
            lines.append("  (no transitions observed for this step)")
        input_violations = validate_sequence(steps)
        return '\n'.join(lines), format_violations(input_violations)

    # --- Seq2Seq path ---
    model, vocab = load_model(checkpoint)
    family = detect_family(steps)
    src = encode_prefix(steps, family, vocab).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        encoder_output, encoder_hidden = model.encoder(src)
        hidden = model.decoder.init_hidden(encoder_hidden)
        sos_input = torch.tensor([SOS], device=DEVICE)
        output, _, _ = model.decoder(sos_input, hidden, encoder_output)
        probs = output.exp().squeeze(0)
        top_probs, top_indices = probs.topk(10)

    lines = [f"Model: Seq2Seq ({checkpoint})",
             f"Family detected: {family.upper()}",
             f"Prefix length: {len(steps)} steps",
             f"Last step: {steps[-1]}",
             "", "─── Top-10 Next Step Predictions ───", ""]
    for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), 1):
        token = vocab.itos[idx]
        if idx >= 6:
            bar = '█' * int(prob * 40)
            lines.append(f"  {rank:2d}. {token:45s} {prob:.4f} {bar}")

    # Input validation
    input_violations = validate_sequence(steps)
    input_info = format_violations(input_violations)

    return '\n'.join(lines), input_info


def predict_until_end(input_text, checkpoint):
    if not input_text.strip():
        return "No input provided.", "", ""
    if not checkpoint:
        return "No checkpoint selected.", "", ""

    steps = parse_input(input_text)
    if not steps:
        return "Could not parse any steps from input.", "", ""

    # --- Markov chain path ---
    if checkpoint == MARKOV_LABEL:
        mc = load_markov()
        completion = mc.predict_completion(steps, max_steps=200)
        lines = [f"Model: Markov Chain (baseline)",
                 f"Prefix length: {len(steps)} steps",
                 f"Generated: {len(completion)} steps",
                 "", "─── Completion ───", ""]
        for i, s in enumerate(completion, 1):
            lines.append(f"  {i:3d}. {s}")
        if not completion:
            lines.append("  (no completion possible from last step)")
        input_violations = validate_sequence(steps)
        full_seq = steps + completion
        output_violations = validate_sequence(full_seq)
        return '\n'.join(lines), format_violations(input_violations), format_violations(output_violations)

    # --- Seq2Seq path ---
    model, vocab = load_model(checkpoint)
    family = detect_family(steps)
    src = encode_prefix(steps, family, vocab).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        encoder_output, encoder_hidden = model.encoder(src)
        hidden = model.decoder.init_hidden(encoder_hidden)
        input_tok = torch.tensor([SOS], device=DEVICE)
        completion = []
        for _ in range(200):
            output, hidden, _ = model.decoder(input_tok, hidden, encoder_output)
            top1 = output.argmax(1)
            tok_idx = top1.item()
            if tok_idx == EOS:
                break
            if tok_idx >= 6:
                completion.append(vocab.itos[tok_idx])
            input_tok = top1

    lines = [f"Family detected: {family.upper()}",
             f"Prefix length: {len(steps)} steps",
             f"Generated: {len(completion)} steps",
             "", "─── Completion ───", ""]
    for i, s in enumerate(completion, 1):
        lines.append(f"  {i:3d}. {s}")
    if not completion:
        lines.append("  (model produced EOS immediately)")

    # Validation
    input_violations = validate_sequence(steps)
    full_seq = steps + completion
    output_violations = validate_sequence(full_seq)

    input_info = format_violations(input_violations)
    output_info = format_violations(output_violations)

    return '\n'.join(lines), input_info, output_info


def run_test_evaluation(checkpoint):
    """Placeholder — not yet implemented."""
    return "🚧 Test evaluation not yet implemented."


def find_anomaly(input_text, checkpoint):
    """Placeholder — not yet implemented."""
    return "🚧 Anomaly detection not yet implemented.", "", ""


def handle_file_upload(file):
    """Read uploaded CSV file and return its content as text."""
    if file is None:
        return ""
    with open(file.name, 'r') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app():
    checkpoints = list_checkpoints()
    default_ckpt = 'best.pt' if 'best.pt' in checkpoints else (checkpoints[0] if checkpoints else None)

    with gr.Blocks(title="Process Sequence Predictor") as app:
        gr.Markdown("# 🔬 Semiconductor Process Sequence Predictor")
        gr.Markdown("Enter process steps (CSV format, one-per-line, or arrow-separated) and predict next steps or full completions.")

        with gr.Row():
            checkpoint_dd = gr.Dropdown(
                choices=checkpoints,
                value=default_ckpt,
                label="Model Checkpoint",
                scale=1
            )

        with gr.Row(equal_height=True):
            # LEFT: Input
            with gr.Column(scale=4):
                input_text = gr.Textbox(
                    label="Input Sequence",
                    placeholder="RECEIVE WAFER LOT\nLOT IDENTIFICATION\nINITIAL WAFER INSPECTION\n...\n\nOr paste CSV (SEQUENCE_ID,STEP) or use arrows: STEP1 -> STEP2 -> ...",
                    lines=20,
                    max_lines=40,
                )
                file_upload = gr.File(
                    label="Upload CSV",
                    file_types=['.csv', '.txt'],
                    type='filepath'
                )
                input_violations = gr.Textbox(
                    label="Input Validation (10 process rules)",
                    lines=4,
                    interactive=False
                )

            # MIDDLE: Buttons
            with gr.Column(scale=1, min_width=180):
                gr.Markdown("### Actions")
                btn_next = gr.Button("🎯 Predict Next Step", variant="primary", size="lg")
                btn_end = gr.Button("🔄 Predict Until End", variant="primary", size="lg")
                btn_test = gr.Button("📊 Test on Dataset", variant="primary", size="lg")
                btn_anomaly = gr.Button("🔍 Find Anomaly", variant="secondary", size="lg", interactive=False)

            # RIGHT: Output
            with gr.Column(scale=4):
                output_text = gr.Textbox(
                    label="Output",
                    lines=20,
                    max_lines=40,
                    interactive=False
                )
                output_violations = gr.Textbox(
                    label="Output Validation (completed sequence rules)",
                    lines=4,
                    interactive=False
                )

        # --- Event handlers ---
        file_upload.change(
            fn=handle_file_upload,
            inputs=[file_upload],
            outputs=[input_text]
        )

        btn_next.click(
            fn=predict_next_step,
            inputs=[input_text, checkpoint_dd],
            outputs=[output_text, input_violations]
        )

        btn_end.click(
            fn=predict_until_end,
            inputs=[input_text, checkpoint_dd],
            outputs=[output_text, input_violations, output_violations]
        )

        btn_test.click(
            fn=run_test_evaluation,
            inputs=[checkpoint_dd],
            outputs=[output_text]
        )

        btn_anomaly.click(
            fn=find_anomaly,
            inputs=[input_text, checkpoint_dd],
            outputs=[output_text, input_violations, output_violations]
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
