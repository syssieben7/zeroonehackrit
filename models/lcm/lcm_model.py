"""
Process-LCM: Large Concept Model adapted for semiconductor process sequences.

Instead of SONAR sentence embeddings, we learn step embeddings in a continuous
space and train a causal Transformer to predict the next step embedding (MSE).

At inference:
  - Next-step: cosine similarity of predicted embedding vs. codebook → top-k
  - Completion: autoregressive greedy decoding via nearest-neighbor in codebook
  - Anomaly: high prediction error at a step → rule violation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class ProcessLCM(nn.Module):
    """
    Process-LCM: Causal Transformer predicting next step embeddings.

    Architecture:
        1. Frontend: step embedding lookup + positional encoding
        2. Core: Causal Transformer decoder (N layers)
        3. Postnet: Linear projection to predict next embedding
    """

    def __init__(self, vocab_size, embed_dim=256, n_heads=8, n_layers=6,
                 dim_feedforward=1024, dropout=0.1, max_seq_len=300):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Frontend: step codebook + positional encoding
        self.step_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len=max(max_seq_len, 512))
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Core: Causal Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable training)
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Postnet: predict next step embedding
        self.postnet = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Output head for discrete classification (auxiliary loss)
        self.classifier = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    def _causal_mask(self, seq_len, device):
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, step_ids, padding_mask=None):
        """
        Forward pass: predict next step embeddings for all positions.

        Args:
            step_ids: (batch, seq_len) token indices
            padding_mask: (batch, seq_len) True where padded

        Returns:
            predicted_embeddings: (batch, seq_len, embed_dim)
            logits: (batch, seq_len, vocab_size) classification logits
        """
        B, S = step_ids.shape
        device = step_ids.device

        # Frontend
        x = self.step_embeddings(step_ids)  # (B, S, D)
        x = self.pos_encoding(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Causal mask
        causal_mask = self._causal_mask(S, device)

        # Core Transformer (using decoder-only style: tgt=memory=x)
        x = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )

        # Postnet: predict embedding of next step
        predicted_embeddings = self.postnet(x)

        # Auxiliary classifier
        logits = self.classifier(x)

        return predicted_embeddings, logits

    def compute_loss(self, step_ids, padding_mask=None, mse_weight=1.0, ce_weight=0.5):
        """
        Combined loss:
          - MSE: predicted embedding vs. actual next step embedding (LCM-style)
          - CE: auxiliary cross-entropy for discrete classification (helps convergence)

        Args:
            step_ids: (batch, seq_len)
            padding_mask: (batch, seq_len) True where padded
        """
        predicted_emb, logits = self.forward(step_ids, padding_mask)

        # Targets: shifted by 1 (predict next step)
        # predicted_emb[:, t] should match embedding of step_ids[:, t+1]
        target_ids = step_ids[:, 1:]  # (B, S-1)
        target_emb = self.step_embeddings(target_ids).detach()  # (B, S-1, D)
        pred_emb = predicted_emb[:, :-1]  # (B, S-1, D)
        pred_logits = logits[:, :-1]  # (B, S-1, V)

        # Mask: ignore PAD positions in targets
        if padding_mask is not None:
            valid_mask = ~padding_mask[:, 1:]  # (B, S-1)
        else:
            valid_mask = (target_ids != 0)  # PAD=0

        # MSE loss (in embedding space)
        mse_per_pos = (pred_emb - target_emb).pow(2).sum(-1)  # (B, S-1)
        mse_loss = (mse_per_pos * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        # Cross-entropy loss (auxiliary)
        ce_loss = F.cross_entropy(
            pred_logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
            ignore_index=0,  # PAD
        )

        total_loss = mse_weight * mse_loss + ce_weight * ce_loss
        return total_loss, mse_loss.item(), ce_loss.item()

    @torch.no_grad()
    def predict_next_step(self, step_ids, top_k=5):
        """
        Predict next step given prefix.

        Returns:
            top_k_indices: (batch, top_k)
            top_k_scores: (batch, top_k)
        """
        predicted_emb, _ = self.forward(step_ids)
        last_pred = predicted_emb[:, -1]  # (B, D) - prediction for next position

        # Cosine similarity against codebook
        codebook = self.step_embeddings.weight  # (V, D)
        last_pred_norm = F.normalize(last_pred, dim=-1)
        codebook_norm = F.normalize(codebook, dim=-1)
        scores = torch.mm(last_pred_norm, codebook_norm.T)  # (B, V)

        # Zero out special tokens (indices 0-5)
        scores[:, :6] = -float('inf')

        top_scores, top_indices = scores.topk(top_k, dim=-1)
        return top_indices, top_scores

    @torch.no_grad()
    def complete_sequence(self, prefix_ids, max_len=200, eos_idx=3):
        """
        Autoregressively complete a sequence using classifier logits.
        Uses incremental decoding: only computes the last position at each step
        by leveraging the full forward once, then appending.

        Args:
            prefix_ids: (1, prefix_len) - single sequence
            max_len: maximum steps to generate
            eos_idx: EOS token index

        Returns:
            list of predicted step indices
        """
        device = prefix_ids.device
        current = prefix_ids.clone()
        predicted = []

        pe_limit = self.pos_encoding.pe.size(1)
        for _ in range(max_len):
            if current.size(1) >= pe_limit:
                break
            # Use classifier logits directly (much faster than cosine sim)
            _, logits = self.forward(current)
            next_logits = logits[:, -1]  # (1, V)
            next_logits[:, :6] = -float('inf')  # mask special tokens
            next_logits[:, eos_idx] = -float('inf')  # don't predict EOS early

            next_idx = next_logits.argmax(dim=-1).item()

            # Check if model is very uncertain (all low confidence) - stop
            if next_logits.max().item() < -10:
                break

            predicted.append(next_idx)
            next_tok = torch.tensor([[next_idx]], device=device)
            current = torch.cat([current, next_tok], dim=1)

        return predicted

    @torch.no_grad()
    def anomaly_scores(self, step_ids):
        """
        Compute per-position prediction error for anomaly detection.
        High error at a position indicates an unexpected/anomalous step.

        Args:
            step_ids: (batch, seq_len)

        Returns:
            errors: (batch, seq_len-1) per-position MSE
            max_error: (batch,) max error per sequence
            mean_error: (batch,) mean error per sequence
        """
        predicted_emb, _ = self.forward(step_ids)
        target_emb = self.step_embeddings(step_ids[:, 1:]).detach()
        pred_emb = predicted_emb[:, :-1]

        errors = (pred_emb - target_emb).pow(2).sum(-1)  # (B, S-1)
        max_error = errors.max(dim=-1).values
        mean_error = errors.mean(dim=-1)

        return errors, max_error, mean_error
