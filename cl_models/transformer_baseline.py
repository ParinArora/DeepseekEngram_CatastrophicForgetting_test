"""Baseline dense Transformer language model used in the experiments.

This is the reference model that the ablations and Engram variant are compared
against. It keeps the familiar Transformer data flow:

`input_ids [B, T] -> embeddings [B, T, D] -> blocks [B, T, D] -> logits [B, T, V]`
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .configs import TransformerConfig
from .common import CausalLMBase, PositionalEmbedding, TransformerBlock


class StandardTransformerLM(CausalLMBase):
    """Standard causal Transformer LM used as the main baseline."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Standard token + position embedding stack feeding a residual backbone.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = PositionalEmbedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            # Each block preserves shape `[B, T, D]`.
            TransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                layer_norm_eps=cfg.layer_norm_eps,
                enable_attention=True,
            )
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        # Embed tokens and positions into the shared residual stream.
        # `input_ids` is `[B, T]`; each embedding lookup produces `[B, T, D]`.
        x = self.token_emb(input_ids) + self.pos_emb(input_ids)
        x = self.dropout(x)

        hidden_states: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            if return_hidden_states:
                # Each entry captures the post-block representation for probing.
                hidden_states.append(x)

        # Final normalization + linear readout convert hidden states to logits.
        # `logits` has shape `[B, T, V]`, where `V` is vocabulary size.
        x = self.ln_f(x)
        logits = self.lm_head(x)

        out: Dict[str, Any] = {"logits": logits}
        if labels is not None:
            out["loss"] = self.compute_next_token_loss(logits, labels)
        if return_hidden_states:
            out["hidden_states"] = hidden_states
        return out
