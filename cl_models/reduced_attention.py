"""Attention-reduced Transformer ablation.

This model keeps the same overall scaffold as the baseline but disables
attention in a subset of blocks so experiments can isolate how much the full
attention stack contributes.

The tensor shapes still match the baseline exactly:
`[B, T] -> [B, T, D] -> [B, T, D] -> [B, T, V]`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .configs import TransformerConfig
from .common import CausalLMBase, PositionalEmbedding, TransformerBlock


class ReducedAttentionTransformerLM(CausalLMBase):
    """
    Transformer ablation with reduced attention capacity.

    The simplest robust ablation is to keep only every N-th attention layer active,
    while leaving the MLP path intact. This gives you a backbone where attention is
    weakened without changing the overall block structure too radically.
    """

    def __init__(self, cfg: TransformerConfig, attention_every_n_layers: int = 2) -> None:
        super().__init__()
        self.cfg = cfg
        # Clamp to at least one so we never construct a modulo-by-zero schedule.
        self.attention_every_n_layers = max(1, attention_every_n_layers)

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = PositionalEmbedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList()
        for i in range(cfg.n_layers):
            # Only every Nth block keeps self-attention; the others behave like
            # MLP-only residual blocks inside the shared TransformerBlock class.
            enable_attention = (i % self.attention_every_n_layers) == 0
            self.blocks.append(
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    layer_norm_eps=cfg.layer_norm_eps,
                    enable_attention=enable_attention,
                )
            )

        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        # Token ids `[B, T]` become dense vectors `[B, T, D]`.
        x = self.token_emb(input_ids) + self.pos_emb(input_ids)
        x = self.dropout(x)

        hidden_states: List[torch.Tensor] = []
        for block in self.blocks:
            # Whether or not attention is enabled inside a block, the outer shape
            # stays `[B, T, D]`.
            x = block(x)
            if return_hidden_states:
                hidden_states.append(x)

        # The final projection turns one vector of length `D` at each position
        # into one vocabulary-sized logit vector.
        x = self.ln_f(x)
        logits = self.lm_head(x)

        out: Dict[str, Any] = {"logits": logits}
        if labels is not None:
            out["loss"] = self.compute_next_token_loss(logits, labels)
        if return_hidden_states:
            out["hidden_states"] = hidden_states
        return out
