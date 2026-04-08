"""MLP-only causal language model used as a stronger ablation baseline.

This keeps the same input/output contract as the Transformer models but removes
self-attention entirely, which makes it useful for isolating how much sequence
mixing the attention mechanism itself is contributing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .configs import TransformerConfig
from .common import CausalLMBase, PositionalEmbedding, MLPOnlyBlock


class MLPOnlyLM(CausalLMBase):
    """
    MLP-only causal LM.

    This is the stripped architecture analogous to the kind of toy setup used in
    test-time-training papers when they remove attention to isolate the effect of
    another mechanism.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # The model still keeps token/position embeddings and residual depth;
        # only the self-attention operation is removed.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = PositionalEmbedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            MLPOnlyBlock(
                d_model=cfg.d_model,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                layer_norm_eps=cfg.layer_norm_eps,
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
        # `input_ids` is `[B, T]`. Adding token and positional embeddings turns
        # it into the residual stream `[B, T, D]`.
        x = self.token_emb(input_ids) + self.pos_emb(input_ids)
        x = self.dropout(x)

        hidden_states: List[torch.Tensor] = []
        for block in self.blocks:
            # Each block keeps shape `[B, T, D]` while mixing features only
            # through position-wise MLP operations.
            x = block(x)
            if return_hidden_states:
                hidden_states.append(x)

        # Final logits are `[B, T, V]`.
        x = self.ln_f(x)
        logits = self.lm_head(x)

        out: Dict[str, Any] = {"logits": logits}
        if labels is not None:
            out["loss"] = self.compute_next_token_loss(logits, labels)
        if return_hidden_states:
            out["hidden_states"] = hidden_states
        return out
