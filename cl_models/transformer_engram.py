"""Transformer backbone augmented with the Engram memory side-path.

This model keeps the baseline Transformer's main residual stream but adds an
extra Engram-produced residual update at selected layers. The input/output
shapes stay the same as the baseline:

`input_ids [B, T] -> residual stream [B, T, D] -> logits [B, T, V]`
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .configs import TransformerConfig, EngramConfig
from .common import CausalLMBase, PositionalEmbedding, TransformerBlock
from .engram_module import EngramMemoryModule


class TransformerWithEngramLM(CausalLMBase):
    """
    Standard causal Transformer with Engram memory injection.

    Engram is injected before selected blocks as a residual side-path.
    This keeps training/eval comparable to the baseline while preserving the
    memory/compute split the official DeepSeek Engram work is aiming for.
    """

    def __init__(self, base_cfg: TransformerConfig, engram_cfg: EngramConfig) -> None:
        super().__init__()
        self.base_cfg = base_cfg
        self.engram_cfg = engram_cfg

        # The dense Transformer backbone stays identical to the baseline so the
        # Engram contribution can be studied as an additive residual path.
        self.token_emb = nn.Embedding(base_cfg.vocab_size, base_cfg.d_model)
        self.pos_emb = PositionalEmbedding(base_cfg.max_seq_len, base_cfg.d_model)
        self.dropout = nn.Dropout(base_cfg.dropout)

        self.blocks = nn.ModuleList([
            # The baseline-style blocks still do the main sequence processing.
            TransformerBlock(
                d_model=base_cfg.d_model,
                n_heads=base_cfg.n_heads,
                mlp_ratio=base_cfg.mlp_ratio,
                dropout=base_cfg.dropout,
                layer_norm_eps=base_cfg.layer_norm_eps,
                enable_attention=True,
            )
            for _ in range(base_cfg.n_layers)
        ])

        self.engram_layers = nn.ModuleDict()
        for layer_id in engram_cfg.layer_ids:
            if 0 <= layer_id < base_cfg.n_layers:
                # Only layers requested by config get an auxiliary Engram module.
                self.engram_layers[str(layer_id)] = EngramMemoryModule(
                    d_model=base_cfg.d_model,
                    layer_id=layer_id,
                    cfg=engram_cfg,
                )

        self.ln_f = nn.LayerNorm(base_cfg.d_model, eps=base_cfg.layer_norm_eps)
        self.lm_head = nn.Linear(base_cfg.d_model, base_cfg.vocab_size, bias=False)

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
        for layer_id, block in enumerate(self.blocks):
            if str(layer_id) in self.engram_layers:
                # Engram is injected residually before the dense Transformer block.
                # Both tensors are `[B, T, D]`, so the addition keeps the same
                # shape and simply changes the content of the residual stream.
                x = x + self.engram_layers[str(layer_id)](hidden_states=x, input_ids=input_ids)
            x = block(x)
            if return_hidden_states:
                hidden_states.append(x)

        # Final logits are `[B, T, V]`, exactly like the baseline.
        x = self.ln_f(x)
        logits = self.lm_head(x)

        out: Dict[str, Any] = {"logits": logits}
        if labels is not None:
            out["loss"] = self.compute_next_token_loss(logits, labels)
        if return_hidden_states:
            out["hidden_states"] = hidden_states
        return out
