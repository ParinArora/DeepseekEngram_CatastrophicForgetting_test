"""Shared building blocks used by all language-model variants in the package.

This module contains the reusable neural-network components that are agnostic to
the continual-learning setup: attention, MLP blocks, positional embeddings, and
the small base interface expected by the adapter/trainer stack.

Every model file in `cl_models/` builds on these pieces, so understanding this
file makes the rest of the package much easier to follow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Simple batch-first causal self-attention block.

    Input and output both use shape `[B, T, D]`:
    - `B` = batch size
    - `T` = sequence length
    - `D` = model width (`d_model`)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build the usual upper-triangular causal mask so position t can only
        # use information from positions <= t.
        t = x.size(1)
        # `causal_mask` has shape [T, T]. `True` entries mark future positions
        # that attention is not allowed to read from.
        causal_mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        # MultiheadAttention preserves the outer shape, so `[B, T, D]` stays
        # `[B, T, D]` after attention.
        out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        return self.dropout(out)


class FeedForward(nn.Module):
    """Standard Transformer MLP block.

    The hidden width is `mlp_ratio * d_model`, matching the common Transformer
    design where each block alternates attention with a wider pointwise network.
    """

    def __init__(self, d_model: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        # The intermediate width is wider than the residual stream so the block
        # can mix features more expressively before projecting back down.
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The sequential module keeps the outer shape `[B, T, D]` unchanged even
        # though it briefly expands the last dimension to `mlp_ratio * D`.
        return self.net(x)


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block.

    Pre-normalization keeps the residual stream stable, and `enable_attention`
    lets ablation models reuse the same block while swapping attention out.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int,
        dropout: float,
        layer_norm_eps: float,
        enable_attention: bool = True,
    ) -> None:
        super().__init__()
        self.enable_attention = enable_attention
        self.ln_1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln_2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout) if enable_attention else nn.Identity()
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_attention:
            # Attention updates the residual stream first when enabled.
            # Shape stays `[B, T, D]` before and after the residual addition.
            x = x + self.attn(self.ln_1(x))
        # The MLP path is always active in every bundled architecture.
        x = x + self.mlp(self.ln_2(x))
        return x


class MLPOnlyBlock(nn.Module):
    """Stripped-down block with no attention, useful for TTT-style ablations."""

    def __init__(self, d_model: int, mlp_ratio: int, dropout: float, layer_norm_eps: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This block is shape-preserving: `[B, T, D] -> [B, T, D]`.
        return x + self.mlp(self.ln(x))


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings.

    Positions are generated on the fly from sequence length, so callers only
    need to supply `input_ids`.
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        # Expand one `[0, ..., T-1]` index vector across the batch.
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        # The embedding lookup changes `[B, T]` integer positions into `[B, T, D]`
        # dense vectors.
        return self.embedding(positions)


class CausalLMBase(nn.Module):
    """
    Shared causal LM interface used by all architectures.

    The model returns a dict so the adapter/trainer can stay architecture-agnostic.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def compute_next_token_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        """Shifted causal-LM cross-entropy.

        The model predicts token `t+1` from the hidden state at position `t`,
        so both tensors are shifted before cross-entropy is computed.
        """
        # If `logits` is `[B, T, V]`, then `shift_logits` becomes `[B, T-1, V]`.
        shift_logits = logits[:, :-1, :].contiguous()
        # If `labels` is `[B, T]`, then `shift_labels` becomes `[B, T-1]`.
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
        )
