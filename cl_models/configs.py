"""Dataclass-based configuration objects shared across the model package.

Keeping configuration in small dataclasses makes experiments easy to read,
serialize, and tweak from scripts without threading long argument lists through
every constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class TransformerConfig:
    """Shared architecture config for all causal LM variants.

    These fields describe the dense Transformer backbone that every architecture
    variant starts from, even if some variants later disable or augment parts of
    that backbone.
    """

    # Vocabulary and sequence layout.
    vocab_size: int
    max_seq_len: int = 512
    # Core model width/depth.
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    # Feed-forward and regularization controls.
    mlp_ratio: int = 4
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    # Token used for padding when a dataset needs fixed-length batching.
    pad_token_id: int = 0


@dataclass
class EngramConfig:
    """
    Engram module config.

    This is inspired by the official DeepSeek Engram demo implementation, but
    adapted into a cleaner single-stream module for causal LM experimentation.
    """

    # Tokenizer used before Engram compresses and hashes token ids.
    tokenizer_name_or_path: str = "gpt2"
    # Target hashed vocabulary size per n-gram order, e.g. [2-gram, 3-gram].
    engram_vocab_size: List[int] = field(default_factory=lambda: [128_000, 128_000])
    # Controls which n-grams are built and how large their embeddings are.
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    # Backbone layer ids where Engram modules are injected.
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    # Hashing and local-mixing details.
    pad_id: int = 0
    seed: int = 0
    kernel_size: int = 4
    # Reused as a branch-count control in the compatibility implementation.
    gating_hidden_multiplier: int = 1
    # Optional dependency injection hook for nonstandard token spaces.
    compressed_tokenizer_factory: Optional[Callable[[str], Any]] = None


@dataclass
class TrainingConfig:
    """Shared trainer config.

    The trainer is intentionally small, so these options mostly correspond
    directly to one concept in the training loop.
    """

    # Optimizer hyperparameters.
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: Optional[float] = 1.0
    # Basic loop controls.
    epochs: int = 1
    device: str = "cpu"
    gradient_accumulation_steps: int = 1
    amp: bool = False
    # Human-facing progress logging cadence.
    log_every: int = 50
