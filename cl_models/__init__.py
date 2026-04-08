"""Public package surface for the bundled continual-learning models.

This module intentionally re-exports the small set of classes that most callers
need so experiment scripts can import from `cl_models` without knowing the
internal file layout.
"""

from .configs import (
    TransformerConfig,
    EngramConfig,
    TrainingConfig,
)
from .transformer_baseline import StandardTransformerLM
from .transformer_engram import TransformerWithEngramLM
from .reduced_attention import ReducedAttentionTransformerLM
from .mlp_only import MLPOnlyLM
from .trainers import LMTrainer
from .adapters import TorchCLModelAdapter

__all__ = [
    # Configuration objects used to instantiate models and training helpers.
    "TransformerConfig",
    "EngramConfig",
    "TrainingConfig",
    # Model families exposed by the package.
    "StandardTransformerLM",
    "TransformerWithEngramLM",
    "ReducedAttentionTransformerLM",
    "MLPOnlyLM",
    # Training/evaluation integration points.
    "LMTrainer",
    "TorchCLModelAdapter",
]
