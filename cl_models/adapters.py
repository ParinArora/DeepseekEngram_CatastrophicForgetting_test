"""Adapters that let plain PyTorch language models plug into the tester.

The tester is intentionally model-agnostic, so this file translates from the
package's concrete PyTorch modules to the small protocol expected by
`tester.py`.

In other words: model files define *how the network computes*, this file defines
*how the network is presented to the continual-learning evaluator*.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
import copy

import torch

from .common import CausalLMBase
from .configs import TrainingConfig
from .trainers import LMTrainer


class TorchCLModelAdapter:
    """
    Adapter that makes a torch LM model compatible with the continual tester.

    Expected task data format:
    - train_data: iterable of batches, each batch is a dict with at least:
        {"input_ids": LongTensor[B, T], "labels": LongTensor[B, T]}
    - eval_data: same idea, or anything your score_fn understands.

    Notes:
    - `score_fn(adapter, data)` is intentionally external, so you can define task-
      specific metrics without rewriting the adapter.
    - `nll(data)` assumes `data` is a dataloader/iterable of standard LM batches.
    """

    def __init__(self, model: CausalLMBase, training_cfg: TrainingConfig) -> None:
        # The trainer owns optimization details; the adapter exposes a simpler
        # interface geared toward continual-learning evaluation.
        self.model = model
        self.training_cfg = training_cfg
        self.trainer = LMTrainer(model, training_cfg)

    def clone(self) -> "TorchCLModelAdapter":
        # Deep-copying a full trainer/optimizer state is sometimes expensive, but
        # this keeps the example simple and works for oracle/reference setups.
        return copy.deepcopy(self)

    def train_on_task(self, task: Any) -> Dict[str, Any]:
        # `TaskSpec` stores task-specific metadata, but only the train loader is
        # needed here.
        return self.trainer.train_task(task.train_data)

    def score(self, data: Any, score_fn: Callable[["TorchCLModelAdapter", Any], float]) -> float:
        # Metric logic lives outside the adapter so benchmarks can define their
        # own accuracy, exact-match, retention, or probe metrics.
        return score_fn(self, data)

    def nll(self, data: Any) -> float:
        # NLL is delegated to the shared trainer because it already knows how to
        # batch data and run the model in evaluation mode.
        return self.trainer.evaluate_nll(data)

    def num_parameters(self) -> int:
        # Count every scalar weight and bias value in the model.
        return sum(p.numel() for p in self.model.parameters())

    def num_trainable_parameters(self) -> int:
        # This excludes any parameter that has `requires_grad=False`.
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def active_parameter_estimate(self) -> Optional[int]:
        # The current bundled models are dense at runtime.
        return None

    def flops_estimate(self) -> Optional[float]:
        # Left as a hook; exact FLOPs are architecture- and sequence-length-dependent.
        return None

    def get_hidden_states(self, data: Any, layers: Sequence[str]) -> Dict[str, Any]:
        # We ignore `layers` here and return all hidden states. A richer version
        # could map names to indices and select only the requested ones.
        return self.trainer.collect_hidden_states(data)


@torch.no_grad()
def lm_accuracy_score(adapter: TorchCLModelAdapter, eval_loader: Iterable[Dict[str, torch.Tensor]]) -> float:
    """
    Token-level next-token accuracy.

    This is a simple default metric. For many CL setups you may want exact match,
    F1, perplexity, or a benchmark-specific metric instead.
    """
    model = adapter.model.to(adapter.training_cfg.device)
    model.eval()

    total_correct = 0
    total_count = 0
    for batch in eval_loader:
        # This metric mirrors the training objective: predict the next token at
        # every supervised position in the sequence.
        batch = {k: v.to(adapter.training_cfg.device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"])
        # `logits` starts as `[B, T, V]`. Dropping the final time step gives
        # `[B, T-1, V]`, which lines up with the shifted labels below.
        logits = outputs["logits"][:, :-1, :]
        labels = batch["labels"][:, 1:]
        preds = logits.argmax(dim=-1)

        mask = labels != -100
        total_correct += int(((preds == labels) & mask).sum().item())
        total_count += int(mask.sum().item())

    return total_correct / max(total_count, 1)
