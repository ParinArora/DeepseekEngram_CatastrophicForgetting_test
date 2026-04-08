"""Shared optimization utilities for the bundled language-model variants.

The repo currently trains every model with the same next-token objective, so a
single trainer keeps experimentation simple and comparable across architectures.

This file sits between the pure model definitions and the continual-learning
tester: the models know how to compute logits and loss, and the trainer knows
how to step an optimizer over batches.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
import logging
import time

import torch
import torch.nn as nn
from torch.optim import AdamW

from .configs import TrainingConfig
from .common import CausalLMBase


logger = logging.getLogger(__name__)


class LMTrainer:
    """
    Shared LM trainer for all current architectures.

    Right now, all bundled models are trained with the same next-token objective,
    so they can share one training loop. If you later add architecture-specific
    losses or regularizers, split them into separate trainer files/classes.
    """

    def __init__(self, model: CausalLMBase, cfg: TrainingConfig) -> None:
        self.model = model.to(cfg.device)
        self.cfg = cfg
        # AdamW is a sensible default for Transformer-style language models.
        self.optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # The training/evaluation loops assume every tensor-like batch field
        # should live on the configured device.
        return {k: v.to(self.cfg.device) for k, v in batch.items()}

    def train_task(self, train_loader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        self.model.train()
        # Clear any stale gradients before starting a fresh task.
        self.optimizer.zero_grad(set_to_none=True)
        step = 0
        running_loss = 0.0
        start = time.perf_counter()
        epoch_mean_losses = []
        num_batches = len(train_loader) if hasattr(train_loader, "__len__") else None

        logger.info(
            "Starting training: epochs=%d, batches_per_epoch=%s",
            self.cfg.epochs,
            num_batches if num_batches is not None else "unknown",
        )

        for epoch in range(self.cfg.epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            for batch in train_loader:
                # Each tensor in `batch` usually has shape `[B, T]`, where `B`
                # is batch size and `T` is sequence length.
                batch = self._move_batch(batch)
                outputs = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                raw_loss = outputs["loss"]
                # Gradient accumulation simulates larger effective batch sizes.
                loss = raw_loss / self.cfg.gradient_accumulation_steps
                loss.backward()

                if self.cfg.max_grad_norm is not None:
                    # Gradient clipping limits unusually large updates that could
                    # destabilize training.
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # Keep logging based on the original loss scale so the numbers
                # are easy to compare with evaluation NLL.
                batch_loss = float(raw_loss.item())
                running_loss += batch_loss
                epoch_loss += batch_loss
                step += 1
                epoch_steps += 1

                if self.cfg.log_every > 0 and step % self.cfg.log_every == 0:
                    logger.info(
                        "Epoch %d/%d | step %d%s | mean_loss=%.4f",
                        epoch + 1,
                        self.cfg.epochs,
                        step,
                        f"/{self.cfg.epochs * num_batches}" if num_batches is not None else "",
                        running_loss / max(step, 1),
                    )

            epoch_mean_loss = epoch_loss / max(epoch_steps, 1)
            epoch_mean_losses.append(epoch_mean_loss)
            logger.info(
                "Finished epoch %d/%d | mean_loss=%.4f",
                epoch + 1,
                self.cfg.epochs,
                epoch_mean_loss,
            )

        elapsed = time.perf_counter() - start
        mean_loss = running_loss / max(step, 1)
        return {
            "num_steps": step,
            "mean_loss": mean_loss,
            "epoch_mean_losses": epoch_mean_losses,
            "train_time_sec": elapsed,
        }

    @torch.no_grad()
    def evaluate_nll(self, eval_loader: Iterable[Dict[str, torch.Tensor]]) -> float:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        for batch in eval_loader:
            # Evaluation reuses the exact same objective as training but without
            # gradient tracking.
            batch = self._move_batch(batch)
            outputs = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
            total_loss += float(outputs["loss"].item())
            total_batches += 1
        return total_loss / max(total_batches, 1)

    @torch.no_grad()
    def collect_hidden_states(
        self,
        eval_loader: Iterable[Dict[str, torch.Tensor]],
    ) -> Dict[str, Any]:
        """
        Collect hidden states for the first batch only.

        This keeps the example simple. For a real probing pipeline, you would
        usually collect many batches and pool/stack them explicitly.
        """
        self.model.eval()
        for batch in eval_loader:
            batch = self._move_batch(batch)
            # Representation probes can build richer tooling on top of this raw
            # hidden-state access point.
            outputs = self.model(input_ids=batch["input_ids"], return_hidden_states=True)
            return {"hidden_states": outputs.get("hidden_states", [])}
        return {"hidden_states": []}
