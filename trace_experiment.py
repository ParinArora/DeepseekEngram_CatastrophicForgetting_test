"""TRACE continual-learning experiment built on top of the shared tester.

This file plays the same role for the TRACE benchmark that
`splitmnist_experiment.py` plays for SplitMNIST:

- it loads one benchmark into the data format expected by this repository
- it converts each task into `TaskSpec` objects for `tester.py`
- it instantiates one of the `cl_models` architectures
- it runs sequential task training and produces JSON-friendly reports

TRACE itself is text-to-text, so this script works with a normal tokenizer and
uses generation-based metrics such as exact match and token-level F1.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoTokenizer
except Exception as exc:
    raise ImportError(
        "This script requires transformers. Install with: pip install transformers"
    ) from exc

# -----------------------------------------------------------------------------
# Flexible imports: support either a package layout (`cl_models`) or flat files.
# -----------------------------------------------------------------------------
try:
    from cl_models import (
        TransformerConfig,
        EngramConfig,
        TrainingConfig,
        StandardTransformerLM,
        TransformerWithEngramLM,
        ReducedAttentionTransformerLM,
        MLPOnlyLM,
        TorchCLModelAdapter,
    )
except Exception:
    from configs import TransformerConfig, EngramConfig, TrainingConfig
    from transformer_baseline import StandardTransformerLM
    from transformer_engram import TransformerWithEngramLM
    from reduced_attention import ReducedAttentionTransformerLM
    from mlp_only import MLPOnlyLM
    from adapters import TorchCLModelAdapter

from tester import (
    TaskSpec,
    ProbeSuiteSpec,
    ContinualTester,
    ContinualTesterConfig,
    aggregate_reports,
)

try:
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
except Exception:
    CSVLogger = None
    TensorBoardLogger = None

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class TraceLightningExperimentLogger:
    """Optional Lightning-style experiment logger for TRACE runs."""

    def __init__(
        self,
        enabled: bool,
        save_dir: str,
        name: str,
        version: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.save_dir = save_dir
        self.name = name
        self.version = version
        self.loggers: List[Any] = []
        self.metrics_path: Optional[Path] = None

        if not self.enabled:
            return

        log_root = Path(save_dir)
        log_root.mkdir(parents=True, exist_ok=True)

        if CSVLogger is not None:
            self.loggers.append(CSVLogger(save_dir=save_dir, name=name, version=version))
        if TensorBoardLogger is not None:
            self.loggers.append(TensorBoardLogger(save_dir=save_dir, name=name, version=version))

        run_dir = log_root / name
        if version:
            run_dir = run_dir / version
        run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = run_dir / "trace_metrics.jsonl"

        if self.loggers:
            logger.info(
                "Enabled Lightning experiment logging: save_dir=%s name=%s version=%s backends=%s",
                save_dir,
                name,
                version if version is not None else "auto",
                ",".join(type(x).__name__ for x in self.loggers),
            )
        else:
            logger.info(
                "Lightning logging requested, but lightning loggers are unavailable; falling back to JSONL logging at %s",
                self.metrics_path,
            )

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        clean = self._sanitize_dict(params)
        for exp_logger in self.loggers:
            try:
                exp_logger.log_hyperparams(clean)
            except Exception:
                logger.exception("Failed to log hyperparameters with %s", type(exp_logger).__name__)

        self._append_jsonl({"type": "hyperparams", "payload": clean})

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled:
            return

        clean = self._sanitize_dict(metrics)
        numeric_metrics = {k: v for k, v in clean.items() if isinstance(v, (int, float))}
        if step is not None:
            numeric_metrics.setdefault("step", step)

        for exp_logger in self.loggers:
            try:
                exp_logger.log_metrics(numeric_metrics, step=step)
            except Exception:
                logger.exception("Failed to log metrics with %s", type(exp_logger).__name__)

        payload = {"type": "metrics", "payload": clean}
        if step is not None:
            payload["step"] = step
        self._append_jsonl(payload)

    def finalize(self, status: str = "success") -> None:
        if not self.enabled:
            return

        self._append_jsonl({"type": "finalize", "status": status})
        for exp_logger in self.loggers:
            finalize = getattr(exp_logger, "finalize", None)
            if callable(finalize):
                try:
                    finalize(status)
                except Exception:
                    logger.exception("Failed to finalize %s", type(exp_logger).__name__)

    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        if self.metrics_path is None:
            return
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._sanitize_dict(record)) + "\n")

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                nested = self._sanitize_dict(value)
                for nested_key, nested_value in nested.items():
                    out[f"{key}/{nested_key}"] = nested_value
            elif isinstance(value, (list, tuple)):
                out[key] = json.dumps([self._sanitize_value(x) for x in value])
            else:
                out[key] = self._sanitize_value(value)
        return out

    def _sanitize_value(self, value: Any) -> Any:
        if value is None:
            return "None"
        if isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return str(value)


def build_experiment_tracker(args, seed: int) -> TraceLightningExperimentLogger:
    # This wrapper keeps the existing experiment flow intact while making it
    # easy to mirror runs into Lightning-compatible loggers.
    experiment_name = args.lightning_experiment_name or f"trace_{args.model}"
    version = f"seed_{seed}"
    return TraceLightningExperimentLogger(
        enabled=args.enable_lightning_logging,
        save_dir=args.lightning_log_dir,
        name=experiment_name,
        version=version,
    )


def summarize_report_metrics(report) -> Dict[str, Any]:
    # Flatten the final report into logger-friendly scalars so each run is easy
    # to compare in TensorBoard/CSV outputs.
    metrics: Dict[str, Any] = {}

    summary = report.summary
    metrics["summary/final_acc"] = summary.final_acc
    metrics["summary/mean_forgetting"] = summary.mean_forgetting
    metrics["summary/worst_forgetting"] = summary.worst_forgetting
    metrics["summary/bwt"] = summary.bwt
    metrics["summary/fwt"] = summary.fwt
    metrics["summary/intransigence"] = summary.intransigence

    for task_name, value in summary.per_task_final.items():
        metrics[f"summary/per_task_final/{task_name}"] = value
    for task_name, value in summary.per_task_best.items():
        metrics[f"summary/per_task_best/{task_name}"] = value
    for task_name, value in summary.per_task_forgetting.items():
        metrics[f"summary/per_task_forgetting/{task_name}"] = value
    for task_name, value in summary.immediate_new_task_scores.items():
        metrics[f"summary/immediate_new_task_scores/{task_name}"] = value

    for stage in report.stages:
        stage_prefix = f"stage_{stage.stage_idx}"
        if stage.trained_task_name is not None:
            metrics[f"{stage_prefix}/trained_task"] = stage.trained_task_name
        for task_name, value in stage.task_scores.items():
            metrics[f"{stage_prefix}/task_scores/{task_name}"] = value
        for metric_name, value in stage.old_domain_nll.items():
            metrics[f"{stage_prefix}/old_domain_nll/{metric_name}"] = value
        for metric_name, value in stage.old_domain_ppl.items():
            metrics[f"{stage_prefix}/old_domain_ppl/{metric_name}"] = value
        for metric_name, value in stage.new_domain_nll.items():
            metrics[f"{stage_prefix}/new_domain_nll/{metric_name}"] = value
        for metric_name, value in stage.new_domain_ppl.items():
            metrics[f"{stage_prefix}/new_domain_ppl/{metric_name}"] = value
        for probe_name, probe_metrics in stage.general_probes.items():
            for metric_name, value in probe_metrics.items():
                metrics[f"{stage_prefix}/general_probes/{probe_name}/{metric_name}"] = value
        for probe_name, probe_metrics in stage.memory_probes.items():
            for metric_name, value in probe_metrics.items():
                metrics[f"{stage_prefix}/memory_probes/{probe_name}/{metric_name}"] = value
        for probe_name, probe_metrics in stage.representation_probes.items():
            for metric_name, value in probe_metrics.items():
                metrics[f"{stage_prefix}/representation_probes/{probe_name}/{metric_name}"] = value
        if stage.resources.train_time_sec is not None:
            metrics[f"{stage_prefix}/resources/train_time_sec"] = stage.resources.train_time_sec
        if stage.resources.eval_time_sec is not None:
            metrics[f"{stage_prefix}/resources/eval_time_sec"] = stage.resources.eval_time_sec
        if stage.resources.train_units_per_sec is not None:
            metrics[f"{stage_prefix}/resources/train_units_per_sec"] = stage.resources.train_units_per_sec
        if stage.resources.eval_units_per_sec is not None:
            metrics[f"{stage_prefix}/resources/eval_units_per_sec"] = stage.resources.eval_units_per_sec

    return metrics


# =============================================================================
# Helpers
# =============================================================================
def set_seed(seed: int) -> None:
    # Seed every random-number source used by the experiment so runs are as
    # repeatable as the underlying hardware allows.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    # The metrics compare normalized text so punctuation, repeated spaces, and
    # capitalization differences do not count as separate answers.
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1(pred: str, gold: str) -> float:
    # Split both strings into normalized tokens, then compute overlap in the
    # same spirit as question-answering F1 metrics.
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    gold_counts: Dict[str, int] = {}
    for t in gold_toks:
        # Count each gold token so repeated words are handled correctly.
        gold_counts[t] = gold_counts.get(t, 0) + 1

    common = 0
    for t in pred_toks:
        if gold_counts.get(t, 0) > 0:
            common += 1
            gold_counts[t] -= 1

    if common == 0:
        return 0.0

    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def normalized_exact_match(pred: str, gold: str) -> float:
    # Exact match is strict after normalization: the full normalized strings
    # must be identical.
    return float(normalize_text(pred) == normalize_text(gold))


def count_examples(data: Any) -> Optional[int]:
    # The tester uses this to report throughput in "examples per second".
    if hasattr(data, "dataset"):
        return len(data.dataset)
    if hasattr(data, "__len__"):
        try:
            return len(data)
        except TypeError:
            return None
    return None


# =============================================================================
# TRACE local-format loading
# =============================================================================
def load_json_list(path: str | Path) -> List[Dict[str, Any]]:
    # Each TRACE split file is expected to be one JSON list of records.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def load_trace_task_dir(task_dir: str | Path) -> Dict[str, List[Dict[str, str]]]:
    # TRACE tasks live in directories with one file per split.
    task_dir = Path(task_dir)
    train_path = task_dir / "train.json"
    eval_path = task_dir / "eval.json"
    test_path = task_dir / "test.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")
    if not eval_path.exists() and not test_path.exists():
        raise FileNotFoundError(
            f"Need at least one of {eval_path} or {test_path} for evaluation"
        )

    train_data = load_json_list(train_path)
    eval_data = load_json_list(eval_path) if eval_path.exists() else load_json_list(test_path)
    test_data = load_json_list(test_path) if test_path.exists() else eval_data

    def validate(records: List[Dict[str, Any]], split_name: str) -> List[Dict[str, str]]:
        # Convert loose JSON values into the exact string-only format that the
        # dataset class below expects.
        out: List[Dict[str, str]] = []
        for i, rec in enumerate(records):
            if "prompt" not in rec or "answer" not in rec:
                raise ValueError(
                    f"{task_dir.name}/{split_name}[{i}] must contain 'prompt' and 'answer' keys"
                )
            out.append(
                {
                    "prompt": str(rec["prompt"]).strip(),
                    "answer": str(rec["answer"]).strip(),
                }
            )
        return out

    return {
        "train": validate(train_data, "train"),
        "eval": validate(eval_data, "eval"),
        "test": validate(test_data, "test"),
    }


def discover_trace_tasks(trace_root: str | Path, task_order: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    # This helper decides which task directories will become the continual
    # learning stream, and in what order.
    trace_root = Path(trace_root)
    if not trace_root.exists():
        raise FileNotFoundError(f"TRACE root does not exist: {trace_root}")

    candidate_dirs = [p for p in trace_root.iterdir() if p.is_dir()]
    valid_dirs = [p for p in candidate_dirs if (p / "train.json").exists()]

    if not valid_dirs:
        raise FileNotFoundError(
            f"No TRACE task directories found under {trace_root}. "
            f"Expected subdirs each containing train.json and eval.json/test.json."
        )

    name_to_dir = {p.name: p for p in valid_dirs}

    if task_order:
        missing = [name for name in task_order if name not in name_to_dir]
        if missing:
            raise ValueError(f"Requested task(s) not found under {trace_root}: {missing}")
        return [(name, name_to_dir[name]) for name in task_order]

    # Stable deterministic default order.
    return sorted([(p.name, p) for p in valid_dirs], key=lambda x: x[0].lower())


# =============================================================================
# Prompt formatting + dataset
# =============================================================================
def build_prompt_text(prompt: str) -> str:
    # Keep this simple and explicit so generations are easier to parse.
    return f"Instruction:\n{prompt.strip()}\n\nAnswer:\n"


class TraceCausalLMDataset(Dataset):
    """
    Converts TRACE prompt/answer pairs into causal-LM training examples.

    Training:
      input = prompt_prefix + answer + eos
      labels = ignore on prompt_prefix, supervise answer + eos

    Evaluation:
      same tensors for NLL if needed, but generation-based metrics use the raw
      prompt/answer fields exposed in the batch.
    """

    def __init__(
        self,
        records: Sequence[Dict[str, str]],
        tokenizer,
        max_seq_len: int,
        add_eos: bool = True,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.add_eos = add_eos

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                raise ValueError("Tokenizer must have either pad_token or eos_token.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        prompt = rec["prompt"]
        answer = rec["answer"]

        # The visible text prefix is what the model sees before it starts
        # generating the answer.
        prefix_text = build_prompt_text(prompt)
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        if self.add_eos and self.tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [self.tokenizer.eos_token_id]

        # `input_ids` is one long 1D token sequence of length T:
        #   prefix tokens followed by answer tokens.
        input_ids = prefix_ids + answer_ids
        # `labels` has the same length T, but the prompt positions are masked
        # with IGNORE_INDEX so loss is only computed on the answer region.
        labels = [IGNORE_INDEX] * len(prefix_ids) + answer_ids

        # Left-truncate long sequences to preserve the answer region.
        if len(input_ids) > self.max_seq_len:
            overflow = len(input_ids) - self.max_seq_len
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        if len(input_ids) == 0:
            raise ValueError(f"Empty tokenized sample at index {idx}")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_text": prompt,
            "answer_text": answer,
            "prefix_text": prefix_text,
        }


def collate_trace_batch(batch: Sequence[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    # Find the longest example in this minibatch so every shorter example can be
    # padded up to the same length.
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    labels = []
    attention_mask = []
    prompt_text = []
    answer_text = []
    prefix_text = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        # Each individual example starts as shape [T_i]. Left-padding turns it
        # into [max_len], and stacking later will produce [B, max_len].
        input_ids.append(
            torch.cat(
                [
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                    item["input_ids"],
                ],
                dim=0,
            )
        )
        labels.append(
            torch.cat(
                [
                    # Prompt tokens and padding should not contribute to loss.
                    torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                    item["labels"],
                ],
                dim=0,
            )
        )
        attention_mask.append(
            torch.cat(
                [
                    # 0 marks padding, 1 marks a real token.
                    torch.zeros(pad_len, dtype=torch.long),
                    torch.ones(seq_len, dtype=torch.long),
                ],
                dim=0,
            )
        )

        prompt_text.append(item["prompt_text"])
        answer_text.append(item["answer_text"])
        prefix_text.append(item["prefix_text"])

    return {
        # Final tensor shapes:
        # - input_ids:      [B, max_len]
        # - labels:         [B, max_len]
        # - attention_mask: [B, max_len]
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "prompt_text": prompt_text,
        "answer_text": answer_text,
        "prefix_text": prefix_text,
    }


@dataclass
class TraceCollator:
    """Picklable collator wrapper so DataLoader workers can reuse TRACE batching."""

    pad_token_id: int

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        # Wrapping the collate function in a top-level dataclass makes it
        # picklable, which is important when DataLoader uses worker processes.
        return collate_trace_batch(batch, pad_token_id=self.pad_token_id)


# =============================================================================
# Generation + metrics
# =============================================================================
@torch.no_grad()
def greedy_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
) -> torch.Tensor:
    model.eval()
    # `generated` starts as [B, T_prefix] and grows by one column each loop.
    generated = input_ids

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated)
        # `outputs["logits"]` has shape [B, T_current, vocab_size], so taking
        # the last time step gives one next-token distribution per example.
        next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        # Concatenating along dim=1 changes the sequence length from T to T+1.
        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and bool((next_token == eos_token_id).all().item()):
            break

    return generated


@torch.no_grad()
def trace_generation_metrics(
    adapter: TorchCLModelAdapter,
    eval_loader: Iterable[Dict[str, Any]],
    tokenizer,
    max_new_tokens: int = 64,
) -> Dict[str, float]:
    model = adapter.model.to(adapter.training_cfg.device)
    model.eval()

    total_em = 0.0
    total_f1 = 0.0
    total = 0

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(adapter.training_cfg.device)

        # Build generation prefix only: trim off supervised answer region by
        # re-tokenizing the explicit prefix_text.
        prefix_ids_list = [
            tokenizer.encode(prefix, add_special_tokens=False)
            for prefix in batch["prefix_text"]
        ]
        max_prefix_len = max(len(x) for x in prefix_ids_list)

        prefix_tensor = []
        for ids in prefix_ids_list:
            pad_len = max_prefix_len - len(ids)
            prefix_tensor.append(
                torch.tensor(
                    # Every prefix becomes one vector of shape [max_prefix_len].
                    [tokenizer.pad_token_id] * pad_len + ids,
                    dtype=torch.long,
                    device=adapter.training_cfg.device,
                )
            )
        # After stacking, `prefix_tensor` has shape [B, max_prefix_len].
        prefix_tensor = torch.stack(prefix_tensor, dim=0)

        generated = greedy_generate(
            model=model,
            input_ids=prefix_tensor,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Slice off the original prefix so only newly generated answer tokens
        # remain. If `generated` is [B, max_prefix_len + K] then `gen_suffix`
        # is [B, K].
        gen_suffix = generated[:, max_prefix_len:]
        pred_texts = tokenizer.batch_decode(gen_suffix, skip_special_tokens=True)

        for pred, gold in zip(pred_texts, batch["answer_text"]):
            pred = pred.strip()
            gold = gold.strip()
            total_em += normalized_exact_match(pred, gold)
            total_f1 += token_f1(pred, gold)
            total += 1

    return {
        "exact_match": total_em / max(total, 1),
        "token_f1": total_f1 / max(total, 1),
    }


def trace_score_fn_factory(tokenizer, max_new_tokens: int):
    # The tester expects one scalar score per task, so this wrapper converts the
    # richer generation metrics into a single number.
    def score_fn(adapter: TorchCLModelAdapter, eval_loader: Iterable[Dict[str, Any]]) -> float:
        # Use normalized exact match as the main CL score.
        metrics = trace_generation_metrics(
            adapter=adapter,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        return metrics["exact_match"]
    return score_fn


def make_general_probe_suite(all_eval_loaders: Dict[str, DataLoader], tokenizer, max_new_tokens: int) -> ProbeSuiteSpec:
    # The general probe re-runs generation metrics over every task's eval set at
    # each continual-learning stage.
    def general_probe_fn(adapter: TorchCLModelAdapter, data: Dict[str, DataLoader], ctx) -> Dict[str, float]:
        out: Dict[str, float] = {}
        ems = []
        f1s = []
        for task_name, loader in data.items():
            metrics = trace_generation_metrics(adapter, loader, tokenizer, max_new_tokens=max_new_tokens)
            out[f"{task_name}_em"] = metrics["exact_match"]
            out[f"{task_name}_f1"] = metrics["token_f1"]
            ems.append(metrics["exact_match"])
            f1s.append(metrics["token_f1"])
        out["mean_em_all_tasks"] = float(sum(ems) / max(len(ems), 1))
        out["mean_f1_all_tasks"] = float(sum(f1s) / max(len(f1s), 1))
        return out

    return ProbeSuiteSpec(
        name="trace_generation_probe_suite",
        data=all_eval_loaders,
        run_fn=general_probe_fn,
    )


# =============================================================================
# Model factory
# =============================================================================
class IdentityCompressedTokenizer:
    """
    Identity compressor for Engram when using an already-tokenized integer space.
    We map tokenizer vocab ids directly to themselves.
    """

    def __init__(self, tokenizer_name_or_path: str, vocab_size: int) -> None:
        self.lookup_table = np.arange(vocab_size, dtype=np.int64)
        self.num_new_token = vocab_size

    def __len__(self) -> int:
        return self.num_new_token

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        return np.asarray(input_ids, dtype=np.int64)


def build_identity_engram_factory(vocab_size: int):
    # Engram expects a callable "tokenizer factory"; this adapter returns one
    # that leaves already-tokenized TRACE ids unchanged.
    class _Factory(IdentityCompressedTokenizer):
        def __init__(self, tokenizer_name_or_path: str) -> None:
            super().__init__(tokenizer_name_or_path, vocab_size=vocab_size)
    return _Factory


def build_model_adapter(
    model_name: str,
    vocab_size: int,
    max_seq_len: int,
    pad_token_id: int,
    device: str,
    epochs: int,
    lr: float,
    log_every: int,
) -> TorchCLModelAdapter:
    # TRACE uses a text tokenizer, so `vocab_size` comes directly from the
    # tokenizer and `max_seq_len` controls the maximum token length after
    # truncation/padding.
    base_cfg = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=256,
        n_heads=8,
        n_layers=6,
        mlp_ratio=4,
        dropout=0.1,
        pad_token_id=pad_token_id,
    )

    train_cfg = TrainingConfig(
        lr=lr,
        epochs=epochs,
        device=device,
        weight_decay=0.01,
        max_grad_norm=1.0,
        log_every=log_every,
    )

    if model_name == "baseline":
        model = StandardTransformerLM(base_cfg)
    elif model_name == "reduced_attention":
        model = ReducedAttentionTransformerLM(base_cfg, attention_every_n_layers=2)
    elif model_name == "mlp_only":
        model = MLPOnlyLM(base_cfg)
    elif model_name == "engram":
        engram_cfg = EngramConfig(
            tokenizer_name_or_path="identity-trace",
            engram_vocab_size=[32000, 32000],
            max_ngram_size=3,
            n_embed_per_ngram=128,
            n_head_per_ngram=4,
            layer_ids=[2, 4],
            pad_id=pad_token_id,
            kernel_size=4,
            compressed_tokenizer_factory=build_identity_engram_factory(vocab_size=vocab_size),
        )
        model = TransformerWithEngramLM(base_cfg, engram_cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return TorchCLModelAdapter(model=model, training_cfg=train_cfg)


# =============================================================================
# Task building
# =============================================================================
def build_trace_dataloader(
    records: Sequence[Dict[str, str]],
    tokenizer,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    # The dataset yields one example at a time; the collator turns a list of
    # variable-length examples into one padded minibatch.
    ds = TraceCausalLMDataset(records, tokenizer=tokenizer, max_seq_len=max_seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=TraceCollator(pad_token_id=tokenizer.pad_token_id),
    )


class CombinedLoader:
    """Simple iterable wrapper that chains multiple evaluation loaders.

    The tester only needs something it can iterate over repeatedly, so this
    lightweight wrapper is enough for cumulative old-domain evaluation.
    """

    def __init__(self, loaders: Sequence[DataLoader]) -> None:
        self.loaders = list(loaders)
        # This synthetic `dataset` length is only used by throughput counters.
        self.dataset = [None] * sum(len(loader.dataset) for loader in self.loaders)

    def __iter__(self):
        for loader in self.loaders:
            yield from loader

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders)


def make_tasks_and_probes(args, tokenizer):
    # Build the continual-learning task stream in the requested order.
    task_dirs = discover_trace_tasks(
        trace_root=args.trace_root,
        task_order=[x.strip() for x in args.task_order.split(",")] if args.task_order else None,
    )

    tasks: List[TaskSpec] = []
    previous_eval_loaders: List[DataLoader] = []
    all_eval_loaders: Dict[str, DataLoader] = {}

    for task_name, task_dir in task_dirs:
        splits = load_trace_task_dir(task_dir)

        train_loader = build_trace_dataloader(
            splits["train"],
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        eval_loader = build_trace_dataloader(
            splits["eval"],
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        all_eval_loaders[task_name] = eval_loader

        tasks.append(
            TaskSpec(
                name=task_name,
                train_data=train_loader,
                eval_data=eval_loader,
                score_fn=trace_score_fn_factory(tokenizer, args.max_new_tokens),
                # Old-domain NLL is evaluated on all earlier task loaders
                # chained together.
                old_domain_eval_data=CombinedLoader(previous_eval_loaders) if previous_eval_loaders else None,
                # New-domain NLL stays focused on the current task.
                new_domain_eval_data=eval_loader,
                metadata={
                    "task_dir": str(task_dir),
                    "train_size": len(splits["train"]),
                    "eval_size": len(splits["eval"]),
                },
            )
        )
        previous_eval_loaders.append(eval_loader)

    general_suites = [make_general_probe_suite(all_eval_loaders, tokenizer, args.max_new_tokens)]
    memory_suites: List[ProbeSuiteSpec] = []
    representation_suites: List[Any] = []

    return tasks, general_suites, memory_suites, representation_suites


# =============================================================================
# Runner
# =============================================================================
def run_one(args, seed: int):
    # One run means: choose one seed, build tokenizer/model/tasks, then pass the
    # whole setup into the shared continual-learning tester.
    set_seed(seed)

    tracker = build_experiment_tracker(args, seed=seed)
    run_started = time.perf_counter()

    logger.info("Starting TRACE run | seed=%d | model=%s | trace_root=%s", seed, args.model, args.trace_root)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

    adapter = build_model_adapter(
        model_name=args.model,
        vocab_size=len(tokenizer),
        max_seq_len=args.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        log_every=args.log_every,
    )

    tasks, general_suites, memory_suites, repr_suites = make_tasks_and_probes(args, tokenizer)

    logger.info(
        "Prepared TRACE run | num_tasks=%d | task_order=%s | vocab_size=%d | device=%s",
        len(tasks),
        ",".join(task.name for task in tasks),
        len(tokenizer),
        args.device,
    )

    tracker.log_hyperparams({
        "dataset": "TRACE",
        "trace_root": args.trace_root,
        "model": args.model,
        "seed": seed,
        "tokenizer_name": args.tokenizer_name,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "max_seq_len": args.max_seq_len,
        "max_new_tokens": args.max_new_tokens,
        "log_every": args.log_every,
        "num_tasks": len(tasks),
        "task_order": [task.name for task in tasks],
        "enable_lightning_logging": args.enable_lightning_logging,
    })

    for task_idx, task in enumerate(tasks):
        logger.info(
            "Task %d/%d | name=%s | train_size=%s | eval_size=%s | task_dir=%s",
            task_idx + 1,
            len(tasks),
            task.name,
            task.metadata.get("train_size"),
            task.metadata.get("eval_size"),
            task.metadata.get("task_dir"),
        )

    # FWT is only computed if explicit baselines are provided. For now we set
    # them to 0.0 as a simple placeholder.
    future_task_baselines = {task.name: 0.0 for task in tasks}
    oracle_new_task_scores = None

    tester = ContinualTester(
        ContinualTesterConfig(
            run_name=f"trace_{args.model}_seed{seed}",
            metadata={
                "dataset": "TRACE",
                "trace_root": args.trace_root,
                "model": args.model,
                "seed": seed,
                "tokenizer_name": args.tokenizer_name,
            },
            evaluate_all_tasks_each_stage=True,
            evaluate_initial_stage=True,
            compute_old_domain_nll=True,
            compute_new_domain_nll=True,
            train_unit_count_fn=count_examples,
            eval_unit_count_fn=count_examples,
            unit_name="examples",
        )
    )

    try:
        report = tester.run(
            model=adapter,
            tasks=tasks,
            general_probe_suites=general_suites,
            memory_probe_suites=memory_suites,
            representation_probe_suites=repr_suites,
            future_task_baselines=future_task_baselines,
            oracle_new_task_scores=oracle_new_task_scores,
        )
    except Exception:
        tracker.finalize(status="failed")
        logger.exception("TRACE run failed | seed=%d | model=%s", seed, args.model)
        raise

    elapsed = time.perf_counter() - run_started
    tracker.log_metrics(summarize_report_metrics(report), step=len(report.stages))
    tracker.log_metrics({"run/total_time_sec": elapsed}, step=len(report.stages))
    tracker.finalize(status="success")

    logger.info(
        "Finished TRACE run | seed=%d | stages=%d | final_acc=%s | mean_forgetting=%s | total_time_sec=%.2f",
        seed,
        len(report.stages),
        report.summary.final_acc,
        report.summary.mean_forgetting,
        elapsed,
    )

    return report


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    # The CLI keeps experiment settings visible and reproducible from the shell.
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=str, required=True,
                        help="Root dir containing TRACE task subdirs, each with train.json/eval.json/test.json")
    parser.add_argument("--task-order", type=str, default="",
                        help="Comma-separated task order, e.g. C-STANCE,FOMC,MeetingBank,Py150")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "engram", "reduced_attention", "mlp_only"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument("--enable-lightning-logging", action="store_true",
                        help="Mirror run metadata/metrics to Lightning-compatible loggers when available")
    parser.add_argument("--lightning-log-dir", type=str, default="lightning_logs",
                        help="Directory used for Lightning-style experiment logs")
    parser.add_argument("--lightning-experiment-name", type=str, default="",
                        help="Optional override for the Lightning experiment name")
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    # A single logging format keeps trainer output and tester output aligned.
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    reports = []
    for run_idx in range(args.num_runs):
        # Each run gets a different seed so multi-run aggregation measures
        # variability instead of duplicating the same result.
        reports.append(run_one(args, seed=args.seed + run_idx))

    if len(reports) == 1:
        final_report = reports[0]
        print(json.dumps(final_report.to_dict(), indent=2))
        if args.save_json:
            final_report.to_json(args.save_json)
            print(f"saved report to {args.save_json}")
    else:
        aggregated = aggregate_reports(reports)
        print(json.dumps(asdict(aggregated), indent=2))
        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(asdict(aggregated), f, indent=2)
            print(f"saved report to {args.save_json}")


if __name__ == "__main__":
    main()
