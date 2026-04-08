"""
SplitMNIST continual-learning experiment using Avalanche for the benchmark stream,
while still using:

- `tester.py` for reporting/metrics
- `cl_models` for the actual architectures

----------------------
Avalanche is the right tool for *standard continual-learning benchmarks* like
SplitMNIST because it gives us a canonical stream of train/test experiences.
However, we still want to evaluate our own architectures and push everything
through our own tester so we keep the richer reporting stack we built.

This script therefore does the following:
1. Uses Avalanche's SplitMNIST benchmark stream.
2. Converts each MNIST sample into a token sequence so it can be consumed by
   the causal-LM-style models currently in `cl_models`.
3. Wraps each Avalanche experience into TaskSpec objects for `tester.py`.
4. Runs the benchmark through our existing continual-learning tester.

How this file fits into the project
-----------------------------------
- `cl_models/` defines the actual neural-network architectures.
- `tester.py` defines the continual-learning evaluation loop and report format.
- this file is the benchmark-specific glue that turns SplitMNIST into the
  tensors, tasks, and probes those shared components understand.

Sequence format
---------------
Each image becomes:
    [pixel_1, pixel_2, ..., pixel_784, SEP, LABEL]

Only the final LABEL token is supervised. This lets us reuse the same model
family for now instead of introducing a separate classifier family.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# Avalanche import.
# We try multiple import styles because Avalanche docs/examples vary a bit
# across versions.
# -----------------------------------------------------------------------------
try:
    from avalanche.benchmarks.classic import SplitMNIST
except Exception:
    try:
        from avalanche.benchmarks import SplitMNIST
    except Exception as exc:
        raise ImportError(
            "Avalanche is required for this script. Install it with: pip install avalanche-lib"
        ) from exc

# -----------------------------------------------------------------------------
# Import the tester you already created.
# -----------------------------------------------------------------------------
from tester import (
    TaskSpec,
    ProbeSuiteSpec,
    RepresentationProbeSpec,
    ContinualTester,
    ContinualTesterConfig,
    aggregate_reports,
)

# -----------------------------------------------------------------------------
# Import the architecture package.
# -----------------------------------------------------------------------------
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


# =============================================================================
# Token-space design for MNIST-as-sequence
# =============================================================================
PAD_ID = 0
PIXEL_OFFSET = 1            # 0..255 -> 1..256
SEP_ID = 257                # separator before the label token
LABEL_OFFSET = 258          # class 0..9 -> 258..267
VOCAB_SIZE = 268            # token ids 0..267 inclusive
SEQ_LEN = 28 * 28 + 2       # pixels + SEP + LABEL
LABEL_PREDICTION_INDEX = -2  # The SEP-position hidden state predicts the trailing label token.


# =============================================================================
# Dataset wrappers
# =============================================================================
class ExperienceSequenceDataset(Dataset):
    """
    Wraps an Avalanche experience dataset as a causal-LM dataset.

    The wrapped dataset is expected to yield (image, label) pairs.
    We turn that into a sequence where the model predicts the class token.
    """

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]

        # Some Avalanche datasets may return tuples with task labels.
        # We only need image + class label here.
        if len(sample) >= 2:
            image, label = sample[0], sample[1]
        else:
            raise ValueError("Unexpected sample format from Avalanche dataset.")

        # Ensure tensor image in [0, 1]. If the benchmark is already transformed,
        # this is usually a torch.Tensor with shape [1, 28, 28].
        if not torch.is_tensor(image):
            image = transforms.ToTensor()(image)

        # `image` starts as [1, 28, 28]. Flattening changes it to [784], one
        # scalar per pixel, which we then remap into a token-id range.
        pixel_values = (image.view(-1) * 255.0).round().long()
        pixel_tokens = pixel_values + PIXEL_OFFSET
        sep_token = torch.tensor([SEP_ID], dtype=torch.long)
        label_token = torch.tensor([LABEL_OFFSET + int(label)], dtype=torch.long)

        # Only the final label token is supervised; the earlier positions act as
        # context so we can reuse a next-token-prediction architecture.
        # Final sizes:
        # - `input_ids`: [786]  -> 784 pixel tokens + 1 SEP + 1 LABEL
        # - `labels`:    [786]  -> mostly IGNORE_INDEX except the last token
        input_ids = torch.cat([pixel_tokens, sep_token, label_token], dim=0)
        labels = torch.full_like(input_ids, fill_value=-100)
        labels[-1] = label_token.item()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "class_label": torch.tensor(int(label), dtype=torch.long),
            "label_token": label_token.squeeze(0),
        }


# =============================================================================
# Engram patch for non-text token spaces
# =============================================================================
class IdentityCompressedTokenizer:
    """
    Identity token compressor for synthetic token spaces like SplitMNIST.

    The Engram implementation in `cl_models` is adapted from a text-tokenizer-
    based design. For SplitMNIST, the token IDs are already canonical, so we
    patch in an identity mapping.
    """

    def __init__(self, vocab_size: int) -> None:
        self.lookup_table = np.arange(vocab_size, dtype=np.int64)
        self.num_new_token = vocab_size

    def __len__(self) -> int:
        return self.num_new_token

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        return np.asarray(input_ids, dtype=np.int64)


def build_splitmnist_engram_identity_tokenizer_factory(vocab_size: int = VOCAB_SIZE):
    """Create a tokenizer factory that behaves like an identity mapping.

    The Engram module expects to receive something tokenizer-like, but
    SplitMNIST already uses a hand-crafted token space, so no text tokenizer is
    needed.
    """
    class _IdentityTokenizerForSplitMNIST(IdentityCompressedTokenizer):
        def __init__(self, tokenizer_name_or_path: str) -> None:
            super().__init__(vocab_size=vocab_size)

    return _IdentityTokenizerForSplitMNIST


# =============================================================================
# Avalanche benchmark helpers
# =============================================================================
@dataclass
class AvalancheTaskBundle:
    task_name: str
    classes: Tuple[int, ...]
    train_loader: DataLoader
    eval_loader: DataLoader
    support_loader: DataLoader
    probe_loader: DataLoader


class CombinedLoader:
    """Re-iterable view over multiple dataloaders."""

    def __init__(self, loaders: Sequence[DataLoader]) -> None:
        self.loaders = list(loaders)
        self.dataset = ConcatDataset([loader.dataset for loader in self.loaders])

    def __iter__(self):
        # Re-create each loader iterator every time so the wrapper can be used
        # repeatedly across evaluation stages.
        for loader in self.loaders:
            yield from loader

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders)


class ClassSubsetDataset(Dataset):
    """
    Utility dataset for per-class support/probe subsets built from torchvision MNIST.
    This is only used for the probe suites.
    """

    def __init__(self, base_dataset: Dataset, indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]


class WrappedSequenceSubset(Dataset):
    """Compose an arbitrary image/label dataset with sequence-tokenization logic.

    This keeps support/probe subsets consistent with the main training/eval
    data format without needing to duplicate raw subset construction logic.
    """

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        if not torch.is_tensor(image):
            image = transforms.ToTensor()(image)
        # This mirrors `ExperienceSequenceDataset.__getitem__` so probe/support
        # batches have exactly the same token format as the main train/eval data.
        pixel_values = (image.view(-1) * 255.0).round().long()
        pixel_tokens = pixel_values + PIXEL_OFFSET
        sep_token = torch.tensor([SEP_ID], dtype=torch.long)
        label_token = torch.tensor([LABEL_OFFSET + int(label)], dtype=torch.long)
        input_ids = torch.cat([pixel_tokens, sep_token, label_token], dim=0)
        labels = torch.full_like(input_ids, fill_value=-100)
        labels[-1] = label_token.item()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "class_label": torch.tensor(int(label), dtype=torch.long),
            "label_token": label_token.squeeze(0),
        }



def take_per_class_indices(
    targets: Sequence[int],
    classes: Sequence[int],
    k: int,
    seed: int,
    exclude_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """Sample up to `k` examples per class with an optional exclusion set.

    The exclusion hook is used to guarantee that support and probe subsets do
    not accidentally share examples.
    """
    rng = random.Random(seed)
    excluded = {int(idx) for idx in (exclude_indices or [])}
    out: List[int] = []
    for c in classes:
        cls_idx = [i for i, y in enumerate(targets) if int(y) == int(c) and i not in excluded]
        rng.shuffle(cls_idx)
        out.extend(cls_idx[: min(k, len(cls_idx))])
    return out


def build_avalanche_splitmnist_bundles(
    data_root: str,
    batch_size: int,
    num_workers: int,
    support_per_class: int,
    probe_per_class: int,
    seed: int,
) -> Tuple[List[AvalancheTaskBundle], DataLoader, Dict[int, DataLoader]]:
    """
    Build Avalanche SplitMNIST task bundles, plus the probe loaders needed by
    the richer reporting suites.
    """
    benchmark = SplitMNIST(
        n_experiences=5,
        seed=seed,
        shuffle=False,
        return_task_id=False,
        class_ids_from_zero_in_each_exp=False,
        train_transform=None,
        eval_transform=None,
        dataset_root=data_root,
    )

    bundles: List[AvalancheTaskBundle] = []

    # Build one task bundle per Avalanche experience.
    for exp_idx, (train_exp, test_exp) in enumerate(zip(benchmark.train_stream, benchmark.test_stream)):
        classes = tuple(int(c) for c in sorted(train_exp.classes_in_this_experience))
        task_name = f"splitmnist_{exp_idx}_{'_'.join(str(c) for c in classes)}"

        # Avalanche provides the canonical train/eval experience split; the
        # wrapper simply converts each sample into the token-sequence format our
        # language-model code expects.
        train_seq = ExperienceSequenceDataset(train_exp.dataset)
        test_seq = ExperienceSequenceDataset(test_exp.dataset)

        train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        eval_loader = DataLoader(test_seq, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Probe loaders are built from raw torchvision MNIST for simplicity and control.
        raw_test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=transforms.ToTensor())
        support_indices = take_per_class_indices(raw_test_ds.targets, classes, k=support_per_class, seed=seed + exp_idx)
        probe_indices = take_per_class_indices(
            raw_test_ds.targets,
            classes,
            k=probe_per_class,
            seed=seed + 1000 + exp_idx,
            exclude_indices=support_indices,
        )
        support_ds = WrappedSequenceSubset(ClassSubsetDataset(raw_test_ds, support_indices))
        probe_ds = WrappedSequenceSubset(ClassSubsetDataset(raw_test_ds, probe_indices))

        support_loader = DataLoader(support_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        probe_loader = DataLoader(probe_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        bundles.append(
            AvalancheTaskBundle(
                task_name=task_name,
                classes=classes,
                train_loader=train_loader,
                eval_loader=eval_loader,
                support_loader=support_loader,
                probe_loader=probe_loader,
            )
        )

    # Overall evaluation loader across the whole test set.
    raw_test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=transforms.ToTensor())
    overall_eval_loader = DataLoader(
        WrappedSequenceSubset(raw_test_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Per-digit probe loaders for memory retention views.
    digit_probe_loaders: Dict[int, DataLoader] = {}
    for digit in range(10):
        digit_indices = take_per_class_indices(raw_test_ds.targets, [digit], k=probe_per_class, seed=seed + 5000 + digit)
        digit_ds = WrappedSequenceSubset(ClassSubsetDataset(raw_test_ds, digit_indices))
        digit_probe_loaders[digit] = DataLoader(digit_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return bundles, overall_eval_loader, digit_probe_loaders


# =============================================================================
# Metrics / probes
# =============================================================================
@torch.no_grad()
def splitmnist_label_accuracy(adapter: TorchCLModelAdapter, eval_loader: Iterable[Dict[str, torch.Tensor]]) -> float:
    """Accuracy on the final class token.

    Even though the class token sits at index `-1`, the causal language model
    predicts it from the hidden state at the previous position, which is why the
    logits come from `LABEL_PREDICTION_INDEX`.
    """
    model = adapter.model.to(adapter.training_cfg.device)
    model.eval()

    total_correct = 0
    total_count = 0
    label_slice = slice(LABEL_OFFSET, LABEL_OFFSET + 10)

    for batch in eval_loader:
        batch = {k: v.to(adapter.training_cfg.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"])
        # `outputs["logits"]` has shape [B, 786, 268]. Selecting
        # `LABEL_PREDICTION_INDEX` keeps only the time step that predicts the
        # class token, and `label_slice` narrows the vocabulary down to the 10
        # class-token ids. The result has shape [B, 10].
        logits = outputs["logits"][:, LABEL_PREDICTION_INDEX, label_slice]
        preds = logits.argmax(dim=-1)
        targets = batch["label_token"] - LABEL_OFFSET
        total_correct += int((preds == targets).sum().item())
        total_count += int(targets.numel())

    return total_correct / max(total_count, 1)


@torch.no_grad()
def collect_label_prediction_representations(
    adapter: TorchCLModelAdapter,
    loader: Iterable[Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect the final-layer hidden state at the label-prediction position.

    In this sequence format, that is the SEP token position because its hidden
    state predicts the next LABEL token in the causal LM.
    """
    model = adapter.model.to(adapter.training_cfg.device)
    model.eval()

    reps: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for batch in loader:
        batch = {k: v.to(adapter.training_cfg.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"], return_hidden_states=True)
        hidden_states = outputs["hidden_states"][-1]
        # `hidden_states` is [B, 786, D]. Selecting one time step changes it to
        # [B, D], one representation vector per example.
        reps.append(hidden_states[:, LABEL_PREDICTION_INDEX, :].detach().cpu())
        labels.append(batch["class_label"].detach().cpu())

    # Concatenation stacks all minibatches into:
    # - representations: [N_examples, D]
    # - labels:          [N_examples]
    return torch.cat(reps, dim=0), torch.cat(labels, dim=0)


def nearest_centroid_accuracy(
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    probe_x: torch.Tensor,
    probe_y: torch.Tensor,
) -> float:
    """Simple no-dependency representation probe.

    The support set defines one centroid per class; the probe set is classified
    by whichever centroid is closest in Euclidean distance.
    """
    classes = sorted(set(int(c) for c in support_y.tolist()))
    centroids = []
    centroid_labels = []
    for c in classes:
        mask = support_y == c
        # `support_x[mask]` is [N_c, D]; averaging over examples gives one
        # centroid vector [1, D] for class `c`.
        centroids.append(support_x[mask].mean(dim=0, keepdim=True))
        centroid_labels.append(c)
    # After concatenation, `centroids_t` is [num_classes, D].
    centroids_t = torch.cat(centroids, dim=0)

    # `torch.cdist(probe_x, centroids_t)` returns [N_probe, num_classes].
    dists = torch.cdist(probe_x, centroids_t)
    pred_idx = dists.argmin(dim=1)
    preds = torch.tensor([centroid_labels[i] for i in pred_idx.tolist()])
    return float((preds == probe_y).float().mean().item())



def make_general_probe_suite(overall_eval_loader: DataLoader) -> ProbeSuiteSpec:
    """General probe = overall accuracy on all 10 digits."""
    def general_probe_fn(adapter: TorchCLModelAdapter, data: DataLoader, ctx) -> Dict[str, float]:
        return {"overall_splitmnist_accuracy": splitmnist_label_accuracy(adapter, data)}

    return ProbeSuiteSpec(
        name="general_splitmnist_suite",
        data=overall_eval_loader,
        run_fn=general_probe_fn,
    )



def make_memory_probe_suite(digit_probe_loaders: Dict[int, DataLoader]) -> ProbeSuiteSpec:
    """Memory probe = per-digit held-out probe accuracy."""
    def memory_probe_fn(adapter: TorchCLModelAdapter, data: Dict[int, DataLoader], ctx) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for digit, loader in data.items():
            # Per-digit slices make it easier to see which classes are retained
            # or forgotten after later tasks arrive.
            out[f"digit_{digit}_probe_acc"] = splitmnist_label_accuracy(adapter, loader)
        return out

    return ProbeSuiteSpec(
        name="digit_memory_probe_suite",
        data=digit_probe_loaders,
        run_fn=memory_probe_fn,
    )



def make_representation_probe_suite(bundles: List[AvalancheTaskBundle]) -> RepresentationProbeSpec:
    """
    Representation probe = nearest-centroid accuracy on the last-layer state
    used to predict the label token.
    """
    name_to_bundle = {bundle.task_name: bundle for bundle in bundles}

    def representation_probe_fn(adapter: TorchCLModelAdapter, data: List[AvalancheTaskBundle], ctx) -> Dict[str, float]:
        if not ctx.seen_task_names:
            return {"centroid_probe_acc": 0.0}

        accs: List[float] = []
        for task_name in ctx.seen_task_names:
            # Each seen task contributes its own support/probe classification
            # problem, and the suite reports their mean.
            bundle = name_to_bundle[task_name]
            support_x, support_y = collect_label_prediction_representations(adapter, bundle.support_loader)
            probe_x, probe_y = collect_label_prediction_representations(adapter, bundle.probe_loader)
            accs.append(nearest_centroid_accuracy(support_x, support_y, probe_x, probe_y))

        return {"centroid_probe_acc": float(sum(accs) / max(len(accs), 1))}

    return RepresentationProbeSpec(
        name="representation_probe_suite",
        layers=["last_hidden_state"],
        data=bundles,
        run_fn=representation_probe_fn,
    )


# =============================================================================
# Model factory
# =============================================================================
def build_model_adapter(
    model_name: str,
    device: str,
    epochs: int,
    lr: float,
    log_every: int,
) -> TorchCLModelAdapter:
    """Instantiate one of the models from cl_models and wrap it in the adapter."""
    # All SplitMNIST variants share the same token space and sequence length, so
    # comparisons across models stay fair.
    base_cfg = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=SEQ_LEN,
        d_model=256,
        n_heads=8,
        n_layers=6,
        mlp_ratio=4,
        dropout=0.1,
        pad_token_id=PAD_ID,
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
        # SplitMNIST uses synthetic token ids rather than text-tokenizer ids, so
        # Engram receives an injected identity tokenizer factory.
        engram_cfg = EngramConfig(
            tokenizer_name_or_path="identity-splitmnist",
            engram_vocab_size=[32_000, 32_000],
            max_ngram_size=3,
            n_embed_per_ngram=128,
            n_head_per_ngram=4,
            layer_ids=[2, 4],
            pad_id=PAD_ID,
            kernel_size=4,
            compressed_tokenizer_factory=build_splitmnist_engram_identity_tokenizer_factory(vocab_size=VOCAB_SIZE),
        )
        model = TransformerWithEngramLM(base_cfg, engram_cfg)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return TorchCLModelAdapter(model=model, training_cfg=train_cfg)


# =============================================================================
# Throughput counters for tester resource fields
# =============================================================================
def count_examples(data: Any) -> Optional[int]:
    """Best-effort example counter for throughput reporting."""
    if hasattr(data, "dataset"):
        return len(data.dataset)
    if hasattr(data, "__len__"):
        try:
            return len(data)
        except TypeError:
            return None
    return None


# =============================================================================
# Experiment builder
# =============================================================================
def make_tasks_and_probes(args):
    """Create tester task specs plus the three probe-suite families.

    Old-domain loaders are constructed cumulatively from previously seen tasks
    so retention NLL measures past-task behavior rather than current-task
    performance repeated under a different name.
    """
    bundles, overall_eval_loader, digit_probe_loaders = build_avalanche_splitmnist_bundles(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        support_per_class=args.support_per_class,
        probe_per_class=args.probe_per_class,
        seed=args.seed,
    )

    tasks: List[TaskSpec] = []
    previous_eval_loaders: List[DataLoader] = []
    for bundle in bundles:
        tasks.append(
            TaskSpec(
                name=bundle.task_name,
                train_data=bundle.train_loader,
                eval_data=bundle.eval_loader,
                score_fn=splitmnist_label_accuracy,
                # Old-domain NLL is evaluated on the union of earlier tasks.
                old_domain_eval_data=CombinedLoader(previous_eval_loaders) if previous_eval_loaders else None,
                # New-domain NLL stays task-local and tracks plasticity.
                new_domain_eval_data=bundle.eval_loader,
                metadata={"classes": bundle.classes},
            )
        )
        # The current task becomes part of the "old domain" pool for all later
        # tasks once it has been appended here.
        previous_eval_loaders.append(bundle.eval_loader)

    general_suites = [make_general_probe_suite(overall_eval_loader)]
    memory_suites = [make_memory_probe_suite(digit_probe_loaders)]
    repr_suites = [make_representation_probe_suite(bundles)]
    return tasks, general_suites, memory_suites, repr_suites



def run_one(args, seed: int):
    """Run one full continual-learning experiment for a given seed."""
    # Seed all libraries before building the model or dataloaders so random task
    # sampling and parameter initialization stay reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    adapter = build_model_adapter(args.model, args.device, args.epochs, args.lr, args.log_every)
    tasks, general_suites, memory_suites, repr_suites = make_tasks_and_probes(args)

    # Explicit FWT baselines. We leave them at 0 for this starter benchmark,
    # but they are still passed explicitly so tester.py doesn't silently invent them.
    future_task_baselines = {task.name: 0.0 for task in tasks}

    # Optional oracle scores for intransigence. None for now.
    oracle_new_task_scores = None

    tester = ContinualTester(
        ContinualTesterConfig(
            run_name=f"splitmnist_avalanche_{args.model}_seed{seed}",
            metadata={
                "dataset": "SplitMNIST(Avalanche)",
                "model": args.model,
                "seed": seed,
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

    return tester.run(
        model=adapter,
        tasks=tasks,
        general_probe_suites=general_suites,
        memory_probe_suites=memory_suites,
        representation_probe_suites=repr_suites,
        future_task_baselines=future_task_baselines,
        oracle_new_task_scores=oracle_new_task_scores,
    )


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Build the CLI used for quick local experiments and smoke tests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "engram", "reduced_attention", "mlp_only"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--support-per-class", type=int, default=16)
    parser.add_argument("--probe-per-class", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--save-json", type=str, default="")
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    """Configure simple console logging for trainers and the tester."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    reports = []
    for run_idx in range(args.num_runs):
        # Each run shifts the seed so aggregated multi-run experiments are
        # deterministic but not identical copies.
        reports.append(run_one(args, seed=args.seed + run_idx))

    if len(reports) == 1:
        final_report = reports[0]
        print(final_report.to_dict())
        if args.save_json:
            final_report.to_json(args.save_json)
            print(f"saved report to {args.save_json}")
    else:
        aggregated_report = aggregate_reports(reports)
        print(json.dumps(asdict(aggregated_report), indent=2))
        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(asdict(aggregated_report), f, indent=2)
            print(f"saved report to {args.save_json}")


if __name__ == "__main__":
    main()
