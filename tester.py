"""Architecture-agnostic continual-learning evaluation utilities.

The central idea of this module is to separate *how a model learns* from *how
we evaluate it over a sequence of tasks*. Any model that implements the small
`ModelAdapter` protocol can plug into the same reporting stack and receive:

- per-stage task scores
- optional old/new-domain NLL diagnostics
- arbitrary probe suites
- summary metrics such as forgetting, BWT, and FWT
- JSON-friendly reports for later analysis

How it fits into the project:
- files in `cl_models/` define trainable model architectures
- experiment scripts such as `trace_experiment.py` and
  `splitmnist_experiment.py` turn raw benchmarks into tasks and probe suites
- this file runs the shared train/evaluate/report loop once those pieces have
  been assembled
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union
import copy
import json
import logging
import math
import statistics
import time


# =============================================================================
# Type aliases
# =============================================================================

# Generic scoring callback: given a model adapter and some data, return a float.
ScoreFn = Callable[["ModelAdapter", Any], float]

# Generic probe callback: given model adapter + probe data + stage context,
# return a dictionary of scalar metrics.
ProbeFn = Callable[["ModelAdapter", Any, "StageContext"], Dict[str, float]]

# Optional resource callback for user-defined memory/FLOP hooks.
# The callback can return any numeric fields. Known keys are pulled into
# explicit ResourceSnapshot fields; everything else goes into `extra`.
ResourceFn = Callable[["ModelAdapter"], Dict[str, Union[int, float]]]

# Optional unit-count callback for throughput.
# Example: number of examples, number of tokens, number of sequences, etc.
UnitCountFn = Callable[[Any], Optional[int]]

logger = logging.getLogger(__name__)


# =============================================================================
# Adapter protocol
# =============================================================================

class ModelAdapter(Protocol):
    """
    Architecture-agnostic interface for the tester.

    Any architecture can be evaluated as long as you wrap it with these methods.
    """

    def clone(self) -> "ModelAdapter":
        """Return a deep copy if you need an untouched reference/oracle model."""
        ...

    def train_on_task(self, task: "TaskSpec") -> Dict[str, Any]:
        """
        Train / adapt the model on one task.

        Returns an optional log dictionary containing anything useful:
        loss curves, step counts, wall-clock time, etc.
        """
        ...

    def score(self, data: Any, score_fn: ScoreFn) -> float:
        """
        Evaluate the model on arbitrary data using the provided score_fn.
        """
        ...

    def nll(self, data: Any) -> float:
        """
        Return negative log-likelihood on data.
        If your model cannot do this, raise NotImplementedError.
        """
        ...

    def num_parameters(self) -> int:
        """Total parameter count."""
        ...

    def num_trainable_parameters(self) -> int:
        """Trainable parameter count."""
        ...

    def active_parameter_estimate(self) -> Optional[int]:
        """
        Return active parameter estimate if sparse; otherwise return None.
        """
        return None

    def flops_estimate(self) -> Optional[float]:
        """
        Return approximate FLOPs for current forward/inference setting if available.
        """
        return None

    def get_hidden_states(self, data: Any, layers: Sequence[str]) -> Dict[str, Any]:
        """
        Optional hook for representation probe suites.
        If unsupported, raise NotImplementedError.
        """
        raise NotImplementedError("Hidden-state extraction is not implemented.")


# =============================================================================
# Specs for tasks and probe suites
# =============================================================================

@dataclass
class TaskSpec:
    """
    One continual-learning task or domain stage.
    """
    name: str
    train_data: Any
    eval_data: Any
    score_fn: ScoreFn

    # Optional old-domain eval set for retention-side language-model metrics.
    old_domain_eval_data: Optional[Any] = None

    # Optional new-domain eval set for plasticity-side language-model metrics.
    # If absent, the tester can fall back to eval_data if you want, but keeping
    # it explicit is cleaner because not all eval_data objects are LM-compatible.
    new_domain_eval_data: Optional[Any] = None

    # Optional per-task metadata (domain name, language, tags, counts, etc.).
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeSuiteSpec:
    """
    Generic suite for any post-stage probe:
    - entity retention
    - factual retention
    - general benchmarks
    - custom diagnostics
    """
    name: str
    data: Any
    run_fn: ProbeFn
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepresentationProbeSpec:
    """
    Specialized representation probe definition.

    This is separate only because representation probes often require selected
    layers plus train/eval examples for probe fitting.
    """
    name: str
    layers: List[str]
    data: Any
    run_fn: ProbeFn
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Stage / report objects
# =============================================================================

@dataclass
class StageContext:
    """
    Context passed into probes and post-stage hooks.

    Probes receive this so they know which stage they are running at and which
    tasks have already been trained.
    """
    stage_idx: int
    trained_task_name: Optional[str]
    seen_task_names: List[str]
    all_task_names: List[str]
    is_initial: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceSnapshot:
    """
    Resource metrics for one stage.

    Throughput:
    - `train_units` and `eval_units` are generic. The caller decides what a unit is.
      Examples: examples, sequences, tokens, documents.
    - `train_units_per_sec` and `eval_units_per_sec` mirror that same unit.

    This object is attached to each stage report so performance analysis and
    quality analysis live in the same final report.
    """
    train_time_sec: Optional[float] = None
    eval_time_sec: Optional[float] = None

    total_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None
    active_parameters: Optional[int] = None
    flops_estimate: Optional[float] = None

    train_units: Optional[int] = None
    eval_units: Optional[int] = None
    train_units_per_sec: Optional[float] = None
    eval_units_per_sec: Optional[float] = None

    peak_gpu_memory_bytes: Optional[int] = None
    peak_cpu_memory_bytes: Optional[int] = None

    extra: Dict[str, Union[int, float]] = field(default_factory=dict)


@dataclass
class StageReport:
    """
    Results after one training stage (or initial stage 0).

    One `StageReport` answers the question:
    "After this training step, how well does the model perform and what did it
    cost to get here?"
    """
    stage_idx: int
    trained_task_name: Optional[str]

    # Scores on each task after this stage.
    task_scores: Dict[str, float] = field(default_factory=dict)

    # Optional old-domain retention metrics.
    old_domain_nll: Dict[str, float] = field(default_factory=dict)
    old_domain_ppl: Dict[str, float] = field(default_factory=dict)

    # Optional new-domain plasticity metrics.
    new_domain_nll: Dict[str, float] = field(default_factory=dict)
    new_domain_ppl: Dict[str, float] = field(default_factory=dict)

    # Arbitrary probe suites (fact/entity/general/representation).
    general_probes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    memory_probes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    representation_probes: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Optional raw logs from the train adapter.
    train_log: Dict[str, Any] = field(default_factory=dict)

    # Resource usage for this stage.
    resources: ResourceSnapshot = field(default_factory=ResourceSnapshot)


@dataclass
class SummaryMetrics:
    """
    Top-level scalar summary metrics derived from all stage reports.

    These are the headline numbers most experiments compare first, while the raw
    per-stage/per-task details remain available in the full report.
    """
    final_acc: Optional[float] = None
    mean_forgetting: Optional[float] = None
    worst_forgetting: Optional[float] = None
    bwt: Optional[float] = None
    fwt: Optional[float] = None
    intransigence: Optional[float] = None

    # Raw per-task summaries.
    per_task_final: Dict[str, float] = field(default_factory=dict)
    per_task_best: Dict[str, float] = field(default_factory=dict)
    per_task_forgetting: Dict[str, float] = field(default_factory=dict)
    immediate_new_task_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RunReport:
    """
    Final report object for one continual-learning run.

    This is the main output artifact that experiment scripts print or serialize.
    """
    metadata: Dict[str, Any]
    task_order: List[str]
    stages: List[StageReport]
    accuracy_matrix: Dict[str, Dict[str, Optional[float]]]
    summary: SummaryMetrics

    # Optional auxiliary reports.
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        # `sanitize_for_json` keeps downstream report writing resilient even if
        # training logs include arrays, tensors, or dataclass instances.
        return sanitize_for_json({
            "metadata": self.metadata,
            "task_order": self.task_order,
            "stages": [self._stage_to_dict(s) for s in self.stages],
            "accuracy_matrix": self.accuracy_matrix,
            "summary": asdict(self.summary),
            "notes": self.notes,
        })

    def to_json(self, path: str) -> None:
        """Save report as JSON, sanitizing non-JSON-native values."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def _stage_to_dict(stage: StageReport) -> Dict[str, Any]:
        return sanitize_for_json(asdict(stage))


# =============================================================================
# JSON sanitization helpers
# =============================================================================

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert common non-JSON objects into JSON-safe equivalents.

    Handles:
    - dataclasses
    - dict / list / tuple / set
    - objects with .tolist() (e.g. numpy arrays, torch tensors on CPU)
    - objects with .item() for scalar conversion
    - bytes
    - fallback to str(...) for unsupported custom objects

    This keeps report serialization robust even if train logs contain arrays/tensors.
    """
    # Primitive JSON-safe types.
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Dataclasses.
    if is_dataclass(obj):
        # Convert dataclasses to dictionaries first, then sanitize recursively.
        return sanitize_for_json(asdict(obj))

    # Mappings.
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    # Sequences.
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]

    # Bytes.
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")

    # Numpy arrays / torch tensors / similar.
    if hasattr(obj, "tolist"):
        try:
            return sanitize_for_json(obj.tolist())
        except Exception:
            pass

    # Numpy scalar / torch scalar / similar.
    if hasattr(obj, "item"):
        try:
            return sanitize_for_json(obj.item())
        except Exception:
            pass

    # Last resort: stringify.
    return str(obj)


def safe_exp(value: float, clamp_max: float = 88.0) -> float:
    """Exponentiate safely for report-friendly perplexity values.

    Very large NLL values can overflow `exp`; clamping preserves a monotonic
    perplexity-like quantity while keeping the report numerically stable.
    """
    return math.exp(min(float(value), clamp_max))


# =============================================================================
# Utility functions for CL metrics
# =============================================================================

def safe_mean(values: List[float]) -> Optional[float]:
    """Return mean(values) or None if empty."""
    return statistics.mean(values) if values else None


def task_to_training_position(stage_reports: List[StageReport]) -> Dict[str, int]:
    """
    Map each task name to the index in `stage_reports` where that task was trained.

    This is more robust than assuming:
      stage index = task index + 1

    and works whether or not there is an initial evaluation stage.
    """
    mapping: Dict[str, int] = {}
    for pos, stage in enumerate(stage_reports):
        if stage.trained_task_name is not None and stage.trained_task_name not in mapping:
            # The first stage whose `trained_task_name` matches a task is the
            # stage immediately after that task was trained.
            mapping[stage.trained_task_name] = pos
    return mapping


def compute_per_task_immediate(stage_reports: List[StageReport], task_names: List[str]) -> Dict[str, float]:
    """
    Immediate score on each task right after that task is learned.

    This implementation does NOT assume an initial stage exists.
    It simply finds the stage whose `trained_task_name == task`.
    """
    immediate: Dict[str, float] = {}
    train_pos = task_to_training_position(stage_reports)

    for task in task_names:
        pos = train_pos.get(task)
        if pos is None:
            continue
        stage = stage_reports[pos]
        # We only keep a value if that task was actually evaluated at that stage.
        if task in stage.task_scores:
            immediate[task] = stage.task_scores[task]

    return immediate


def compute_per_task_best_after_learning(
    stage_reports: List[StageReport],
    task_names: List[str],
) -> Dict[str, float]:
    """
    Best score observed for each task AFTER that task has been learned.

    This avoids counting pre-learning scores as the task's "best", which would
    distort forgetting metrics in some setups.
    """
    best: Dict[str, float] = {}
    train_pos = task_to_training_position(stage_reports)

    for task in task_names:
        pos = train_pos.get(task)
        if pos is None:
            continue

        # Only consider stages at or after the task was first trained.
        vals = [
            s.task_scores[task]
            for s in stage_reports[pos:]
            if task in s.task_scores
        ]
        if vals:
            best[task] = max(vals)

    return best


def compute_per_task_final(stage_reports: List[StageReport], task_names: List[str]) -> Dict[str, float]:
    """
    Final-stage score for each task.
    """
    # The final stage acts as the "after all tasks" evaluation snapshot.
    final_stage = stage_reports[-1]
    return {task: final_stage.task_scores[task] for task in task_names if task in final_stage.task_scores}


def compute_per_task_forgetting(
    stage_reports: List[StageReport],
    task_names: List[str],
    use_immediate_reference: bool = False,
) -> Dict[str, float]:
    """
    Forgetting per task.

    If use_immediate_reference=False:
      forgetting(task_j) = best_score_after_learning(task_j) - final_score(task_j)

    If use_immediate_reference=True:
      forgetting(task_j) = immediate_post_learning_score(task_j) - final_score(task_j)

    The best-after-learning version is more common for "worst damage from peak".
    The immediate version is closer to "damage since first acquisition".
    """
    final_scores = compute_per_task_final(stage_reports, task_names)
    immediate_scores = compute_per_task_immediate(stage_reports, task_names)
    best_scores = compute_per_task_best_after_learning(stage_reports, task_names)

    forgetting: Dict[str, float] = {}
    for task in task_names:
        if task not in final_scores:
            continue

        if use_immediate_reference:
            ref = immediate_scores.get(task)
        else:
            ref = best_scores.get(task)

        if ref is not None:
            # Positive forgetting means the model ended worse than its earlier
            # reference point; negative values imply later improvement.
            forgetting[task] = ref - final_scores[task]

    return forgetting


def compute_final_acc(stage_reports: List[StageReport], task_names: List[str]) -> Optional[float]:
    """
    Final average task score across tasks at the final stage.
    """
    final_scores = compute_per_task_final(stage_reports, task_names)
    # This is simply the mean of the final row of the accuracy matrix.
    return safe_mean(list(final_scores.values()))


def compute_bwt(stage_reports: List[StageReport], task_names: List[str]) -> Optional[float]:
    """
    Backward Transfer (BWT).

    Standard practical interpretation:
      average over tasks j of (final_score_on_task_j - immediate_post_learning_score_on_task_j)

    Negative values imply forgetting.
    """
    final_scores = compute_per_task_final(stage_reports, task_names)
    immediate_scores = compute_per_task_immediate(stage_reports, task_names)

    vals: List[float] = []
    for task in task_names:
        if task in final_scores and task in immediate_scores:
            # Improvement after first learning yields positive BWT; forgetting
            # yields negative BWT.
            vals.append(final_scores[task] - immediate_scores[task])

    return safe_mean(vals)


def compute_fwt(
    stage_reports: List[StageReport],
    task_names: List[str],
    future_task_baselines: Optional[Dict[str, float]],
) -> Optional[float]:
    """
    Forward Transfer (FWT).

    We define:
      for task j, use the score on task j *before* training task j,
      minus an explicit baseline score for task j.

    IMPORTANT:
    - If `future_task_baselines` is None, this returns None.
    - It does NOT silently default to 0.0, because that is usually wrong.
    - If a baseline dict is provided but is missing a task, a KeyError is raised.

    This makes FWT opt-in and explicit.
    """
    if future_task_baselines is None:
        return None

    vals: List[float] = []
    train_pos = task_to_training_position(stage_reports)

    for task in task_names:
        if task not in future_task_baselines:
            raise KeyError(f"Missing FWT baseline for task: {task}")

        pos = train_pos.get(task)
        if pos is None:
            continue

        # Pre-task stage is the report immediately before the task-training stage.
        pre_pos = pos - 1
        if pre_pos < 0:
            # No pre-task stage exists, so skip.
            continue

        pre_stage_score = stage_reports[pre_pos].task_scores.get(task)
        if pre_stage_score is None:
            # If the future task was not evaluated before training, skip.
            continue

        baseline = future_task_baselines[task]
        # Positive FWT means the model already does better on the future task
        # than the provided baseline before that task is trained.
        vals.append(pre_stage_score - baseline)

    return safe_mean(vals)


def compute_intransigence(
    stage_reports: List[StageReport],
    task_names: List[str],
    oracle_new_task_scores: Optional[Dict[str, float]],
) -> Optional[float]:
    """
    Intransigence measures inability to learn new tasks well.

    Here we use:
      oracle_new_task_score(task_j) - immediate_post_learning_score(task_j)

    Lower is better; 0 means the continual learner matches the oracle.
    """
    if oracle_new_task_scores is None:
        return None

    immediate_scores = compute_per_task_immediate(stage_reports, task_names)
    vals: List[float] = []

    for task in task_names:
        if task in oracle_new_task_scores and task in immediate_scores:
            # A larger gap means the continual learner is further from the
            # oracle that was trained in a more favorable setting.
            vals.append(oracle_new_task_scores[task] - immediate_scores[task])

    return safe_mean(vals)


def build_accuracy_matrix(
    stage_reports: List[StageReport],
    task_names: List[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Convert stage reports to a readable accuracy matrix.

    Unlike the earlier version, this fills missing entries with None instead of
    silently omitting them, which makes partial evaluation schedules clearer.
    """
    matrix: Dict[str, Dict[str, Optional[float]]] = {}

    for s in stage_reports:
        if s.trained_task_name is None:
            row_name = f"stage_{s.stage_idx}_init"
        else:
            row_name = f"stage_{s.stage_idx}_after_{s.trained_task_name}"

        matrix[row_name] = {}
        for task in task_names:
            # Missing evaluations are kept as `None` so downstream analysis can
            # tell the difference between "not measured" and a real score of 0.
            matrix[row_name][task] = s.task_scores.get(task, None)

    return matrix


# =============================================================================
# Main tester
# =============================================================================

@dataclass
class ContinualTesterConfig:
    """
    Global tester configuration.
    """
    run_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Whether to evaluate all tasks after every stage.
    # Strongly recommended if you want a full R matrix and meaningful FWT.
    evaluate_all_tasks_each_stage: bool = True

    # Whether to run initial stage-0 evaluation before any CL training.
    evaluate_initial_stage: bool = True

    # Whether to compute old-domain NLL/PPL whenever old_domain_eval_data exists.
    compute_old_domain_nll: bool = True

    # Whether to compute new-domain NLL/PPL whenever new_domain_eval_data exists.
    compute_new_domain_nll: bool = True

    # Optional external resource hook.
    resource_fn: Optional[ResourceFn] = None

    # Optional callbacks to count "units" for throughput.
    # Example unit choices:
    # - number of examples
    # - number of sequences
    # - number of tokens
    train_unit_count_fn: Optional[UnitCountFn] = None
    eval_unit_count_fn: Optional[UnitCountFn] = None
    unit_name: str = "units"

    # Optional user notes.
    notes: Dict[str, Any] = field(default_factory=dict)


class ContinualTester:
    """
    Architecture-agnostic continual-learning evaluator.

    This class does NOT assume anything about the architecture itself.
    It only assumes the provided model adapter can:
    - train on a task
    - score a dataset
    - optionally compute NLL
    - optionally expose hidden states
    """

    def __init__(self, config: ContinualTesterConfig):
        self.config = config

    def run(
        self,
        model: ModelAdapter,
        tasks: Sequence[TaskSpec],
        general_probe_suites: Optional[Sequence[ProbeSuiteSpec]] = None,
        memory_probe_suites: Optional[Sequence[ProbeSuiteSpec]] = None,
        representation_probe_suites: Optional[Sequence[RepresentationProbeSpec]] = None,
        future_task_baselines: Optional[Dict[str, float]] = None,
        oracle_new_task_scores: Optional[Dict[str, float]] = None,
    ) -> RunReport:
        """
        Run the full continual-learning evaluation.
        """
        # Normalize optional suite inputs so the rest of the method can treat
        # them uniformly.
        general_probe_suites = list(general_probe_suites or [])
        memory_probe_suites = list(memory_probe_suites or [])
        representation_probe_suites = list(representation_probe_suites or [])

        # `task_names` defines the canonical task order used everywhere else in
        # the report.
        task_names = [t.name for t in tasks]
        stage_reports: List[StageReport] = []

        # ---------------------------------------------------------------------
        # Stage 0: evaluate the initial model before any continual-learning step.
        # ---------------------------------------------------------------------
        if self.config.evaluate_initial_stage:
            logger.info("Evaluating initial stage before any task training.")
            ctx = StageContext(
                stage_idx=0,
                trained_task_name=None,
                seen_task_names=[],
                all_task_names=task_names,
                is_initial=True,
            )
            initial_stage = self._evaluate_stage(
                model=model,
                current_task=None,
                tasks=tasks,
                stage_idx=0,
                trained_task_name=None,
                general_probe_suites=general_probe_suites,
                memory_probe_suites=memory_probe_suites,
                representation_probe_suites=representation_probe_suites,
                ctx=ctx,
                train_log={},
                train_time_sec=None,
            )
            stage_reports.append(initial_stage)

        # ---------------------------------------------------------------------
        # Sequential CL loop: train on each task and evaluate afterward.
        # ---------------------------------------------------------------------
        for task_idx, task in enumerate(tasks, start=1):
            logger.info("Stage %d/%d: training on %s", task_idx, len(tasks), task.name)
            train_start = time.perf_counter()
            train_log = model.train_on_task(task)
            train_time_sec = time.perf_counter() - train_start

            # task_idx is 1-based here, so this includes the task we just trained.
            seen_task_names = task_names[:task_idx]
            ctx = StageContext(
                stage_idx=task_idx,
                trained_task_name=task.name,
                seen_task_names=seen_task_names,
                all_task_names=task_names,
                is_initial=False,
            )

            stage = self._evaluate_stage(
                model=model,
                current_task=task,
                tasks=tasks,
                stage_idx=task_idx,
                trained_task_name=task.name,
                general_probe_suites=general_probe_suites,
                memory_probe_suites=memory_probe_suites,
                representation_probe_suites=representation_probe_suites,
                ctx=ctx,
                train_log=train_log,
                train_time_sec=train_time_sec,
            )
            stage_reports.append(stage)
            logger.info("Stage %d/%d: finished evaluating %s", task_idx, len(tasks), task.name)

        # ---------------------------------------------------------------------
        # Build summary metrics from all stage reports.
        # ---------------------------------------------------------------------
        # All of the scalar CL metrics are derived from the stage reports, so
        # the raw stage history remains the source of truth.
        summary = self._build_summary(
            stage_reports=stage_reports,
            task_names=task_names,
            future_task_baselines=future_task_baselines,
            oracle_new_task_scores=oracle_new_task_scores,
        )

        accuracy_matrix = build_accuracy_matrix(stage_reports, task_names)

        # Run-level metadata records experiment context once so every stage does
        # not need to duplicate it.
        metadata = {
            "run_name": self.config.run_name,
            "task_count": len(tasks),
            "task_names": task_names,
            "unit_name": self.config.unit_name,
            "model_total_parameters": model.num_parameters(),
            "model_trainable_parameters": model.num_trainable_parameters(),
            **self.config.metadata,
        }

        return RunReport(
            metadata=metadata,
            task_order=task_names,
            stages=stage_reports,
            accuracy_matrix=accuracy_matrix,
            summary=summary,
            notes=self.config.notes,
        )

    # -------------------------------------------------------------------------
    # Internal stage evaluation
    # -------------------------------------------------------------------------

    def _evaluate_stage(
        self,
        model: ModelAdapter,
        current_task: Optional[TaskSpec],
        tasks: Sequence[TaskSpec],
        stage_idx: int,
        trained_task_name: Optional[str],
        general_probe_suites: Sequence[ProbeSuiteSpec],
        memory_probe_suites: Sequence[ProbeSuiteSpec],
        representation_probe_suites: Sequence[RepresentationProbeSpec],
        ctx: StageContext,
        train_log: Dict[str, Any],
        train_time_sec: Optional[float],
    ) -> StageReport:
        """
        Evaluate task scores, old-domain NLL, new-domain NLL, probe suites,
        and resources for one stage.
        """
        stage = StageReport(
            stage_idx=stage_idx,
            trained_task_name=trained_task_name,
            train_log=train_log,
        )

        # Measure evaluation time separately.
        eval_start = time.perf_counter()

        # ---------------------------------------------------------------------
        # Decide which tasks to evaluate.
        # ---------------------------------------------------------------------
        tasks_to_eval: List[TaskSpec]
        if self.config.evaluate_all_tasks_each_stage:
            tasks_to_eval = list(tasks)
        else:
            # If not evaluating all tasks, evaluate only tasks that should be "seen".
            if trained_task_name is None:
                tasks_to_eval = []
            else:
                # Include everything up to and including the currently trained task.
                trained_idx = next(i for i, t in enumerate(tasks) if t.name == trained_task_name)
                tasks_to_eval = list(tasks[: trained_idx + 1])

        # ---------------------------------------------------------------------
        # Task evaluations -> this fills the R matrix.
        # ---------------------------------------------------------------------
        for task in tasks_to_eval:
            # The tester does not know what a "score" means; the task's metric
            # function defines that contract.
            score = model.score(task.eval_data, task.score_fn)
            stage.task_scores[task.name] = score

        # ---------------------------------------------------------------------
        # Old-domain NLL/PPL, if available.
        # ---------------------------------------------------------------------
        if self.config.compute_old_domain_nll:
            seen_task_names = set(ctx.seen_task_names)
            for task in tasks:
                if task.name not in seen_task_names:
                    continue
                if task.old_domain_eval_data is None:
                    continue
                try:
                    nll_val = model.nll(task.old_domain_eval_data)
                    stage.old_domain_nll[task.name] = nll_val
                    stage.old_domain_ppl[task.name] = safe_exp(nll_val)
                except NotImplementedError:
                    # If a model cannot compute NLL, skip cleanly.
                    pass

        # ---------------------------------------------------------------------
        # New-domain NLL/PPL, if available.
        #
        # By default this only evaluates the CURRENT task's new-domain NLL, since
        # this is primarily a plasticity metric.
        # ---------------------------------------------------------------------
        if self.config.compute_new_domain_nll and current_task is not None:
            if current_task.new_domain_eval_data is not None:
                try:
                    nll_val = model.nll(current_task.new_domain_eval_data)
                    stage.new_domain_nll[current_task.name] = nll_val
                    stage.new_domain_ppl[current_task.name] = safe_exp(nll_val)
                except NotImplementedError:
                    pass

        # ---------------------------------------------------------------------
        # General benchmark probes.
        # ---------------------------------------------------------------------
        for suite in general_probe_suites:
            if not suite.enabled:
                continue
            # Probe suites are intentionally opaque to the tester so callers can
            # attach arbitrary diagnostics without changing this core code.
            stage.general_probes[suite.name] = suite.run_fn(model, suite.data, ctx)

        # ---------------------------------------------------------------------
        # Memory-specific probes (entity/fact/local pattern retention).
        # ---------------------------------------------------------------------
        for suite in memory_probe_suites:
            if not suite.enabled:
                continue
            stage.memory_probes[suite.name] = suite.run_fn(model, suite.data, ctx)

        # ---------------------------------------------------------------------
        # Representation probes.
        # ---------------------------------------------------------------------
        for suite in representation_probe_suites:
            if not suite.enabled:
                continue
            stage.representation_probes[suite.name] = suite.run_fn(model, suite.data, ctx)

        eval_time_sec = time.perf_counter() - eval_start

        # ---------------------------------------------------------------------
        # Resource snapshot.
        # ---------------------------------------------------------------------
        resources = self._build_resource_snapshot(
            model=model,
            current_task=current_task,
            evaluated_tasks=tasks_to_eval,
            train_time_sec=train_time_sec,
            eval_time_sec=eval_time_sec,
        )

        stage.resources = resources
        return stage

    def _build_resource_snapshot(
        self,
        model: ModelAdapter,
        current_task: Optional[TaskSpec],
        evaluated_tasks: Sequence[TaskSpec],
        train_time_sec: Optional[float],
        eval_time_sec: Optional[float],
    ) -> ResourceSnapshot:
        """
        Build a ResourceSnapshot, including explicit throughput fields and
        optional peak memory fields from the resource hook.
        """
        resources = ResourceSnapshot(
            train_time_sec=train_time_sec,
            eval_time_sec=eval_time_sec,
            total_parameters=model.num_parameters(),
            trainable_parameters=model.num_trainable_parameters(),
            active_parameters=model.active_parameter_estimate(),
            flops_estimate=model.flops_estimate(),
        )

        # ---------------------------------------------------------------------
        # Built-in throughput using unit-count callbacks, if provided.
        # ---------------------------------------------------------------------
        if current_task is not None and self.config.train_unit_count_fn is not None:
            try:
                train_units = self.config.train_unit_count_fn(current_task.train_data)
                resources.train_units = train_units
                if train_units is not None and train_time_sec and train_time_sec > 0:
                    # Throughput is derived rather than measured directly.
                    resources.train_units_per_sec = train_units / train_time_sec
            except Exception as exc:
                # The tester is designed to degrade gracefully if optional hooks
                # fail; the rest of the report is still useful.
                resources.extra["train_unit_count_error"] = 1.0
                print(f"[warn] train_unit_count_fn failed: {exc}")

        if evaluated_tasks and self.config.eval_unit_count_fn is not None:
            try:
                eval_units = 0
                valid = False
                for task in evaluated_tasks:
                    c = self.config.eval_unit_count_fn(task.eval_data)
                    if c is not None:
                        eval_units += c
                        valid = True
                if valid:
                    resources.eval_units = eval_units
                    if eval_time_sec and eval_time_sec > 0:
                        resources.eval_units_per_sec = eval_units / eval_time_sec
            except Exception as exc:
                resources.extra["eval_unit_count_error"] = 1.0
                print(f"[warn] eval_unit_count_fn failed: {exc}")

        # ---------------------------------------------------------------------
        # External resource hook.
        #
        # Known keys are promoted to first-class fields.
        # Unknown keys go into `extra`.
        # ---------------------------------------------------------------------
        if self.config.resource_fn is not None:
            try:
                hook_metrics = dict(self.config.resource_fn(model))
                known_keys = {
                    "peak_gpu_memory_bytes",
                    "peak_cpu_memory_bytes",
                    "train_units",
                    "eval_units",
                    "train_units_per_sec",
                    "eval_units_per_sec",
                    "flops_estimate",
                    "active_parameters",
                }

                if "peak_gpu_memory_bytes" in hook_metrics:
                    resources.peak_gpu_memory_bytes = int(hook_metrics.pop("peak_gpu_memory_bytes"))

                if "peak_cpu_memory_bytes" in hook_metrics:
                    resources.peak_cpu_memory_bytes = int(hook_metrics.pop("peak_cpu_memory_bytes"))

                if "train_units" in hook_metrics:
                    resources.train_units = int(hook_metrics.pop("train_units"))

                if "eval_units" in hook_metrics:
                    resources.eval_units = int(hook_metrics.pop("eval_units"))

                if "train_units_per_sec" in hook_metrics:
                    resources.train_units_per_sec = float(hook_metrics.pop("train_units_per_sec"))

                if "eval_units_per_sec" in hook_metrics:
                    resources.eval_units_per_sec = float(hook_metrics.pop("eval_units_per_sec"))

                if "flops_estimate" in hook_metrics:
                    resources.flops_estimate = float(hook_metrics.pop("flops_estimate"))

                if "active_parameters" in hook_metrics:
                    resources.active_parameters = int(hook_metrics.pop("active_parameters"))

                # Any remaining hook metrics go into `extra`.
                resources.extra.update(hook_metrics)

            except Exception as exc:
                resources.extra["resource_hook_error"] = 1.0
                print(f"[warn] resource_fn failed: {exc}")

        return resources

    # -------------------------------------------------------------------------
    # Summary builder
    # -------------------------------------------------------------------------

    def _build_summary(
        self,
        stage_reports: List[StageReport],
        task_names: List[str],
        future_task_baselines: Optional[Dict[str, float]],
        oracle_new_task_scores: Optional[Dict[str, float]],
    ) -> SummaryMetrics:
        """
        Build final summary metrics from stage reports.
        """
        # These helper outputs are also stored directly in the summary so later
        # analysis can inspect the exact per-task values behind the aggregates.
        immediate_scores = compute_per_task_immediate(stage_reports, task_names)
        per_task_final = compute_per_task_final(stage_reports, task_names)
        per_task_best = compute_per_task_best_after_learning(stage_reports, task_names)
        per_task_forgetting = compute_per_task_forgetting(
            stage_reports=stage_reports,
            task_names=task_names,
            use_immediate_reference=False,  # default: best-after-learning to final
        )

        summary = SummaryMetrics(
            final_acc=compute_final_acc(stage_reports, task_names),
            mean_forgetting=safe_mean(list(per_task_forgetting.values())),
            worst_forgetting=max(per_task_forgetting.values()) if per_task_forgetting else None,
            bwt=compute_bwt(stage_reports, task_names),
            fwt=compute_fwt(stage_reports, task_names, future_task_baselines=future_task_baselines),
            intransigence=compute_intransigence(
                stage_reports, task_names, oracle_new_task_scores=oracle_new_task_scores
            ),
            per_task_final=per_task_final,
            per_task_best=per_task_best,
            per_task_forgetting=per_task_forgetting,
            immediate_new_task_scores=immediate_scores,
        )
        return summary


# =============================================================================
# Aggregation helpers across seeds / task orders
# =============================================================================

@dataclass
class AggregatedScalar:
    """Mean/std/min/max summary for one scalar reported across multiple runs."""
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]
    n: int


@dataclass
class AggregatedReport:
    """
    Aggregated summary across multiple RunReport objects.
    """
    run_count: int
    aggregated_summary: Dict[str, AggregatedScalar]
    per_task_forgetting: Dict[str, AggregatedScalar]
    per_task_final: Dict[str, AggregatedScalar]


def _aggregate_values(values: List[float]) -> AggregatedScalar:
    """Aggregate one metric across runs, handling empty/singleton cases cleanly."""
    if not values:
        return AggregatedScalar(None, None, None, None, 0)
    if len(values) == 1:
        v = values[0]
        return AggregatedScalar(v, 0.0, v, v, 1)
    return AggregatedScalar(
        mean=statistics.mean(values),
        std=statistics.stdev(values),
        min=min(values),
        max=max(values),
        n=len(values),
    )


def aggregate_reports(reports: Sequence[RunReport]) -> AggregatedReport:
    """
    Aggregate multiple run reports (e.g., across seeds or task orders).
    """
    if not reports:
        raise ValueError("No reports provided for aggregation.")

    # Only top-level scalar summaries are aggregated here; the raw stage history
    # remains available in each individual RunReport.
    scalar_keys = ["final_acc", "mean_forgetting", "worst_forgetting", "bwt", "fwt", "intransigence"]

    aggregated_summary: Dict[str, AggregatedScalar] = {}
    for key in scalar_keys:
        vals: List[float] = []
        for r in reports:
            v = getattr(r.summary, key)
            if v is not None:
                vals.append(v)
        aggregated_summary[key] = _aggregate_values(vals)

    all_task_names = reports[0].task_order
    per_task_forgetting: Dict[str, AggregatedScalar] = {}
    per_task_final: Dict[str, AggregatedScalar] = {}

    for task in all_task_names:
        forgetting_vals: List[float] = []
        final_vals: List[float] = []
        for r in reports:
            if task in r.summary.per_task_forgetting:
                forgetting_vals.append(r.summary.per_task_forgetting[task])
            if task in r.summary.per_task_final:
                final_vals.append(r.summary.per_task_final[task])

        per_task_forgetting[task] = _aggregate_values(forgetting_vals)
        per_task_final[task] = _aggregate_values(final_vals)

    return AggregatedReport(
        run_count=len(reports),
        aggregated_summary=aggregated_summary,
        per_task_forgetting=per_task_forgetting,
        per_task_final=per_task_final,
    )


# =============================================================================
# Example adapter / example helper functions
# =============================================================================

class DummyModelAdapter:
    """
    Tiny example adapter to show the required interface.

    This is NOT a real model. Replace this with your architecture wrapper.
    """

    def __init__(self):
        self.learned_tasks: List[str] = []

    def clone(self) -> "DummyModelAdapter":
        return copy.deepcopy(self)

    def train_on_task(self, task: TaskSpec) -> Dict[str, Any]:
        # The dummy model simply records that the task was seen so the example
        # metrics can improve over time.
        self.learned_tasks.append(task.name)
        return {
            "trained_task": task.name,
            "fake_loss": 0.1,
            "num_updates": 100,
        }

    def score(self, data: Any, score_fn: ScoreFn) -> float:
        return score_fn(self, data)

    def nll(self, data: Any) -> float:
        # Fake NLL. Replace with real loss computation.
        return 1.0 / max(len(self.learned_tasks), 1)

    def num_parameters(self) -> int:
        return 1_000_000

    def num_trainable_parameters(self) -> int:
        return 1_000_000

    def active_parameter_estimate(self) -> Optional[int]:
        return None

    def flops_estimate(self) -> Optional[float]:
        return None

    def get_hidden_states(self, data: Any, layers: Sequence[str]) -> Dict[str, Any]:
        return {layer: [[0.0, 1.0], [1.0, 0.0]] for layer in layers}


def example_score_fn(model: ModelAdapter, data: Any) -> float:
    """
    Very small example metric.
    A real score_fn would run predictions and compute accuracy/F1/etc.
    """
    if isinstance(model, DummyModelAdapter):
        return min(1.0, 0.2 * len(model.learned_tasks))
    return 0.0


def example_general_probe_fn(model: ModelAdapter, data: Any, ctx: StageContext) -> Dict[str, float]:
    return {
        "score": 0.5 + 0.01 * ctx.stage_idx,
    }


def example_memory_probe_fn(model: ModelAdapter, data: Any, ctx: StageContext) -> Dict[str, float]:
    return {
        "entity_accuracy": 0.4 + 0.02 * ctx.stage_idx,
        "fact_accuracy": 0.35 + 0.015 * ctx.stage_idx,
    }


def example_representation_probe_fn(model: ModelAdapter, data: Any, ctx: StageContext) -> Dict[str, float]:
    return {
        "probe_score_layer_1": 0.6 + 0.01 * ctx.stage_idx,
        "probe_score_layer_2": 0.55 + 0.008 * ctx.stage_idx,
    }


def example_unit_count_fn(data: Any) -> Optional[int]:
    """
    Example throughput counter.
    Replace this with token counting / example counting / sequence counting.
    """
    return 100


def example_resource_fn(model: ModelAdapter) -> Dict[str, Union[int, float]]:
    """
    Example external resource hook.

    Any unknown keys remain in ResourceSnapshot.extra.
    """
    return {
        "peak_gpu_memory_bytes": 123_456_789,
        "peak_cpu_memory_bytes": 987_654_321,
        "some_custom_metric": 42.0,
    }


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # This block doubles as executable documentation: it shows the smallest
    # end-to-end setup needed to drive the tester with any compliant adapter.
    tasks = [
        TaskSpec(
            name="task_a",
            train_data={"train": "A"},
            eval_data={"eval": "A"},
            score_fn=example_score_fn,
            old_domain_eval_data={"old_nll_eval": "A"},
            new_domain_eval_data={"new_nll_eval": "A"},
        ),
        TaskSpec(
            name="task_b",
            train_data={"train": "B"},
            eval_data={"eval": "B"},
            score_fn=example_score_fn,
            old_domain_eval_data={"old_nll_eval": "B"},
            new_domain_eval_data={"new_nll_eval": "B"},
        ),
        TaskSpec(
            name="task_c",
            train_data={"train": "C"},
            eval_data={"eval": "C"},
            score_fn=example_score_fn,
            old_domain_eval_data={"old_nll_eval": "C"},
            new_domain_eval_data={"new_nll_eval": "C"},
        ),
    ]

    general_suites = [
        ProbeSuiteSpec(
            name="general_ability_suite",
            data={"benchmark": "general"},
            run_fn=example_general_probe_fn,
        )
    ]

    memory_suites = [
        ProbeSuiteSpec(
            name="memory_retention_suite",
            data={"benchmark": "memory"},
            run_fn=example_memory_probe_fn,
        )
    ]

    representation_suites = [
        RepresentationProbeSpec(
            name="representation_suite",
            layers=["layer_1", "layer_2"],
            data={"probe_data": "repr"},
            run_fn=example_representation_probe_fn,
        )
    ]

    # Explicit FWT baselines are now required for FWT.
    future_task_baselines = {
        "task_a": 0.0,
        "task_b": 0.0,
        "task_c": 0.0,
    }

    oracle_new_task_scores = {
        "task_a": 0.9,
        "task_b": 0.9,
        "task_c": 0.9,
    }

    config = ContinualTesterConfig(
        run_name="demo_run",
        metadata={
            "architecture": "dummy_model",
            "engram_enabled": False,
            "seed": 123,
        },
        train_unit_count_fn=example_unit_count_fn,
        eval_unit_count_fn=example_unit_count_fn,
        unit_name="examples",
        resource_fn=example_resource_fn,
    )

    tester = ContinualTester(config)
    model = DummyModelAdapter()

    report = tester.run(
        model=model,
        tasks=tasks,
        general_probe_suites=general_suites,
        memory_probe_suites=memory_suites,
        representation_probe_suites=representation_suites,
        future_task_baselines=future_task_baselines,
        oracle_new_task_scores=oracle_new_task_scores,
    )

    print(json.dumps(report.to_dict(), indent=2))
