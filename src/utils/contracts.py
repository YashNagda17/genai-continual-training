"""Backend-independent boundary records. Payloads persisted in replay must be JSON."""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable
import hashlib
import json


def fingerprint(value):
    """Hash canonical JSON so equivalent dictionary orders produce the same identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Record:
    sample_id: str
    dataset_id: str
    revision: str
    source_id: str
    task_id: str
    arrival: int
    split: str
    inputs: Any
    target: Any
    tokens: int
    storage_bytes: int
    class_id: str | None = None
    group_id: str | None = None
    practice: dict | None = None


def practice_record(record, cut=None):
    """Use a retained past answer while preserving the observation and its source."""
    if record.practice is None:
        return record
    ids = record.practice["answer_ids"]
    ids = ids if cut is None else ids[:cut]
    updated = replace(record, target={"text": record.practice["answer_text"], "ids": list(ids)},
                      tokens=record.tokens - len(record.target["ids"]) + len(ids))
    return replace(updated, storage_bytes=len(json.dumps(asdict(updated), ensure_ascii=False).encode()))


def training_records(records, task_id=None):
    """Materialize a unique train-only view, optionally restricted to one task.
    Batch sampling may repeat IDs later; source membership itself must be unique.
    """
    records = tuple(records)
    return records


@dataclass(frozen=True)
class Batch:
    records: tuple[Record, ...]
    payload: Any
    valid_tokens: int
    supervised_tokens: int

    @property
    def ids(self):
        """Preserve record order when aligning scores, labels and update reports."""
        return tuple(r.sample_id for r in self.records)

@dataclass(frozen=True)
class ScoreSpec:
    kind: str
    reduction: str
    parameter_view: str = ""
    frozen_prefix: int = 0
    temperature: float = 1.0


@dataclass(frozen=True)
class ScoreResult:
    ids: tuple[str, ...]
    values: Any
    model_revision: str
    representation: str
    reduction: str
    cost: dict = field(default_factory=dict)
    cpu_bytes: int = 0
    accelerator_bytes: int = 0

    def validate(self, batch, spec):
        """Check score identity, order and representation against the requested batch.
        This verifies metadata and buffer accounting, not backend-specific tensor shapes.
        Return the same result so runtime callers can validate and consume it together.
        """
        if tuple(self.ids) != batch.ids or not self.model_revision:
            raise ValueError("Scores are not aligned or lack model provenance")
        if self.reduction != spec.reduction:
            raise ValueError("Score reduction mismatch")
        if spec.parameter_view and self.representation != spec.parameter_view:
            raise ValueError("Score parameter-view mismatch")
        if min(self.cpu_bytes, self.accelerator_bytes) < 0:
            raise ValueError("Invalid score-buffer byte count")
        return self


@dataclass(frozen=True)
class Objective:
    reduction: str
    current_weight: float = 1.0
    replay_weight: float = 1.0
    separate_means: bool = False
    kd_weight: float = 0.0
    kd_temperature: float = 1.0
    kd_temperature_squared: bool = False
    kl_direction: str = "teacher_to_student"
    token_mode: str = "all"
    token_allowance: float = 0.0


@dataclass(frozen=True)
class TokenTargets:
    """Flat float32 CPU baseline and weight, one value per supervised token."""
    baseline: Any
    weight: Any


@dataclass(frozen=True)
class AdapterPlan:
    """Which shared adapters can change, where extra rank goes, and the per-block learning rate.

    Block keys are ``encoder.7`` strings. ``new_rank`` and ``lr_scale`` map those keys to a
    rank and a multiplier in ``(0, 1]``. The experiment runner builds the plan.
    """
    trainable: tuple[str, ...]
    new_rank: dict = field(default_factory=dict)
    lr_scale: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingBatch:
    batch: Batch
    origins: tuple[str, ...]
    teacher_targets: ScoreResult | None = None
    teacher_task: str | None = None
    token_targets: TokenTargets | None = None


@dataclass(frozen=True)
class TrainLimits:
    max_attempts: int
    max_steps: int
    max_tokens: int


@dataclass(frozen=True)
class FreezePlan:
    prefix: int
    refresh: bool
    scores: tuple[float, ...] = ()


@dataclass
class UpdateHooks:
    plan_update: Callable[[dict], FreezePlan] | None = None
    gradient_subset: str = ""
    token_targets: Callable | None = None


@dataclass(frozen=True)
class StepReport:
    consumed_ids: tuple[str, ...]
    success: bool
    optimizer_step: int
    tokens: int
    loss_sums: dict
    loss_counts: dict
    diagnostics: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    skip_reason: str | None = None


@dataclass(frozen=True)
class SelectionResult:
    candidate_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    provenance: dict


@dataclass(frozen=True)
class ReplayPlan:
    tasks: tuple[str, ...]
    probabilities: tuple[float, ...]
    replay_probability: float
    teachers: dict[str, str]
    current_teacher: str | None


@dataclass(frozen=True)
class ModelInfo:
    identity: str
    total_parameters: int
    trainable_parameters: int
    capabilities: frozenset[str]
    cpu_bytes: int
    accelerator_bytes: int
    checkpoint_bytes: int
    vocabulary: str
    loss_fingerprint: str
    parameter_views: tuple[str, ...]
    layer_names: tuple[str, ...] = ()
    forward_cost: float = 0.0
    backward_costs: tuple[float, ...] = ()
    legal_prefixes: tuple[int, ...] = (0,)

    def validate(self):
        """Check role metadata before allocating a learner, probe or teacher.
        Layer costs must align with the layer order, and legal prefixes leave a trainable path.
        """
        if not 0 < self.total_parameters <= 700_000_000:
            raise ValueError("Model roles must have at most 700M total parameters")
        if not 0 <= self.trainable_parameters <= self.total_parameters:
            raise ValueError("Invalid trainable parameter count")
        if min(self.cpu_bytes, self.accelerator_bytes, self.checkpoint_bytes) < 0:
            raise ValueError("Model memory estimates must be nonnegative")
        if not self.identity or not self.vocabulary or not self.loss_fingerprint:
            raise ValueError("Model identity, vocabulary and loss fingerprint are required")
        if len(self.layer_names) != len(self.backward_costs):
            raise ValueError("Layer cost layout mismatch")
        if self.forward_cost < 0 or any(v < 0 for v in self.backward_costs):
            raise ValueError("Layer costs must be nonnegative")
        if 0 not in self.legal_prefixes or any(
                p < 0 or p >= len(self.layer_names) for p in self.legal_prefixes if p):
            raise ValueError("Legal prefixes must include zero and leave a trainable path")


@dataclass(frozen=True)
class ResourceBudget:
    cpu_bytes: int
    accelerator_bytes: int
    disk_bytes: int = 10_000_000_000

    def validate_role(self, info):
        """Check one role against CPU/device bounds; aggregate payload checks belong to runtime."""
        info.validate()
        if info.cpu_bytes > self.cpu_bytes or info.accelerator_bytes > self.accelerator_bytes:
            raise ValueError("Model role exceeds declared residency budget")


@dataclass(frozen=True)
class ReplayBudget:
    examples: int | None
    tokens: int
    bytes: int


@dataclass(frozen=True)
class RunSpec:
    task_ids: tuple[str, ...]
    seed: int
    batch_size: int
    max_attempts: int
    max_steps: int
    max_tokens: int
    passes: int
    exhaustion: str
    reduction: str
    replay: ReplayBudget
    resources: ResourceBudget
    checkpoint_every: int
    identity: dict
    evaluation_spec: dict
    resume: str = "auto"
    accumulation: int = 1
    access: str = "task_chunks"

    def scientific(self):
        """Build the identity used to decide whether an attempt can be resumed.
        Resume mode and checkpoint cadence are operational choices.
        Keep data, training and scientific budget settings in the returned identity.
        """
        spec = asdict(self)
        for key in ("resume", "checkpoint_every"):
            spec.pop(key)
        spec["resources"].pop("disk_bytes")
        return spec


def model_boundary(condition, message):
    """Validate a concrete model's shape, identity or allocation request at its boundary."""
    if not condition:
        raise ValueError(message)


@dataclass
class RunContext:
    model: Any
    dataset: Any
    spec: RunSpec
    output: Any


@dataclass(frozen=True)
class RunResult:
    status: str
    checkpoint: str
    completed_tasks: tuple[str, ...]
    metrics: tuple[dict, ...]
    counters: dict
