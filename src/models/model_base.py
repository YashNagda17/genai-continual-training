"""Native Torch model execution obligations; architecture and weights are supplied."""

from abc import ABC, abstractmethod

from ..utils.numerics import move_to_device, resolve_device


class ModelBase(ABC):
    def __init__(self, device="auto"):
        """Resolve Torch placement without loading model weights or allocating model tensors."""
        self.device = resolve_device(device)
        self.runtime_root = None

    def bind_runtime(self, output, resources):
        """Give concrete executors a run-owned location for bounded numerical offload state."""
        self.runtime_root = output

    def bind_dataset(self, dataset):
        """Inject collation callbacks while keeping dataset implementations outside model code."""
        self.collate = dataset.collate
        self.collate_prompts = getattr(dataset, "collate_prompts", None)

    @property
    def base_storage_bytes(self):
        """Immutable external model assets charged alongside run-owned artifacts."""
        return 0

    def to_device(self, payload):
        """Move nested batch tensors to this model while preserving dtypes and autograd state."""
        return move_to_device(payload, self.device)

    @property
    @abstractmethod
    def info(self):
        """ModelInfo available before allocating tensors."""

    @abstractmethod
    def describe_role(self, role, initialization):
        """Return ModelInfo before allocating an auxiliary/probe/teacher role."""

    @abstractmethod
    def fork(self, role, initialization):
        """Create an unloaded, isolated wrapper at a reproducible named initialization.
        Keep its parameters, optimizer and RNG separate from the learner.
        Fixed probes and new CSReL rounds must not inherit evolving learner weights.
        """

    @abstractmethod
    def activate(self):
        """Load/restore the active numerical state within the advertised byte bounds."""

    @abstractmethod
    def offload(self):
        """Release residency, preserving complete numerical state on bounded storage."""

    @abstractmethod
    def release(self):
        """Release this auxiliary role's tensors and transient offload assets."""

    @abstractmethod
    def parameter_counts(self):
        """Return (unique total, trainable) counts after loading, including frozen base."""

    @abstractmethod
    def get_weights(self):
        """Return an independent copy of parameters and persistent model buffers.
        Mutating the returned snapshot must not mutate the live model.
        """

    @abstractmethod
    def set_weights(self, weights):
        """Install weights into the same architecture without replacing this wrapper.
        Reset optimizer, scheduler, scaler and step state so they match the new starting weights.
        All pipelines sharing this object must observe the installed weights.
        """

    @abstractmethod
    def train(self, k, *, batch_factory, objective, hooks, limits):
        """Execute k logical passes; return StepReports in attempt order.

        No independent sampling/shuffle is permitted. The executor alone steps its
        optimizer and scheduler. Call hooks.plan_update(supervised_statistics)
        before backward, enforce the returned prefix through weight decay/momentum,
        and report supervised Fisher/subset statistics. A skipped update consumes
        inputs and updates the scaler, but not scheduler or optimizer-step count.
        Save boundaries must have no outstanding graph or accumulated gradient.
        """

    @abstractmethod
    def score(self, batch, score_spec):
        """Return ScoreResult without changing parameters, buffers or training RNG.

        Supported kinds are loss, last_layer_gradients, lora_b_gradient (one
        batch-mean vector), and logits. Each requested representation/reduction
        must match exactly. Teacher logits use score_spec.temperature.
        """

    @abstractmethod
    def evaluate(self, view, evaluation_spec, progress=None):
        """Return finite named metrics; preserve all numerical/training/RNG state.
        When progress is set, call progress(done, total) after each scored batch.
        """

    @abstractmethod
    def predict(self, inputs, generation_spec):
        """Generate with native Torch on self.device, using explicit decoding settings."""

    @abstractmethod
    def save(self, path, state_kind):
        """Stream a self-contained state directory.

        resume includes optimizer, scheduler, scaler, backend RNG and counters.
        frozen contains exact immutable weights and compatibility metadata only.
        Files must fit info.checkpoint_bytes; shared immutable assets may be
        content-addressed references only when independently verified on load.
        """

    @abstractmethod
    def load(self, path, state_kind):
        """Restore complete numerical state for resume, including optimizer and backend RNG.
        For frozen snapshots, install the saved weights with fresh optimizer/scheduler/scaler/steps.
        Preserve wrapper identity and verify metadata/assets before accepting the saved state.
        """
