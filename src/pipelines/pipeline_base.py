"""Composition interface shared by standalone paper methods and hybrids."""

from abc import ABC, abstractmethod
import copy
import random

from ..utils.runtime import PipelineRuntime
from ..utils.state import save_weights, tuple_tree
from ..utils.numerics import resolve_device


class PipelineBase(PipelineRuntime, ABC):
    def __init__(self, config, model=None, dataset=None):
        """Store injected adapters, policy RNG and the optional active auxiliary reference.
        Run-specific cursors, replay and checkpoint storage are initialized by the runtime.
        """
        self.config = config
        self.policy = {}
        self.rng = random.Random()
        self.model, self.dataset = model, dataset
        self.context = None
        self.auxiliary = None
        self.auxiliary_key = None
        self._checkpoint = ""

    def components(self):
        """Name the standalone component; hybrids override this mapping to expose child policies."""
        return {"method": self}

    @property
    def device(self):
        """Follow the shared learner's Torch device; unbound numerical steps use CPU."""
        return self.model.device if self.model is not None else resolve_device("cpu")

    def bind_instruction_dataset(self, dataset, raw, split_spec=None):
        """Preprocess an instruction source with the bound local model and share that dataset.

        Official splits remain unless split_spec sets validation_fraction. Truncation comes from
        the source manifest unless split_spec overrides it. The model must implement attach_instruction.
        """
        from ..datasets.dataset_instruction import InstructionPreprocessing
        spec = {"validation_fraction": 0.0, "seed": 0}
        if split_spec:
            spec.update(split_spec)
        truncation = spec.pop("truncation", dataset.manifest.get("source", {}).get("truncation", "drop"))
        views = dataset.split(raw, spec)
        prepared, report = dataset.preprocess(
            views, InstructionPreprocessing(**self.model.preprocessing_kwargs(truncation=truncation)))
        self.dataset = dataset
        for component in self.components().values():
            component.dataset = dataset
            if component.context is not None:
                component.context.dataset = dataset
        self.model.attach_instruction(dataset)
        return prepared, report

    def set_model(self, model):
        """Share the exact learner wrapper across this owner and all registered components.
        Update any bound context as well, so later lifecycle calls use the replacement.
        No weights or optimizer state are copied; callers switch between completed operations.
        """
        if self.context is not None and self.model is not None and self.model is not model:
            self.model.offload()
        for component in self.components().values():
            component.model = model
            if component.context is not None:
                component.context.model = model
        if self.context is not None:
            model.bind_runtime(self.context.output, self.context.spec.resources)
            model.bind_dataset(self.context.dataset)
        return self

    def get_weights(self):
        """Request an independent backend-native parameter/buffer snapshot from the adapter."""
        return self.model.get_weights()

    def set_weights(self, weights):
        """Install weights in the existing wrapper; the adapter resets numerical training state.
        Every pipeline sharing that wrapper sees the update immediately.
        """
        self.model.set_weights(weights)
        return self

    def save_weights(self, path):
        """Save a frozen-weight snapshot independently of replay and lifecycle checkpoints."""
        return save_weights(self.model, path)

    def load_weights(self, path):
        """Load frozen weights into the current wrapper with fresh optimizer/scheduler/scaler state."""
        self.model.load(path, "frozen")
        return self

    def get_policy_state(self):
        """Copy every registered component's policy and RNG without copying model tensors.
        Deep copies prevent later policy mutations from changing the saved snapshot.
        Lifecycle cursors and replay belong to full checkpoints, not this composition helper.
        """
        return {name: {"policy": copy.deepcopy(component.policy), "rng": component.rng.getstate()}
                for name, component in self.components().items()}

    def set_policy_state(self, state):
        """Restore matching component policies and RNG streams from an independent snapshot.
        Convert JSON-restored RNG lists back to tuples before passing them to Random.setstate.
        Leave the shared model, replay membership and lifecycle progress unchanged.
        """
        for name, component in self.components().items():
            component.policy = copy.deepcopy(state[name]["policy"])
            component.rng.setstate(tuple_tree(state[name]["rng"]))
        return self

    def required_capabilities(self):
        """Let paper components declare the learner operations they require; the base requires none."""
        return set()

    def auxiliary_roles(self):
        """Let components declare isolated roles and initializations needed by the run."""
        return ()

    def dependencies(self):
        """Let components name frozen artifacts that a full checkpoint must retain."""
        return ()

    def teacher_targets(self, batch, task, objective):
        """Default to supervised-only training; routed methods override this hook."""
        return None

    def observe(self, report, records):
        """Methods with update-clock policies override this hook."""

    def after_memory_commit(self):
        """Methods with task-boundary policies override this hook."""

    def log_sample_progress(self, phase, done, total):
        """Print one line each time this sweep crosses another 5% of its samples."""
        if total <= 0 or done <= 0:
            return
        bucket = min(20, (done * 20) // total)
        if bucket < 1:
            return
        marks = getattr(self, "_sample_marks", None)
        if marks is None:
            self._sample_marks = marks = {}
        if marks.get(phase) == bucket:
            return
        marks[phase] = bucket
        print(f"{phase} {done}/{total} samples ({bucket * 5}%)", flush=True)

    def current_batch(self, selector=None, *, peek=False):
        """Scan arrivals, then print when the training cursor crosses another 5%."""
        before = self.progress.get("cursor", 0) if isinstance(self.progress, dict) else 0
        batch = super().current_batch(selector, peek=peek)
        done = self.progress.get("cursor", 0) if isinstance(self.progress, dict) else 0
        if done != before and self.current:
            self.log_sample_progress(f"train task={self.task_id}", done, len(self.current))
        return batch

    def evaluate(self, view, evaluation_spec):
        """Score held-out records and print each new 5% of that view."""
        if any(r.split == "train" for r in view):
            raise ValueError("Evaluation requires held-out records")
        self._activate(self.model)
        task = view[0].task_id if view else ""

        def progress(done, total, _task=task):
            self.log_sample_progress(f"evaluate task={_task}", done, total)

        return self.model.evaluate(view, evaluation_spec, progress=progress)

    @abstractmethod
    def open_task(self):
        """Prepare current-task policy state."""

    @abstractmethod
    def next_update(self):
        """Return records, origins, objective, teacher task and hooks, or None."""

    @abstractmethod
    def finish_task(self):
        """Propose the complete next replay membership."""
