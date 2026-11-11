"""Common lifecycle and guarded access for the summarization pipelines."""

from abc import ABC, abstractmethod
import copy
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ..utils.contracts import fingerprint
from ..utils.runtime import PipelineRuntime
from ..utils.state import read_json, tuple_tree, write_json
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
        """Scan arrivals, then print when this epoch's cursor crosses another 5%.

        Later epochs reuse the admitted rows, so their cursor stays at the end.
        The repeat cursor is the one that moves, and the last batch of a repeat
        epoch resets it while advancing the pass count.
        """
        progress = self.progress if isinstance(self.progress, dict) else {}
        before = (progress.get("cursor", 0), progress.get("repeat_cursor", 0), progress.get("pass", 1))
        batch = super().current_batch(selector, peek=peek)
        if not batch or not self.current or not isinstance(self.progress, dict):
            return batch
        total = len(self.current)
        cursor = self.progress.get("cursor", 0)
        repeat = self.progress.get("repeat_cursor", 0)
        epoch_pass = self.progress.get("pass", 1)
        if epoch_pass != before[2]:
            epoch, done = epoch_pass, total
        elif cursor != before[0]:
            epoch, done = 1, cursor
        elif repeat != before[1]:
            epoch, done = epoch_pass + 1, repeat
        else:
            return batch
        passes = getattr(getattr(self, "spec", None), "passes", epoch)
        self.log_sample_progress(f"train task={self.task_id} epoch {epoch}/{passes}", done, total)
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
        """Return records, origins, objective and hooks, or None."""

    @abstractmethod
    def finish_task(self):
        """Propose the complete next replay membership."""


class SummarizationPipeline(PipelineBase):
    """Shared checkpoint/cost plumbing for the three independent summarization jobs."""

    def input_identity(self):
        return {}

    def run(self, context):
        started = time.perf_counter()
        path = Path(context.output) / "run.json"
        started_complete = path.is_file() and read_json(path).get("status") == "completed"
        try:
            return super().run(context)
        finally:
            if path.is_file() and hasattr(self, "artifacts") and not started_complete:
                with self.artifacts.lock():
                    manifest = read_json(path)
                    if manifest["identity"] == self.artifacts.identity:
                        elapsed = time.perf_counter() - started
                        costs = [event.get("measured", {}) for event in getattr(self, "events", ())]
                        peaks = [cost.get("peak_device_allocated_bytes") or cost.get("process_peak_cpu_bytes")
                                 for cost in costs]
                        measured = [value for value in peaks if value is not None]
                        manifest.update(wall_clock_seconds=manifest.get("wall_clock_seconds", 0) + elapsed,
                                        gpu_hours=manifest.get("gpu_hours", 0) + (elapsed / 3600 if self.device.type != "cpu" else 0),
                                        peak_memory_mb=max(measured) / 1024 ** 2 if measured else None,
                                        model_params=self.model.info.total_parameters, inference_latency_ms=None,
                                        result_type="null", result_reason="No scientific comparison assessed")
                        write_json(path, manifest)

    def bind(self, context):
        super().bind(context)
        self.resolved.update(inputs=self.input_identity(), preprocessing=self.dataset.fingerprint,
                             source=self.dataset.manifest.get('source', {}), torch=str(torch.__version__))
        self.resolved.update(implementation='summarization-pipelines-v1',
                             model_settings=asdict(self.model.settings) if hasattr(self.model, 'settings') else None,
                             model_revision=getattr(self.model, 'revision', None),
                             model_config=self.model.config.to_dict() if getattr(self.model, 'config', None) is not None else None)
        if getattr(self.model, 'gradient_checkpointing', False):
            self.resolved['gradient_checkpointing'] = True
        # Once per bind; numerical forwards never hash tokens or parameters.
        self.artifacts.identity = fingerprint(self.resolved)

    def _commit(self):
        if self.context is not None:
            self.boundary()

    def _cost(self, category, records, measured, generated=()):
        details = {"category": category, "rows": len(records),
                   "input_tokens": sum(len(r.inputs.get("ids", ())) for r in records),
                   "new_tokens": sum(map(len, generated)),
                   "forward_pass_units": len(records) * (3 if category == 'probe_pass' else 1),
                   "measured": measured}
        if self.context is not None:
            self.record_event("analysis_cost", **details)
        return details

    def _score(self, records, spec, category, model=None):
        records = tuple(records)
        if any(r.split != "train" for r in records):
            raise ValueError("Policy scoring requires training records")
        batch = self.dataset.collate(records)
        result = (model or self.model).score(batch, spec).validate(batch, spec)
        values = result.values
        # tensors = [values] if torch.is_tensor(values) else values["gradients"]
        # if any(not torch.isfinite(value).all() for value in tensors):
        #     raise ValueError("Nonfinite policy scores")
        # Temporary: consume scores on their compute device without a host finite check.
        self._cost(category, records, result.cost)
        return values

    def _generate(self, records, category, model=None, *, held_out=False, guard=None):
        records = tuple(records)
        if any((r.split not in {"validation", "test"}) if held_out else r.split != "train" for r in records):
            raise ValueError("Generation view has the wrong data split")
        spec = dict(self.config.generation)
        if guard:
            spec["script_guard"] = guard
        batch = self.dataset.collate_sources(records)
        result = (model or self.model).predict(batch, spec)
        if tuple(result["ids"]) != batch.ids:
            raise ValueError("Generation changed row order")
        self._cost(category, records, result["cost"], result["token_ids"])
        return result

    def held_out(self, language, split):
        named = self.dataset.views.get(f"{language}/{split}")
        rows = tuple(named if named is not None else
                     (r for r in self.dataset.views.get(split, ()) if r.task_id == language))
        if not rows or any(r.split != split or r.task_id != language for r in rows):
            raise ValueError(f"Required {language} {split} view is missing or invalid")
        return rows

    def _open_view(self):
        return ()  # Analysis jobs never open a training archive.

    def open_task(self):
        pass

    def finish_task(self):
        return ()

    def _evaluate_seen(self):
        pass  # Each job owns its explicit validation/reporting phases.
