from dataclasses import asdict, replace
from pathlib import Path
import torch
import copy
import csv
import random
import time


from .state import Artifacts, ReplayStore, write_json
from .contracts import RunResult, TrainLimits, TrainingBatch, UpdateHooks


class PipelineRuntime:
    def validate_runtime(self):
        """Preflight learner capabilities and declared residency bounds before allocating roles.
        Paper-specific configuration validity is assumed during the current logic-review stage.
        """
        info, spec = self.model.info, self.spec
        spec.resources.validate_role(info)
        if not self.required_capabilities() <= info.capabilities:
            raise ValueError(f"Missing model capabilities: {self.required_capabilities() - info.capabilities}")
        for role, initialization in self.auxiliary_roles():
            spec.resources.validate_role(self.model.describe_role(role, initialization))

    def bind(self, context):
        """Bind one model, dataset and run specification to the owner and its components.
        Capture initial policies once so a fresh attempt can reset interrupted policy work.
        """
        if not hasattr(self, "_initial_policies"):
            self._initial_policies = {k: copy.deepcopy(v.policy) for k, v in self.components().items()}
        self.context = context
        self.set_model(context.model)
        self.dataset, self.spec = context.dataset, context.spec
        for component in self.components().values():
            component.dataset, component.spec = self.dataset, self.spec
        self.validate_runtime()
        self.resolved = {
            "schema": 1, "method": type(self).__name__, "spec": self.spec.scientific(),
            "config": {k: asdict(v.config) for k, v in self.components().items()},
            "model": self.model.info.identity, "dataset": type(self.dataset).__name__,
            "device": str(self.model.device),
        }
        self.artifacts = Artifacts(context.output, type(self).__name__, self.spec.resources.disk_bytes,
                                     external_bytes=lambda: self.model.base_storage_bytes)

    def _activate(self, model):
        """Validate metadata, activate the wrapper and verify its loaded parameter counts.
        When a run is bound, also enforce that run's per-role residency bounds.
        """
        model.info.validate()
        if self.context is not None:
            self.spec.resources.validate_role(model.info)
        model.activate()
        if tuple(model.parameter_counts()) != ( model.info.total_parameters, model.info.trainable_parameters):
            raise ValueError("Loaded parameter counts disagree with pre-allocation metadata")

    def acquire_auxiliary(self, role, initialization, *, checkpoint=None, state_kind="frozen"):
        """Offload the learner before creating and activating one isolated numerical role.
        Repeated requests for the same role reuse it while fitting or scoring is in progress.
        The owner must release that role before switching to another initialization.
        """
        key = [role, initialization]
        if self.auxiliary is None:
            self.model.offload()
            info = self.model.describe_role(role, initialization)
            self.spec.resources.validate_role(info)
            self.auxiliary = self.model.fork(role, initialization)
            if checkpoint is not None:
                self.auxiliary.load(checkpoint, state_kind)
            if self.auxiliary is self.model or (checkpoint is None and self.auxiliary.info != info):
                raise ValueError("Auxiliary must be isolated and match advertised metadata")
            self.auxiliary_key = key
            self._activate(self.auxiliary)
        elif self.auxiliary_key != key:
            raise ValueError("Release the previous auxiliary before switching roles")
        return self.auxiliary

    def release_auxiliary(self, *, restore_learner=True):
        """Release the auxiliary; analysis callers may defer restoring the idle learner."""
        if self.auxiliary is not None:
            self.auxiliary.release()
            self.auxiliary = None
            self.auxiliary_key = None
            if restore_learner:
                self._activate(self.model)

    def record_event(self, kind, **details):
        """Advance the audit watermark and attach task identity to one completed observation."""
        self.event += 1
        self.events.append({"event": self.event, "task": self.task_id, "kind": kind, **details})

    def _materialize_events(self):
        """Replace device scalars in unsaved events (step losses) with numbers, in one transfer per dtype.

        Executors leave step losses on the device so the host does not wait after every update."""
        start = getattr(self, "_materialized", 0)
        if start > len(self.events):
            start = 0
        found = []

        def collect(container, key, value):
            if isinstance(value, torch.Tensor):
                found.append((container, key, value))
            elif isinstance(value, (dict, list)):
                for inner_key, inner in (value.items() if isinstance(value, dict) else enumerate(value)):
                    collect(value, inner_key, inner)

        collect(None, None, self.events[start:])
        for integral in (False, True):
            group = [(c, k, t) for c, k, t in found if t.numel() == 1 and (t.dtype.is_floating_point is not integral)]
            if group:
                dtype = torch.int64 if integral else torch.float32
                values = torch.stack([t.detach().reshape(()).to(dtype) for _, _, t in group]).cpu().tolist()
                for (container, key, _), number in zip(group, values):
                    container[key] = number
        for container, key, tensor in found:
            if isinstance(container[key], torch.Tensor):
                container[key] = tensor.detach().cpu().tolist()
        self._materialized = len(self.events)

    @property
    def task_id(self):
        return self.spec.task_ids[self.task_index] if self.task_index < len(self.spec.task_ids) else None

    def _initialize(self):
        """Reset lifecycle state and replay for a new attempt using the captured initial policies.
        Seed each component by run seed and stable component name, preserving independent streams.
        Activate the supplied learner without replacing its initial weights.
        """
        self.auxiliary, self.auxiliary_key = None, None
        self.task_index, self.event = 0, 0
        self.phase, self.progress, self.metrics, self.events = "open", {}, [], []
        self.counters = {"attempts": 0, "steps": 0, "tokens": 0}
        self.replay = ReplayStore(budget=self.spec.replay)
        for name, component in self.components().items():
            component.policy = copy.deepcopy(self._initial_policies[name])
            component.rng.seed(f"{self.spec.seed}:{name}")
        self._activate(self.model)

    def run(self, context):
        self.bind(context)
        with self.artifacts.lock():
            manifest_path = Path(context.output) / "run.json"
            if manifest_path.exists():
                raise FileExistsError(f"Run output already exists: {context.output}; choose a fresh directory")
            self._initialize()
            manifest = {"identity": self.artifacts.identity, "status": "running", "resolved": self.resolved,
                        "spec": asdict(self.spec), "config": {k: asdict(c.config) for k, c in self.components().items()}}
            write_json(manifest_path, manifest)
            try:
                for self.task_index, _task in enumerate(self.spec.task_ids):
                    self.current = self._open_view()
                    self.progress = {"cursor": 0, "queue": [], "admitted": [], "repeat_cursor": 0, "pass": 1,
                                     "attempts": 0, "steps": 0, "tokens": 0}
                    self.open_task()
                    self.phase = "training"
                    self.record_event("task_open")
                    while True:
                        limited = self._limit_reached()
                        update = None if limited else self.next_update()
                        if update is None:
                            self.record_event("training_complete", reason=(
                                "limits" if limited else self.progress.get("stop_reason", "exhausted")),
                                attempts=self.progress["attempts"], steps=self.progress["steps"])
                            break
                        records, origins, objective, hooks = update
                        batch = self.collate(records)
                        if self.progress["tokens"] + batch.valid_tokens > self.spec.max_tokens:
                            self.record_event("token_limit")
                            break
                        report = self.execute(self.model, TrainingBatch(batch, tuple(origins)), objective, hooks)
                        for key, amount in (("attempts", 1), ("steps", int(report.success)), ("tokens", report.tokens)):
                            self.counters[key] += amount
                            self.progress[key] += amount
                        self.observe(report, records)
                        self.progress.pop("branch", None)
                        self.record_event("learner_update", ids=list(batch.ids), origins=list(origins),
                                          report=self.report_summary(report))
                    self.phase = "memory"
                    self.replay = self.replay.proposal(self.finish_task())
                    self.after_memory_commit()
                    self.current, self.progress = (), {}
                    self.record_event("memory_commit", ids=[r.sample_id for r in self.replay.records], storage=self.replay.usage)
                    self.phase = "evaluation"
                    self._evaluate_seen()
                    self.record_event("evaluation_complete")
                self.task_index = len(self.spec.task_ids)
                self.phase = "completed"
                self._write_metrics()
                manifest["status"] = "completed"
                write_json(manifest_path, manifest)
                return RunResult("completed", "", self.spec.task_ids, tuple(self.metrics), dict(self.counters))
            except BaseException as error:
                manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                error=f"{type(error).__name__}: {error}")
                write_json(manifest_path, manifest)
                raise
            finally:
                if self.auxiliary is not None:
                    self.auxiliary.release()
                    self.auxiliary = None
                self.auxiliary_key = None
                self.model.release()

    def _open_view(self):
        """Materialize the current training view without changing its order."""
        records = tuple(self.dataset.training_view(self.task_id))
        if not records:
            raise ValueError("Required training partition is empty")
        if set(r.sample_id for r in records) & set(
                r.sample_id for r in self.replay.records if r.task_id != self.task_id):
            raise ValueError("Current and historical observation identities overlap")
        if list(records) != sorted(records, key=lambda r: (r.arrival, r.sample_id)):
            raise ValueError("Training view must preserve canonical arrival order")
        payload_bytes = sum(r.storage_bytes for r in (*records, *self.replay.records))
        role_cpu = max([self.model.info.cpu_bytes, *(
            self.model.describe_role(role, initialization).cpu_bytes
            for role, initialization in self.auxiliary_roles())])
        if payload_bytes + role_cpu > self.spec.resources.cpu_bytes:
            raise ValueError("Current chunk, replay and resident role exceed the CPU byte budget")
        return records

    def _limit_reached(self):
        """Stop when any per-task attempt, successful-step or consumed-token bound is reached."""
        return any(self.progress[key] >= limit for key, limit in (
            ("attempts", self.spec.max_attempts), ("steps", self.spec.max_steps),
            ("tokens", self.spec.max_tokens)))

    def collate(self, records):
        """Delegate training collation while preserving the exact requested record order.
        This boundary prevents held-out records from entering learner or auxiliary scoring.
        """
        records = tuple(records)
        if any(r.split != "train" for r in records):
            raise ValueError("Held-out examples cannot enter training or scoring")
        batch = self.dataset.collate(records)
        if batch.ids != tuple(r.sample_id for r in records):
            raise ValueError("Collation changed example order")
        return batch

    def execute(self, model, training_batch, objective, hooks=None):
        """Ask the model to consume exactly one supplied update attempt with its objective and hooks.
        Validate consumed IDs, tokens and reporting without choosing data or stepping optimizers here.
        Record host wall time separately from any synchronized hardware costs supplied by the adapter.
        """
        start = time.monotonic()
        reports = tuple(model.train(
            1, batch_factory=lambda: iter((training_batch,)), objective=objective,
            hooks=hooks or UpdateHooks(),
            limits=TrainLimits(1, 1, training_batch.batch.valid_tokens)))
        if len(reports) != 1 or tuple(reports[0].consumed_ids) != training_batch.batch.ids:
            raise ValueError("Executor must report exactly the supplied update attempt")
        report = reports[0]
        if report.tokens != training_batch.batch.valid_tokens:
            raise ValueError("Executor token count disagrees with collated batch")
        if not report.success and not report.skip_reason:
            raise ValueError("Skipped update needs an explicit reason")
        if report.skip_reason != "no_active_tokens" and any(v <= 0 for v in report.loss_counts.values()):
            raise ValueError("Loss denominators must be positive")
        self.record_event("execution_cost", role="learner" if model is self.model else "auxiliary",
                          wall_seconds=time.monotonic() - start, measured=report.cost)
        return report

    @staticmethod
    def report_summary(report):
        """Copy report fields for persistence while dropping transient per-example gradient arrays."""
        # Exclude transient arrays before asdict can duplicate a large tensor buffer.
        diagnostics = {k: v for k, v in report.diagnostics.items() if k not in {"class_gradients", "token_surprise"}}
        return asdict(replace(report, diagnostics=diagnostics))

    def learner_pass_order(self, items, pass_number):
        """Reconstruct a pass permutation independently of replay/policy randomness."""
        items = list(items)
        if self.spec.identity.get("shuffle_training", False):
            random.Random(f"{self.spec.seed}:learner-order:{self.task_index}:{pass_number}").shuffle(items)
        return items

    def current_batch(self, selector=None, *, peek=False):
        """Queue admitted arrivals, repeating only admitted IDs; peek leaves the batch queued."""
        progress = self.progress
        by_id = {r.sample_id: r for r in self.current}
        ordered_current = self.learner_pass_order(self.current, 1)
        while not progress["queue"] and progress["cursor"] < len(self.current):
            start = progress["cursor"]
            records = ordered_current[start:start + self.spec.batch_size]
            selected = selector(records) if selector else tuple(r.sample_id for r in records)
            if not set(selected) <= set(r.sample_id for r in records):
                raise ValueError("Admission selected an ineligible record")
            progress["cursor"] += len(records)
            progress["queue"] = list(selected)
            progress["admitted"].extend(selected)
            self.record_event("arrival", observed=[r.sample_id for r in records], admitted=list(selected))
        if not progress["queue"]:
            if (self.spec.exhaustion == "repeat_admitted" and progress["admitted"]
                    and progress["pass"] < self.spec.passes):
                start = progress["repeat_cursor"]
                ordered_ids = self.learner_pass_order(progress["admitted"], progress["pass"] + 1)
                ids = ordered_ids[start:start + self.spec.batch_size]
                progress["queue"] = ids
                progress["repeat_cursor"] += len(ids)
                if progress["repeat_cursor"] == len(progress["admitted"]):
                    progress["pass"] += 1
                    progress["repeat_cursor"] = 0
            else:
                progress["stop_reason"] = "exhausted" if progress["admitted"] else "no_current_update"
                return ()
        result = tuple(by_id[i] for i in progress["queue"][:self.spec.batch_size])
        if not peek:
            del progress["queue"][:len(result)]
        return result

    def evaluate(self, view, evaluation_spec):
        """Evaluate held-out records with the learner; the adapter preserves numerical training state."""
        if any(r.split == "train" for r in view):
            raise ValueError("Evaluation requires held-out records")
        self._activate(self.model)
        return self.model.evaluate(view, evaluation_spec)

    def _evaluate_seen(self):
        """Evaluate each seen task once at the current task boundary."""
        for task in self.spec.task_ids[:self.task_index + 1]:
            if any(r["checkpoint_task"] == self.task_id and r["task"] == task for r in self.metrics):
                continue
            view = tuple(self.dataset.evaluation_view(task))
            metrics = self.evaluate(view, self.spec.evaluation_spec) if view else {}
            predictions = metrics.pop("_predictions", None)
            if predictions is not None:
                stage = self.spec.task_ids.index(task)
                prediction_dir = Path(self.context.output) / "predictions"
                prediction_dir.mkdir(parents=True, exist_ok=True)
                write_json(prediction_dir / f"stage-{self.task_index + 1}-task-{stage + 1}.json", predictions)
            self.metrics.append({"checkpoint_task": self.task_id, "task": task, "metrics": metrics})
            self.record_event("evaluation", task_evaluated=task, metrics=metrics)

    def _write_metrics(self):
        with (Path(self.context.output) / "metrics.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("checkpoint_task", "task", "metric", "value"))
            for row in self.metrics:
                for name, value in row["metrics"].items():
                    writer.writerow((row["checkpoint_task"], row["task"], name, value))
        self._materialize_events()
        write_json(Path(self.context.output) / "events.json", self.events)

    def predict(self, inputs, generation_spec):
        """Activate the shared learner for inference and release residency when generation finishes."""
        self._activate(self.model)
        try:
            return self.model.predict(inputs, generation_spec)
        finally:
            self.model.offload()
