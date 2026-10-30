"""TSR fixed-initialization LoRA-B signatures, source routing and era teachers."""

from dataclasses import dataclass

from .pipeline_base import PipelineBase
from ..utils.contracts import Objective, ReplayPlan, ScoreSpec
from ..utils.numerics import cosine_tensor, sample, softmax, tensor


@dataclass(frozen=True)
class TSRConfig:
    replay_probability: float
    routing_temperature: float
    kd_weight: float
    kd_temperature: float
    kd_temperature_squared: bool
    probe_initialization: str
    gradient_view: str
    probe_batches: int
    memory_per_task: int
    replacement: bool


class TSRPipeline(PipelineBase):
    def __init__(self, config, model=None, dataset=None):
        """Track immutable historical signatures/teachers and any unfinished current probe."""
        super().__init__(config, model, dataset)
        self.policy = {"tasks": {}, "current_signature": None, "probe": None}

    def required_capabilities(self):
        """Require the learner to execute the routed distillation objective."""
        return {"distillation"}

    def auxiliary_roles(self):
        """Declare the fixed probe and learner-compatible historical teacher before allocation."""
        return (("probe", self.config.probe_initialization), ("teacher", self.model.info.identity))


    def dependencies(self):
        """Return teacher references that must survive with this policy's recovery checkpoint."""
        return tuple(record["teacher"] for record in self.policy["tasks"].values())

    def probe_batch(self, batch, model, reduction):
        """Score one batch in the fixed LoRA-B coordinate view without taking an optimizer step."""
        spec = ScoreSpec("lora_b_gradient", reduction, self.config.gradient_view)
        return model.score(batch, spec)

    def accumulate_probe(self, result, ids):
        """Add a batch-mean probe gradient to the saved task-signature sum.
        Advance the record cursor and batch count together, keeping the exact observed IDs.
        The final signature weights batches equally, including a smaller last batch.
        """
        state = self.policy["probe"]
        gradient = tensor(result.values)
        total = gradient if state["sum"] is None else tensor(state["sum"], gradient.device) + gradient
        state["sum"] = total.tolist()
        state["cursor"] += len(ids)
        state["batches"] += 1
        state["ids"].extend(ids)

    def task_signature(self, model, reduction):
        """Average the completed probe sum and attach its coordinate and initialization provenance.
        Copy the probed IDs so later probe bookkeeping cannot change this signature.
        """
        state = self.policy["probe"]
        return {
            "values": (tensor(state["sum"], model.device) / state["batches"]).tolist(),
            "initialization": self.config.probe_initialization,
            "view": self.config.gradient_view, "ids": list(state["ids"]),
            "weighting": "mean_of_batch_gradients", "reduction": reduction,
            "model": model.info.identity,
        }

    def probe_task(self, records, owner):
        """Acquire the same isolated probe initialization for each current task.
        Resume its partial sum/cursor and score up to the configured number of batches.
        Checkpoint after each observation, then finalize the signature and release the probe.
        """
        probe_state = self.policy["probe"]
        if probe_state is None:
            probe_state = {"cursor": 0, "batches": 0, "sum": None, "ids": []}
            self.policy["probe"] = probe_state
        model = owner.acquire_auxiliary("probe", self.config.probe_initialization)
        while (probe_state["cursor"] < len(records)
               and probe_state["batches"] < self.config.probe_batches):
            start = probe_state["cursor"]
            batch = owner.collate(records[start:start + owner.spec.batch_size])
            result = self.probe_batch(batch, model, owner.spec.reduction)
            self.accumulate_probe(result, batch.ids)
            limit = min(len(records), self.config.probe_batches * owner.spec.batch_size)
            owner.log_sample_progress(f"probe task={owner.task_id}", probe_state["cursor"], limit)
            owner.record_event("probe", ids=list(batch.ids), cost=result.cost)
            owner.boundary()
        self.policy["current_signature"] = self.task_signature(model, owner.spec.reduction)
        owner.release_auxiliary()
        self.policy["probe"] = None
        return self.policy["current_signature"]

    def routing_weights(self, signature, historical_signatures):
        """Softmax signed cosine similarities to historical signatures using the routing temperature."""
        if not historical_signatures:
            return ()
        previous = tensor([old["values"] for old in historical_signatures], self.device)
        similarities = cosine_tensor(previous, signature["values"]) / self.config.routing_temperature
        return softmax(similarities)

    @staticmethod
    def select_teacher(tasks, probabilities):
        """Choose the most compatible historical task; task ID resolves equal routing weights."""
        return min(zip(tasks, probabilities), key=lambda pair: (-pair[1], pair[0]))[0]

    def route_replay(self, signature, replay):
        """Route only over tasks that still have retained records.
        Pair each task's routing weight with its own immutable era teacher.
        Use the highest-weight teacher for current data; empty history has no teacher or replay.
        """
        tasks = replay.tasks
        if not tasks:
            return ReplayPlan((), (), self.config.replay_probability, {}, None)
        historical = [self.policy["tasks"][task] for task in tasks]
        probabilities = self.routing_weights(signature, [r["signature"] for r in historical])
        teachers = {task: record["teacher"] for task, record in zip(tasks, historical)}
        top = self.select_teacher(tasks, probabilities)
        return ReplayPlan(tasks, probabilities, self.config.replay_probability, teachers, top)

    def retrieve_replay(self, records, count):
        """Sample uniformly within the routed task using the declared replacement policy.
        Without replacement, a short task memory produces a smaller batch.
        """
        if not records:
            return ()
        if not self.config.replacement:
            count = min(count, len(records))
        return sample(records, [1 / len(records)] * len(records),
                      count, self.config.replacement, self.rng)

    def draw_branch(self, plan):
        """Draw current versus replay once for the whole update.
        A replay draw samples a task using routing weights and uses that task's teacher.
        A current draw uses the most compatible teacher; empty history always selects current.
        """
        if not plan.tasks:
            return {"origin": "current", "task": None}
        if self.rng.random() < plan.replay_probability:
            task = sample(plan.tasks, plan.probabilities, 1, True, self.rng)[0]
            return {"origin": "replay", "task": task}
        return {"origin": "current", "task": plan.current_teacher}

    def objective(self, reduction, has_teacher):
        """Specify supervised loss and optional KL for the model executor.
        KD temperature/scaling are independent of the temperature used for task routing.
        """
        return Objective(reduction, kd_weight=self.config.kd_weight if has_teacher else 0,
                         kd_temperature=self.config.kd_temperature,
                         kd_temperature_squared=self.config.kd_temperature_squared)

    def teacher_outputs(self, batch, task, objective, owner):
        """Resolve the routed task's frozen teacher and request detached targets.
        The runtime loads/releases the teacher before the learner constructs its training graph.
        Skip teacher inference for the first task or a zero distillation weight.
        """
        if task is None or objective.kd_weight == 0:
            return None
        reference = self.policy["tasks"][task]["teacher"]
        spec = ScoreSpec("logits", owner.spec.reduction, temperature=objective.kd_temperature)
        result = owner.teacher_scores(reference, batch, spec)
        owner.record_event("teacher_inference", task_teacher=task, ids=list(batch.ids), cost=result.cost)
        return result

    def snapshot_task(self, owner):
        """Freeze the learner once at the task boundary and pair it with the task signature.
        Store that pair before forcing a checkpoint so resume cannot create a different era.
        The historical entry remains unchanged during subsequent tasks.
        """
        if owner.task_id not in self.policy["tasks"]:
            teacher = owner.artifacts.blob(owner.model)
            self.policy["tasks"][owner.task_id] = {
                "signature": self.policy["current_signature"], "teacher": teacher}
            owner.record_event("teacher_snapshot", teacher=teacher)
            owner.boundary(force=True)

    def open_task(self):
        """Probe permitted current training data before beginning routed learner updates."""
        self.probe_task(self.current, self)

    def next_update(self):
        """Keep the sampled source branch in progress until the update is consumed.
        Choose current data or uniform replay within that source, preserving its teacher pairing.
        Return the objective and teacher task for execution by the shared lifecycle.
        """
        plan = self.route_replay(self.policy["current_signature"], self.replay)
        if "branch" not in self.progress:
            self.progress["branch"] = self.draw_branch(plan)
        branch = self.progress["branch"]
        if branch["origin"] == "current":
            records = self.current_batch()
        else:
            eligible = self.replay.task(branch["task"])
            records = self.retrieve_replay(eligible, self.spec.batch_size)
        if not records:
            return None
        objective = self.objective(self.spec.reduction, branch["task"] is not None)
        return records, [branch["origin"]] * len(records), objective, branch["task"], None

    def teacher_targets(self, batch, task, objective):
        """Use this standalone pipeline as the owner of teacher loading and event recording."""
        return self.teacher_outputs(batch, task, objective, self)

    def build_task_memory(self, records):
        """Choose up to memory_per_task current records uniformly without replacement."""
        count = min(len(records), self.config.memory_per_task)
        return tuple(self.rng.sample(records, count))

    def finish_task(self):
        """Snapshot the era teacher, then append this task's subsample to native TSR history."""
        self.snapshot_task(self)
        return self.replay.records + self.build_task_memory(self.current)
