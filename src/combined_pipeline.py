"""One learner lifecycle: OASIS admission -> TSR routing -> SAR/aL -> CSReL."""

from dataclasses import dataclass

from .pipelines.pipeline_base import PipelineBase


@dataclass(frozen=True)
class CombinedConfig:
    recipe: str = "oasis_tsr_al_sar_csrel_task_chunks"


class CombinedPipeline(PipelineBase):
    def __init__(self, *, oasis, tsr, al_sar, csrel, model=None, dataset=None):
        """Compose the four policy objects under one lifecycle owner and optional shared learner."""
        super().__init__(CombinedConfig(), model, dataset)
        self.oasis, self.tsr, self.al_sar, self.csrel = oasis, tsr, al_sar, csrel
        if model is not None:
            self.set_model(model)

    def components(self):
        """Expose stable component names for model binding and policy/RNG checkpoint round trips."""
        return {"combined": self, "oasis": self.oasis, "tsr": self.tsr,
                "al_sar": self.al_sar, "csrel": self.csrel}

    def required_capabilities(self):
        """Union learner requirements; CSReL scoring capabilities belong to its auxiliary roles."""
        return (self.oasis.required_capabilities() | self.tsr.required_capabilities()
                | self.al_sar.required_capabilities())

    def auxiliary_roles(self):
        """Declare CSReL fit roles, the fixed TSR probe and historical teachers for residency checks."""
        return (*self.csrel.auxiliary_roles(),
                ("probe", self.tsr.config.probe_initialization),
                ("teacher", self.model.info.identity))


    def dependencies(self):
        """Retain the historical teacher artifacts required by TSR's active policy."""
        return self.tsr.dependencies()

    def open_task(self):
        """Reconcile retained exposure and probe the entire permitted current training view.
        TSR signatures are computed before OASIS filtering changes the learner's current batches.
        """
        self.al_sar.reconcile(self.replay.records)
        self.tsr.probe_task(self.current, self)

    def _admit(self, records):
        # All observed current records participate in active-population accounting,
        # including rejected records with zero direct exposure.
        """Apply OASIS to current arrivals and record its selected IDs for the owner's batch queue."""
        self.al_sar.register(records)
        result = self.oasis.select_current(self.collate(records), self.model, self.spec.reduction)
        self.record_event("oasis_selection", selection=result.__dict__)
        return result.selected_ids

    def next_update(self):
        """Use TSR to choose the update source and its coherent historical teacher.
        Current batches pass through OASIS; replay batches use SAR within the chosen task.
        Attach aL hooks and the TSR objective while keeping the source decision pending until consumed.
        """
        plan = self.tsr.route_replay(self.tsr.policy["current_signature"], self.replay)
        if "branch" not in self.progress:
            self.progress["branch"] = self.tsr.draw_branch(plan)
        branch = self.progress["branch"]
        # Establish a nonempty admitted source even for alpha=1. An all-rejected
        # task cannot silently become replay-only training.
        if not self.progress["admitted"] and not self.current_batch(self._admit, peek=True):
            return None
        if branch["origin"] == "current":
            records = self.current_batch(self._admit)
        else:
            eligible = self.replay.task(branch["task"])
            count = self.spec.batch_size if self.al_sar.config.replacement else min(
                self.spec.batch_size, len(eligible))
            records = self.al_sar.retrieve_replay(eligible, count)
        if not records:
            return None
        objective = self.tsr.objective(self.spec.reduction, branch["task"] is not None)
        return (records, [branch["origin"]] * len(records), objective, branch["task"],
                self.al_sar.hooks(self.model.info))

    def teacher_targets(self, batch, task, objective):
        """Compute TSR teacher outputs through the combined owner's isolated auxiliary runtime."""
        return self.tsr.teacher_outputs(batch, task, objective, self)

    def observe(self, report, records):
        """Apply aL-SAR's successful-update statistics and hybrid exposure clock."""
        self.al_sar.observe_update(report, records, self.model.info)

    def finish_task(self):
        """Capture the current era teacher, then let CSReL propose the next replay memory.
        All current records remain coreset candidates, including those rejected by OASIS.
        The shared lifecycle commits the proposal after checking its storage budget.
        """
        self.tsr.snapshot_task(self)
        return self.csrel.build_memory(
            self.current, self.replay, self.spec.task_ids[:self.task_index + 1], self)

    def after_memory_commit(self):
        """Finalize CSReL reference losses and reconcile SAR counts with the accepted memory.
        Restrict future task routing to tasks that still have retained examples.
        Physical teacher pruning is separate because recovery checkpoints may still need old blobs.
        """
        self.csrel.commit()
        self.al_sar.reconcile(self.replay.records)
        # Required teacher blobs survive while either retained recovery state
        # references them; active routing cannot use evicted tasks.
        self.tsr.policy["tasks"] = {task: record for task, record in self.tsr.policy["tasks"].items()
                                    if task in self.replay.tasks}
