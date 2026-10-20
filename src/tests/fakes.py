"""Small numerical executor to test orchestration without downloading models.

This is a two-coordinate squared-error model, not a transformer/LoRA reproduction.
Its named gradient capabilities exercise the interface and routing only.
"""

from dataclasses import asdict, replace
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ..datasets.dataset_base import DatasetBase
from ..models.model_base import ModelBase
from ..utils.contracts import Batch, ModelInfo, Record, ScoreResult, StepReport, fingerprint


def record(sample_id, task="a", x=(1.0, 0.5), target=1.0, label="positive", split="train"):
    return Record(sample_id, "tiny", "v1", sample_id, task, int(sample_id[-1]),
                  split, list(x), target, 2, 64, label)


class TinyDataset(DatasetBase):
    def __init__(self, records=None):
        self.records = tuple(records or (
            record("a0"), record("a1", x=(.5, 1.0), target=-1, label="negative"),
            record("a2", x=(1.0, 1.0)), record("a3", x=(2.0, .5)),
            record("b0", "b", x=(.7, 1.0)), record("b1", "b", x=(1.2, -.5)),
            record("b2", "b", x=(1.0, 2.0), target=-1, label="negative"),
            record("b3", "b", x=(.4, 1.1))))

    @property
    def fingerprint(self):
        return fingerprint([asdict(r) for r in self.records])

    @property
    def has_classes(self):
        return all(r.class_id is not None for r in self.records)

    def load(self, source_spec):
        self.records = tuple(Record(**r) for r in json.loads(Path(source_spec).read_text()))
        return self.records, {"fingerprint": self.fingerprint}

    def split(self, raw_view, split_spec):
        return {name: tuple(r for r in raw_view if r.split == name)
                for name in ("train", "validation", "test")}

    def preprocess(self, view, preprocessing_spec):
        return tuple(view), {"dropped": []}

    def fetch(self, view, batch_spec, cursor):
        if cursor >= len(view):
            return None, cursor
        records = tuple(view[cursor:cursor + batch_spec["size"]])
        return self.collate(records), cursor + len(records)

    def training_view(self, task_id):
        return tuple(r for r in self.records if r.task_id == task_id and r.split == "train")

    def evaluation_view(self, task_id):
        return tuple(replace(r, sample_id="eval-" + r.sample_id, split="test")
                     for r in self.records if r.task_id == task_id)

    def collate(self, records):
        return Batch(tuple(records), torch.tensor([r.inputs for r in records], dtype=torch.float64),
                     sum(r.tokens for r in records), len(records))

    def save(self, path):
        Path(path).write_text(json.dumps([asdict(r) for r in self.records]))


class Controller:
    def __init__(self, fail=None, occurrence=1, skip=False):
        self.fail, self.occurrence, self.skip = fail, occurrence, skip
        self.calls, self.log = {}, []
        self.resident = set()
        self.peak_resident = 0

    def hit(self, operation):
        self.calls[operation] = self.calls.get(operation, 0) + 1
        if self.fail == operation and self.calls[operation] == self.occurrence:
            raise KeyboardInterrupt(operation)


class TinyModel(ModelBase):
    def __init__(self, role="learner", initialization="tiny-base", controller=None, device="cpu"):
        super().__init__(device)
        self.role, self.initialization = role, initialization
        self.controller = controller or Controller()
        dtype = torch.float64 if self.device.type == "cpu" else torch.float32
        self._weights = torch.nn.Parameter(torch.tensor([0.1, -0.2], dtype=dtype, device=self.device))
        self.steps = 0

    @property
    def weights(self):
        return self._weights.detach().tolist()

    @weights.setter
    def weights(self, values):
        with torch.no_grad():
            self._weights.copy_(torch.as_tensor(values, dtype=self._weights.dtype, device=self._weights.device))

    @property
    def info(self):
        return self.describe_role(self.role, self.initialization)

    def describe_role(self, role, initialization):
        return ModelInfo(
            identity=f"{role}:{initialization}", total_parameters=2, trainable_parameters=2,
            capabilities=frozenset(("loss", "last_layer_gradients", "lora_b_gradient",
                                    "fixed_zero_lora_b", "logits", "distillation",
                                    "subset_gradients", "feature_gradient", "layer_fisher", "freeze_aware")),
            cpu_bytes=128, accelerator_bytes=128 if self.device.type != "cpu" else 0, checkpoint_bytes=4096,
            vocabulary="tiny-v1", loss_fingerprint="squared-error-example-mean",
            parameter_views=("head", "lora-b", "subset"),
            layer_names=("lower", "upper"), forward_cost=1,
            backward_costs=(4, 2), legal_prefixes=(0, 1))

    def fork(self, role, initialization):
        return TinyModel(role, initialization, self.controller, self.device)

    def activate(self):
        self._weights = torch.nn.Parameter(self._weights.detach().to(self.device))
        self.controller.resident.add(id(self))
        self.controller.peak_resident = max(self.controller.peak_resident, len(self.controller.resident))

    def offload(self):
        self._weights = torch.nn.Parameter(self._weights.detach().cpu())
        self.controller.resident.discard(id(self))

    def release(self):
        self.offload()

    def parameter_counts(self):
        return 2, 2

    def get_weights(self):
        return self._weights.detach().clone()

    def set_weights(self, weights):
        self.weights, self.steps = weights, 0

    def score(self, batch, score_spec):
        self.controller.hit(f"{self.role}.score")
        inputs = self.to_device(batch.payload.to(self._weights.dtype))
        targets = self._weights.new_tensor([r.target for r in batch.records])
        with torch.no_grad():
            predictions = inputs @ self._weights
            gradients = 2 * (predictions - targets).unsqueeze(-1) * inputs
            if score_spec.kind in {"last_layer_gradients", "subset_gradients"}:
                values = gradients
            elif score_spec.kind == "lora_b_gradient":
                values = gradients.mean(dim=0)
            elif score_spec.kind == "loss":
                values = (predictions - targets).square()
            elif score_spec.kind == "logits":
                values = torch.stack((predictions / score_spec.temperature,
                                      torch.zeros_like(predictions)), dim=-1)
            else:
                raise NotImplementedError(score_spec.kind)
        self.controller.log.append((self.role, "score", score_spec.kind, batch.ids, tuple(self.weights)))
        return ScoreResult(batch.ids, values, fingerprint(self.weights), score_spec.parameter_view,
                           score_spec.reduction)

    def train(self, k, *, batch_factory, objective, hooks, limits):
        """Use Torch autograd and a stateless SGD update for the tiny regression fixture.
        Supervised gradient statistics remain separate from the optional KL objective.
        This exercises tensor handoff, not a transformer or real LoRA representation.
        """
        reports = []
        for _ in range(k):
            for training in batch_factory():
                self.controller.hit(f"{self.role}.train")
                inputs = self.to_device(training.batch.payload.to(self._weights.dtype))
                targets = self._weights.new_tensor([r.target for r in training.batch.records])
                predictions = inputs @ self._weights
                losses = (predictions - targets).square()
                gradients = (2 * (predictions - targets).unsqueeze(-1) * inputs).detach()
                squared = gradients.square().sum(dim=-1).mean().item()
                statistics = {"feature_norm_sq": squared}
                plan = hooks.plan_update(statistics) if hooks.plan_update else None
                prefix = plan.prefix if plan else 0
                diagnostics = {
                    "feature_norm_sq": squared, "freeze_prefix": prefix,
                    "layer_fisher": gradients.square().mean(dim=0).tolist(),
                    "gradient_subset": hooks.gradient_subset,
                    "class_gradients": gradients[:, prefix:],
                }
                before = tuple(self.weights)
                skip = self.controller.skip and self.role == "learner"
                if not skip:
                    if objective.separate_means:
                        loss = losses.new_zeros(())
                        for origin, weight in (("current", objective.current_weight),
                                               ("replay", objective.replay_weight)):
                            mask = torch.tensor([o == origin for o in training.origins], device=self.device)
                            if mask.any():
                                loss = loss + weight * losses[mask].mean()
                    else:
                        loss = losses.mean()
                    if objective.kd_weight and training.teacher_targets is not None:
                        logits = torch.stack((predictions / objective.kd_temperature,
                                              torch.zeros_like(predictions)), dim=-1)
                        teacher = self.to_device(training.teacher_targets.values)
                        kl = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(teacher, dim=-1),
                                      reduction="batchmean")
                        scale = objective.kd_temperature ** 2 if objective.kd_temperature_squared else 1
                        loss = loss + objective.kd_weight * scale * kl
                    gradient = torch.autograd.grad(loss, self._weights)[0]
                    with torch.no_grad():
                        self._weights[prefix:] -= .01 * gradient[prefix:]
                    self.steps += 1
                self.controller.log.append((self.role, "train", training.batch.ids, before,
                                            tuple(self.weights), objective, training.teacher_task))
                reports.append(StepReport(
                    training.batch.ids, not skip, self.steps, training.batch.valid_tokens,
                    {"supervised": (inputs @ self._weights.detach() - targets).square().sum().item()},
                    {"supervised": len(training.batch.ids)}, diagnostics,
                    skip_reason="test overflow" if skip else None))
        return reports

    def evaluate(self, view, evaluation_spec):
        self.controller.hit("evaluate")
        inputs = self._weights.new_tensor([r.inputs for r in view])
        targets = self._weights.new_tensor([r.target for r in view])
        return {"negative_mse": -(inputs @ self._weights.detach() - targets).square().mean().item()}

    def predict(self, inputs, generation_spec):
        return (self._weights.new_tensor(inputs) @ self._weights.detach()).tolist()

    def save(self, path, state_kind):
        self.controller.hit(f"{self.role}.save")
        Path(path).mkdir(parents=True, exist_ok=True)
        Path(path, "model.json").write_text(json.dumps(
            {"weights": self.weights, "steps": self.steps, "state_kind": state_kind}))

    def load(self, path, state_kind):
        value = json.loads(Path(path, "model.json").read_text())
        if value["state_kind"] != state_kind:
            raise ValueError("Wrong checkpoint kind")
        self.weights = value["weights"]
        self.steps = value["steps"] if state_kind == "resume" else 0
