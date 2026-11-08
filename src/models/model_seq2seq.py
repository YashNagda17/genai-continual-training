"""Shared encoder-decoder execution for mT5. Transformers supplies the architecture."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import copy
import gc
import math
import random
try:
    import resource
except ImportError:
    resource = None
import shutil
import tempfile
import time

import torch
from torch import nn
from torch.nn import functional as F

from .model_base import ModelBase
from ..utils.contracts import (
    AdapterPlan, ModelInfo, ScoreResult, StepReport, model_boundary,
)
from ..utils.state import Checkpoints, read_json, write_json


@dataclass(frozen=True)
class Seq2SeqSettings:
    """Numerical choices for one encoder-decoder role. Full fine-tuning is rejected."""
    mode: str = "lora"
    rank: int = 16
    alpha: float = 32
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    total_steps: int | None = None
    seed: int = 0
    max_source_length: int = 256
    max_target_length: int = 64
    max_batch_size: int = 18
    logit_chunk_size: int = 32
    max_score_bytes: int = 2 * 1024 ** 3
    working_bytes: int = 1024 ** 3
    optimizer_dtype: str = "float32"


_PROJECTIONS = {"q", "k", "v", "o", "wi_0", "wi_1", "wo"}
_PARTS = {"SelfAttention": "self_attention", "EncDecAttention": "cross_attention",
          "DenseReluDense": "feed_forward"}
_BANDS = ("1-2", "3-4", "5-6", "7-8")


def cpu_tree(value):
    """Copy optimizer tensors to CPU without changing the container shape."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [cpu_tree(item) for item in value]
        return type(value)(items)
    return value


def _adapter_parameter_names(module):
    return {name for name, _parameter in module.named_parameters()
            if name.endswith((".A", ".B", ".extra_A", ".extra_B"))}


def _count(module, *, trainable_only=False):
    """Torch parameters() already removes tied aliases."""
    return sum(p.numel() for p in module.parameters() if not trainable_only or p.requires_grad)


def _seed_for(seed, name):
    return random.Random(f"{seed}:{name}").getrandbits(63)


def _init_A(parameter, seed=None):
    """Kaiming fill on the parameter's device. A meta tensor has no storage to fill."""
    if parameter.device.type == "meta":
        return
    generator = None
    if seed is not None:
        generator = torch.Generator(device=parameter.device).manual_seed(seed)
    nn.init.kaiming_uniform_(parameter, a=math.sqrt(5), generator=generator)


class Seq2SeqLoRA(nn.Module):
    """Frozen projection plus (alpha / rank) B A, with B starting at zero."""

    def __init__(self, base, rank, alpha, *, block_key, block_index, part, projection, deterministic, seed):
        super().__init__()
        self.base = base
        self.block_key, self.block_index = block_key, block_index
        self.part, self.projection = part, projection
        self.scale = alpha / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        if deterministic:
            _init_A(self.A, seed)
        else:
            _init_A(self.A)
        self.base.requires_grad_(False)

    @property
    def weight(self):
        """The frozen projection. The built-in feed-forward reads this dtype."""
        return self.base.weight

    def add_extra(self, rank, seed):
        extra_A = nn.Parameter(self.base.weight.new_empty(rank, self.base.in_features))
        _init_A(extra_A, seed)
        self.extra_A = extra_A
        self.extra_B = nn.Parameter(self.base.weight.new_zeros(self.base.out_features, rank))
        self.extra_scale = 2.0

    def clear_extra(self):
        for name in ("extra_A", "extra_B"):
            if name in self._parameters:
                del self._parameters[name]
        self.extra_scale = None

    def forward(self, inputs):
        update = F.linear(F.linear(inputs, self.A), self.B) * self.scale
        if getattr(self, "extra_scale", None):
            update = update + F.linear(F.linear(inputs, self.extra_A), self.extra_B) * self.extra_scale
        return self.base(inputs) + update

    def delta(self):
        """Effective update in float32, including the LoRA scale and any extra rank."""
        value = (self.B.float() @ self.A.float()) * self.scale
        if getattr(self, "extra_scale", None):
            value = value + (self.extra_B.float() @ self.extra_A.float()) * self.extra_scale
        return value


def _adapter_identity(name):
    """Map ``encoder.block.0.layer.0.SelfAttention.q`` to the contract names."""
    pieces = name.split(".")
    if len(pieces) < 7 or pieces[1] != "block" or pieces[-1] not in _PROJECTIONS:
        return None
    part = _PARTS.get(pieces[-2])
    if part is None or pieces[0] not in {"encoder", "decoder"}:
        return None
    block = int(pieces[2]) + 1
    return pieces[0], block, part, pieces[-1]


def _detach_value(*values):
    """Stop the backward pass at this output. Values are unchanged."""
    value = values[-1]
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_value(item) for item in value)
    items = getattr(value, "items", None)
    if callable(items):
        for key, item in list(value.items()):
            if torch.is_tensor(item):
                value[key] = item.detach()
    return value


class Seq2SeqLMModel(ModelBase):
    """Encoder reads the article. Decoder writes the summary one token at a time."""

    model_id = ""
    model_type = ""
    untie_head = False
    expected_parameters = None

    def __init__(self, settings=None, *, device="auto", revision="main", cache_dir=None,
                 collate=None, collate_sources=None, role="learner", initialization="base",
                 config=None, tokenizer=None, model_factory=None, pretrained=True):
        super().__init__(device)
        self.settings = settings or Seq2SeqSettings()
        model_boundary(self.settings.mode == "lora", "Seq2seq training is LoRA only")
        model_boundary(self.settings.optimizer_dtype != "float16", "Seq2seq training does not use fp16")
        self.revision = revision
        self.cache_dir = Path(cache_dir or Path(__file__).parent / "model_weights")
        self.collate, self.collate_sources = collate, collate_sources
        self.role, self.initialization = role, initialization
        self.config, self._tokenizer = config, tokenizer
        self.model_factory, self.pretrained = model_factory, pretrained
        self.module = self.optimizer = self.scheduler = self.scaler = None
        self.steps = self.weight_revision = 0
        self._weights_identity = None
        self._profiles, self._prepared = {}, False
        self._offload = None
        self._save_full_weights = False
        self._base_path, self._base_bytes = None, 0
        self.rng_state = None
        self.resources = None
        self._new_rank = {}
        self._trainable = None
        self._lr_scale = {}
        self._plan = None
        self._cut_hooks = []
        self._plan_fingerprint = None
        self._merge_deltas = None
        self._merge_state = None
        self._script_ids = None
        self.artifact = None

    def _load_dtype(self):
        """bf16 on devices that support it. fp32 elsewhere. Never fp16."""
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        backend = getattr(torch, self.device.type, None)
        supported = getattr(backend, "is_bf16_supported", None)
        if self.device.type != "cpu" and callable(supported) and supported():
            return torch.bfloat16
        return torch.float32

    def _prepare(self):
        if self._prepared:
            return
        if self.config is None or self._tokenizer is None or self.model_factory is None:
            from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
            if self.config is None:
                self.config = AutoConfig.from_pretrained(
                    self.model_id, revision=self.revision, cache_dir=self.cache_dir)
            if self._tokenizer is None:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_id, revision=self.revision, cache_dir=self.cache_dir)
                except ImportError as error:
                    raise ImportError(
                        "mT5's SentencePiece tokenizer requires the sentencepiece package"
                    ) from error
            self.model_factory = self.model_factory or (
                lambda config: AutoModelForSeq2SeqLM.from_config(config, attn_implementation="eager"))
        self._vocabulary = self.model_id
        self._prepared = True

    @property
    def tokenizer(self):
        self._prepare()
        return self._tokenizer

    def encode(self, text):
        """Encode plain text. mT5 has no chat template."""
        return self.tokenizer.encode(text, add_special_tokens=True)

    def preprocessing_kwargs(self, *, max_source=None, max_target=None):
        """Arguments for Seq2SeqPreprocessing. This class does not import the dataset."""
        tok = self.tokenizer
        return {"encode": self.encode,
                "eos_id": tok.eos_token_id,
                "pad_id": tok.pad_token_id if tok.pad_token_id is not None else 0,
                "max_source": max_source or self.settings.max_source_length,
                "max_target": max_target or self.settings.max_target_length,
                "identity": {"model": self.model_id, "revision": self.revision,
                             "vocabulary": self._vocabulary}}

    def bind_dataset(self, dataset):
        super().bind_dataset(dataset)
        self.collate_sources = getattr(dataset, "collate_sources", self.collate_sources)

    def _untie_lm_head(self, module):
        """Keep a separate output head when the Hub checkpoint says it is not tied."""
        if not self.untie_head:
            return
        if module.lm_head.weight is module.shared.weight:
            module.lm_head.weight = nn.Parameter(module.shared.weight.detach().clone())
        tied = getattr(module, "_tied_weights_keys", None)
        if isinstance(tied, dict) and "lm_head.weight" in tied:
            module._tied_weights_keys = {key: value for key, value in tied.items() if key != "lm_head.weight"}

    def _attach_lora(self, module, *, deterministic):
        for name, child in list(module.named_modules()):
            identity = _adapter_identity(name)
            if identity is None or not isinstance(child, nn.Linear):
                continue
            stack, block, part, projection = identity
            parent, attribute = name.rsplit(".", 1)
            seed = _seed_for(self.settings.seed, name)
            setattr(module.get_submodule(parent), attribute, Seq2SeqLoRA(
                child, self.settings.rank, self.settings.alpha, block_key=f"{stack}.{block}",
                block_index=block - 1, part=part, projection=projection,
                deterministic=deterministic, seed=seed))
        self._mark_adapters(module)

    def _mark_adapters(self, module):
        """Train adapter factors only. The base weights, embeddings, and output head stay frozen."""
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(name.endswith((".A", ".B", ".extra_A", ".extra_B")))

    def _apply_mask(self, module):
        """Freeze shared adapters outside the installed plan. Extra rank stays trainable."""
        if self._trainable is None:
            return
        for child in module.modules():
            if not isinstance(child, Seq2SeqLoRA):
                continue
            shared = child.block_key in self._trainable
            child.A.requires_grad_(shared)
            child.B.requires_grad_(shared)

    def _add_new_rank(self, module, new_rank):
        for name, child in module.named_modules():
            if not isinstance(child, Seq2SeqLoRA) or child.block_key not in new_rank:
                continue
            rank = int(new_rank[child.block_key])
            child.add_extra(rank, _seed_for(self.settings.seed, "extra:" + name))

    def _build(self, config):
        module = self.model_factory(copy.deepcopy(config))
        self._untie_lm_head(module)
        return module

    def _profile(self):
        self._prepare()
        key = (self.role != "serving", self.settings.rank, self.settings.alpha,
               tuple(sorted(self._new_rank.items())), self.untie_head,
               None if self._trainable is None else tuple(sorted(self._trainable)))
        if key not in self._profiles:
            with torch.device("meta"):
                module = self._build(self.config)
                base = _count(module)
                if self.role != "serving":
                    self._attach_lora(module, deterministic=self.role == "probe")
                    self._add_new_rank(module, self._new_rank)
                    self._apply_mask(module)
            if self.expected_parameters is not None and self.untie_head:
                model_boundary(base == self.expected_parameters,
                               f"{self.model_id} has {base} base parameters, expected {self.expected_parameters}")
            self._profiles[key] = (_count(module), _count(module, trainable_only=True), base)
        return self._profiles[key]

    def describe_role(self, role, initialization):
        saved_role, self.role = self.role, role
        try:
            total, trainable, _base = self._profile()
        finally:
            self.role = saved_role
        identity = f"{self.model_id}:{role}:{initialization}"
        weight_bytes = total * 4
        saved = trainable if role != "serving" and not self._save_full_weights else total
        peak = weight_bytes + trainable * 12 + self.settings.working_bytes + self.settings.max_score_bytes
        capabilities = frozenset({"example_loss", "token_loss", "generation", "probe_gradient"})
        blocks = self.config.num_layers + self.config.num_decoder_layers
        return ModelInfo(identity, total, trainable, capabilities=capabilities,
                         cpu_bytes=peak, accelerator_bytes=peak if self.device.type != "cpu" else 0,
                         checkpoint_bytes=saved * 4 + trainable * 12 + 1024 ** 2,
                         vocabulary=self._vocabulary, loss_fingerprint="seq2seq-summary-ce-v1",
                         parameter_views=("cell", "block"),
                         layer_names=tuple(f"block.{index}" for index in range(blocks)),
                         forward_cost=float(total), backward_costs=tuple(2.0 for _ in range(blocks)),
                         legal_prefixes=(0,))

    @property
    def info(self):
        return self.describe_role(self.role, self.initialization)

    def fork(self, role, initialization):
        """Unloaded isolated role. A probe does not copy the evolving learner."""
        self._prepare()
        child = type(self)(self.settings, device=str(self.device), revision=self.revision,
                           cache_dir=self.cache_dir, collate=self.collate, collate_sources=self.collate_sources,
                           role=role, initialization=initialization, config=copy.deepcopy(self.config),
                           tokenizer=self.tokenizer, model_factory=self.model_factory, pretrained=self.pretrained)
        child._profiles = self._profiles
        child.runtime_root, child.resources = self.runtime_root, self.resources
        return child

    def _assets(self):
        if not self.pretrained:
            return
        from huggingface_hub import snapshot_download
        self._base_path = Path(snapshot_download(
            self.model_id, revision=self.revision, cache_dir=self.cache_dir,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "pytorch_model.bin"]))
        self._base_bytes = sum(path.stat().st_size for path in self._base_path.rglob("*") if path.is_file())

    @property
    def base_storage_bytes(self):
        return self._base_bytes

    def upgrade_parameters(self, dtype=None):
        self.module.to(dtype=torch.float32 if dtype is None else dtype)

    def _optimizer_parameters(self):
        groups, new_params = [], []
        by_block = {}
        for name, parameter in self.module.named_parameters():
            if not parameter.requires_grad or not name.endswith((".A", ".B", ".extra_A", ".extra_B")):
                continue
            if ".extra_" in name:
                new_params.append(parameter)
                continue
            owner = name.rsplit(".", 1)[0]
            module = self.module.get_submodule(owner)
            by_block.setdefault(module.block_key, []).append(parameter)
        scales = getattr(self, "_lr_scale", {})
        for block_key in sorted(by_block):
            groups.append({"params": by_block[block_key],
                           "lr": self.settings.learning_rate * float(scales.get(block_key, 1.0))})
        if new_params:
            groups.append({"params": new_params, "lr": self.settings.learning_rate})
        return groups

    def _reset_optimizer(self):
        groups = self._optimizer_parameters()
        self.module.zero_grad(set_to_none=True)
        self.optimizer = torch.optim.AdamW(groups, lr=self.settings.learning_rate,
                                           weight_decay=self.settings.weight_decay) if groups else None
        if self.optimizer is None:
            self.scheduler = self.scaler = None
            return
        def schedule(step):
            if self.settings.warmup_steps and step < self.settings.warmup_steps:
                return (step + 1) / self.settings.warmup_steps
            if self.settings.total_steps is None:
                return 1.0
            return max(0.0, (self.settings.total_steps - step)
                       / max(1, self.settings.total_steps - self.settings.warmup_steps))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=False)

    def _load_checkpoint(self, module):
        path = self._base_path / "pytorch_model.bin"
        safetensors = list(self._base_path.glob("*.safetensors"))
        if safetensors:
            from safetensors.torch import load_file
            state = load_file(safetensors[0], device="cpu")
        else:
            model_boundary(path.is_file(), f"{self.model_id} has no pytorch_model.bin or safetensors")
            state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = module.load_state_dict(state, strict=False)
        ignored = set(getattr(module, "_keys_to_ignore_on_load_unexpected", []))
        bad = [key for key in unexpected if key not in ignored]
        model_boundary(not missing and not bad, f"Checkpoint mismatch missing={missing} unexpected={bad}")

    def activate(self):
        if self.module is not None:
            return
        self.info.validate()
        if self.resources is not None:
            self.resources.validate_role(self.info)
        self._assets()
        load_dtype = self._load_dtype()
        model_boundary(load_dtype != torch.float16, "Refusing to load a seq2seq model in fp16")
        devices = [] if self.device.type == "cpu" else [self.device]
        with torch.random.fork_rng(devices=devices, device_type=self.device.type), torch.device(self.device):
            self._restore_rng(self._new_rng(self.settings.seed))
            module = self._build(self.config)
            if self.pretrained:
                self._load_checkpoint(module)
            if self.role != "serving":
                self._restore_rng(self._new_rng())
                self._attach_lora(module, deterministic=self.role == "probe")
                self._add_new_rank(module, self._new_rank)
                self._mark_adapters(module)
                self._apply_mask(module)
            elif self._merge_deltas:
                with torch.no_grad():
                    for name, value in self._merge_state.items():
                        parameter = module.get_parameter(name)
                        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
                    for name, delta in self._merge_deltas.items():
                        weight = module.get_parameter(name + ".weight")
                        weight.add_(delta.to(device=weight.device, dtype=weight.dtype))
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        self.module = module.to(device=self.device, dtype=load_dtype)
        self.module.config.use_cache = False
        if self.role != "serving" and self._trainable is not None:
            self._cut_backward()
        self.upgrade_parameters()
        self._reset_optimizer()
        self.rng_state = self.rng_state or self._new_rng()
        if self._offload is not None:
            self._read_state(Path(self._offload.name))

    def _seed(self):
        return _seed_for(self.settings.seed, self.initialization)

    def parameter_counts(self):
        return _count(self.module), _count(self.module, trainable_only=True)

    def _clear_cuts(self):
        for handle in self._cut_hooks:
            handle.remove()
        self._cut_hooks = []

    def _lowest_trainable(self, stack):
        indexes = [module.block_index for module in self.module.modules()
                   if isinstance(module, Seq2SeqLoRA) and module.block_key.startswith(stack + ".")
                   and any(parameter.requires_grad for parameter in module.parameters())]
        return min(indexes) if indexes else None

    def _cut_backward(self):
        """Detach the block below the lowest trainable one so its backward is not run."""
        self._clear_cuts()
        for stack in ("encoder", "decoder"):
            lowest = self._lowest_trainable(stack)
            if lowest == 0:
                continue
            if lowest is None:
                module = self.module.get_submodule(stack)
            else:
                module = self.module.get_submodule(f"{stack}.block")[lowest - 1]
            self._cut_hooks.append(module.register_forward_hook(_detach_value))

    def apply_plan(self, plan):
        """Install adapter trainability before the first update and build a fresh optimizer."""
        self.activate()
        if not isinstance(plan, AdapterPlan):
            plan = AdapterPlan(tuple(plan["trainable"]), dict(plan.get("new_rank") or {}),
                               dict(plan.get("lr_scale") or {}), dict(plan.get("identity") or {}))
        trainable, new_rank, scales = set(plan.trainable), dict(plan.new_rank), dict(plan.lr_scale)
        for module in self.module.modules():
            if isinstance(module, Seq2SeqLoRA):
                module.clear_extra()
        self._new_rank = new_rank
        self._trainable = set(trainable)
        self._lr_scale = scales
        self._plan = plan
        self._add_new_rank(self.module, new_rank)
        self._apply_mask(self.module)
        self._profiles.clear()
        self._cut_backward()
        self.optimizer = None
        self._reset_optimizer()
        trainable_count = _count(self.module, trainable_only=True)
        new_count = sum(parameter.numel() for name, parameter in self.module.named_parameters()
                        if parameter.requires_grad and ".extra_" in name)
        self._plan_fingerprint = {"identity": plan.identity, "trainable": sorted(trainable),
                                  "new_rank": new_rank, "lr_scale": scales,
                                  "trainable_parameters": trainable_count, "new_parameters": new_count}
        self.weight_revision += 1
        return {"trainable_parameters": trainable_count, "new_parameters": new_count,
                "fingerprint": self._plan_fingerprint}

    def _capture_rng(self):
        states = {"cpu": torch.random.get_rng_state()}
        if self.device.type != "cpu":
            states["device"] = getattr(torch, self.device.type).get_rng_state(self.device).cpu()
        return states

    def _restore_rng(self, states):
        torch.random.set_rng_state(states["cpu"])
        if self.device.type != "cpu":
            getattr(torch, self.device.type).set_rng_state(states["device"], self.device)

    def _new_rng(self, seed=None):
        seed = self._seed() if seed is None else seed
        states = {"cpu": torch.Generator(device="cpu").manual_seed(seed).get_state()}
        if self.device.type != "cpu":
            states["device"] = torch.Generator(device=self.device).manual_seed(seed).get_state().cpu()
        return states

    def _synchronize(self):
        if self.device.type != "cpu":
            getattr(torch, self.device.type).synchronize(self.device)

    def _begin_timed_block(self):
        self._synchronize()
        backend = getattr(torch, self.device.type, None)
        if self.device.type != "cpu" and hasattr(backend, "reset_peak_memory_stats"):
            backend.reset_peak_memory_stats(self.device)
        return time.perf_counter()

    def _timing_cost(self, start, **fields):
        self._synchronize()
        cost = {"device": str(self.device), "synchronized_wall_seconds": time.perf_counter() - start,
                "process_peak_cpu_bytes": None if resource is None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                **fields}
        backend = getattr(torch, self.device.type, None)
        if self.device.type != "cpu" and hasattr(backend, "max_memory_allocated"):
            cost["peak_device_allocated_bytes"] = backend.max_memory_allocated(self.device)
        return cost

    @contextmanager
    def _execution(self, *, mode="inference", advance=False):
        if mode not in {"training", "inference"}:
            raise ValueError("Execution mode must be training or inference")
        active = self.module is not None
        self.activate()
        caller = self._capture_rng()
        modes = [(item, item.training) for item in self.module.modules()]
        flags = [(parameter, parameter.requires_grad) for parameter in self.module.parameters()]
        buffers = [] if advance else [(buffer, buffer.detach().clone()) for buffer in self.module.buffers()]
        self._restore_rng(self.rng_state)
        self.module.train(mode == "training")
        try:
            yield mode
        finally:
            if advance:
                self.rng_state = self._capture_rng()
            self._restore_rng(caller)
            for module, saved in modes:
                module.training = saved
            for parameter, flag in flags:
                parameter.requires_grad_(flag)
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
            if not active:
                self.offload()

    def _payload(self, batch):
        payload = batch.payload if hasattr(batch, "payload") else batch
        return self.to_device(payload)

    def _forward(self, payload):
        """One call into the built-in seq2seq model. It shifts the labels and runs both stacks."""
        return self.module(input_ids=payload["input_ids"], attention_mask=payload.get("attention_mask"),
                           labels=payload.get("labels"), use_cache=False, return_dict=True)

    def _losses(self, payload):
        """Per-example mean NLL from the model's own logits. Padding labels are ignored."""
        labels = payload["labels"]
        logits = self._forward(payload).logits
        valid = labels != -100
        counts = valid.sum(-1).clamp_min(1)
        chosen = labels.reshape(-1)[valid.reshape(-1)]
        flat = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1])[valid.reshape(-1)], chosen, reduction="none")
        owners = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(valid)[valid]
        sums = torch.zeros(len(counts), device=logits.device, dtype=flat.dtype).scatter_add(0, owners, flat)
        return sums / counts, sums, counts, flat

    def _reduce(self, means, sums, counts, reduction):
        if reduction == "per_example_mean":
            return means.mean()
        return sums.sum() / counts.sum()

    def score(self, batch, score_spec):
        with self._execution():
            payload = self._payload(batch)
            start = self._begin_timed_block()
            if score_spec.kind in {"example_loss", "loss"}:
                values = self._losses(payload)[0].detach().float()
            elif score_spec.kind == "token_loss":
                values = self._losses(payload)[3].detach().float()
            elif score_spec.kind == "probe_gradient":
                values = self._probe_gradient(payload, score_spec)
            else:
                raise ValueError(f"Seq2seq model cannot score {score_spec.kind}")
            cost = self._timing_cost(start)
            revision = f"{self.model_id}:{self.steps}"
        tensor = values if torch.is_tensor(values) else values["gradients"][0]
        nbytes = tensor.numel() * tensor.element_size() if torch.is_tensor(values) else sum(
            item.numel() * item.element_size() for item in values["gradients"])
        on_cpu = tensor.device.type == "cpu"
        return ScoreResult(batch.ids, values, revision, score_spec.parameter_view or score_spec.kind,
                           score_spec.reduction, cost, cpu_bytes=nbytes if on_cpu else 0,
                           accelerator_bytes=0 if on_cpu else nbytes)

    def _probe_gradient(self, payload, score_spec):
        """Batch-mean B gradients on an isolated probe, grouped by block or cell."""
        model_boundary(self.role == "probe" and score_spec.parameter_view in {"cell", "block"},
                       "Probe gradients require a probe role and a cell or block view")
        selected = [(name, p) for name, p in self.module.named_parameters() if name.endswith(".B")]
        means, sums, counts, _ = self._losses(payload)
        gradients = torch.autograd.grad(self._reduce(means, sums, counts, score_spec.reduction),
                                        [p for _, p in selected])
        grouped = {}
        for (name, _), gradient in zip(selected, gradients):
            module = self.module.get_submodule(name.rsplit(".", 1)[0])
            key = module.block_key if score_spec.parameter_view == "block" else (
                f"{module.block_key.rsplit('.', 1)[0]}.{module.part}.{_BANDS[module.block_index // 2]}")
            grouped.setdefault(key, []).append(gradient.detach().flatten())
        keys = tuple(sorted(grouped))
        return {"groups": keys, "gradients": tuple(torch.cat(grouped[key]) for key in keys)}

    def _optimizer_update(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        parameters = [parameter for parameter in self.module.parameters() if parameter.grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.settings.max_grad_norm) if parameters else loss.new_zeros(())
        success = bool(torch.isfinite(loss.detach()) and torch.isfinite(norm))
        if success:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.steps += 1
            self.weight_revision += 1
            self._weights_identity = str(self.steps)
        self.optimizer.zero_grad(set_to_none=True)
        return success

    def _weighted_loss(self, flat, targets):
        weight = targets.weight.detach().to(device=flat.device, dtype=flat.dtype)
        active = weight > 0
        if int(active.sum().item()) == 0:
            return None, 0.0, 0
        terms = weight * flat
        return terms.sum() / active.sum(), float(terms.detach().sum().item()), int(active.sum().item())

    def train(self, k, *, batch_factory, objective, hooks, limits):
        """Run caller-ordered updates; zero active weights skip AdamW entirely."""
        model_boundary(objective.token_mode in {"all", "weighted"} and not objective.kd_weight,
                       "Seq2seq supports supervised or weighted training")
        model_boundary(not hooks or (not hooks.plan_update and not hooks.gradient_subset),
                       "Seq2seq does not implement causal freeze hooks")
        reports, attempts, updates, tokens = [], 0, 0, 0
        with self._execution(mode="training", advance=True):
            for _ in range(k):
                for training in batch_factory():
                    if attempts >= limits.max_attempts or updates >= limits.max_steps or tokens + training.batch.valid_tokens > limits.max_tokens:
                        return reports
                    start = self._begin_timed_block()
                    means, sums, counts, flat = self._losses(self._payload(training.batch))
                    diagnostics = {"plan": self._plan_fingerprint}
                    if objective.token_mode == "weighted":
                        loss, numer, denom = self._weighted_loss(flat, training.token_targets)
                        diagnostics["token_surprise"] = flat.detach().float().tolist()
                    else:
                        model_boundary(training.token_targets is None, "Unweighted training takes no token targets")
                        loss = self._reduce(means, sums, counts, objective.reduction)
                        numer = float(sums.detach().sum())
                        denom = int(counts.detach().sum())
                    success = loss is not None and self._optimizer_update(loss)
                    reason = None if success else ("no_active_tokens" if loss is None else "nonfinite_loss_or_gradients")
                    attempts, updates, tokens = attempts + 1, updates + int(success), tokens + training.batch.valid_tokens
                    reports.append(StepReport(training.batch.ids, success, self.steps, training.batch.valid_tokens,
                                              {"supervised": numer}, {"supervised": denom}, diagnostics,
                                              self._timing_cost(start), reason))
                    del means, sums, counts, flat, loss
        return reports

    def evaluate(self, view, evaluation_spec, progress=None):
        size = evaluation_spec.get("batch_size", self.settings.max_batch_size)
        total, count = 0.0, 0
        predictions = []
        with self._execution():
            start = self._begin_timed_block()
            for offset in range(0, len(view), size):
                records = view[offset:offset + size]
                payload = self._payload(self.collate(records))
                with torch.no_grad():
                    _means, sums, counts, _flat = self._losses(payload)
                total += float(sums.sum().item())
                count += int(counts.sum().item())
                if "generation" in evaluation_spec:
                    generated = self.predict(self.collate_sources(records), evaluation_spec["generation"])
                    if evaluation_spec.get("return_predictions"):
                        predictions.extend({"id": record.sample_id, "input": record.inputs["text"],
                                            "target": record.target["text"], "prediction": text}
                                           for text, record in zip(generated["texts"], records))
                if progress is not None:
                    progress(offset + len(records), len(view))
        result = {"negative_nll": -total / count, "perplexity": math.exp(min(80.0, total / count))}
        result.update(self._timing_cost(start))
        if evaluation_spec.get("return_predictions"):
            result["_predictions"] = predictions
        return result

    def _script_masks(self):
        if self._script_ids is not None:
            return self._script_ids
        special = {self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, self.tokenizer.unk_token_id,
                   getattr(self.config, "decoder_start_token_id", None)}
        devanagari, latin = [], []
        for index in range(len(self.tokenizer)):
            if index in special:
                continue
            text = self.tokenizer.decode([index])
            if any("\u0900" <= character <= "\u097F" for character in text):
                devanagari.append(index)
            if any("A" <= character <= "Z" or "a" <= character <= "z" for character in text):
                latin.append(index)
        self._script_ids = {"devanagari": torch.tensor(devanagari, dtype=torch.long, device=self.device),
                            "latin": torch.tensor(latin, dtype=torch.long, device=self.device)}
        return self._script_ids

    def _guard(self, spec):
        guard = None if spec is None else spec.get("script_guard")
        if not guard:
            return None
        from transformers import LogitsProcessor
        masks = self._script_masks()
        blocked = masks[guard["block"]] if "block" in guard else None
        penalized = masks[guard["penalize"]] if "penalize" in guard else None
        penalty = float(guard.get("penalty", 2.0))

        class _Guard(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if blocked is not None and blocked.numel():
                    scores.index_fill_(1, blocked.to(scores.device), float("-inf"))
                if penalized is not None and penalized.numel():
                    scores[:, penalized.to(scores.device)] -= penalty
                return scores
        return _Guard()

    def predict(self, inputs, generation_spec):
        """Greedy summary. ``token_ids`` keep the first EOS and drop padding."""
        with self._execution(), torch.no_grad():
            start = self._begin_timed_block()
            payload = self._payload(inputs)
            max_new = int(generation_spec.get("max_new_tokens", self.settings.max_target_length))
            guard = self._guard(generation_spec)
            output = self.module.generate(
                input_ids=payload["input_ids"], attention_mask=payload.get("attention_mask"),
                max_new_tokens=max_new, do_sample=False,
                logits_processor=[guard] if guard is not None else None)
            rows = output.tolist()
            start_id = getattr(self.config, "decoder_start_token_id", self.tokenizer.pad_token_id)
            if rows and rows[0] and rows[0][0] == start_id:
                rows = [row[1:] for row in rows]
            token_ids = self._trim_generated(rows)
            texts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            ids = inputs.ids if hasattr(inputs, "ids") else ()
            return {"ids": ids, "texts": texts, "token_ids": token_ids, "cost": self._timing_cost(start)}

    def _trim_generated(self, rows):
        eos, pad = self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        trimmed = []
        for row in rows:
            kept = []
            for token in row:
                if token == pad and token != eos:
                    break
                kept.append(int(token))
                if token == eos:
                    break
            trimmed.append(kept)
        return trimmed

    def effective_updates(self):
        """Per-projection ``(alpha / rank) B A``, plus extra rank, in float32."""
        self.activate()
        return {name: module.delta().detach() for name, module in self.module.named_modules()
                if isinstance(module, Seq2SeqLoRA)}

    def shrink_adapters(self, k):
        """Replace every adapter by the top-k SVD factors of its effective update, at scale 1."""
        self.activate()
        with torch.no_grad():
            for module in self.module.modules():
                if not isinstance(module, Seq2SeqLoRA):
                    continue
                delta = module.delta()
                _U, singular, _Vh = torch.linalg.svd(delta, full_matrices=False)
                keep = min(k, singular.numel())
                U, S, Vh = _U[:, :keep], singular[:keep], _Vh[:keep, :]
                sqrt = S.sqrt()
                module.A = nn.Parameter((sqrt.unsqueeze(1) * Vh).to(dtype=module.A.dtype))
                module.B = nn.Parameter((U * sqrt.unsqueeze(0)).to(dtype=module.B.dtype))
                module.scale = 1.0
                module.clear_extra()
        self.settings = replace(self.settings, rank=k, alpha=float(k))
        self._new_rank = {}
        self._profiles.clear()
        self.steps = 0
        self.weight_revision += 1
        self._weights_identity = "0"
        self.rng_state = self._new_rng()
        self.optimizer = None
        self._reset_optimizer()
        return self.get_weights()

    def merged_copy(self):
        """Inference role whose base weights are ``W + ΔW``. The learner is unchanged."""
        self.activate()
        state = {}
        for name, parameter in self.module.named_parameters():
            if name.endswith((".A", ".B", ".extra_A", ".extra_B")):
                continue
            target = name.replace(".base.", ".")
            state[target] = parameter.detach().clone()
        deltas = {name: value.detach().clone() for name, value in self.effective_updates().items()}
        child = self.fork("serving", "merged")
        child.pretrained = False
        child._merge_state = state
        child._merge_deltas = deltas
        return child

    def _weights(self, *, changed_only=False):
        """Adapter factors when the frozen base can be rebuilt. Full snapshots keep every parameter."""
        eligible = _adapter_parameter_names(self.module)
        return {"parameters": {name: parameter.detach() for name, parameter in self.module.named_parameters()
                               if not changed_only or name in eligible},
                "buffers": {name: buffer.detach() for name, buffer in self.module.named_buffers()}}

    def _install(self, weights, *, changed_only=False):
        """Install exact tensor names and shapes into the existing Parameter objects."""
        live = {"parameters": dict(self.module.named_parameters()), "buffers": dict(self.module.named_buffers())}
        eligible = _adapter_parameter_names(self.module)
        for kind, tensors in live.items():
            incoming = weights.get(kind, {})
            expected = eligible if kind == "parameters" and changed_only else set(tensors)
            model_boundary(set(incoming) == expected and all(value.shape == tensors[name].shape
                           for name, value in incoming.items()), "Checkpoint tensor names or shapes differ")
        with torch.no_grad():
            for kind, tensors in live.items():
                for name, value in weights.get(kind, {}).items():
                    target = tensors[name]
                    target.copy_(value.to(device=target.device, dtype=target.dtype))

    def get_weights(self):
        with self._execution():
            weights = self._weights()
            return {"compatibility": self._compatibility(),
                    **{kind: {name: tensor.clone() for name, tensor in values.items()} for kind, values in weights.items()}}

    def _compatibility(self):
        return f"{self.model_id}:{self.revision}:{self.settings.rank}:{self.settings.alpha}"

    def set_weights(self, weights):
        """Install a snapshot in this wrapper and start a fresh optimizer."""
        self.activate()
        names = set(weights["parameters"])
        adapter = _adapter_parameter_names(self.module)
        if names == set(dict(self.module.named_parameters())):
            self._save_full_weights = True
            changed_only = False
        else:
            model_boundary(names == adapter, "Checkpoint tensor names or shapes differ")
            changed_only = True
        self._install(weights, changed_only=changed_only)
        self._weights_identity = "0"
        self.initialization = "weights"
        self.steps = 0
        self.weight_revision += 1
        self.rng_state = self._new_rng()
        self.optimizer = None
        self._reset_optimizer()

    def _plan_record(self):
        if self._plan is None:
            return None
        return {"trainable": list(self._plan.trainable), "new_rank": dict(self._plan.new_rank),
                "lr_scale": dict(self._plan.lr_scale), "identity": dict(self._plan.identity)}

    def _restore_layout(self, metadata):
        self._new_rank = {key: int(value) for key, value in dict(metadata.get("new_rank") or {}).items()}
        trainable = metadata.get("trainable")
        self._trainable = None if trainable is None else set(trainable)
        self._lr_scale = {key: float(value) for key, value in dict(metadata.get("lr_scale") or {}).items()}
        plan = metadata.get("plan_record")
        self._plan = None if not plan else AdapterPlan(
            tuple(plan["trainable"]), dict(plan.get("new_rank") or {}),
            dict(plan.get("lr_scale") or {}), dict(plan.get("identity") or {}))
        self._plan_fingerprint = metadata.get("plan")
        rank, alpha = int(metadata["rank"]), float(metadata["alpha"])
        if self.settings.rank != rank or self.settings.alpha != alpha:
            self.settings = replace(self.settings, rank=rank, alpha=alpha)
            self._profiles.clear()

    def _metadata(self, kind, record):
        return {"schema": 1, "state_kind": kind, "compatibility": self._compatibility(),
                "execution_identity": self.info.identity, "model": self.model_id, "revision": self.revision,
                "base_bytes": self._base_bytes, "steps": self.steps,
                "weight_revision": self.weight_revision, "initialization": self.initialization,
                "weights_identity": self._weights_identity or str(self.steps),
                "plan": self._plan_fingerprint, "plan_record": self._plan_record(),
                "rank": self.settings.rank, "alpha": self.settings.alpha, "new_rank": self._new_rank,
                "trainable": None if self._trainable is None else sorted(self._trainable),
                "lr_scale": self._lr_scale, "full_weights": self._save_full_weights, "record": record}

    def _training_state(self):
        return {"optimizer": cpu_tree(self.optimizer.state_dict()), "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(), "rng": cpu_tree(self.rng_state), "steps": self.steps,
                "weight_revision": self.weight_revision, "weights_identity": self._weights_identity}

    def save(self, path, state_kind, *, record=None):
        """Save an adapter snapshot. A merged serving copy is not written."""
        path = Path(path)
        if self.role == "serving":
            raise ValueError("A merged copy is built in memory and is not saved")
        if self.module is None and self._offload is not None and state_kind in {"resume", "frozen"}:
            source = Path(self._offload.name)
            saved = read_json(source / "metadata.json")
            if state_kind == "resume" and saved["state_kind"] != "resume":
                self.activate()
            else:
                path.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / "weights.pt", path / "weights.pt")
                if state_kind == "resume":
                    shutil.copyfile(source / "training.pt", path / "training.pt")
                write_json(path / "metadata.json", {**saved, "state_kind": state_kind})
                return
        self.activate()
        path.mkdir(parents=True, exist_ok=True)
        torch.save(cpu_tree(self._weights(changed_only=not self._save_full_weights)), path / "weights.pt")
        if state_kind == "resume":
            torch.save(self._training_state(), path / "training.pt")
        write_json(path / "metadata.json", self._metadata(state_kind, record))

    def save_english_checkpoint(self, path, *, passes, validation, prefix):
        """Base reference plus the English adapter. The directory is written once."""
        self.save(path, "english", record={"prefix": prefix, "passes": passes, "validation": validation,
                                           "writer": "english_checkpoint"})

    def save_hindi_checkpoint(self, path):
        """Final Hindi adapter and the plan that produced it."""
        self.save(path, "hindi")

    def save_shrunk_adapter(self, path, *, k_energy, k_generation, test_table):
        """Inference export at scale 1. This is not an exact-resume checkpoint."""
        self.save(path, "inference", record={"k_energy": k_energy, "k_generation": k_generation,
                                             "test_table": test_table})

    def _frozen_identity(self, metadata):
        self._weights_identity = metadata["weights_identity"]
        self.artifact = metadata.get("record")
        if self.role == "learner":
            self.initialization = "weights:" + self._weights_identity

    def load(self, path, state_kind):
        path = Path(path)
        metadata = read_json(path / "metadata.json")
        self._prepare()
        self._restore_layout(metadata)
        if state_kind == "resume":
            self.initialization = metadata["initialization"]
        else:
            self._frozen_identity(metadata)
        self._base_bytes = metadata["base_bytes"]
        self._save_full_weights = metadata.get("full_weights", False)
        if self.module is not None:
            self._read_state(path)
            return
        target = self._offload_directory()
        names = ["weights.pt", "metadata.json", *(("training.pt",) if state_kind == "resume" else ())]
        for name in names:
            shutil.copyfile(path / name, Path(target.name) / name)
        previous, self._offload = self._offload, target
        if previous is not None:
            previous.cleanup()
        self.steps = metadata["steps"] if state_kind == "resume" else 0
        self.weight_revision = metadata["weight_revision"]

    def _read_state(self, path):
        """Install a saved snapshot. Resume restores the optimizer; every other kind starts fresh."""
        path = Path(path)
        metadata = read_json(path / "metadata.json")
        self._restore_layout(metadata)
        self._save_full_weights = metadata.get("full_weights", False)
        self._apply_mask(self.module)
        self._cut_backward()
        self._install(torch.load(path / "weights.pt", map_location="cpu", weights_only=True),
                      changed_only=not self._save_full_weights)
        self.optimizer = None
        self._reset_optimizer()
        if metadata["state_kind"] == "resume":
            training = torch.load(path / "training.pt", map_location="cpu", weights_only=True)
            self.optimizer.load_state_dict(training["optimizer"])
            self.scheduler.load_state_dict(training["scheduler"])
            self.scaler.load_state_dict(training["scaler"])
            self.rng_state = training["rng"]
            self.steps, self.weight_revision = training["steps"], training["weight_revision"]
            self._weights_identity = training["weights_identity"]
        else:
            self.steps, self.weight_revision = 0, metadata["weight_revision"]
            self._frozen_identity(metadata)
            self.rng_state = self._new_rng()

    def _reserve_storage(self, additional):
        root = Path(self.runtime_root or self.cache_dir / "standalone")
        root.mkdir(parents=True, exist_ok=True)
        if self.resources is not None:
            Checkpoints(root, self.info.identity, self.resources.disk_bytes,
                        external_bytes=self.base_storage_bytes).reserve(additional)

    def _offload_directory(self):
        root = Path(self.runtime_root or self.cache_dir / "standalone")
        root.mkdir(parents=True, exist_ok=True)
        self._reserve_storage(self.info.checkpoint_bytes)
        folder = root / ".offload"
        folder.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=folder, prefix=self.role + "-")

    def offload(self):
        """A merged copy is dropped. Every other role keeps a private snapshot."""
        if self.module is None:
            return
        if self.role == "serving":
            self.release()
            return
        self._clear_cuts()
        target = self._offload_directory()
        self.save(target.name, "resume" if self.optimizer is not None else "frozen")
        previous, self._offload = self._offload, target
        self.module = self.optimizer = self.scheduler = self.scaler = None
        if previous is not None:
            previous.cleanup()
        gc.collect()
        if self.device.type != "cpu":
            getattr(torch, self.device.type).empty_cache()

    def release(self):
        self._clear_cuts()
        self.module = self.optimizer = self.scheduler = self.scaler = None
        if self._offload is not None:
            self._offload.cleanup()
            self._offload = None
        gc.collect()
        if self.device.type != "cpu":
            getattr(torch, self.device.type).empty_cache()
