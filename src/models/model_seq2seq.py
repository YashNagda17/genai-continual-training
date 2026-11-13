from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import copy
import gc
import math
import os
import random
try:
    import resource
except ImportError:
    resource = None
import shutil
import tempfile
import time

# Blocking GPU fence. The driver reads this when the XPU context is created, and otherwise spins one core.
os.environ.setdefault("EventHostSynchronizeLinuxUserFenceKmdWait", "1")

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from torch.utils._pytree import tree_map_only
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig,
                          LogitsProcessorList, SequenceBiasLogitsProcessor, SuppressTokensLogitsProcessor)

from .model_base import ModelBase
from ..utils.contracts import ( AdapterPlan, ModelInfo, ScoreResult, StepReport, model_boundary, )
from ..utils.state import Artifacts, read_json, write_json


@dataclass(frozen=True)
class Seq2SeqSettings:
    """Numerical choices for one encoder-decoder role. Full fine-tuning is rejected."""
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
    max_batch_size: int = 32
    logit_chunk_size: int = 32  # Target positions per cross-entropy call over the full vocabulary
    max_score_bytes: int = 2 * 1024 ** 3
    working_bytes: int = 1024 ** 3


def _band(block_index):
    """Map zero-based blocks 0 and 1 to the band "1-2"."""
    first = block_index // 2 * 2 + 1
    return f"{first}-{first + 1}"


class _ChunkedPositionNLL(torch.autograd.Function):
    """First-order CE with one logits-gradient buffer instead of one per slice."""

    @staticmethod
    def forward(ctx, logits, labels, chunk_size):
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        losses = torch.empty_like(flat_labels, dtype=torch.float32)
        for start in range(0, flat_labels.numel(), chunk_size):
            stop = start + chunk_size
            losses[start:stop] = F.cross_entropy(
                flat_logits[start:stop].float(), flat_labels[start:stop],
                ignore_index=-100, reduction="none")
        ctx.save_for_backward(flat_logits, flat_labels)
        ctx.chunk_size, ctx.logits_shape = chunk_size, logits.shape
        return losses.view_as(labels)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        upstream = grad_output.reshape(-1)
        gradient = torch.empty_like(logits)
        for start in range(0, labels.numel(), ctx.chunk_size):
            stop = start + ctx.chunk_size
            targets = labels[start:stop]
            valid = targets != -100
            probabilities = F.softmax(logits[start:stop].float(), dim=-1)
            probabilities.scatter_add_(1, targets.masked_fill(~valid, 0).unsqueeze(1),
                                       -valid.to(probabilities.dtype).unsqueeze(1))
            probabilities.mul_((upstream[start:stop].float() * valid).unsqueeze(1))
            gradient[start:stop].copy_(probabilities)
        return gradient.view(ctx.logits_shape), None, None


def cpu_tree(value):
    return tree_map_only(torch.Tensor, lambda item: item.detach().cpu(), value)


def _adapter_parameter_names(module):
    return {name for name, _parameter in module.named_parameters()
            if name.endswith((".A", ".B", ".extra_A", ".extra_B"))}


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
        result = self.base(inputs)
        shape = result.shape
        hidden = F.linear(inputs, self.A).reshape(-1, self.A.shape[0])
        result = torch.addmm(result.reshape(-1, shape[-1]), hidden, self.B.T, alpha=self.scale)
        if getattr(self, "extra_scale", None):
            hidden = F.linear(inputs, self.extra_A).reshape(-1, self.extra_A.shape[0])
            result = torch.addmm(result, hidden, self.extra_B.T, alpha=self.extra_scale)
        return result.reshape(shape)

    def delta(self):
        """Effective update in float32, including the LoRA scale and any extra rank."""
        value = (self.B.float() @ self.A.float()) * self.scale
        if getattr(self, "extra_scale", None):
            value = value + (self.extra_B.float() @ self.extra_A.float()) * self.extra_scale
        return value

    def delta_norm(self):
        """Frobenius norm from rank-sized Gram matrices; preserve extra-rank cross terms."""
        if getattr(self, "extra_scale", None):
            return torch.linalg.vector_norm(self.delta())
        A, B = self.A.float(), self.B.float()
        squared = ((B.T @ B) * (A @ A.T).T).sum()
        return squared.clamp_min(0).sqrt() * abs(self.scale)


def _detach_value(_module, _inputs, value):
    """Stop backward through a frozen block while preserving its structured output."""
    return tree_map_only(torch.Tensor, lambda item: item.detach(), value)


class Seq2SeqLMModel(ModelBase):

    model_id = ""
    stack_paths = {"encoder": "model.encoder", "decoder": "model.decoder"}
    blocks_attribute = "layers"

    def __init__(self, settings=None, *, device="auto", revision="main", cache_dir=None,
                 collate=None, collate_sources=None, role="learner", initialization="base",
                 config=None, tokenizer=None, model_factory=None, pretrained=True):
        super().__init__(device)
        self.settings = settings or Seq2SeqSettings()
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
        self._merge_state = None
        self._script_ids = None
        self.artifact = None

    def _prepare(self):
        if self._prepared:
            return
        source = {"revision": self.revision, "cache_dir": self.cache_dir}
        if self.config is None:
            self.config = AutoConfig.from_pretrained(self.model_id, **source)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **source)
        self.model_factory = self.model_factory or (
            lambda config: AutoModelForSeq2SeqLM.from_config(config, attn_implementation="eager"))
        self._prepared = True

    @property
    def tokenizer(self):
        self._prepare()
        return self._tokenizer

    def preprocessing_kwargs(self, *, max_source=None, max_target=None):
        tok = self.tokenizer
        return {"encode": self.encode,
                "eos_id": tok.eos_token_id,
                "pad_id": tok.pad_token_id if tok.pad_token_id is not None else 0,
                "max_source": max_source or self.settings.max_source_length,
                "max_target": max_target or self.settings.max_target_length,
                "identity": {"model": self.model_id, "revision": self.revision,
                             "vocabulary": self.model_id}}

    def bind_dataset(self, dataset):
        super().bind_dataset(dataset)
        self.collate_sources = getattr(dataset, "collate_sources", self.collate_sources)

    def _attach_lora(self, module, *, deterministic):
        raise NotImplementedError('The mBART adapter supplies projection names')

    def _mark_adapters(self, module):
        """Train adapter factors only. The base weights, embeddings, and output head stay frozen."""
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(name.endswith((".A", ".B", ".extra_A", ".extra_B")))

    def _adapters(self, module=None):
        return ((name, layer) for name, layer in (self.module if module is None else module).named_modules()
                if isinstance(layer, Seq2SeqLoRA))

    def _apply_mask(self, module):
        """Freeze shared adapters outside the installed plan. Extra rank stays trainable."""
        if self._trainable is None:
            return
        for _, child in self._adapters(module):
            shared = child.block_key in self._trainable
            child.A.requires_grad_(shared)
            child.B.requires_grad_(shared)

    def _add_new_rank(self, module, new_rank):
        for name, child in self._adapters(module):
            if child.block_key in new_rank:
                child.add_extra(int(new_rank[child.block_key]), _seed_for(self.settings.seed, "extra:" + name))

    def _build(self, config):
        return self.model_factory(copy.deepcopy(config))

    def _profile(self):
        self._prepare()
        key = (self.role != "serving", self.settings.rank, self.settings.alpha,
               tuple(sorted(self._new_rank.items())),
               None if self._trainable is None else tuple(sorted(self._trainable)))
        if key not in self._profiles:
            with torch.device("meta"):
                module = self._build(self.config)
                if self.role != "serving":
                    self._attach_lora(module, deterministic=self.role == "probe")
                    self._add_new_rank(module, self._new_rank)
                    self._apply_mask(module)
            adapter_names = _adapter_parameter_names(module)
            adapters = sum(p.numel() for name, p in module.named_parameters() if name in adapter_names)
            self._profiles[key] = (module.num_parameters(), module.num_parameters(only_trainable=True), adapters)
        return self._profiles[key]

    def describe_role(self, role, initialization):
        saved_role, self.role = self.role, role
        try:
            total, trainable, adapters = self._profile()
        finally:
            self.role = saved_role
        identity = f"{self.model_id}:{role}:{initialization}"
        weight_bytes = total * 4
        saved = adapters if role != "serving" and not self._save_full_weights else total
        peak = weight_bytes + trainable * 12 + self.settings.working_bytes + self.settings.max_score_bytes
        capabilities = frozenset({"example_loss", "token_loss", "generation", "probe_gradient", "token_weight_hook",
                                  "adapter_plans", "adapter_updates", "adapter_shrink", "zero_probe"})
        blocks = self.config.num_layers + self.config.num_decoder_layers
        return ModelInfo(identity, total, trainable, capabilities=capabilities,
                         cpu_bytes=peak, accelerator_bytes=peak if self.device.type != "cpu" else 0,
                         checkpoint_bytes=saved * 4 + trainable * 12 + 1024 ** 2,
                         vocabulary=self.model_id, loss_fingerprint="seq2seq-summary-ce-v1",
                         parameter_views=("cell", "block"),
                         layer_names=tuple(f"block.{index}" for index in range(blocks)),
                         forward_cost=float(total), backward_costs=tuple(2.0 for _ in range(blocks)),
                         legal_prefixes=(0,))

    @property
    def info(self):
        return self.describe_role(self.role, self.initialization)

    def rank_costs(self):
        """Adapter parameters per encoder/decoder block rank, measured on meta without loading weights."""
        self._prepare()
        # Building a module can draw from the global random generator; keep that state unchanged,
        # so measuring sizes never changes how a later randomly initialized model starts.
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            module = self._build(self.config)
            self._attach_lora(module, deterministic=True)
        costs = {}
        for _, child in self._adapters(module):
            if child.block_index == 0:
                stack = child.block_key.split(".")[0]
                costs[stack] = costs.get(stack, 0) + child.A.shape[1] + child.B.shape[0]
        return costs["encoder"], costs["decoder"]

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
        if not self.pretrained or self._base_path is not None:
            return
        self._base_path = Path(snapshot_download(
            self.model_id, revision=self.revision, cache_dir=self.cache_dir,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "pytorch_model.bin"]))
        self._base_bytes = sum(path.stat().st_size for path in self._base_path.rglob("*") if path.is_file())

    @property
    def base_storage_bytes(self):
        return self._base_bytes

    def _optimizer_parameters(self):
        by_block, new_params = {}, []
        for _, adapter in self._adapters():
            for name, parameter in adapter.named_parameters(recurse=False):
                if parameter.requires_grad:
                    target = new_params if name.startswith("extra_") else by_block.setdefault(adapter.block_key, [])
                    target.append(parameter)
        groups = [{"params": by_block[key], "lr": self.settings.learning_rate * float(self._lr_scale.get(key, 1.0))}
                  for key in sorted(by_block)]
        if new_params:
            groups.append({"params": new_params, "lr": self.settings.learning_rate})
        return groups

    def _reset_optimizer(self):
        groups = self._optimizer_parameters()
        self.module.zero_grad(set_to_none=True)
        # Fused AdamW on CUDA/XPU: fewer kernel launches from the host per step.
        self.optimizer = torch.optim.AdamW(groups, lr=self.settings.learning_rate,
                                           weight_decay=self.settings.weight_decay,
                                           fused=self.device.type in {"cuda", "xpu"}) if groups else None
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
                self._apply_mask(module)
            elif self._merge_state is not None:
                module.load_state_dict(self._merge_state, strict=False, assign=False)
                module.requires_grad_(False)
        self.module = module.to(device=self.device, dtype=torch.float32)
        if self.device.type != "cpu":
            getattr(torch, self.device.type).empty_cache()
        self.module.config.use_cache = False
        if self.role != "serving":
            self.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self.role != "serving" and self._trainable is not None:
            self._cut_backward()
        self._reset_optimizer()
        self.rng_state = self.rng_state or self._new_rng()
        if self._offload is not None:
            self._read_state(Path(self._offload.name))

    def parameter_counts(self):
        return self.module.num_parameters(), self.module.num_parameters(only_trainable=True)

    def _clear_cuts(self):
        for handle in self._cut_hooks:
            handle.remove()
        self._cut_hooks = []

    def _lowest_trainable(self, stack):
        indexes = [module.block_index for _, module in self._adapters()
                   if module.block_key.startswith(stack + ".")
                   and any(parameter.requires_grad for parameter in module.parameters())]
        return min(indexes) if indexes else None

    def _cut_backward(self):
        """Detach the block below the lowest trainable one so its backward is not run."""
        self._clear_cuts()
        for stack in ("encoder", "decoder"):
            lowest = self._lowest_trainable(stack)
            if lowest == 0:
                continue
            path = self.stack_paths[stack]
            if lowest is not None:
                path += f".{self.blocks_attribute}.{lowest - 1}"
            self._cut_hooks.append(self.module.get_submodule(path).register_forward_hook(_detach_value))

    def apply_plan(self, plan):
        """Install adapter trainability before the first update and build a fresh optimizer."""
        self.activate()
        trainable, new_rank, scales = set(plan.trainable), dict(plan.new_rank), dict(plan.lr_scale)
        for _, module in self._adapters():
            module.clear_extra()
        self._new_rank = new_rank
        self._trainable = set(trainable)
        self._lr_scale = scales
        self._plan = plan
        self._add_new_rank(self.module, new_rank)
        self._apply_mask(self.module)
        self._profiles.clear()
        self._cut_backward()
        self._reset_optimizer()
        trainable_count = self.module.num_parameters(only_trainable=True)
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
        seed = _seed_for(self.settings.seed, self.initialization) if seed is None else seed
        states = {"cpu": torch.Generator(device="cpu").manual_seed(seed).get_state()}
        if self.device.type != "cpu":
            states["device"] = torch.Generator(device=self.device).manual_seed(seed).get_state().cpu()
        return states

    def _synchronize(self):
        if self.device.type != "cpu":
            getattr(torch, self.device.type).synchronize(self.device)

    def release_evaluation_memory(self):
        if self.device.type != "cpu":
            gc.collect()
            getattr(torch, self.device.type).empty_cache()

    def _begin_timed_block(self):
        self._synchronize()
        self._reset_peak_memory()
        return time.perf_counter()

    def _timing_cost(self, start, *, synchronize=True, **fields):
        if synchronize:
            self._synchronize()
        cost = {"device": str(self.device),
                "synchronized_wall_seconds" if synchronize else "wall_seconds": time.perf_counter() - start,
                "process_peak_cpu_bytes": None if resource is None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                **fields}
        if not synchronize:
            cost["synchronized"] = False
        backend = getattr(torch, self.device.type, None)
        if self.device.type != "cpu" and hasattr(backend, "max_memory_allocated"):
            cost["peak_device_allocated_bytes"] = backend.max_memory_allocated(self.device)
        return cost

    @contextmanager
    def _execution(self, *, mode="inference", advance=False):
        active = self.module is not None
        self.activate()
        caller = self._capture_rng()
        modes = [(item, item.training) for item in self.module.modules()]
        flags = [(parameter, parameter.requires_grad) for parameter in self.module.parameters()]
        buffers = [] if advance else [(buffer, buffer.detach().clone()) for buffer in self.module.buffers()]
        self._restore_rng(self.rng_state)
        training = mode == "training"
        if any(saved != training for _, saved in modes):
            self.module.train(training)
        try:
            yield mode
        finally:
            if advance:
                self.rng_state = self._capture_rng()
            self._restore_rng(caller)
            for module, saved in modes:
                if module.training != saved:
                    module.training = saved
            for parameter, flag in flags:
                if parameter.requires_grad != flag:
                    parameter.requires_grad_(flag)
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
            if not active:
                self.offload()

    def _payload(self, batch):
        payload = batch.payload if hasattr(batch, "payload") else batch
        return self.to_device(payload)

    def _sdpa_masks(self):
        return getattr(self.module.config, "_attn_implementation", None) == "sdpa"

    def _source_mask(self, attention_mask):
        """Prebuild SDPA's 4D source mask to avoid the library's host scalar read."""
        if attention_mask is None or not self._sdpa_masks():
            return attention_mask
        return attention_mask.bool()[:, None, None, :]

    def _causal_mask(self, length, device):
        """Decoder self-attention mask for SDPA. Passing none makes the library build all-ones and read it back."""
        if not self._sdpa_masks():
            return None
        return torch.ones((length, length), dtype=torch.bool, device=device).tril_()[None, None]

    def _forward(self, payload):
        """Teacher-forced logits without allocating the library's second vocabulary loss."""
        decoder_input_ids = self.module.prepare_decoder_input_ids_from_labels(payload["labels"])
        causal = self._causal_mask(decoder_input_ids.shape[1], decoder_input_ids.device)
        return self.module(input_ids=payload["input_ids"], attention_mask=self._source_mask(payload.get("attention_mask")),
                           decoder_input_ids=decoder_input_ids, decoder_attention_mask=causal,
                           use_cache=False, return_dict=True)

    def _position_nll(self, logits, labels):
        """Bound per-position CE scratch to 256 MiB, with the configured chunk as a floor."""
        if self.settings.logit_chunk_size < 1:
            raise ValueError("logit_chunk_size must be positive")
        vocab = logits.shape[-1]
        rows = labels.numel()
        scratch_rows = max(1, (256 * 1024 ** 2) // max(vocab * logits.element_size(), 1))
        chunk = max(self.settings.logit_chunk_size, min(rows or 1, scratch_rows))
        return _ChunkedPositionNLL.apply(logits, labels, chunk)

    def _losses(self, payload):
        """Per-example mean NLL from the model's own logits. Padding labels are ignored."""
        labels = payload["labels"]
        logits = self._forward(payload).logits
        position = self._position_nll(logits, labels)
        valid = labels != -100
        counts = valid.sum(-1).clamp_min(1)
        sums = position.sum(-1)
        return sums / counts, sums, counts, position

    def _reduce(self, means, sums, counts, reduction):
        return means.mean() if reduction == "per_example_mean" else sums.sum() / counts.sum()

    def score(self, batch, score_spec):
        with self._execution():
            payload = self._payload(batch)
            start = self._begin_timed_block()
            if score_spec.kind in {"example_loss", "loss", "token_loss"}:
                with torch.no_grad():
                    means, _sums, _counts, position = self._losses(payload)
                    if score_spec.kind == "token_loss":
                        values = position.detach().reshape(-1).index_select(0, payload["supervised_indices"]).float()
                    else:
                        values = means.detach().float()
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
                f"{module.block_key.rsplit('.', 1)[0]}.{module.part}.{_band(module.block_index)}")
            grouped.setdefault(key, []).append(gradient.detach().flatten())
        keys = tuple(sorted(grouped))
        return {"groups": keys, "gradients": tuple(torch.cat(grouped[key]) for key in keys)}

    def _optimizer_update(self, loss):
        """Backward, clipping and AdamW in fp32 without host finite checks or gradient scaling."""
        loss.backward()
        parameters = [parameter for group in self.optimizer.param_groups for parameter in group['params']
                      if parameter.grad is not None]
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, self.settings.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.steps += 1
        self.weight_revision += 1
        self._weights_identity = str(self.steps)
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def _weighted_loss(self, flat, targets, counts=None):
        weight = targets.weight.detach().to(device=flat.device, dtype=flat.dtype)
        model_boundary(weight.shape == flat.shape, "Token weights must align with supervised tokens")
        active = None if targets.all_positive else weight > 0
        if flat.numel() == 0 or (not targets.all_positive and int(active.sum().item()) == 0):
            return None, 0.0, 0
        terms = weight * flat
        if counts is not None:
            owners = torch.repeat_interleave(torch.arange(len(counts), device=flat.device), counts,
                                            output_size=flat.numel())
            sums = flat.new_zeros(len(counts)).scatter_add(0, owners, terms)
            if targets.all_positive:
                means = sums / counts
                return means.mean(), means.detach().sum(), len(means)
            row_counts = flat.new_zeros(len(counts)).scatter_add(0, owners, active.to(flat.dtype))
            included = row_counts > 0
            means = torch.where(included, sums / row_counts.clamp_min(1), 0.0)
            denominator = included.sum()
            return means.sum() / denominator, means.detach().sum(), denominator
        denominator = flat.numel() if targets.all_positive else active.sum()
        return terms.sum() / denominator, terms.detach().sum(), denominator

    def _reset_peak_memory(self):
        backend = getattr(torch, self.device.type, None)
        if self.device.type != "cpu" and hasattr(backend, "reset_peak_memory_stats"):
            backend.reset_peak_memory_stats(self.device)

    def train(self, k, *, batch_factory, objective, hooks, limits):
        """Run caller-ordered updates; zero active weights skip AdamW entirely."""
        model_boundary(objective.token_mode in {"all", "weighted"}, "Seq2seq supports supervised or weighted training")
        reports, attempts, updates, tokens = [], 0, 0, 0
        with self._execution(mode="training", advance=True):
            for _ in range(k):
                for training in batch_factory():
                    if attempts >= limits.max_attempts or updates >= limits.max_steps or tokens + training.batch.valid_tokens > limits.max_tokens:
                        return reports
                    self._reset_peak_memory()
                    start = time.perf_counter()
                    payload = self._payload(training.batch)
                    means, sums, counts, position = self._losses(payload)
                    diagnostics = {"plan": self._plan_fingerprint}
                    if objective.token_mode == "weighted":
                        # Backward needs the loss graph; only the policy hook sees detached NLL.
                        flat = position.reshape(-1).index_select(0, payload["supervised_indices"])
                        targets = (hooks.token_targets(training.batch, flat.detach())
                                   if hooks and hooks.token_targets else training.token_targets)
                        model_boundary(targets is not None, "Weighted training requires token targets")
                        row_counts = counts if hooks and hooks.token_targets and objective.reduction == 'per_example_mean' else None
                        loss, numer, denom = self._weighted_loss(flat, targets, row_counts)
                        diagnostics["token_surprise"] = flat.detach()  # Device tensor; not saved
                    else:
                        model_boundary(training.token_targets is None, "Unweighted training takes no token targets")
                        loss = self._reduce(means, sums, counts, objective.reduction)
                        numer = denom = None
                    success = loss is not None and self._optimizer_update(loss)
                    if numer is None:
                        numer = sums.detach().sum()
                        denom = training.batch.supervised_tokens
                    reason = None if success else ("no_active_tokens" if loss is None else "nonfinite_loss_or_gradients")
                    attempts, updates, tokens = attempts + 1, updates + int(success), tokens + training.batch.valid_tokens
                    reports.append(StepReport(training.batch.ids, success, self.steps, training.batch.valid_tokens,
                                              {"supervised": numer}, {"supervised": denom}, diagnostics,
                                              self._timing_cost(start, synchronize=False), reason))
                    del means, sums, counts, position, loss
        return reports

    def evaluate(self, view, evaluation_spec, progress=None):
        try:
            return self._evaluate_view(view, evaluation_spec, progress)
        finally:
            self.release_evaluation_memory()

    def _evaluate_view(self, view, evaluation_spec, progress=None):
        size = evaluation_spec.get("batch_size", self.settings.max_batch_size)
        predictions = []
        total_sum = torch.zeros((), device=self.device)
        total_count = torch.zeros((), device=self.device)
        with self._execution():
            start = self._begin_timed_block()
            for offset in range(0, len(view), size):
                records = view[offset:offset + size]
                payload = self._payload(self.collate(records))
                with torch.no_grad():
                    _means, sums, counts, _position = self._losses(payload)
                total_sum = total_sum + sums.sum()
                total_count = total_count + counts.sum()
                del payload, _means, sums, counts, _position
                if "generation" in evaluation_spec:
                    generated = self.predict(self.collate_sources(records), evaluation_spec["generation"])
                    if evaluation_spec.get("return_predictions"):
                        predictions.extend({"id": record.sample_id, "input": record.inputs["text"],
                                            "target": record.target["text"], "prediction": text}
                                           for text, record in zip(generated["texts"], records))
                if progress is not None:
                    progress(offset + len(records), len(view))
        total, count = torch.stack((total_sum, total_count)).detach().cpu().tolist()
        count = int(count)
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
        masks = self._script_masks()
        processors = LogitsProcessorList()
        if "block" in guard and masks[guard["block"]].numel():
            processors.append(SuppressTokensLogitsProcessor(masks[guard["block"]].tolist(), device=self.device))
        if "penalize" in guard and masks[guard["penalize"]].numel():
            processors.append(SequenceBiasLogitsProcessor(
                [[[token], -float(guard.get("penalty", 2.0))] for token in masks[guard["penalize"]].tolist()]))
        return processors

    def _greedy_ids(self, input_ids, attention_mask, max_new, guard, no_repeat=0, *, stop_on_eos=True):
        """Use explicit greedy defaults instead of inheriting checkpoint decoding policies."""
        start_id = int(getattr(self.config, "decoder_start_token_id", self.tokenizer.pad_token_id))
        if max_new <= 0:
            return input_ids.new_full((input_ids.shape[0], 1), start_id)
        pad_id = self.tokenizer.pad_token_id
        config = GenerationConfig(max_new_tokens=max_new, do_sample=False, num_beams=1, use_cache=True,
            decoder_start_token_id=start_id, eos_token_id=int(self.tokenizer.eos_token_id) if stop_on_eos else None,
            pad_token_id=int(self.tokenizer.eos_token_id) if pad_id is None else int(pad_id),
            no_repeat_ngram_size=max(0, no_repeat))
        previous, self.module.generation_config = self.module.generation_config, config
        try:
            return self.module.generate(input_ids=input_ids, attention_mask=attention_mask,
                                        generation_config=config, logits_processor=guard)
        finally:
            self.module.generation_config = previous

    def predict(self, inputs, generation_spec):
        """Greedy summary. ``token_ids`` keep the first EOS and drop padding."""
        with self._execution(), torch.inference_mode():
            start = self._begin_timed_block()
            payload = self._payload(inputs)
            max_new = int(generation_spec.get("max_new_tokens", self.settings.max_target_length))
            guard = self._guard(generation_spec)
            output = self._greedy_ids(
                payload["input_ids"], payload.get("attention_mask"), max_new, guard,
                int(generation_spec.get("no_repeat_ngram_size", 0)),
                stop_on_eos=bool(generation_spec.get("stop_on_eos", True)))
            token_ids = self._trim_generated(output[:, 1:].tolist())
            texts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            ids = inputs.ids if hasattr(inputs, "ids") else ()
            return {"ids": ids, "texts": texts, "token_ids": token_ids,
                    "cost": self._timing_cost(start, decoder_steps=output.shape[1] - 1)}

    def _trim_generated(self, rows):
        eos, pad = self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        trimmed = []
        for row in rows:
            stop = next((i + int(token == eos) for i, token in enumerate(row) if token in (eos, pad)), len(row))
            trimmed.append([int(token) for token in row[:stop]])
        return trimmed

    def effective_updates(self):
        """Per-projection ``(alpha / rank) B A``, plus extra rank, in float32."""
        self.activate()
        return {name: module.delta().detach() for name, module in self._adapters()}

    def effective_update_norms(self):
        """Per-projection norms for location maps without full shared-update matrices."""
        self.activate()
        with torch.no_grad():
            return {name: module.delta_norm() for name, module in self._adapters()}

    def reset_probe(self):
        """Freeze the loaded checkpoint's function beneath a fresh zero-B probe."""
        model_boundary(self.role == "probe", "Reset only an isolated probe role")
        self.activate()
        self._clear_cuts()
        with torch.no_grad():
            for name, child in self._adapters():
                child.base.weight.add_(child.delta().to(child.base.weight.dtype))
                child.clear_extra()
                _init_A(child.A, _seed_for(self.settings.seed, name))
                child.B.zero_()
        self._new_rank, self._lr_scale, self._trainable = {}, {}, None
        self._plan = self._plan_fingerprint = None
        self._save_full_weights = True
        self._mark_adapters(self.module)
        self._profiles.clear()
        self._reset_optimizer()
        self.steps = 0
        self.rng_state = self._new_rng()

    def shrink_adapters(self, k):
        """Replace every adapter by the top-k SVD factors of its effective update, at scale 1."""
        self.activate()
        with torch.no_grad():
            for _, module in self._adapters():
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
        self._apply_mask(self.module)
        self.steps = 0
        self.weight_revision += 1
        self._weights_identity = "0"
        self.rng_state = self._new_rng()
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
            state[target] = parameter.detach().to(device="cpu", copy=True)
        for name, buffer in self.module.named_buffers():
            state[name] = buffer.detach().to(device="cpu", copy=True)
        with torch.no_grad():
            for name, module in self._adapters():
                state[name + ".weight"].add_(module.delta().detach().cpu())
        child = self.fork("serving", "merged")
        child.pretrained = False
        child._merge_state = state
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
        state = {name: value for kind in live for name, value in weights.get(kind, {}).items()}
        self.module.load_state_dict(state, strict=False, assign=False)

    def get_weights(self):
        with self._execution():
            weights = self._weights()
            return {"compatibility": self._compatibility(),
                    **{kind: {name: tensor.clone() for name, tensor in values.items()} for kind, values in weights.items()}}

    def _compatibility(self):
        return f"{self.model_id}:{self.revision}:{self.settings.rank}:{self.settings.alpha}"

    def set_weights(self, weights):
        self.activate()
        full = set(weights["parameters"]) == set(dict(self.module.named_parameters()))
        self._save_full_weights = self._save_full_weights or full
        self._install(weights, changed_only=not full)
        self._weights_identity = "0"
        self.initialization = "weights"
        self.steps = 0
        self.weight_revision += 1
        self.rng_state = self._new_rng()
        self._reset_optimizer()

    def _plan_record(self):
        return None if self._plan is None else {**asdict(self._plan), "trainable": list(self._plan.trainable)}

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
        if self.module is None and self._offload is not None and state_kind in {"offload", "frozen"}:
            source = Path(self._offload.name)
            saved = read_json(source / "metadata.json")
            if state_kind == "offload" and saved["state_kind"] != "offload":
                self.activate()
            else:
                path.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / "weights.pt", path / "weights.pt")
                if state_kind == "offload":
                    shutil.copyfile(source / "training.pt", path / "training.pt")
                write_json(path / "metadata.json", {**saved, "state_kind": state_kind})
                return
        self.activate()
        path.mkdir(parents=True, exist_ok=True)
        torch.save(cpu_tree(self._weights(changed_only=not self._save_full_weights)), path / "weights.pt")
        if state_kind == "offload":
            torch.save(self._training_state(), path / "training.pt")
        write_json(path / "metadata.json", self._metadata(state_kind, record))

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
        if state_kind == "offload":
            self.initialization = metadata["initialization"]
        else:
            self._frozen_identity(metadata)
        self._base_bytes = metadata["base_bytes"]
        self._save_full_weights = metadata.get("full_weights", False)
        if self.module is not None:
            self._read_state(path)
            return
        target = self._offload_directory()
        names = ["weights.pt", "metadata.json", *(("training.pt",) if state_kind == "offload" else ())]
        for name in names:
            shutil.copyfile(path / name, Path(target.name) / name)
        previous, self._offload = self._offload, target
        if previous is not None:
            previous.cleanup()
        self.steps = metadata["steps"] if state_kind == "offload" else 0
        self.weight_revision = metadata["weight_revision"]

    def _read_state(self, path):
        """Install a saved snapshot. Private offload restores live optimizer state; public exports start fresh."""
        path = Path(path)
        metadata = read_json(path / "metadata.json")
        self._restore_layout(metadata)
        self._save_full_weights = metadata.get("full_weights", False)
        self._apply_mask(self.module)
        self._cut_backward()
        self._install(torch.load(path / "weights.pt", map_location="cpu", weights_only=True),
                      changed_only=not self._save_full_weights)
        self._reset_optimizer()
        if metadata["state_kind"] == "offload":
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

    def _offload_directory(self):
        root = Path(self.runtime_root or self.cache_dir / "standalone")
        root.mkdir(parents=True, exist_ok=True)
        if self.resources is not None:
            Artifacts(root, self.info.identity, self.resources.disk_bytes,
                        external_bytes=self.base_storage_bytes).reserve(self.info.checkpoint_bytes)
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
        self.save(target.name, "offload" if self.optimizer is not None else "frozen")
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
        self._script_ids = None
        if self._offload is not None:
            self._offload.cleanup()
            self._offload = None
        gc.collect()
        if self.device.type != "cpu":
            getattr(torch, self.device.type).empty_cache()