"""Shared encoder-decoder execution for mT5. Transformers supplies the architecture.

Native Torch owns LoRA, losses, gradients, optimization, and generation. Constructors
do not download assets or allocate parameters. Load in bf16 where the device supports
it, otherwise fp32. Never fp16: T5-family weights overflow in fp16.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import copy
import gc
import hashlib
import math
import shutil
import tempfile
import time

import torch
from torch import nn
from torch.nn import functional as F

from .model_base import ModelBase
from ..utils.contracts import (
    AdapterPlan, ModelInfo, ScoreResult, StepReport, fingerprint, model_boundary,
)
from ..utils.numerics import resolve_device
from ..utils.state import Checkpoints, base_assets, read_json, tensor_fingerprint, write_json


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


def _count(module, *, trainable_only=False):
    """Count each Parameter object once so tied embeddings are not added twice."""
    seen, total = set(), 0
    for parameter in module.parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += parameter.numel()
    return total


def _seed_for(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()
    return int(digest[:8], 16)


def _init_A(parameter, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    nn.init.kaiming_uniform_(parameter, a=math.sqrt(5), generator=generator)


class Seq2SeqLoRA(nn.Module):
    """Frozen projection plus (alpha / rank) B A, with B starting at zero.

    ``extra_*`` is a second zero-start pair. Its scale is ``alpha / rank`` with
    ``alpha = 2 * rank``, so the pair starts at scale 2 and B = 0.
    """

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
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.base.requires_grad_(False)

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


def _detach_block_output(module, inputs, output):
    """Stop the backward pass at this block's output. Values are unchanged."""
    return tuple(item.detach() if torch.is_tensor(item) else item for item in output)


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
        self._base_path, self._base_files, self._base_bytes = None, None, 0
        self.rng_state = None
        self.resources = None
        self._new_rank = {}
        self._cut_hooks = []
        self._plan_fingerprint = None
        self._merge_deltas = None
        self._script_ids = None

    def _uses_lora(self):
        return self.role != "serving"

    def _load_dtype(self):
        """bf16 on devices that support it. fp32 elsewhere. Never fp16."""
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        backend = getattr(torch, self.device.type, None)
        supported = getattr(backend, "is_bf16_supported", None)
        if self.device.type != "cpu" and callable(supported) and supported():
            return torch.bfloat16
        return torch.float32

    def _optimizer_dtype(self):
        return torch.float32

    def _prepare(self):
        if self._prepared:
            return
        if self.config is None or self._tokenizer is None or self.model_factory is None:
            import transformers
            from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
            if self.config is None:
                self.config = AutoConfig.from_pretrained(
                    self.model_id, revision=self.revision, cache_dir=self.cache_dir)
            self._pin_revision()
            if self._tokenizer is None:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_id, revision=self.revision, cache_dir=self.cache_dir)
                except ImportError as error:
                    raise ImportError(
                        "mT5's SentencePiece tokenizer requires the sentencepiece package"
                    ) from error
            self._library_version = transformers.__version__
            self.model_factory = self.model_factory or (
                lambda config: AutoModelForSeq2SeqLM.from_config(config, attn_implementation="eager"))
        else:
            self._library_version = getattr(self, "_library_version", "injected")
        model_boundary(not self.pretrained or self.revision not in ("main", "master"),
                       "An injected pretrained config needs a pinned model revision")
        if self.model_type:
            model_boundary(self.config.model_type == self.model_type, "Model architecture does not match class")
        self._config_data = self.config.to_dict()
        self._vocabulary = fingerprint({"vocabulary": self._tokenizer.get_vocab(),
                                        "special": self._tokenizer.special_tokens_map})
        self._prepared = True

    def _pin_revision(self):
        import re
        if self.revision and re.fullmatch(r"[0-9a-f]{40}", self.revision):
            return
        pinned = getattr(self.config, "_commit_hash", None)
        if isinstance(pinned, str) and re.fullmatch(r"[0-9a-f]{40}", pinned):
            self.revision = pinned
            return
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RevisionResolutionError
        try:
            resolved = HfApi().resolve_revision(
                self.model_id, revision=self.revision, cache_dir=self.cache_dir, local_files_only=True)
        except RevisionResolutionError:
            resolved = HfApi().resolve_revision(
                self.model_id, revision=self.revision, cache_dir=self.cache_dir, local_files_only=False)
        self.revision = resolved.resolved

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
                             "vocabulary": self._vocabulary, "transformers": self._library_version}}

    def bind_dataset(self, dataset):
        super().bind_dataset(dataset)
        self.collate_sources = getattr(dataset, "collate_sources", self.collate_sources)

    def _untie_lm_head(self, module):
        """Keep a separate output head when the Hub checkpoint says it is not tied.

        ``MT5Config`` forces ``tie_word_embeddings = True`` even if the file says false.
        The google/mt5-small checkpoint stores ``lm_head.weight`` on its own, and the
        parameter count is the untied 300,176,768.
        """
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
        key = (self._uses_lora(), self.settings.rank, self.settings.alpha,
               tuple(sorted(self._new_rank.items())), self.untie_head)
        if key not in self._profiles:
            with torch.device("meta"):
                module = self._build(self.config)
                base = _count(module)
                if self._uses_lora():
                    self._attach_lora(module, deterministic=self.role == "probe")
                    self._add_new_rank(module, self._new_rank)
            if self.expected_parameters is not None and self.untie_head:
                model_boundary(base == self.expected_parameters,
                               f"{self.model_id} has {base} base parameters, expected {self.expected_parameters}")
            layout = [(name, tuple(parameter.shape), parameter.requires_grad)
                      for name, parameter in module.named_parameters()]
            self._profiles[key] = (layout, _count(module), _count(module, trainable_only=True), base)
        return self._profiles[key]

    def describe_role(self, role, initialization):
        saved_role, self.role = self.role, role
        try:
            layout, total, trainable, _base = self._profile()
        finally:
            self.role = saved_role
        identity = fingerprint({"model": self.model_id, "revision": self.revision,
                                "config": self._config_data, "settings": asdict(self.settings),
                                "role": role, "initialization": initialization, "layout": layout,
                                "vocabulary": self._vocabulary, "transformers": self._library_version,
                                "pretrained": self.pretrained, "torch": str(torch.__version__),
                                "device": str(self.device), "untie_head": self.untie_head})
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
        child._library_version = self._library_version
        child._profiles = self._profiles
        child.runtime_root, child.resources = self.runtime_root, self.resources
        return child

    def _assets(self):
        if not self.pretrained:
            self._base_files = {}
            return
        from huggingface_hub import snapshot_download
        self._base_path = Path(snapshot_download(
            self.model_id, revision=self.revision, cache_dir=self.cache_dir,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "pytorch_model.bin"]))
        actual = base_assets(self._base_path)
        model_boundary(self._base_files in (None, actual), "Referenced base assets changed")
        self._base_files = actual
        self._base_bytes = sum(path.stat().st_size for path in self._base_path.rglob("*") if path.is_file())

    @property
    def base_storage_bytes(self):
        return self._base_bytes

    def upgrade_parameters(self, dtype=None):
        model_boundary(self.module is not None, "Load the model before upgrading its dtype")
        model_boundary(self.optimizer is None, "Upgrade parameters before the optimizer is built")
        self.module.to(dtype=torch.float32 if dtype is None else dtype)

    def _optimizer_parameters(self):
        groups, new_params = [], []
        by_block = {}
        for name, parameter in self.module.named_parameters():
            if not parameter.requires_grad:
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
            state = load_file(safetensors[0])
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
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(torch.Generator().manual_seed(self.settings.seed).get_state())
            module = self._build(self.config)
            if self.pretrained:
                self._load_checkpoint(module)
            if self._uses_lora():
                torch.random.set_rng_state(torch.Generator().manual_seed(self._seed()).get_state())
                self._attach_lora(module, deterministic=self.role == "probe")
                self._add_new_rank(module, self._new_rank)
                self._mark_adapters(module)
            elif self._merge_deltas:
                with torch.no_grad():
                    for name, value in self._merge_state.items():
                        parameter = module.get_parameter(name)
                        parameter.copy_(value.to(dtype=parameter.dtype))
                    for name, delta in self._merge_deltas.items():
                        weight = module.get_parameter(name + ".weight")
                        weight.add_(delta.to(dtype=weight.dtype))
        self.module = module.to(device=self.device, dtype=load_dtype)
        self.module.config.use_cache = False
        self.upgrade_parameters()
        self._reset_optimizer()
        self.rng_state = self.rng_state or self._new_rng()
        if self._offload is not None:
            self._read_state(Path(self._offload.name))

    def _seed(self):
        key = f"{self.settings.seed}:{self.initialization}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)

    def parameter_counts(self):
        model_boundary(self.module is not None, "Load the model before counting parameters")
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
            if lowest is None or lowest == 0:
                continue
            block = self.module.get_submodule(f"{stack}.block")[lowest - 1]
            self._cut_hooks.append(block.register_forward_hook(_detach_block_output))

    def _as_plan(self, plan):
        if isinstance(plan, AdapterPlan):
            return plan
        return AdapterPlan(tuple(plan["trainable"]), dict(plan.get("new_rank") or {}),
                           dict(plan.get("lr_scale") or {}), dict(plan.get("identity") or {}))

    def _block_count(self, stack):
        return self.config.num_decoder_layers if stack == "decoder" else self.config.num_layers

    def apply_plan(self, plan):
        """Install adapter trainability before the first update and build a fresh optimizer.

        Frozen shared adapters get ``requires_grad = False``. Leaving them trainable
        with a zero learning rate would still run their backward and hide the cost gap.
        New rank stays at the full learning rate. Applying the plan does not change a
        prediction, because the new B factors start at zero.
        """
        self.activate()
        plan = self._as_plan(plan)
        valid = {f"{stack}.{block}" for stack in ("encoder", "decoder")
                 for block in range(1, self._block_count(stack) + 1)}
        trainable, new_rank, scales = set(plan.trainable), dict(plan.new_rank), dict(plan.lr_scale)
        model_boundary(trainable <= valid and set(new_rank) <= valid and set(scales) <= valid,
                       "Adapter plan names an unknown block")
        model_boundary(all(isinstance(rank, int) and rank > 0 for rank in new_rank.values()),
                       "New rank must be a positive integer")
        model_boundary(all(0 < float(scale) <= 1 for scale in scales.values()),
                       "Learning-rate scales must be in (0, 1]")
        for module in self.module.modules():
            if isinstance(module, Seq2SeqLoRA):
                module.clear_extra()
        self._new_rank = new_rank
        self._add_new_rank(self.module, new_rank)
        self._lr_scale = scales
        for module in self.module.modules():
            if not isinstance(module, Seq2SeqLoRA):
                continue
            shared = module.block_key in trainable
            module.A.requires_grad_(shared)
            module.B.requires_grad_(shared)
        for module in self.module.modules():
            if isinstance(module, Seq2SeqLoRA) and module.block_key not in trainable:
                model_boundary(not module.A.requires_grad and not module.B.requires_grad,
                               "Frozen adapters must not stay requires_grad")
        self._profiles.clear()
        self._cut_backward()
        self.optimizer = None
        self._reset_optimizer()
        trainable_count = _count(self.module, trainable_only=True)
        new_count = sum(parameter.numel() for name, parameter in self.module.named_parameters()
                        if parameter.requires_grad and ".extra_" in name)
        self._plan_fingerprint = fingerprint({"identity": plan.identity, "trainable": sorted(trainable),
                                              "new_rank": new_rank, "lr_scale": scales,
                                              "trainable_parameters": trainable_count, "new_parameters": new_count})
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

    def _new_rng(self):
        seed = self._seed()
        states = {"cpu": torch.Generator(device="cpu").manual_seed(seed).get_state()}
        if self.device.type != "cpu":
            states["device"] = torch.Generator(device=self.device).manual_seed(seed).get_state().cpu()
        return states

    def _synchronize(self):
        if self.device.type != "cpu":
            getattr(torch, self.device.type).synchronize(self.device)

    def _begin_timed_block(self):
        self._synchronize()
        return time.perf_counter()

    def _timing_cost(self, start, **fields):
        self._synchronize()
        return {"device": str(self.device), "synchronized_wall_seconds": time.perf_counter() - start, **fields}

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
        model_boundary(payload["input_ids"].shape[0] <= self.settings.max_batch_size, "Batch exceeds the batch bound")
        model_boundary(payload["input_ids"].shape[1] <= self.settings.max_source_length, "Article exceeds the source bound")
        if "labels" in payload:
            model_boundary(payload["labels"].shape[1] <= self.settings.max_target_length, "Summary exceeds the target bound")
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
        counts = valid.sum(-1)
        model_boundary(bool((counts > 0).all()), "Every example needs a supervised summary token")
        chosen = labels.reshape(-1)[valid.reshape(-1)]
        flat = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1])[valid.reshape(-1)], chosen, reduction="none")
        owners = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(valid)[valid]
        sums = torch.zeros(len(counts), device=logits.device, dtype=flat.dtype).scatter_add(0, owners, flat)
        return sums / counts, sums, counts, flat

    def _reduce(self, means, sums, counts, reduction):
        model_boundary(reduction in {"per_example_mean", "per_token_mean"}, "Unknown loss reduction")
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
                model_boundary(score_spec.reduction == "none", "token_loss uses reduction none")
                values = self._losses(payload)[3].detach().float()
            elif score_spec.kind == "probe_gradient":
                values = self._probe_gradient(payload, score_spec)
            else:
                raise ValueError(f"Seq2seq model cannot score {score_spec.kind}")
            cost = self._timing_cost(start)
            revision = fingerprint((self.info.identity, self.steps, self.weight_revision,
                                    self._weights_identity, self._plan_fingerprint))
        nbytes = self._value_bytes(values)
        on_cpu = self._value_device(values) == "cpu"
        return ScoreResult(batch.ids, values, revision, score_spec.parameter_view or score_spec.kind,
                           score_spec.reduction, cost, cpu_bytes=nbytes if on_cpu else 0,
                           accelerator_bytes=0 if on_cpu else nbytes)

    def _value_bytes(self, values):
        if torch.is_tensor(values):
            return values.numel() * values.element_size()
        return sum(item.numel() * item.element_size() for item in values["gradients"])

    def _value_device(self, values):
        tensor = values if torch.is_tensor(values) else values["gradients"][0]
        return tensor.device.type

    def _probe_gradient(self, payload, score_spec):
        """Mean target-loss gradient on probe B factors, grouped by cell or block."""
        model_boundary(self.role == "probe", "Probe gradients are taken on a probe role")
        model_boundary(score_spec.parameter_view in {"cell", "block"}, "Probe gradients group by cell or block")
        selected = [parameter for name, parameter in self.module.named_parameters() if name.endswith(".B")]
        means, sums, counts, _flat = self._losses(payload)
        loss = self._reduce(means, sums, counts, score_spec.reduction)
        gradients = torch.autograd.grad(loss, selected, allow_unused=False)
        grouped = {}
        for (name, _parameter), gradient in zip(
                [(name, parameter) for name, parameter in self.module.named_parameters() if name.endswith(".B")],
                gradients):
            module = self.module.get_submodule(name.rsplit(".", 1)[0])
            if score_spec.parameter_view == "block":
                key = module.block_key
            else:
                key = f"{module.block_key.rsplit('.', 1)[0]}.{module.part}.{_BANDS[(module.block_index) // 2]}"
            grouped.setdefault(key, []).append(gradient.detach().flatten())
        keys = tuple(sorted(grouped))
        return {"groups": keys, "gradients": tuple(torch.cat(grouped[key]) for key in keys)}

    def _plan_context(self):
        return {"plan": self._plan_fingerprint}

    def _optimizer_update(self, loss, revision_context):
        model_boundary(self.optimizer is not None, "This role has no trainable parameters")
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
            self._weights_identity = fingerprint((
                self._weights_identity or self.info.identity, revision_context, self.steps,
                tensor_fingerprint(self._capture_rng()), str(self.device), str(torch.__version__)))
        self.optimizer.zero_grad(set_to_none=True)
        return success

    def _weighted_loss(self, flat, targets):
        weight = targets.weight.detach().to(device=flat.device, dtype=flat.dtype)
        model_boundary(weight.ndim == 1 and weight.numel() == flat.numel(),
                       "Token targets must cover every supervised token")
        active = weight > 0
        if int(active.sum().item()) == 0:
            return None, 0.0, 0
        terms = weight * flat
        return terms.sum() / active.sum(), float(terms.detach().sum().item()), int(active.sum().item())

    def train(self, k, *, batch_factory, objective, hooks, limits):
        """One microbatch per update. The caller chooses the rows. Weighted mode can take no step."""
        hooks = hooks or type("Hooks", (), {"plan_update": None, "gradient_subset": ""})()
        model_boundary(getattr(hooks, "plan_update", None) is None and not getattr(hooks, "gradient_subset", ""),
                       "Seq2seq training does not implement freeze-prefix or subset-gradient hooks")
        model_boundary(objective.kd_weight == 0 and not objective.separate_means,
                       "Seq2seq training does not implement distillation or separate source means")
        model_boundary(objective.token_mode in {"all", "weighted"}, "Seq2seq token mode must be all or weighted")
        reports, attempts, updates, tokens = [], 0, 0, 0
        with self._execution(mode="training", advance=True):
            for _ in range(k):
                for training in batch_factory():
                    if attempts >= limits.max_attempts or updates >= limits.max_steps or tokens + training.batch.valid_tokens > limits.max_tokens:
                        return reports
                    start = self._begin_timed_block()
                    payload = self._payload(training.batch)
                    _means, sums, counts, flat = self._losses(payload)
                    context = {"ids": training.batch.ids, "origins": training.origins,
                               "objective": asdict(objective), **self._plan_context()}
                    if objective.token_mode == "weighted":
                        model_boundary(training.token_targets is not None, "Weighted mode requires token targets")
                        loss, numer, denom = self._weighted_loss(flat, training.token_targets)
                        surprise = flat.detach().float().cpu().tolist()
                        if loss is None:
                            success, reason = False, "no_active_tokens"
                        else:
                            success = self._optimizer_update(loss, context)
                            reason = None if success else "nonfinite_loss_or_gradients"
                        report_sum, report_count = numer, denom
                        diagnostics = {"token_surprise": surprise, **self._plan_context()}
                    else:
                        model_boundary(training.token_targets is None, "token_mode all does not take token targets")
                        loss = self._reduce(_means, sums, counts, objective.reduction)
                        success = self._optimizer_update(loss, context)
                        reason = None if success else "nonfinite_loss_or_gradients"
                        report_sum = float(sums.detach().sum().item())
                        report_count = int(counts.detach().sum().item())
                        diagnostics = dict(self._plan_context())
                    attempts, updates, tokens = attempts + 1, updates + int(success), tokens + training.batch.valid_tokens
                    reports.append(StepReport(
                        training.batch.ids, success, self.steps, training.batch.valid_tokens,
                        {"supervised": report_sum}, {"supervised": report_count}, diagnostics,
                        self._timing_cost(start), reason))
                    del flat, sums, counts
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
        self._script_ids = {"devanagari": torch.tensor(devanagari, dtype=torch.long),
                            "latin": torch.tensor(latin, dtype=torch.long)}
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
            model_boundary("labels" not in payload, "Generation requires source-only inputs")
            max_new = int(generation_spec.get("max_new_tokens", self.settings.max_target_length))
            model_boundary(not generation_spec.get("do_sample", False), "Summarization generation is greedy")
            guard = self._guard(generation_spec)
            output = self.module.generate(
                input_ids=payload["input_ids"], attention_mask=payload.get("attention_mask"),
                max_new_tokens=max_new, do_sample=False,
                logits_processor=[guard] if guard is not None else None)
            rows = output.cpu().tolist()
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
        model_boundary(isinstance(k, int) and k > 0, "Shrink rank must be a positive integer")
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
        self._weights_identity = tensor_fingerprint(self._weights())
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
            state[target] = parameter.detach().cpu().clone()
        deltas = {name: value.detach().cpu().clone() for name, value in self.effective_updates().items()}
        child = self.fork("serving", "merged")
        child.pretrained = False
        child._merge_state = state
        child._merge_deltas = deltas
        return child

    def _weights(self):
        return {"parameters": {name: parameter.detach().cpu() for name, parameter in self.module.named_parameters()},
                "buffers": {name: buffer.detach().cpu() for name, buffer in self.module.named_buffers()}}

    def get_weights(self):
        with self._execution():
            weights = self._weights()
            return {"compatibility": self._compatibility(),
                    **{kind: {name: tensor.clone() for name, tensor in values.items()} for kind, values in weights.items()}}

    def _compatibility(self):
        layout = self._profile()[0]
        return fingerprint({"model": self.model_id, "revision": self.revision, "config": self._config_data,
                            "layout": layout, "rank": self.settings.rank, "alpha": self.settings.alpha,
                            "new_rank": self._new_rank, "vocabulary": self._vocabulary,
                            "transformers": self._library_version, "untie_head": self.untie_head})

    def set_weights(self, weights):
        self.activate()
        model_boundary(weights["compatibility"] == self._compatibility(), "Incompatible model weight layout")
        live = dict(self.module.named_parameters())
        with torch.no_grad():
            for name, value in weights["parameters"].items():
                live[name].copy_(value.to(device=live[name].device, dtype=live[name].dtype))
        self._weights_identity = tensor_fingerprint(weights)
        self.initialization = "weights:" + self._weights_identity
        self._save_full_weights = True
        self.steps = 0
        self.weight_revision += 1
        self.rng_state = self._new_rng()
        self.optimizer = None
        self._reset_optimizer()

    def _metadata(self, kind):
        return {"schema": 1, "state_kind": kind, "compatibility": self._compatibility(),
                "execution_identity": self.info.identity, "model": self.model_id, "revision": self.revision,
                "base_files": self._base_files, "base_bytes": self._base_bytes, "steps": self.steps,
                "weight_revision": self.weight_revision, "initialization": self.initialization,
                "weights_identity": self._weights_identity, "plan": self._plan_fingerprint}

    def save(self, path, state_kind):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.activate()
        torch.save(self._weights(), path / "weights.pt")
        if state_kind == "resume" and self.optimizer is not None:
            torch.save({"optimizer": cpu_tree(self.optimizer.state_dict()),
                        "scheduler": self.scheduler.state_dict(), "steps": self.steps,
                        "weight_revision": self.weight_revision, "weights_identity": self._weights_identity,
                        "rng": cpu_tree(self.rng_state)}, path / "training.pt")
        write_json(path / "metadata.json", self._metadata(state_kind))

    def load(self, path, state_kind):
        path = Path(path)
        metadata = read_json(path / "metadata.json")
        model_boundary(metadata["state_kind"] == state_kind, "Checkpoint kind differs")
        self._prepare()
        model_boundary(metadata["compatibility"] == self._compatibility(), "Incompatible model checkpoint")
        self.activate()
        state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        self.set_weights({**state, "compatibility": metadata["compatibility"]})
        if state_kind == "resume" and (path / "training.pt").is_file() and self.optimizer is not None:
            training = torch.load(path / "training.pt", map_location="cpu", weights_only=True)
            self.optimizer.load_state_dict(training["optimizer"])
            self.scheduler.load_state_dict(training["scheduler"])
            self.steps = training["steps"]
            self.weight_revision = training["weight_revision"]
            self._weights_identity = training["weights_identity"]
            self.rng_state = training["rng"]

    def _read_state(self, path):
        self.load(path, "resume" if (Path(path) / "training.pt").is_file() else "frozen")

    def _reserve_storage(self, additional):
        root = Path(self.runtime_root or self.cache_dir / "standalone")
        root.mkdir(parents=True, exist_ok=True)
        if self.resources is not None:
            Checkpoints(root, self.info.identity, self.resources.disk_bytes,
                        external_bytes=self.base_storage_bytes).reserve(additional)

    def offload(self):
        if self.module is None:
            return
        self._clear_cuts()
        root = Path(self.runtime_root or self.cache_dir / "standalone")
        folder = root / ".offload"
        folder.mkdir(parents=True, exist_ok=True)
        target = tempfile.TemporaryDirectory(dir=folder, prefix=self.role + "-")
        self.save(target.name, "resume" if self.optimizer is not None else "frozen")
        previous, self._offload = self._offload, target
        self.module = self.optimizer = self.scheduler = self.scaler = None
        if previous is not None:
            previous.cleanup()
        gc.collect()

    def release(self):
        self._clear_cuts()
        self.module = self.optimizer = self.scheduler = self.scaler = None
        if self._offload is not None:
            self._offload.cleanup()
            self._offload = None
        gc.collect()
