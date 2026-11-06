"""Shared causal-LM execution using Transformers architectures and native Torch.

The wrapper owns numerical state; paper policies and dataset collation stay with callers.
Imports and constructors do not download assets or allocate model parameters.
Numerical work uses the resolved Torch device: CUDA/ROCm when available, otherwise Intel XPU,
with CPU only when neither accelerator is present. Weights load in float16, then
upgrade_parameters casts them to the optimizer dtype (float32 by default) before AdamW.
Policy reductions and score buffers stay float32. GradScaler stays disabled at that default.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import bisect
import copy
import gc
import hashlib
import math
import random
try:
    import resource
except ImportError:  # Windows has no resource module; peak RSS stays unmeasured.
    resource = None
import shutil
import tempfile
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model_base import ModelBase
from ..utils.contracts import ModelInfo, ScoreResult, StepReport, UpdateHooks, fingerprint, model_boundary
from ..utils import numerics
from ..utils.numerics import resolve_device
from ..utils.state import Checkpoints, base_assets, read_json, write_json, tensor_fingerprint


@dataclass(frozen=True)
class CausalLMSettings:
    """Numerical choices and explicit bounds; working_bytes is a caller-supplied peak estimate."""
    mode: str = "lora"
    rank: int = 16
    alpha: float = 32
    learning_rate: float = 2e-4
    weight_decay: float = .01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    total_steps: int | None = None
    seed: int = 0
    max_length: int = 1024
    max_batch_size: int = 2
    gradient_subset_size: int = 4096
    logit_chunk_size: int = 64
    max_score_bytes: int = 2 * 1024 ** 3
    working_bytes: int = 1024 ** 3
    autocast: str = "float32"
    optimizer_dtype: str = "float32"
    compile: bool = False
    forward_cost: float | None = None
    backward_costs: tuple | None = None


class LoRALinear(nn.Module):
    """Keep the immutable projection and add alpha/rank * B(Ax), with B initially zero."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    @property
    def weight(self):
        """Expose shape metadata expected by architecture implementations."""
        return self.base.weight

    @property
    def bias(self):
        """Retain the original projection bias without adapting or duplicating it."""
        return self.base.bias

    def forward(self, inputs):
        return self.base(inputs) + F.linear(F.linear(inputs, self.A), self.B) * self.scale


def cpu_tree(value):
    """Copy optimizer/RNG tensors to CPU without changing scalar or container structure."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


class CausalLMModel(ModelBase):
    model_id = ""
    model_type = ""
    lora_targets = ("q_proj", "k_proj", "v_proj", "o_proj")
    template_options = {}

    def __init__(self, settings=None, *, device="auto", revision="main", cache_dir=None,
                 collate=None, collate_prompts=None, role="learner", initialization="base",
                 config=None, tokenizer=None, model_factory=None, pretrained=True):
        """Keep allocation lazy; optional config/factory inputs support local architecture instances.
        Production loading resolves and pins the source commit before obtaining any weights.
        Dataset callbacks are injected rather than importing a concrete dataset implementation.
        """
        self.device = resolve_device(device)
        self.runtime_root = None
        self.settings = settings or CausalLMSettings()
        self.revision = revision
        self.cache_dir = Path(cache_dir or Path(__file__).parent / "model_weights")
        self.collate, self.collate_prompts = collate, collate_prompts
        self.role, self.initialization = role, initialization
        self.config, self._tokenizer = config, tokenizer
        self.model_factory, self.pretrained = model_factory, pretrained
        self.module = self.optimizer = self.scheduler = self.scaler = None
        self.steps = self.weight_revision = 0
        self._weights_identity = None
        self._profiles, self._prepared = {}, False
        self._offload = None
        self._save_full_weights = False
        self._base_path, self._base_files = None, None
        self._base_bytes = 0
        self.rng_state = None
        self.resources = None
        self._compiled_forward = None

    def _load_dtype(self):
        """Initial resident weights. The optimizer dtype is applied later by upgrade_parameters."""
        return torch.float16

    def _optimizer_dtype(self):
        """Parameter dtype installed before AdamW. Float32 avoids float16 gradients."""
        if self.settings.optimizer_dtype == "float16":
            return torch.float16
        return torch.float32

    def _compute_dtype(self):
        """Forwards use the dtype the optimizer will see."""
        return self._optimizer_dtype()

    @property
    def use_lora(self):
        """Fixed probes always use LoRA, independently of the learner training mode."""
        return self.role == "probe" or self.settings.mode == "lora"

    def _seed(self, initialization=None):
        """Give each named initialization a stable stream independent of the evolving learner."""
        key = f"{self.settings.seed}:{initialization or self.initialization}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)

    def _prepare(self):
        """Resolve config/tokenizer identity and inspect a parameter-free meta model.
        This obtains exact tied-parameter counts and adapter coordinates before weight allocation.
        No production tensors or optimizer states are retained from this inspection.
        """
        if self._prepared:
            return
        if self.config is None or self._tokenizer is None or self.model_factory is None:
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            if self.config is None:
                self.config = AutoConfig.from_pretrained(self.model_id, revision=self.revision,
                                                        cache_dir=self.cache_dir)
            self._pin_revision()
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, revision=self.revision, cache_dir=self.cache_dir)
            self._library_version = transformers.__version__
            self.model_factory = self.model_factory or (
                lambda config: AutoModelForCausalLM.from_config(config, attn_implementation="eager"))
        else:
            self._library_version = getattr(self, "_library_version", "injected")
        model_boundary(not self.pretrained or self.revision not in ("main", "master"),
                       "An injected pretrained config needs a pinned model revision")
        self._config_data = self.config.to_dict()
        model_boundary(self.config.model_type == self.model_type, "Model architecture does not match class")
        self._vocabulary = fingerprint({"vocabulary": self._tokenizer.get_vocab(),
                                        "special": self._tokenizer.special_tokens_map})
        self._template = fingerprint({"chat_template": self._tokenizer.chat_template,
                                      "options": self.template_options, "allow_reasoning": False})
        self._prepared = True

    def _pin_revision(self):
        """Replace a branch name with the immutable commit of the cached snapshot.
        Transformers 5 no longer records that commit on the config object.
        """
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
        """Load tokenizer/config metadata on first explicit access."""
        self._prepare()
        return self._tokenizer

    def render(self, messages):
        """Render a generation prompt with the class's fixed non-thinking template options."""
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, **self.template_options)

    def encode(self, text):
        """Templates already contain special tokens; do not insert a second BOS or EOS."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def preprocessing_kwargs(self, *, max_length=None, truncation="drop"):
        """Return InstructionPreprocessing arguments without coupling models to datasets."""
        tok = self.tokenizer
        return {"encode": self.encode, "render": self.render, "eos_id": tok.eos_token_id,
                "pad_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                "max_length": max_length or self.settings.max_length, "truncation": truncation,
                "allow_reasoning": False,
                "identity": {"model": self.model_id, "revision": self.revision,
                             "vocabulary": self._vocabulary, "template": self._template,
                             "enable_thinking": False, "transformers": self._library_version}}

    def _attach_lora(self, module, enabled):
        """Wrap only declared decoder projections; preserve the architecture's tied output head."""
        if enabled:
            for name, child in list(module.named_modules()):
                if (name.startswith("model.layers.") and isinstance(child, nn.Linear)
                        and name.rsplit(".", 1)[-1] in self.lora_targets):
                    parent, attribute = name.rsplit(".", 1)
                    setattr(module.get_submodule(parent), attribute,
                            LoRALinear(child, self.settings.rank, self.settings.alpha))
            for name, parameter in module.named_parameters():
                parameter.requires_grad_(name.endswith((".A", ".B")))
        else:
            module.requires_grad_(True)

    def _group(self, name):
        """Order embeddings, decoder blocks and final output parameters; aliases count once."""
        if name.startswith("model.embed_tokens."):
            return 0
        if name.startswith("model.layers."):
            return int(name.split(".")[2]) + 1
        return self.config.num_hidden_layers + 1

    def _profile(self, lora):
        """Inspect a meta architecture and record stable parameter layouts and legal prefixes."""
        self._prepare()
        if lora not in self._profiles:
            with torch.random.fork_rng(devices=[]), torch.device("meta"):
                module = self.model_factory(copy.deepcopy(self.config))
                self._attach_lora(module, lora)
            layout = [(n, tuple(p.shape), p.requires_grad) for n, p in module.named_parameters()]
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            groups = [0] * (self.config.num_hidden_layers + 2)
            for name, parameter in module.named_parameters():
                groups[self._group(name)] += parameter.numel()
            legal = tuple(i for i in range(len(groups))
                          if any(train and self._group(n) >= i for n, _, train in layout))
            self._profiles[lora] = (layout, total, trainable, groups, legal)
        return self._profiles[lora]

    def _compatibility(self):
        """Bind weights to their base, layout and tokenizer, independently of training role."""
        layout = self._profile(self.use_lora)[0]
        return fingerprint({"model": self.model_id, "revision": self.revision,
                            "config": self._config_data, "layout": layout,
                            "base_seed": None if self.pretrained else self.settings.seed,
                            "rank": self.settings.rank, "alpha": self.settings.alpha,
                            "vocabulary": self._vocabulary, "template": self._template,
                            "transformers": self._library_version})

    def describe_role(self, role, initialization):
        """Advertise conservative tensor/checkpoint bounds before constructing a numerical role.
        Working and score buffers are explicit bounds; parameter-based FLOPs are estimates.
        Callers can replace the layer costs with measurements for their batch/sequence shape.
        """
        lora = role == "probe" or self.settings.mode == "lora"
        layout, total, trainable, groups, legal = self._profile(lora)
        identity = fingerprint({"model": self.model_id, "revision": self.revision,
                                "config": self._config_data, "settings": asdict(self.settings),
                                "role": role, "initialization": initialization, "layout": layout,
                                "vocabulary": self._vocabulary, "template": self._template,
                                "transformers": self._library_version, "pretrained": self.pretrained,
                                "torch": str(torch.__version__), "device": str(self.device)})
        param_bytes = 2 if self.settings.optimizer_dtype == "float16" else 4
        weight_bytes = total * param_bytes
        saved_bytes = (trainable if lora and not self._save_full_weights else total) * param_bytes
        numerical_bytes = weight_bytes + trainable * 12
        peak = numerical_bytes + self.settings.working_bytes + self.settings.max_score_bytes
        capabilities = {"loss", "last_layer_gradients", "subset_gradients", "logits", "distillation",
                        "feature_gradient", "layer_fisher", "freeze_aware", "generation",
                        "token_loss", "prompt_embedding"}
        if lora:
            capabilities.update(("lora_b_gradient", "fixed_zero_lora_b"))
        return ModelInfo(identity, total, trainable, capabilities=frozenset(capabilities),
                         cpu_bytes=peak, accelerator_bytes=peak if self.device.type != "cpu" else 0,
                         checkpoint_bytes=saved_bytes + trainable * 12 + 1024 ** 2,
                         vocabulary=self._vocabulary, loss_fingerprint="causal-response-ce-v1",
                         parameter_views=("head", "lm_head", "lora-b", "subset"),
                         layer_names=("embeddings", *(f"block.{i}" for i in range(len(groups) - 2)), "output"),
                         forward_cost=(float(sum(groups)) if self.settings.forward_cost is None
                                       else self.settings.forward_cost),
                         backward_costs=(tuple(2. * n for n in groups) if self.settings.backward_costs is None
                                         else self.settings.backward_costs),
                         legal_prefixes=legal)

    @property
    def info(self):
        """Describe this role without allocating its production tensors."""
        return self.describe_role(self.role, self.initialization)

    def fork(self, role, initialization):
        """Create an unloaded isolated role; its immutable metadata/tokenizer may be shared."""
        self._prepare()
        child = type(self)(self.settings, device=str(self.device), revision=self.revision,
                           cache_dir=self.cache_dir, collate=self.collate, collate_prompts=self.collate_prompts,
                           role=role, initialization=initialization, config=copy.deepcopy(self.config),
                           tokenizer=self.tokenizer, model_factory=self.model_factory, pretrained=self.pretrained)
        child._profiles = self._profiles
        child._library_version = self._library_version
        child._save_full_weights = self._save_full_weights
        child.runtime_root, child.resources = self.runtime_root, self.resources
        return child

    def bind_runtime(self, output, resources):
        """Place private numerical offload files inside the receiving run's storage boundary.
        Copy an inactive model's saved state before releasing its previous private directory.
        Moving between attempts therefore preserves weights and optimizer state without loading tensors.
        """
        super().bind_runtime(output, resources)
        self.resources = resources
        if self._offload is not None and not Path(self._offload.name).resolve().is_relative_to(Path(output).resolve()):
            previous = self._offload
            target = self._offload_directory()
            for file in Path(previous.name).iterdir():
                shutil.copyfile(file, Path(target.name) / file.name)
            self._offload = target
            previous.cleanup()

    def attach_instruction(self, dataset):
        """Bind an instruction dataset preprocessed with this local tokenizer and template.

        CLINC, Schuster, Bitext, and TRACE share this check. The model does not import a fetcher.
        """
        self.bind_dataset(dataset)
        return self

    def bind_dataset(self, dataset):
        """Bind dataset callbacks and require the same non-thinking tokenizer/template identity."""
        super().bind_dataset(dataset)
        actual = dataset.manifest["preprocessing"]
        expected = self.preprocessing_kwargs()
        model_boundary(actual["identity"] == expected["identity"]
                       and actual.get("allow_reasoning") is False
                       and actual["eos_id"] == expected["eos_id"]
                       and actual["pad_id"] == expected["pad_id"]
                       and actual["max_length"] <= self.settings.max_length,
                       "Dataset preprocessing does not match this non-thinking model")

    def _new_rng(self):
        """Create local CPU/device RNG streams without seeding global generators."""
        seed = self._seed()
        states = {"cpu": torch.Generator(device="cpu").manual_seed(seed).get_state()}
        if self.device.type != "cpu":
            states["device"] = torch.Generator(device=self.device).manual_seed(seed).get_state().cpu()
        return states

    def _capture_rng(self):
        """Capture only the CPU and selected device streams used by this role."""
        states = {"cpu": torch.random.get_rng_state()}
        if self.device.type != "cpu":
            states["device"] = getattr(torch, self.device.type).get_rng_state(self.device).cpu()
        return states

    def _restore_rng(self, states):
        """Restore saved streams without changing generators on other accelerators."""
        torch.random.set_rng_state(states["cpu"])
        if self.device.type != "cpu":
            getattr(torch, self.device.type).set_rng_state(states["device"], self.device)

    def _assets(self):
        """Resolve one immutable base snapshot and hash referenced files before loading tensors."""
        if self.pretrained:
            from huggingface_hub import snapshot_download
            self._base_path = Path(snapshot_download(
                self.model_id, revision=self.revision, cache_dir=self.cache_dir,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]))
            actual = base_assets(self._base_path)
            model_boundary(self._base_files in (None, actual), "Referenced base assets changed")
            self._base_files = actual
            self._base_bytes = sum(p.stat().st_size for p in self._base_path.rglob("*") if p.is_file())
        else:
            self._base_files = {}

    @property
    def base_storage_bytes(self):
        """Charge the verified immutable cache once alongside run-owned state."""
        return self._base_bytes

    def upgrade_parameters(self, dtype=None):
        """Cast the loaded module before any optimizer state exists.
        Loading stays float16; pass float32, or rely on settings.optimizer_dtype, for training.
        """
        model_boundary(self.module is not None, "Load the model before upgrading its dtype")
        model_boundary(self.optimizer is None, "Upgrade parameters before the optimizer is built")
        target = self._optimizer_dtype() if dtype is None else dtype
        self.module.to(dtype=target)

    def _reset_optimizer(self):
        """Keep every eligible parameter in AdamW; grad=None excludes frozen weights and momentum."""
        names = {n for n, _, train in self._profile(self.use_lora)[0] if train}
        parameters = [p for n, p in self.module.named_parameters() if n in names]
        self.module.zero_grad(set_to_none=True)
        self.optimizer = torch.optim.AdamW(parameters, lr=self.settings.learning_rate,
                                          weight_decay=self.settings.weight_decay)
        def schedule(step):
            if self.settings.warmup_steps and step < self.settings.warmup_steps:
                return (step + 1) / self.settings.warmup_steps
            if self.settings.total_steps is None:
                return 1.
            return max(0., (self.settings.total_steps - step)
                       / max(1, self.settings.total_steps - self.settings.warmup_steps))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.compute_dtype = self._compute_dtype()
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.compute_dtype == torch.float16)

    def activate(self):
        """Load a fresh base plus the exact saved numerical overlay, one role at a time.
        Weights load in float16, then upgrade to the optimizer dtype before AdamW is created.
        Existing offload state is restored after building the same architecture and optimizer.
        """
        if self.module is not None:
            return
        self.info.validate()
        if self.resources is not None:
            self.resources.validate_role(self.info)
            self._reserve_storage(self.info.checkpoint_bytes)
        self._assets()
        if self.resources is not None:
            self._reserve_storage(self.info.checkpoint_bytes)
        load_dtype = self._load_dtype()
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(torch.Generator().manual_seed(self.settings.seed).get_state())
            if self.pretrained:
                from transformers import AutoModelForCausalLM
                module = AutoModelForCausalLM.from_pretrained(
                    self._base_path, dtype=load_dtype, attn_implementation="eager", use_safetensors=True)
            else:
                module = self.model_factory(copy.deepcopy(self.config))
            torch.random.set_rng_state(torch.Generator().manual_seed(self._seed()).get_state())
            self._attach_lora(module, self.use_lora)
        self.module = module.to(device=self.device, dtype=load_dtype)
        self.module.config.use_cache = False
        self.upgrade_parameters()
        self._reset_optimizer()
        self.rng_state = self.rng_state or self._new_rng()
        if self._offload is not None:
            self._read_state(Path(self._offload.name))
        self._compiled_forward = None

    def parameter_counts(self):
        """Count tied parameters once; restore baseline eligibility after every frozen-prefix update."""
        return (sum(p.numel() for p in self.module.parameters()),
                sum(p.numel() for p in self.module.parameters() if p.requires_grad))

    def _synchronize(self):
        if self.device.type != "cpu":
            getattr(torch, self.device.type).synchronize(self.device)

    def _torch_backend(self):
        """Return the Torch backend module for the resolved device, if any."""
        if self.device.type == "cpu":
            return None
        return getattr(torch, self.device.type, None)

    def _supports_event_timing(self):
        backend = self._torch_backend()
        return backend is not None and hasattr(backend, "Event")

    def _begin_timed_block(self):
        """Start timing with Torch events on accelerators; wall clock on CPU fallback."""
        self._synchronize()
        backend = self._torch_backend()
        if backend is not None:
            backend.reset_peak_memory_stats(self.device)
        if self._supports_event_timing():
            backend = self._torch_backend()
            start = backend.Event(enable_timing=True)
            with torch.device(self.device):
                start.record()
            return ("torch", start)
        return ("wall", time.perf_counter())

    def _timing_cost(self, start, **fields):
        """Finish timing; Torch ops report device-active seconds from recorded events."""
        cost = {"device": str(self.device), **fields}
        if start[0] == "torch":
            backend = self._torch_backend()
            end = backend.Event(enable_timing=True)
            with torch.device(self.device):
                end.record()
            backend.synchronize(self.device)
            cost["device_active_seconds"] = start[1].elapsed_time(end) / 1000.0
            cost["peak_device_allocated_bytes"] = backend.max_memory_allocated(self.device)
        else:
            self._synchronize()
            cost["synchronized_wall_seconds"] = time.perf_counter() - start[1]
        return cost

    @contextmanager
    def _execution(self, *, mode="inference", advance=False):
        """Isolate one public operation in either training or inference module mode.

        Switching is implicit from the entry point: ``train`` uses ``mode="training"``;
        ``score``, ``evaluate``, ``predict``, and ``get_weights`` use inference
        (``module.eval()`` semantics via ``module.train(False)``). Inference scoring
        may still enable autograd for gradient probes; only ``predict`` and held-out
        loss paths wrap work in ``torch.no_grad()``. Training alone may advance the
        role-local RNG when ``advance=True``.
        """
        if mode not in {"training", "inference"}:
            raise ValueError("Execution mode must be training or inference")
        active = self.module is not None
        self.activate()
        caller_rng = self._capture_rng()
        modes = [(m, m.training) for m in self.module.modules()]
        flags = [(p, p.requires_grad) for p in self.module.parameters()]
        buffers = [] if advance else [(b, b.detach().clone()) for b in self.module.buffers()]
        self._restore_rng(self.rng_state)
        self.module.train(mode == "training")
        try:
            with torch.autocast(self.device.type, dtype=self.compute_dtype,
                                enabled=self.compute_dtype != torch.float32):
                yield mode
        finally:
            if advance:
                self.rng_state = self._capture_rng()
            self._restore_rng(caller_rng)
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
        """Enforce configured allocation bounds before sending a collated batch to the device."""
        payload = batch.payload
        shape = payload["input_ids"].shape
        model_boundary(shape[0] <= self.settings.max_batch_size and shape[1] <= self.settings.max_length,
                       "Batch exceeds configured model working bounds")
        model_boundary(all(not any(tag in str(r.target).lower() for tag in
                       ("<think>", "</think>", "<analysis>", "</analysis>")) for r in batch.records),
                       "Non-thinking training requires final-response targets")
        return self.to_device(payload)

    def _decoder(self, **kwargs):
        """Run the decoder forward, compiling it once when this role asked for torch.compile."""
        if self._compiled_forward is None:
            forward = self.module.model.forward
            if self.settings.compile:
                forward = torch.compile(forward, dynamic=True)
            self._compiled_forward = forward
        return self._compiled_forward(**kwargs)

    def _features(self, payload):
        """Obtain final token features without allocating logits for prompt/padding tokens."""
        return self._decoder(input_ids=payload["input_ids"],
                             attention_mask=payload["attention_mask"],
                             use_cache=False, return_dict=True).last_hidden_state

    def _batched_head_gradients(self, payload, reduction):
        """One decoder pass for every example's tied last-layer gradient.

        A single-example reduction is that example's token mean for both batch reductions,
        so the batch differentiates each of those means. The output-projection part is the
        outer product of logit gradients and hidden states. A tied embedding adds each
        example's rows from the decoder gradient. Untied heads omit that scatter.
        """
        input_ids = payload["input_ids"]
        labels = payload["labels"]
        embed = self.module.get_input_embeddings()
        tied = embed.weight is self.module.lm_head.weight
        model_boundary(reduction in {"per_example_mean", "per_token_mean"}, "Unknown gradient reduction")
        batch, seq = input_ids.shape
        hidden = self.module.lm_head.in_features
        vocab = self.module.lm_head.out_features
        self._buffer_bound(batch * vocab * hidden + batch * max(seq - 1, 0) * vocab)
        if tied:
            embedded = embed(input_ids).detach().requires_grad_(True)
            features = self._decoder(inputs_embeds=embedded, attention_mask=payload["attention_mask"],
                                     use_cache=False, return_dict=True).last_hidden_state
            means, _, counts = self._losses(features, labels)
            # One-example reductions both differentiate that example's token mean.
            token_grad = torch.autograd.grad(means.sum(), embedded)[0]
            padding_idx = getattr(embed, "padding_idx", None)
            if padding_idx is not None:
                token_grad = token_grad.masked_fill((input_ids == padding_idx).unsqueeze(-1), 0)
        else:
            with torch.no_grad():
                features = self._features(payload)
                _, _, counts = self._losses(features, labels)
            token_grad = None
        shifted = features[:, :-1]
        with torch.no_grad():
            targets = labels[:, 1:]
            valid = targets != -100
            logits = self.module.lm_head(shifted).float()
            safe = targets.clamp(min=0)
            grad_logits = logits.softmax(-1)
            chosen = grad_logits.gather(-1, safe.unsqueeze(-1)) - 1
            grad_logits.scatter_(-1, safe.unsqueeze(-1), chosen)
            scale = counts.to(grad_logits.dtype).clamp_min(1)[:, None, None]
            grad_logits = torch.where(valid.unsqueeze(-1), grad_logits / scale, torch.zeros_like(grad_logits))
            result = torch.matmul(grad_logits.transpose(1, 2), shifted.float())
            if token_grad is not None:
                index = (torch.arange(batch, device=result.device)[:, None] * vocab + input_ids).reshape(-1)
                result.view(batch * vocab, hidden).index_add_(0, index, token_grad.reshape(-1, hidden).float())
        return result.view(batch, -1)

    def _head_loss(self, features, labels):
        """Compute one small vocabulary projection in float32 for stable causal cross entropy."""
        return F.cross_entropy(self.module.lm_head(features).float(), labels, reduction="none")

    def _token_nll(self, features, labels):
        """Return flat causal NLL, the row that owns each token, and the per-row counts.
        Order is row-major over supervised positions. Prompt and padding labels contribute nothing.
        Checkpoint head chunks during autograd so vocabulary-sized intermediates are recomputed.
        """
        targets = labels[:, 1:]
        valid = targets != -100
        counts = valid.sum(-1)
        model_boundary(bool((counts > 0).all()), "Every example needs a causally supervised target")
        selected = features[:, :-1][valid]
        chosen = targets[valid]
        owners = torch.arange(labels.shape[0], device=self.device)[:, None].expand_as(valid)[valid]
        parts = []
        for start in range(0, len(chosen), self.settings.logit_chunk_size):
            stop = start + self.settings.logit_chunk_size
            part = (checkpoint(self._head_loss, selected[start:stop], chosen[start:stop], use_reentrant=False)
                    if torch.is_grad_enabled() else self._head_loss(selected[start:stop], chosen[start:stop]))
            parts.append(part)
        return torch.cat(parts), owners, counts

    def _losses(self, features, labels):
        """Sum the flat token NLL into one mean and one sum per example."""
        flat, owners, counts = self._token_nll(features, labels)
        sums = torch.zeros(len(counts), device=self.device).scatter_add(0, owners, flat)
        return sums / counts, sums, counts

    @staticmethod
    def _reduce(means, sums, counts, reduction):
        """Keep per-example averaging distinct from averaging all supervised response tokens."""
        return {"per_example_mean": lambda: means.mean(),
                "per_token_mean": lambda: sums.sum() / counts.sum()}[reduction]()

    def supervised_loss(self, means, sums, counts, objective, origins):
        """Apply separately normalized current/replay losses when the supplied objective requests it."""
        if not objective.separate_means:
            return self._reduce(means, sums, counts, objective.reduction)
        loss = sums.sum() * 0
        for origin, weight in (("current", objective.current_weight), ("replay", objective.replay_weight)):
            mask = torch.tensor([o == origin for o in origins], device=self.device)
            if mask.any():
                loss = loss + weight * self._reduce(means[mask], sums[mask], counts[mask], objective.reduction)
        return loss

    def _batch_identity(self, payload):
        """Tie scores to token identities and masks. Labels are included when the batch has them."""
        keys = [key for key in ("input_ids", "attention_mask", "labels") if key in payload]
        return fingerprint({key: payload[key].detach().cpu().tolist() for key in keys})

    def distillation_loss(self, features, payload, targets, objective, origins):
        """Evaluate exact KL(teacher || student) in supervised-token chunks.
        Teacher logits have already been divided by temperature by score_logits.
        Transfer only each CPU target chunk; preserve the requested token/example denominator.
        """
        model_boundary(targets.cost.get("vocabulary") == self._vocabulary
                       and targets.cost.get("token_positions") == self._batch_identity(payload),
                       "Teacher vocabulary or supervised token positions differ")
        labels = payload["labels"][:, 1:]
        valid = labels != -100
        positions = valid.flatten().nonzero().flatten()
        hidden = features[:, :-1].reshape(-1, features.shape[-1])
        values = targets.values.reshape(-1, self.config.vocab_size)
        parts = []
        def kl_chunk(x, teacher):
            student = self.module.lm_head(x).float() / objective.kd_temperature
            teacher = teacher.to(self.device)
            return F.kl_div(F.log_softmax(student, -1), F.log_softmax(teacher, -1),
                            log_target=True, reduction="none").sum(-1)
        for start in range(0, len(positions), self.settings.logit_chunk_size):
            index = positions[start:start + self.settings.logit_chunk_size]
            teacher = values.index_select(0, index.cpu())
            parts.append(checkpoint(kl_chunk, hidden[index], teacher, use_reentrant=False))
        owners = torch.arange(labels.shape[0], device=self.device)[:, None].expand_as(valid)[valid]
        sums = features.new_zeros(labels.shape[0], dtype=torch.float32).scatter_add(0, owners, torch.cat(parts))
        counts = valid.sum(-1)
        loss = self.supervised_loss(sums / counts, sums, counts, objective, origins)
        return loss * (objective.kd_temperature ** 2 if objective.kd_temperature_squared else 1.)

    def _buffer_bound(self, elements):
        """Check float32 score storage before materializing a large gradient/logit result."""
        model_boundary(elements * 4 <= self.settings.max_score_bytes, "Score buffer exceeds configured byte bound")

    def _coordinates(self, view="subset"):
        """Sample stable global parameter coordinates without allocating a full random permutation."""
        layout = [(n, math.prod(shape)) for n, shape, train in self._profile(self.use_lora)[0] if train]
        ends, total = [], 0
        for _, size in layout:
            total += size
            ends.append(total)
        seed = self._seed(view)
        chosen = sorted(random.Random(seed).sample(range(total), min(total, self.settings.gradient_subset_size)))
        grouped = {}
        for index in chosen:
            position = bisect.bisect_right(ends, index)
            grouped.setdefault(layout[position][0], []).append(index - (ends[position - 1] if position else 0))
        return grouped

    def score_loss(self, payload):
        """Return one response-token mean per ordered example, without constructing a training graph."""
        with torch.no_grad():
            return self._losses(self._features(payload), payload["labels"])[0].float()

    def score_token_loss(self, payload):
        """Return detached per-token NLL, flat in row-major order over supervised tokens."""
        with torch.no_grad():
            return self._token_nll(self._features(payload), payload["labels"])[0].float()

    def score_prompt_embedding(self, payload):
        """Return one L2-normalized mean of the final hidden state over attended prompt tokens."""
        model_boundary("labels" not in payload and "attention_mask" in payload,
                       "Prompt embeddings need a prompt-only batch")
        with torch.no_grad():
            hidden = self._features(payload).float()
            mask = payload["attention_mask"].to(dtype=hidden.dtype).unsqueeze(-1)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.)
            norm = torch.linalg.vector_norm(mean, dim=-1, keepdim=True)
            return mean / torch.where(norm > 0, norm, torch.ones_like(norm))

    def score_logits(self, payload, temperature):
        """Buffer temperature-scaled teacher logits on CPU at causally shifted positions.
        Non-supervised positions stay zero and are excluded by the distillation mask.
        The full output allocation is checked before any vocabulary projection is made.
        """
        labels = payload["labels"][:, 1:]
        shape = (*labels.shape, self.config.vocab_size)
        self._buffer_bound(math.prod(shape))
        output = torch.zeros(shape, dtype=torch.float32)
        with torch.no_grad():
            features = self._features(payload)[:, :-1]
            positions = (labels != -100).nonzero()
            for start in range(0, len(positions), self.settings.logit_chunk_size):
                index = positions[start:start + self.settings.logit_chunk_size]
                rows, columns = index.unbind(-1)
                output[rows.cpu(), columns.cpu()] = (
                    self.module.lm_head(features[rows, columns]).float() / temperature).cpu()
        return output

    def score_gradients(self, payload, kind, view, reduction):
        """Use autograd.grad so scoring never overwrites existing training gradients.
        Head requests include all uses of tied weights and differentiate the whole batch at once.
        LoRA-B requests use the fixed probe and differentiate the batch objective.
        The returned buffer stays on the parameter device.
        """
        if kind == "last_layer_gradients":
            return self._batched_head_gradients(payload, reduction)
        named = dict(self.module.named_parameters())
        coordinates = None
        if kind == "lora_b_gradient":
            selected = {n: p for n, p in named.items() if n.endswith(".B")}
        else:
            coordinates = self._coordinates(view)
            selected = {n: named[n] for n in coordinates}
        for parameter in selected.values():
            parameter.requires_grad_(True)
        width = sum(len(coordinates[n]) if coordinates is not None else p.numel() for n, p in selected.items())
        rows = 1 if kind == "lora_b_gradient" else len(payload["input_ids"])
        self._buffer_bound(rows * width)
        device = next(iter(selected.values())).device if selected else self.device
        result = torch.empty((rows, width), dtype=torch.float32, device=device)
        for i in range(rows):
            current = payload if kind == "lora_b_gradient" else {k: v[i:i + 1] for k, v in payload.items()}
            means, sums, counts = self._losses(self._features(current), current["labels"])
            loss = self._reduce(means, sums, counts, reduction)
            gradients = torch.autograd.grad(loss, tuple(selected.values()), allow_unused=True)
            offset = 0
            for (name, parameter), gradient in zip(selected.items(), gradients):
                value = torch.zeros_like(parameter) if gradient is None else gradient
                value = value.flatten()
                if coordinates is not None:
                    value = value[coordinates[name]]
                result[i, offset:offset + value.numel()].copy_(value.detach())
                offset += value.numel()
            del gradients
        return result[0] if kind == "lora_b_gradient" else result

    def score(self, batch, score_spec):
        """Dispatch one explicit representation with model, vocabulary and buffer provenance."""
        with self._execution() as execution_mode:
            payload = self._payload(batch)
            start = self._begin_timed_block()
            if score_spec.kind in {"token_loss", "prompt_embedding"}:
                model_boundary(score_spec.reduction == "none", f"{score_spec.kind} uses reduction none")
            if score_spec.kind == "loss":
                values = self.score_loss(payload)
            elif score_spec.kind == "token_loss":
                values = self.score_token_loss(payload)
            elif score_spec.kind == "prompt_embedding":
                values = self.score_prompt_embedding(payload)
            elif score_spec.kind == "logits":
                values = self.score_logits(payload, score_spec.temperature)
            else:
                values = self.score_gradients(payload, score_spec.kind, score_spec.parameter_view, score_spec.reduction)
            cost = self._timing_cost(start, execution_mode=execution_mode, vocabulary=self._vocabulary,
                                     token_positions=self._batch_identity(payload),
                                     output_bytes=values.numel() * values.element_size())
            revision = fingerprint((self.info.identity, self.steps, self.weight_revision, self._weights_identity))
        nbytes = values.numel() * values.element_size()
        on_cpu = values.device.type == "cpu"
        return ScoreResult(batch.ids, values, revision, score_spec.parameter_view, score_spec.reduction,
                           cost, cpu_bytes=nbytes if on_cpu else 0,
                           accelerator_bytes=0 if on_cpu else nbytes)

    def feature_statistics(self, payload, objective, origins):
        """Differentiate only the final features for the pre-backward freeze decision.
        Restore randomness before the actual training forward so dropout uses the same draw.
        This additional forward is charged separately in the step's cost report.
        """
        rng = self._capture_rng()
        with torch.no_grad():
            features = self._features(payload)
        self._restore_rng(rng)
        features = features.detach().requires_grad_(True)
        means, sums, counts = self._losses(features, payload["labels"])
        loss = self.supervised_loss(means, sums, counts, objective, origins)
        gradient = torch.autograd.grad(loss, features)[0]
        return {"feature_norm_sq": gradient.float().square().sum().item()}

    def apply_freeze(self, prefix):
        """Exclude frozen groups from backward and AdamW, retaining their optimizer state for reuse."""
        eligible = {n for n, _, train in self._profile(self.use_lora)[0] if train}
        for name, parameter in self.module.named_parameters():
            parameter.requires_grad_(name in eligible and self._group(name) >= prefix)
        self.optimizer.zero_grad(set_to_none=True)

    def example_contributions(self, means, sums, counts, objective, origins):
        """Scale example losses so their mean equals the actual supervised batch objective.
        This keeps Fisher/SAR statistics consistent with token weighting and separate source means.
        """
        size = len(means)
        if not objective.separate_means:
            return means if objective.reduction == "per_example_mean" else size * sums / counts.sum()
        values = sums * 0
        for origin, weight in (("current", objective.current_weight), ("replay", objective.replay_weight)):
            mask = torch.tensor([o == origin for o in origins], device=self.device)
            if mask.any():
                contribution = (means / mask.sum() if objective.reduction == "per_example_mean"
                                else sums / counts[mask].sum())
                values = values + size * weight * contribution * mask
        return values

    def gradient_statistics(self, means, sums, counts, hooks, plan, objective, origins):
        """Collect supervised per-example Fisher traces and sampled unfrozen SAR coordinates.
        Reuse the current forward graph without touching .grad or including distillation.
        Discard each full parameter gradient after reducing its norm/sampled coordinates.
        """
        refresh = plan is not None and plan.refresh
        coordinates = self._coordinates(hooks.gradient_subset) if hooks.gradient_subset else {}
        named = {n: p for n, p in self.module.named_parameters() if p.requires_grad
                 and (refresh or n in coordinates)}
        groups = len(self.info.layer_names)
        traces = torch.zeros(groups, device=self.device, dtype=torch.float32) if refresh and named else None
        active = {n: indices for n, indices in coordinates.items() if n in named}
        self._buffer_bound(len(means) * sum(map(len, active.values())))
        rows = []
        if named:
            values = self.example_contributions(means, sums, counts, objective, origins)
            for loss in values:
                gradients = torch.autograd.grad(loss, tuple(named.values()), retain_graph=True, allow_unused=True)
                pieces = []
                for (name, parameter), gradient in zip(named.items(), gradients):
                    if gradient is None:
                        gradient = torch.zeros_like(parameter)
                    if traces is not None:
                        traces[self._group(name)] += gradient.detach().float().square().sum() / len(means)
                    if name in active:
                        pieces.append(gradient.detach().flatten()[active[name]].float())
                if pieces:
                    rows.append(torch.cat(pieces))
        return {"layer_fisher": [0.] * groups if traces is None else traces.tolist(),
                "fisher_refresh": refresh, "gradient_subset": hooks.gradient_subset,
                "class_gradients": torch.stack(rows) if rows else None}

    def optimizer_update(self, loss, *, revision_context=None):
        """Commit finite updates only; nonfinite attempts consume data/RNG and update the scaler.
        grad=None prevents frozen weights from receiving momentum or weight decay.
        Scheduler and successful-step counters advance only after an actual optimizer update.
        """
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        parameters = [p for p in self.module.parameters() if p.grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.settings.max_grad_norm)
        success = bool(torch.isfinite(loss.detach()) and torch.isfinite(norm))
        if success:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.steps += 1
            self.weight_revision += 1
            if revision_context is None:
                self._weights_identity = tensor_fingerprint(self._weights(changed_only=not self._save_full_weights))
            else:
                self._weights_identity = fingerprint((
                    self._weights_identity or self.info.identity, revision_context, self.steps,
                    tensor_fingerprint(self._capture_rng()), str(self.device), str(torch.__version__)))
        elif self.scaler.is_enabled():
            self.scaler.update(new_scale=max(1., self.scaler.get_scale() / 2))
        self.optimizer.zero_grad(set_to_none=True)
        return success

    def _masked_token_loss(self, flat, targets, objective):
        """Return the weighted or hinge loss, its detached numerator, and the active count.
        A zero count means the batch has nothing to practise. The mask is not differentiated.
        """
        weight = targets.weight.detach().to(device=flat.device, dtype=flat.dtype)
        baseline = targets.baseline.detach().to(device=flat.device, dtype=flat.dtype)
        model_boundary(weight.dtype == flat.dtype and weight.ndim == 1
                       and tuple(targets.weight.shape) == tuple(targets.baseline.shape)
                       and targets.weight.dtype == torch.float32 and targets.baseline.dtype == torch.float32
                       and targets.weight.device.type == "cpu" and targets.baseline.device.type == "cpu",
                       "Token targets must be flat float32 CPU tensors")
        if objective.token_mode == "weighted":
            terms = weight * flat
            active = weight > 0
        else:
            excess = flat - baseline - objective.token_allowance
            terms = weight * excess.clamp(min=0)
            active = (weight > 0) & (excess.detach() > 0)
        denom = int(active.sum().item())
        numer = 0. if denom == 0 else float(terms.detach().sum().item())
        if denom == 0:
            return None, numer, denom
        return terms.sum() / denom, numer, denom

    def _token_update(self, training, objective, hooks, payload, start, execution_mode):
        """Train on token weights, or skip the optimizer when the active count is zero.
        AdamW is not called on that skip, so weight decay cannot move a recovered token.
        """
        model_boundary(objective.token_mode in {"weighted", "hinge"}, "Unknown token mode")
        model_boundary(not objective.separate_means and objective.kd_weight == 0,
                       "Token modes do not combine with separate means or distillation")
        targets = training.token_targets
        model_boundary(targets is not None, "Token modes require token targets")
        self.apply_freeze(0)
        features = self._features(payload)
        flat, _, _ = self._token_nll(features, payload["labels"])
        model_boundary(flat.numel() == targets.weight.numel() == training.batch.supervised_tokens,
                       "Token targets must cover every supervised token")
        surprise = flat.detach().float().cpu().tolist()
        loss, numer, denom = self._masked_token_loss(flat, targets, objective)
        diagnostics = {"freeze_prefix": 0, "token_surprise": surprise,
                       "gradient_subset": hooks.gradient_subset}
        if denom == 0:
            success, reason = False, "no_active_tokens"
            del features, flat
        else:
            context = {"tokens": self._batch_identity(payload), "ids": training.batch.ids,
                       "origins": training.origins, "objective": asdict(objective), "prefix": 0,
                       "token_targets": fingerprint({"baseline": targets.baseline.detach().cpu().tolist(),
                                                     "weight": targets.weight.detach().cpu().tolist()})}
            success = self.optimizer_update(loss, revision_context=context)
            reason = None if success else "nonfinite_loss_or_gradients"
            del features, flat, loss
        cost = self._timing_cost(
            start, execution_mode=execution_mode, forward_passes=1, per_example_backward_observations=0,
            process_peak_cpu_bytes=(None if resource is None
                                     else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            layer_cost_kind="supplied" if self.settings.backward_costs else "parameter_estimate")
        return StepReport(training.batch.ids, success, self.steps, training.batch.valid_tokens,
                          {"supervised": numer}, {"supervised": denom}, diagnostics, cost, reason)

    def train(self, k, *, batch_factory, objective, hooks, limits):
        """Consume the supplied finite stream with no internal shuffling or replay decisions.
        Plan freezing before the actual backward, report supervised statistics separately from KL,
        and leave no gradients or live graphs at the checkpoint boundary after each attempt.
        token_mode all keeps that supervised loss. weighted and hinge use token targets.
        """
        reports, attempts, updates, tokens = [], 0, 0, 0
        hooks = hooks or UpdateHooks()
        model_boundary(objective.token_mode in {"all", "weighted", "hinge"}, "Unknown token mode")
        with self._execution(mode="training", advance=True) as execution_mode:
            for _ in range(k):
                for training in batch_factory():
                    if attempts >= limits.max_attempts or updates >= limits.max_steps or tokens + training.batch.valid_tokens > limits.max_tokens:
                        return reports
                    start = self._begin_timed_block()
                    payload = self._payload(training.batch)
                    if objective.token_mode != "all":
                        report = self._token_update(training, objective, hooks, payload, start, execution_mode)
                        attempts, updates, tokens = (attempts + 1, updates + int(report.success),
                                                     tokens + training.batch.valid_tokens)
                        reports.append(report)
                        continue
                    model_boundary(training.token_targets is None, "token_mode all does not take token targets")
                    self.apply_freeze(0)
                    statistics = self.feature_statistics(payload, objective, training.origins) if hooks.plan_update else {}
                    plan = hooks.plan_update(statistics) if hooks.plan_update else None
                    self.apply_freeze(plan.prefix if plan else 0)
                    features = self._features(payload)
                    means, sums, counts = self._losses(features, payload["labels"])
                    diagnostics = self.gradient_statistics(means, sums, counts, hooks, plan, objective, training.origins)
                    diagnostics.update(statistics, freeze_prefix=plan.prefix if plan else 0)
                    loss = self.supervised_loss(means, sums, counts, objective, training.origins)
                    if objective.kd_weight and training.teacher_targets is not None:
                        model_boundary(training.teacher_targets.ids == training.batch.ids, "Teacher IDs differ")
                        loss = loss + objective.kd_weight * self.distillation_loss(
                            features, payload, training.teacher_targets, objective, training.origins)
                    logged_sum = (means.sum() if objective.reduction == "per_example_mean" else sums.sum()).detach().item()
                    logged_count = len(means) if objective.reduction == "per_example_mean" else counts.sum().item()
                    revision_context = {"tokens": self._batch_identity(payload), "ids": training.batch.ids,
                                        "origins": training.origins, "objective": asdict(objective),
                                        "prefix": plan.prefix if plan else 0}
                    success = self.optimizer_update(loss, revision_context=revision_context)
                    attempts, updates, tokens = attempts + 1, updates + int(success), tokens + training.batch.valid_tokens
                    cost = self._timing_cost(start, execution_mode=execution_mode,
                                             forward_passes=1 + int(hooks.plan_update is not None),
                                             per_example_backward_observations=len(means) if hooks.gradient_subset or (plan and plan.refresh) else 0,
                                             process_peak_cpu_bytes=(None if resource is None
                                              else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                                             layer_cost_kind="supplied" if self.settings.backward_costs else "parameter_estimate")
                    reports.append(StepReport(training.batch.ids, success, self.steps, training.batch.valid_tokens,
                                              {"supervised": logged_sum}, {"supervised": logged_count}, diagnostics, cost,
                                              None if success else "nonfinite_loss_or_gradients"))
                    del features, means, sums, counts, loss
        return reports

    def evaluate(self, view, evaluation_spec, progress=None):
        """Evaluate caller-collated held-out records, preserving the learner's numerical state.
        Report token-weighted NLL/perplexity and optionally exact-match generation accuracy.
        text_metrics names functions in numerics and averages them over generated strings.
        The dataset retains ownership of response masks and prompt-only generation collation.
        """
        size = evaluation_spec.get("batch_size", self.settings.max_batch_size)
        metric_names = tuple(evaluation_spec.get("text_metrics") or ())
        if metric_names and "generation" not in evaluation_spec:
            raise ValueError("text_metrics require generation")
        total, count, correct = 0., 0, 0
        metric_totals = dict.fromkeys(metric_names, 0.0)
        predictions = []
        with self._execution() as execution_mode:
            timing_start = self._begin_timed_block()
            for start in range(0, len(view), size):
                records = view[start:start + size]
                batch = self.collate(records)
                payload = self._payload(batch)
                with torch.no_grad():
                    _, sums, counts = self._losses(self._features(payload), payload["labels"])
                total += sums.sum().item()
                count += counts.sum().item()
                if "generation" in evaluation_spec:
                    generated = self.predict(self.collate_prompts(records), evaluation_spec["generation"])
                    pairs = list(zip(generated["texts"], records))
                    correct += sum(text.strip() == record.target["text"].strip() for text, record in pairs)
                    for name in metric_names:
                        fn = getattr(numerics, name, None)
                        model_boundary(callable(fn) and not name.startswith("_"), f"Unknown text metric {name}")
                        metric_totals[name] += sum(float(fn(text, record.target["text"])) for text, record in pairs)
                    if evaluation_spec.get("return_predictions"):
                        predictions.extend({"id": record.sample_id,
                                            "messages": record.inputs["messages"],
                                            "target": record.target["text"], "prediction": text,
                                            "exact_match": text.strip() == record.target["text"].strip()}
                                           for text, record in pairs)
                if progress is not None:
                    progress(start + len(records), len(view))
        result = {"negative_nll": -total / count, "perplexity": math.exp(min(80., total / count))}
        if "generation" in evaluation_spec:
            result["exact_match"] = correct / len(view)
        for name, amount in metric_totals.items():
            result[name] = amount / len(view)
        result.update(self._timing_cost(timing_start, execution_mode=execution_mode))
        if evaluation_spec.get("return_predictions"):
            result["_predictions"] = predictions
        return result

    def _generation_payload(self, inputs):
        """Accept prompt-only Batch/dict inputs or conversations rendered with the fixed template."""
        if hasattr(inputs, "payload"):
            model_boundary("labels" not in inputs.payload, "Generation requires prompt-only collation")
            return inputs.ids, self.to_device(inputs.payload)
        if isinstance(inputs, dict):
            model_boundary("labels" not in inputs, "Generation requires prompt-only inputs")
            return (), self.to_device(inputs)
        conversations = [inputs] if inputs and isinstance(inputs[0], dict) else inputs
        encoded = [self.encode(self.render(messages)) for messages in conversations]
        width = max(map(len, encoded))
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        ids = torch.full((len(encoded), width), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, tokens in enumerate(encoded):
            ids[i, -len(tokens):] = torch.tensor(tokens, device=self.device)
            mask[i, -len(tokens):] = True
        return (), {"input_ids": ids, "attention_mask": mask}

    def predict(self, inputs, generation_spec):
        """Generate completions only, with an isolated decoding seed and non-thinking prompts."""
        with self._execution() as execution_mode, torch.no_grad():
            timing_start = self._begin_timed_block()
            ids, payload = self._generation_payload(inputs)
            model_boundary(payload["input_ids"].shape[0] * generation_spec.get("num_beams", 1) <= self.settings.max_batch_size,
                           "Generation batch exceeds configured working bounds")
            allowed = {"max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "num_beams"}
            options = {k: v for k, v in generation_spec.items() if k in allowed}
            options.setdefault("max_new_tokens", 128)
            options.setdefault("do_sample", False)
            model_boundary(payload["input_ids"].shape[1] + options["max_new_tokens"] <= self.settings.max_length,
                           "Generation exceeds configured sequence bound")
            seed = generation_spec.get("seed", self.settings.seed)
            torch.random.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            if self.device.type != "cpu":
                getattr(torch, self.device.type).set_rng_state(
                    torch.Generator(device=self.device).manual_seed(seed).get_state(), self.device)
            output = self.module.generate(**payload, **options,
                                          pad_token_id=self.preprocessing_kwargs()["pad_id"],
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          use_cache=True)
            tokens = output[:, payload["input_ids"].shape[1]:].cpu().tolist()
            return {"ids": ids, "texts": self.tokenizer.batch_decode(tokens, skip_special_tokens=True),
                    "token_ids": self._trim_generated(tokens),
                    "cost": self._timing_cost(timing_start, execution_mode=execution_mode)}

    def _trim_generated(self, rows):
        """Keep tokens through the first EOS and drop padding. Text decoding stays on the raw rows."""
        eos = self.tokenizer.eos_token_id
        pad = self.preprocessing_kwargs()["pad_id"]
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

    def _weights(self, *, changed_only=False):
        """Serialize canonical parameter names once, preserving tied-parameter identity on reload."""
        eligible = {n for n, _, train in self._profile(self.use_lora)[0] if train}
        return {"parameters": {n: p.detach().cpu() for n, p in self.module.named_parameters()
                               if not changed_only or not self.use_lora or n in eligible},
                "buffers": {n: b.detach().cpu() for n, b in self.module.named_buffers()}}

    def _install(self, weights, *, changed_only=False):
        """Copy tensors into existing Parameter objects so shared wrapper/optimizer references survive."""
        live = dict(self.module.named_parameters())
        eligible = {n for n, _, train in self._profile(self.use_lora)[0] if train}
        expected = eligible if changed_only and self.use_lora else set(live)
        buffers = dict(self.module.named_buffers())
        model_boundary(set(weights["parameters"]) == expected
                       and set(weights["buffers"]) == set(buffers)
                       and all(tuple(t.shape) == tuple(live[n].shape) for n, t in weights["parameters"].items())
                       and all(tuple(t.shape) == tuple(buffers[n].shape) for n, t in weights["buffers"].items()),
                       "Checkpoint tensor names or shapes differ")
        with torch.no_grad():
            for name, value in weights["parameters"].items():
                parameter = self.module.get_parameter(name)
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            for name, value in weights["buffers"].items():
                buffer = self.module.get_buffer(name)
                buffer.copy_(value.to(device=buffer.device, dtype=buffer.dtype))

    def get_weights(self):
        """Return independent CPU parameter/buffer copies with architecture compatibility identity."""
        with self._execution():
            weights = self._weights()
            return {"compatibility": self._compatibility(),
                    **{kind: {n: t.clone() for n, t in values.items()} for kind, values in weights.items()}}

    def set_weights(self, weights):
        """Install a compatible snapshot into this wrapper and start a fresh numerical training state."""
        self.activate()
        model_boundary(weights["compatibility"] == self._compatibility(), "Incompatible model weight layout")
        self._install(weights)
        self._weights_identity = tensor_fingerprint({k: weights[k] for k in ("parameters", "buffers")})
        self.initialization = "weights:" + self._weights_identity
        # Arbitrary external snapshots may change the nominally frozen base too.
        self._save_full_weights = True
        self.steps = 0
        self.weight_revision += 1
        self.rng_state = self._new_rng()
        self._reset_optimizer()

    def _metadata(self, kind):
        """Record compatibility, immutable asset hashes and the non-thinking policy."""
        return {"schema": 1, "state_kind": kind, "compatibility": self._compatibility(),
                "execution_identity": self.info.identity, "model": self.model_id, "revision": self.revision,
                "base_files": self._base_files, "base_bytes": self._base_bytes, "steps": self.steps, "weight_revision": self.weight_revision,
                "enable_thinking": False, "full_weights": self._save_full_weights,
                "initialization": self.initialization, "weights_identity": self._weights_identity}

    def _training_state(self):
        """Keep all state needed to continue the next update exactly, without storing graphs."""
        return {"optimizer": cpu_tree(self.optimizer.state_dict()), "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(), "rng": cpu_tree(self.rng_state), "steps": self.steps,
                "weight_revision": self.weight_revision, "weights_identity": self._weights_identity}

    def save(self, path, state_kind):
        """Save weights separately from numerical resume state, avoiding repeated immutable bases.
        Inactive roles copy their private disk state without loading optimizer tensors into RAM.
        The outer checkpoint manager owns atomic publication, checksums and rolling retention.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.module is None and self._offload is not None:
            source = Path(self._offload.name)
            saved = read_json(source / "metadata.json")
            if state_kind == "resume" and saved["state_kind"] != "resume":
                self.activate()
                return self.save(path, state_kind)
            shutil.copyfile(source / "weights.pt", path / "weights.pt")
            if state_kind == "resume":
                shutil.copyfile(source / "training.pt", path / "training.pt")
            write_json(path / "metadata.json", {**saved, "state_kind": state_kind})
        else:
            self.activate()
            torch.save(self._weights(changed_only=not self._save_full_weights), path / "weights.pt")
            if state_kind == "resume":
                torch.save(self._training_state(), path / "training.pt")
            write_json(path / "metadata.json", self._metadata(state_kind))

    def _read_state(self, path):
        """Restore a complete numerical state, or reset optimization when loading frozen weights."""
        metadata = read_json(path / "metadata.json")
        model_boundary(metadata["compatibility"] == self._compatibility(), "Incompatible model checkpoint")
        model_boundary(metadata["base_files"] == self._base_files, "Immutable base checksum mismatch")
        if metadata["state_kind"] == "resume":
            model_boundary(metadata["execution_identity"] == self.info.identity, "Resume settings differ")
        self._save_full_weights = metadata.get("full_weights", False)
        self._install(torch.load(path / "weights.pt", map_location="cpu", weights_only=True),
                      changed_only=not self._save_full_weights)
        if metadata["state_kind"] == "resume":
            state = torch.load(path / "training.pt", map_location="cpu", weights_only=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.scaler.load_state_dict(state["scaler"])
            self.rng_state = state["rng"]
            self.steps, self.weight_revision = state["steps"], state["weight_revision"]
            self._weights_identity = state["weights_identity"]
        else:
            self.steps, self.weight_revision = 0, metadata["weight_revision"]
            self._frozen_identity(metadata)
            self.rng_state = self._new_rng()
            self._reset_optimizer()

    def _frozen_identity(self, metadata):
        """Keep frozen inputs distinguishable while retaining independent teacher role identities."""
        self._weights_identity = fingerprint((metadata["execution_identity"], metadata["weights_identity"],
                                              metadata["steps"], metadata["weight_revision"]))
        if self.role == "learner":
            self.initialization = "weights:" + self._weights_identity

    def _reserve_storage(self, additional):
        """Account for shared base assets once, as well as run-owned staging/offload files."""
        root = Path(self.runtime_root or self.cache_dir / "standalone")
        root.mkdir(parents=True, exist_ok=True)
        if self.resources is not None:
            Checkpoints(root, self.info.identity, self.resources.disk_bytes,
                        external_bytes=self.base_storage_bytes).reserve(additional)

    def _offload_directory(self):
        """Charge staging files to the run before creating one private numerical snapshot."""
        root = Path(self.runtime_root or self.cache_dir / "standalone")
        root.mkdir(parents=True, exist_ok=True)
        self._reserve_storage(self.info.checkpoint_bytes)
        folder = root / ".offload"
        folder.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=folder, prefix=self.role + "-")

    def load(self, path, state_kind):
        """Validate source metadata; inactive roles keep a private copy safe from checkpoint pruning."""
        path = Path(path)
        metadata = read_json(path / "metadata.json")
        model_boundary(metadata["state_kind"] == state_kind, "Checkpoint kind differs")
        self._prepare()
        if state_kind == "resume":
            self.initialization = metadata["initialization"]
        else:
            self._frozen_identity(metadata)
        model_boundary(metadata["compatibility"] == self._compatibility(), "Incompatible model checkpoint")
        if self.module is not None:
            self._read_state(path)
        else:
            self._base_files = metadata["base_files"]
            self._base_bytes = metadata["base_bytes"]
            self._save_full_weights = metadata.get("full_weights", False)
            target = self._offload_directory()
            for name in ("weights.pt", "metadata.json", *(('training.pt',) if state_kind == "resume" else ())):
                shutil.copyfile(path / name, Path(target.name) / name)
            previous, self._offload = self._offload, target
            if previous is not None:
                previous.cleanup()
            self.steps = metadata["steps"] if state_kind == "resume" else 0
            self.weight_revision = metadata["weight_revision"]

    def offload(self):
        """Persist full numerical state before releasing all model/optimizer/device tensors.
        Keep the previous private snapshot until the replacement has been written successfully.
        Checkpoint publication remains responsible for durable recovery after a hard interruption.
        """
        if self.module is None:
            return
        self._compiled_forward = None
        target = self._offload_directory()
        self.save(target.name, "resume")
        previous, self._offload = self._offload, target
        self.module = self.optimizer = self.scheduler = self.scaler = None
        if previous is not None:
            previous.cleanup()
        gc.collect()
        if self.device.type != "cpu":
            getattr(torch, self.device.type).empty_cache()

    def release(self):
        """Drop an unneeded auxiliary and its private state; required frozen teachers live elsewhere."""
        self._compiled_forward = None
        self.module = self.optimizer = self.scheduler = self.scaler = None
        if self._offload is not None:
            self._offload.cleanup()
            self._offload = None
        gc.collect()
        if self.device.type != "cpu":
            getattr(torch, self.device.type).empty_cache()
