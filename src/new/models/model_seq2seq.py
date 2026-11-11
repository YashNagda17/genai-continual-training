"""Encoder-decoder LoRA in native Torch: training, scoring, probe gradients, greedy writing, shrink and merge.

Transformers supplies the architecture and base weights. Base weights stay frozen in fp32."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import gc
import math
import random
import re

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from transformers import AutoTokenizer, MT5ForConditionalGeneration

from ..utils.numerics import read_json, resolve_device, write_json

WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "model_weights"
PROJECTIONS = {"q", "k", "v", "o", "wi_0", "wi_1", "wo"}
PARTS = {"SelfAttention": "self_attention", "EncDecAttention": "cross_attention", "DenseReluDense": "feed_forward"}
LAYOUT = ("rank", "alpha", "new_rank", "trainable", "lr_scale")


@dataclass(frozen=True)
class Seq2SeqSettings:
    rank: int = 16
    alpha: float = 32.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    seed: int = 0
    max_source_length: int = 256
    max_target_length: int = 64
    logit_chunk_size: int = 32  # Target positions per cross-entropy call over the 250k vocabulary


@lru_cache(maxsize=None)
def load_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id, cache_dir=WEIGHTS)


def seed_for(seed, name):
    return random.Random(f"{seed}:{name}").getrandbits(63)


def init_A(parameter, seed):
    generator = torch.Generator(device=parameter.device).manual_seed(seed)
    nn.init.kaiming_uniform_(parameter, a=math.sqrt(5), generator=generator)


class ChunkedNLL(torch.autograd.Function):
    """Per-position cross-entropy with one logits-gradient buffer instead of one per slice."""

    @staticmethod
    def forward(ctx, logits, labels, chunk):
        flat, targets = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        losses = torch.empty_like(targets, dtype=torch.float32)
        for start in range(0, targets.numel(), chunk):
            losses[start:start + chunk] = F.cross_entropy(flat[start:start + chunk].float(),
                                                          targets[start:start + chunk], reduction="none")
        ctx.save_for_backward(flat, targets)
        ctx.chunk, ctx.shape = chunk, logits.shape
        return losses.view_as(labels)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        flat, targets = ctx.saved_tensors
        upstream, gradient = upstream.reshape(-1), torch.empty_like(flat)
        for start in range(0, targets.numel(), ctx.chunk):
            stop = start + ctx.chunk
            valid = targets[start:stop] != -100
            probabilities = F.softmax(flat[start:stop].float(), dim=-1)
            probabilities.scatter_add_(1, targets[start:stop].masked_fill(~valid, 0).unsqueeze(1),
                                       -valid.float().unsqueeze(1))
            gradient[start:stop] = probabilities * (upstream[start:stop].float() * valid).unsqueeze(1)
        return gradient.view(ctx.shape), None, None


class ChunkedKL(torch.autograd.Function):
    """Per-position KL(teacher || student) at temperature T, zero at padding, in vocabulary-sized chunks."""

    @staticmethod
    def forward(ctx, logits, teacher, mask, temperature, chunk):
        flat, target, valid = logits.reshape(-1, logits.shape[-1]), teacher.reshape(-1, teacher.shape[-1]), mask.reshape(-1)
        values = torch.zeros(valid.shape, dtype=torch.float32, device=logits.device)
        for start in range(0, valid.numel(), chunk):
            p = F.log_softmax(target[start:start + chunk].float() / temperature, -1)
            q = F.log_softmax(flat[start:start + chunk].float() / temperature, -1)
            values[start:start + chunk] = (p.exp() * (p - q)).sum(-1) * valid[start:start + chunk]
        ctx.save_for_backward(flat, target, valid)
        ctx.temperature, ctx.chunk, ctx.shape = temperature, chunk, logits.shape
        return values.view(mask.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        flat, target, valid = ctx.saved_tensors
        upstream, gradient = upstream.reshape(-1), torch.empty_like(flat)
        for start in range(0, valid.numel(), ctx.chunk):
            stop = start + ctx.chunk
            difference = (F.softmax(flat[start:stop].float() / ctx.temperature, -1)
                          - F.softmax(target[start:stop].float() / ctx.temperature, -1))
            gradient[start:stop] = difference * (upstream[start:stop] * valid[start:stop] / ctx.temperature).unsqueeze(1)
        return gradient.view(ctx.shape), None, None, None, None


class LoRALinear(nn.Module):
    """W x + scale B A x, with B starting at zero. E's extra rank has its own factors at scale 2."""

    def __init__(self, base, rank, scale, block, cell, seed):
        super().__init__()
        self.base, self.block, self.cell, self.scale, self.extra_scale = base, block, cell, scale, 0.0
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        init_A(self.A, seed)

    @property
    def weight(self):
        """The frozen projection; the feed-forward reads its dtype."""
        return self.base.weight

    def add_extra(self, rank, seed):
        self.extra_A = nn.Parameter(self.base.weight.new_empty(rank, self.base.in_features))
        self.extra_B = nn.Parameter(self.base.weight.new_zeros(self.base.out_features, rank))
        init_A(self.extra_A, seed)
        self.extra_scale = 2.0

    def clear_extra(self):
        self._parameters.pop("extra_A", None)
        self._parameters.pop("extra_B", None)
        self.extra_scale = 0.0

    def forward(self, x):
        out = self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)
        if self.extra_scale:
            out = out + self.extra_scale * F.linear(F.linear(x, self.extra_A), self.extra_B)
        return out

    def delta(self):
        """Effective update Delta W in float32, including any extra rank."""
        value = self.scale * (self.B.float() @ self.A.float())
        if self.extra_scale:
            value = value + self.extra_scale * (self.extra_B.float() @ self.extra_A.float())
        return value


def detach_output(_module, _inputs, output):
    """Stop the backward pass at this output. Values are unchanged."""
    if torch.is_tensor(output):
        return output.detach()
    if isinstance(output, tuple):
        return tuple(item.detach() if torch.is_tensor(item) else item for item in output)
    for key, value in output.items():
        if torch.is_tensor(value):
            output[key] = value.detach()
    return output


def block_repeats(generated, scores, size):
    """Forbid any token that would repeat an n-gram already written. Stays on the device."""
    if size < 1 or generated.shape[1] < size:
        return scores
    windows = generated.unfold(1, size, 1)
    matched = (windows[:, :, :-1] == generated[:, generated.shape[1] - size + 1:][:, None, :]).all(dim=-1)
    penalty = torch.zeros(matched.shape, dtype=scores.dtype, device=scores.device).masked_fill_(matched, -math.inf)
    return scores.scatter_add_(1, windows[:, :, -1], penalty)


class Seq2SeqLMModel:
    model_id = ""

    def __init__(self, settings=None, device="auto"):
        self.settings = settings or Seq2SeqSettings()
        self.device = resolve_device(device)
        self.tokenizer = load_tokenizer(self.model_id)
        self.layout = {"rank": self.settings.rank, "alpha": self.settings.alpha, "new_rank": {},
                       "trainable": None, "lr_scale": {}}
        self.module = self.optimizer = self.scheduler = self._scripts = None
        self.steps, self.hooks = 0, []

    def adapters(self):
        return [(name, m) for name, m in self.module.named_modules() if isinstance(m, LoRALinear)]

    def trainable_parameters(self):
        return sum(p.numel() for p in self.module.parameters() if p.requires_grad)

    def build(self, layout=None):
        """Frozen base weights plus LoRA on every projection of all 16 blocks, in the given layout."""
        self.release()
        self.layout = dict(layout or self.layout)
        module = MT5ForConditionalGeneration.from_pretrained(self.model_id, cache_dir=WEIGHTS,
                                                             attn_implementation="eager")
        assert sum(p.numel() for p in module.parameters()) <= 700_000_000
        module.requires_grad_(False)
        scale = self.layout["alpha"] / self.layout["rank"]
        for name, child in list(module.named_modules()):
            pieces = name.split(".")
            if pieces[-1] in PROJECTIONS and pieces[-2] in PARTS and pieces[1:2] == ["block"]:
                block = int(pieces[2])
                cell = f"{pieces[0]}.{PARTS[pieces[-2]]}.{block // 2 * 2 + 1}-{block // 2 * 2 + 2}"
                parent, attribute = name.rsplit(".", 1)
                setattr(module.get_submodule(parent), attribute, LoRALinear(
                    child, self.layout["rank"], scale, f"{pieces[0]}.{block + 1}", cell, seed_for(self.settings.seed, name)))
        self.module = module.to(self.device)
        self.module.config.use_cache = False
        self.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self._install_layout()
        self.steps, self.pristine_base = 0, True

    def apply_plan(self, plan):
        """C2/D/E: which shared adapters train, where extra rank goes, and per-block learning rates."""
        self.layout.update(trainable=sorted(plan.trainable), new_rank=dict(plan.new_rank), lr_scale=dict(plan.lr_scale))
        self._install_layout()

    def _install_layout(self):
        trainable = self.layout["trainable"]
        for name, m in self.adapters():
            if m.block in self.layout["new_rank"] and not m.extra_scale:
                m.add_extra(int(self.layout["new_rank"][m.block]), seed_for(self.settings.seed, "extra:" + name))
            m.A.requires_grad_(trainable is None or m.block in trainable)
            m.B.requires_grad_(trainable is None or m.block in trainable)
        self._cut_backward()
        self._reset_optimizer()

    def _cut_backward(self):
        """Detach the block below the lowest trainable one, so the backward pass stops there (C2's cost)."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        if self.layout["trainable"] is None:
            return
        for stack in ("encoder", "decoder"):
            indexes = [int(m.block.split(".")[1]) - 1 for _, m in self.adapters()
                       if m.block.startswith(stack) and any(p.requires_grad for p in m.parameters())]
            if indexes and min(indexes) == 0:
                continue
            target = getattr(self.module, stack)
            target = target.block[min(indexes) - 1] if indexes else target
            self.hooks.append(target.register_forward_hook(detach_output))

    def _reset_optimizer(self):
        groups = {}
        for _, m in self.adapters():
            scale = float(self.layout["lr_scale"].get(m.block, 1.0))
            for key, parameter in m._parameters.items():
                if parameter.requires_grad:
                    groups.setdefault(1.0 if key.startswith("extra") else scale, []).append(parameter)
        lr, warmup = self.settings.learning_rate, self.settings.warmup_steps
        self.optimizer = torch.optim.AdamW([{"params": p, "lr": lr * s} for s, p in sorted(groups.items())],
                                           lr=lr, weight_decay=self.settings.weight_decay) if groups else None
        self.scheduler = self.optimizer and torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0)

    def _logits(self, batch):
        batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
        labels = batch["labels"]
        return self.module(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           decoder_input_ids=self.module._shift_right(labels), use_cache=False).logits, labels

    def _losses(self, batch, with_logits=False):
        """Per-position NLL (zero at padding) and the supervised-token mask."""
        logits, labels = self._logits(batch)
        nll = ChunkedNLL.apply(logits, labels, self.settings.logit_chunk_size)
        return (nll, labels != -100, logits) if with_logits else (nll, labels != -100)

    @torch.no_grad()
    def teacher_logits(self, batch):
        """A frozen teacher's logits at the same teacher-forced positions (TSR)."""
        self.module.eval()
        return self._logits(batch)[0]

    def train_step(self, batch, token_weights=None, row_weights=None, teacher=None, kd_weight=0.0, kd_temperature=2.0):
        """One AdamW update. Each row's loss is its mean over summary tokens; rows are averaged, or combined with
        `row_weights` (CSReL's separate current and replay means). `token_weights(nll, mask)` reweights tokens (G3).
        A `teacher` adds kd_weight T^2 KL(teacher || student) at temperature T per token (TSR)."""
        targets = teacher.teacher_logits(batch) if teacher else None
        self.module.train()
        nll, mask, logits = self._losses(batch, with_logits=True)
        weights = token_weights(nll.detach(), mask) if token_weights else mask.float()
        per_token = nll * weights
        if teacher:
            per_token = per_token + kd_weight * kd_temperature ** 2 * ChunkedKL.apply(
                logits, targets, mask, kd_temperature, self.settings.logit_chunk_size)
        rows = per_token.sum(-1) / mask.sum(-1).clamp_min(1)
        loss = (rows * row_weights.to(rows.device)).sum() if row_weights is not None else rows.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in self.optimizer.param_groups for p in g["params"]],
                                       self.settings.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.steps += 1
        return loss.detach()

    @torch.no_grad()
    def token_losses(self, batch):
        """Each row's supervised-token NLL, as lists."""
        self.module.eval()
        nll, mask = self._losses(batch)
        return [row[:count] for row, count in zip(nll.float().cpu().tolist(), mask.sum(-1).tolist())]

    def reset_probe(self):
        """C1: fold the loaded adapter into the base, then put a fresh zero-B adapter everywhere."""
        self.pristine_base = False
        with torch.no_grad():
            for name, m in self.adapters():
                m.base.weight += m.delta()
                m.clear_extra()
                init_A(m.A, seed_for(self.settings.seed, name))
                m.B.zero_()
        self.layout.update(new_rank={}, trainable=None, lr_scale={})
        self._install_layout()

    def probe_gradients(self, batch):
        """Batch-mean gradient of the loss for every B, flattened, by adapter name."""
        self.module.eval()
        nll, mask = self._losses(batch)
        adapters = self.adapters()
        gradients = torch.autograd.grad((nll.sum(-1) / mask.sum(-1)).mean(), [m.B for _, m in adapters])
        return {name: gradient.flatten() for (name, _), gradient in zip(adapters, gradients)}

    def _script_ids(self):
        if self._scripts is None:
            pieces = self.tokenizer.convert_ids_to_tokens(list(range(len(self.tokenizer))))
            special = set(self.tokenizer.all_special_ids)

            def ids(test, skip_special=True):
                return torch.tensor([i for i, piece in enumerate(pieces) if piece and test(piece)
                                     and not (skip_special and i in special)], dtype=torch.long, device=self.device)
            self._scripts = {"sentinel": ids(lambda p: re.fullmatch(r"▁?<extra_id_\d+>", p), False),
                             "devanagari": ids(lambda p: any("ऀ" <= c <= "ॿ" for c in p)),
                             "latin": ids(lambda p: any(c.isascii() and c.isalpha() for c in p))}
        return self._scripts

    @torch.no_grad()
    def generate(self, batch, max_new=64, no_repeat=3, guard=None):
        """Greedy summaries with a KV cache. mT5's sentinels are blocked. The G5 guard blocks Devanagari
        for English articles (`guard="english"`) or lowers Latin tokens by 2 for Hindi ones."""
        self.module.eval()
        scripts = self._script_ids()
        ids, mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)
        encoded = self.module.encoder(input_ids=ids, attention_mask=mask)
        eos, pad = self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        out = torch.full((len(ids), max_new + 1), self.module.config.decoder_start_token_id, device=self.device)
        alive, past = torch.ones(len(ids), dtype=torch.bool, device=self.device), None
        for step in range(max_new):
            result = self.module(attention_mask=mask, encoder_outputs=encoded, decoder_input_ids=out[:, step:step + 1],
                                 past_key_values=past, use_cache=True)
            past, scores = result.past_key_values, result.logits[:, -1].float()
            scores[:, scripts["sentinel"]] = -math.inf
            scores = block_repeats(out[:, :step + 1], scores, no_repeat)
            if guard == "english":
                scores[:, scripts["devanagari"]] = -math.inf
            elif guard == "hindi":
                scores[:, scripts["latin"]] -= 2.0
            chosen = torch.where(alive, scores.argmax(-1), pad)
            out[:, step + 1] = chosen
            alive &= chosen != eos
            if (step + 1) % 4 == 0 and not alive.any():  # One host check every four steps
                out = out[:, :step + 2]
                break
        rows = []
        for row in out[:, 1:].tolist():
            kept = []
            for token in row:
                if token == pad:
                    break
                kept.append(token)
                if token == eos:
                    break
            rows.append(kept)
        return self.tokenizer.batch_decode(rows, skip_special_tokens=True), rows

    @torch.no_grad()
    def deltas(self):
        return {name: m.delta() for name, m in self.adapters()}

    @torch.no_grad()
    def shrink(self, k):
        """F: replace each adapter by the top-k SVD factors of its Delta W, at scale 1."""
        self.pristine_base = False
        for _, m in self.adapters():
            U, S, Vh = torch.linalg.svd(m.delta(), full_matrices=False)
            root = S[:k].sqrt()
            m.A, m.B = nn.Parameter(root[:, None] * Vh[:k]), nn.Parameter(U[:, :k] * root)
            m.scale = 1.0
            m.clear_extra()
        self.layout.update(rank=k, alpha=float(k), new_rank={})

    @torch.no_grad()
    def merge(self):
        """Serve W + Delta W with plain projections, so writing costs nothing extra per token."""
        self.pristine_base = False
        for name, m in self.adapters():
            m.base.weight += m.delta()
            parent, attribute = name.rsplit(".", 1)
            setattr(self.module.get_submodule(parent), attribute, m.base)
        self.optimizer = None

    def save(self, path, record=None, training=False):
        """Adapter factors and layout; `training` adds optimizer, scheduler, steps and RNG for resume."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({name: p.detach().cpu() for name, p in self.module.named_parameters()
                    if name.rsplit(".", 1)[-1] in ("A", "B", "extra_A", "extra_B")}, path / "weights.pt")
        if training:
            torch.save({"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                        "steps": self.steps, "rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None},
                       path / "training.pt")
        write_json(path / "metadata.json", {**self.layout, "model": self.model_id, "record": record})

    def load(self, path, training=False):
        """Rebuild for the saved layout, unless only the adapter weights differ (adapter swap)."""
        path = Path(path)
        metadata = read_json(path / "metadata.json")
        layout = {key: metadata[key] for key in LAYOUT}
        if self.module is None or not self.pristine_base or layout != self.layout or training:
            self.build(layout)
        parameters = dict(self.module.named_parameters())
        with torch.no_grad():
            for name, value in torch.load(path / "weights.pt", map_location=self.device).items():
                parameters[name].copy_(value)
        if training:
            state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.steps = state["steps"]
            torch.set_rng_state(state["rng"])
            if state["cuda_rng"] is not None:
                torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        return metadata

    def release(self):
        for handle in self.hooks:
            handle.remove()
        self.module = self.optimizer = self.scheduler = None
        self.hooks = []
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
