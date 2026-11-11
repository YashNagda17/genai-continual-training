"""Hindi-only LoRA probe for Qwen/Qwen3-0.6B.

Same protocol as scripts/probe_mbart25.py: score English and Hindi, train on Hindi
only, score again. Articles are packed with the Qwen chat template. The prepared
mT5 ids are not reused. This script does not replace the mT5 experiment runner.

From the repository root, with the project venv active:

  source ~/venv/bin/activate
  python scripts/probe_qwen.py --eval-only
  python scripts/probe_qwen.py
"""

import os

os.environ.setdefault("EventHostSynchronizeLinuxUserFenceKmdWait", "1")

from pathlib import Path
import argparse
import json
import math
import random
import sys
import time

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.dataset_instruction import SummarizationDataset
from src.utils.numerics import chrf, resolve_device, rouge_l, script_share

MODEL_ID = "Qwen/Qwen3-0.6B"
PARAMETER_CAP = 700_000_000
PREPARED = ROOT / "results/code/summarization-passes5/prepared"
INSTRUCTION = "Summarize the article.\n\n"
GENERATION = {"max_new_tokens": 64, "no_repeat_ngram_size": 3, "stop_on_eos": True}
PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


class CausalLoRA(nn.Module):
    """Frozen projection plus (alpha / rank) B A, with B starting at zero."""

    def __init__(self, base, rank, alpha, seed):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        generator = torch.Generator(device=base.weight.device).manual_seed(seed)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5), generator=generator)
        self.base.requires_grad_(False)

    def forward(self, inputs):
        result = self.base(inputs)
        shape = result.shape
        hidden = F.linear(inputs, self.A).reshape(-1, self.A.shape[0])
        result = torch.addmm(result.reshape(-1, shape[-1]), hidden, self.B.T, alpha=self.scale)
        return result.reshape(shape)


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


def load_dataset(prepared):
    dataset = SummarizationDataset()
    dataset.load({"path": str(prepared)})
    return dataset


def select(dataset, key, count):
    rows = dataset.views[key]
    if len(rows) < count:
        raise ValueError(f"{key} has {len(rows)} rows, needed {count}")
    return list(rows[:count])


def chat_ids(tokenizer, messages, add_generation_prompt):
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=False)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def prompt_ids(tokenizer, article, max_source):
    """User turn plus the closed think block, trimmed so the prompt fits max_source."""
    body = tokenizer.encode(str(article), add_special_tokens=False)[:max_source]
    ids = []
    while True:
        text = tokenizer.decode(body, skip_special_tokens=True)
        ids = chat_ids(tokenizer, [{"role": "user", "content": INSTRUCTION + text}], True)
        if len(ids) <= max_source or not body:
            return ids[:max_source]
        body = body[:max(0, len(body) - (len(ids) - max_source))]


def retokenize(records, tokenizer, max_source, max_target):
    # Qwen has no unk id. <|im_end|> is the eos token, so the summary is trained to stop there.
    end_id = int(tokenizer.eos_token_id)
    chosen = []
    for record in records:
        prompt = prompt_ids(tokenizer, record.inputs["text"], max_source)
        summary = tokenizer.encode(str(record.target["text"]), add_special_tokens=False)[:max_target]
        chosen.append({
            "id": record.sample_id,
            "article": record.inputs["text"],
            "target": record.target["text"],
            "prompt": prompt,
            "input_ids": prompt + summary + [end_id],
            "labels": [-100] * len(prompt) + summary + [end_id],
        })
    return chosen


def collate(rows, pad_id, *, prompts=False):
    """Right-pad training rows so position 0 is the first real token. Left-pad prompts for generation."""
    sequences = [row["prompt"] if prompts else row["input_ids"] for row in rows]
    width = max(len(seq) for seq in sequences)
    ids, labels, mask = [], [], []
    for row, seq in zip(rows, sequences):
        pad = width - len(seq)
        if prompts:
            ids.append([pad_id] * pad + seq)
            labels.append([-100] * width)
            mask.append([0] * pad + [1] * len(seq))
        else:
            ids.append(seq + [pad_id] * pad)
            labels.append(row["labels"] + [-100] * pad)
            mask.append([1] * len(seq) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def chunk_size_for(logits):
    """Grow the cross-entropy slice until the fp32 scratch is about 256 MiB."""
    vocab = logits.shape[-1]
    rows = max(1, logits.numel() // vocab)
    scratch_rows = max(1, (256 * 1024 ** 2) // max(vocab * 4, 1))
    return max(32, min(rows, scratch_rows))


def position_nll(logits, labels):
    return _ChunkedPositionNLL.apply(logits, labels, chunk_size_for(logits))


def batch_loss(model, batch):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    use_cache=False, return_dict=True)
    labels = batch["labels"][:, 1:]
    position = position_nll(outputs.logits[:, :-1], labels)
    valid = labels != -100
    counts = valid.sum(-1).clamp_min(1)
    return (position.sum(-1) / counts).mean()


def attach_lora(module, rank, alpha):
    attached = 0
    for name, child in list(module.named_modules()):
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in PROJECTIONS or not isinstance(child, nn.Linear):
            continue
        parent_name, attribute = name.rsplit(".", 1)
        parent = module.get_submodule(parent_name)
        seed = random.Random(f"0:{name}").getrandbits(63)
        setattr(parent, attribute, CausalLoRA(child, rank, alpha, seed))
        attached += 1
    return attached


def parameter_counts(module):
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable


def generate(model, batch, eos_id, pad_id):
    """Greedy continuation. Left padding keeps the last logit on a real token."""
    prompt_width = batch["input_ids"].shape[1]
    output = model.generate(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        max_new_tokens=GENERATION["max_new_tokens"], do_sample=False,
        no_repeat_ngram_size=GENERATION["no_repeat_ngram_size"],
        eos_token_id=eos_id, pad_token_id=pad_id, use_cache=True)
    return output[:, prompt_width:]


def score_rows(model, tokenizer, rows, batch_size, pad_id, eos_id):
    model.eval()
    scored = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch = move(collate(chunk, pad_id, prompts=True), model.device)
        with torch.inference_mode():
            token_ids = generate(model, batch, eos_id, pad_id)
        texts = tokenizer.batch_decode(token_ids.tolist(), skip_special_tokens=True)
        for row, text in zip(chunk, texts):
            text = text.strip()
            scored.append({
                "id": row["id"], "prediction": text, "target": row["target"],
                "article": row["article"], "chrf": chrf(text, row["target"]),
                "rouge_l": 100 * rouge_l(text, row["target"]),
                "devanagari_share": script_share(text, "devanagari"),
            })
    return scored


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def score_both(model, tokenizer, rows, batch_size, pad_id, eos_id):
    scored = {}
    for language in ("english", "hindi"):
        scored[language] = score_rows(model, tokenizer, rows[language], batch_size, pad_id, eos_id)
        print(f"  {language:8}  chrF {mean(scored[language], 'chrf'):6.2f}   "
              f"ROUGE-L {mean(scored[language], 'rouge_l'):6.2f}   "
              f"Devanagari letters {100 * mean(scored[language], 'devanagari_share'):5.1f}%")
    return scored


def train_hindi(model, rows, batch_size, epochs, learning_rate, pad_id):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    steps = (len(rows) + batch_size - 1) // batch_size
    losses = []
    synchronize(model.device)
    started = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        for step, start in enumerate(range(0, len(rows), batch_size), start=1):
            batch = move(collate(rows[start:start + batch_size], pad_id), model.device)
            loss = batch_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            value = float(loss.detach())
            losses.append(value)
            print(f"  epoch {epoch}/{epochs}  step {step:4d}/{steps}  loss {value:.4f}")
    synchronize(model.device)
    return {"steps": len(losses), "epochs": epochs, "examples": len(rows), "batch_size": batch_size,
            "loss_first": losses[0], "loss_last": losses[-1], "seconds": time.perf_counter() - started}


def show_examples(title, scored, count):
    print(f"\n{title}")
    for language in ("english", "hindi"):
        for row in scored[language][:count]:
            print(f"\n[{language}] {row['id']}  chrF {row['chrf']:.1f}")
            print(f"  article:    {row['article'][:240].replace(chr(10), ' ')}")
            print(f"  reference:  {row['target'][:240].replace(chr(10), ' ')}")
            print(f"  predicted:  {row['prediction'][:240].replace(chr(10), ' ')}")


def write_report(path, payload):
    path.mkdir(parents=True, exist_ok=True)
    (path / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    before, after = payload["before"], payload.get("after")
    lines = [
        "# Qwen3-0.6B summarization probe",
        "",
        f"Model: `{payload['model']}`",
        f"Parameters: {payload['parameters']:,} total, {payload['trainable']:,} trainable",
        f"Articles per language: {payload['eval_n']}. Chat instruction: `{payload['instruction'].strip()}`. "
        "Thinking is off.",
        "",
        "Qwen3-0.6B is a tied-embedding causal model. Summaries are trained only on the tokens",
        "after the user turn. Scores are this project's chrF and whitespace ROUGE-L.",
        "",
        "| | English chrF | English ROUGE-L | Hindi chrF | Hindi ROUGE-L |",
        "|---|---:|---:|---:|---:|",
        f"| Before Hindi | {before['english']['chrf']:.2f} | {before['english']['rouge_l']:.2f} | "
        f"{before['hindi']['chrf']:.2f} | {before['hindi']['rouge_l']:.2f} |",
    ]
    if after:
        train = payload["hindi_run"]
        lines.append(
            f"| After {train['examples']} Hindi examples | {after['english']['chrf']:.2f} | "
            f"{after['english']['rouge_l']:.2f} | {after['hindi']['chrf']:.2f} | {after['hindi']['rouge_l']:.2f} |")
        lines.extend(["",
                      f"English chrF change: {after['english']['chrf'] - before['english']['chrf']:+.2f}. "
                      f"Hindi chrF change: {after['hindi']['chrf'] - before['hindi']['chrf']:+.2f}.",
                      f"Hindi loss {train['loss_first']:.3f} to {train['loss_last']:.3f} "
                      f"in {train['seconds'] / 60:.1f} minutes, {train['epochs']} epochs, "
                      f"batch {train['batch_size']}. No English replay."])
    (path / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_adapter(model, path):
    path.mkdir(parents=True, exist_ok=True)
    state = {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
             if name.endswith((".A", ".B"))}
    torch.save({"model": MODEL_ID, "adapters": state}, path / "adapter.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--output", type=Path, default=ROOT / "results/code/qwen-probe")
    parser.add_argument("--eval-n", type=int, default=8)
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--train-rows", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--max-source", type=int, default=256)
    parser.add_argument("--max-target", type=int, default=64)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.prepared)
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=hub)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=hub, dtype=torch.float32, attn_implementation="sdpa")
    device = resolve_device(args.device)
    model.to(device)
    model.requires_grad_(False)
    if not args.eval_only:
        attached = attach_lora(model, args.rank, args.alpha)
        if attached == 0:
            raise RuntimeError("No LoRA parameters were attached")
    total, trainable = parameter_counts(model)
    if total > PARAMETER_CAP:
        raise RuntimeError(f"{args.model} has {total:,} parameters, above the {PARAMETER_CAP:,} cap")
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    eos_id = int(tokenizer.eos_token_id)
    rows = {
        "english": retokenize(select(dataset, "english/validation", args.eval_n), tokenizer,
                              args.max_source, args.max_target),
        "hindi": retokenize(select(dataset, "hindi/validation", args.eval_n), tokenizer,
                            args.max_source, args.max_target),
    }
    if not args.eval_only:
        rows["hindi_train"] = retokenize(
            select(dataset, "hindi/train", args.train_rows), tokenizer, args.max_source, args.max_target)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Device {device}, float32. {total:,} parameters, {trainable:,} trainable. "
          f"Chunked loss, batch {args.batch_size}.")

    print(f"\nBefore Hindi, {args.eval_n} validation articles per language")
    before = score_both(model, tokenizer, rows, args.batch_size, pad_id, eos_id)
    show_examples("Sample summaries before Hindi", before, args.show)
    training = None
    after = None
    if not args.eval_only:
        print(f"\nHindi-only LoRA: {args.train_rows} examples, {args.epochs} epochs, "
              f"batch {args.batch_size}, lr {args.lr}")
        training = train_hindi(model, rows["hindi_train"], args.batch_size, args.epochs, args.lr, pad_id)
        print("\nAfter Hindi")
        after = score_both(model, tokenizer, rows, args.batch_size, pad_id, eos_id)
        show_examples("Sample summaries after Hindi", after, args.show)
        print(f"\nEnglish chrF change {mean(after['english'], 'chrf') - mean(before['english'], 'chrf'):+.2f}. "
              f"Hindi chrF change {mean(after['hindi'], 'chrf') - mean(before['hindi'], 'chrf'):+.2f}.")
        save_adapter(model, args.output / "hindi")

    def pack(scored):
        return {language: {"chrf": mean(rows_for_language, "chrf"),
                           "rouge_l": mean(rows_for_language, "rouge_l"),
                           "devanagari_share": mean(rows_for_language, "devanagari_share"),
                           "rows": rows_for_language}
                for language, rows_for_language in scored.items()}

    write_report(args.output, {
        "model": args.model, "parameters": total, "trainable": trainable,
        "instruction": INSTRUCTION, "enable_thinking": False,
        "eval_n": args.eval_n, "device": str(device), "dtype": "float32",
        "before": pack(before), "after": None if after is None else pack(after), "hindi_run": training,
    })
    print(f"\nWrote {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
