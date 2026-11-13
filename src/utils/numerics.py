from collections import Counter
import sys
import unicodedata

import torch

_cpu_notices = set()


def _notice_cpu(reason):
    """Print each distinct CPU placement once so an accelerator miss is visible."""
    if reason in _cpu_notices:
        return
    _cpu_notices.add(reason)
    print(f"Using CPU: {reason}", file=sys.stderr, flush=True)


def resolve_device(preferred="auto"):
    """Route CUDA/ROCm, then Intel XPU, with CPU as the available-device fallback.
    Explicit CPU avoids accelerator probing; indexed requests keep a valid device index.
    CPU results are printed so a fallback is not silent.
    """
    if preferred is None or str(preferred) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu:0")
        _notice_cpu("no CUDA or XPU is available. Pass device=\"cuda\" or device=\"xpu\" "
                    "when an accelerator is present.")
        return torch.device("cpu")
    device = torch.device(preferred)
    if device.type == "cpu":
        _notice_cpu("device was set to cpu. Pass device=\"cuda\" or device=\"xpu\" " "to run on an accelerator.")
        return device
    backend = getattr(torch, device.type, None)
    if (device.type in {"cuda", "xpu"} and backend is not None
            and backend.is_available()
            and (device.index is None or device.index < backend.device_count())):
        return device
    _notice_cpu(f"requested {preferred} is not available. Pass device=\"cuda\" or device=\"xpu\".")
    return torch.device("cpu")


def move_to_device(value, device):
    """Move nested batch/optimizer tensors without casting token IDs or detaching gradients."""
    if isinstance(value, torch.Tensor):
        # A copy from ordinary (pageable) host memory waits for every queued kernel first.
        # Pinned memory lets the copy queue behind them while the host moves on.
        non_blocking = value.device.type == "cpu" and torch.device(device).type in {"cuda", "xpu"}
        if non_blocking and not value.is_pinned():
            value = value.pin_memory()
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def tensor(values, device=None):
    """Detach policy inputs and keep tensor inputs on their device unless explicitly moved.
    Use float32 so squared gradients stay in the same precision as the model.
    Python sequences default to CPU; callers with a model pass its resolved device.
    """
    if isinstance(values, torch.Tensor):
        return values.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(values, dtype=torch.float32, device=device if device is not None else "cpu")


def unit_vectors(values):
    """Normalize along the final axis; zero vectors remain zero.
    Scale by the largest coordinate first to avoid overflow in squared gradient norms.
    """
    values = tensor(values)
    scale = values.abs().amax(dim=-1, keepdim=True)
    scaled = values / torch.where(scale > 0, scale, torch.ones_like(scale))
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    return scaled / torch.where(norm > 0, norm, torch.ones_like(norm))


def cosine_tensor(a, b):
    """Compute signed cosine on the input device, including broadcasted rows of vectors."""
    a = tensor(a)
    b = tensor(b, a.device)
    return (unit_vectors(a) * unit_vectors(b)).sum(dim=-1)


def cosine(a, b):
    """Return one signed cosine scalar; a zero vector contributes zero."""
    return cosine_tensor(a, b).item()


def _as_tokens(value):
    if isinstance(value, str):
        return value.split()
    if isinstance(value, torch.Tensor):
        return value.detach().reshape(-1).tolist()
    return list(value)


def _pair_device(left, right, device):
    if device is not None:
        return torch.device(device)
    for value in (left, right):
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def rouge_l(prediction, reference, device=None):
    """Unofficial ROUGE-L F1 over whitespace tokens, with no stemming or synonym table.
    Both-empty strings score 1.
    """
    device = _pair_device(prediction, reference, device)
    left, right = _as_tokens(prediction), _as_tokens(reference)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = torch.zeros(len(right) + 1, device=device, dtype=torch.float32)
    for item in left:
        current = previous.new_zeros(len(right) + 1)
        for j, other in enumerate(right, start=1):
            current[j] = previous[j - 1] + 1 if item == other else torch.maximum(previous[j], current[j - 1])
        previous = current
    overlap = previous[-1]
    precision, recall = overlap / len(left), overlap / len(right)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall / (precision + recall)).item()


def chrf(hyp, ref, n=6, beta=2):
    """Unofficial character n-gram F-beta, effective orders, on a 0–100 scale."""
    hyp, ref = ''.join(hyp.split()), ''.join(ref.split())
    if not hyp or not ref:
        return 0.0
    if n < 1 or beta <= 0:
        raise ValueError('Positive n and beta required')
    return _chrf_nonempty(hyp, ref, n, beta)


def _chrf_nonempty(hyp, ref, n, beta):
    """Shared arithmetic for nonempty, whitespace-stripped character sequences."""
    precision, recall = [], []
    for order in range(1, min(n, len(hyp), len(ref)) + 1):
        left = Counter(hyp[i:i + order] for i in range(len(hyp) - order + 1))
        right = Counter(ref[i:i + order] for i in range(len(ref) - order + 1))
        overlap = sum((left & right).values())
        precision.append(overlap / sum(left.values()))
        recall.append(overlap / sum(right.values()))
    p, r = sum(precision) / len(precision), sum(recall) / len(recall)
    return 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0


def script_share(text, script='devanagari'):
    """Fraction of letters in a Unicode script; punctuation is ignored."""
    letters = [c for c in text if c.isalpha()]
    return sum(script.upper() in unicodedata.name(c, '') for c in letters) / len(letters) if letters else 0.0


def energy_rank(singular_values, fraction=0.9):
    """Shared rank reaching the aggregate squared-singular-value energy target."""
    if not 0 < fraction <= 1:
        raise ValueError('Energy fraction must be in (0, 1]')
    values = list(singular_values.values()) if isinstance(singular_values, dict) else singular_values
    if isinstance(values, torch.Tensor) and values.ndim == 1:
        values = [values]
    if not len(values):
        return 1
    if isinstance(values[0], (int, float)):
        values = [values]
    rows = [tensor(row) for row in values]
    energy = rows[0].new_zeros(max(row.numel() for row in rows))
    for row in rows:
        energy[:row.numel()] += row.square()
    total = energy.sum()
    if total == 0:
        return 1
    return min(len(energy), int(torch.searchsorted(energy.cumsum(0), fraction * total)) + 1)


def paired_bootstrap(a, b, resamples=1000, seed=0):
    """Paired mean difference with a percentile 95% bootstrap interval."""
    if len(a) != len(b) or not len(a) or resamples < 1:
        raise ValueError('Nonempty paired vectors and positive resamples required')
    delta = tensor(a, 'cpu') - tensor(b, 'cpu')
    generator = torch.Generator().manual_seed(seed)
    means = torch.stack([delta[torch.randint(len(delta), (len(delta),), generator=generator)].mean()
                         for _ in range(resamples)])
    low, high = torch.quantile(means, torch.tensor([0.025, 0.975])).tolist()
    return {'delta': float(delta.mean()), 'low': low, 'high': high, 'resamples': resamples}


def interaction(s_a, s_rev, s_blk, s_d):
    return s_d - s_rev - s_blk + s_a


def top_overlap(rank_1, rank_2, k):
    if k < 1 or min(len(rank_1), len(rank_2)) < k:
        raise ValueError('Rankings must each contain at least k rows')
    return len(set(rank_1[:k]) & set(rank_2[:k])) / k


def cohen_kappa(r1, r2):
    if not r1 or len(r1) != len(r2):
        raise ValueError('Nonempty paired ratings required')
    left, right = Counter(r1), Counter(r2)
    observed = sum(a == b for a, b in zip(r1, r2)) / len(r1)
    chance = sum(left[key] * right[key] for key in set(left) | set(right)) / len(r1) ** 2
    return (observed - chance) / (1 - chance) if chance < 1 else None


def summary_metrics(predictions, references, language):
    """Summarization decision metrics, all on a 0–100 scale."""
    if not predictions or len(predictions) != len(references):
        raise ValueError('Nonempty aligned summaries required')
    return {'rouge_l': 100 * sum(rouge_l(a, b) for a, b in zip(predictions, references)) / len(predictions),
            'chrf': sum(chrf(a, b) for a, b in zip(predictions, references)) / len(predictions),
            'right_language': 100 * sum((script_share(text) > 0.5) == (language == 'hindi')
                                        for text in predictions) / len(predictions)}
