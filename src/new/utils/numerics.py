"""Device routing, JSON files, timing, and the summary metrics."""

from collections import Counter
from pathlib import Path
import json
import os
import unicodedata

import torch
from torch.nn import functional as F


def resolve_device(preferred="auto"):
    """An NVIDIA GPU when one is present, otherwise the CPU."""
    if preferred not in (None, "auto"):
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    print("Using CPU: no NVIDIA GPU is available", flush=True)
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_mb(device):
    return torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None


def write_json(path, value):
    """Write a temporary file, then replace, so readers see the old or the new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rouge_l(prediction, reference):
    """ROUGE-L F1 over whitespace tokens, with no stemming. Safe for Hindi."""
    left, right = prediction.split(), reference.split()
    if not left or not right:
        return float(left == right)
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0]
        for j, other in enumerate(right, 1):
            current.append(previous[j - 1] + 1 if item == other else max(previous[j], current[j - 1]))
        previous = current
    if not previous[-1]:
        return 0.0
    precision, recall = previous[-1] / len(left), previous[-1] / len(right)
    return 2 * precision * recall / (precision + recall)


def chrf(hyp, ref, n=6, beta=2):
    """Character n-gram F-beta on a 0-100 scale, safe for any script."""
    hyp, ref = "".join(hyp.split()), "".join(ref.split())
    if not hyp or not ref:
        return 0.0
    precision, recall = [], []
    for order in range(1, min(n, len(hyp), len(ref)) + 1):
        left = Counter(hyp[i:i + order] for i in range(len(hyp) - order + 1))
        right = Counter(ref[i:i + order] for i in range(len(ref) - order + 1))
        overlap = sum((left & right).values())
        precision.append(overlap / sum(left.values()))
        recall.append(overlap / sum(right.values()))
    p, r = sum(precision) / len(precision), sum(recall) / len(recall)
    return 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0


def script_share(text, script="devanagari"):
    """dev(s): the fraction of letters in one Unicode script."""
    letters = [c for c in text if c.isalpha()]
    return sum(script.upper() in unicodedata.name(c, "") for c in letters) / len(letters) if letters else 0.0


def right_language(text, language):
    """Hindi summaries are mostly Devanagari; English ones are not (dev(s) > 0.5 means slipped)."""
    return (script_share(text) > 0.5) == (language == "hindi")


def summary_metrics(predictions, references, language):
    """ROUGE-L, chrF, and the right-language share, all on a 0-100 scale."""
    count = len(predictions)
    return {"rouge_l": 100 * sum(map(rouge_l, predictions, references)) / count,
            "chrf": sum(map(chrf, predictions, references)) / count,
            "right_language": 100 * sum(right_language(t, language) for t in predictions) / count}


def cosine(a, b):
    """Signed cosine; a zero vector gives zero."""
    return float(F.cosine_similarity(a.float(), b.float(), dim=0))


def spearman(left, right):
    """Rank correlation with averaged ties; None when either side is constant."""
    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        result, i = [0.0] * len(values), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in order[i:j + 1]:
                result[k] = (i + j) / 2
            i = j + 1
        return torch.tensor(result) - (len(values) - 1) / 2
    a, b = ranks(left), ranks(right)
    return cosine(a, b) if a.norm() > 0 and b.norm() > 0 else None


def energy_rank(spectra, fraction=0.9):
    """k_E: the smallest shared rank holding `fraction` of all squared singular values."""
    energy = torch.zeros(max(s.numel() for s in spectra))
    for s in spectra:
        energy[:s.numel()] += s.float().cpu().square()
    return min(len(energy), int(torch.searchsorted(energy.cumsum(0), fraction * energy.sum())) + 1)


def paired_bootstrap(a, b, resamples=1000, seed=0):
    """Paired mean difference with a percentile 95% interval over resampled articles."""
    delta = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    means = torch.stack([delta[torch.randint(len(delta), (len(delta),), generator=generator)].mean()
                         for _ in range(resamples)])
    low, high = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"delta": float(delta.mean()), "low": low, "high": high, "resamples": resamples}
