"""Torch policy arithmetic, device routing and shared metrics."""

import torch


def resolve_device(preferred="auto"):
    """Route CUDA/ROCm, then Intel XPU, with CPU as the available-device fallback.
    Explicit CPU avoids accelerator probing; indexed requests keep a valid device index.
    """
    if preferred is None or str(preferred) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu:0")
        return torch.device("cpu")
    device = torch.device(preferred)
    if device.type == "cpu":
        return device
    backend = getattr(torch, device.type, None)
    if (device.type in {"cuda", "xpu"} and backend is not None
            and backend.is_available()
            and (device.index is None or device.index < backend.device_count())):
        return device
    return torch.device("cpu")


def move_to_device(value, device):
    """Move nested batch/optimizer tensors without casting token IDs or detaching gradients."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def tensor(values, device=None):
    """Detach policy inputs and keep tensor inputs on their device unless explicitly moved.
    Use float32 on CPU for reference arithmetic and float16 on accelerators for portability.
    Python sequences default to CPU; callers with a model pass its resolved device.
    """
    target = torch.device(device) if device is not None else (
        values.device if isinstance(values, torch.Tensor) else torch.device("cpu"))
    dtype = torch.float32 if target.type == "cpu" else torch.float16
    if isinstance(values, torch.Tensor):
        return values.detach().to(device=target, dtype=dtype)
    return torch.as_tensor(values, dtype=dtype, device=target)


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


def softmax(values):
    """Use Torch's stable softmax and return plain probabilities for policy storage."""
    values = tensor(values)
    return tuple(torch.softmax(values, dim=-1).tolist()) if values.numel() else ()


def sigmoid(x):
    """Evaluate one logistic probability using Torch's stable sigmoid kernel."""
    return torch.sigmoid(tensor(x)).item()


def sample(items, probabilities, count, replacement, rng):
    """Sample with a local CPU Torch generator seeded from the saved policy RNG.
    Keep small sampling state on CPU across devices without changing global Torch RNG.
    Return repeated records when replacement permits them, preserving exposure multiplicity.
    """
    if count < 0 or len(items) != len(probabilities):
        raise ValueError("Invalid sampling request")
    weights = tensor(probabilities, "cpu")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if items and not torch.isclose(weights.sum(), weights.new_tensor(1.0)):
        raise ValueError("Probabilities must sum to one")
    if count == 0:
        return ()
    if not items or (not replacement and count > (weights > 0).sum().item()):
        raise ValueError("Insufficient positive-probability records")
    generator = torch.Generator(device="cpu").manual_seed(rng.getrandbits(63))
    indices = torch.multinomial(weights, count, replacement=replacement, generator=generator)
    return tuple(items[index] for index in indices.tolist())


def quotas(tasks, capacity):
    """Split capacity equally across sorted task IDs; earlier IDs receive the remainder."""
    ordered = sorted(tasks)
    if capacity < 0 or len(ordered) != len(set(ordered)):
        raise ValueError("Invalid quota request")
    if not ordered:
        return {}
    base, remainder = divmod(capacity, len(ordered))
    return {task: base + (i < remainder) for i, task in enumerate(ordered)}


def continual_metrics(matrix, task_index):
    """Compute old-task transfer and forgetting on a higher-is-better score matrix.
    Use only observed columns from each prior row; negative forgetting is preserved.
    Old-task averages are undefined until at least one earlier task exists.
    """
    t = task_index
    row = tensor(matrix[t])
    if t == 0:
        return {"current": row[0].item(), "backward_transfer": None, "forgetting": None,
                "undefined_reason": "No previous task"}
    initial = tensor([matrix[j][j] for j in range(t)], row.device)
    previous_best = torch.stack([tensor([matrix[k][j] for k in range(j, t)], row.device).max()
                                 for j in range(t)])
    return {
        "current": row[t].item(),
        "backward_transfer": (row[:t] - initial).mean().item(),
        "forgetting": (previous_best - row[:t]).mean().item(),
    }
