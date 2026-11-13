"""Optional cached top-k-plus-tail distillation for the isolated mBART pilot."""

import torch
from .model_mbart import MBart25Summarizer


def compress_probabilities(logits, top_k):
    """Keep actual probability mass, with all omitted tokens in one bucket."""
    values, indices = logits.float().topk(min(top_k, logits.shape[-1] - 1), dim=-1)
    probabilities = (values - logits.float().logsumexp(-1, keepdim=True)).exp()
    tail = (1 - probabilities.sum(-1, keepdim=True)).clamp_min(0)
    buckets = torch.cat((probabilities, tail), -1)
    return indices.cpu(), (buckets / buckets.sum(-1, keepdim=True)).cpu()


def bucket_kl(logits, indices, probabilities):
    """KL(teacher || student) over top-k tokens and the residual vocabulary."""
    indices = indices.to(logits.device)
    q = probabilities.to(device=logits.device, dtype=torch.float32)
    z = logits.float()
    normalizer = z.logsumexp(-1, keepdim=True)
    selected = z.gather(-1, indices) - normalizer
    rest = z.scatter(-1, indices, float('-inf')).logsumexp(-1, keepdim=True) - normalizer
    log_p = torch.cat((selected, rest), -1)
    return (torch.special.xlogy(q, q) - q * log_p).sum(-1)


class CachedDistillation:
    """Base training API with an opt-in, ID-aligned cache; evaluation remains CE."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_cache = {}
        self.distillation_strength = 0.0
        self._teacher_rows = None

    def cache_teacher(self, batch, top_k=32):
        if any(record.split != 'train' for record in batch.records):
            raise ValueError('Teacher caching requires training records')
        result = {}
        with self._execution(), torch.no_grad():
            payload = self._payload(batch)
            logits = self._forward(payload).logits
            for row, identifier in enumerate(batch.ids):
                valid = payload['labels'][row] != -100
                indices, probabilities = compress_probabilities(logits[row][valid], top_k)
                result[identifier] = {'indices': indices, 'probabilities': probabilities,
                                      'labels': payload['labels'][row][valid].cpu()}
        return result

    def train(self, k, *, batch_factory, objective, hooks, limits):
        if self.distillation_strength and (objective.reduction != 'per_example_mean' or objective.token_mode != 'all'):
            raise ValueError('Distillation pilot requires unweighted per-example CE')

        def batches():
            for training in batch_factory():
                self._teacher_rows = [self.teacher_cache.get(i) for i in training.batch.ids]
                yield training
        try:
            return super().train(k, batch_factory=batches, objective=objective, hooks=hooks, limits=limits)
        finally:
            self._teacher_rows = None

    def _losses(self, payload):
        if not self.distillation_strength or self._teacher_rows is None or not self.module.training:
            return super()._losses(payload)
        labels = payload['labels']
        logits = self._forward(payload).logits
        position = self._position_nll(logits, labels)
        counts = (labels != -100).sum(-1).clamp_min(1)
        sums = position.sum(-1)
        penalties = []
        for row, cached in enumerate(self._teacher_rows):
            if cached is None:
                penalties.append(logits.new_zeros(()))
                continue
            valid = labels[row] != -100
            if not torch.equal(labels[row][valid].cpu(), cached['labels']):
                raise ValueError('Teacher prefixes differ from the current replay labels')
            selected = logits[row][valid]
            terms = [bucket_kl(selected[start:start + 16], cached['indices'][start:start + 16],
                               cached['probabilities'][start:start + 16])
                     for start in range(0, len(selected), 16)]
            penalties.append(torch.cat(terms).mean())
        means = sums / counts + self.distillation_strength * torch.stack(penalties)
        return means, sums, counts, position


class DistillationMBart(CachedDistillation, MBart25Summarizer):
    """mBART executor with opt-in cached teacher distillation."""
