import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from .model_mbart import MBart25Summarizer


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


class PaperMBart(MBart25Summarizer):
    """Standard mBART unless the pipeline explicitly enables a paper objective."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_weight = None
        self.teacher_checkpoint = None
        self.kd_weight, self.kd_temperature = 0.5, 2.0
        self._origins = None
        self._teacher = None
        self.teacher_cost = None

    def train(self, k, *, batch_factory, objective, hooks, limits):
        def batches():
            for item in batch_factory():
                self._origins = item.origins
                yield item
        try:
            return super().train(k, batch_factory=batches, objective=objective, hooks=hooks, limits=limits)
        finally:
            self._origins = None

    def _losses(self, payload):
        if self._origins is None or not self.module.training:
            return super()._losses(payload)
        if self.replay_weight is None and self.teacher_checkpoint is None:
            return super()._losses(payload)
        target = None
        if self.teacher_checkpoint:
            if self._teacher is None:
                self._teacher = self.fork('teacher', str(self.teacher_checkpoint))
                self._teacher.load(self.teacher_checkpoint, 'english')
            with self._teacher._execution(), torch.no_grad():
                start = self._teacher._begin_timed_block()
                target = self._teacher._forward(payload).logits.detach()
                self.teacher_cost = self._teacher._timing_cost(start)
        logits = self._forward(payload).logits
        position = self._position_nll(logits, payload['labels'])
        mask = payload['labels'] != -100
        counts = mask.sum(-1).clamp_min(1)
        sums = position.sum(-1)
        means = sums / counts
        if target is not None:
            kl = ChunkedKL.apply(logits, target, mask, self.kd_temperature, self.settings.logit_chunk_size)
            means = means + self.kd_weight * self.kd_temperature ** 2 * kl.sum(-1) / counts
        if self.replay_weight is not None:
            n = len(self._origins)
            current, replay = self._origins.count('current'), self._origins.count('replay')
            weights = [n * self.replay_weight / replay if origin == 'replay' else n / current
                       for origin in self._origins]
            means = means * means.new_tensor(weights)
        return means, sums, counts, position

    def release(self):
        if self._teacher is not None:
            self._teacher.release()
            self._teacher = None
        super().release()
