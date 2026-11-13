"""Fixed-base signatures, batch-level replay routing, and era-teacher KD on mBART."""

from dataclasses import dataclass, replace
import math
import random

import torch

from .block_selection import BlockSelectionPipeline
from .csrel import CSReLReplayPipeline
from .revision import RevisionConfig
from ..utils.numerics import cosine


@dataclass(frozen=True)
class TSRConfig(RevisionConfig):
    probe_batches: int = 10
    routing_temperature: float = 0.1
    replay_probability: float = 0.5
    kd_weight: float = 0.5
    kd_temperature: float = 2.0

    def __post_init__(self):
        super().__post_init__()
        if (self.probe_batches < 1 or self.routing_temperature <= 0 or
                not 0 <= self.replay_probability < 1 or self.kd_temperature <= 0 or self.kd_weight < 0):
            raise ValueError('Invalid TSR signature, routing or distillation settings')
        if self.stage != 'hindi' or self.selector != 'none':
            raise ValueError('TSR owns the Hindi/replay branch schedule')


class TSRPipeline(CSReLReplayPipeline):
    replay_weight = None

    def schedule(self, count, seed):
        rng, branches, seen = random.Random(f'{seed}:tsr-branches'), [], 0
        while seen < count:
            replay = rng.random() < self.config.replay_probability
            branches.append(replay)
            seen += not replay
        return branches

    def extra_updates(self, dataset, settings):
        count = math.ceil(len(dataset.training_view('hindi')) / settings.batch_size) * settings.hindi_passes
        return len(self.schedule(count, settings.seed)) - count

    def open_task(self):
        super().open_task()
        if 'tsr_schedule' not in self.policy:
            count = math.ceil(len(self.current) / self.config.hindi_batch) * self.spec.passes
            self.policy['tsr_schedule'] = self.schedule(count, self.spec.seed)
        if 'signatures' not in self.policy:
            probe = self.acquire_auxiliary('probe', 'tsr-fixed-base')
            # Identical base and zero-B factors for both tasks, independent of arm seed.
            probe.release()
            probe.settings = replace(probe.settings, seed=0)
            probe.activate()
            signatures = {}
            for language, source in (('english', tuple(self.dataset.training_view('english'))), ('hindi', self.current)):
                rows = source[:self.config.probe_batches * self.config.hindi_batch]
                values = BlockSelectionPipeline._mean_gradient(self, rows, 'block', f'tsr-{language}', probe)
                signatures[language] = torch.cat(list(values.values())).cpu()
            if any(not torch.isfinite(v).all() or v.norm() == 0 for v in signatures.values()):
                raise ValueError('TSR requires finite nonzero task signatures')
            similarity = cosine(signatures['english'], signatures['hindi'])
            self.policy['signatures'] = {'cosine': {'english': similarity}, 'weights': {'english': 1.0},
                'vectors': {k: v.tolist() for k, v in signatures.items()},
                'routing_temperature': self.config.routing_temperature,
                'english_ids': [r.sample_id for r in self.dataset.training_view('english')[:self.config.probe_batches * self.config.hindi_batch]],
                'hindi_ids': [r.sample_id for r in self.current[:self.config.probe_batches * self.config.hindi_batch]]}
            self.policy['work'].clear()
            self.release_auxiliary()

    def next_update(self):
        if self.policy['pending'] is not None:
            return self._pending_update()
        index = self.progress['attempts']
        if index >= len(self.policy['tsr_schedule']):
            return None
        if self.policy['tsr_schedule'][index]:
            current = ()
            replay = tuple(self.rng.sample(list(self.pool), min(self.config.hindi_batch, len(self.pool))))
        else:
            current, replay = self.current_batch(), ()
            if not current:
                raise ValueError('TSR schedule exhausted Hindi rows prematurely')
        self.policy['pending'] = {'current': [r.sample_id for r in current], 'replay': [r.sample_id for r in replay]}
        return self._pending_update()

    def _pending_update(self):
        self.model.teacher_checkpoint = self.english_checkpoint
        self.model.kd_weight, self.model.kd_temperature = self.config.kd_weight, self.config.kd_temperature
        return super()._pending_update()

    def observe(self, report, records):
        super().observe(report, records)
        self._cost('teacher_inference', records, self.model.teacher_cost or {})
