"""F: search a shared adapter rank using generation on validation only."""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .pipeline_base import SummarizationPipeline
from ..utils.numerics import energy_rank, summary_metrics
from ..utils.state import write_json


@dataclass(frozen=True)
class ShrinkConfig:
    full_rank: int
    energy_fraction: float = 0.9
    tolerance: float = 1.0
    grid: tuple = (12, 8, 6, 4)
    analysis_batch: int = 8
    generation: dict = field(default_factory=lambda: {'max_new_tokens': 64, 'do_sample': False})

    def __post_init__(self):
        if self.full_rank < 1 or self.analysis_batch < 1 or any(k < 1 for k in self.grid):
            raise ValueError('Positive ranks and batch size required')
        if not 0 < self.energy_fraction <= 1 or self.tolerance < 0 or self.generation.get('do_sample', False):
            raise ValueError('Invalid energy fraction, tolerance, or decoding')


class GenerationShrinkPipeline(SummarizationPipeline):
    def __init__(self, config, model=None, dataset=None, *, checkpoint=None, state_kind='hindi', validation=None):
        super().__init__(config, model, dataset)
        self.checkpoint, self.state_kind = checkpoint, state_kind
        self.validation = {key: tuple(rows) for key, rows in (validation or {}).items()}
        self.policy = {'baseline': None, 'k_energy': None, 'table': [], 'work': {}, 'direction': None,
                       'next_rank': None, 'k_generation': None, 'result': None, 'shapes': {}}

    def required_capabilities(self):
        return {'generation', 'adapter_updates', 'adapter_shrink'}

    def input_identity(self):
        return {'method_status': 'project', 'checkpoint': str(self.checkpoint), 'state_kind': self.state_kind,
                'validation': {key: [r.sample_id for r in rows] for key, rows in self.validation.items()}}

    def open_task(self):
        if self.checkpoint is None:
            raise ValueError('Shrinking requires a finished, immutable checkpoint')
        self.model.load(self.checkpoint, self.state_kind)
        self.model.activate()
        self._views()

    def _views(self):
        views = self.validation or {language: self.held_out(language, 'validation') for language in ('english', 'hindi')}
        if set(views) != {'english', 'hindi'} or any(not rows or any(
                r.split != 'validation' or r.task_id != language for r in rows) for language, rows in views.items()):
            raise ValueError('Shrink decisions require English and Hindi validation rows, never test')
        return views

    def _evaluate_rank(self, model):
        views = self._views()
        work = self.policy['work']
        scores = {}
        for language, rows in views.items():
            state = work.setdefault(language, {'cursor': 0, 'texts': []})
            while state['cursor'] < len(rows):
                part = rows[state['cursor']:state['cursor'] + self.config.analysis_batch]
                state['texts'] += self._generate(part, 'shrink_generation', model, held_out=True)['texts']
                state['cursor'] += len(part)
                self._commit()
            scores[language] = summary_metrics(state['texts'], [r.target['text'] for r in rows], language)
        return scores

    def passes(self, scores, baseline):
        return all(scores[language][metric] >= baseline[language][metric] - self.config.tolerance
                   for language in ('english', 'hindi') for metric in ('rouge_l', 'chrf', 'right_language'))

    def _candidate(self, rank):
        fresh = self.auxiliary is None
        candidate = self.acquire_auxiliary('shrink', f'rank-{rank}', checkpoint=self.checkpoint, state_kind=self.state_kind)
        if fresh:
            # Every candidate starts at the original checkpoint, never the previous truncation.
            candidate.shrink_adapters(rank)
            self._commit()
        return candidate

    def shrink_search(self):
        if self.policy['result'] is not None:
            return self.policy['result']
        self._views()
        if self.policy['k_energy'] is None:
            self.model.activate()
            spectra = {}
            for name, delta in self.model.effective_updates().items():
                spectra[name] = torch.linalg.svdvals(delta.float())
                self.policy['shapes'][name] = list(delta.shape)
            if not spectra:
                raise ValueError('No adapters to shrink')
            self.policy['k_energy'] = energy_rank(spectra, self.config.energy_fraction)
            if self.policy['k_energy'] > self.config.full_rank:
                raise ValueError('full_rank does not cover the effective adapter rank')
            self._commit()
        if self.policy['baseline'] is None:
            self.policy['baseline'] = self._evaluate_rank(self.model)
            self.policy['work'] = {}
            self.policy['next_rank'] = self.policy['k_energy']
            self._commit()
        while self.policy['next_rank'] is not None:
            rank = self.policy['next_rank']
            candidate = self._candidate(rank)
            scores = self._evaluate_rank(candidate)
            passed = self.passes(scores, self.policy['baseline'])
            size = sum(min(rank, *shape) * sum(shape) * 4 for shape in self.policy['shapes'].values())
            self.release_auxiliary()
            self.policy['table'].append({'rank': rank, 'scores': scores, 'passed': passed, 'adapter_tensor_bytes': size})
            self.policy['work'] = {}
            if self.policy['direction'] is None:
                self.policy['direction'] = 'down' if passed else 'up'
            if passed:
                self.policy['k_generation'] = rank
            tested = {row['rank'] for row in self.policy['table']}
            if self.policy['direction'] == 'down':
                remaining = sorted((k for k in set(self.config.grid) if k < rank and k not in tested), reverse=True)
                self.policy['next_rank'] = remaining[0] if passed and remaining else None
            else:
                remaining = sorted(k for k in set((*self.config.grid, self.config.full_rank)) if rank < k <= self.config.full_rank and k not in tested)
                self.policy['next_rank'] = remaining[0] if not passed and remaining else None
            self._commit()
        chosen = self.policy['k_generation']
        if chosen is None:
            raise ValueError('No candidate, including full rank, passed the generation test')
        candidate = self._candidate(chosen)
        output = Path(self.context.output) / 'shrunk-adapter'
        self.artifacts.pin_artifact(output, candidate.info.checkpoint_bytes,
            lambda path: candidate.save_shrunk_adapter(path, k_energy=self.policy['k_energy'], k_generation=chosen,
                                                       test_table=self.policy['table']))
        self.release_auxiliary()
        self.policy['result'] = {'k_E': self.policy['k_energy'], 'k_G': chosen, 'baseline': self.policy['baseline'],
                                  'table': self.policy['table'], 'adapter': 'shrunk-adapter'}
        self._commit()
        return self.policy['result']

    def next_update(self):
        result = self.shrink_search()
        write_json(Path(self.context.output) / 'shrink.json', result)
        self.metrics = [{'checkpoint_task': 'shrink', 'task': f'rank-{row["rank"]}/{language}',
                         'metrics': {**scores, 'passed': int(row['passed']),
                                     'adapter_tensor_bytes': row['adapter_tensor_bytes']}}
                        for row in result['table'] for language, scores in row['scores'].items()]
        return None
