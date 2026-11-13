"""Greedy reducible-loss English memory selection for the mBART CSReL adaptation."""

from dataclasses import asdict, replace
import math
from pathlib import Path
import time

from ..utils.contracts import Objective, ScoreSpec, TrainLimits, TrainingBatch, fingerprint
from ..utils.state import Artifacts, write_json
from .revision import RevisionPipeline


class CSReLSelection:
    def __init__(self, directory, dataset, factory, settings, checkpoint):
        self.directory, self.dataset, self.factory = Path(directory), dataset, factory
        self.settings, self.checkpoint = settings, str(checkpoint)

    def run(self):
        s, directory = self.settings, self.directory
        candidates = tuple(self.dataset.training_view('english'))
        capacity = len(self.dataset.pool_view())
        if not 0 < capacity <= len(candidates) or not 1 <= s.csrel_rounds <= capacity:
            raise ValueError('Invalid CSReL capacity or rounds')
        if any(r.split != 'train' or r.task_id != 'english' for r in candidates):
            raise ValueError('CSReL candidates must be English training records')
        identity = fingerprint({'dataset': self.dataset.fingerprint, 'checkpoint': self.checkpoint,
                                'settings': asdict(s), 'capacity': capacity})
        directory.mkdir(parents=True, exist_ok=False)
        store = Artifacts(directory, identity, external_bytes=lambda: model.base_storage_bytes if model else 0)
        state = {'event': 0, 'phase': 'reference', 'cursor': 0, 'fit_step': 0,
                 'reference': {}, 'scores': {}, 'selected': [], 'rounds': [], 'seconds': 0.0,
                 'costs': {'analysis_seconds': {}, 'forward_pass_units': 0, 'tokens_per_language': {'english': 0}}}
        model = None
        by_id = {r.sample_id: r for r in candidates}
        quotas = [capacity // s.csrel_rounds] * s.csrel_rounds
        quotas[-1] += capacity - sum(quotas)

        def fresh(reference=False):
            nonlocal model
            if model is not None:
                model.release()
            model = self.factory()
            model.settings = replace(model.settings, seed=s.selection_seed, warmup_steps=0)
            model.bind_runtime(directory, None)
            model.bind_dataset(self.dataset)
            if reference:
                model.load(self.checkpoint, 'english')
            model.activate()

        with store.lock():
            try:
                fresh(reference=True)
                while state['phase'] != 'done':
                    started = time.perf_counter()
                    category = {'reference': 'reference_scoring', 'score': 'subset_scoring', 'fit': 'subset_training'}[state['phase']]
                    charged, units = (), 1
                    selected = set(state['selected'])
                    if state['phase'] in {'reference', 'score'}:
                        rows = candidates if state['phase'] == 'reference' else tuple(
                            r for r in candidates if r.sample_id not in selected)
                        part = rows[state['cursor']:state['cursor'] + s.analysis_batch]
                        if part:
                            charged = part
                            values = model.score(self.dataset.collate(part), ScoreSpec('example_loss', 'none')).values.tolist()
                            table = state['reference'] if state['phase'] == 'reference' else state['scores']
                            table.update((r.sample_id, value if state['phase'] == 'reference' else
                                          value - state['reference'][r.sample_id]) for r, value in zip(part, values))
                            state['cursor'] += len(part)
                        if state['cursor'] == len(rows):
                            if state['phase'] == 'score':
                                picked = sorted(state['scores'], key=lambda i: (-state['scores'][i], i))[
                                    :quotas[len(state['rounds'])]]
                                state['selected'] += picked
                                state['rounds'].append({'ids': picked, 'mean_score': sum(
                                    state['scores'][i] for i in picked) / len(picked)})
                            state.update(cursor=0, fit_step=0, scores={},
                                         phase='done' if len(state['rounds']) == len(quotas) else 'fit')
                            if state['phase'] != 'done':
                                fresh()
                    else:
                        rows = tuple(by_id[i] for i in state['selected'])
                        batches = math.ceil(len(rows) / s.batch_size)
                        if state['fit_step'] < batches * s.english_passes:
                            start = state['fit_step'] % batches * s.batch_size
                            part = rows[start:start + s.batch_size]
                            charged, units = part, 3
                            batch = self.dataset.collate(part)
                            report, = model.train(1, batch_factory=lambda: [TrainingBatch(batch, ('current',) * len(part))],
                                objective=Objective('per_example_mean'), hooks=None, limits=TrainLimits(1, 1, batch.valid_tokens))
                            if not report.success:
                                raise RuntimeError('CSReL subset update failed')
                            state['fit_step'] += 1
                        if state['fit_step'] == batches * s.english_passes:
                            state['phase'] = 'score'
                    elapsed = time.perf_counter() - started
                    state['seconds'] += elapsed
                    costs = state['costs']
                    costs['analysis_seconds'][category] = costs['analysis_seconds'].get(category, 0) + elapsed
                    costs['forward_pass_units'] += units * len(charged)
                    costs['tokens_per_language']['english'] += sum(len(r.inputs['ids']) for r in charged)
                    state['event'] += 1
                memory = tuple(by_id[i] for i in sorted(state['selected'], key=lambda i: by_id[i].arrival))
                write_json(directory / 'memory.json', [asdict(r) for r in memory])
                report = {'pool': str(directory / 'memory.json'), 'rounds': state['rounds'],
                          'selection_seconds': state['seconds'], 'candidate_count': len(candidates),
                          'capacity': capacity, 'model': model.model_id, 'costs': state['costs']}
                write_json(directory / 'selection.json', report)
                return memory, report
            finally:
                if model is not None:
                    model.release()


class CSReLReplayPipeline(RevisionPipeline):
    """Same revision lifecycle, with CSReL's separate current/replay CE means."""
    replay_weight = 1.0

    @staticmethod
    def _check_pool(pool):
        if not pool or len({r.sample_id for r in pool}) != len(pool) or any(
                r.split != 'train' or r.task_id != 'english' for r in pool):
            raise ValueError('Paper baselines require a unique English training memory')

    def input_identity(self):
        return {**super().input_identity(), 'method_status': 'adaptation', 'replay_weight': self.replay_weight}

    def _pending_update(self):
        self.model.replay_weight = self.replay_weight
        return super()._pending_update()
