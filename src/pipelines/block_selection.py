from dataclasses import dataclass
from pathlib import Path

import torch

from .pipeline_base import SummarizationPipeline
from ..utils.contracts import AdapterPlan, ScoreSpec
from ..utils.numerics import cosine, cosine_tensor
from ..utils.state import write_json


@dataclass(frozen=True)
class BlockSelectionConfig:
    stability_threshold: float
    seed: int = 0
    sample_size: int = 64
    analysis_batch: int = 8
    encoder_blocks: int = 12
    decoder_blocks: int = 12
    shared_rank: int = 16
    encoder_block_cost: int = 18432    # Adapter parameters per rank in one encoder block (mBART)
    decoder_block_cost: int = 26624   # ... and in one decoder block

    def __post_init__(self):
        if not -1 <= self.stability_threshold <= 1:
            raise ValueError('Invalid map threshold')
        if any(type(v) is not int or v < 1 for v in (self.sample_size, self.analysis_batch,
                self.encoder_blocks, self.decoder_blocks, self.shared_rank)):
            raise ValueError('Positive sizes and ranks required')


def cell_of(name):
    pieces = name.split('.')
    if pieces[0] == 'model' and len(pieces) > 4 and pieces[2] == 'layers':
        # mBART: model.encoder.layers.3.self_attn.q_proj, or .fc1 / .fc2 for the feed-forward
        stack, block = pieces[1], int(pieces[3])
        part = 'feed_forward' if pieces[-1] in ('fc1', 'fc2') else {
            'self_attn': 'self_attention', 'encoder_attn': 'cross_attention'}[pieces[-2]]
        band = block // 2 * 2 + 1
        return f'{stack}.{part}.{band}-{band + 1}'
    raise ValueError(f'Not an mBART projection: {name}')


def spearman(left, right):
    def ranks(values):
        values = torch.as_tensor(values, dtype=torch.float32)
        order = values.argsort()
        ranks = torch.empty_like(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2
            start = end
        return ranks - ranks.mean()
    a, b = ranks(left), ranks(right)
    return cosine(a, b) if a.norm() > 0 and b.norm() > 0 else None


class BlockSelectionPipeline(SummarizationPipeline):
    def __init__(self, config, model=None, dataset=None, *, pool=(), hindi=(),
                 english_checkpoint=None, hindi_checkpoint=None):
        super().__init__(config, model, dataset)
        self.rng.seed(config.seed)
        self.pool, self.hindi = tuple(pool), tuple(hindi)
        self.english_checkpoint, self.hindi_checkpoint = english_checkpoint, hindi_checkpoint
        self.policy = {'samples': None, 'sample_index': 0, 'maps': [], 'work': {}, 'change': {}, 'result': None}

    def input_identity(self):
        return {'method_status': 'ablation', 'english_checkpoint': str(self.english_checkpoint), 'hindi_checkpoint': str(self.hindi_checkpoint),
                'pool': [r.sample_id for r in self.pool], 'hindi': [r.sample_id for r in self.hindi]}

    def required_capabilities(self):
        return {'probe_gradient', 'adapter_updates', 'zero_probe'}

    def _change_map(self, path, kind):
        role = self.acquire_auxiliary('analysis', str(path), checkpoint=path, state_kind=kind)
        result = {}
        for name, value in role.effective_update_norms().items():
            cell = cell_of(name)
            result[cell] = result.get(cell, 0.0) + value
        if result:
            totals = torch.stack(list(result.values())).detach().cpu().tolist()
            result = dict(zip(result, totals))
        self.release_auxiliary(restore_learner=False)
        return result

    def _mean_gradient(self, rows, view, key, probe):
        work = self.policy['work'].setdefault(key, {'cursor': 0, 'sum': {}, 'groups': None})
        while work['cursor'] < len(rows):
            part = rows[work['cursor']:work['cursor'] + self.config.analysis_batch]
            values = self._score(part, ScoreSpec('probe_gradient', 'per_example_mean', parameter_view=view),
                                 'probe_pass', probe)
            groups = list(values['groups'])
            if work['groups'] is not None and groups != work['groups']:
                raise ValueError('Probe group layout changed during a sweep')
            work['groups'] = groups
            gradients = values['gradients']
            sizes = [gradient.numel() for gradient in gradients]
            if gradients:
                # Restore the JSON totals with one transfer across all groups.
                previous = torch.tensor([item for group, size in zip(groups, sizes)
                                         for item in work['sum'].get(group, [0.0] * size)],
                                        dtype=gradients[0].dtype)
                previous = previous.to(gradients[0].device, non_blocking=True)
                packed = (previous + len(part) * torch.cat(gradients)).detach().cpu().tolist()
                offset = 0
                for group, size in zip(groups, sizes):
                    work['sum'][group] = packed[offset:offset + size]
                    offset += size
            work['cursor'] += len(part)
        groups = list(work['sum'])
        sizes = [len(work['sum'][group]) for group in groups]
        packed = torch.tensor([item for group in groups for item in work['sum'][group]], dtype=torch.float32)
        means = packed.to(probe.device, non_blocking=True) / len(rows)
        return dict(zip(groups, means.split(sizes)))

    def location_maps(self):
        if self.policy['result'] is not None:
            return self.policy['result']
        if any(r.split != 'train' or r.task_id != 'english' or not r.practice
               or r.practice.get('writer') != 'english_checkpoint' for r in self.pool):
            raise ValueError('C1 English probes may read only the fixed English pool')
        if any(r.split != 'train' or r.task_id != 'hindi' for r in self.hindi):
            raise ValueError('C1 Hindi probes require Hindi training rows')
        n = self.config.sample_size
        if any(len({r.sample_id for r in rows}) != len(rows) for rows in (self.pool, self.hindi)):
            raise ValueError('Probe samples require unique observation IDs')
        if min(len(self.pool), len(self.hindi)) < 2 * n:
            raise ValueError('C1 needs two disjoint samples per language')
        if not self.english_checkpoint or not self.hindi_checkpoint:
            raise ValueError('C1 needs saved English and Hindi-only checkpoints')
        for label, path, kind in [('english', self.english_checkpoint, 'english'), ('hindi', self.hindi_checkpoint, 'hindi')]:
            if label not in self.policy['change']:
                self.policy['change'][label] = self._change_map(path, kind)
        if self.policy['samples'] is None:
            self.policy['samples'] = {language: self.rng.sample([r.sample_id for r in rows], 2 * n)
                                       for language, rows in [('english', self.pool), ('hindi', self.hindi)]}
        fresh = self.auxiliary is None
        probe = self.acquire_auxiliary('probe', 'english-zero-B', checkpoint=self.english_checkpoint, state_kind='english')
        if fresh:
            probe.reset_probe()
        lookup = {r.sample_id: r for r in (*self.pool, *self.hindi)}
        while self.policy['sample_index'] < 2:
            index = self.policy['sample_index']
            result = {}
            for view in ('cell', 'block'):
                means = {}
                for language in ('english', 'hindi'):
                    ids = self.policy['samples'][language][index * n:(index + 1) * n]
                    rows = tuple(lookup[i] for i in ids)
                    means[language] = self._mean_gradient(rows, view, f'{view}/{language}', probe)
                groups = sorted(means['english'])
                metrics = [torch.stack((means['hindi'][group].norm(),
                                       cosine_tensor(means['english'][group], means['hindi'][group])))
                           for group in groups]
                rows = torch.stack(metrics).detach().cpu().tolist() if metrics else []
                result[view] = {group: {'pull': pull, 'agreement': agreement}
                                for group, (pull, agreement) in zip(groups, rows)}
            self.policy['maps'].append(result)
            self.policy['sample_index'] += 1
            self.policy['work'] = {}
        cells = sorted(self.policy['maps'][0]['cell'])
        correlation = spearman([self.policy['maps'][0]['cell'][key]['agreement'] for key in cells],
                               [self.policy['maps'][1]['cell'][key]['agreement'] for key in cells])
        blocks = {key: sum(row['block'][key]['agreement'] for row in self.policy['maps']) / 2
                  for key in self.policy['maps'][0]['block']}
        self.release_auxiliary(restore_learner=False)
        self.policy['result'] = {'change': self.policy['change'], 'samples': self.policy['maps'],
                                  'correlation': correlation,
                                  'stable': correlation is not None and correlation >= self.config.stability_threshold,
                                  'blocks': blocks}
        return self.policy['result']

    def adapter_plan(self, arm):
        blocks = {stack: [f'{stack}.{i}' for i in range(1, size + 1)] for stack, size in
                  [('encoder', self.config.encoder_blocks), ('decoder', self.config.decoder_blocks)]}
        all_blocks = blocks['encoder'] + blocks['decoder']
        if arm == 'all':
            trainable = all_blocks
        elif arm == 'late':
            trainable = blocks['encoder'][-2:] + blocks['decoder'][-2:]
        elif arm == 'early_late':
            trainable = sorted({ids[0] for ids in blocks.values()} | {ids[-1] for ids in blocks.values()})
        else:
            raise ValueError('Unknown block-placement arm')
        count = self.config.shared_rank * sum(self.config.encoder_block_cost if key.startswith('encoder.') else self.config.decoder_block_cost for key in trainable)
        return AdapterPlan(tuple(trainable), identity={'arm': arm, 'trainable_parameters': count})


    def next_update(self):
        result = self.location_maps()
        write_json(Path(self.context.output) / 'location_maps.json', result)
        self.metrics = [{'checkpoint_task': 'maps', 'task': f'sample-{i + 1}/{view}/{group}', 'metrics': values}
                        for i, sample in enumerate(result['samples']) for view, groups in sample.items()
                        for group, values in groups.items()]
        return None
