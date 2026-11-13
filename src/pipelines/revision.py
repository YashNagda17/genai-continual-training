from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import csv
import math

import torch

from .pipeline_base import PipelineBase, SummarizationPipeline
from ..utils.contracts import Objective, Record, ScoreSpec, TokenTargets, UpdateHooks, practice_record
from ..utils.numerics import chrf, move_to_device, script_share, summary_metrics
from ..utils.state import ReplayStore, read_json, write_json


@dataclass(frozen=True)
class RevisionConfig:
    stage: str
    selector: str = 'random'
    replay_text: str = 'reference'
    repair_stop: bool = False
    hindi_batch: int = 16
    replay_per_batch: int = 2
    top_fraction: float = 0.5
    warmup_fraction: float = 0.1
    past_filter_fraction: float = 0.2
    repair_beta: float = 1.0
    stop_similarity: float = 0.9
    refresh_steps: int = 200
    hindi_passes: int = 1
    extra_random_steps: int = 0
    analysis_batch: int = 8
    prefix: str = 'plain'
    generation: dict = field(default_factory=lambda: {'max_new_tokens': 64, 'do_sample': False})

    def __post_init__(self):
        if self.stage not in {'english', 'hindi'} or self.selector not in {'none', 'random', 'loss', 'drift', 'full'}:
            raise ValueError('Unknown revision stage or selector')
        if self.replay_text not in {'reference', 'past'} or (self.replay_text == 'past' and self.selector in {'none', 'full'}):
            raise ValueError('Past replay requires a pool selector')
        if self.repair_stop and self.selector not in {'loss', 'drift'}:
            raise ValueError('G3 requires a loss or drift parent')
        if self.extra_random_steps < 0 or (self.extra_random_steps and self.selector != 'random'):
            raise ValueError('Extra updates belong to the random compute control')
        if any(type(v) is not int or v < 1 for v in (self.hindi_batch, self.hindi_passes, self.analysis_batch, self.refresh_steps)):
            raise ValueError('Batch sizes, passes, and refresh cadence must be positive integers')
        if self.replay_per_batch < 0 or (self.selector in {'random', 'loss', 'drift'} and not self.replay_per_batch):
            raise ValueError('Pool selectors require positive replay slots')
        if any(not 0 <= v <= 1 for v in (self.top_fraction, self.warmup_fraction, self.past_filter_fraction, self.stop_similarity)) or self.repair_beta < 0:
            raise ValueError('Invalid revision fraction or repair weight')
        if self.generation.get('do_sample', False):
            raise ValueError('The plan uses greedy generation')
        if self.stage == 'english' and (self.repair_stop or self.replay_text != 'reference' or self.extra_random_steps):
            raise ValueError('English training has no replay or repair policy')


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def load_pool(path):
    """Load the English-stage artifact, never regenerate its summaries in Hindi."""
    value = read_json(Path(path) / 'manifest.json')
    if value['writer'] != 'english_checkpoint':
        raise ValueError('Pool must be written by the English checkpoint')
    return tuple(Record(**row['record']) for row in value['records'])


class RevisionPipeline(SummarizationPipeline):
    def __init__(self, config, model=None, dataset=None, *, pool=(), english_checkpoint=None, adapter_plan=None):
        super().__init__(config, model, dataset)
        self.pool = tuple(pool)
        self.english_checkpoint = None if english_checkpoint is None else str(english_checkpoint)
        self.adapter_plan = adapter_plan
        self.policy = {'ranking': [], 'scores': {}, 'retired': [], 'active': [], 'ever_active': [],
                       'rank_cursor': 0, 'warmup_done': False, 'pending': None, 'work': {},
                       'pass_rows': 0, 'completed_passes': 0, 'validated_passes': 0,
                       'extra': 0, 'full_cursor': 0,
                       'refresh_at': 0, 'filtered': [], 'eligible': None, 'english_ready': False}

    def input_identity(self):
        status = 'project' if self.config.repair_stop or self.config.replay_text == 'past' or self.config.selector == 'drift' else (
            'adaptation' if self.config.selector == 'loss' else 'ablation')
        return {'method_status': status, 'evaluation_schedule': 'after_training',
                'english_checkpoint': self.english_checkpoint, 'pool': [asdict(r) for r in self.pool],
                'adapter_plan': None if self.adapter_plan is None else _plain(asdict(self.adapter_plan))}

    def required_capabilities(self):
        return ({'example_loss', 'token_loss', 'generation'}
                | ({'token_weight_hook'} if self.config.repair_stop else set())
                | ({'adapter_plans'} if self.adapter_plan is not None else set()))

    def validate_runtime(self):
        super().validate_runtime()
        if self.spec.task_ids != (self.config.stage,) or self.spec.batch_size != self.config.hindi_batch:
            raise ValueError('RunSpec must name exactly the configured stage and batch size')
        if self.spec.reduction != 'per_example_mean' or self.spec.accumulation != 1:
            raise ValueError('Revision uses one combined per-example mean microbatch')
        if self.config.stage == 'hindi':
            if not self.english_checkpoint or self.spec.passes != self.config.hindi_passes:
                raise ValueError('Hindi runs need an English checkpoint and the fixed pass count')
            self._check_pool(self.pool)
        if self.spec.passes > 1 and self.spec.exhaustion != 'repeat_admitted':
            raise ValueError('Multiple fixed passes require repeat_admitted')

    @staticmethod
    def _check_pool(pool):
        if not pool or len({r.sample_id for r in pool}) != len(pool) or any(
                r.split != 'train' or r.task_id != 'english' or not r.practice
                or r.practice.get('writer') != 'english_checkpoint' for r in pool):
            raise ValueError('Hindi policy access requires a unique English training pool with saved practice data')

    def _open_view(self):
        return PipelineBase._open_view(self)

    def open_task(self):
        required = math.ceil(len(self.current) / self.spec.batch_size) * self.spec.passes + self.config.extra_random_steps
        if min(self.spec.max_attempts, self.spec.max_steps) < required:
            raise ValueError('Run budget cannot cover the fixed training passes')
        if self.config.stage == 'english':
            if not self.pool:
                self.pool = tuple(self.dataset.pool_view())
            train_ids = {r.sample_id for r in self.current}
            if not self.pool or len({r.sample_id for r in self.pool}) != len(self.pool) or any(r.sample_id not in train_ids or r.split != 'train' for r in self.pool):
                raise ValueError('The English pool must be drawn from this English training view')
        else:
            metadata = read_json(Path(self.english_checkpoint) / 'metadata.json')
            if metadata['state_kind'] != 'english' or (metadata.get('record') or {}).get('writer') != 'english_checkpoint':
                raise ValueError('Hindi training must start from the English-stage artifact')
            self.model.load(self.english_checkpoint, 'english')
            if self.adapter_plan is not None:
                if self.adapter_plan.identity.get('arm', '').startswith('room_') and not self.adapter_plan.identity.get('matched_parameters'):
                    raise ValueError('Validate the complete extra-room family before training')
                self.model.apply_plan(self.adapter_plan)
            if self.config.replay_text == 'past':
                self.filter_past(self.pool)
            # Charge the complete fixed pool even for X; it cannot hide in an unbudgeted archive.
            ReplayStore(self.pool, budget=self.spec.replay)
        current_tokens = sum(len(r.inputs['ids']) for r in self.current) * self.spec.passes
        rows = self.pool
        extra = self.config.replay_per_batch if self.config.selector != 'none' else 0
        if self.config.stage == 'hindi' and self.config.selector == 'full':
            rows, extra = tuple(self.dataset.training_view('english')), self.config.hindi_batch
        bound = current_tokens + required * extra * max((len(r.inputs['ids']) for r in rows), default=0)
        bound += self.config.extra_random_steps * self.config.hindi_batch * max(len(r.inputs['ids']) for r in self.current)
        if self.config.stage == 'english':
            bound, extra = current_tokens, 0
        if self.spec.max_tokens < bound:
            raise ValueError('Token budget cannot cover the fixed passes and replay')
        settings = getattr(self.model, 'settings', None)
        if settings and self.config.hindi_batch + extra > settings.max_batch_size:
            raise ValueError('Configured model batch bound does not cover current plus replay rows')

    def _eligible(self):
        ids = self.policy['eligible']
        return tuple(r for r in self.pool if ids is None or r.sample_id in ids)

    def filter_past(self, pool):
        self._check_pool(pool)
        scores = {r.sample_id: chrf(r.practice['answer_text'], r.target['text']) for r in pool}
        bottom = set(sorted(scores, key=lambda i: (scores[i], i))[:math.floor(len(pool) * self.config.past_filter_fraction)])
        eligible = [r.sample_id for r in pool if r.sample_id not in bottom
                    and script_share(r.practice['answer_text']) <= 0.5 and len(r.practice['answer_text'].split()) >= 5]
        if not eligible:
            raise ValueError('G2 filter left no eligible past summaries')
        self.policy.update(eligible=eligible, filtered=[r.sample_id for r in pool if r.sample_id not in eligible])
        return tuple(r for r in pool if r.sample_id in eligible)

    def write_past(self, pool):
        if self.config.stage != 'english' or any(r.split != 'train' or r.task_id != 'english' for r in pool):
            raise ValueError('Only the English stage may write past summaries for its training pool')
        work = self.policy['work'].setdefault('write', {'cursor': 0, 'records': []})
        self.model.activate()
        print(f"write english pool {work['cursor']}/{len(pool)} samples", flush=True)
        while work['cursor'] < len(pool):
            rows = tuple(pool[work['cursor']:work['cursor'] + self.config.analysis_batch])
            generated = self._generate(rows, 'pool_generation')
            baseline = self._score(rows, ScoreSpec('token_loss', 'none'), 'pool_scoring')
            baseline = baseline.detach().cpu().tolist()
            offset = 0
            written = []
            for row, text, ids in zip(rows, generated['texts'], generated['token_ids']):
                count = len(row.target['ids'])
                old = baseline[offset:offset + count]
                offset += count
                written.append(replace(row, practice={'answer_text': text, 'answer_ids': ids,
                    'baseline': old, 'past_baseline': [], 'loss_before': sum(old) / len(old),
                    'writer': 'english_checkpoint'}))
            nonempty = tuple(practice_record(r) for r in written if r.practice['answer_ids'])
            past = self._score(nonempty, ScoreSpec('token_loss', 'none'), 'pool_scoring').detach().cpu().tolist() if nonempty else []
            offset = 0
            for row in written:
                count = len(row.practice['answer_ids'])
                row.practice['past_baseline'] = past[offset:offset + count] if count else []
                offset += count
                work['records'].append(asdict(row))
            work['cursor'] += len(rows)
            self.log_sample_progress("write english pool", work['cursor'], len(pool))
        return tuple(Record(**row) for row in work['records'])

    def rank_pool(self, pool=None):
        pool = self._eligible() if pool is None else tuple(pool)
        self._check_pool(pool)
        if not {r.sample_id for r in pool} <= {r.sample_id for r in self.pool}:
            raise ValueError('Ranking may access only the declared fixed pool')
        if self.config.selector not in {'loss', 'drift'}:
            self.policy.update(ranking=sorted(r.sample_id for r in pool), warmup_done=True)
            return tuple(self.policy['ranking'])
        work = self.policy['work'].setdefault('rank', {'cursor': 0, 'scores': {}})
        while work['cursor'] < len(pool):
            rows = pool[work['cursor']:work['cursor'] + self.config.analysis_batch]
            if self.config.selector == 'loss':
                losses = self._score(rows, ScoreSpec('example_loss', 'none'), 'pool_scoring').tolist()
                scores = [loss - r.practice['loss_before'] for r, loss in zip(rows, losses)]
            elif self.config.selector == 'drift':
                texts = self._generate(rows, 'pool_generation')['texts']
                scores = [1.0 if script_share(text) > 0.5 else 1 - chrf(text, r.practice['answer_text']) / 100
                          for r, text in zip(rows, texts)]
            else:
                scores = [0.0] * len(rows)
            work['scores'].update(zip((r.sample_id for r in rows), scores))
            work['cursor'] += len(rows)
        self.policy['scores'] = dict(work['scores'])
        self.policy['ranking'] = sorted(work['scores'], key=lambda i: (-work['scores'][i], i))
        self.policy['warmup_done'] = True
        self.policy['work'].pop('rank', None)
        return tuple(self.policy['ranking'])

    def replay_slots(self, count=None):
        count = self.config.replay_per_batch if count is None else count
        pool = self._eligible()
        if count > len(pool):
            raise ValueError('Not enough distinct pool rows for the replay slots')
        ranking = self.policy['ranking']
        eligible_ids = {r.sample_id for r in pool}
        available = [i for i in ranking if i not in self.policy['retired'] and i in eligible_ids]
        top = min(math.ceil(self.config.top_fraction * count), len(available)) if (
            self.policy['warmup_done'] and self.config.selector in {'loss', 'drift'}) else 0
        chosen = []
        while len(chosen) < top:
            index = self.policy['rank_cursor'] % len(ranking)
            self.policy['rank_cursor'] += 1
            if ranking[index] in available and ranking[index] not in chosen:
                chosen.append(ranking[index])
        self.policy['active'] = list(chosen)
        self.policy['ever_active'] = sorted(set(self.policy['ever_active']) | set(chosen))
        rest = [r.sample_id for r in pool if r.sample_id not in chosen]
        chosen += self.rng.sample(rest, count - len(chosen))
        by_id = {r.sample_id: r for r in pool}
        return tuple(practice_record(by_id[i]) if self.config.replay_text == 'past' else by_id[i] for i in chosen)

    def retire(self):
        ids = self.policy['active']
        work = self.policy['work'].setdefault('refresh', {'ids': list(ids), 'cursor': 0})
        by_id = {r.sample_id: r for r in self.pool}
        while work['cursor'] < len(work['ids']):
            rows = tuple(by_id[i] for i in work['ids'][work['cursor']:work['cursor'] + self.config.analysis_batch])
            generated = self._generate(rows, 'stop_regeneration')['texts']
            for row, text in zip(rows, generated):
                if script_share(text) <= 0.5 and chrf(text, row.practice['answer_text']) / 100 >= self.config.stop_similarity:
                    self.policy['retired'].append(row.sample_id)
                    if self.context:
                        self.record_event('retirement', sample_id=row.sample_id, step=self.progress['steps'])
            work['cursor'] += len(rows)
        self.policy['refresh_at'] = self.progress['steps']
        self.policy['work'].pop('refresh', None)
        return tuple(self.policy['retired'])

    def token_targets(self, batch, surprise):
        baselines, spans, offset = [], [], 0
        key = 'past_baseline' if self.config.replay_text == 'past' else 'baseline'
        for row in batch.records:
            count = len(row.target['ids'])
            if row.task_id == 'english':
                before = row.practice[key]
                if len(before) != count:
                    raise ValueError('Saved token baseline no longer aligns with replay text')
                spans.append((offset, count, len(baselines)))
                baselines.extend(before)
            offset += count
        if offset != surprise.numel():
            raise ValueError('Token targets and training batch disagree')
        weights = torch.ones_like(surprise, dtype=torch.float32)
        if spans:
            # Pinned, so the copy queues behind the forward instead of waiting for it.
            before = move_to_device(torch.tensor(baselines, dtype=surprise.dtype), surprise.device)
            for offset, count, start in spans:
                now = surprise[offset:offset + count]
                weight = 1 + self.config.repair_beta * (now - before[start:start + count]).clamp_min(0)
                weights[offset:offset + count] = weight / weight.mean()
        return TokenTargets(torch.zeros_like(weights), weights.detach(), all_positive=True)

    def _generate_held_out(self, rows, work, text_key, category, label, **options):
        """Collect held-out predictions in batches using the current in-memory work state."""
        print(f"{label} {work['cursor']}/{len(rows)} samples", flush=True)
        while work['cursor'] < len(rows):
            part = rows[work['cursor']:work['cursor'] + self.config.analysis_batch]
            work[text_key] += self._generate(part, category, held_out=True, **options)['texts']
            work['cursor'] += len(part)
            self.log_sample_progress(label, work['cursor'], len(rows))

    def _validation_pass(self):
        if self.policy['completed_passes'] < self.spec.passes:
            raise ValueError('English validation requires every training epoch to finish')
        if 'final_validation' in self.policy:
            return self.policy['final_validation']['scores']
        work = self.policy['work'].setdefault('validation', {'cursor': 0, 'predictions': []})
        rows = self.held_out('english', 'validation')
        label = f"validation english epoch {self.policy['completed_passes']}/{self.spec.passes}"
        self._generate_held_out(rows, work, 'predictions', 'validation_generation', label)
        scores = summary_metrics(work['predictions'], [r.target['text'] for r in rows], 'english')
        self.policy['final_validation'] = {'scores': scores, 'texts': work['predictions']}
        self.policy['validated_passes'] = self.policy['completed_passes']
        self.policy['work'].pop('validation', None)
        self.model.release_evaluation_memory()
        return scores

    def next_update(self):
        if self.policy['pending'] is not None:
            return self._pending_update()
        if self.config.stage == 'hindi':
            total = math.ceil(len(self.current) / self.config.hindi_batch) * self.config.hindi_passes
            if not self.policy['warmup_done'] and self.progress['attempts'] >= math.ceil(total * self.config.warmup_fraction):
                self.rank_pool()
            if self.config.repair_stop and self.progress['steps'] - self.policy['refresh_at'] >= self.config.refresh_steps:
                self.retire()
        current = self.current_batch()
        if not current and self.config.stage == 'hindi' and self.policy['extra'] < self.config.extra_random_steps:
            start = (self.policy['extra'] * self.config.hindi_batch) % len(self.current)
            current = tuple(self.current[(start + i) % len(self.current)] for i in range(min(len(self.current), self.config.hindi_batch)))
            self.policy['extra'] += 1
        if not current:
            return None
        replay = ()
        if self.config.stage == 'hindi':
            if self.config.selector == 'full':
                full = tuple(self.dataset.training_view('english'))
                if not full or any(r.split != 'train' for r in full):
                    raise ValueError('Full-English reference requires its declared English training view')
                start = self.policy['full_cursor']
                replay = tuple(full[(start + i) % len(full)] for i in range(self.config.hindi_batch))
                self.policy['full_cursor'] += len(replay)
            elif self.config.selector != 'none':
                replay = self.replay_slots()
        self.policy['pending'] = {'current': [r.sample_id for r in current], 'replay': [r.sample_id for r in replay]}
        return self._pending_update()

    def _pending_update(self):
        pending = self.policy['pending']
        current = {r.sample_id: r for r in self.current}
        replay = {r.sample_id: r for r in (self.dataset.training_view('english') if self.config.selector == 'full' else self.pool)}
        rows = tuple(current[i] for i in pending['current'])
        history = tuple(replay[i] for i in pending['replay'])
        if self.config.replay_text == 'past':
            history = tuple(practice_record(r) for r in history)
        hooks = UpdateHooks(token_targets=self.token_targets) if self.config.repair_stop else None
        return (rows + history, ['current'] * len(rows) + ['replay'] * len(history),
                Objective('per_example_mean', token_mode='weighted' if self.config.repair_stop else 'all'), hooks)

    def observe(self, report, records):
        self.policy['pending'] = None
        self.record_event('revision_exposure', category='update', rows=len(records), forward_pass_units=3 * len(records),
                          tokens_per_language={language: sum(len(r.inputs['ids']) for r in records if r.task_id == language)
                                               for language in ('english', 'hindi')})
        if self.config.stage == 'english':
            self.policy['pass_rows'] += len(records)
            if self.policy['pass_rows'] >= len(self.current):
                self.policy['completed_passes'] += 1
                self.policy['pass_rows'] = 0

    def finish_task(self):
        if self.config.stage == 'hindi':
            self.artifacts.pin_artifact(Path(self.context.output) / 'hindi', self.model.info.checkpoint_bytes,
                                        lambda path: self.model.save(path, 'hindi'))
            return self.pool
        scores = self._validation_pass()
        if not self.policy['english_ready']:
            reference = self.artifacts.pin(self.model, {'kind': 'english', 'writer': 'english_checkpoint',
                'prefix': self.config.prefix, 'passes': self.policy['completed_passes'],
                'selection': 'final_epoch', 'validation': scores})
            self.policy.update(english_ready=True, english_checkpoint=reference)
        written = self.write_past(self.pool)
        self.artifacts.pin_pool([{'id': r.sample_id, 'summary': r.practice['answer_text'],
            'baseline': r.practice['baseline'], 'loss_before': r.practice['loss_before'], 'record': asdict(r)} for r in written],
            english_checkpoint=self.policy['english_checkpoint'])
        return written

    def after_memory_commit(self):
        self.policy['work'].pop('write', None)

    def _evaluate_seen(self):
        for split in ('validation', 'test'):
            if split == 'test':
                print(f"test after {self.spec.passes} training epochs", flush=True)
            for language in ('english', 'hindi'):
                for guarded in (False, True):
                    key = f'{language}/{split}/{"guarded" if guarded else "plain"}'
                    if any(row['task'] == key for row in self.metrics):
                        continue
                    rows = self.held_out(language, split)
                    work = self.policy['work'].setdefault('evaluation', {'cursor': 0, 'texts': []})
                    cached = (self.policy.get('final_validation') if self.config.stage == 'english'
                              and key == 'english/validation/plain' else None)
                    if cached is not None:
                        work.update(cursor=len(rows), texts=cached['texts'])
                    guard = ({'block': 'devanagari'} if language == 'english' else {'penalize': 'latin', 'penalty': 2.0}) if guarded else None
                    self._generate_held_out(rows, work, 'texts',
                                            'guard_generation' if guarded else 'evaluation_generation',
                                            f"generate {key}", guard=guard)
                    scores = cached['scores'] if cached is not None else summary_metrics(
                        work['texts'], [r.target['text'] for r in rows], language)
                    self._save_scores('final', key, scores)
                    self.metrics.append({'checkpoint_task': self.config.stage, 'task': key, 'metrics': scores})
                    folder = Path(self.context.output) / 'predictions'
                    folder.mkdir(exist_ok=True)
                    write_json(folder / (key.replace('/', '-') + '.json'), [
                        {'id': r.sample_id, 'target': r.target['text'], 'prediction': text} for r, text in zip(rows, work['texts'])])
                    self.policy['work'].pop('evaluation', None)
                    if cached is not None:
                        self.policy.pop('final_validation', None)
                self.model.release_evaluation_memory()

    def _save_scores(self, pass_number, task, scores):
        fields = ('pass', 'task', 'rouge_l', 'chrf', 'right_language')
        row = {'pass': pass_number, 'task': task, **{key: scores[key] for key in fields[2:]}}
        saved = [item for item in self.policy.get('score_history') or []
                 if not (item.get('pass') == pass_number and item.get('task') == task)]
        saved.append(row)
        self.policy['score_history'] = saved
        path = Path(self.context.output) / 'scores.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(tuple(item[key] for key in fields) for item in saved)
