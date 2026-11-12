"""Small paired replay/KD experiment. Protocol: /design/distillation-pilot.md."""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import random
import statistics
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
from .datasets.dataset_instruction import SummarizationDataset
from .models.model_distillation import DistillationMT5
from .models.model_seq2seq import Seq2SeqSettings
from .utils.contracts import Objective, TrainingBatch, TrainLimits, fingerprint
from .utils.numerics import summary_metrics
from .utils.state import Checkpoints, read_json, write_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evaluate(model, dataset, views, max_tokens):
    result = {}
    for language, records in views.items():
        print(f'evaluate {language}: {len(records)} articles', flush=True)
        metrics = model.evaluate(records, {'batch_size': 1, 'return_predictions': True,
            'generation': {'max_new_tokens': max_tokens, 'do_sample': False, 'no_repeat_ngram_size': 3}})
        predictions = metrics.pop('_predictions')
        metrics.update(summary_metrics([p['prediction'] for p in predictions],
                                       [p['target'] for p in predictions], language))
        result[language] = {'metrics': metrics, 'predictions': predictions}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', default='results/code/summarization-passes5/prepared')
    parser.add_argument('--checkpoint', default='results/code/summarization-passes5/phase1/final/english')
    parser.add_argument('--output', default='results/code/distillation-pilot')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--hindi-batch', type=int, default=2)
    parser.add_argument('--replay-batch', type=int, default=1)
    parser.add_argument('--eval-examples', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=32)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if min(args.steps, args.hindi_batch, args.replay_batch, args.eval_examples,
           args.max_new_tokens, args.top_k, args.threads) < 1 or args.strength <= 0:
        parser.error('Counts and strength must be positive')
    if args.hindi_batch + args.replay_batch > 32 or args.steps > 100:
        parser.error('Pilot permits at most 32 rows per batch and 100 updates')
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset = SummarizationDataset()
    dataset.load({'path': args.prepared})
    metadata = read_json(Path(args.checkpoint) / 'metadata.json')
    prep = dataset.manifest['preprocessing']
    if metadata['model'] != DistillationMT5.model_id or prep['identity']['model'] != metadata['model']:
        parser.error('This pilot requires matching mT5-small checkpoint and prepared data')
    # Selection is fixed across seeds and arms, independently from training ordering.
    views = {lang: tuple(sorted(dataset.views[f'{lang}/validation'], key=lambda r: r.sample_id)[:args.eval_examples])
             for lang in ('english', 'hindi')}
    if any(len(rows) != args.eval_examples for rows in views.values()):
        parser.error('Insufficient validation records')
    config = {**vars(args), 'checkpoint_sha256': digest(Path(args.checkpoint) / 'weights.pt'),
              'prepared_identity': dataset.fingerprint, 'schema': 1,
              'torch': str(torch.__version__), 'validation_ids': {k: [r.sample_id for r in v] for k, v in views.items()}}
    if (root / 'config.json').exists() and read_json(root / 'config.json') != config:
        parser.error('Settings or inputs changed; choose a new output directory')
    write_json(root / 'config.json', config)

    def factory(seed, directory):
        model = DistillationMT5(Seq2SeqSettings(rank=metadata['rank'], alpha=metadata['alpha'],
            seed=seed, learning_rate=args.learning_rate, max_source_length=prep['max_source'],
            max_target_length=prep['max_target']), device=args.device)
        model.bind_runtime(directory, None)
        model.bind_dataset(dataset)
        model.load(args.checkpoint, metadata['state_kind'])
        model.activate()
        return model

    all_results = []
    with Checkpoints(root, fingerprint(config)).lock():
        for seed in args.seeds:
            folder = root / f'seed-{seed}'
            folder.mkdir(exist_ok=True)
            rng = random.Random(seed)
            hindi = list(dataset.training_view('hindi'))
            pool = list(dataset.pool_view())
            rng.shuffle(hindi)
            if len(hindi) < args.steps * args.hindi_batch or len(pool) < args.replay_batch:
                parser.error('Insufficient training data')
            batches = [tuple(hindi[i * args.hindi_batch:(i + 1) * args.hindi_batch]) +
                       tuple(rng.sample(pool, args.replay_batch)) for i in range(args.steps)]
            selected = {r.sample_id: r for batch in batches for r in batch[args.hindi_batch:]}
            write_json(folder / 'selection.json', [[r.sample_id for r in batch] for batch in batches])
            cache_path = folder / 'teacher.pt'
            if not cache_path.exists() or not (folder / 'before.json').exists():
                print(f'seed {seed}: load English teacher', flush=True)
                teacher = factory(seed, folder / 'teacher-runtime')
                try:
                    started = time.perf_counter()
                    cache = {}
                    for record in selected.values():
                        cache.update(teacher.cache_teacher(dataset.collate((record,)), args.top_k))
                    cache_seconds = time.perf_counter() - started
                    torch.save(cache, folder / 'teacher.tmp')
                    (folder / 'teacher.tmp').replace(cache_path)
                    write_json(folder / 'cache.json', {'seconds': cache_seconds, 'bytes': cache_path.stat().st_size,
                        'sha256': digest(cache_path), 'examples': len(cache), 'model_params': teacher.info.total_parameters})
                    write_json(folder / 'before.json', evaluate(teacher, dataset, views, args.max_new_tokens))
                finally:
                    teacher.release()
            cache_info = read_json(folder / 'cache.json')
            if digest(cache_path) != cache_info['sha256']:
                raise ValueError('Teacher cache checksum differs')
            cache = torch.load(cache_path, map_location='cpu', weights_only=True)
            arms = {}
            for arm, strength in (('replay', 0.0), ('replay_kd', args.strength)):
                directory = folder / arm
                directory.mkdir(exist_ok=True)
                if (directory / 'result.json').exists():
                    arms[arm] = read_json(directory / 'result.json')
                    continue
                print(f'seed {seed}: {arm}', flush=True)
                model = factory(seed, directory)
                store = Checkpoints(directory, fingerprint({'config': config, 'seed': seed, 'arm': arm}))
                try:
                    model.teacher_cache, model.distillation_strength = cache, strength
                    state = {'event': 0, 'training_seconds': 0.0, 'steps': []}
                    if (directory / 'latest.json').exists():
                        checkpoint, state = store.latest()
                        model.load(checkpoint / 'learner', 'resume')
                    for index in range(state['event'], args.steps):
                        started = time.perf_counter()
                        batch = dataset.collate(batches[index])
                        item = TrainingBatch(batch, ('current',) * args.hindi_batch + ('replay',) * args.replay_batch)
                        report, = model.train(1, batch_factory=lambda: (item,), objective=Objective('per_example_mean'),
                            hooks=None, limits=TrainLimits(1, 1, batch.valid_tokens))
                        if model.device.type == 'cuda':
                            torch.cuda.synchronize(model.device)
                        seconds = time.perf_counter() - started
                        nll = float(report.loss_sums['supervised']) / float(report.loss_counts['supervised'])
                        if not report.success or not torch.isfinite(torch.tensor(nll)):
                            raise RuntimeError('Failed or nonfinite update')
                        state['training_seconds'] += seconds
                        state['event'] = index + 1
                        state['steps'].append({'step': index + 1, 'supervised_nll': nll, 'seconds': seconds,
                                               'cost': report.cost})
                        store.publish(state, model, None, (), completed=index + 1 == args.steps)
                        print(f'{arm}: update {index + 1}/{args.steps}, {seconds:.2f}s, NLL {nll:.4f}', flush=True)
                    after = evaluate(model, dataset, views, args.max_new_tokens)
                    arms[arm] = {'after': after, **state, 'strength': strength}
                    write_json(directory / 'result.json', arms[arm])
                finally:
                    model.release()
            before = read_json(folder / 'before.json')
            delta = {lang: arms['replay_kd']['after'][lang]['metrics']['chrf'] -
                          arms['replay']['after'][lang]['metrics']['chrf'] for lang in views}
            verdict = ('positive' if delta['english'] > 0 and delta['hindi'] >= -0.5 else
                       'null' if delta['english'] == 0 and delta['hindi'] == 0 else 'negative')
            result = {'seed': seed, 'before': before, 'arms': arms, 'cache': cache_info,
                      'chrf_delta_kd_minus_replay': delta, 'result_type': verdict,
                      'note': 'Exploratory small subset; not a scientific result or evidence for mBART.'}
            write_json(folder / 'report.json', result)
            all_results.append(result)
        rows = []
        for result in all_results:
            for arm in ('before', 'replay', 'replay_kd'):
                scores = result['before'] if arm == 'before' else result['arms'][arm]['after']
                for lang, value in scores.items():
                    rows.append({'seed': result['seed'], 'arm': arm, 'language': lang,
                                 **{k: value['metrics'][k] for k in ('chrf', 'rouge_l', 'negative_nll')}})
        with (root / 'report.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        aggregate = {lang: {'mean_delta': statistics.mean(r['chrf_delta_kd_minus_replay'][lang] for r in all_results),
            'std_delta': statistics.stdev(r['chrf_delta_kd_minus_replay'][lang] for r in all_results) if len(all_results) > 1 else None}
            for lang in views}
        write_json(root / 'report.json', {'config': config, 'seeds': all_results, 'aggregate': aggregate})
        lines = ['# Small distillation comparison', '', 'Exploratory mT5 run; validation subset only.', '',
                 '| Seed | Arm | Language | chrF | ROUGE-L | NLL |', '|---|---|---|---:|---:|---:|']
        lines.extend(f"| {r['seed']} | {r['arm']} | {r['language']} | {r['chrf']:.4f} | {r['rouge_l']:.4f} | {-r['negative_nll']:.4f} |" for r in rows)
        for r in all_results:
            lines.extend(['', f"Seed {r['seed']}: {r['result_type']}. KD minus replay chrF: {r['chrf_delta_kd_minus_replay']}.",
                f"Training seconds: replay {r['arms']['replay']['training_seconds']:.2f}; KD {r['arms']['replay_kd']['training_seconds']:.2f}; teacher cache {r['cache']['seconds']:.2f}."])
        (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(root / 'report.md', flush=True)


if __name__ == '__main__':
    main()
