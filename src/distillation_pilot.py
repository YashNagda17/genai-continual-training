"""Small paired replay/KD experiment. Protocol: /design/distillation-pilot.md."""

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import random
import time

if __name__ == '__main__':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
from .datasets.dataset_instruction import SummarizationDataset
from .models.model_distillation import DistillationMBart
from .models.model_seq2seq import Seq2SeqSettings
from .utils.contracts import Objective, TrainingBatch, TrainLimits, fingerprint
from .utils.numerics import summary_metrics
from .utils.state import Artifacts, read_json, write_json


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


def main(argv=None, *, model_class=None):
    model_class = model_class or DistillationMBart
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', default='results/code/mbart-1000-1epoch/prepared')
    parser.add_argument('--checkpoint', default='results/code/mbart-1000-1epoch/phase1/english/pinned/english/summarize')
    parser.add_argument('--output', default='results/code/distillation-pilot')
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--steps', type=int, default=4)
    duration.add_argument('--epochs', type=int)
    parser.add_argument('--model', choices=('mbart',), default='mbart')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--hindi-batch', type=int, default=2)
    parser.add_argument('--replay-batch', type=int, default=1)
    parser.add_argument('--eval-examples', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=32)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if min(args.steps, args.hindi_batch, args.replay_batch, args.eval_examples,
           args.max_new_tokens, args.top_k, args.threads) < 1 or args.strength <= 0:
        parser.error('Counts and strength must be positive')
    if args.epochs is not None and args.epochs < 1:
        parser.error('Epochs must be positive')
    if args.hindi_batch + args.replay_batch > 32 or (args.epochs is None and args.steps > 100):
        parser.error('Pilot permits at most 32 rows per batch and 100 updates')
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output is not empty: {root}; choose a fresh directory")
    root.mkdir(parents=True, exist_ok=True)
    dataset = SummarizationDataset()
    dataset.load({'path': args.prepared})
    if args.epochs is not None:
        args.steps = math.ceil(len(dataset.training_view('hindi')) / args.hindi_batch) * args.epochs
        if not args.steps:
            parser.error('Hindi training view is empty')
    metadata = read_json(Path(args.checkpoint) / 'metadata.json')
    prep = dataset.manifest['preprocessing']
    if metadata['model'] != model_class.model_id or prep['identity']['model'] != metadata['model']:
        parser.error('This pilot requires matching mBART checkpoint and prepared data')
    # Selection is fixed across arms, independently from training ordering.
    views = {lang: tuple(sorted(dataset.views[f'{lang}/validation'], key=lambda r: r.sample_id)[:args.eval_examples])
             for lang in ('english', 'hindi')}
    if any(len(rows) != args.eval_examples for rows in views.values()):
        parser.error('Insufficient validation records')
    config = {**vars(args), 'checkpoint_sha256': digest(Path(args.checkpoint) / 'weights.pt'),
              'prepared_identity': dataset.fingerprint, 'schema': 3,
              'torch': str(torch.__version__), 'validation_ids': {k: [r.sample_id for r in v] for k, v in views.items()}}
    write_json(root / 'config.json', config)

    def factory(seed, directory):
        model = model_class(Seq2SeqSettings(rank=metadata['rank'], alpha=metadata['alpha'],
            seed=seed, learning_rate=args.learning_rate, max_source_length=prep['max_source'],
            max_target_length=prep['max_target']), device=args.device, gradient_checkpointing=args.gradient_checkpointing)
        model.bind_runtime(directory, None)
        model.bind_dataset(dataset)
        model.load(args.checkpoint, metadata['state_kind'])
        model.activate()
        return model

    with Artifacts(root, fingerprint(config)).lock():
        seed, folder = args.seed, root
        rng = random.Random(seed)
        hindi = list(dataset.training_view('hindi'))
        pool = list(dataset.pool_view())
        rng.shuffle(hindi)
        if not hindi or (args.epochs is None and len(hindi) < args.steps * args.hindi_batch) or len(pool) < args.replay_batch:
            parser.error('Insufficient training data')
        current = ([tuple(hindi[i:i + args.hindi_batch]) for i in range(0, len(hindi), args.hindi_batch)] * args.epochs
                   if args.epochs is not None else
                   [tuple(hindi[i * args.hindi_batch:(i + 1) * args.hindi_batch]) for i in range(args.steps)])
        batches = [rows + tuple(rng.sample(pool, args.replay_batch)) for rows in current]
        selected = {r.sample_id: r for rows in batches for r in rows[-args.replay_batch:]}
        write_json(folder / 'selection.json', [[r.sample_id for r in batch] for batch in batches])
        cache_path = folder / 'teacher.pt'
        print('load English teacher', flush=True)
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
        cache_info = {'seconds': cache_seconds, 'bytes': cache_path.stat().st_size,
                      'sha256': digest(cache_path), 'examples': len(cache)}
        arms = {}
        for arm, strength in (('replay', 0.0), ('replay_kd', args.strength)):
            directory = folder / arm
            directory.mkdir(exist_ok=True)
            print(arm, flush=True)
            model = factory(seed, directory)
            store = Artifacts(directory, fingerprint({'config': config, 'seed': seed, 'arm': arm}))
            try:
                model.teacher_cache, model.distillation_strength = cache, strength
                state = {'event': 0, 'training_seconds': 0.0, 'steps': []}
                for index in range(args.steps):
                    started = time.perf_counter()
                    batch = dataset.collate(batches[index])
                    item = TrainingBatch(batch, ('current',) * len(current[index]) + ('replay',) * args.replay_batch)
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
                    print(f'{arm}: update {index + 1}/{args.steps}, {seconds:.2f}s, NLL {nll:.4f}', flush=True)
                after = evaluate(model, dataset, views, args.max_new_tokens)
                checkpoint = directory / 'hindi'
                store.pin_artifact(checkpoint, model.info.checkpoint_bytes, lambda path: model.save(path, 'hindi'))
                arms[arm] = {'after': after, **state, 'strength': strength, 'checkpoint': str(checkpoint)}
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
                  'note': 'Exploratory small subset; not a scientific result.'}
        rows = []
        for arm in ('before', 'replay', 'replay_kd'):
            scores = result['before'] if arm == 'before' else result['arms'][arm]['after']
            for lang, value in scores.items():
                rows.append({'arm': arm, 'language': lang,
                             **{k: value['metrics'][k] for k in ('chrf', 'rouge_l', 'negative_nll')}})
        with (root / 'report.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        aggregate = {lang: {'delta': delta[lang]} for lang in views}
        write_json(root / 'report.json', {'config': config, **result, 'aggregate': aggregate})
        lines = ['# Small distillation comparison', '', 'Exploratory mBART run; validation subset only.', '',
                 '| Arm | Language | chrF | ROUGE-L | NLL |', '|---|---|---:|---:|---:|']
        lines.extend(f"| {r['arm']} | {r['language']} | {r['chrf']:.4f} | {r['rouge_l']:.4f} | {-r['negative_nll']:.4f} |" for r in rows)
        lines.extend(['', f"{result['result_type']}. KD minus replay chrF: {delta}.",
            f"Training seconds: replay {arms['replay']['training_seconds']:.2f}; KD {arms['replay_kd']['training_seconds']:.2f}; teacher cache {cache_info['seconds']:.2f}."])
        (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        for language in views:
            replay = arms['replay']['after'][language]['metrics']['chrf']
            kd = arms['replay_kd']['after'][language]['metrics']['chrf']
            print(f"{language} chrF: replay={replay:.4f}, KD={kd:.4f}, delta={kd - replay:+.4f}", flush=True)
        print(root / 'report.md', flush=True)
        return read_json(root / 'report.json')


if __name__ == '__main__':
    main()
