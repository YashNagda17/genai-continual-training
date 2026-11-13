"""Separate mBART CSReL/TSR comparisons; does not run or modify project phases."""

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import csv

import torch

from .datasets.dataset_instruction import SummarizationDataset
from .models.model_baselines import PaperMBart
from .models.model_seq2seq import Seq2SeqSettings
from .pipelines.csrel import CSReLSelection, CSReLReplayPipeline
from .pipelines.tsr import TSRConfig, TSRPipeline
from .pipelines.revision import load_pool
from .phases.settings import PhaseSettings, align_settings
from .phases.support import hindi_pipeline, revision_config, run_hindi
from .baseline_source import local_dataset, prepare_local_source
from .phases.artifacts import compare_arms
from .utils.contracts import fingerprint
from .utils.state import Artifacts, read_json, write_json


@dataclass(frozen=True)
class BaselineSettings(PhaseSettings):
    selection_seed: int = 0
    csrel_rounds: int = 5
    csrel_replay_weight: float = 1.0
    tsr_probe_batches: int = 10
    tsr_routing_temperature: float = 0.1
    tsr_replay_probability: float = 0.5
    tsr_kd_weight: float = 0.5
    tsr_kd_temperature: float = 2.0


def run_baselines(directory, dataset, factory, settings, checkpoint, pool):
    directory = Path(directory)
    print('Selecting CSReL English memory', flush=True)
    memory, selection = CSReLSelection(directory / 'csrel_memory', dataset, factory, settings, checkpoint).run()
    print('Running random replay control', flush=True)
    arms = {'random': run_hindi(directory / 'random', 'random', hindi_pipeline(settings, pool, checkpoint),
                               dataset, factory, settings)}
    for name, rows in (('csrel', memory), ('csrel_random', pool)):
        print(f'Running {name}: {settings.hindi_passes} Hindi epochs', flush=True)
        pipeline = CSReLReplayPipeline(revision_config(settings, stage='hindi'), pool=rows, english_checkpoint=checkpoint)
        pipeline.replay_weight = settings.csrel_replay_weight
        arms[name] = run_hindi(directory / name, name, pipeline, dataset, factory, settings,
                              selection_seconds=selection['selection_seconds'] if name == 'csrel' else 0)
    config = TSRConfig(**asdict(revision_config(settings, stage='hindi', selector='none')),
        probe_batches=settings.tsr_probe_batches, routing_temperature=settings.tsr_routing_temperature,
        replay_probability=settings.tsr_replay_probability, kd_weight=settings.tsr_kd_weight,
        kd_temperature=settings.tsr_kd_temperature)
    print(f'Running TSR: {settings.hindi_passes} Hindi epochs plus replay updates', flush=True)
    arms['tsr'] = run_hindi(directory / 'tsr', 'tsr', TSRPipeline(config, pool=pool, english_checkpoint=checkpoint),
                           dataset, factory, settings)
    comparisons = {f'{a}-vs-{b}': compare_arms(arms[a], arms[b])
                   for a, b in (('csrel', 'csrel_random'), ('csrel', 'random'), ('tsr', 'random'))}
    rows = [{'arm': name, 'task': task, 'chrf': value['chrf'], 'rouge_l': value['rouge_l'],
             'right_language': value['right_language']} for name, arm in arms.items()
            for task, value in arm['metrics'].items() if task.endswith('/plain')]
    with (directory / 'report.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Exploratory mBART baseline comparison', '',
             '| Arm | Task | chrF | ROUGE-L |', '|---|---|---:|---:|']
    lines.extend(f"| {r['arm']} | {r['task']} | {r['chrf']:.4f} | {r['rouge_l']:.4f} |" for r in rows)
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    report = {'model': 'mbart', 'protocol': 'exploratory', 'settings': asdict(settings),
              'selection': selection, 'arms': arms, 'comparisons': comparisons}
    write_json(directory / 'report.json', report)
    for row in rows:
        print(f"{row['arm']} {row['task']} chrF={row['chrf']:.4f}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='results/code/mbart-local-source')
    parser.add_argument('--output', default='results/code/mbart-baselines-1epoch')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Prepare cached local XL-Sum and train the English source before running baselines')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--analysis-batch-size', type=int, default=16)
    parser.add_argument('--replay-per-batch', type=int, default=2)
    parser.add_argument('--train-examples', type=int, default=2000, help='Per language, with --from-scratch')
    parser.add_argument('--validation-examples', type=int, default=200, help='Per language, with --from-scratch')
    parser.add_argument('--test-examples', type=int, default=500, help='Per language, with --from-scratch')
    parser.add_argument('--pool-examples', type=int, default=500, help='English memory size, with --from-scratch')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--max-source-length', type=int, default=512, help='Input token limit, with --from-scratch')
    parser.add_argument('--max-target-length', type=int, default=256, help='Target token limit, with --from-scratch')
    parser.add_argument('--teacher-device', default=None, help='Use cpu to keep the TSR teacher off a small GPU')
    args = parser.parse_args(argv)
    sizes = (args.epochs, args.batch_size, args.analysis_batch_size, args.replay_per_batch,
             args.train_examples, args.validation_examples, args.test_examples, args.pool_examples,
             args.max_new_tokens, args.max_source_length, args.max_target_length)
    if min(sizes) < 1:
        parser.error('Epochs, example counts, batch sizes, replay slots, and generation length must be positive')
    if args.from_scratch and args.pool_examples > args.train_examples:
        parser.error('--pool-examples cannot exceed --train-examples')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error('Keep baseline output separate from the phase source')
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output is not empty: {output}; choose a fresh directory')
    settings = BaselineSettings(protocol='pilot', seed=args.seed, batch_size=args.batch_size,
        analysis_batch=args.analysis_batch_size, replay_per_batch=args.replay_per_batch,
        english_passes=args.epochs, hindi_passes=args.epochs, max_new_tokens=args.max_new_tokens,
        csrel_rounds=min(5, args.pool_examples))

    def model_factory(rank=16, alpha=32.0, max_source=None, max_target=None):
        model = PaperMBart(Seq2SeqSettings(rank=rank, alpha=alpha, seed=args.seed,
            learning_rate=settings.learning_rate, max_source_length=max_source or args.max_source_length,
            max_target_length=max_target or args.max_target_length,
            max_batch_size=max(args.batch_size + args.replay_per_batch, args.analysis_batch_size)),
            device=args.device, gradient_checkpointing=args.gradient_checkpointing)
        model.teacher_device = args.teacher_device
        return model

    if args.from_scratch:
        if source.exists() and any(source.iterdir()):
            parser.error(f'Source is not empty: {source}; use a fresh source or omit --from-scratch to reuse it')
        print('Preparing fresh local baseline source; no experiment phases will run', flush=True)
        model = model_factory()
        try:
            dataset = local_dataset(model, train=args.train_examples, validation=args.validation_examples,
                                    test=args.test_examples, pool=args.pool_examples, seed=args.seed)
            settings = align_settings(settings, model)
        finally:
            model.release()
        prepare_local_source(source, dataset, model_factory, settings)
    if not source.is_dir():
        parser.error(f'--source folder does not exist: {source}. Supply the actual local mBART run folder, '
                     'or use --from-scratch to prepare data and train the English source locally.')
    source_report = 'source.json' if (source / 'source.json').is_file() else 'phase1/report.json'
    required = ('prepared/manifest.json', 'prepared/records.jsonl', source_report)
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        parser.error(f'Incomplete mBART source run: {source}. Missing: {", ".join(missing)}. '
                     'Imported reports and smoke runs cannot supply the baseline training inputs.')
    dataset = SummarizationDataset()
    dataset.load({'path': str(source / 'prepared')})
    report = read_json(source / source_report)
    checkpoint = Path(report['context']['english_checkpoint'])
    pool_path = Path(report['context']['pool'])
    if not checkpoint.is_absolute():
        checkpoint = (source / checkpoint).resolve()
    if not pool_path.is_absolute():
        pool_path = (source / pool_path).resolve()
    if not (checkpoint / 'metadata.json').is_file() or not pool_path.exists():
        parser.error(f'English checkpoint or replay pool is missing. Checkpoint: {checkpoint}; '
                     f'pool: {pool_path}. If this run was copied from Colab, update these paths '
                     f'in the local source {source_report} to the copied artifacts.')
    metadata = read_json(checkpoint / 'metadata.json')
    prep = dataset.manifest['preprocessing']
    if metadata['model'] != PaperMBart.model_id or prep['identity']['model'] != metadata['model']:
        parser.error('Matching mBART checkpoint and prepared data required')
    pool = load_pool(pool_path)
    expected = {r.sample_id: r for r in dataset.pool_view()}
    if not pool or len(pool) != len(expected) or {r.sample_id for r in pool} != set(expected) or any(
            r.inputs != expected[r.sample_id].inputs or r.target != expected[r.sample_id].target for r in pool):
        parser.error('A nonempty English replay pool matching the prepared source data is required')
    if len(pool) < settings.csrel_rounds:
        settings = replace(settings, csrel_rounds=len(pool))

    def factory():
        return model_factory(metadata['rank'], metadata['alpha'], prep['max_source'], prep['max_target'])

    model = factory()
    try:
        settings = align_settings(settings, model)
    finally:
        model.release()
    identity = {'settings': asdict(settings), 'dataset': dataset.fingerprint, 'checkpoint': str(checkpoint),
                'gradient_checkpointing': args.gradient_checkpointing, 'teacher_device': args.teacher_device}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}; choose a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', identity)
    with Artifacts(output, fingerprint(identity)).lock():
        return run_baselines(output, dataset, factory, settings, checkpoint, pool)


if __name__ == '__main__':
    main()
