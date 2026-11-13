"""Separate mBART CSReL/TSR comparisons; does not run or modify project phases."""

import argparse
from dataclasses import asdict, dataclass
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
    memory, selection = CSReLSelection(directory / 'csrel_memory', dataset, factory, settings, checkpoint).run()
    arms = {'random': run_hindi(directory / 'random', 'random', hindi_pipeline(settings, pool, checkpoint),
                               dataset, factory, settings)}
    for name, rows in (('csrel', memory), ('csrel_random', pool)):
        pipeline = CSReLReplayPipeline(revision_config(settings, stage='hindi'), pool=rows, english_checkpoint=checkpoint)
        pipeline.replay_weight = settings.csrel_replay_weight
        arms[name] = run_hindi(directory / name, name, pipeline, dataset, factory, settings,
                              selection_seconds=selection['selection_seconds'] if name == 'csrel' else 0)
    config = TSRConfig(**asdict(revision_config(settings, stage='hindi', selector='none')),
        probe_batches=settings.tsr_probe_batches, routing_temperature=settings.tsr_routing_temperature,
        replay_probability=settings.tsr_replay_probability, kd_weight=settings.tsr_kd_weight,
        kd_temperature=settings.tsr_kd_temperature)
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
    parser.add_argument('--source', default='results/code/mbart-1000-1epoch')
    parser.add_argument('--output', default='results/code/mbart-baselines-1epoch')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error('Epochs must be positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error('Keep baseline output separate from the phase source')
    dataset = SummarizationDataset()
    dataset.load({'path': str(source / 'prepared')})
    report = read_json(source / 'phase1/report.json')
    checkpoint = Path(report['context']['english_checkpoint'])
    pool_path = Path(report['context']['pool'])
    if not checkpoint.is_absolute():
        checkpoint = (source / checkpoint).resolve()
    if not pool_path.is_absolute():
        pool_path = (source / pool_path).resolve()
    metadata = read_json(checkpoint / 'metadata.json')
    prep = dataset.manifest['preprocessing']
    if metadata['model'] != PaperMBart.model_id or prep['identity']['model'] != metadata['model']:
        parser.error('Matching mBART checkpoint and prepared data required')
    pool = load_pool(pool_path)
    expected = {r.sample_id: r for r in dataset.pool_view()}
    if len(pool) != 500 or {r.sample_id for r in pool} != set(expected) or any(
            r.inputs != expected[r.sample_id].inputs or r.target != expected[r.sample_id].target for r in pool):
        parser.error('A matching 500-example English pool from the new mBART run is required')
    settings = BaselineSettings(protocol='pilot', seed=args.seed, batch_size=16, analysis_batch=16,
        english_passes=args.epochs, hindi_passes=args.epochs, max_new_tokens=256)

    def factory():
        return PaperMBart(Seq2SeqSettings(rank=metadata['rank'], alpha=metadata['alpha'], seed=args.seed,
            learning_rate=settings.learning_rate, max_source_length=prep['max_source'], max_target_length=prep['max_target']),
            device=args.device, gradient_checkpointing=args.gradient_checkpointing)

    model = factory()
    try:
        settings = align_settings(settings, model)
    finally:
        model.release()
    identity = {'settings': asdict(settings), 'dataset': dataset.fingerprint, 'checkpoint': str(checkpoint),
                'gradient_checkpointing': args.gradient_checkpointing}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}; choose a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', identity)
    with Artifacts(output, fingerprint(identity)).lock():
        return run_baselines(output, dataset, factory, settings, checkpoint, pool)


if __name__ == '__main__':
    main()
