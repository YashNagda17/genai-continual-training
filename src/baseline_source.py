"""Prepare a fresh local English checkpoint and replay pool for paper baselines."""

from dataclasses import asdict
from pathlib import Path
import json
import random

from .datasets.dataset_instruction import Seq2SeqPreprocessing, SummarizationDataset
from .phases.data import choose_ids, xlsum_directory
from .phases.support import revision_config, run_pipeline
from .pipelines.revision import RevisionPipeline
from .utils.contracts import fingerprint
from .utils.state import Artifacts, write_json


def local_dataset(model, *, train=2000, validation=200, test=500, pool=500, seed=1):
    """Use the cached XL-Sum data; never fetch an old experiment from Drive."""
    if min(train, validation, test, pool) < 1 or pool > train:
        raise ValueError('Dataset sizes must be positive and the pool cannot exceed English training')
    root = xlsum_directory()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    limits = {'train': train, 'validation': validation, 'test': test}
    samples = {(language, split): [] for language in ('english', 'hindi') for split in limits}
    seen = dict.fromkeys(samples, 0)
    streams = {key: random.Random(f'{seed}:{key[0]}:{key[1]}') for key in samples}
    print('Selecting local XL-Sum rows with a streaming reservoir', flush=True)
    with (root / 'records.jsonl').open(encoding='utf-8') as handle:
        for arrival, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get('task'), row.get('split'))
            if key not in samples or not row.get('text', '').strip() or not row.get('summary', '').strip():
                continue
            seen[key] += 1
            count = limits[key[1]]
            if len(samples[key]) < count:
                samples[key].append((arrival, row))
            else:
                position = streams[key].randrange(seen[key])
                if position < count:
                    samples[key][position] = (arrival, row)
    for (language, split), rows in samples.items():
        if len(rows) < limits[split]:
            raise ValueError(f'{language}/{split} has only {len(rows)} usable rows; requested {limits[split]}')
    dataset = SummarizationDataset()
    dataset.manifest = {'dataset_id': manifest['dataset_id'], 'revision': manifest['revision'],
                        'source': manifest, 'selection': {'method': 'streaming_reservoir', 'seed': seed,
                                                        'per_language': limits, 'usable_rows': {
                                                            '/'.join(key): value for key, value in seen.items()}}}
    raw = tuple(dataset._record(row, arrival) for arrival, row in sorted(
        [item for rows in samples.values() for item in rows], key=lambda item: item[0]))
    selected = dataset.split(raw, {'seed': seed})
    subsets = {f'{language}/{split}': [r.sample_id for r in selected[split] if r.task_id == language]
               for language in ('english', 'hindi') for split in limits}
    dataset.preprocess(selected, Seq2SeqPreprocessing(**model.preprocessing_kwargs(), prefix='summarize: '))
    for name, ids in subsets.items():
        dataset.expose(name, ids)
    dataset.expose('english/pool', choose_ids(dataset.training_view('english'), pool, seed + 1))
    return dataset


def prepare_local_source(directory, dataset, factory, settings):
    """Train English only; do not invoke any experiment phase handlers."""
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Source is not empty: {directory}; use a fresh source or omit --from-scratch')
    directory.mkdir(parents=True, exist_ok=True)
    identity = fingerprint({'dataset': dataset.fingerprint, 'settings': asdict(settings)})
    with Artifacts(directory, identity).lock():
        dataset.save(directory / 'prepared')
        write_json(directory / 'preparation.json', {'status': 'running', 'settings': asdict(settings)})
        try:
            print(f'Training local English source: {settings.english_passes} epochs', flush=True)
            english = directory / 'english'
            pipeline = RevisionPipeline(revision_config(settings, stage='english'), pool=dataset.pool_view())
            arm = run_pipeline(pipeline, factory, dataset, english, 'english', settings,
                               settings.english_passes, 'english')
            context = {'english_checkpoint': str(Path(arm['checkpoint']).relative_to(directory)),
                       'pool': str((english / 'pinned' / 'pool').relative_to(directory))}
            write_json(directory / 'source.json', {'format': 'baseline-source-v1', 'context': context,
                                                 'english': arm, 'settings': asdict(settings)})
            write_json(directory / 'preparation.json', {'status': 'completed', 'settings': asdict(settings)})
            return context
        except BaseException as error:
            write_json(directory / 'preparation.json', {'status': 'failed', 'error': str(error)})
            raise
