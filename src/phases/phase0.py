"""Phase 0: one update, readable-summary artifacts, and training-only truncation checks."""

from ..utils.contracts import Objective, TrainLimits, TrainingBatch
from ..utils.state import Checkpoints, read_json, write_json
from .artifacts import annotate, save_report
from .settings import generation_spec
from .support import phase


def summary_word_coverage(records, encode, decode, limit):
    totals = {}
    for record in records:
        words = record.target['text'].casefold().split()
        article = set(record.inputs['text'].casefold().split())
        prefix = set(decode(list(encode(record.inputs['text']))[:limit]).casefold().split())
        bucket = totals.setdefault(record.task_id, {'rows': 0, 'full': 0, 'kept': 0})
        bucket['rows'] += 1
        bucket['full'] += sum(word in article for word in words)
        bucket['kept'] += sum(word in article and word in prefix for word in words)
    for bucket in totals.values():
        bucket['coverage'] = bucket['kept'] / bucket['full'] if bucket['full'] else None
    return totals


@phase(0)
def run(directory, dataset, model_factory, settings, context, clock):
    final = directory / 'final' / 'batch'
    if clock.body.get('ready') and (final / 'complete.json').is_file():
        return save_report(directory, 0, clock.body['details'], {'phase0': clock.body['details']})
    model = model_factory()
    try:
        model.bind_runtime(directory, None)
        english = tuple(dataset.training_view('english'))
        rows = english[:settings.batch_size]
        trained = model.train(1, batch_factory=lambda: [TrainingBatch(dataset.collate(rows), ('current',) * len(rows))],
                              objective=Objective('per_example_mean'), hooks=None, limits=TrainLimits(1, 1, 100000))
        if not trained or not trained[0].success:
            raise ValueError('Phase 0 did not complete its update')
        summaries = []
        for start in range(0, min(20, len(english)), settings.analysis_batch):
            part = english[start:min(start + settings.analysis_batch, 20)]
            generated = model.predict(dataset.collate_sources(part), generation_spec(settings))
            summaries += [{'id': row.sample_id, 'target': row.target['text'], 'text': text}
                          for row, text in zip(part, generated['texts'])]
        if not summaries or not any(row['text'].strip() for row in summaries):
            raise ValueError('Phase 0 generated no nonempty summary')
        write_json(directory / 'summaries.json', summaries)
        Checkpoints(directory, 'phase0').pin_artifact(final, model.info.checkpoint_bytes,
                                                    lambda target: model.save(target, 'frozen'))
        coverage = summary_word_coverage(
            [record for language in ('english', 'hindi') for record in dataset.training_view(language)],
            model.encode, model.tokenizer.decode, model.settings.max_source_length)
        ratios = dataset.manifest.get('preprocessing', {}).get('report', {}).get('pilot', {})
        measured = trained[0].cost
        peak = measured.get('peak_device_allocated_bytes') or measured.get('process_peak_cpu_bytes')
        ran = {'name': 'batch', 'status': 'completed', 'steps': 1,
               'epochs': {'requested': len(rows) / len(english), 'completed': len(rows) / len(english), 'steps': 1},
               'accuracies': {}, 'time': {'wall_clock_seconds': clock.seconds(),
               'gpu_hours': clock.seconds() / 3600 if model.device.type != 'cpu' else 0,
               'peak_memory_mb': peak / 1024 ** 2 if peak else None}, 'checkpoint': str(final), 'final_model': str(final)}
        details = annotate({'trained': True, 'generated': True, 'summary': summaries[0],
            'readability_review': 'pending', 'summary_count': len(summaries),
            'tokens_per_character': {language: stats.get('tokens_per_character') for language, stats in ratios.items()},
            'truncation': coverage}, clock, [ran], str(final))
        clock.update(ready=True, details=details)
        return save_report(directory, 0, details, {'phase0': details})
    finally:
        model.release()
