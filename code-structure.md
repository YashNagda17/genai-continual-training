# Code structure

All executable code belongs in `src/`. Use base classes with concrete subclasses; wildcards below stand for descriptive implementation names.

```text
src/
├── datasets/
│   ├── data/
│   ├── dataset_base.py
│   └── dataset_*.py
├── models/
│   ├── model_weights/
│   ├── model_base.py
│   └── model_*.py
├── pipelines/
│   ├── pipeline_base.py
│   └── <paper>.py
├── utils/
├── tests/
├── combined_pipeline.py
└── main.py
```

| Component | Responsibility / public operations |
|---|---|
| `DatasetBase` | Source processing; `fetch`, `preprocess`, `split`, `load`, `save`. Raw artifacts in `datasets/data/` |
| `ModelBase` | Transformer execution; `train(k)` for k logical passes, `evaluate`, `predict(x)`, `save`, `load`. Base assets in `models/model_weights/` |
| `PipelineBase` | One independently runnable paper method per module/package; `run`, `evaluate`, `predict`, `save`, `load` |
| `utils/` | Shared records, storage, configuration, and measurement |
| `CombinedPipeline` | Compose paper components across datasets/models into one hybrid run |
| `main.py` | Resolve and execute the configured model/method/task-sequence run matrix |

Detailed interfaces and ownership: [design/README.md](/design/README.md).

