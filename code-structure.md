# Code structure

Executable code belongs in `src/`. The active implementation follows [Idea_Refined_2.md](/Idea_Refined_2.md).

```text
src/
├── datasets/
│   ├── dataset_base.py
│   ├── dataset_instruction.py   # SummarizationDataset, Seq2SeqPreprocessing
│   ├── dataset_sources.py       # XL-Sum download and canonical JSONL
│   └── data/                    # Local data assets
├── models/
│   ├── model_base.py
│   ├── model_seq2seq.py         # Native Torch seq2seq execution and adapters
│   ├── model_mt5.py
│   └── model_weights/           # Local base assets
├── pipelines/
│   ├── pipeline_base.py
│   ├── revision.py
│   ├── adapter_room.py
│   └── generation_shrink.py
├── phases/
│   ├── phase0.py … phase7.py    # Experiment decisions
│   ├── settings.py             # Protocol settings
│   ├── data.py                 # Subsets, prefix preparation, smoke fixture
│   ├── support.py              # Shared execution and resume lifecycle
│   ├── artifacts.py            # Model exports and reports
│   └── serving.py              # One timing pass, no warm-up
├── summarization_experiment.py
├── utils/
├── tests/                      # Summarization and phase behavior
├── litreview/                  # Preserved research support
└── researchflow/               # Preserved paper workflow support
```

| Boundary | Owner |
|---|---|
| Source identity, official splits, selected IDs, tokens, collation | `DatasetBase` / `SummarizationDataset` |
| Torch training, inference, adapter operations, device and optimizer state | `ModelBase` / `Seq2SeqLMModel` |
| Replay, location maps, block plans, shrink search | The three concrete pipelines |
| Experiment order, gates, seeded arms, winner selection | Phase modules and shared execution |
| Checkpoint transactions, retention, numerical metrics and resource accounting | `utils/` |
| Phase JSON/CSV/Markdown, logs and final adapter exports | `phases/artifacts.py` |

Use namespace packages and direct module imports. Constructors and imports do not download assets. Model roles use available CUDA/ROCm or XPU with CPU fallback and stay below 700M total parameters. [Design contracts](/design/README.md) define the interfaces. Legacy pipeline/model/dataset code is removed; [paper references](/Papers/README.md) and research workflows remain.

