# Datasets and DatasetBase

Only XL-Sum English/Hindi and `SummarizationDataset` are active. [dataset_sources.py](/src/datasets/dataset_sources.py) downloads the official language archives and converts them to canonical JSONL; [dataset_instruction.py](/src/datasets/dataset_instruction.py) contains `SummarizationDataset` and `Seq2SeqPreprocessing`. The filename is retained for direct-import compatibility. Instruction datasets and other source converters are removed.

The dataset owns source identity, official split membership, immutable finite views, preprocessing and collation. It never chooses replay or imports a model/pipeline. A record keeps its source article ID, language task, arrival order, split, article and reference summary. Manifests record plain settings and ordered protocol IDs rather than content hashes.

| Operation | Required behavior |
|---|---|
| `load` | Read canonical JSONL or saved prepared records/manifest without downloading |
| `split` | Keep official train/validation/test membership; report overlap |
| `preprocess` | Tokenize article and summary separately, preserve IDs, report empty/truncated rows |
| `training_view`, `evaluation_view` | Return immutable ordered tuples, respecting exposed protocol subsets |
| `expose` | Bind the runner's ordered ID list to a named view and persist it in the manifest |
| `save` | Save prepared records and source/split/preprocessing/subset settings |

The unused dataset `fetch` method and `has_classes` property have been removed
from the base interface and implementation. Runtime code owns batch cursors and
uses `collate` directly. Source downloading still uses `DatasetSource.fetch`.

The runner draws 2,000 training, 200 validation, and 500 test articles per language, and a 500-row English pool inside English training. It selects before tokenization so unused official rows are not processed. Every selected article is then prefixed with `summarize: ` so the encoder is told the task. A later language-prefix variant preserves exactly those same selected IDs.

`Seq2SeqPreprocessing` receives model-provided encode/tokenizer identity, EOS/pad IDs, source/target limits (256/64), and the `summarize: ` task prompt. Phase 6 may replace that prompt with `"summarize in {language}: "`. Article and summary retain their first tokens and end with EOS. Empty article/summary rows are reported and dropped. The manifest records prefix, limits, tokenizer and per-language pilot measurements, including tokens per character and truncation counts.

`collate(records)` right-pads encoder IDs and boolean attention masks, with target padding masked by `-100`. `collate_sources(records)` supplies encoder inputs only. The model transfers transient Torch batches to its resolved device without changing IDs/mask dtypes. `practice_record` swaps a reference target for a saved past summary outside dataset code.

Labeled payloads also include `supervised_indices`: a flat int64 tensor computed on CPU, indexing non-padding positions in the row-major flattened labels. Scoring and weighted training transfer these indices with the batch and use fixed-size device gathers instead of GPU boolean compaction.

Collation explicitly allocates on CPU even when an ambient Torch device context selects an accelerator. Dataset views are never migrated to GPU. Device batch copies are owned by the model executor and must not be retained by completed evaluation sweeps.

Hindi training and policy collation may access only Hindi training rows and the fixed English pool; the declared full-English reference is the exception. Validation reaches declared decisions, and test reaches evaluation/serving/reports. Access guards belong to the pipeline. Class/group annotations, chat templates and causal prefix collation are not used.
