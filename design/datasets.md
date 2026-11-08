# Datasets and DatasetBase

Owns `src/datasets/`, `dataset_base.py`, and one concrete subclass per source format or processing behavior. Different dataset names with identical processing use configuration, not duplicate classes. CLINC, Schuster, Bitext, and TRACE share [InstructionDataset](/src/datasets/dataset_instruction.py). Their fetch classes in `dataset_sources.py` only convert source files into that JSONL schema. Raw storage is governed by [data.md](/design/data.md).

Produce identifiable, reproducible records and causal task views. Dataset code does not decide replay membership, importance scores, optimization, or paper-specific task grouping.

## Record contract

A canonical record contains `sample_id`, `dataset_id`, source revision/row identity, `task_id`, arrival position, split, input payload, target payload, and optional group/format metadata. Identity is stable across tokenization, shuffling, and resume. Preserve a source ID and duplicate-group ID when distinct observations share content; a content hash alone is not a unique observation ID.

For instruction tuning, preserve message roles and the supervised response separately. Rendering and tokenization may change representation, never the underlying target or provenance. Model-specific preprocessing receives a pinned tokenizer, template, truncation policy, and loss-mask policy; these enter the prepared-view fingerprint.

## Base-class operations

| Operation | Required behavior |
|---|---|
| `load(source_spec)` | Validate source identity/schema; return a raw view and provenance manifest. No implicit downloads during import. |
| `split(raw_view, split_spec)` | Return immutable train/validation/test membership. Preserve official splits unless an explicit protocol replaces them. |
| `preprocess(view, preprocessing_spec)` | Return a prepared view with the same IDs and split membership; report dropped/truncated records and reasons. |
| `fetch(view, batch_spec, cursor)` | Return the next batch plus next cursor; signal exhaustion explicitly. No silent wraparound, cross-split sampling, or global iterator. |
| `save(path)` | Persist source/split/preprocessing manifests and permitted processed artifacts; never save learner or replay-policy state. |

`load()` must also support reloading a saved manifest and verifying referenced artifacts. Keep views immutable; iterator cursors belong to the caller so two runs cannot advance each other's data.

The injected Python adapter additionally exposes `training_view(task_id)`, `evaluation_view(task_id)`, and `collate(records)`. Views are immutable tuples of canonical records; the initial implementation accepts finite task chunks. Canonical payloads and metadata must be JSON-serializable (large assets may use content-addressed references, whose backing bytes must be included in resource estimates). Collation creates backend tensors, masks, and counts without changing record order. The pipeline fetches only the open task's training view, and opens held-out views only in evaluation. `fingerprint` binds source, splits, preprocessing, and task membership for resume.

## Split and exposure rules

- Split before fitting any preprocessing statistics. Fit data-dependent transforms on training records only. Group related examples, conversations, documents, or alternate renderings to prevent leakage; preserve temporal order when required by the protocol.
- Apply source-level duplicate and overlap checks before assigning stream tasks. If authoritative benchmark splits overlap, report the overlap and the chosen protocol instead of silently changing the benchmark.
- Expose only the current task's training view and retained replay to policy code. The evaluation path receives held-out views separately. Knowing task metadata or a fixed task schedule is distinct from inspecting future task content.
- Keep source arrival order separate from training shuffle order. An epoch does not mean a second arrival. Single-pass mode cannot rewind data; task-chunk mode can revisit `D_t` only while that task is open.
- Task IDs are not class labels. A dataset adapter must declare whether genuine class/group annotations exist; it cannot manufacture classes from task IDs to satisfy aL-SAR.

## Batch contract

Model-provided preprocessing sets `allow_reasoning=False`. Targets containing `<think>`/`</think>` or `<analysis>`/`</analysis>` are dropped with a recorded reason, without silently rewriting responses. Older standalone preprocessing configurations keep their existing behavior unless this option is set.

`collate` builds native Torch tensor payloads, normally on CPU. The model adapter transfers them to its resolved device and chooses supported floating-point precision while preserving integer token IDs and boolean masks. Persist record manifests as plain JSON; transient collated tensors are reconstructed after resume.

Every batch carries ordered IDs, task/origin labels, tensors or payloads, valid-token masks, supervised-token masks, and valid counts. For causal SFT, supervise only the configured response tokens; exclude padding and masked prompt tokens. Record the EOS and truncation policy. Reject or explicitly drop examples with no supervised target tokens; do not turn them into zero-loss examples.

Variable-length collation must preserve ID alignment. Packing is optional and requires sample boundaries and segment masks that prevent unintended cross-example attention and allow per-example scoring. Disable packing on a scoring path until its per-example equivalence is validated.

## Generation-aware replay additions

Used by the [generation replay recipe](/design/generation-replay.md). Datasets stay unaware of practice answers.

- `InstructionPreprocessing.reserve_response: int = 0`. With `prompt_left`, keep `len(prompt) <= max_length - max(len(response), reserve_response)`. Today's rule fills `max_length` with prompt plus gold, so on a long-prompt row `predict` has no room to generate and raises its sequence bound. The value enters the prepared-view fingerprint. Zero preserves current behavior.
- `InstructionDataset.collate_prefixes(records, lengths)`. Left-pads `prompt_ids + target["ids"][:length]`, with no labels, in record order. Repeated IDs are allowed, for the two runs of a repair test.
- `collate` needs no change. A row from `practice_record` already carries its answer or cut as `target`, and its count as `tokens`.
- The dataset may not assign groups. Question groups are a policy decision in [answer-selection.md](/design/answer-selection.md). `class_id` stays genuine labels only.
- No new source or fetcher. The recipe uses the existing TRACE snapshot and `TraceSource`, plus CLINC for the class-group check.

## Summarization plan additions

Used by the [summarization plan](/design/summarization-plan.md). Summarization records need encoder-decoder processing, so they get their own subclass, `SummarizationDataset(DatasetBase)` in `src/datasets/dataset_summarization.py`. `InstructionDataset` is unchanged.

**Record.** `input = {"text": article}`, `target = {"text": summary}`, and `metadata.language` (`english`, `hindi`, and later `marathi`, `gujarati`). `task_id` is the language. `sample_id` comes from XL-Sum's article ID and language, so the same article never gets two IDs.

**`Seq2SeqPreprocessing`:**

| Field | Meaning |
|---|---|
| `tokenizer` | The pinned mT5 tokenizer |
| `max_source`, `max_target` | 256 and 64 in the plan. The source keeps its **first** tokens, since news puts facts first. The target keeps its first tokens, and the row records `target_truncated` |
| `prefix` | `None`, or a template such as `"summarize in {language}: "`, prepended to the article before tokenizing. G4 uses it. It enters the prepared-view fingerprint |

Rows with an empty article or summary are dropped with a recorded reason. Truncation counts are reported per language.

**Operations:**

- `collate(records)`: encoder input IDs and mask, plus labels padded with `-100`, in record order.
- `collate_sources(records)`: encoder inputs only, with no labels, for generation. Held-out records go through the same guard as elsewhere.
- `practice_record(record)` from utilities swaps the target to a past summary for G2. The dataset itself never sees past summaries.

**Protocol subsets.** The experiment runner, not the dataset, draws the fixed ID lists: 2,000 train, the 500-row English pool inside English train, 200 validation, and 500 test per language. The dataset exposes them as views, and the lists are saved in the manifest.

**Pilot measurements.** The dataset reports tokens per character per language, the share of articles longer than `max_source`, and the share of summaries longer than `max_target`. The runner computes the truncation-coverage check.

**Not used by this plan:** chat templates, `allow_reasoning`, `reserve_response`, `collate_prefixes`, and class or group annotations.
