# Datasets and DatasetBase

Owns `src/datasets/`, `dataset_base.py`, and one concrete subclass per source format or processing behavior. Different dataset names with identical processing use configuration, not duplicate classes. Raw storage is governed by [data.md](/design/data.md).

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

