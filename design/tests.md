# Correctness and fidelity checks

Owns `src/tests/`. Test mathematical and lifecycle boundaries that could invalidate the research. Use deterministic tiny records/models; routine checks must not require downloading production weights or running a full experiment.

Keep the routine suite compact: hand-computed equations, weight/policy handoff, device routing, standalone/hybrid smoke runs, and representative recovery failures. Consolidate repeated setup and failure phases into table-driven cases. Avoid tests of the fixture itself, Python abstract-class mechanics, or compositions checked only against the same helper calls. The checklist below also includes later production-adapter validation; a passing tiny-model suite does not establish all of it.

## Required checks by owner

| Area | Evidence required before integration |
|---|---|
| Datasets | Stable identity/cursor and manifest reload; disjoint split groups; target-preserving preprocessing; response/padding masks; no target in generation prompt; future/evicted-record access denied |
| Model execution | Per-example loss/gradient agreement with separate forwards/backwards; correct variable-length reduction; probe/evaluation state isolation; teacher immutability; accumulation equivalence |
| Replay storage | Global/per-task units and caps; zero capacity; deterministic allocation; eviction denies active access while recovery-only blobs remain inaccessible and counted |
| CSReL | Loss-difference ordering including noisy examples; round-dependent ranking; empty initialization; zero quota/underfill; old-task reselection restricted to retained records; reference-loss provenance |
| aL-SAR | Signed similarity/probabilities; active-population decay and duplicate counts; native/hybrid eviction; cold start and zero denominator; staged-state rollback on skipped updates; prefix/cadence; frozen-weight invariance and reactivation; reject unsupported accumulation |
| OASIS | Last-layer per-example scores; exact small SIREN including higher-order terms and negative cosine; prior-state normalization; reproducible shared-threshold sampling and calibration integral; empty admission and task-boundary state continuity |
| TSR | Common-initialization LoRA-B signature; routing distribution; batch-level replay choice; origin-matched teachers and current-task top teacher; KL direction and masks; empty-history behavior |
| Composition | One learner optimizer; TSR source/teacher preserved by SAR; no OASIS replay filtering; no branch redraw after rejected arrivals; empty/exhausted current source and replay probabilities zero/one terminate; skipped-update attempts bounded; auxiliary access logged separately |
| Recovery and measurement | Interrupt learner/auxiliary training, admission, probe, memory construction, and evaluation; restore queues/RNG/staged state; compare with uninterrupted execution; task commit before evaluation; torn-save fallback; missing teacher is an error; no duplicate events |
| Run attempts | Separate attempt directories; `auto|required|never` semantics; explicit choice for multiple attempts; no overwrite or state merge across attempts |
| Retention and residency | Latest-plus-one pruning only after commit; preserve live teacher/replay/fallback references; completion cleanup; bounded CPU/GPU teacher loading; include staging peaks; save failure preserves the prior state |
| SLM scope | Reject more than 700M total parameters in every role, including a frozen/quantized large base with a small adapter; verify tied parameters count once and total/trainable counts differ |
| Evaluation | Task-matrix arithmetic including negative forgetting, null first-task values, metric direction/scale, and distinction between before-task and after-learning transfer |

## Validation sequence

1. **Contract tests:** hand-computable cases and tiny Torch CPU model checks for supported numerical operations; no accelerator is required.
2. **Algorithm checks:** compare a minimal implementation against the paper equations and, where available, a pinned official implementation. Exercise a case that distinguishes the real mechanism from the prohibited shortcut; matching method names or output shapes is insufficient.
3. **Two-task smoke run:** test standalone pipelines first, then a compatible hybrid. Include current selection, replay, teacher routing, task-end coreset construction, evaluation, and checkpoint restore. State any features the tiny model cannot exercise.
4. **Later experimental validation:** use the supplied models/datasets/protocol, repeated seeds, matched controls, and complete cost/storage reporting. This stage establishes empirical behavior; earlier stages cannot establish accuracy, retention, or speedup claims.

Keep ablations distinguishable from missing features. A test skipping unavailable gradient support must not mark a pipeline fully validated. Record failure and unavailable checks explicitly. Do not write brittle tests for Markdown wording, class-name existence alone, or private implementation structure.

Report executed, failed, and unavailable checks.

## Setup check

`SetupModelTests` in [test_pipelines.py](/src/tests/test_pipelines.py) loads the three cached causal language models and runs one short forward on the resolved device. It is skipped unless `GENAI_SETUP_TESTS=1`.

Run it only when the task is setup: confirming downloaded weights, the Torch build, or accelerator placement. Do not run it for algorithm, pipeline, contract, or paper changes, and do not add it to the routine command below. A passing routine suite does not require this check.

```sh
GENAI_SETUP_TESTS=1 python -m unittest src.tests.test_pipelines.SetupModelTests -v
```

`resolve_device` prints to stderr when the resolved device is CPU, including an explicit `cpu` choice and a fallback from a missing CUDA or XPU device. That line means the run is not on an accelerator; pass `device="xpu:0"` or `device="cuda:0"` when that device is present.
