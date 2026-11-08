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

## Generation-aware replay additions

Evidence required for the [generation replay recipe](/design/generation-replay.md), in `src/tests/test_generation_replay.py`, using the tiny fixtures in `fakes.py` extended with `predict`, `token_loss`, and `collate_prompts`. Worked numbers come from [Idea_Refined.md](/Idea_Refined.md) and were made up to show the method, so these test arithmetic, not behavior.

| Area | Evidence required |
|---|---|
| Answer selection | Disagreement by hand, including 0 for identical answers; `gap` ordering across the messy/clear table; strong model still fitted on the whole pool under `candidate_limit`; exact group quota total and stable ties; seeded k-means reproducible; older-task reselection reusing stored strong scores; same seeds give same samples after resume |
| Past self | Routing formula with drift; `b = 0` equals TSR weights; greedy answer reproduced by the saved copy, so a pass at write time; EOS trim and `truncated`; rejected answers never replayed; older answers unchanged by reselection; commit is atomic with membership; `retain_copies=False` keeps files needed by a retained checkpoint |
| Repair and stop | `S = 2.575` and `w = 2.5` from `d = 1.03`, `g = 0.75`, `beta = 2`; targets 0.46 and 0.61; rounds use 7, 7, then 3 tokens; 260 generations for 60 rows, 20 failing, 5 tokens, 2 runs; gain 0 with no rollouts when nothing follows the token; negative gain clamped; floor zeroes small gains; shuffled and random policies reproducible |
| Clock | A turn counts when nothing is active; a row that slips after everything recovered is found by the next refresh; fall-through moves to another topic, then to current; `skip` terminates at replay probability 1 |
| Model execution | Hinge and weighted losses by hand; an all-inactive batch leaves every parameter and optimizer state unchanged, including under weight decay; truncated-row surprises equal full-row surprises at kept positions; `token_loss` agrees with `score` loss after reduction; refresh and writing preserve learner, optimizer, and training RNG |
| Data | `reserve_response` leaves room for generation on a long prompt; `collate_prefixes` order and left padding; held-out records rejected at prompt and prefix collation |
| Recovery | Interrupt during task-end writing, each refresh stage, and between refresh and the pending branch; compare with an uninterrupted run; row state round-trips through JSON with integer positions intact; no double count of turns |
| Measurement | Every generation and forward appears in one cost category; `max_task_seconds` stops at a safe boundary |

Do not treat these as evidence for retention, cost, or the main claim. Only the [first test and arms](/design/generation-replay-experiments.md) can show that.

## Summarization plan additions

Evidence required for the [summarization plan](/design/summarization-plan.md), in `src/tests/test_summarization.py`. Use a tiny `MT5Config` built locally (for example 2 encoder and 2 decoder blocks, `d_model` 32, a vocabulary of a few hundred tokens), with no downloads. Worked numbers come from [Idea_Refined_2.md](/Idea_Refined_2.md), which were made up to show the method, so these check arithmetic, not behavior.

| Area | Evidence required |
|---|---|
| Metrics | `chrf` by hand on a short pair, and 100 for identical strings; `script_share` on Latin, Devanagari, and mixed text; `paired_bootstrap` reproducible from its seed; `interaction(0, 3, 2, 3) = -2`; `top_overlap` of two hand rankings; `cohen_kappa` by hand; `energy_rank` on given singular values |
| Revision | B's ranking from the plan's table (Election +0.8 first, Science +0.1 low); drift `= 1` for a slipped summary; the half-ranked, half-random slot rule with stable ties and no duplicate in a batch; the G2 filter keeps and drops the plan's two rows; G3 weights average 1 per row and equal 1 when nothing got worse; the stop rule retires only at `sim >= 0.9` in English, and the next-ranked row takes the slot |
| Access | In the Hindi stage, only Hindi training rows and pool rows reach policy code; test rows are refused at every collation; validation reaches only declared decisions |
| Adapter plans | Trainable counts 2,555,904, 638,976, and 638,976 on the real per-rank costs (computed, not loaded); frozen adapters get no gradient and no optimizer update; with only late blocks trainable, no gradient is computed for earlier encoder blocks; new rank leaves outputs unchanged at the first step; E's rank arithmetic gives 14 and 659,456 for four decoder blocks, and 4 and 638,976 for the spread; `lr_scale` is 50% at agreement −0.5 and stops at the floor |
| Probe | Probe gradients leave the learner's weights, optimizer state, and RNG unchanged; grouped by cell and by block, they sum to the flat gradient |
| Shrink | Full-rank shrink reproduces outputs within tolerance, including the LoRA scale; `k_E` from the energy rule; the search steps down and stops at the first failure, or steps up when `k_E` fails |
| Generation | The script guard never emits a blocked token, and never masks EOS; greedy `predict` is deterministic; the merged copy matches unmerged outputs within tolerance; the prefix is stored on the prepared view |
| Model execution | Per-example seq2seq loss equals separate forwards; labels padded with `-100` don't count; bf16 or fp32 load, never fp16 |
| Recovery | Interrupt during past-summary writing, rescoring, and stop regeneration; compare with an uninterrupted run; the English checkpoint artifact is identical across all Hindi runs |
| Pre-registration | A scientific Hindi job is refused when the file is missing or its settings differ; C2 scientific jobs are refused before `c1_prediction.json` exists |

`SetupModelTests` gains one mT5-small case: load the pinned snapshot, check 300,176,768 parameters with an untied head, and run one forward and one greedy summary. It stays skipped unless `GENAI_SETUP_TESTS=1`.

These checks aren't evidence for retention, cost, or the main claims. Only the [scientific runs](/design/summarization-experiments.md) can show those.
