# Implementation contract

**Active-code scope (2026-11-08):** only the mT5/XL-Sum implementation of
[Idea_Refined_2.md](/Idea_Refined_2.md), its three pipelines, phase driver, and
supporting tests remain executable. Legacy paper pipelines, causal adapters,
instruction datasets, and their experiment scripts/tests are removed at the
user's request. Paper sources, reviews, and historical design documents remain
research references; their old code maps are not active interfaces.

Status: the active model/data adapters, three pipelines, and Phase 0–7 driver are implemented. See [execution and validation limits](/src/README.md). Scientific runs and training reproductions have not been performed.

**Current plan, 2026-11-08:** [Idea_Refined_2.md](/Idea_Refined_2.md), English → Hindi summarization on mT5-small with LoRA only. Its implementation design is [summarization-plan.md](/design/summarization-plan.md). Phases share resume, final exports and JSON/CSV/Markdown logging. Validation uses tiny CPU models; no scientific result is claimed.

## Authority and scope

- These contracts resolve implementation details under [Idea.md](/Idea.md), [code-structure.md](/code-structure.md), and [ARCHITECTURE.md](/ARCHITECTURE.md). Change a contract before implementing incompatible behavior.
- **Current plan:** the [multilingual summarization plan](/design/summarization-plan.md). It uses its own revision, adapter-placement, and shrink pipelines, borrows CSReL's reducible-loss idea and TSR's fixed probe, and doesn't call OASIS, aL-SAR, the four-paper hybrid, or the generation replay recipe. Placement and shrink rules are in [summarization-experiments.md](/design/summarization-experiments.md); the model operations they call are in [models.md](/design/models.md).
- Earlier CSReL, aL-SAR, OASIS, TSR and generation-replay designs are historical research references. Their executable code is removed. [Paper references](/Papers/README.md) retain derivations, source identities, evidence and related work.
- Record mechanism/access changes as distinct adaptations or ablations. Reproduction claims require resolving source ambiguities against pinned code and validating the stated protocol.
- All executable code, including tests/plotting, belongs in `src/`; reports and run outputs belong in `results/`. This overrides the generic skill code layout. Existing `src/litreview/` and `src/researchflow/` remain workflow support.
- [SKILLS.md](/SKILLS.md) routes research work; [general guidelines](/skills/general_guidelines.md) define shared workflow conventions.
- Model/data identities, hardware, optimization settings, numeric budgets, and experimental acceptance margins are later inputs.

## SLM and resource constraints

- **Maximum 700,000,000 total parameters per model**, including learners, teachers, probes, and auxiliary models. Count frozen/trainable base, heads, and adapters; tied parameters count once. Smaller models are allowed; quantization does not change the count. Larger published models require a labeled scale adaptation within this ceiling.
- Preflight aggregate CPU/GPU/disk budgets, including resident roles, activations, gradients, optimizer state, logits, replay, caches, and checkpoint staging. Runs are sequential by default.
- [model-weights.md](/design/model-weights.md) owns mandatory phase-aware resume, rolling retention, and bounded residency for learner and auxiliary training.
- Log candidate algorithmic improvements in [optimizations.md](/optimizations.md).

## Document map

Read this shared contract once, then the affected component and paper reference.

| Code area / class | Owning document |
|---|---|
| `src/` and shared invariants | This document |
| `src/datasets/`, `dataset_base.py`, `DatasetBase`, concrete dataset subclasses | [datasets.md](/design/datasets.md) |
| `src/datasets/data/` | [data.md](/design/data.md) |
| `src/models/`, `model_base.py`, `ModelBase`, concrete model subclasses | [models.md](/design/models.md) |
| `src/models/model_weights/` | [model-weights.md](/design/model-weights.md) |
| `src/pipelines/`, `pipeline_base.py`, `PipelineBase` | [pipelines.md](/design/pipelines.md) |
| `src/utils/`, shared records and replay storage | [utils.md](/design/utils.md) |
| `src/tests/` | [tests.md](/design/tests.md) |
| Current plan: guardrails, module layout, reuse, old and new features | [summarization-plan.md](/design/summarization-plan.md) |
| `src/models/model_seq2seq.py`, `src/models/model_mt5.py` | [models.md](/design/models.md) |
| `src/datasets/dataset_instruction.py` (`SummarizationDataset`), `XLSumSource` | [datasets.md](/design/datasets.md), [data.md](/design/data.md) |
| `src/pipelines/revision.py`, `RevisionPipeline` | [revision.md](/design/revision.md) |
| `src/pipelines/adapter_room.py`, `AdapterRoomPipeline`; `src/pipelines/generation_shrink.py`, `GenerationShrinkPipeline` | [summarization-experiments.md](/design/summarization-experiments.md), [models.md](/design/models.md) |
| `src/summarization_experiment.py`, `src/phases/phase0.py` … `phase7.py` | [summarization-experiments.md](/design/summarization-experiments.md) |

A paper module may become a same-named package while preserving its public contract.

## Research invariants

1. At task `t`, training may access current training data `D_t` and the explicitly retained history `M_<t`. Validation is for declared tuning/evaluation only. Test and future-task records never enter selection, probes, reference-model training, replay, or gradient computation.
2. The proposed hybrid uses **task chunks**: `D_t` remains accessible through task-end coreset construction. Reject a single-pass configuration for this recipe. Rejected OASIS samples are still observed training data and remain eligible for task-end CSReL under this access policy.
3. Account separately for retained replay payload, historical candidate access, and auxiliary state; an example-count cap does not bound total storage.
4. At task boundaries, replay contains only **CSReL-retained** records; OASIS admission is a separate decision. Archives and recovery-only payloads cannot be reopened through the active training view. Count historical probes/reference access against the same policy.
5. One model executor owns learner updates. Paper components supply policies; auxiliary training/probing remains isolated from learner state.
6. A score is tied to its sample IDs, model revision, parameter view, loss reduction, and computation time. Do not reuse a score after those assumptions change unless the paper explicitly defines it as persistent state. The recipe's stored baselines and age-limited repair gains are declared persistent in [repair-stop.md](/design/repair-stop.md).
7. Standalone and hybrid runs have distinct identities; see [pipelines.md](/design/pipelines.md).

**Current plan invariants.** Invariants 2 and 4 describe the task-chunk hybrid and don't apply to the current plan. In their place:

8. The task stream is English, then Hindi. In the Hindi stage, training and policy code may access the Hindi training view and the fixed **English pool** of 500 records, with the past summaries and baselines saved for them. No other English training data is accessible, except to the declared `full_english` reference arm.
9. Validation is read only for declared decisions: English checkpoint choice, winner rules, the C1 prediction check, and F's shrink test. Test reaches only evaluation and reports.
10. Every Hindi-stage run loads the same immutable English checkpoint for its prefix variant. It is a new run with transferred weights, not a resume.
11. Scientific and pilot runs use the fixed settings in [summarization-experiments.md](/design/summarization-experiments.md). A finished report is reused only when those settings and the prepared data still match.

## Dependency and ownership boundary

The entry point is `summarization_experiment.py`: `phases -> pipelines -> {models, datasets, utils}`. Models and datasets may depend on shared utility records, but not on pipeline implementations or each other. Tokenization is injected into datasets. Utilities cannot import concrete models, datasets, or pipelines. Earlier paper contracts remain historical references alongside the sources in [Papers/](/Papers/).

| Concern | Single owner |
|---|---|
| Record identity, split membership, arrival order | Dataset adapter |
| Tensor execution, optimizer, scheduler, gradient accumulation | Model adapter |
| Revision, maps, adapter plans and shrink policy state | Corresponding active pipeline |
| Task lifecycle and replay commit | The active pipeline and shared runtime |
| Serialization, budget validation, metric arithmetic | Utilities |
| Local smoke preparation | `phases/data.py` |
| Phase order, config resolution, process exit status | `summarization_experiment.py` |

Validation requirements live in [tests.md](/design/tests.md); experiment controls live in [combined-pipeline.md](/design/combined-pipeline.md).

For the current plan, scope is one process and device, `google/mt5-small` (Qwen3-0.6B as the fallback), and XL-Sum English and Hindi, with the experiment runner as its entry point. Initial implementation scope for the earlier work is one process/device, the four standalone methods, and the specified hybrid. Distributed training, automatic task discovery, alternative recipes, and optional paper variants need separate contracts. Model/data choices and unresolved paper-code discrepancies remain explicit inputs, not inferred defaults.

use source ~/venv/bin/activate to activate virtual environment
