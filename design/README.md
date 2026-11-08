# Implementation contract

Status: algorithm orchestration implemented with injected model/data adapters, 2026-10-20. See the [callable API and validation limits](/src/README.md). Production adapters, the experiment matrix, and training reproductions remain future work.

**Current plan, 2026-11-08:** [Idea_Refined_2.md](/Idea_Refined_2.md), English → Hindi summarization on mT5-small with LoRA only. Its implementation design is [summarization-plan.md](/design/summarization-plan.md), which also lists the old features this plan doesn't need and the new ones it adds. Model/data adapters and the three pipeline components are implemented with tiny CPU validation. The scientific experiment runner and reporting remain pending; no scientific result is claimed.

## Authority and scope

- These contracts resolve implementation details under [Idea.md](/Idea.md), [code-structure.md](/code-structure.md), and [ARCHITECTURE.md](/ARCHITECTURE.md). Change a contract before implementing incompatible behavior.
- **Current plan:** the [multilingual summarization plan](/design/summarization-plan.md). It uses its own revision, adapter-placement, and shrink pipelines, borrows CSReL's reducible-loss idea and TSR's fixed probe, and doesn't call OASIS, aL-SAR, the four-paper hybrid, or the generation replay recipe. Placement and shrink rules are in [summarization-experiments.md](/design/summarization-experiments.md); the model operations they call are in [models.md](/design/models.md).
- Implemented earlier methods, kept unchanged: **CSReL, aL-SAR, OASIS, TSR**. The earlier [generation replay recipe](/design/generation-replay.md) extends CSReL and TSR, adds our own [Repair and Stop mechanism](/design/repair-stop.md), and runs beside the four-paper hybrid. Neither is part of the current plan. Older shortlists/model suggestions are superseded. [Paper references](/Papers/README.md) hold derivations, source identities, evidence, and related work; component contracts hold implementation choices.
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
| `src/pipelines/csrel.py`, `CSReLPipeline` | [csrel.md](/design/csrel.md) |
| `src/pipelines/al_sar.py`, `ALSARPipeline` | [al-sar.md](/design/al-sar.md) |
| `src/pipelines/oasis.py`, `OASISPipeline` | [oasis.md](/design/oasis.md) |
| `src/pipelines/tsr.py`, `TSRPipeline` | [tsr.md](/design/tsr.md) |
| `src/utils/`, shared records and replay storage | [utils.md](/design/utils.md) |
| `src/tests/` | [tests.md](/design/tests.md) |
| `src/combined_pipeline.py`, `CombinedPipeline` | [combined-pipeline.md](/design/combined-pipeline.md) |
| `src/combined_pipeline.py`, `GenerationReplayPipeline` | [generation-replay.md](/design/generation-replay.md) |
| `src/pipelines/answer_selection.py`, `AnswerSelectionPipeline`, `FixedMemorySelection` | [answer-selection.md](/design/answer-selection.md) |
| `src/pipelines/past_self_replay.py`, `PastSelfReplayPipeline` | [past-self-replay.md](/design/past-self-replay.md) |
| `src/pipelines/repair_stop.py`, `RepairStopPipeline` | [repair-stop.md](/design/repair-stop.md) |
| `src/generation_replay_experiment.py` | [generation-replay-experiments.md](/design/generation-replay-experiments.md) |
| `src/example.py` | [main.md](/design/main.md) |
| `src/main.py` | [main.md](/design/main.md) |
| `src/csrel_experiment.py`, `MemoryBaselinePipeline` | [main.md](/design/main.md), [csrel.md](/design/csrel.md) |
| Current plan: guardrails, module layout, reuse, old and new features | [summarization-plan.md](/design/summarization-plan.md) |
| `src/models/model_seq2seq.py`, `src/models/model_mt5.py` | [models.md](/design/models.md) |
| `src/datasets/dataset_instruction.py` (`SummarizationDataset`), `XLSumSource` | [datasets.md](/design/datasets.md), [data.md](/design/data.md) |
| `src/pipelines/revision.py`, `RevisionPipeline` | [revision.md](/design/revision.md) |
| `src/pipelines/adapter_room.py`, `AdapterRoomPipeline`; `src/pipelines/generation_shrink.py`, `GenerationShrinkPipeline` | [summarization-experiments.md](/design/summarization-experiments.md), [models.md](/design/models.md) |
| `src/summarization_experiment.py` | [summarization-experiments.md](/design/summarization-experiments.md) |

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
11. Settings and predictions are pre-registered. A scientific run records the pre-registration hash and is refused if its settings differ.

## Dependency and ownership boundary

`example.py` is the present sample entry. The later matrix entry is `main.py`. Both call `combined_pipeline -> pipelines -> {models, datasets, utils}`. Models and datasets may depend on shared utility records, but not on pipeline implementations or each other. Tokenization is supplied to datasets through an injected preprocessing interface. Utilities cannot import concrete models, datasets, or pipelines.

| Concern | Single owner |
|---|---|
| Record identity, split membership, arrival order | Dataset adapter |
| Tensor execution, optimizer, scheduler, gradient accumulation | Model adapter |
| Paper equations and policy state | Corresponding paper pipeline |
| Task lifecycle, component composition, memory commit | Standalone pipeline or `CombinedPipeline` |
| Serialization, budget validation, metric arithmetic | Utilities |
| Sample run | `example.py` |
| Run matrix, config resolution, process exit status | `main.py` |

Validation requirements live in [tests.md](/design/tests.md); experiment controls live in [combined-pipeline.md](/design/combined-pipeline.md).

For the current plan, scope is one process and device, `google/mt5-small` (Qwen3-0.6B as the fallback), and XL-Sum English and Hindi, with the experiment runner as its entry point. Initial implementation scope for the earlier work is one process/device, the four standalone methods, and the specified hybrid. Distributed training, automatic task discovery, alternative recipes, and optional paper variants need separate contracts. Model/data choices and unresolved paper-code discrepancies remain explicit inputs, not inferred defaults.

use source ~/venv/bin/activate to activate virtual environment
