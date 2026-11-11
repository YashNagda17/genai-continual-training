# Multilingual summarization plan

Implementation design for [Idea_Refined_2.md](/Idea_Refined_2.md), the final experiment plan: teach an English news summarizer to write Hindi with LoRA alone, letting the model's own writing decide what to revise. This file owns the guardrails, the module layout, the reuse map, and the list of old features this plan does not need.

Status: model/data adapters, the three pipelines and Phase 0–7 orchestration are implemented with tiny CPU validation. The driver saves recovery state, final adapters and JSON/CSV/Markdown reports. Scientific execution and paper-level interpretation remain unmeasured; see the [execution guide](/src/README.md).

Read with: [revision.md](/design/revision.md) (experiments A, B, G1, G2, G3), [models.md](/design/models.md) (adapter names, `AdapterPlan`, probe gradients, shrinking), and [summarization-experiments.md](/design/summarization-experiments.md) (C1, C2, E, F, phases, arms, settings, statistics, and reports).

## What changes from the earlier recipes

| | Earlier recipes | This plan |
|---|---|---|
| Task stream | TRACE or CLINC topics, many tasks | Two languages: English, then Hindi. A third language is optional |
| Model | Causal LMs (SmolLM2, LFM2.5, Qwen3) | **mT5-small**, an encoder-decoder. Qwen3-0.6B is the fallback |
| Fine-tuning | LoRA by default, full fine-tuning allowed | **LoRA only**. Base weights are always frozen |
| Replay memory | Built at task end by CSReL or OASIS | A **fixed pool of 500 English training examples**, saved at the English stage |
| Methods | OASIS, aL-SAR, TSR, CSReL, generation replay | Our own plan: revision policies, adapter placement, and generation-tested shrinking. It borrows CSReL's reducible-loss idea and TSR's fixed probe |
| Evaluation | Task matrix | English and Hindi before and after the Hindi update, with and without a script guard |

## Design guardrails

1. **LoRA only, one model under 700M.** mT5-small has 300,176,768 parameters. Every role, including the probe, stays under the [ceiling](/design/README.md).
2. **mT5-small and XL-Sum are explicitly requested** by the plan. This is the allowed exception to "existing assets first". No other new model, dataset, or inference backend is added.
3. **Reuse before adding.** Use existing code if it fits, then subclass it, and only then write new code. Every new item appears in the tables below.
4. **Change the owner first.** Each concern has one [owning contract](/design/README.md). Shared additions go in that contract's "Summarization plan additions" section. Paper contracts (`csrel.md`, `tsr.md`, `oasis.md`, `al-sar.md`) and the earlier recipe contracts don't change, apart from a one-line "Current plan" status note at the top of each.
5. **Every arm has a question, a control, and a decision rule.** These are listed in [summarization-experiments.md](/design/summarization-experiments.md).
6. **Numbers are fixed before the runs.** Settings and predictions are pre-registered, and the runner refuses a scientific Hindi run whose settings differ from the pre-registered file.
7. **Decisions use validation, never test.** "Best" arms are chosen on validation by a pre-registered rule. Test scores are computed only for reporting.
8. **Honest claims.** Nothing counts until it's measured in `results/`. Steps A–D are measurements. G1–G3, E, and F are labelled `project` mechanisms ([method identity](/design/pipelines.md)). Novelty stays unverified.
9. **Ideas go in [optimizations.md](/optimizations.md)**, not in design files.

## Code guardrails

| Area | Rule |
|---|---|
| Active scope | Keep only this plan's models, source, pipelines and behavior tests. Earlier paper references and research workflows remain available |
| Layout | Modules go where the table below says. Namespace packages, relative imports, no `__init__.py` |
| Dependencies | Models contain no method or arm names. Datasets know nothing of replay or past summaries. Utilities import no concrete model, dataset, or pipeline. The three new pipelines never import each other; the experiment runner composes them |
| Numerics | Native Torch for training, calculations, and generation, with GPU detection and CPU fallback through `resolve_device`. mT5 runs in fp32 |
| Learner updates | Only the executor steps an optimizer. `predict`, `score`, probes, and shrink tests preserve weights, optimizer state, modes, and training RNG |
| Data access | During the Hindi stage, training and policy code see only the Hindi training view and the English pool. Validation is read only for declared decisions. Test reaches only evaluation |
| State | Policy state is plain JSON. Each component has its own RNG. Generation is greedy, so it needs no seed |
| Resume | Long loops keep a cursor and stop at safe boundaries, as in existing pipelines ([model-weights.md](/design/model-weights.md)) |
| Cost | Every training, scoring, generation, probe, and timing pass records a [cost category](/design/utils.md) |
| Tests | Compact, table-driven checks in `src/tests/test_summarization.py` on a tiny locally built mT5 config, with no downloads ([tests.md](/design/tests.md)) |
| Outputs | The current directory, unless `--output` names another one. Each arm and seed gets its own attempt directory. Never delete an existing attempt |

## Module layout

### New modules

| Module | Contents | Base | Contract |
|---|---|---|---|
| `src/models/model_seq2seq.py` | `Seq2SeqSettings`, `Seq2SeqLMModel`: the shared encoder-decoder executor | `ModelBase` | [models.md](/design/models.md) |
| `src/models/model_mt5.py` | `MT5SmallModel`, `model_id = "google/mt5-small"` | `Seq2SeqLMModel` | [models.md](/design/models.md) |
| `src/datasets/dataset_instruction.py` | `Seq2SeqPreprocessing`, `SummarizationDataset` | `DatasetBase` | [datasets.md](/design/datasets.md) |
| `src/pipelines/revision.py` | `RevisionConfig`, `RevisionPipeline`: experiments X, A, B, G1, G2, G3, the full-English reference | `PipelineBase` | [revision.md](/design/revision.md) |
| `src/pipelines/adapter_room.py` | `AdapterRoomConfig`, `AdapterRoomPipeline`: maps (C1), block plans (C2), new room (E) | `PipelineBase` | [summarization-experiments.md](/design/summarization-experiments.md), [models.md](/design/models.md) |
| `src/pipelines/generation_shrink.py` | `ShrinkConfig`, `GenerationShrinkPipeline`: experiment F | `PipelineBase` | [summarization-experiments.md](/design/summarization-experiments.md), [models.md](/design/models.md) |
| `src/summarization_experiment.py` | Orders the phases, prepares data, and writes the attempt root | None | [summarization-experiments.md](/design/summarization-experiments.md) |
| `src/phases/phase0.py` … `phase7.py` | One phase each, calling the existing pipelines | None | [summarization-experiments.md](/design/summarization-experiments.md) |
| `src/tests/test_summarization.py` | Plan checks | None | [tests.md](/design/tests.md) |

### Additions to existing modules

| Module | Addition | Contract |
|---|---|---|
| `src/datasets/dataset_sources.py` | `XLSumSource` | [data.md](/design/data.md) |
| `src/utils/contracts.py` | `AdapterPlan`, `GenerationSpec.script_guard` | [utils.md](/design/utils.md) |
| `src/utils/numerics.py` | `chrf`, `script_share`, `paired_bootstrap`, `interaction`, `top_overlap`, `cohen_kappa`, `energy_rank` | [utils.md](/design/utils.md) |
| `src/tests/fakes.py` | A tiny seq2seq fixture with `predict`, `score`, and adapter plans | [tests.md](/design/tests.md) |

## Reuse map

| Plan step | Reused as it is | New on top |
|---|---|---|
| Two-stage stream | `PipelineRuntime.run`, `current_batch`, `ReplayStore`, `LimitedViews` | Branching every Hindi run from one saved English checkpoint |
| Pool and past summaries | `Record.practice` (`answer_text`, `answer_ids`, `baseline`, `writer`), `practice_record` | `writer = "english_checkpoint"` |
| Loss-based choice (B) | `score` with per-example target loss, the CSReL reducible-loss idea | Ranking by loss increase from the saved checkpoint |
| Repair weights (G3) | `Objective.token_mode = "weighted"`, `TokenTargets`, the no-step rule on zero active tokens | Weights from `log p_before - log p_now` |
| Probe gradients (C1, E) | TSR's fixed zero-B LoRA probe idea and `fork(role, initialization)` | Per-cell and per-block gradients on an encoder-decoder |
| Similarity and ROUGE | `numerics.rouge_l` | `numerics.chrf` |
| Baseline arms | `MemoryBaselinePipeline` naming (`new_only`, `random`, `cumulative`) | Mapped to X, A, and the full-English reference |
| Resume and retention | `PipelineRuntime.save`, `load`, `boundary`, `Checkpoints.publish` | An immutable English-checkpoint artifact shared by all arms |
| Reports | `src/phases/artifacts.py` owns shared records, final exports and JSON/CSV/Markdown | Bootstrap deltas, interaction, overlap, explicit decision/status fields |

## Old features this plan does not need

Their executable code and dedicated tests are removed. Original paper sources, reviews, research workflows and historical design references remain available.

| Feature | Owner | Why it isn't needed |
|---|---|---|
| OASIS admission | [oasis.md](/design/oasis.md) | Every Hindi example is trained on. There's no admission decision |
| aL-SAR retrieval and freezing | [al-sar.md](/design/al-sar.md) | Block choice is a fixed, pre-registered plan, not a Fisher-based prefix |
| Four-paper hybrid | [combined-pipeline.md](/design/combined-pipeline.md) | Not a comparison arm in this plan |
| TSR routing, era teachers, KD | [tsr.md](/design/tsr.md) | One old language, so there's nothing to route between. No distillation. Only the fixed probe is reused. Routing returns only with a third language (Phase 7) |
| CSReL reference and subset model training | [csrel.md](/design/csrel.md) | The English checkpoint is the reference. No helper models are trained |
| Generation replay recipe: answer disagreement, drift routing, repair test rollouts, hinge loss, sentence cuts | [generation-replay.md](/design/generation-replay.md) and its parts | Replaced by the simpler G1–G3 rules in [revision.md](/design/revision.md) |
| `prompt_embedding`, k-means groups | [models.md](/design/models.md), [answer-selection.md](/design/answer-selection.md) | No question groups |
| `reserve_response` | [datasets.md](/design/datasets.md) | Encoder-decoder inputs and outputs have separate length limits |
| Chat templates, `allow_reasoning` | [datasets.md](/design/datasets.md) | mT5 takes plain text, optionally with a prefix |
| Full fine-tuning mode, Fisher, SAR coordinates, OASIS last-layer gradients | [models.md](/design/models.md) | LoRA only, and no paper method that needs them |
| CLINC, Schuster, Bitext, and TRACE sources | [data.md](/design/data.md) | XL-Sum only |

## New features this plan needs

| Feature | Owner |
|---|---|
| Encoder-decoder executor for mT5-small, with LoRA on self-attention, cross-attention, and feed-forward projections | [models.md](/design/models.md) |
| Freezing adapters by block, so that frozen blocks also stop the backward pass | [models.md](/design/models.md) |
| Extra zero-start rank per block, and learning-rate multipliers per adapter group | [models.md](/design/models.md) |
| Per-cell probe gradients | [models.md](/design/models.md) |
| SVD shrinking of the effective update, and a merged inference copy | [models.md](/design/models.md) |
| Which blocks train, where new rank goes, and how far to shrink | [summarization-experiments.md](/design/summarization-experiments.md) |
| Greedy generation with a script guard (blocked and penalized token masks) | [models.md](/design/models.md) |
| XL-Sum source, summarization records, an optional language prefix | [data.md](/design/data.md), [datasets.md](/design/datasets.md) |
| Revision policies: random, loss increase, drift, past summaries, repair and stop | [revision.md](/design/revision.md) |
| chrF, script share, paired bootstrap, interaction, overlap, kappa, energy rank | [utils.md](/design/utils.md) |
| Fixed settings, validation-based winner rules, serving timing, blind manual-check sheets | [summarization-experiments.md](/design/summarization-experiments.md) |

## Run lifecycle

```text
English stage (once per prefix variant):
  train on 2,000 English rows for all epochs, then validate and keep final-epoch weights
  save the English checkpoint as an immutable artifact
  write past summaries and per-token baselines for the 500 pool rows
  score both languages on validation and test, and time one summary

Hindi stage (once per arm and seed):
  load the English checkpoint and the pool
  apply the arm's AdapterPlan (default: all blocks, no new room)
  train on Hindi, with the arm's revision policy filling replay slots
  save the Hindi checkpoint
  evaluate both languages, with and without the script guard

Analysis jobs (no training):
  C1 maps on the English checkpoint, F shrink on chosen checkpoints, serving timing
```

Each Hindi-stage run is a new run that loads weights from the English checkpoint. It isn't a resume of the English run.

## Build order

Each step is usable and tested before the next. Stop where a [gate](/design/summarization-experiments.md) says to.

| Step | Builds | Why here |
|---|---|---|
| S0 | XL-Sum source, `SummarizationDataset` in `dataset_instruction.py`, `Seq2SeqLMModel` with LoRA, greedy generation, chrF and script share | Phase 0: one batch and one summary |
| S1 | English stage, pool writing, X, script guard, serving timing | Phase 1 and the gate |
| S2 | `RevisionPipeline`: random, loss, drift, past | Tier 1 revision arms |
| S3 | Block freezing with backward cut, `AdapterRoomPipeline` maps and block plans | C1, C2, D |
| S4 | Generation shrink | F needs only finished checkpoints |
| S5 | Repair and stop, extra rank, learning-rate multipliers | Tier 2: G3 and E |
| S6 | Prefix branch, compute-matched control | Tier 3 |
