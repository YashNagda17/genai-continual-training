# Cost-Efficient Incremental Fine-Tuning

Determine how much representative historical data is needed to update a fine-tuned model on new arrivals while preserving earlier knowledge. Vary replay budgets and compare with new-data-only fine-tuning and complete-data retraining. Measure old/new-task performance, forgetting, training time, and storage; seek performance close to complete retraining at lower cost.

## Current plan (2026-11-08)

The current plan is [Idea_Refined_2.md](/Idea_Refined_2.md). An English news summarizer (mBART, LoRA only) learns Hindi from XL-Sum. A fixed pool of 500 English examples is replayed at a fixed budget. The model's own writing decides what to revise, what to revise with, when to stop, and how small the adapter becomes, and every cost is counted: training, analysis, serving latency, and storage. It compares against Hindi-only fine-tuning, random replay, and replay of the full English set. Implementation design: [summarization-plan.md](/design/summarization-plan.md).

The table below retains the CSReL and TSR foundations of the original framing.
Their baseline adaptations are implemented under `src/pipelines/csrel.py` and `src/pipelines/tsr.py`.
They run through `python -m src.baseline_experiment`, separately from the five report phases.

## Retained paper foundations

| Stage | Paper | Role |
|---|---|---|
| Historical-task routing | [TSR, July 2026](https://arxiv.org/abs/2607.15587) | Prioritize compatible old tasks for transfer and route distillation teachers |
| Task-end memory construction | [CSReL, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/24a8d40f6656e542f3fd43bac678e71b-Abstract-Conference.html) | Train auxiliary scoring models and select reducible-loss coresets for later tasks |

Implementation boundaries: [design/README.md](/design/README.md). Paper details: [Papers/README.md](/src/docs/Papers/README.md).

