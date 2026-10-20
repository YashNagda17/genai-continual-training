# Cost-Efficient Incremental Fine-Tuning

Determine how much representative historical data is needed to update a fine-tuned model on new arrivals while preserving earlier knowledge. Vary replay budgets and compare with new-data-only fine-tuning and complete-data retraining. Measure old/new-task performance, forgetting, training time, and storage; seek performance close to complete retraining at lower cost.

## Pipeline

| Stage | Paper | Role |
|---|---|---|
| Current-data admission | [OASIS, ACL 2026](https://aclanthology.org/2026.acl-long.158/) | Select incoming examples using relative last-layer gradient informativeness and redundancy |
| Historical-task routing | [TSR, July 2026](https://arxiv.org/abs/2607.15587) | Prioritize compatible old tasks for transfer and route distillation teachers |
| Replay retrieval / layer updates | [aL-SAR, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/760564ebba4797d0dcf1678e96e8cbcb-Abstract-Conference.html) | Sample within the routed task using effective frequency; adaptively freeze layers |
| Task-end memory construction | [CSReL, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/24a8d40f6656e542f3fd43bac678e71b-Abstract-Conference.html) | Train auxiliary scoring models and select reducible-loss coresets for later tasks |

Implementation boundaries: [design/README.md](/design/README.md). Paper details: [Papers/README.md](/Papers/README.md).

