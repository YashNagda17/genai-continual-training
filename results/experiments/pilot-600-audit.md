# CLINC 600-per-task pilot audit ? 2026-11-04

Three domains (banking, credit_cards, travel); 600 training and 150 validation examples per domain; seed 17; global memory 60; five learner/reference/subset passes; learning rate 0.001. Training was ordered, not shuffled. These are validation pilot results, not final test results.

| Method | Final exact match | Forgetting (points) | Recorded invocation minutes |
|---|---:|---:|---:|
| new_only | 30.67% | 41.00 | 21.63 |
| random | 81.56% | -0.67 | 28.32 |
| csrel | 54.44% | 18.00 | 47.34 |
| cumulative | 47.56% | 13.00 | 42.27 |

All four manifests and final lifecycle states recorded completion. New-only, random and CSReL each completed 2,250 learner updates; cumulative completed 4,500. Runtime comparisons are observational because execution conditions differed.

## Findings

All arms initially scored 39.33% on banking. Of 150 predictions, 60 were bill_balance and 29 order_checks. This is label concentration; a causal effect of training order is not established.

CSReL banking memory initially covered 7/15 intents; before travel it covered 3/15 (routing 15, account_blocked 10, report_fraud 5). Random covered 15/15 initially and 14/15 before travel. This is a plausible contributor to retention differences, not a proven sole cause.

Cumulative processed all seen records, but in task/intent blocks. Finite auxiliary passes do not establish convergence.

## Prepared ablation

The opt-in --shuffle-training flag permutes learner records per pass, including mixed-task cumulative views, independently of replay RNG. Auxiliary fitting order and memory-selection equations remain unchanged. Real-model results for this ablation and Kaggle execution are not yet available. Compare with identical data/hyperparameters in a new output directory.

Compact metrics: [pilot-600-results](/results/experiments/pilot-600-results.json). Large checkpoints, datasets, weights, and raw run outputs remain ignored.
