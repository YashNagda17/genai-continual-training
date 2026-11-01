# Challenge Report: Upgrades to the Incremental Fine-Tuning Hybrid

Date: 2026-11-01 | Venue target(s): ICLR / NeurIPS / ACL, inferred from the current stack because the request named none | Lit review used: [literature-review-pipeline-upgrades-2026-11-01.md](/results/literature-review/literature-review-pipeline-upgrades-2026-11-01.md)

## Idea Statement (verbatim)

From [Idea.md](/Idea.md):

> Determine how much representative historical data is needed to update a fine-tuned model on new arrivals while preserving earlier knowledge. Vary replay budgets and compare with new-data-only fine-tuning and complete-data retraining. Measure old/new-task performance, forgetting, training time, and storage; seek performance close to complete retraining at lower cost.

The request adds: find alternate papers, including CFD and biology, that upgrade this OASIS → TSR → aL-SAR → CSReL pipeline for accuracy and cost.

## Evidence Base

- Lit review: [literature-review-pipeline-upgrades-2026-11-01.md](/results/literature-review/literature-review-pipeline-upgrades-2026-11-01.md)
- Current contracts: [combined pipeline](/design/combined-pipeline.md), [paper index](/Papers/README.md)
- Additional sources named in the review: GREATS, LESS, RoD, TreeLoRA, ASO-LoRA, SD-LoRA, CoDyRA, LISA, Govoeyi and Richter, van der Meer and Bendor

## Gap-Coverage Map

| Lit-review gap | Addressed by the current hybrid? | Strength | Notes |
|---|---|---|---|
| Score examples by the quantity they change, not by gradient energy | No | none | OASIS uses relative energy plus redundancy. GREATS and DWR are the missing comparison. |
| Let the replay share move with measured forgetting | No | none | TSR draws a fixed branch probability. RoD is the missing comparison. |
| Stop one task vector from cancelling opposite subskills | No | none | Stated as a TSR limitation. TreeLoRA is the layer-wise alternative. |
| Show that Fisher prefix freezing beats a dumb layer or rank budget | No | none | LISA and CoDyRA supply the controls. Neither has been run here. |
| Show that a buffer beats rehearsal-free LoRA on this instruction stream | No | none | SD-LoRA and ASO-LoRA win on other streams. The hybrid has no such arm. |
| Report semantic accuracy, format validity, bytes, and FLOPs together under 700M | Partially | weak | The contracts require the ledger. No scientific run exists. |
| A new selector family | No | none | The gap is not open. The selectors exist. |

## Novelty Audit

| Component | Closest prior work | Verdict |
|---|---|---|
| Current-batch gradient admission | OASIS itself; GradNorm and MaxLoss inside GREATS | Already done. Energy is the weak form. |
| Probe-aligned admission | GREATS (NeurIPS 2024), LESS (ICML 2024) | Already done for static fine-tuning. Unaddressed only as a bounded-stream substitution against OASIS. |
| Fixed task routing by one gradient signature | TSR | Already done. |
| Layer-wise gradient routing | TreeLoRA (ICML 2025) | Already done. Incremental if only the tree is reimplemented. |
| Soft isolation versus transfer | ASO-LoRA (ACL 2026), O-LoRA | Already done. The gain over hard orthogonality is about 1.7 points on short T5 streams and reverses on one long order. |
| Replay-share as an output of reducible loss | RoD (arXiv 2026); CSReL for the writer | The share is unaddressed inside this hybrid. The loss difference is not. |
| Fisher layer freezing and frequency retrieval | aL-SAR | Already done. |
| Random layer sampling at matched FLOPs | LISA (NeurIPS 2024) | The control is published and has not been run. Genuinely unaddressed as a comparison, not as a method. |
| Rank as a forgetting brake | CoDyRA | Already done on CLIP and TRACE. Incremental as a port. |
| Reducible-loss memory | CSReL | Already done. |
| Rehearsal-free LoRA | SD-LoRA (ICLR 2025) | Already done on vision CIL. Unaddressed on this instruction contract. |
| Adjoint-weighted sampling | DWR, as implemented for PINNs by Govoeyi and Richter | The identity is old. The instruction-stream use is unaddressed and may be formally invalid. See the domain report. |
| Prediction-error replay priority | van der Meer and Bendor as a hypothesis; MIR as the ML method | The ML version is older than three years. Not a new sampler. |

Chain of verification, checked against the review's table readings:

- GREATS Table 2 is 54.2 versus GradNorm 50.4 on the nine-subject MMLU slice, not a full MMLU score. Holds.
- LESS does not beat full data on Llama-2-7B MMLU or BBH. The abstract's "often" does not cover those cells. Holds.
- TreeLoRA loses to HiDe-LoRA on ImageNet-R, 71.94 versus 72.28. Holds.
- SD-LoRA's Table 2 gap versus InfLoRA on 20-task ImageNet-R is 5.37 points, not the prose's 7.68. Holds.
- RoD is one seed and uses 4B–35B continued pretraining, not a 700M instruction buffer. Holds.

No verdict above flips under those checks. The "already done" labels stay.

## Impact Audit

- Who benefits: a lab that must update a small instruct model when a new task arrives and cannot keep the historical corpus.
- At what scale: one model under 700M, one stream, one GPU. Not a pretraining pipeline.
- Counterfactual: without this project the same lab would tune a replay percentage, keep a random buffer, and fine-tune. GREATS, RoD, and SD-LoRA would still exist. None of them is wired to this access contract.
- Improvement threshold: not crossed by any number in the current repository. There is no trained hybrid result. A port of TreeLoRA's 0.08-point CIFAR margin, or ASO-LoRA's 1.7-point classification margin, would not change a deployment decision.
- Compute fairness: GREATS's ghost score is close to ordinary training (71.3 versus 76.2 examples per second) and its direct score is not (4.2). LESS pays a warmup and a datastore before any gain. RoD pays a specialist unless that cost is removed. An upgrade that adds those costs without a matched-FLOP table is a compute allocation, not a result.

## Assumption Audit

| Assumption | Category | Consequence if false | Evidence required |
|---|---|---|---|
| Task boundaries are known and the current task stays readable until CSReL | Data | OASIS, TSR probes, and task-end coresets are different algorithms | A stream with no labels for the boundary |
| A held-out current-task probe represents the quantity we care about | Evaluation | Goal-weighted admission optimizes the wrong functional | A probe-swap ablation, including a noise probe |
| Last-layer gradient energy ranks examples the way a retention probe would | Model | OASIS admits the wrong rows | Paired scores on one batch |
| One probe vector is a stable task signature | Model | TSR routes to a cancelled average | Per-layer cosines with opposite signs |
| A fixed replay probability is near the useful rate | Optimization | The hybrid spends the budget on the wrong source | A sweep at 0, 0.25, 0.5, and 1 |
| Fisher per FLOP beats random layer sampling | Model | aL-SAR's criterion is bookkeeping | Matched-FLOP LISA-style arm |
| Stored gold targets plus an era teacher are needed | Deployment | SD-LoRA-style updates match the hybrid and the buffer is wasted | Rehearsal-free arm on the same stream |
| 7B paper numbers transfer under 700M | Scale | The ranking reverses | Repeat the winning arm at the project model |
| Instruction loss and exact format move together | Evaluation | A loss win is a format loss | Separate the two metrics, as LESS warns |

## Evaluation Weakness Audit

- Missing baselines: random and gradient-norm admission; a fixed-ratio sweep; random layer sampling; a fixed small LoRA rank; a rehearsal-free LoRA. GREATS, LISA, CoDyRA, and SD-LoRA are the citations.
- Benchmark issues: TRACE is the closest public stream, and the published runs sit at 1B–7B. Vision CIL is the wrong saturation story to quote as an instruction result.
- Ablation gaps: the hybrid has never been compared with any of its four papers removed, because the scientific run has not happened. Stacking four new papers would make that worse.
- Statistical issues: RoD is one seed. TreeLoRA's vision margins sit inside a tenth of a point. ASO-LoRA copies older baseline columns.
- OOD / stress absence: no format shift, no label noise, no empty-history task, no probe that does not match the test.

## Venue Acceptance Check

ICLR and NeurIPS accept a new training rule when the ablation isolates it and the gain is visible against the obvious cheaper rule. They reject a four-module system whose only evidence is that each module was published. ACL accepts instruction continual-learning papers, and it also accepts ASO-LoRA-scale gains when the protocol is the community protocol. A 1-point gain on a private stream, with no TRACE number and no released selection cost, is the usual reject. The idea as "a better pipeline" falls short of all three. The idea as a pre-registered substitution study can clear the empirical bar if the margins and the negative arms are reported, including the arms that lose.

## Brutal Verdict

1. The hybrid is a composition of four published methods. Shipping it, or swapping in four newer published methods, is an engineering report.
2. The headline papers do not support a large gain. TreeLoRA's vision edge is tenths of a point and reverses on ImageNet-R. ASO-LoRA's edge over O-LoRA reverses on one order. LESS loses to full data on two of three Llama-2-7B tasks.
3. The strongest new papers use the wrong access model. RoD wants the pretraining corpus and a second converged model. GREATS and LESS want a static target probe and 7B models. SD-LoRA wants class-incremental images and no buffer.
4. Gradient energy, a fixed replay rate, and Fisher freezing are all unjustified until they beat alignment, a ratio sweep, and random layer sampling. Those controls are simpler than the hybrid.
5. A CFD or biology citation does not raise accuracy. The DWR identity and the replay-tag hypothesis only change which ablation is mandatory.

## Contribution Ladder

Current rung: Weak as a novelty claim. Viable only as a reproduction with a ledger, and that reproduction has not been run.

What moves it up one rung: one substitution, isolated, at matched bytes and FLOPs, beating OASIS-or-uniform and losing to a named control if the mechanism is removed. That is Viable.

What would make it Strong: the same run also includes the rehearsal-free arm and the noise-probe arm, and the paper reports the frontier rather than a single winning ratio. That is a characterization of when goal-weighted replay spends its budget, which is the Strong version.

High-value is not available. The field already has the selectors. A new problem class would require a benchmark this review did not find a hole large enough to invent.

## Cross-Domain Pollination

TRIZ negations:

- The field treats a large gradient as importance. Negate it: only the part aligned with a named functional counts.
- The field picks a replay rate. Negate it: the rate is the fraction of slots old data can win.
- The field uses one task similarity. Negate it: similarity can change sign across layers.
- The field stores the example it retrieves. Negate it: retrieval can be a tag, and the write can happen later under a different test.
- The field assumes a buffer. Negate it: isolation without stored examples is the baseline that has to lose.

Conventional path. Keep the hybrid, replace OASIS with GREATS, and add the baselines GREATS already uses. Exploratory. Low risk. The likely outcome is a small accuracy change bought with a probe the project must then defend.

Blend A, CFD. Dual-weighted residuals score a residual by the adjoint of a quantity of interest. Map the residual to the candidate gradient and the adjoint to a retention probe. The mechanism is GREATS with the probe defined as old-task loss, not current loss. Boden: transformational if the paper's object of selection changes from "informative example" to "example that moves a named retention functional." The transformation is in the objective, not in a new network.

Blend B, budget allocation. RoD is already an auction between new and old examples. Organizational learning adds nothing once the auction is stated. Boden: combinational. Use it as the replay-share arm, not as a second theory.

Blend C, admission control. OASIS is an admission gate and CSReL is a later write. Queuing theory would drop arrivals that fail the gate. The current contract does the opposite and keeps rejects eligible for the coreset. A tag-then-capture rule, motivated by synaptic tagging, would let CSReL keep a reject only when a later probe says its loss is still reducible. Boden: combinational. It is a real ablation and a weak theory, because tagging in hippocampus is not a theorem about tokens.

Selected path: Blend A, with Blend B as the replay-share measurement and the LISA and SD-LoRA arms as destructive controls. Justification: every strong paper in the review is a version of "weight the update by the functional you will be judged on." Stacking routers does not add a second idea. Transformational relative to the current energy score; not transformational relative to GREATS.

## Refined High-Value Pivot

- Anchor gap: admission and replay share are optimized by local scores that are not the retention functional.
- Selected path: Blend A.
- Null hypotheses, to be tested before any expected gain is quoted:
  - H0-1: probe-aligned admission does not change retained semantic accuracy relative to OASIS at the same number of admitted examples.
  - H0-2: a demand rule does not move the adaptation-forgetting frontier relative to the best of the fixed probabilities {0, 0.25, 0.5, 1}.
  - H0-3: per-layer signatures do not beat one task signature.
  - H0-4: Fisher prefix freezing does not beat random layer sampling at the same expected unfrozen FLOPs.
  - H0-5: a rehearsal-free LoRA does not match the hybrid on the instruction stream.
- Falsifiable claim, as a pre-registered target rather than a measured result: on a fixed instruction stream and a model at or below 700M, probe-aligned admission plus demand-routed replay changes retained semantic accuracy by at least 3 points relative to the current hybrid and relative to uniform replay, at a selection-plus-training FLOP increase of at most 20% and at the same replay-byte cap. A gain below 1 point, or a gain that remains when the probe is replaced by noise, rejects the claim.
- Explain-it test: Keep the examples that move old-task loss, and spend replay only when old loss is actually rising. Do not add a new module for every paper in the review.
- What to keep: task chunks, the byte and FLOP ledger, CSReL as the writer until a writer ablation says otherwise, era teachers only on the replay arm that uses distillation.
- What to drop: SIREN-style redundancy as a required stage until alignment is beaten; TSR's fixed 0.5 as an untested default; aL-SAR's Fisher rule until it beats random sampling; any plan that imports RoD's specialist or SD-LoRA's vision head unchanged; biology as a performance claim.
- Experimental plan: one stream, one model under 700M, seeds at least three. Arms are new-only, uniform replay, fixed-ratio sweep, current hybrid, probe-aligned admission with the hybrid's other stages unchanged, demand-routed share with the writer unchanged, random layer sampling, rehearsal-free LoRA, and a noise probe. Metrics are retained semantic accuracy, new-task semantic accuracy, format validity, retained bytes, auxiliary bytes, and selection-plus-training FLOPs. Oracle: train on all current data plus the full historical training sets from a fixed initialization, and log that this arm is allowed to violate the byte cap. The gap from the best bounded arm to that oracle is the headroom.
- Required controls for the obvious confounds: same token budget; same byte cap; probe ids disjoint from test; noise probe; random layer arm; ratio sweep; rehearsal-free arm; three seeds.
- Most informative negative result: probe alignment ties OASIS, and the noise probe ties both. That would mean the last-layer instruction gradient is too noisy for a first-order functional weight, and the review's main transfer from GREATS and DWR would be false in this setting.
- Expected venue fit: a negative or mixed characterization fits a workshop now and a conference only if the ledger and the arms above are complete. It does not fit as a methods paper that claims a new selector.

## Multi-Lens Validation

- Theorist: the only mechanism with a real identity behind it is the first-order functional error, residual times adjoint. GREATS is that expansion for a validation loss. DWR is that expansion for a PDE functional. OASIS energy is the residual norm, which the identity does not justify. The demand rule is a heuristic auction unless the two scores are on a common scale. RoD's source-specific references are an attempt at that scale. A cached commit loss is a further approximation and has to be labeled as one.
- Practitioner: a 0.5B instruct model, one TRACE-sized stream, and the arms above are a few controlled runs, not a new training stack. The hard part is the probe gradient at admission time without doubling the backward cost. GREATS's ghost inner product is the implementation to copy. The specialist model in RoD should not be implemented.
- Skeptic: the damaging question is whether any gain is just "we trained on a luckier 25%." The noise-probe arm is the answer. The result that kills the pivot is a noise probe that matches the real probe.
- Janusian: yes. The useful paper plots retained accuracy against replay bytes and against probe quality. It does not announce that goal weighting dominates energy. Accuracy versus replay, and transfer versus isolation, are the tensions. SD-LoRA and RoD sit on opposite sides of the second one.
- Revisions made: the claim is a pre-registered 3-point target with an explicit reject region, not a promised gain. Layer-wise routing and Fisher freezing are controls, not part of the claimed method. CSReL stays until a writer ablation removes it. Biology and CFD do not appear in the claim sentence.

## Fact checks that stayed unresolved

- CoDyRA's "LLaMA-3-1B" row repeats TreeLoRA's 36.14, which TreeLoRA labels LLaMA-3.2-1B. The checkpoints may differ.
- SD-LoRA's arXiv id 2501.13198 comes from the OpenAlex record whose title matches. The ICLR PDF was the text actually read.
- LISA and CoDyRA code URLs were not verified.
- OpenAlex citation counts are too incomplete to use.
