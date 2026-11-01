# Domain Grounding Report: Goal-Weighted Replay for a Bounded Instruction Update

Date: 2026-11-01 | Domain: sequential supervised fine-tuning, with CFD adjoint weighting and synaptic tagging as checked analogies | Inputs: [Challenge-pipeline-upgrades-2026-11-01.md](/results/challenger/Challenge-pipeline-upgrades-2026-11-01.md), [literature-review-pipeline-upgrades-2026-11-01.md](/results/literature-review/literature-review-pipeline-upgrades-2026-11-01.md), [combined pipeline](/design/combined-pipeline.md)

The challenger artifact exists and was used. This report checks the pivot, not a fresh idea.

## Current Idea

Keep a small fine-tune under a replay-byte cap. Admit current examples and replay old examples according to how much they move a declared retention loss, and spend replay only when that loss is rising. Compare that rule with energy admission, a fixed replay rate, random layer sampling, and a rehearsal-free update.

Two-sentence form. The training slot should go to the example whose gradient agrees with the loss we will be graded on. A large gradient that points somewhere else is not evidence the example is useful.

## Domain Interpretation

- Primary domain: online and continual supervised learning under a finite memory.
- Secondary domains: goal-oriented finite-element adaptivity, and the neurobiology of replay and synaptic tagging.
- Why this framing: the pipeline's decisions are selection and freezing decisions. CFD supplies the error identity those decisions are missing. Biology supplies a competing story about tags and later consolidation. Neither secondary domain is the system being trained.

## Claim-to-Theory Map

| Claim | Domain | Theory / principle | Assumptions | Source read | Status |
|---|---|---|---|---|---|
| A first-order step on an example reduces a probe loss in proportion to the gradient inner product | Optimization | Taylor expansion of a smooth loss | The step is small, the loss is differentiable, and higher-order terms are small | GREATS, NeurIPS 2024; LESS states the Adam variant | Partially supported. True locally. False if the probe loss is non-smooth or the Adam normalizer dominates the direction. |
| Weighting by gradient energy optimizes the same thing as weighting by a named functional | CFD / optimization | Dual-weighted residual identity: the functional error is the residual tested against the adjoint, not the residual norm | Linear or linearized functional, a usable adjoint in a space that is not Galerkin-orthogonal to the residual, and a residual that means equation error | Govoeyi and Richter, arXiv:2604.01835, stating the Becker–Rannacher identity for a Poisson problem | Violated by OASIS if the claim is that energy targets retention. Energy targets residual size. Supported only as a warning. |
| The same identity applies unchanged to instruction cross-entropy | Transfer | The structural map is residual ↔ candidate gradient and adjoint ↔ probe gradient | The probe loss must be the functional, the map must be first-order, and the failure mode must have a counterpart | Comparison of GREATS with Govoeyi and Richter | Partially supported. The first-order map holds. The PDE assumptions do not. The analogy is valid as a first-order weighting argument and invalid as a finite-element guarantee. |
| Replay share should rise only when old loss rises | Online learning | A scarce budget should be spent on the constraint that is currently violated | Old and new scores share a scale, and the old score is zero when nothing was forgotten | RoD, arXiv:2609.40089 | Partially supported for a pretraining corpus with a reference model. Unsupported for a cached commit-time loss until that cache is shown to rank the same examples. |
| One average gradient is a task signature | Geometry of gradients | An average preserves direction only if the summands are not opposed | Subskills inside the task are aligned | TSR limitation already stated in [TSR-summary.md](/Papers/TSR-summary.md); TreeLoRA uses layer-wise directions | Partially supported. The cancellation failure is real. TreeLoRA does not prove the layer split fixes it on this stream. |
| Fisher information per FLOP is the right layer-plasticity state | Synaptic metaplasticity | A synapse needs an internal state on more than one timescale, and the update rule must read that state | The state must predict interference, not merely correlate with recent gradient energy | Zenke and Laborieux, arXiv:2405.16922 | Partially supported as a requirement for some state. Unsupported as a requirement for aL-SAR's particular Fisher ratio. LISA has no such state and is the empirical counter-hypothesis. |
| An admission tag should gate the later memory write | Synaptic tagging and capture | A weak tag decays unless plasticity-related products from a strong event are captured | Tags and products are separable, and the capture window is finite | Chong, Ang, and Sajikumar, Communications Biology, 2025, on hippocampal slices | Analogy is suggestive but not formally valid. The slice result does not imply a token rule. The current contract, which keeps OASIS rejects eligible for CSReL, is a design choice, not a biological consequence. |
| Awake retrieval should be prioritized by prediction error and should tag a later write | Systems consolidation | Replay probability rises with prediction error; awake replay marks memories for sleep consolidation | The cited rodent pattern holds, and a token buffer is the analogue of a neural pattern | van der Meer and Bendor, Trends in Neurosciences, 2025 | Partially supported as a hypothesis about rodents. Unsupported as an accuracy claim for this model. MIR is the existing ML counterpart and is older than the review window. |
| Rehearsal is required to control forgetting | Continual learning | No such general theorem was found | Class-incremental linear readouts can share a low-loss region without stored images | SD-LoRA, ICLR 2025 | Incorrect as a general claim. Unresolved for long instruction targets. |
| Composing four published selectors approaches joint retraining | Optimization | Joint retraining minimizes the empirical risk on the union | The composed scores would have to be the gradient of that risk | No source supports the composition | Unsupported. The cumulative-data run is an oracle measurement, not a consequence of the hybrid. |

## Assumption Audit

| Assumption | Why it matters | If false | Evidence needed |
|---|---|---|---|
| Probe loss is differentiable in the weights that admission is allowed to read | The inner product is otherwise the wrong object | Alignment scores noise | Check the probe gradient norm and its variance across probe resamples |
| The probe is drawn from the same conditional as the reported test, and is disjoint from it | Otherwise the adjoint is for the wrong functional | A gain that vanishes on a fresh test | Fixed split, published ids |
| Candidate and probe gradients are not so aligned that every example receives the same score | DWR and GREATS both fail when the weight carries no contrast | Selection collapses to random | Report the score range inside a batch |
| Commit-time loss remains comparable after the template and the mask change | RoD's reference loss is defined on a fixed tokenization | Demand routing follows a preprocessing artifact | Hash the template with the cached loss |
| Layer Fisher traces are not just recency | Metaplasticity requires the state to distinguish old interference from the current batch | aL-SAR freezes the layers the new task needs | Compare against a trace that is not reset and against random sampling |
| The capture window in a slice (up to 9 hours in the strong-before-weak mouse CA1 experiment, n = 7 slices at that interval) has a training-step counterpart | Otherwise tag-then-capture has no timescale to implement | The gate is an arbitrary delay | Do not import 9 hours. Test a delay of one task only. |

## Corrections to the Current Idea

- Claim: OASIS admission is justified because a larger gradient is more informative.
  - Why it is weak: the dual-weighted residual identity scores the residual by its effect on a functional. GREATS's Table 2 beats GradNorm by 3.8 points on its MMLU slice and essentially ties it on TydiQA. Energy is a baseline, not a justification.
  - Corrected view: report energy, alignment with a current probe, and alignment with a retention probe as three scores.
  - Source: Govoeyi and Richter; GREATS.

- Claim: a replay probability can be chosen once and left alone.
  - Why it is weak: RoD's frontier moves with the ratio, and the legal Nemotron comparison at matched adaptation is 0.071 versus 0.125 validation-loss forgetting against 10% replay.
  - Corrected view: the probability is a measured outcome. The fixed values 0, 0.25, 0.5, and 1 are the controls.
  - Source: RoD. The transfer to a cached commit loss is not yet supported.

- Claim: biology says rejected OASIS examples should be kept out of CSReL.
  - Why it is wrong as a deduction: Chong et al. show that a weak hippocampal pathway can still be stabilized when an earlier strong pathway supplies plasticity-related products, including at a 9-hour interval in their strong-before-weak slices. That is associativity across synapses, not a filter on dataset rows. van der Meer and Bendor argue that awake replay tags memories for later consolidation. They do not measure language-model buffers.
  - Corrected view: tag-then-capture is an optional ablation. The contract that keeps rejects eligible for CSReL stays until that ablation wins.
  - Source: Chong, Ang, and Sajikumar 2025; van der Meer and Bendor 2025.

- Claim: Fisher freezing is the plasticity rule metaplasticity recommends.
  - Why it is weak: Zenke and Laborieux require a multi-timescale internal state that the update consults. They do not derive a prefix of layers chosen by Fisher per FLOP. LISA is a zero-state rule that can beat LoRA on a single fine-tune, which does not prove it retains old tasks and does disprove any claim that a complicated state is free.
  - Corrected view: treat aL-SAR's trace as one state and random sampling as the no-state control.
  - Source: Zenke and Laborieux, arXiv:2405.16922; LISA.

- Claim: rehearsal-free results imply the buffer is wasted, or the reverse.
  - Why it is wrong: SD-LoRA's ImageNet-R 20-task table shows 75.26 against InfLoRA 69.89 without stored images. That is a vision readout. It does not speak to long targets. The hybrid's buffer is not theoretically required, and it is not theoretically useless.
  - Corrected view: one rehearsal-free arm, same stream, same FLOP cap.
  - Source: SD-LoRA Table 2, not the prose margins.

## Operating Regime and Failure Regime

Where the weighting argument should work:

- The probe is a small, fixed, held-out set from the same tasks the metric reports.
- Steps are small enough that a first-order inner product ranks the batch.
- Old and new losses are computed with the same mask and template.
- The model is small enough that a ghost inner product fits in the same backward budget as OASIS's last-layer score.

Where it should fail:

- The probe is a handful of prompts that do not match the test, or it overlaps the test.
- Template tokens dominate the gradient, so every example looks aligned.
- Task shift is large inside one update, and the Taylor step is a poor ranker. RoD's own single-seed frontier should not be expected to survive that unchanged.
- The retention functional depends on generation length or format, while the probe loss is token cross-entropy. LESS already reports that this pair can move apart.
- The adjoint network in the PINN paper is a second network for a linear elliptic equation. Copying that second network into the instruction run recreates RoD's specialist and breaks the cost target.

Boundary that matters most: whether a retention probe has any ranking power beyond a noise probe. If it does not, stop the goal-weighting story and keep the budget study of replay rate, which does not need an adjoint.

## Theory-Backed Reformulation

- Refined idea statement: Under a fixed byte cap, fill each training slot with the candidate that has the largest first-order predicted reduction in a predeclared probe loss. Use a current-task probe for new examples and the rise from a cached commit-time loss for stored examples. Do not add a second trained reference model.
- Mechanism: first-order functional weighting. The probe gradient is a stand-in for an adjoint. It is a stand-in, not a solution of an adjoint PDE.
- Why theory says it may work: if the probe loss is the quantity in the paper's claim, the inner product is the directional derivative of that quantity. Residual size is not.
- What must be true: shared tokenization, a disjoint probe, a step size inside the linear regime, and a score that varies inside the batch.

## Falsifiable Claims

1. At a matched admitted count, probe-aligned admission changes retained semantic accuracy relative to OASIS. Reject if the absolute gap is under 1 point or if a noise probe matches the real probe.
2. The fraction of replay slots produced by the demand rule lands on the frontier traced by the fixed probabilities {0, 0.25, 0.5, 1}, within the run-to-run spread across at least three seeds. Reject if a fixed probability dominates it on both adaptation and retention.
3. Random layer sampling at the same expected unfrozen FLOPs matches aL-SAR. If it does, the Fisher rule is not part of the claim.

## Experiment Implications

- Stream and model: one instruction stream, one model at or below 700M. TRACE is the public reference point, repeated at a size the cap allows. Do not cite 7B cells as results of that run.
- Baselines required by the theory: uniform and energy admission; the ratio sweep; full current data plus full history as an oracle that is allowed to break the byte cap; rehearsal-free LoRA; random layer sampling.
- Stress tests: noise probe; probe drawn from the wrong task; template or mask change between commit and replay; empty history; a task whose within-task layer gradients have mixed signs.
- Do not run: a second specialist model, a PINN, a hippocampal-slice protocol, or a 9-hour capture window.

## Analogy check

DWR to probe-aligned admission. The structural property is "an error component matters only through its effect on a declared output." The map is formal at first order and fails as a bound: instruction loss is not a Poisson energy, the network is not a conforming subspace, and Galerkin orthogonality is the wrong language. Use the identity to choose the ablation. Do not cite a Poisson error ratio as an expected accuracy gain.

Synaptic tagging to OASIS-then-CSReL. The structural property is "a mark and a later consolidation product are different events." The biology result that was actually read is the mouse CA1 slice experiment: weak stimulation decays when alone, and a strong-before-weak pairing still left both pathways potentiated at intervals up to 9 hours in that preparation. The assumptions do not hold for tokens. There is no protein product, no second pathway, and no measured decay time in steps. The analogy is suggestive and not formally valid. The only permitted use is an ablation that the domain does not predict will win.

Awake replay to SAR. The structural property is "reactivation priority tracks prediction error and writes a tag rather than an immediate decision." van der Meer and Bendor state this as a reading of rodent work and as a hypothesis. The ML counterpart already exists in loss-increase retrieval. SAR's frequency score is a different mechanism and should be ablated against loss increase, not against a neuroscience diagram.

## Return versus patch

The core premise that survives is the first-order weighting argument plus a replay-rate sweep. The premises that do not survive are "energy is the right score," "biology gates the coreset," and "the hybrid approaches retraining." Those are corrections, not a broken experiment. No return to the challenger is required. The challenger pivot already dropped the biological gate from the claim sentence.

## Remaining Unknowns

- Whether a ghost inner product for SGD ranks the same instruction examples as the Adam feature LESS uses.
- Whether a cached scalar commit loss, without RoD's specialist, preserves the demand ranking.
- Whether SD-LoRA's low-loss path exists for a shared generative head.
- The unresolved CoDyRA versus TreeLoRA model-name mismatch on the 1B TRACE row.
- OpenAlex citation counts, which were too incomplete to use.
