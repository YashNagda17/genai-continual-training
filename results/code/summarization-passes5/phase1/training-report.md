# Phase 1 training and evaluation results

Training and saved evaluations completed; serving benchmarks were interrupted. This report does not declare the whole Phase 1 experiment completed.

**Scientific gate: `fix_english`.** English validation chrF is 20.7666; lead-1 (copy the first sentence) scores 22.0174. Exploratory Phase 2 was explicitly authorized; the baseline gate remains failed.

mT5-small, FP32, LoRA rank 16 / alpha 32. **4 English epochs and 4 Hindi epochs per seed**, despite the `passes5` folder name. 2,000 train / 200 validation / 500 test articles per language; 500 English pool rows. English seed 0; Hindi seeds 1, 2, 3. Prompt `summarize: `; input/output limits 256/64; learning rate 0.001; warm-up 1%; no repeated trigram.

## Scores

After-Hindi values are means across the three independent Hindi seeds. The English-trained checkpoint is shared. Higher is better; these are not accuracy percentages.

| Split | Language | Metric | Before Hindi | After Hindi mean ± SD | Change |
|---|---|---|---:|---:|---:|
| validation | english | chrf | 20.7666 | 14.2607 ± 0.2247 | -6.5059 |
| validation | english | rouge_l | 14.8388 | 3.3194 ± 0.2533 | -11.5194 |
| validation | english | right_language | 100.0000 | 99.6667 ± 0.2887 | -0.3333 |
| validation | hindi | chrf | 13.5387 | 22.0398 ± 0.3702 | +8.5010 |
| validation | hindi | rouge_l | 12.0834 | 18.7211 ± 0.2866 | +6.6376 |
| validation | hindi | right_language | 100.0000 | 100.0000 ± 0.0000 | +0.0000 |
| test | english | chrf | 20.5259 | 15.0529 ± 0.2487 | -5.4730 |
| test | english | rouge_l | 14.8116 | 3.5964 ± 0.4962 | -11.2153 |
| test | english | right_language | 100.0000 | 99.8667 ± 0.1155 | -0.1333 |
| test | hindi | chrf | 14.0076 | 22.9928 ± 0.5678 | +8.9851 |
| test | hindi | rouge_l | 12.6323 | 19.3770 ± 0.4463 | +6.7447 |
| test | hindi | right_language | 98.0000 | 100.0000 ± 0.0000 | +2.0000 |

## Interpretation

English validation forgetting is **6.5059 chrF points**; Hindi validation gain is **8.5010 points**. This is a forgetting signal, but the English baseline is below lead-1, so it does not satisfy the scientific starting-quality requirement. Validation decides the gate; test results are report-only. ROUGE-L is the project's unofficial whitespace metric; right-language share is a Unicode-script proxy, not a semantic language detector.

## Recorded costs

- English: 500 updates; 10.58 minutes of updates; 31.39 minutes for the complete arm.
- Hindi: 1500 updates across three seeds; 32.79 minutes of updates; 82.68 minutes summed across arms.
- These are historical XPU-host measurements, not predictions for this PC. Recorded GPU hours include arm wall time, not GPU kernel busy time. Serving results are incomplete.

## Reuse and limitations

The imported adapter files and pool passed integrity checks. Prediction IDs/references and chrF were checked against the prepared subsets; ROUGE-L and script scores come from the saved arm reports. Base revision was recorded as `main`, so the source does not pin an immutable base-model commit. Exploratory Phase 2 reuses the saved English checkpoint, the same subsets and pool, and Hindi seed 1 for all methods. It does not repeat English training or Phase 1 serving.

Past-summary filtering leaves 400 of 500 pool examples eligible.
