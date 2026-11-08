# Tests

The active tests are [test_summarization.py](/src/tests/test_summarization.py) and [test_phases.py](/src/tests/test_phases.py). Legacy pipeline, causal-model and instruction-source tests were removed with their implementations.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest src.tests.test_summarization src.tests.test_phases -q
```

Keep behavior tests small and tied to the [current plan](/Idea_Refined_2.md). Do not test Markdown wording, abstract-class boilerplate or helpers that merely mirror implementation.

| Area | Evidence |
|---|---|
| Metrics | Hand-computable chrF/script share, bootstrap, interaction, overlap, kappa and energy rank |
| Data | Official XL-Sum splits/IDs, selected subsets, truncation/prefix, masked right-padding, source archive conversion |
| Model | Tiny local mT5 loss/masking, deterministic generation, script guards, probe isolation, adapter masks, merging and shrink equivalence |
| Revision | Loss/drift ranks, replay slots, past filtering, repair weights, retirement and immutable English artifacts |
| Placement | Per-rank counts, equal C2 sizes, zero-output new rank, slowdown, two-sample stability and strict 3% family matching |
| Shrink | Validation-only search, both search directions, rank table and saved selected adapter |
| Recovery | Interrupted policy work matches uninterrupted work, finished arms are reused, reports can be rebuilt without training |
| Phases | Gates/winners use validation, pre-registration rejects mismatches, seed attempts and summaries persist, each phase saves reports/status/final models |
| Serving | One pass without warm-up, separate merged/unmerged/reference logs and completed-result reuse |

Routine execution uses a tiny random Torch model on CPU, with no downloads. Report exactly which tests ran. Static compilation and mock/CPU checks do not establish accelerator behavior, native-paper reproduction, retention, quality or speedup.
