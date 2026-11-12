# Bounded English distillation comparison

User-authorized exploratory experiment, 2026-11-12. Compare reference replay with
the identical replay plus cached English-teacher probabilities. This is separate
from the phase matrix and uses the locally available trained mT5 checkpoint.

Default: four updates, two Hindi training rows and one English pool row per update,
four fixed validation examples per language, seed 1, learning rate 0.001, greedy
generation up to 64 tokens. These small-run settings are not the scientific protocol.
Both arms start from the same immutable English adapter, use the same row order,
optimizer settings and RNG seed. No test rows are read. CLI supports multiple seeds.

Cache top-32 probabilities and one residual-vocabulary probability per supervised
position on English reference-summary prefixes, temperature 1. Student KL uses the
same top-token identities and residual bucket. This is a coarse-grained KL, not
full-vocabulary KL. The loss is batch-mean supervised CE plus strength times the
batch-mean KL (zero KL for Hindi rows); default strength 1. Teacher runs once,
before students. Only one model is resident. Cache time and bytes are reported.

The isolated model subclass owns numerical distillation. Existing phase execution
is unchanged. Checkpoints retain optimizer/RNG and the completed update cursor via
the existing transactional Checkpoints implementation; source weights are read-only.

Report before/after validation chrF, ROUGE-L, language correctness and NLL, predictions,
training time, cache cost and memory. A promising seed requires English chrF above
replay alone with Hindi chrF no more than 0.5 below replay. Otherwise record a negative
or null result. A tiny single-seed run cannot establish an improvement; no tuning on
these results is presented as independent confirmation.
