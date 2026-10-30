# Raw and prepared data storage

Owns `src/datasets/data/`. `download_clinc_oos.sh` writes the `plus` configuration of `clinc/clinc_oos` there as canonical JSONL: one task `clinc`, the intent name as `class` and as the supervised response, and the Hub splits `train` / `validation` / `test` unchanged, including out-of-scope.

- Namespace artifacts by dataset ID and immutable source revision. Record source URI/local origin, checksum, schema, and source usage constraints where supplied. Do not treat a mutable upstream dataset name as a revision.
- Keep downloaded raw data immutable. Derived split/preprocessing artifacts must have separate fingerprints, including split seed/grouping, tokenizer/template revision, normalization, and truncation rules.
- A configured external cache is allowed; its manifest must resolve the same content. Paths alone are not content identity. Never redownload or regenerate silently when a referenced revision is unavailable.
- Keep replay payload and replay manifests in run-owned artifacts under `results/code/runs/<run_id>/<attempt_id>/`. A pointer into a complete raw archive is not proof of bounded historical storage: materialize the retained payload or count the accessible backing store.
- An archival copy may support later audit or a cumulative-data baseline. The bounded-history learner receives a restricted view that cannot enumerate or reopen that archive. Report archive size separately from retained learner-accessible state.
- Checkpoints and selection logs must not create uncounted usable historical copies. Diagnostic logs normally store IDs, scores, and hashes; payload-bearing error logs are evaluation artifacts and must remain inaccessible to training.
- Reject corrupt, mismatched, or missing artifacts. Empty required training partitions fail preflight; declared empty replay/optional evaluation views have typed empty results. Path changes require matching content hashes.

