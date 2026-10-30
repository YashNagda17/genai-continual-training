# Raw and prepared data storage

Owns `src/datasets/data/`. `download_datasets.sh` calls the source fetchers in `src/datasets/dataset_sources.py`. Each fetcher writes canonical JSONL for [InstructionDataset](/src/datasets/dataset_instruction.py): one directory `src/datasets/data/<dataset_id>/<commit>/` with `records.jsonl` and `manifest.json`. Importing a fetcher does not download. A second fetch of the same commit returns the existing files when the checksum matches and refuses to overwrite different bytes.

| Dataset | Source | Tasks and labels |
|---|---|---|
| `clinc_oos` | `clinc/clinc_oos`, config `plus` | One task `clinc`. The intent name is `class` and the supervised response. Official `train` / `validation` / `test`, including out-of-scope. Truncation `drop`. |
| `schuster` | `edmdias/multilingual-task-oriented-dialog`, config `en` | Tasks are the intent prefix `alarm`, `reminder`, and `weather`. The full intent label is `class` and the supervised response. Slots are not supervised. Hub `eval` is stored as `validation`. License CC-BY-SA-4.0. Truncation `drop`. |
| `bitext_customer_support` | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | Task is the category. `class` is the intent. The supervised response is the assistant reply. The file has no official split: within each category, seed 0 assigns the first 10% of instruction groups to test and the next 10% to validation, with the percentage truncated toward zero. License CDLA-Sharing-1.0. Truncation `prompt_left`. |
| `trace` | `YeMoKoo/TRACE` file `TRACE-Benchmark.zip`, subset `LLM-CL-Benchmark_5000` | Eight tasks in benchmark order: C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten. Lima is not part of this stream. `eval.json` is `validation`. `class` is the answer for FOMC, C-STANCE, NumGLUE-cm, and NumGLUE-ds, and the leading `A`–`D` choice for ScienceQA. MeetingBank, Py150, and 20Minuten have no class. The supervised response is the full answer. Truncation `prompt_left`. |

- Namespace artifacts by dataset ID and immutable source revision. Record source URI/local origin, checksum, schema, and source usage constraints where supplied. Do not treat a mutable upstream dataset name as a revision.
- Keep downloaded raw data immutable. Derived split/preprocessing artifacts must have separate fingerprints, including split seed/grouping, tokenizer/template revision, normalization, and truncation rules.
- A configured external cache is allowed; its manifest must resolve the same content. Paths alone are not content identity. Never redownload or regenerate silently when a referenced revision is unavailable.
- Keep replay payload and replay manifests in run-owned artifacts under `results/code/runs/<run_id>/<attempt_id>/`. A pointer into a complete raw archive is not proof of bounded historical storage: materialize the retained payload or count the accessible backing store.
- An archival copy may support later audit or a cumulative-data baseline. The bounded-history learner receives a restricted view that cannot enumerate or reopen that archive. Report archive size separately from retained learner-accessible state.
- Checkpoints and selection logs must not create uncounted usable historical copies. Diagnostic logs normally store IDs, scores, and hashes; payload-bearing error logs are evaluation artifacts and must remain inaccessible to training.
- Reject corrupt, mismatched, or missing artifacts. Empty required training partitions fail preflight; declared empty replay/optional evaluation views have typed empty results. Path changes require matching content hashes.

