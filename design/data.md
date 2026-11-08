# Raw and prepared data storage

Owns `src/datasets/data/`. `download_datasets.sh` calls the sole active source, `XLSumSource`, in `src/datasets/dataset_sources.py`. It writes `src/datasets/data/xlsum/<commit>/records.jsonl` and `manifest.json` for [SummarizationDataset](/design/datasets.md). Imports do not download; a second fetch preserves an existing complete conversion. Downloaded raw assets are preserved during code cleanup.

The source table below retains historical dataset provenance. Only the XL-Sum converter remains executable.

| Dataset | Source | Tasks and labels |
|---|---|---|
| `clinc_oos` | `clinc/clinc_oos`, config `plus` | One task `clinc`. The intent name is `class` and the supervised response. Official `train` / `validation` / `test`, including out-of-scope. Truncation `drop`. This file is not a multi-task stream. |
| `clinc_oos_domains` | same Hub snapshot, config `plus` | Ten tasks in official `clinc/oos-eval` `data/domains.json` order: `banking`, `credit_cards`, `kitchen_and_dining`, `home`, `auto_and_commute`, `travel`, `utility`, `work`, `small_talk`, `meta`. Each domain lists its 15 intent names. The intent is `class` and the supervised response. `oos` rows are omitted and counted. Official splits remain. Truncation `drop`. |
| `schuster` | `edmdias/multilingual-task-oriented-dialog`, config `en` | Tasks are the intent prefix `alarm`, `reminder`, and `weather`. The full intent label is `class` and the supervised response. Slots are not supervised. Hub `eval` is stored as `validation`. License CC-BY-SA-4.0. Truncation `drop`. |
| `bitext_customer_support` | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | Task is the category. `class` is the intent. The supervised response is the assistant reply. The file has no official split: within each category, seed 0 assigns the first 10% of instruction groups to test and the next 10% to validation, with the percentage truncated toward zero. License CDLA-Sharing-1.0. Truncation `prompt_left`. |
| `trace` | `YeMoKoo/TRACE` file `TRACE-Benchmark.zip`, subset `LLM-CL-Benchmark_5000` | Eight tasks in benchmark order: C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten. Lima is not part of this stream. `eval.json` is `validation`. `class` is the answer for FOMC, C-STANCE, NumGLUE-cm, and NumGLUE-ds, and the leading `A`–`D` choice for ScienceQA. MeetingBank, Py150, and 20Minuten have no class. The supervised response is the full answer. Truncation `prompt_left`. |
| `xlsum` | `csebuetnlp/xlsum`, files `data/<language>_XLSum_v2.0.tar.bz2` | **Current plan.** One task per language: `english` and `hindi` now, `marathi` and `gujarati` only for Phase 7. The article is the input and the summary is the target. Official train / validation / test. License CC-BY-NC-SA-4.0. Written in the canonical JSONL for [SummarizationDataset](/src/datasets/dataset_instruction.py), not the instruction rows. Truncation is applied by `Seq2SeqPreprocessing` in that same file. |

**XL-Sum notes.** `XLSumSource` reads the per-language tar files directly. `./download_datasets.sh xlsum` fetches English and Hindi. Each saved row is `id` (`<language>/<article id>`), `task`, `language`, `split` (`val` stored as `validation`), `text`, and `summary`. CLINC, Schuster, Bitext and TRACE converter code is removed. Additional languages require an explicit protocol extension.

- Namespace artifacts by dataset ID and the revision string used to download them. Record source URI, schema, and source usage constraints where supplied.
- Keep downloaded raw data in place. Derived split and preprocessing settings, including the tokenizer, normalization, and truncation rules, are stored in the prepared manifest.
- A configured external cache is allowed; its manifest must resolve the same content. Paths alone are not content identity. Never redownload or regenerate silently when a referenced revision is unavailable.
- Keep replay payload and replay manifests in run-owned artifacts under `results/code/runs/<run_id>/<attempt_id>/`. A pointer into a complete raw archive is not proof of bounded historical storage: materialize the retained payload or count the accessible backing store.
- An archival copy may support later audit or a cumulative-data baseline. The bounded-history learner receives a restricted view that cannot enumerate or reopen that archive. Report archive size separately from retained learner-accessible state.
- Checkpoints and selection logs must not create uncounted usable historical copies. Diagnostic logs normally store IDs, scores, and hashes; payload-bearing error logs are evaluation artifacts and must remain inaccessible to training.
- Reject missing artifacts. Empty required training partitions fail preflight; declared empty replay or optional evaluation views have typed empty results.

