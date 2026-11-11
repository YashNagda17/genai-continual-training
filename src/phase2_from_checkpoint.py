"""Report imported Phase 1 training and prepare an explicit exploratory Phase 2."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from .datasets.dataset_instruction import SummarizationDataset
from .models.model_mt5 import MT5SmallModel
from .models.model_seq2seq import Seq2SeqSettings
from .phases.phase2 import run as run_phase2
from .phases.settings import PhaseSettings
from .phases.support import gate_decision, lead_baseline, script_share_of_forgetting
from .utils.contracts import fingerprint
from .utils.numerics import chrf
from .utils.state import Checkpoints, read_json, write_json


def rebase_paths(value, original, source):
    """Rebase source-owned absolute paths without modifying the imported records."""
    if isinstance(value, dict):
        return {key: rebase_paths(child, original, source) for key, child in value.items()}
    if isinstance(value, list):
        return [rebase_paths(child, original, source) for child in value]
    if isinstance(value, str) and (value == original or value.startswith(original + "/")):
        return str(source.joinpath(*value[len(original):].strip("/").split("/")).resolve())
    return value


def source_settings(english_run, hindi_run, manifest):
    """Recover recorded settings and require the archived phase identity to match."""
    resolved = english_run["resolved"]
    model = resolved["model_settings"]
    method = resolved["config"]["method"]
    spec = english_run["spec"]
    config = resolved["model_config"]
    return PhaseSettings(protocol="scientific", seed=spec["seed"], batch_size=spec["batch_size"],
        english_passes=spec["passes"], hindi_passes=hindi_run["spec"]["passes"],
        task_prefix=manifest["preprocessing"].get("prefix") or "",
        analysis_batch=method["analysis_batch"], max_new_tokens=method["generation"]["max_new_tokens"],
        encoder_blocks=config["num_layers"], decoder_blocks=config["num_decoder_layers"],
        stability_threshold=0.5, sample_size=64, learning_rate=model["learning_rate"],
        warmup_fraction=model["warmup_steps"] / spec["max_steps"],
        no_repeat_ngram=method["generation"].get("no_repeat_ngram_size", 0))


def inspect_source(source):
    """Verify source payloads and saved chrF on CPU; allocate no model weights."""
    source = Path(source).resolve()
    dataset = SummarizationDataset()
    dataset.load({"path": str(source / "prepared")})
    english = read_json(source / "phase1/english/arm.json")
    hindi = read_json(source / "phase1/hindi_only/arm.json")
    english_run = read_json(source / "phase1/english/run.json")
    hindi_run = read_json(source / "phase1/hindi_only/seed-1/run.json")
    settings = source_settings(english_run, hindi_run, dataset.manifest)
    expected = fingerprint({"version": 2, "phase": 1, "settings": asdict(settings),
                            "dataset": dataset.fingerprint})
    progress = read_json(source / "phase1/progress.json")
    if progress["identity"] != expected:
        raise ValueError("Archived Phase 1 identity does not match recovered settings and prepared data")
    old_root = english["attempt"].replace("\\", "/").rsplit("/phase1/", 1)[0]
    english = rebase_paths(english, old_root, source)
    hindi = rebase_paths(hindi, old_root, source)
    runs = [english, *hindi.get("seed_runs", ())]
    if len(runs) != 4 or {run["seed"] for run in runs[1:]} != set(settings.hindi_seeds):
        raise ValueError("Expected a completed English arm and all three Hindi seeds")
    hashes = {}

    def remember(path):
        path = Path(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        hashes[path.relative_to(source).as_posix()] = digest
        return digest

    def artifact(folder):
        folder = Path(folder)
        marker = folder / "complete.json"
        complete = read_json(marker)
        for name, digest in complete["files"].items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder.resolve()) or remember(path) != digest:
                raise ValueError(f"Artifact verification failed: {path}")
        remember(marker)

    for path in (source / "prepared/manifest.json", source / "prepared/records.jsonl",
                 source / "phase1/progress.json", source / "phase1/english/run.json",
                 source / "phase1/english/arm.json", source / "phase1/hindi_only/arm.json"):
        remember(path)
    for run in runs:
        if run["status"] != "completed":
            raise ValueError(f"Incomplete source arm: {run['name']}")
        folder = Path(run["attempt"])
        remember(folder / "arm.json")
        remember(folder / "run.json")
        checkpoint = Path(run["checkpoint"])
        if not checkpoint.is_relative_to(source):
            raise ValueError("Source checkpoint is outside the imported run")
        # Final exports have complete markers; pinning does too for the English adapter.
        artifact(Path(run["final_model"]))
        metadata = read_json(checkpoint / "metadata.json")
        if metadata["model"] != MT5SmallModel.model_id or metadata["rank"] != 16 or metadata["alpha"] != 32:
            raise ValueError("Source adapter does not match mT5-small rank-16 configuration")
        remember(checkpoint / "metadata.json")
        remember(checkpoint / "weights.pt")
        if hashes[(checkpoint / "weights.pt").relative_to(source).as_posix()] != hashes[
                (Path(run["final_model"]) / "weights.pt").relative_to(source).as_posix()]:
            raise ValueError("Pinned checkpoint and final export differ")
        for task, scores in run["metrics"].items():
            language, split, _ = task.split("/")
            expected_rows = dataset.views[f"{language}/{split}"]
            path = folder / "predictions" / (task.replace("/", "-") + ".json")
            rows = read_json(path)
            expected_refs = {row.sample_id: row.target["text"] for row in expected_rows}
            if len(rows) != len(expected_refs) or {row["id"]: row["target"] for row in rows} != expected_refs:
                raise ValueError(f"Prediction IDs/references do not match prepared data: {path}")
            actual = statistics.mean(chrf(row["prediction"], row["target"]) for row in rows)
            if not math.isclose(actual, scores["chrf"], abs_tol=1e-5):
                raise ValueError(f"Saved chrF does not match predictions: {path}")
            remember(path)
    for task, scores in hindi["metrics"].items():
        for metric, value in scores.items():
            mean = statistics.mean(run["metrics"][task][metric] for run in runs[1:])
            if not math.isclose(value, mean, abs_tol=1e-5):
                raise ValueError("Hindi aggregate differs from the recorded seed arms")
    pool_folder = source / "phase1/english/pinned/pool"
    artifact(pool_folder)
    pool = read_json(pool_folder / "manifest.json")
    train_ids = {row.sample_id for row in dataset.training_view("english")}
    pool_ids = [row["id"] for row in pool["records"]]
    if len(pool_ids) != 500 or len(set(pool_ids)) != 500 or not set(pool_ids) <= train_ids:
        raise ValueError("Invalid English training replay pool")
    from .pipelines.revision import RevisionConfig, RevisionPipeline, load_pool
    loaded_pool = load_pool(pool_folder)
    past = RevisionPipeline(RevisionConfig("hindi", replay_text="past"), pool=loaded_pool)
    eligible = past.filter_past(loaded_pool)
    lead = {language: lead_baseline(dataset, language) for language in ("english", "hindi")}
    before, after = english["metrics"], hindi["metrics"]
    forget = before["english/validation/plain"]["chrf"] - after["english/validation/plain"]["chrf"]
    gain = after["hindi/validation/plain"]["chrf"] - before["hindi/validation/plain"]["chrf"]
    gate = gate_decision(forget, gain, before["english/validation/plain"]["chrf"], lead["english"]["chrf"])
    spread = hindi["metric_std"]["english/validation/plain"]["chrf"]
    details = {"gate": gate, "forget": forget, "gain": gain, "lead_1": lead,
        "english_validation_chrf": before["english/validation/plain"]["chrf"],
        "x_seed_spread": spread, "confirm_revision": spread > settings.max_seed_spread,
        "script_share_validation": script_share_of_forgetting(before["english/validation/plain"]["chrf"],
            after["english/validation/plain"]["chrf"], after["english/validation/guarded"]["chrf"]),
        "decision_split": "validation", "serving": {"status": "incomplete", "timings_usable": False},
        "source_phase_seconds": progress.get("phase_seconds"),
        "past_replay_eligible": len(eligible), "past_replay_filtered": 500 - len(eligible)}
    context = {"english_checkpoint": english["checkpoint"], "pool": str(pool_folder),
               "english_metrics": before, "hindi_only": hindi, "phase1": details}
    return dataset, settings, english_run["resolved"]["model_settings"], english, context, hashes


def write_training_report(source, settings, english, context):
    """Export report-ready scores without declaring the interrupted phase completed."""
    folder = source / "phase1"
    hindi, details = context["hindi_only"], context["phase1"]
    rows = []
    for split in ("validation", "test"):
        for language in ("english", "hindi"):
            task = f"{language}/{split}/plain"
            for metric in ("chrf", "rouge_l", "right_language"):
                before = english["metrics"][task][metric]
                after = hindi["metrics"][task][metric]
                rows.append({"split": split, "language": language, "metric": metric,
                             "before_hindi": before, "after_hindi_mean": after,
                             "after_hindi_std": hindi["metric_std"][task][metric], "change": after - before})
    payload = {"status": "training_evaluation_complete_serving_incomplete", "scientific_gate": details["gate"],
               "settings": asdict(settings), "details": details, "scores": rows,
               "timing": {"english": english["time"], "hindi_three_seeds": hindi["time"],
                          "english_update_seconds": english["update_seconds"],
                          "hindi_update_seconds_three_seeds": hindi["update_seconds"]}}
    write_json(folder / "training-report.json", payload)
    with (folder / "training-report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Phase 1 training and evaluation results", "",
        "Training and saved evaluations completed; serving benchmarks were interrupted. "
        "This report does not declare the whole Phase 1 experiment completed.", "",
        f"**Scientific gate: `{details['gate']}`.** English validation chrF is "
        f"{details['english_validation_chrf']:.4f}; lead-1 (copy the first sentence) scores "
        f"{details['lead_1']['english']['chrf']:.4f}. Exploratory Phase 2 was explicitly authorized; "
        "the baseline gate remains failed.", "",
        f"mT5-small, FP32, LoRA rank 16 / alpha 32. **{settings.english_passes} English epochs and "
        f"{settings.hindi_passes} Hindi epochs per seed**, despite the `passes5` folder name. "
        "2,000 train / 200 validation / 500 test articles per language; 500 English pool rows. "
        "English seed 0; Hindi seeds 1, 2, 3. Prompt `summarize: `; input/output limits 256/64; "
        f"learning rate {settings.learning_rate}; warm-up {settings.warmup_fraction:.0%}; no repeated trigram.", "",
        "## Scores", "", "After-Hindi values are means across the three independent Hindi seeds. "
        "The English-trained checkpoint is shared. Higher is better; these are not accuracy percentages.", "",
        "| Split | Language | Metric | Before Hindi | After Hindi mean ± SD | Change |",
        "|---|---|---|---:|---:|---:|"]
    lines += [f"| {r['split']} | {r['language']} | {r['metric']} | {r['before_hindi']:.4f} | "
              f"{r['after_hindi_mean']:.4f} ± {r['after_hindi_std']:.4f} | {r['change']:+.4f} |" for r in rows]
    lines += ["", "## Interpretation", "",
        f"English validation forgetting is **{details['forget']:.4f} chrF points**; Hindi validation gain is "
        f"**{details['gain']:.4f} points**. This is a forgetting signal, but the English baseline is below "
        "lead-1, so it does not satisfy the scientific starting-quality requirement. Validation decides "
        "the gate; test results are report-only. ROUGE-L is the project's unofficial whitespace metric; "
        "right-language share is a Unicode-script proxy, not a semantic language detector.", "",
        "## Recorded costs", "",
        f"- English: {english['steps']} updates; {english['update_seconds'] / 60:.2f} minutes of updates; "
        f"{english['wall_clock_seconds'] / 60:.2f} minutes for the complete arm.",
        f"- Hindi: {hindi['steps']} updates across three seeds; {hindi['update_seconds'] / 60:.2f} minutes "
        f"of updates; {hindi['wall_clock_seconds'] / 60:.2f} minutes summed across arms.",
        "- These are historical XPU-host measurements, not predictions for this PC. Recorded GPU hours "
        "include arm wall time, not GPU kernel busy time. Serving results are incomplete.", "",
        "## Reuse and limitations", "",
        "The imported adapter files and pool passed integrity checks. Prediction IDs/references and "
        "chrF were checked against the prepared subsets; ROUGE-L and script scores come from the saved "
        "arm reports. Base revision was recorded as `main`, so the source does not pin an immutable "
        "base-model commit. Exploratory Phase 2 reuses the saved English checkpoint, the same subsets "
        "and pool, and Hindi seed 1 for all methods. It does not repeat English training or Phase 1 serving.",
        "", f"Past-summary filtering leaves {details['past_replay_eligible']} of 500 pool examples eligible.", ""]
    (folder / "training-report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/code/summarization-passes5")
    parser.add_argument("--output", default="results/code/phase2-exploratory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exploratory", action="store_true", help="Keep the failed gate; explicitly allow a pilot comparison")
    parser.add_argument("--run", action="store_true", help="Start Phase 2; otherwise verify and prepare reports only")
    args = parser.parse_args(argv)
    if not args.exploratory:
        parser.error("This entry point requires --exploratory; it does not bypass the scientific gate silently")
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Use a separate output root outside the imported run")
    dataset, original_settings, model_settings, english, context, hashes = inspect_source(source)
    settings = replace(original_settings, protocol="pilot", seed=original_settings.hindi_seeds[0])
    provenance = {"exploratory": True, "source_gate": context["phase1"]["gate"],
        "source_training_evaluation_complete": True, "source_serving_complete": False,
        "hindi_seed": settings.seed, "source_identity": fingerprint(hashes)}
    context["phase1_handoff"] = provenance
    manifest = {"schema": 1, "source_root": Path(os.path.relpath(source, output)).as_posix(),
                "settings": asdict(settings), "provenance": provenance, "source_hashes": hashes}
    manifest = json.loads(json.dumps(manifest))
    output.mkdir(parents=True, exist_ok=True)
    with Checkpoints(output, "phase2-handoff").lock():
        path = output / "handoff.json"
        if path.exists() and read_json(path) != manifest:
            raise ValueError("Handoff inputs/settings changed; choose a new output root")
        write_training_report(source, original_settings, english, context)
        write_json(path, manifest)
        print(f"Phase 1 Markdown: {source / 'phase1/training-report.md'}", flush=True)
        print(f"Source gate remains {provenance['source_gate']}; exploratory pilot, Hindi seed {settings.seed}.", flush=True)
        print(f"Phase 2 ready: five arms, {settings.hindi_passes} epochs each; no English retraining.", flush=True)
        if not args.run:
            return output
        import torch
        device = torch.device(args.device)
        if (not torch.cuda.is_available() or device.type != "cuda"
                or (device.index is not None and device.index >= torch.cuda.device_count())):
            raise ValueError("This launch requires CUDA; no silent CPU fallback")
        def factory():
            config = replace(Seq2SeqSettings(**model_settings), seed=settings.seed, logit_chunk_size=32,
                             warmup_steps=0)
            return MT5SmallModel(config, device=args.device, revision="main")
        run_phase2(output, dataset, factory, settings, context)
        write_json(output / "progress.json", {"status": "completed", "completed_phases": [2],
                                              "exploratory": True, "source_gate": provenance["source_gate"]})
    print(f"Phase 2 results: {output / 'phase2/report.md'}", flush=True)
    return output


if __name__ == "__main__":
    main()
