"""Phase 1: English checkpoint, Hindi-only update X, and the script-guard share G5.

X is a confirmation arm, so it runs every Hindi seed. Its seed spread decides whether the best revision is confirmed too."""

from ..pipelines.revision import RevisionPipeline
from ..pipelines.revision import load_pool
from .support import (revision_config, forgetting, gain, gate_decision, lead_baseline, phase, run_pipeline, score,
                      script_share_of_forgetting)
from .artifacts import annotate, save_report
from .serving import benchmark


@phase(1)
def run(directory, dataset, model_factory, settings, context, clock):
    english_dir = directory / "english"
    english_pipeline = RevisionPipeline(revision_config(settings, stage="english"), pool=dataset.pool_view())
    english_arm = run_pipeline(
        english_pipeline, model_factory, dataset, english_dir, "english", settings, settings.english_passes,
        "english")
    english_checkpoint = english_arm["checkpoint"]
    pool = english_dir / "pinned" / "pool"
    hindi_dir = directory / "hindi_only"
    hindi_arm = run_pipeline(
        RevisionPipeline(revision_config(settings, stage="hindi", selector="none"),
                         pool=load_pool(pool), english_checkpoint=english_checkpoint),
        model_factory, dataset, hindi_dir, "hindi", settings, settings.hindi_passes, "hindi_only",
        seeds=settings.hindi_seeds, selector="none", replay_text="reference", repair_stop=False)
    before = english_arm["metrics"]
    after = hindi_arm["metrics"]
    forget = forgetting(before, hindi_arm)
    hindi_gain = gain(before, hindi_arm)
    share = script_share_of_forgetting(score(before, "english/test/plain"), score(after, "english/test/plain"),
                                       score(after, "english/test/guarded"))
    # G4 is decided on validation, so the test split never chooses an experiment.
    share_validation = script_share_of_forgetting(
        score(before, "english/validation/plain"), score(after, "english/validation/plain"),
        score(after, "english/validation/guarded"))
    lead = {language: lead_baseline(dataset, language) for language in ("english", "hindi")}
    english_score = score(before, "english/validation/plain")
    spread = (hindi_arm.get("metric_std") or {}).get("english/validation/plain", {}).get("chrf")
    confirm_revision = spread is not None and spread > settings.max_seed_spread
    serving = benchmark(directory / "serving", dataset, model_factory, settings,
        {"base": None, "english": english_checkpoint, "hindi_only": hindi_arm["checkpoint"]},
        reference={"english": english_checkpoint, "hindi": hindi_arm["checkpoint"]})
    details = annotate(
        {"gate": gate_decision(forget, hindi_gain, english_score, lead["english"]["chrf"]),
         "forget": forget, "gain": hindi_gain, "lead_1": lead, "english_validation_chrf": english_score,
         "x_seed_spread": spread, "confirm_revision": confirm_revision,
         "script_share_of_forgetting": share, "script_share_validation": share_validation,
         "decision_split": "validation", "serving": serving},
        clock, [english_arm, hindi_arm], hindi_arm.get("final_model"))
    update = {"english_checkpoint": str(english_checkpoint), "pool": str(pool),
              "english_metrics": before, "hindi_only": hindi_arm, "phase1": details}
    return save_report(directory, 1, details, update)
