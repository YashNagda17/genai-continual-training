"""Phase 1: English checkpoint, Hindi-only update X, and the script-guard share G5."""

from ..pipelines.revision import RevisionPipeline
from ..pipelines.revision import load_pool
from .support import revision_config, forgetting, gain, gate_decision, phase, run_pipeline, score, script_share_of_forgetting
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
        model_factory, dataset, hindi_dir, "hindi", settings, settings.hindi_passes, "hindi_only", selector="none", replay_text="reference", repair_stop=False)
    before = english_arm["metrics"]
    after = hindi_arm["metrics"]
    forget = forgetting(before, hindi_arm)
    hindi_gain = gain(before, hindi_arm)
    share = script_share_of_forgetting(score(before, "english/test/plain"), score(after, "english/test/plain"),
                                       score(after, "english/test/guarded"))
    serving = benchmark(directory / "serving", dataset, model_factory, settings,
        {"base": None, "english": english_checkpoint, "hindi_only": hindi_arm["checkpoint"]},
        reference={"english": english_checkpoint, "hindi": hindi_arm["checkpoint"]})
    details = annotate(
        {"gate": gate_decision(forget, hindi_gain), "forget": forget, "gain": hindi_gain,
         "script_share_of_forgetting": share, "decision_split": "validation", "serving": serving},
        clock, [english_arm, hindi_arm], hindi_arm.get("final_model"))
    update = {"english_checkpoint": str(english_checkpoint), "pool": str(pool),
              "english_metrics": before, "hindi_only": hindi_arm, "phase1": details}
    return save_report(directory, 1, details, update)
