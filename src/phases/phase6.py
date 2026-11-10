"""Phase 6: a separate English checkpoint that tells the model which language to write.

G4 runs only when G5's guard recovered at least ``g4_min_share`` of the English drop on validation:
if English rarely slips into Hindi, a language tag has nothing to fix."""

from ..pipelines.revision import RevisionPipeline
from .support import revision_config, run_hindi
from ..pipelines.revision import load_pool
from .artifacts import annotate, publish_final, save_report
from .support import phase, run_pipeline
from .data import with_prefix

PREFIX = "summarize in {language}: "


@phase(6)
def run(directory, dataset, model_factory, settings, context, clock):
    share = context["phase1"].get("script_share_validation")
    if not settings.force_conditional and (share is None or share < settings.g4_min_share):
        winner = context["overall_winner"]
        final = publish_final(directory, winner["name"], winner["checkpoint"]) or winner.get("final_model")
        details = annotate({"status": "skipped", "prefix": PREFIX, "script_share_validation": share,
                            "reason": f"G5 recovered {share} of the English drop; G4 needs {settings.g4_min_share}"},
                           clock, [], final)
        return save_report(directory, 6, details, {"phase6": details})
    token_model = model_factory()
    prefixed = with_prefix(dataset, token_model, PREFIX)
    token_model.release()
    english_dir = directory / "english"
    english_arm = run_pipeline(
        RevisionPipeline(revision_config(settings, stage="english", prefix="tagged"), pool=prefixed.pool_view()),
        model_factory, prefixed, english_dir, "english", settings, settings.english_passes, "english_prefix", selector="none",
        replay_text="reference", repair_stop=False)
    checkpoint = english_arm["checkpoint"]
    hindi = run_hindi(
        directory / "hindi_only", "hindi_only_prefix",
        RevisionPipeline(revision_config(settings, stage="hindi", selector="none"),
                         pool=load_pool(english_dir / "pinned" / "pool"), english_checkpoint=checkpoint),
        prefixed, model_factory, settings, selector="none")
    details = annotate(
        {"status": "completed", "script_share_validation": share, "prefix": PREFIX, "stored_prefix": prefixed.manifest["preprocessing"]["prefix"],
         "english": english_arm["checkpoint"], "hindi": hindi["checkpoint"]},
        clock, [english_arm, hindi], hindi.get("final_model"))
    return save_report(directory, 6, details, {"prefix_english": english_arm, "prefix_hindi": hindi,
                                               "phase6": details})
