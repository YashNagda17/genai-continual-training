"""Phase 6: a separate English checkpoint that tells the model which language to write."""

from ..pipelines.revision import RevisionPipeline
from .support import revision_config, run_hindi
from ..pipelines.revision import load_pool
from .artifacts import annotate, save_report
from .support import phase, run_pipeline
from .data import with_prefix

PREFIX = "summarize in {language}: "


@phase(6)
def run(directory, dataset, model_factory, settings, context, clock):
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
        {"prefix": PREFIX, "stored_prefix": prefixed.manifest["preprocessing"]["prefix"],
         "english": english_arm["checkpoint"], "hindi": hindi["checkpoint"]},
        clock, [english_arm, hindi], hindi.get("final_model"))
    return save_report(directory, 6, details, {"prefix_english": english_arm, "prefix_hindi": hindi,
                                               "phase6": details})
