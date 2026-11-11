"""Phase 7: drift ranking with past-summary replay together (G1 + G2), and the combinations the pipelines cannot run.

It runs right after Phase 2 and needs only Phase 1's English checkpoint and pool. ``room_placed_frozen``
needs Phase 4's room plans, so Phase 4 runs it."""

from .support import hindi_pipeline, run_hindi
from ..pipelines.revision import load_pool
from .artifacts import annotate, save_report
from .support import phase

UNSUPPORTED = (
    {"name": "reverse_order", "reason": "Hindi training starts from the English checkpoint"},
    {"name": "third_language", "reason": "This preparation uses English and Hindi only"},
    {"name": "past_without_reference", "reason": "The pool filter currently requires reference summaries"},
)


@phase(7)
def run(directory, dataset, model_factory, settings, context, clock):
    pool = load_pool(context["pool"])
    checkpoint = context["english_checkpoint"]
    together = run_hindi(directory / "drift_past", "drift_past",
                        hindi_pipeline(settings, pool, checkpoint, selector="drift", replay_text="past"),
                        dataset, model_factory, settings, selector="drift", replay_text="past")
    details = annotate(
        {"drift_past": together["attempt"], "skipped": list(UNSUPPORTED),
         "room_placed_frozen": "runs in Phase 4, after the room plans"},
        clock, [together], together.get("final_model"))
    return save_report(directory, 7, details, {"drift_past": together, "phase7": details})
