"""Phase 7: the combinations the current pipelines can run, and the ones they cannot."""

from .support import hindi_pipeline, run_hindi
from ..pipelines.revision import load_pool
from .artifacts import annotate, save_report
from .support import phase, room_plans

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
    skipped = list(UNSUPPORTED)
    frozen = None
    if context.get("room_status") != "completed":
        skipped.append({"name": "room_placed_frozen",
                        "reason": context.get("room_reason") or context.get("room_status")})
    else:
        plans, reason = room_plans(settings, context, frozen=True)
        chosen = next((plan for plan in plans if plan.identity["arm"] == "room_placed_frozen"), None)
        if chosen is None and not any(item["name"] == "room_placed_frozen" for item in skipped):
            skipped.append({"name": "room_placed_frozen", "reason": reason})
        if chosen is not None:
            revision = context["best_revision"]
            pipeline = hindi_pipeline(settings, pool, checkpoint, selector=revision["selector"],
                                      replay_text=revision["replay_text"], repair_stop=revision["repair_stop"])
            pipeline.adapter_plan = chosen
            frozen = run_hindi(directory / "room_placed_frozen", "room_placed_frozen", pipeline, dataset,
                              model_factory, settings, selector=revision["selector"],
                              replay_text=revision["replay_text"], repair_stop=revision["repair_stop"])
    runs = [together] if frozen is None else [together, frozen]
    details = annotate(
        {"drift_past": together["attempt"], "skipped": skipped,
         "room_placed_frozen": None if frozen is None else frozen["attempt"]},
        clock, runs, together.get("final_model"))
    return save_report(directory, 7, details, {"drift_past": together, "phase7": details})
