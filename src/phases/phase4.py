"""Phase 4: extra rank where Hindi clashes with English."""

from .support import hindi_pipeline, run_hindi
from ..pipelines.revision import load_pool
from .artifacts import annotate, compare_arms, publish_final, save_report
from .support import choose_winner, phase, room_plans


@phase(4)
def run(directory, dataset, model_factory, settings, context, clock):
    plans, reason = room_plans(settings, context)
    if not plans:
        status = "skipped" if not context["maps"]["stable"] or not any(
            value < 0 for value in context["maps"]["blocks"].values()) else "rejected"
        details = {"status": status, "reason": reason}
        final = publish_final(directory, context["combined"]["name"], context["combined"]["checkpoint"]) or context["combined"].get("final_model")
        details = annotate(details, clock, [], final)
        update = {"room_status": status, "room_reason": reason, "room_arms": {},
                  "overall_winner": context["combined"], "phase4": details}
        return save_report(directory, 4, details, update)
    pool = load_pool(context["pool"])
    checkpoint = context["english_checkpoint"]
    revision = context["best_revision"]
    trained = {}
    for plan in plans:
        name = plan.identity["arm"]
        pipeline = hindi_pipeline(settings, pool, checkpoint, selector=revision["selector"],
                                  replay_text=revision["replay_text"], repair_stop=revision["repair_stop"])
        pipeline.adapter_plan = plan
        def per_seed(seed, name=name):
            seeded, reason = room_plans(settings, context, seed)
            if not seeded:
                raise ValueError(reason)
            return next(item for item in seeded if item.identity["arm"] == name)
        trained[name] = run_hindi(directory / name, name, pipeline, dataset, model_factory, settings,
                                 plan_factory=per_seed,
                                 selector=revision["selector"], replay_text=revision["replay_text"],
                                 repair_stop=revision["repair_stop"])
    winner = choose_winner(context["english_metrics"], [context["combined"], *trained.values()],
                           context["arms"]["random"])
    comparison = compare_arms(trained["room_placed"], trained["room_spread"])
    details = {"status": "completed", "arms": list(trained), "overall_winner": winner["name"],
               "comparisons": [comparison], "placement_counts": any(
                   result["counts"] for result in comparison["metrics"].values())}
    final = publish_final(directory, winner["name"], winner["checkpoint"]) or winner.get("final_model")
    details = annotate(details, clock, list(trained.values()), final)
    return save_report(directory, 4, details, {"room_status": "completed", "room_reason": "",
                                               "room_arms": trained, "overall_winner": winner, "phase4": details})
