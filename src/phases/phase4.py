"""Phase 4: extra rank where Hindi clashes with English, then the overall winner's confirmation seeds.

E1-E3 are screened on one seed. E4 runs only when placement beats both controls. The overall winner then runs
every Hindi seed, so X, A, and the winner carry the seed-averaged claims."""

from .support import combined_pipeline, hindi_pipeline, run_hindi
from ..pipelines.revision import load_pool
from .artifacts import annotate, compare_arms, publish_final, save_report
from .support import beats, choose_winner, phase, room_plans


def _room_runner(directory, dataset, model_factory, settings, context, pool, checkpoint):
    revision = context["best_revision"]

    def run(plan, seeds=None):
        name = plan.identity["arm"]
        pipeline = hindi_pipeline(settings, pool, checkpoint, selector=revision["selector"],
                                  replay_text=revision["replay_text"], repair_stop=revision["repair_stop"])
        pipeline.adapter_plan = plan

        def per_seed(seed):
            seeded, reason = room_plans(settings, context, seed, names=(name,) if name == "room_placed_slow" else
                                        ("room_placed", "room_random", "room_spread"))
            if not seeded:
                raise ValueError(reason)
            return next(item for item in seeded if item.identity["arm"] == name)
        return run_hindi(directory / name, name, pipeline, dataset, model_factory, settings, plan_factory=per_seed,
                         seeds=seeds, selector=revision["selector"], replay_text=revision["replay_text"],
                         repair_stop=revision["repair_stop"])
    return run


def _confirm(directory, dataset, model_factory, settings, context, pool, checkpoint, winner, plans, run_room):
    """Train the winner's missing seeds. Its screening seed is reused from the original attempt."""
    if winner["name"] == "random":
        return winner
    if winner["name"] == "combined":
        return run_hindi(directory.parent / "phase3" / "combined", "combined",
                         combined_pipeline(settings, pool, checkpoint, context["best_revision"], context["best_blocks"]),
                         dataset, model_factory, settings, seeds=settings.hindi_seeds,
                         selector=winner["selector"], replay_text=winner["replay_text"],
                         repair_stop=winner["repair_stop"], blocks=context["best_blocks"])
    return run_room(plans[winner["name"]], seeds=settings.hindi_seeds)


@phase(4)
def run(directory, dataset, model_factory, settings, context, clock):
    pool = load_pool(context["pool"])
    checkpoint = context["english_checkpoint"]
    before = context["english_metrics"]
    plans, reason = room_plans(settings, context)
    if not plans:
        status = "skipped" if not context["maps"]["stable"] or not any(
            value < 0 for value in context["maps"]["blocks"].values()) else "rejected"
        winner = _confirm(directory, dataset, model_factory, settings, context, pool, checkpoint,
                          context["combined"], {}, None)
        final = publish_final(directory, winner["name"], winner["checkpoint"]) or winner.get("final_model")
        details = annotate({"status": status, "reason": reason, "overall_winner": winner["name"],
                            "confirmed_seeds": [run.get("seed") for run in winner.get("seed_runs", [winner])]},
                           clock, [winner], final)
        update = {"room_status": status, "room_reason": reason, "room_arms": {},
                  "overall_winner": winner, "phase4": details}
        return save_report(directory, 4, details, update)
    run_room = _room_runner(directory, dataset, model_factory, settings, context, pool, checkpoint)
    by_name = {plan.identity["arm"]: plan for plan in plans}
    trained = {name: run_room(plan) for name, plan in by_name.items()}
    placement_wins = beats(before, trained["room_placed"], [trained["room_random"], trained["room_spread"]])
    e4_reason = None
    if placement_wins or settings.force_conditional:
        slow, e4_reason = room_plans(settings, context, names=("room_placed_slow",))
        if slow:
            by_name["room_placed_slow"] = slow[0]
            trained["room_placed_slow"] = run_room(slow[0])
    else:
        e4_reason = "room_placed did not beat room_random and room_spread by 1 point"
    winner = choose_winner(before, [context["combined"], *trained.values()], context["arms"]["random"])
    winner = _confirm(directory, dataset, model_factory, settings, context, pool, checkpoint, winner, by_name, run_room)
    if winner["name"] in trained:
        trained[winner["name"]] = winner
    comparisons = [compare_arms(trained["room_placed"], trained[other]) for other in ("room_random", "room_spread")]
    if winner["name"] != "random":
        comparisons.append(compare_arms(winner, context["arms"]["random"]))
    details = {"status": "completed", "arms": list(trained), "overall_winner": winner["name"],
               "confirmed_seeds": [run.get("seed") for run in winner.get("seed_runs", [winner])],
               "placement_wins": placement_wins, "room_placed_slow": e4_reason or "ran",
               "comparisons": comparisons, "placement_counts": any(
                   result["counts"] for result in comparisons[1]["metrics"].values())}
    final = publish_final(directory, winner["name"], winner["checkpoint"]) or winner.get("final_model")
    details = annotate(details, clock, list(trained.values()) + ([winner] if winner["name"] not in trained else []),
                       final)
    return save_report(directory, 4, details, {"room_status": "completed", "room_reason": "",
                                               "room_arms": trained, "overall_winner": winner, "phase4": details})
