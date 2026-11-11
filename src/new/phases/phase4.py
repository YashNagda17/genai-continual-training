"""Phase 4: new rank where Hindi clashes with English (E), then the overall winner's confirmation seeds.

E1-E3 are screened on one seed. E4 runs only when placement beats both controls."""

import shutil

from ..pipelines.adapter_room import ROOM_ARMS, block_plan, room_plans
from .support import beats, brief, choose_winner, compare_arms, phase, revision_arm, revision_of


@phase(4)
def run(directory, dataset, settings, context):
    english, arms = context["english"], context["arms"]
    before, shared = english["metrics"], block_plan(context["best_blocks"], settings)
    revision = revision_of(context["best_revision"])

    def room(name, seeds=None):
        family = (name,) if name in ("room_placed_slow", "room_placed_frozen") else ROOM_ARMS
        return revision_arm(directory / name, name, settings, dataset, english, seeds, **revision,
                            plan_for_seed=lambda seed: room_plans(settings, context["maps"], shared, seed, family)[0][name])

    def confirm(winner):
        """Train the winner's missing seeds; its screening seed is reused."""
        if winner["name"] == "random":
            return winner
        if winner["name"] == "combined":
            return revision_arm(directory.parent / "phase3" / "combined", "combined", settings, dataset, english,
                                settings.hindi_seeds, plan=shared, **revision)
        return room(winner["name"], settings.hindi_seeds)

    plans, reason = room_plans(settings, context["maps"], shared, settings.hindi_seeds[0])
    status = "completed" if plans else "skipped" if reason in ("unstable_map", "no_clashing_blocks") else "rejected"
    trained, details = {}, {"status": status, "reason": reason}
    if plans:
        trained = {name: room(name) for name in ROOM_ARMS}
        placement_wins = beats(before, trained["room_placed"], [trained["room_random"], trained["room_spread"]])
        if placement_wins or settings.force_conditional:
            trained["room_placed_slow"] = room("room_placed_slow")
        winner = confirm(choose_winner(before, [context["combined"], *trained.values()], arms["random"]))
        trained.update({winner["name"]: winner} if winner["name"] in trained else {})
        comparisons = [compare_arms(trained["room_placed"], trained[other]) for other in ("room_random", "room_spread")]
        details.update(placement_wins=placement_wins, comparisons=comparisons,
                       placement_counts=any(result["counts"] for result in comparisons[1]["metrics"].values()))
    else:
        winner = confirm(context["combined"])
    final = directory / "final" / winner["name"]
    if not final.exists():
        shutil.copytree(winner["checkpoint"], final)
    details.update(overall_winner=winner["name"], final_model=str(final),
                   confirmed_seeds=[run["seed"] for run in winner.get("seed_runs", [winner])],
                   winner_vs_random=compare_arms(winner, arms["random"]) if winner["name"] != "random" else None,
                   runs=brief([*trained.values(), winner]))
    return details, {"room_status": details["status"], "room_reason": reason, "room_arms": trained,
                     "overall_winner": winner, "phase4": details}
