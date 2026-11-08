"""Phase 2: random, loss, drift, past summaries, and repair-and-stop."""

from ..utils.numerics import top_overlap
from .support import hindi_pipeline, run_hindi
from ..pipelines.revision import load_pool
from .artifacts import annotate, compare_arms, save_report
from .support import lower_forgetting, phase


@phase(2)
def run(directory, dataset, model_factory, settings, context, clock):
    pool = load_pool(context["pool"])
    checkpoint = context["english_checkpoint"]
    specs = (
        ("random", dict(selector="random")),
        ("loss", dict(selector="loss")),
        ("drift", dict(selector="drift")),
        ("past", dict(selector="random", replay_text="past")),
    )
    arms = {}
    for name, overrides in specs:
        arms[name] = run_hindi(directory / name, name, hindi_pipeline(settings, pool, checkpoint, **overrides),
                              dataset, model_factory, settings)
    parent = lower_forgetting(context["english_metrics"], [arms["loss"], arms["drift"]])
    repair = run_hindi(directory / "repair_stop", "repair_stop",
                      hindi_pipeline(settings, pool, checkpoint, selector=parent["selector"], repair_stop=True),
                      dataset, model_factory, settings, selector=parent["selector"], replay_text="reference",
                      repair_stop=True, parent=parent["name"])
    arms["repair_stop"] = repair
    overlaps = {}
    for loss, drift in zip(arms["loss"].get("seed_runs", [arms["loss"]]),
                           arms["drift"].get("seed_runs", [arms["drift"]])):
        width = min(100, len(loss["ranking"]), len(drift["ranking"]))
        overlaps[str(loss["seed"])] = top_overlap(loss["ranking"], drift["ranking"], width) if width else None
    details = annotate(
        {"g3_parent": parent["name"], "top_overlap": overlaps,
         "filter_drops": len(arms["past"]["filtered"]), "retired": len(arms["repair_stop"]["retired"]),
         "arms": list(arms), "comparisons": [compare_arms(arms[a], arms[b]) for a, b in (
             ("loss", "random"), ("drift", "loss"), ("past", "random"), ("repair_stop", parent["name"]))]},
        clock, list(arms.values()), arms[parent["name"]].get("final_model"))
    return save_report(directory, 2, details, {"arms": arms, "g3_parent": parent["name"], "phase2": details})
