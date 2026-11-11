"""Phase 2: what to revise and with what. A runs every Hindi seed; B, G1, G2 and G3 are screened on one.

The CSReL and TSR paper baselines run here too, on the same Hindi data, seeds and English memory budget."""

from ..pipelines.csrel import CSReLPipeline
from ..pipelines.tsr import TSRPipeline
from .support import brief, compare_arms, lower_forgetting, phase, revision_arm, run_arm


def baselines(directory, dataset, settings, english, seeds=None):
    """CSReL's memory and learner, its random-memory control (same learner), and TSR."""
    memory = run_arm("csrel_memory", directory / "csrel_memory", settings, lambda model, output, seed: CSReLPipeline(
        settings, model, dataset, output, english["checkpoint"], seed))
    weight, cost = settings.csrel_replay_weight, {"selection_seconds": memory["selection_seconds"]}
    return {"csrel": revision_arm(directory / "csrel", "csrel", settings, dataset, english, seeds, pool_path=memory["pool"],
                                  replay_weight=weight, fields=cost),
            "csrel_random": revision_arm(directory / "csrel_random", "csrel_random", settings, dataset, english, seeds,
                                         replay_weight=weight),
            "tsr": revision_arm(directory / "tsr", "tsr", settings, dataset, english, seeds, pipeline=TSRPipeline,
                                selector="none")}, memory


@phase(2)
def run(directory, dataset, settings, context):
    english = context["english"]
    arms = {}
    for name, config in (("random", {}), ("loss", {"selector": "loss"}), ("drift", {"selector": "drift"}),
                         ("past", {"replay_text": "past"})):
        seeds = settings.hindi_seeds if name == "random" else None
        arms[name] = revision_arm(directory / name, name, settings, dataset, english, seeds, **config)
    parent = lower_forgetting(english["metrics"], [arms["loss"], arms["drift"]])
    arms["repair_stop"] = revision_arm(directory / "repair_stop", "repair_stop", settings, dataset, english,
                                       selector=parent["selector"], repair_stop=True)
    found, memory = baselines(directory, dataset, settings, english)
    arms.update(found)
    loss, drift = arms["loss"]["ranking"], arms["drift"]["ranking"]
    width = min(100, len(loss), len(drift))
    details = {"g3_parent": parent["name"], "filter_drops": len(arms["past"]["filtered"]),
               "top_overlap": len(set(loss[:width]) & set(drift[:width])) / width if width else None,
               "retired": len(arms["repair_stop"]["retired"]), "retired_share": arms["repair_stop"]["retired_share"],
               "comparisons": [compare_arms(arms[a], arms[b]) for a, b in (
                   ("loss", "random"), ("drift", "loss"), ("past", "random"), ("repair_stop", parent["name"]),
                   ("csrel", "csrel_random"), ("csrel", "random"), ("tsr", "random"))],
               "csrel_selection": {key: memory[key] for key in ("rounds", "selection_seconds", "overlap_with_random_pool")},
               "runs": brief(arms.values())}
    return details, {"arms": arms, "phase2": details}
